"""Evaluation: Dice, IoU, HD95, inference time, GPU memory."""

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MedSAM-main"))
from segment_anything import sam_model_registry

from .config import BenchmarkConfig
from .strategies import STRATEGY_REGISTRY, build_strategy


def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return (2.0 * intersection) / (pred.sum() + gt.sum())


def iou_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    if union == 0:
        return 1.0
    return intersection / union


def hausdorff_95(pred, gt):
    """Compute 95th percentile Hausdorff distance between binary masks.

    Uses distance transform approach for efficiency.
    Returns distance in pixels. If either mask is empty, returns image diagonal.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    max_dist = np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)

    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return max_dist

    # Surface points (boundary pixels)
    from scipy.ndimage import binary_erosion
    pred_boundary = pred ^ binary_erosion(pred)
    gt_boundary = gt ^ binary_erosion(gt)

    if not pred_boundary.any():
        pred_boundary = pred
    if not gt_boundary.any():
        gt_boundary = gt

    # Distance from pred boundary to nearest gt point
    dt_gt = distance_transform_edt(~gt)
    dist_pred_to_gt = dt_gt[pred_boundary]

    # Distance from gt boundary to nearest pred point
    dt_pred = distance_transform_edt(~pred)
    dist_gt_to_pred = dt_pred[gt_boundary]

    hd95 = max(
        np.percentile(dist_pred_to_gt, 95),
        np.percentile(dist_gt_to_pred, 95),
    )
    return float(hd95)


def get_bbox_from_mask(mask):
    """Extract tight bounding box from binary mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        H, W = mask.shape
        return np.array([0, 0, W, H])
    return np.array([
        np.min(x_indices), np.min(y_indices),
        np.max(x_indices), np.max(y_indices),
    ])


@torch.no_grad()
def evaluate_strategy(strategy_name, config=None):
    """Evaluate a trained model on the test set. Returns results dict."""
    if config is None:
        config = BenchmarkConfig()

    device = torch.device(config.DEVICE)
    checkpoint_path = os.path.join(config.WORK_DIR, strategy_name, "model_best.pth")

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {strategy_name.upper()}")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: checkpoint not found at {checkpoint_path}")
        return None

    # Load model (full model for all strategies — fair inference comparison)
    sam_model = sam_model_registry["vit_b"](checkpoint=config.SAM_CHECKPOINT)
    model = build_strategy(strategy_name, sam_model, config).to(device)
    del sam_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')}")

    # Load test data
    test_gt_dir = os.path.join(config.TEST_DATA, "gts")
    test_img_dir = os.path.join(config.TEST_DATA, "imgs")
    gt_files = sorted(glob.glob(os.path.join(test_gt_dir, "*.npy")))
    print(f"  Test images: {len(gt_files)}")

    torch.cuda.reset_peak_memory_stats(device)
    results = []

    for gt_file in tqdm(gt_files, desc=f"  Eval {strategy_name}"):
        name = os.path.basename(gt_file)
        img_path = os.path.join(test_img_dir, name)
        if not os.path.exists(img_path):
            continue

        img = np.load(img_path, allow_pickle=True)
        gt = np.load(gt_file, allow_pickle=True)
        gt_binary = (gt > 0).astype(np.uint8)

        img_tensor = torch.tensor(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        bbox = get_bbox_from_mask(gt_binary)
        bbox_np = bbox[None, :]

        # Timed inference
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.amp.autocast("cuda"):
            if strategy_name == "ptsam":
                pred_logits = model(image=img_tensor)
            else:
                pred_logits = model(img_tensor, bbox_np)

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000

        pred_mask = (torch.sigmoid(pred_logits) > 0.5).cpu().numpy().squeeze().astype(np.uint8)

        # Resize pred to gt resolution if needed
        if pred_mask.shape != gt_binary.shape:
            pred_resized = torch.nn.functional.interpolate(
                torch.tensor(pred_mask[None, None, :, :]).float(),
                size=gt_binary.shape, mode="nearest",
            ).numpy().squeeze().astype(np.uint8)
        else:
            pred_resized = pred_mask

        results.append({
            "name": name,
            "dice": dice_score(pred_resized, gt_binary),
            "iou": iou_score(pred_resized, gt_binary),
            "hd95": hausdorff_95(pred_resized, gt_binary),
            "inference_ms": inference_ms,
        })

    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    # Summary statistics
    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    hd95s = [r["hd95"] for r in results]
    times = [r["inference_ms"] for r in results]

    summary = {
        "strategy": strategy_name,
        "n_images": len(results),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "hd95_mean": float(np.mean(hd95s)),
        "hd95_std": float(np.std(hd95s)),
        "inference_ms_mean": float(np.mean(times)),
        "inference_ms_std": float(np.std(times)),
        "peak_memory_mb": peak_memory,
    }

    print(f"\n  Results for {strategy_name.upper()}:")
    print(f"    Dice: {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f}")
    print(f"    IoU:  {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f}")
    print(f"    HD95: {summary['hd95_mean']:.2f} +/- {summary['hd95_std']:.2f}")
    print(f"    Inference: {summary['inference_ms_mean']:.1f}ms +/- {summary['inference_ms_std']:.1f}ms")
    print(f"    Peak memory: {peak_memory:.0f} MB")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {"summary": summary, "per_image": results}
