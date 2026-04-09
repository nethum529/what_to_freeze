"""
Evaluate trained benchmark models on the test set (full model with image encoder).

NOTE: This is a legacy standalone script. For the reproducible benchmark
with paper-recommended per-strategy hyperparameters, use run_benchmark.py.

Usage:
    python scripts/evaluate.py --all
    python scripts/evaluate.py --strategy medsam --checkpoint work_dir/benchmark_etis/medsam/model_best.pth
"""

import argparse
import glob
import json
import os
import sys
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MedSAM-main"))
from segment_anything import sam_model_registry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.benchmark.strategies import MedSAMStrategy, PPSAMStrategy, PTSAMStrategy, BaseSAMStrategy
from scripts.benchmark.config import BenchmarkConfig


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


STRATEGY_MAP = {
    "basesam": BaseSAMStrategy,
    "medsam": MedSAMStrategy,
    "ppsam": PPSAMStrategy,
    "ptsam": PTSAMStrategy,
}


def load_model(strategy_name, checkpoint_path, sam_checkpoint, device):
    config = BenchmarkConfig()
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    from scripts.benchmark.strategies import build_strategy
    model = build_strategy(strategy_name, sam_model, config).to(device)
    del sam_model
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        H, W = mask.shape
        return np.array([0, 0, W, H])
    return np.array([np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)])


@torch.no_grad()
def evaluate_model(model, strategy_name, test_path, device):
    gt_dir = join(test_path, "gts")
    img_dir = join(test_path, "imgs")
    gt_files = sorted(glob.glob(join(gt_dir, "*.npy")))

    results = []
    for gt_file in tqdm(gt_files, desc=f"Evaluating {strategy_name}"):
        name = os.path.basename(gt_file)
        img_path = join(img_dir, name)
        if not os.path.exists(img_path):
            continue

        img = np.load(img_path, allow_pickle=True)
        gt = np.load(gt_file, allow_pickle=True)
        gt_binary = (gt > 0).astype(np.uint8)

        img_tensor = torch.tensor(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        bbox = get_bbox_from_mask(gt_binary)
        bbox_np = bbox[None, :]

        pred_logits = model(img_tensor, bbox_np)
        pred_mask = (torch.sigmoid(pred_logits) > 0.5).cpu().numpy().squeeze()

        results.append({
            "name": name,
            "dice": dice_score(pred_mask, gt_binary),
            "iou": iou_score(pred_mask, gt_binary),
        })

    return results


def summarize(results, strategy_name):
    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    summary = {
        "strategy": strategy_name,
        "n_images": len(results),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
    }
    print(f"\n{'='*50}")
    print(f"Strategy: {strategy_name}")
    print(f"  Dice: {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f}")
    print(f"  IoU:  {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f}")
    print(f"  N:    {summary['n_images']}")
    print(f"{'='*50}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--strategy", choices=["basesam", "medsam", "ppsam", "ptsam"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--test_path", default="data/npy/ETIS_test")
    parser.add_argument("--sam_checkpoint", default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--work_dir", default="work_dir/benchmark_etis")
    parser.add_argument("--output", default="results/etis/evaluate_results.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    all_results = {}
    device = torch.device(args.device)

    if args.all:
        strategies = [
            ("basesam", None),
            ("medsam", join(args.work_dir, "medsam", "model_best.pth")),
            ("ppsam", join(args.work_dir, "ppsam", "model_best.pth")),
            ("ptsam", join(args.work_dir, "ptsam", "model_best.pth")),
        ]
        for name, ckpt in strategies:
            if ckpt is not None and not os.path.exists(ckpt):
                print(f"Skipping {name}: {ckpt} not found")
                continue
            model = load_model(name, ckpt, args.sam_checkpoint, device)
            results = evaluate_model(model, name, args.test_path, device)
            all_results[name] = {"summary": summarize(results, name), "per_image": results}
            del model
            torch.cuda.empty_cache()
    elif args.strategy and args.checkpoint:
        model = load_model(args.strategy, args.checkpoint, args.sam_checkpoint, device)
        results = evaluate_model(model, args.strategy, args.test_path, device)
        all_results[args.strategy] = {"summary": summarize(results, args.strategy), "per_image": results}
    else:
        parser.error("Either --all or both --strategy and --checkpoint required")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
