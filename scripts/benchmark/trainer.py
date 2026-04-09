"""Training loop with AMP, validation, timing, and GPU monitoring."""

import json
import os
import random
import sys
import time

import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MedSAM-main"))
from segment_anything import sam_model_registry

from .config import BenchmarkConfig
from .dataset import AugmentedNpyDataset, CachedEmbeddingDataset, NpyDataset
from .gpu_monitor import GPUMonitor
from .strategies import STRATEGY_REGISTRY, build_strategy


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_train_val(dataset, val_fraction, seed):
    """Deterministic train/val split of a dataset by indices."""
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _estimate_epoch_time(model, dataloader, device, config, is_cached=False):
    """Time a single batch to estimate epoch duration."""
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    batch = next(iter(dataloader))

    torch.cuda.synchronize()
    start = time.perf_counter()

    if is_cached:
        embedding, gt2D, boxes, _ = batch
        embedding, gt2D = embedding.to(device), gt2D.to(device)
        boxes_np = boxes.numpy()
        with torch.amp.autocast("cuda"):
            pred = model(precomputed_embedding=embedding, box=boxes_np)
            loss = seg_loss_fn(pred, gt2D.float()) + ce_loss_fn(pred, gt2D.float())
        scaler.scale(loss).backward()
    else:
        image, gt2D, boxes, _ = batch
        image, gt2D = image.to(device), gt2D.to(device)
        boxes_np = boxes.numpy()
        with torch.amp.autocast("cuda"):
            pred = model(image, boxes_np)
            loss = seg_loss_fn(pred, gt2D.float()) + ce_loss_fn(pred, gt2D.float())
        scaler.scale(loss).backward()

    torch.cuda.synchronize()
    batch_time = time.perf_counter() - start

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    n_batches = len(dataloader)
    epoch_est = batch_time * n_batches
    return epoch_est


def train_strategy(strategy_name, config=None):
    """Train one strategy. Returns dict with losses, timing, metadata."""
    if config is None:
        config = BenchmarkConfig()

    _seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    gpu_monitor = GPUMonitor()

    print(f"\n{'='*60}")
    print(f"  TRAINING: {strategy_name.upper()}")
    print(f"{'='*60}")

    # Build model
    sam_model = sam_model_registry["vit_b"](checkpoint=config.SAM_CHECKPOINT)
    model = build_strategy(strategy_name, sam_model, config).to(device)
    del sam_model
    torch.cuda.empty_cache()

    # Each strategy uses its paper-recommended hyperparameters
    is_ptsam = strategy_name == "ptsam"
    hp = config.get_strategy_hparams(strategy_name)
    lr = hp["lr"]
    num_epochs = hp["num_epochs"]
    batch_size = hp["batch_size"]

    trainable_params = model.get_trainable_params()
    trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  Total params:     {total_count:>12,}")
    print(f"  Trainable params: {trainable_count:>12,}")
    print(f"  AMP: {config.AMP} | Grad checkpoint: {config.GRADIENT_CHECKPOINT and strategy_name != 'ptsam'}")
    print(f"  Batch size: {batch_size} | LR: {lr} | WD: {hp['weight_decay']} | Epochs: {num_epochs}")

    # Build dataset
    _, bbox_shift = STRATEGY_REGISTRY[strategy_name]
    use_augmentation = is_ptsam and config.PTSAM_AUGMENTATION
    is_cached = is_ptsam and not use_augmentation

    if use_augmentation:
        # PT-SAM with augmentation: raw images, frozen encoder runs each batch
        train_full = AugmentedNpyDataset(config.TRAIN_DATA, bbox_shift, augment=True)
        val_full = NpyDataset(config.TRAIN_DATA, bbox_shift)
        # Same deterministic split for both datasets
        n = len(train_full)
        n_val = max(1, int(n * config.VAL_SPLIT))
        rng = np.random.RandomState(config.SEED)
        indices = rng.permutation(n)
        train_idx = indices[: n - n_val].tolist()
        val_idx = indices[n - n_val :].tolist()
        train_dataset = Subset(train_full, train_idx)
        val_dataset = Subset(val_full, val_idx)
        print(f"  Augmentation: ON (frozen encoder per-batch, no cached embeddings)")
    elif is_cached:
        cache_dir = os.path.join(config.WORK_DIR, "ptsam_cache")
        full_dataset = CachedEmbeddingDataset(config.TRAIN_DATA, cache_dir, bbox_shift)
        train_dataset, val_dataset = _split_train_val(full_dataset, config.VAL_SPLIT, config.SEED)
    else:
        full_dataset = NpyDataset(config.TRAIN_DATA, bbox_shift)
        train_dataset, val_dataset = _split_train_val(full_dataset, config.VAL_SPLIT, config.SEED)

    print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Estimate time
    epoch_est = _estimate_epoch_time(model, train_loader, device, config, is_cached)
    total_est = epoch_est * num_epochs
    print(f"\n  Estimated time per epoch: {_format_time(epoch_est)}")
    print(f"  Estimated total time:    {_format_time(total_est)}")
    print()

    # Optimizer, scheduler, and loss — all per-strategy from paper
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=hp["weight_decay"])
    scheduler = None
    if is_ptsam:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    dice_weight = hp["dice_weight"]
    ce_weight = hp["ce_weight"]
    scaler = torch.amp.GradScaler("cuda")

    # Save directory
    save_dir = os.path.join(config.WORK_DIR, strategy_name)
    os.makedirs(save_dir, exist_ok=True)

    # Training state
    best_val_loss = float("inf")
    best_epoch = -1
    train_losses = []
    val_losses = []
    epoch_times = []
    gpu_utils = []

    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        n_steps = 0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{num_epochs} [train]",
                          leave=False):
            if is_cached:
                embedding, gt2D, boxes, _ = batch
                embedding, gt2D = embedding.to(device), gt2D.to(device)
                boxes_np = boxes.numpy()
            else:
                image, gt2D, boxes, _ = batch
                image, gt2D = image.to(device), gt2D.to(device)
                boxes_np = boxes.numpy()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                if is_cached:
                    pred = model(precomputed_embedding=embedding, box=boxes_np)
                else:
                    pred = model(image, boxes_np)
                loss = dice_weight * seg_loss_fn(pred, gt2D.float()) + ce_weight * ce_loss_fn(pred, gt2D.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_steps += 1

        train_loss = running_loss / max(n_steps, 1)
        train_losses.append(train_loss)

        # --- Validate ---
        model.eval()
        val_running = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if is_cached:
                    embedding, gt2D, boxes, _ = batch
                    embedding, gt2D = embedding.to(device), gt2D.to(device)
                    boxes_np = boxes.numpy()
                else:
                    image, gt2D, boxes, _ = batch
                    image, gt2D = image.to(device), gt2D.to(device)
                    boxes_np = boxes.numpy()

                with torch.amp.autocast("cuda"):
                    if is_cached:
                        pred = model(precomputed_embedding=embedding, box=boxes_np)
                    else:
                        pred = model(image, boxes_np)
                    loss = dice_weight * seg_loss_fn(pred, gt2D.float()) + ce_weight * ce_loss_fn(pred, gt2D.float())

                val_running += loss.item()
                val_steps += 1

        val_loss = val_running / max(val_steps, 1)
        val_losses.append(val_loss)

        # --- Scheduler step ---
        if scheduler is not None:
            scheduler.step()

        # --- Timing ---
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = time.time() - run_start
        remaining = (epoch_time * (num_epochs - epoch - 1))

        # --- GPU check ---
        util = gpu_monitor.check_and_warn(config.GPU_UTIL_WARN_THRESHOLD)
        gpu_utils.append(util)
        peak_mem = gpu_monitor.get_peak_memory_mb()

        # --- Checkpoint ---
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "strategy": strategy_name,
                 "val_loss": val_loss, "train_loss": train_loss},
                os.path.join(save_dir, "model_best.pth"),
            )
            improved = " *best*"

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "strategy": strategy_name,
             "val_loss": val_loss, "train_loss": train_loss},
            os.path.join(save_dir, "model_latest.pth"),
        )

        # --- Log ---
        gpu_str = f"{util*100:.0f}%" if util >= 0 else "N/A"
        print(
            f"  Epoch {epoch+1:>2}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}{improved} | "
            f"Time: {_format_time(epoch_time)} | ETA: {_format_time(remaining)} | "
            f"GPU: {gpu_str} | Mem: {peak_mem:.0f}MB"
        )

    total_time = time.time() - run_start
    peak_memory = gpu_monitor.get_peak_memory_mb()

    print(f"\n  {strategy_name.upper()} complete:")
    print(f"    Total time:      {_format_time(total_time)}")
    print(f"    Best val loss:   {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"    Final train loss: {train_losses[-1]:.4f}")
    print(f"    Peak GPU memory: {peak_memory:.0f} MB")

    # Save artifacts
    result = {
        "strategy": strategy_name,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": hp["weight_decay"],
        "dice_weight": hp["dice_weight"],
        "ce_weight": hp["ce_weight"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_time_seconds": total_time,
        "peak_memory_mb": peak_memory,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "gpu_utilizations": gpu_utils,
    }
    with open(os.path.join(save_dir, "train_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, "val_losses.npy"), np.array(val_losses))

    # Cleanup
    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()

    return result
