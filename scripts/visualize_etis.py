"""
Generate visualization figures for the ETIS benchmark results.

Produces:
  - Training + validation loss curves (overlay of 3 strategies)
  - Metric comparison bar charts (Dice, IoU, HD95)
  - Per-image Dice box plot
  - Params vs Performance scatter
  - Comparison table (JSON + text)

Usage:
    python scripts/visualize_etis.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results/etis"
WORK_DIR = "work_dir/benchmark_etis"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

STRATEGIES = ["basesam", "medsam", "ppsam", "ptsam"]
COLORS = {"basesam": "#9E9E9E", "medsam": "#2196F3", "ppsam": "#FF9800", "ptsam": "#4CAF50"}
LABELS = {
    "basesam": "Base SAM (Zero-shot)",
    "medsam": "MedSAM (Enc+Dec)",
    "ppsam": "PP-SAM (Enc+Prompt)",
    "ptsam": "PT-SAM (Tokens only)",
}


def load_results():
    path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(path) as f:
        return json.load(f)


def load_losses():
    losses = {}
    for s in STRATEGIES:
        train_path = os.path.join(WORK_DIR, s, "train_losses.npy")
        val_path = os.path.join(WORK_DIR, s, "val_losses.npy")
        if os.path.exists(train_path) and os.path.exists(val_path):
            losses[s] = {
                "train": np.load(train_path),
                "val": np.load(val_path),
            }
    return losses


def load_train_results():
    results = {}
    for s in STRATEGIES:
        path = os.path.join(WORK_DIR, s, "train_result.json")
        if os.path.exists(path):
            with open(path) as f:
                results[s] = json.load(f)
    return results


def plot_training_curves(losses):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for s, data in losses.items():
        epochs = range(1, len(data["train"]) + 1)
        axes[0].plot(epochs, data["train"], color=COLORS[s], label=LABELS[s],
                     linewidth=2, marker="o", markersize=4)
        axes[1].plot(epochs, data["val"], color=COLORS[s], label=LABELS[s],
                     linewidth=2, marker="o", markersize=4)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (Dice + BCE)", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss (Dice + BCE)", fontsize=12)
    axes[1].set_title("Validation Loss", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("ETIS-LaribPolypDB: Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    save = os.path.join(FIG_DIR, "training_curves.png")
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


def plot_metric_comparison(results):
    os.makedirs(FIG_DIR, exist_ok=True)
    available = [s for s in STRATEGIES if s in results]
    labels = [LABELS[s] for s in available]
    colors = [COLORS[s] for s in available]

    dice_m = [results[s]["summary"]["dice_mean"] for s in available]
    dice_s = [results[s]["summary"]["dice_std"] for s in available]
    iou_m = [results[s]["summary"]["iou_mean"] for s in available]
    iou_s = [results[s]["summary"]["iou_std"] for s in available]
    hd95_m = [results[s]["summary"]["hd95_mean"] for s in available]
    hd95_s = [results[s]["summary"]["hd95_std"] for s in available]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, means, stds, ylabel, title in [
        (axes[0], dice_m, dice_s, "Dice Score", "Dice Coefficient"),
        (axes[1], iou_m, iou_s, "IoU Score", "Intersection over Union"),
        (axes[2], hd95_m, hd95_s, "HD95 (pixels)", "Hausdorff Distance 95%"),
    ]:
        bars = ax.bar(labels, means, yerr=stds, color=colors,
                      capsize=8, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")

    n = results[available[0]]["summary"]["n_images"]
    fig.suptitle(f"ETIS-LaribPolypDB Test Set (N={n})", fontsize=14, y=1.02)
    plt.tight_layout()
    save = os.path.join(FIG_DIR, "metric_comparison.png")
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


def plot_dice_distribution(results):
    os.makedirs(FIG_DIR, exist_ok=True)
    available = [s for s in STRATEGIES if s in results]
    labels = [LABELS[s] for s in available]
    colors = [COLORS[s] for s in available]

    data = [[r["dice"] for r in results[s]["per_image"]] for s in available]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Per-Image Dice Distribution", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    save = os.path.join(FIG_DIR, "dice_distribution.png")
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


def plot_params_vs_performance(results, train_results):
    os.makedirs(FIG_DIR, exist_ok=True)
    available = [s for s in STRATEGIES if s in results and s in train_results]
    labels = {"medsam": "MedSAM", "ppsam": "PP-SAM", "ptsam": "PT-SAM"}
    colors_list = [COLORS[s] for s in available]

    fig, ax = plt.subplots(figsize=(8, 5))
    for s, color in zip(available, colors_list):
        params = train_results[s]["trainable_params"]
        dice = results[s]["summary"]["dice_mean"]
        dice_std = results[s]["summary"]["dice_std"]
        ax.errorbar(params, dice, yerr=dice_std, fmt="o", color=color,
                    markersize=12, capsize=8, linewidth=2, label=labels[s],
                    markeredgecolor="black", markeredgewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Parameter Efficiency", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save = os.path.join(FIG_DIR, "params_vs_dice.png")
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


def save_comparison_table(results, train_results):
    """Save comparison as text file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    available = [s for s in STRATEGIES if s in results]
    labels = {"medsam": "MedSAM", "ppsam": "PP-SAM", "ptsam": "PT-SAM"}

    lines = []
    lines.append("Strategy | Trainable | Dice | IoU | HD95 | Infer(ms) | Train Time | Mem(MB)")
    lines.append("-" * 85)

    for s in available:
        e = results[s]["summary"]
        t = train_results.get(s, {})
        params = t.get("trainable_params", 0)
        if params >= 1e6:
            p_str = f"{params/1e6:.1f}M"
        elif params >= 1e3:
            p_str = f"{params/1e3:.1f}K"
        else:
            p_str = str(params)
        tt = t.get("total_time_seconds", 0)
        tt_str = f"{tt/60:.1f}m" if tt >= 60 else f"{tt:.0f}s"
        lines.append(
            f"{labels[s]:<8} | {p_str:>9} | "
            f"{e['dice_mean']:.3f}±{e['dice_std']:.2f} | "
            f"{e['iou_mean']:.3f}±{e['iou_std']:.2f} | "
            f"{e['hd95_mean']:.1f}±{e['hd95_std']:.1f} | "
            f"{e['inference_ms_mean']:>6.0f}     | "
            f"{tt_str:>10} | {e['peak_memory_mb']:.0f}"
        )

    table_text = "\n".join(lines)
    save = os.path.join(RESULTS_DIR, "comparison_table.txt")
    with open(save, "w") as f:
        f.write(table_text)
    print(f"\nSaved: {save}")
    print(table_text)


if __name__ == "__main__":
    results = load_results()
    losses = load_losses()
    train_results = load_train_results()

    if losses:
        plot_training_curves(losses)
    if results:
        plot_metric_comparison(results)
        plot_dice_distribution(results)
    if results and train_results:
        plot_params_vs_performance(results, train_results)
        save_comparison_table(results, train_results)

    print(f"\nAll figures saved to {FIG_DIR}/")
