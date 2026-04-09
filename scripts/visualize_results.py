"""
Generate comparison visualizations for the freeze-strategy benchmark.

Usage:
    python scripts/visualize_results.py
"""

import json
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np


def load_results():
    with open("results/pilot_kvasir.json") as f:
        return json.load(f)


def load_losses():
    losses = {}
    for strategy in ["medsam", "ppsam", "ptsam"]:
        path = join("work_dir", "benchmark", f"{strategy}_kvasir", "losses.npy")
        if os.path.exists(path):
            losses[strategy] = np.load(path)
    return losses


def load_metadata():
    meta = {}
    for strategy in ["medsam", "ppsam", "ptsam"]:
        path = join("work_dir", "benchmark", f"{strategy}_kvasir", "metadata.npy")
        if os.path.exists(path):
            meta[strategy] = np.load(path, allow_pickle=True).item()
    return meta


def plot_training_curves(losses, save_path="results/figures/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"medsam": "#2196F3", "ppsam": "#FF9800", "ptsam": "#4CAF50"}
    labels = {"medsam": "MedSAM (Mask Decoder)", "ppsam": "PP-SAM (Prompt Encoder)", "ptsam": "PT-SAM (Prompt Tokens)"}

    for strategy, loss_arr in losses.items():
        epochs = range(1, len(loss_arr) + 1)
        ax.plot(epochs, loss_arr, marker="o", color=colors[strategy],
                label=labels[strategy], linewidth=2, markersize=6)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Dice + BCE Loss", fontsize=12)
    ax.set_title("Training Loss Curves (Kvasir-SEG, Fixed Encoder)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_metric_comparison(results, save_path="results/figures/metric_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    strategies = ["medsam", "ppsam", "ptsam"]
    labels = ["MedSAM\n(Mask Decoder\n4.06M params)", "PP-SAM\n(Prompt Encoder\n6.2K params)", "PT-SAM\n(Prompt Tokens\n2.0K params)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    dice_means = [results[s]["summary"]["dice_mean"] for s in strategies]
    dice_stds = [results[s]["summary"]["dice_std"] for s in strategies]
    iou_means = [results[s]["summary"]["iou_mean"] for s in strategies]
    iou_stds = [results[s]["summary"]["iou_std"] for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dice
    bars = axes[0].bar(labels, dice_means, yerr=dice_stds, color=colors,
                       capsize=8, edgecolor="black", linewidth=0.8, alpha=0.85)
    axes[0].set_ylabel("Dice Score", fontsize=12)
    axes[0].set_title("Dice Coefficient", fontsize=13)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, dice_means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{mean:.3f}", ha="center", fontsize=11, fontweight="bold")

    # IoU
    bars = axes[1].bar(labels, iou_means, yerr=iou_stds, color=colors,
                       capsize=8, edgecolor="black", linewidth=0.8, alpha=0.85)
    axes[1].set_ylabel("IoU Score", fontsize=12)
    axes[1].set_title("Intersection over Union", fontsize=13)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, iou_means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{mean:.3f}", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Freeze-Strategy Benchmark: Kvasir-SEG Test Set (N=100)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_image_distribution(results, save_path="results/figures/dice_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = ["medsam", "ppsam", "ptsam"]
    labels = ["MedSAM (Mask Decoder)", "PP-SAM (Prompt Encoder)", "PT-SAM (Prompt Tokens)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    data = []
    for s in strategies:
        dices = [r["dice"] for r in results[s]["per_image"]]
        data.append(dices)

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Per-Image Dice Distribution (Kvasir-SEG Test, N=100)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_params_vs_performance(results, metadata, save_path="results/figures/params_vs_dice.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = ["ptsam", "ppsam", "medsam"]
    labels = ["PT-SAM", "PP-SAM", "MedSAM"]
    colors = ["#4CAF50", "#FF9800", "#2196F3"]

    for s, label, color in zip(strategies, labels, colors):
        params = metadata[s]["trainable_params"]
        dice = results[s]["summary"]["dice_mean"]
        dice_std = results[s]["summary"]["dice_std"]
        ax.errorbar(params, dice, yerr=dice_std, fmt="o", color=color,
                    markersize=12, capsize=8, linewidth=2, label=label,
                    markeredgecolor="black", markeredgewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Parameter Efficiency: Params vs Performance", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    results = load_results()
    losses = load_losses()
    metadata = load_metadata()

    plot_training_curves(losses)
    plot_metric_comparison(results)
    plot_per_image_distribution(results)
    plot_params_vs_performance(results, metadata)

    print("\nAll visualizations saved to results/figures/")
