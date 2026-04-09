"""Quick verification: visual check that augmentations produce correct output."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchmark.dataset import AugmentedNpyDataset


def main():
    data_root = os.path.join("data", "npy", "ETIS_train")
    if not os.path.isdir(data_root):
        print(f"ERROR: {data_root} not found")
        sys.exit(1)

    ds = AugmentedNpyDataset(data_root, augment=True)
    if len(ds) == 0:
        print("ERROR: dataset is empty")
        sys.exit(1)

    out_dir = os.path.join("results", "etis", "augmentation_check")
    os.makedirs(out_dir, exist_ok=True)

    sample_idx = 0
    n_views = 10

    fig, axes = plt.subplots(n_views, 2, figsize=(8, 4 * n_views))
    fig.suptitle(f"Sample {sample_idx}: 10 augmented views", fontsize=14)

    for i in range(n_views):
        img_tensor, mask_tensor, bbox, name = ds[sample_idx]
        img = img_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        mask = mask_tensor.numpy().squeeze()  # (1,H,W) -> (H,W)

        # Validation checks
        assert img.min() >= 0.0, f"View {i}: image min={img.min()}"
        assert img.max() <= 1.0, f"View {i}: image max={img.max()}"
        assert set(np.unique(mask)).issubset({0, 1}), f"View {i}: mask values={np.unique(mask)}"
        assert img.shape[:2] == mask.shape, f"View {i}: shape mismatch img={img.shape} mask={mask.shape}"

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"View {i} - Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"View {i} - Mask (sum={mask.sum()})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "augmentation_verification.png")
    plt.savefig(out_path, dpi=100)
    plt.close()

    print(f"All {n_views} views passed validation checks:")
    print("  - Image values in [0, 1]")
    print("  - Mask is binary (0/1)")
    print("  - Image and mask shapes match")
    print(f"  - Visual output saved to {out_path}")


if __name__ == "__main__":
    main()
