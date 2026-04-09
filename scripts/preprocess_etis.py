"""
Preprocess ETIS-LaribPolypDB dataset into MedSAM npy format.

Converts 196 PNG colonoscopy images + binary masks into:
  - Images: 1024x1024x3, float64, normalized [0,1]
  - Masks:  1024x1024, uint8, binary {0,1}

Split: 160 train / 36 test (deterministic, seed=42).

Usage:
    python scripts/preprocess_etis.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

SEED = 42
RAW_DIR = "data/ETIS-LaribPolypDB"
OUT_DIR = "data/npy"
IMG_SIZE = 1024
TRAIN_COUNT = 160


def main():
    img_dir = os.path.join(RAW_DIR, "images")
    mask_dir = os.path.join(RAW_DIR, "masks")

    filenames = sorted(os.listdir(img_dir), key=lambda x: int(x.split(".")[0]))
    assert len(filenames) == 196, f"Expected 196 images, found {len(filenames)}"

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(filenames))
    train_indices = indices[:TRAIN_COUNT]
    test_indices = indices[TRAIN_COUNT:]

    train_files = [filenames[i] for i in train_indices]
    test_files = [filenames[i] for i in test_indices]

    print(f"Total: {len(filenames)} | Train: {len(train_files)} | Test: {len(test_files)}")

    splits = {"train": train_files, "test": test_files, "seed": SEED}

    for split_name, file_list in [("ETIS_train", train_files), ("ETIS_test", test_files)]:
        img_out = os.path.join(OUT_DIR, split_name, "imgs")
        gt_out = os.path.join(OUT_DIR, split_name, "gts")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(gt_out, exist_ok=True)

        for fname in tqdm(file_list, desc=f"Processing {split_name}"):
            name = fname.replace(".png", ".npy")

            img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_arr = np.array(img, dtype=np.float64) / 255.0
            assert img_arr.shape == (IMG_SIZE, IMG_SIZE, 3)
            np.save(os.path.join(img_out, name), img_arr)

            mask = Image.open(os.path.join(mask_dir, fname)).convert("L")
            mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            mask_arr = (np.array(mask) > 127).astype(np.uint8)
            assert mask_arr.shape == (IMG_SIZE, IMG_SIZE)
            assert mask_arr.max() <= 1
            np.save(os.path.join(gt_out, name), mask_arr)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ETIS_split.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # Verification
    for split_name in ["ETIS_train", "ETIS_test"]:
        imgs = os.listdir(os.path.join(OUT_DIR, split_name, "imgs"))
        gts = os.listdir(os.path.join(OUT_DIR, split_name, "gts"))
        assert len(imgs) == len(gts), f"{split_name}: img/gt count mismatch"
        sample_img = np.load(os.path.join(OUT_DIR, split_name, "imgs", imgs[0]))
        sample_gt = np.load(os.path.join(OUT_DIR, split_name, "gts", gts[0]))
        assert sample_img.shape == (IMG_SIZE, IMG_SIZE, 3)
        assert 0.0 <= sample_img.min() and sample_img.max() <= 1.0
        assert sample_gt.shape == (IMG_SIZE, IMG_SIZE)
        assert set(np.unique(sample_gt)).issubset({0, 1})
        print(f"  {split_name}: {len(imgs)} images verified OK")

    print(f"\nSplit manifest saved to {OUT_DIR}/ETIS_split.json")
    print("Done.")


if __name__ == "__main__":
    main()
