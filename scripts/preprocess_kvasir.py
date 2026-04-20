"""Preprocess Kvasir-SEG raw JPGs into the .npy format NpyDataset expects.

Input:  data/Kvasir-SEG/images/*.jpg  (1000 polyp JPG images, variable resolution)
        data/Kvasir-SEG/masks/*.jpg   (1000 binary-ish JPG masks, same basenames)
Output: data/npy/KvasirSEG_train/imgs/*.npy   (900 x 1024x1024x3 float64 [0,1])
        data/npy/KvasirSEG_train/gts/*.npy    (900 x 1024x1024 uint8 {0,1})
        data/npy/KvasirSEG_test/imgs/*.npy    (100 x same spec)
        data/npy/KvasirSEG_test/gts/*.npy     (100 x same spec)
        data/npy/KvasirSEG_split.json         ({"train": [...], "test": [...], "seed": 42})

Split: 900 train / 100 test, seed=42, via np.random.RandomState(42).permutation(1000).
This matches the count and seed convention used for ETIS and conceptually follows the
PraNet 900/100 split size used by PP-SAM (arXiv:2405.16740). This is NOT a bit-exact
replication of PraNet's published folder assignment -- which would require downloading
their Google Drive archive. The split manifest at KvasirSEG_split.json is published for
reproducibility.

Image: float64 [0,1] (matches the ETIS preprocessor output dtype; NpyDataset coerces
to float32 at load time via tensor.float(), so disk float64 is functionally equivalent).
Mask:  uint8 {0,1} after threshold >127 (cleanly recovers binary from JPEG compression).

Usage: python scripts/preprocess_kvasir.py
Prereq: Download raw data from https://datasets.simula.no/downloads/kvasir-seg.zip
        Unzip to data/Kvasir-SEG/ (so data/Kvasir-SEG/images/ and masks/ exist).
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

SEED = 42
RAW_DIR = "data/Kvasir-SEG"
OUT_DIR = "data/npy"
IMG_SIZE = 1024
TRAIN_COUNT = 900
EXPECTED_COUNT = 1000
DOWNLOAD_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def main():
    img_dir = os.path.join(RAW_DIR, "images")
    mask_dir = os.path.join(RAW_DIR, "masks")

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        print(f"ERROR: raw Kvasir-SEG data not found at '{RAW_DIR}'.", file=sys.stderr)
        print(f"Download from: {DOWNLOAD_URL}", file=sys.stderr)
        print(f"Unzip so that '{img_dir}' and '{mask_dir}' exist.", file=sys.stderr)
        sys.exit(1)

    filenames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    assert len(filenames) == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} Kvasir images at {img_dir}, got {len(filenames)}"
    )

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(filenames))
    train_indices = indices[:TRAIN_COUNT]
    test_indices = indices[TRAIN_COUNT:]

    train_files = [filenames[i] for i in train_indices]
    test_files = [filenames[i] for i in test_indices]

    print(f"Total: {len(filenames)} | Train: {len(train_files)} | Test: {len(test_files)}")

    splits = {"train": train_files, "test": test_files, "seed": SEED}

    for split_name, file_list in [("KvasirSEG_train", train_files), ("KvasirSEG_test", test_files)]:
        img_out = os.path.join(OUT_DIR, split_name, "imgs")
        gt_out = os.path.join(OUT_DIR, split_name, "gts")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(gt_out, exist_ok=True)

        for fname in tqdm(file_list, desc=f"Processing {split_name}"):
            name = fname.replace(".jpg", ".npy")

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
    with open(os.path.join(OUT_DIR, "KvasirSEG_split.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # Verification
    for split_name in ["KvasirSEG_train", "KvasirSEG_test"]:
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

    print(f"\nSplit manifest saved to {OUT_DIR}/KvasirSEG_split.json")
    print("Done.")


if __name__ == "__main__":
    main()
