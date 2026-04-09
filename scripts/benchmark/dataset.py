"""Dataset classes for the ETIS benchmark."""

import glob
import os
import random

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


def build_ptsam_augmentation():
    """Augmentation pipeline matching Piater et al. CVPRW 2025.

    Spatial transforms applied jointly to image+mask.
    Photometric transforms applied to image only.
    Input images are float32 [0,1] HWC; masks are uint8 HW.
    """
    return A.Compose([
        # Spatial (joint image+mask)
        A.RandomResizedCrop(
            size=(1024, 1024), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=0, p=0.5),
        A.ElasticTransform(alpha=120, sigma=6.0, border_mode=0, p=0.3),
        # Photometric (image only)
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3,
        ),
    ])


class NpyDataset(Dataset):
    """Loads preprocessed npy images + masks, returns image tensor + bbox."""

    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            f for f in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(f)))
        ]
        self.bbox_shift = bbox_shift
        print(f"NpyDataset: {len(self.gt_path_files)} images from {data_root}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            os.path.join(self.img_path, img_name), "r", allow_pickle=True
        )
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0

        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)
        gt2D = (gt > 0).astype(np.uint8)

        if gt2D.sum() == 0:
            # No foreground — return full-image bbox as fallback
            H, W = gt2D.shape
            bboxes = np.array([0, 0, W, H])
        else:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


class CachedEmbeddingDataset(Dataset):
    """Loads precomputed image embeddings + masks for PT-SAM training."""

    def __init__(self, data_root, cache_dir, bbox_shift=20):
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            f for f in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(f)))
        ]
        self.bbox_shift = bbox_shift
        print(f"CachedEmbeddingDataset: {len(self.gt_path_files)} samples from {data_root}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        cache_name = img_name.replace(".npy", ".pt")
        embedding = torch.load(
            os.path.join(self.cache_dir, cache_name),
            map_location="cpu",
            weights_only=True,
        )

        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)
        gt2D = (gt > 0).astype(np.uint8)

        if gt2D.sum() == 0:
            H, W = gt2D.shape
            bboxes = np.array([0, 0, W, H])
        else:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            embedding,
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


class AugmentedNpyDataset(Dataset):
    """Loads raw npy images + masks with augmentation for PT-SAM training.

    Unlike CachedEmbeddingDataset, returns raw images so the frozen encoder
    runs each forward pass. This allows augmentation of the input images
    before encoding, matching the paper's methodology.
    """

    def __init__(self, data_root, bbox_shift=20, augment=True):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            f for f in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(f)))
        ]
        self.bbox_shift = bbox_shift
        self.augment = augment
        self.transform = build_ptsam_augmentation() if augment else None
        mode = "augmented" if augment else "raw"
        print(f"AugmentedNpyDataset: {len(self.gt_path_files)} images from {data_root} ({mode})")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            os.path.join(self.img_path, img_name), allow_pickle=True
        ).astype(np.float32)  # writable float32 copy for augmentation
        gt = np.load(self.gt_path_files[index], allow_pickle=True)
        gt2D = (gt > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img_1024, mask=gt2D)
            img_1024 = np.clip(augmented["image"], 0.0, 1.0)
            gt2D = augmented["mask"]

        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, H, W)

        # Compute bbox from (augmented) mask
        if gt2D.sum() == 0:
            H, W = gt2D.shape
            bboxes = np.array([0, 0, W, H])
        else:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )
