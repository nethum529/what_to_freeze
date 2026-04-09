"""
Unified training script for the freeze-strategy benchmark.

Trains all strategies with the full image encoder (not precomputed embeddings)
so the comparison reflects actual paper methodologies:

  - MedSAM:  freeze prompt_encoder, train image_encoder + mask_decoder  (~93.7M params)
  - PP-SAM:  freeze mask_decoder,   train image_encoder + prompt_encoder (~89.7M params)
  - PT-SAM:  freeze everything,     learn 8 prompt tokens only          (~2.0K params)

Usage:
    python scripts/train_benchmark.py --strategy medsam --num_epochs 10
    python scripts/train_benchmark.py --strategy ppsam  --num_epochs 10
    python scripts/train_benchmark.py --strategy ptsam  --num_epochs 10
"""

import argparse
import glob
import os
import random
import sys
import time
from os.path import join

import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MedSAM-main"))
from segment_anything import sam_model_registry


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            f for f in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(f)))
        ]
        self.bbox_shift = bbox_shift
        print(f"Number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(join(self.img_path, img_name), "r", allow_pickle=True)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0

        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(gt == random.choice(label_ids.tolist()))

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


# ---------------------------------------------------------------------------
# Strategy A: MedSAM -- freeze prompt encoder, train image encoder + mask decoder
# ---------------------------------------------------------------------------
class MedSAMStrategy(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )

    def get_trainable_params(self):
        return list(self.image_encoder.parameters()) + list(self.mask_decoder.parameters())


# ---------------------------------------------------------------------------
# Strategy B: PP-SAM -- freeze mask decoder, train image encoder + prompt encoder
#             Uses larger bbox perturbation (0-50px, expansion only)
# ---------------------------------------------------------------------------
class PPSAMStrategy(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=box_torch, masks=None,
        )
        # mask_decoder weights frozen via requires_grad=False but must stay
        # in the compute graph so gradients flow back through to prompt_encoder
        # and image_encoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )

    def get_trainable_params(self):
        return list(self.image_encoder.parameters()) + list(self.prompt_encoder.parameters())


# ---------------------------------------------------------------------------
# Strategy C: PT-SAM -- freeze everything, learn 8 prompt tokens (2,048 params)
# ---------------------------------------------------------------------------
class PTSAMStrategy(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, num_tokens=8):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        embed_dim = prompt_encoder.embed_dim  # 256
        self.learned_tokens = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

    def forward(self, image, box=None):
        with torch.no_grad():
            image_embedding = self.image_encoder(image)

        B = image_embedding.shape[0]
        sparse_embeddings = self.learned_tokens.unsqueeze(0).expand(B, -1, -1)

        with torch.no_grad():
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(B, -1, self.prompt_encoder.image_embedding_size[0],
                     self.prompt_encoder.image_embedding_size[1])

        # mask_decoder must be in compute graph for gradients to flow to learned_tokens
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )

    def get_trainable_params(self):
        return [self.learned_tokens]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
STRATEGY_MAP = {
    "medsam": (MedSAMStrategy, 20),   # (class, bbox_shift)
    "ppsam": (PPSAMStrategy, 50),
    "ptsam": (PTSAMStrategy, 20),
}


def train(args):
    # NOTE: This is a legacy standalone script. For the reproducible benchmark
    # with paper-recommended per-strategy hyperparameters, use run_benchmark.py.
    device = torch.device(args.device)
    strategy_cls, bbox_shift = STRATEGY_MAP[args.strategy]

    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = strategy_cls(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"Strategy: {args.strategy}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    dataset = NpyDataset(args.data_path, bbox_shift=bbox_shift)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    save_dir = join(args.work_dir, "benchmark", f"{args.strategy}_kvasir")
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    losses = []
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for step, (image, gt2D, boxes, _) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        ):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            with torch.amp.autocast("cuda"):
                pred = model(image, boxes_np)
                gt2D_float = gt2D.float()
                loss = seg_loss(pred, gt2D_float) + ce_loss(pred, gt2D_float)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {epoch_loss:.4f} | Time: {elapsed/60:.1f}m")

        checkpoint = {"model": model.state_dict(), "epoch": epoch, "strategy": args.strategy}
        torch.save(checkpoint, join(save_dir, "model_latest.pth"))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, join(save_dir, "model_best.pth"))

    total_time = time.time() - start_time

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title(f"{args.strategy} Training Loss (Kvasir-SEG)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice + BCE Loss")
    plt.grid(True)
    plt.savefig(join(save_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    np.save(join(save_dir, "losses.npy"), np.array(losses))
    metadata = {
        "strategy": args.strategy,
        "trainable_params": trainable_count,
        "total_params": total_params,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_loss": best_loss,
        "final_loss": losses[-1],
        "training_time_seconds": total_time,
        "num_samples": len(dataset),
    }
    np.save(join(save_dir, "metadata.npy"), metadata)

    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["medsam", "ppsam", "ptsam"])
    parser.add_argument("--data_path", default="data/npy/KvasirSEG_train")
    parser.add_argument("--checkpoint", default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--work_dir", default="work_dir")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
