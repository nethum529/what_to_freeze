"""Centralized configuration for the ETIS freeze-strategy benchmark.

Each strategy uses its paper-recommended hyperparameters.
The training recipe is part of the method — standardizing hyperparameters
across architectures with 2K vs 93M parameters would invalidate results.
Evaluation protocol (dataset, split, metrics, seed) is what's standardized.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    # --- Shared / infrastructure ---
    SEED: int = 42
    BBOX_SHIFT_MEDSAM: int = 20
    BBOX_SHIFT_PPSAM: int = 50
    BBOX_SHIFT_PTSAM: int = 20
    NUM_PROMPT_TOKENS: int = 8
    SAM_CHECKPOINT: str = "work_dir/SAM/sam_vit_b_01ec64.pth"
    TRAIN_DATA: str = "data/npy/ETIS_train"
    TEST_DATA: str = "data/npy/ETIS_test"
    WORK_DIR: str = "work_dir/benchmark_etis"
    RESULTS_DIR: str = "results/etis"
    DEVICE: str = "cuda:0"
    GPU_UTIL_WARN_THRESHOLD: float = 0.90
    GRADIENT_CHECKPOINT: bool = True
    AMP: bool = True
    VAL_SPLIT: float = 0.2
    NUM_WORKERS: int = 0

    # --- MedSAM (Ma et al., Nature Communications 2024) ---
    # Paper: AdamW, lr=1e-4, wd=0.01, batch=160 (20×A100), 150 epochs on 1.57M images
    # Loss: BCE + Dice (equal weight), no augmentation, bbox perturbation 0-20px
    MEDSAM_LR: float = 1e-4
    MEDSAM_NUM_EPOCHS: int = 25          # scaled from 150 for ETIS (160 images)
    MEDSAM_BATCH_SIZE: int = 1           # scaled from 160 for single GPU
    MEDSAM_WEIGHT_DECAY: float = 0.01
    MEDSAM_LOSS_CE_WEIGHT: float = 1.0
    MEDSAM_LOSS_DICE_WEIGHT: float = 1.0

    # --- PP-SAM (Rahman et al., CVPRW 2024) ---
    # Paper: AdamW, lr=1e-4, wd=1e-4, batch=1, 100 epochs on Kvasir (900 images)
    # Loss: CE + mIoU (weights unspecified), no augmentation, bbox perturbation 0-50px
    # No LR scheduler, no data augmentation
    PPSAM_LR: float = 1e-4
    PPSAM_NUM_EPOCHS: int = 100          # paper: 100 on full Kvasir
    PPSAM_BATCH_SIZE: int = 1
    PPSAM_WEIGHT_DECAY: float = 1e-4     # paper: 1e-4 (NOT 0.01)
    PPSAM_LOSS_CE_WEIGHT: float = 1.0    # paper: weights unspecified, using equal
    PPSAM_LOSS_DICE_WEIGHT: float = 1.0

    # --- PT-SAM (Piater et al., CVPRW 2025) ---
    # Paper: AdamW, lr=0.05, wd=0.01, batch=2, 1000 epochs × 20 steps = 20K steps
    # Loss: 0.2 CE + 0.8 Dice, cosine annealing, heavy augmentation
    PTSAM_LR: float = 0.05
    PTSAM_NUM_EPOCHS: int = 312          # mapped from 1000×20 steps to ETIS
    PTSAM_BATCH_SIZE: int = 2
    PTSAM_WEIGHT_DECAY: float = 0.01
    PTSAM_LOSS_CE_WEIGHT: float = 0.2
    PTSAM_LOSS_DICE_WEIGHT: float = 0.8
    PTSAM_AUGMENTATION: bool = True

    def get_strategy_hparams(self, strategy_name):
        """Return paper-recommended hyperparameters for a given strategy."""
        valid = {"medsam", "ppsam", "ptsam"}
        if strategy_name not in valid:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Must be one of: {sorted(valid)}"
            )
        prefix = strategy_name.upper()
        return {
            "lr": getattr(self, f"{prefix}_LR"),
            "num_epochs": getattr(self, f"{prefix}_NUM_EPOCHS"),
            "batch_size": getattr(self, f"{prefix}_BATCH_SIZE"),
            "weight_decay": getattr(self, f"{prefix}_WEIGHT_DECAY"),
            "dice_weight": getattr(self, f"{prefix}_LOSS_DICE_WEIGHT"),
            "ce_weight": getattr(self, f"{prefix}_LOSS_CE_WEIGHT"),
        }
