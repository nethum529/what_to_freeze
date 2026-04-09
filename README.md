# What to Freeze: A Unified Benchmark for SAM Fine-Tuning Strategies in Medical Image Segmentation

## The Problem

The Segment Anything Model (SAM) has become a popular foundation for medical image segmentation, and several recent papers propose different ways to adapt it. MedSAM (Ma et al., Nature Communications 2024) freezes the prompt encoder and trains the image encoder and mask decoder (~93.7M parameters). PP-SAM (Rahman et al., CVPRW 2024) takes the opposite approach, freezing the mask decoder and training the image encoder and prompt encoder (~89.7M parameters). PT-SAM (Piater et al., CVPRW 2025) goes further and freezes the entire model, learning only 8 prompt tokens (~2,048 parameters).

Each of these papers reports strong results. The issue is that they all evaluate on different datasets, with different metrics, under different conditions. There is no way to tell from the existing literature which strategy actually works best, because no one has compared them on the same data with the same evaluation protocol. Each paper selects the comparison that makes its method look strongest, and the result is a fragmented landscape where claims of improvement are difficult to verify.

## Our Approach

This project builds a benchmark that evaluates all three freeze strategies under identical conditions. We use the ETIS-LaribPolypDB dataset (196 colonoscopy images, split 160/36 for training and testing with a fixed seed) and standardise everything that should be standardised: the dataset, the data split, the base model (SAM ViT-B), and the evaluation metrics (Dice coefficient, IoU, and Hausdorff Distance at the 95th percentile).

Critically, we do **not** standardise the training hyperparameters across strategies. Each method uses its own paper-recommended learning rate, weight decay, epoch count, loss function weighting, and augmentation policy. This is a deliberate design choice. The training recipe is part of the method -- forcing a single learning rate on both a 93.7M-parameter model and a 2,048-parameter model would not be a fair comparison. What we standardise is the evaluation, not the training.

| Strategy | What's Frozen | What's Trained | Trainable Params | Epochs | Learning Rate |
|----------|---------------|----------------|-----------------|--------|---------------|
| MedSAM | Prompt encoder | Image encoder + mask decoder | ~93.7M | 25 | 1e-4 |
| PP-SAM | Mask decoder | Image encoder + prompt encoder | ~89.7M | 100 | 1e-4 |
| PT-SAM | Everything | 8 learned prompt tokens | ~2,048 | 312 | 0.05 |

## What We Hope to Achieve

The immediate goal is a clear, reproducible answer to the question: given the same medical imaging task, which freeze strategy produces the best segmentation? Beyond that, we want this benchmark to serve as a framework that can be extended with new strategies and new datasets over time.

Longer-term, we are working towards a research publication that contributes both the benchmark itself and a novel evaluation metric that captures aspects of segmentation quality that Dice and IoU miss. Current metrics treat all pixels equally, but in clinical practice, errors near organ boundaries or in small structures matter far more than errors in the interior of a large region.

## Future Directions

- **More datasets and modalities.** ETIS covers colonoscopy polyps, but the findings may not generalise to CT, MRI, X-ray, or ultrasound. Expanding the benchmark across modalities would strengthen any conclusions significantly.
- **Base SAM baseline.** Adding a zero-shot SAM evaluation (no fine-tuning at all, just the pretrained checkpoint with bounding box prompts) would answer the question of whether fine-tuning even helps on this dataset.
- **Matched-parameter comparisons.** PT-SAM trains 2,048 parameters while MedSAM trains 93.7 million. Comparing against LoRA or adapter methods at matched parameter counts would help isolate whether the advantage comes from the method or simply from the number of parameters.
- **Prompt robustness testing.** PP-SAM showed that SAM is sensitive to bounding box quality, but this was only tested on polyp images. Evaluating robustness across modalities with varying levels of prompt perturbation is an open problem.
- **Novel evaluation metric.** Existing metrics (Dice, IoU) are volume-based and do not penalise boundary errors proportionally. Developing a metric that weights clinically significant regions more heavily would be a meaningful contribution.

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU (the benchmark uses ~2 GB VRAM)
- [Git LFS](https://git-lfs.com/) (model checkpoints are stored with LFS)

### Setup

```bash
# Clone the repo (includes LFS files)
git lfs install
git clone https://github.com/nethum529/what_to_freeze.git
cd what_to_freeze

# Create the conda environment
conda create -n medsam python=3.10 -y
conda activate medsam

# Install all dependencies
pip install -r requirements.txt
pip install -e ./MedSAM-main
```

You will also need the pretrained SAM ViT-B checkpoint. Download `sam_vit_b_01ec64.pth` from the [SAM model zoo](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place it in `work_dir/SAM/`.

### Preprocessing

Before training, convert the raw ETIS images to the numpy format used by the benchmark:

```bash
python scripts/preprocess_etis.py
```

This produces 160 training and 36 test samples in `data/npy/`.

### Running the Benchmark

**Train and evaluate all three strategies:**

```bash
python scripts/run_benchmark.py
```

This will cache image encoder embeddings (for PT-SAM), train each strategy with its paper-recommended hyperparameters, evaluate on the test set, and print a comparison table.

**Run a single strategy:**

```bash
python scripts/run_benchmark.py --strategy medsam
python scripts/run_benchmark.py --strategy ppsam
python scripts/run_benchmark.py --strategy ptsam
```

**Quick test with fewer epochs:**

```bash
python scripts/run_benchmark.py --epochs 5
```

**Evaluate only (skip training, use existing checkpoints):**

```bash
python scripts/run_benchmark.py --skip-train
```

**Train a single strategy with a custom epoch count:**

```bash
python scripts/train_benchmark.py --strategy medsam --num_epochs 10
```

### Visualising Results

After evaluation, generate comparison figures:

```bash
python scripts/visualize_etis.py
```

This produces training curves, metric bar charts, score distributions, and a parameter-efficiency scatter plot in `results/etis/`.

## Repository Structure

```
what_to_freeze/
├── scripts/
│   ├── benchmark/          # Core benchmark modules
│   │   ├── config.py       # Hyperparameters (per-strategy, paper-cited)
│   │   ├── strategies.py   # MedSAM, PP-SAM, PT-SAM strategy classes
│   │   ├── trainer.py      # Training loop with AMP and validation
│   │   ├── evaluator.py    # Dice, IoU, HD95 evaluation
│   │   └── dataset.py      # Dataset loaders and augmentation
│   ├── run_benchmark.py    # Full pipeline orchestrator
│   ├── train_benchmark.py  # Standalone training script
│   ├── evaluate.py         # Standalone evaluation script
│   ├── preprocess_etis.py  # Raw PNG to npy conversion
│   └── visualize_etis.py   # Result visualisation
├── MedSAM-main/            # Upstream MedSAM codebase (SAM + medical fine-tuning)
├── data/
│   └── ETIS-LaribPolypDB/  # Raw dataset (196 images + masks)
└── work_dir/
    └── benchmark_etis/     # Trained checkpoints and results
```

## Acknowledgements

This project builds on [MedSAM](https://github.com/bowang-lab/MedSAM) by Jun Ma et al. and the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) by Meta AI. The ETIS-LaribPolypDB dataset is from Silva et al. (2014).
