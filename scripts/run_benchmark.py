"""
Top-level orchestrator for the ETIS freeze-strategy benchmark.

Runs four SAM strategies end-to-end:
  1. Evaluate Base SAM (zero-shot, no training)
  2. Cache PT-SAM embeddings
  3. Train PT-SAM, MedSAM, PP-SAM
  4. Evaluate all on test set
  5. Generate comparison table and visualizations

Usage:
    python scripts/run_benchmark.py                    # full benchmark
    python scripts/run_benchmark.py --strategy basesam  # zero-shot baseline only
    python scripts/run_benchmark.py --strategy medsam   # single strategy
    python scripts/run_benchmark.py --skip-train        # evaluate only
    python scripts/run_benchmark.py --epochs 5          # quick test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MedSAM-main"))

from benchmark.cache_embeddings import precompute_embeddings
from benchmark.config import BenchmarkConfig
from benchmark.evaluator import evaluate_strategy
from benchmark.trainer import train_strategy


def print_comparison_table(all_eval, all_train):
    """Print formatted comparison table."""
    strategies = ["basesam", "medsam", "ppsam", "ptsam"]
    labels = {"basesam": "Base SAM", "medsam": "MedSAM", "ppsam": "PP-SAM", "ptsam": "PT-SAM"}

    n_test = 0
    for s in strategies:
        if s in all_eval:
            n_test = all_eval[s]["summary"]["n_images"]
            break

    print(f"\n{'='*90}")
    print(f"  BENCHMARK RESULTS: ETIS-LaribPolypDB (N={n_test} test images)")
    print(f"{'='*90}")
    header = (
        f"{'Strategy':<10} | {'Trainable':>12} | {'Dice':>12} | {'IoU':>12} | "
        f"{'HD95':>12} | {'Time/img':>10} | {'Train Time':>10} | {'Mem (MB)':>8}"
    )
    print(header)
    print("-" * len(header))

    for s in strategies:
        if s not in all_eval:
            continue
        e = all_eval[s]["summary"]
        t = all_train.get(s, {})

        params = t.get("trainable_params", 0)
        if params >= 1e6:
            params_str = f"{params/1e6:.1f}M"
        elif params >= 1e3:
            params_str = f"{params/1e3:.1f}K"
        else:
            params_str = str(params)

        train_time = t.get("total_time_seconds", 0)
        if train_time >= 3600:
            time_str = f"{train_time/3600:.1f}h"
        elif train_time >= 60:
            time_str = f"{train_time/60:.1f}m"
        else:
            time_str = f"{train_time:.0f}s"

        print(
            f"{labels[s]:<10} | {params_str:>12} | "
            f"{e['dice_mean']:.3f}±{e['dice_std']:.2f} | "
            f"{e['iou_mean']:.3f}±{e['iou_std']:.2f} | "
            f"{e['hd95_mean']:>5.1f}±{e['hd95_std']:<4.1f} | "
            f"{e['inference_ms_mean']:>7.0f}ms | "
            f"{time_str:>10} | "
            f"{e['peak_memory_mb']:>7.0f}"
        )

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="ETIS Freeze-Strategy Benchmark")
    parser.add_argument("--strategy", choices=["basesam", "medsam", "ppsam", "ptsam"],
                        help="Run single strategy (default: all four)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, evaluate existing checkpoints")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device")
    args = parser.parse_args()

    # Build config with overrides
    overrides = {}
    if args.epochs is not None:
        overrides["MEDSAM_NUM_EPOCHS"] = args.epochs
        overrides["PPSAM_NUM_EPOCHS"] = args.epochs
        overrides["PTSAM_NUM_EPOCHS"] = args.epochs
    if args.batch_size is not None:
        overrides["MEDSAM_BATCH_SIZE"] = args.batch_size
        overrides["PPSAM_BATCH_SIZE"] = args.batch_size
        overrides["PTSAM_BATCH_SIZE"] = args.batch_size
    if args.device is not None:
        overrides["DEVICE"] = args.device
    config = BenchmarkConfig(**overrides)

    strategies = [args.strategy] if args.strategy else ["basesam", "ptsam", "medsam", "ppsam"]

    # Verify prerequisites
    assert os.path.exists(config.SAM_CHECKPOINT), f"SAM checkpoint not found: {config.SAM_CHECKPOINT}"
    assert os.path.exists(config.TRAIN_DATA), f"Training data not found: {config.TRAIN_DATA}. Run: python scripts/preprocess_etis.py"
    assert os.path.exists(config.TEST_DATA), f"Test data not found: {config.TEST_DATA}"

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.WORK_DIR, exist_ok=True)

    benchmark_start = time.time()
    all_train = {}
    all_eval = {}

    # Phase 1: Cache embeddings for PT-SAM (skipped when augmentation is on)
    if "ptsam" in strategies and not args.skip_train and not config.PTSAM_AUGMENTATION:
        cache_dir = os.path.join(config.WORK_DIR, "ptsam_cache")
        print("\n[Phase 1] Caching PT-SAM embeddings...")
        precompute_embeddings(
            config.SAM_CHECKPOINT, config.TRAIN_DATA, cache_dir, config.DEVICE,
        )
    elif "ptsam" in strategies and config.PTSAM_AUGMENTATION:
        print("\n[Phase 1] Skipping embedding cache (augmentation enabled, encoder runs per-batch)")

    # Phase 2: Training (skip basesam — zero-shot, no training)
    trainable_strategies = [s for s in strategies if s != "basesam"]
    if not args.skip_train and trainable_strategies:
        print(f"\n[Phase 2] Training {len(trainable_strategies)} strategies...")
        for strat in trainable_strategies:
            result = train_strategy(strat, config)
            all_train[strat] = result
    else:
        # Load existing training results
        for strat in trainable_strategies:
            result_path = os.path.join(config.WORK_DIR, strat, "train_result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    all_train[strat] = json.load(f)

    # Phase 3: Evaluation
    if not args.skip_eval:
        print(f"\n[Phase 3] Evaluating on test set...")
        for strat in strategies:
            result = evaluate_strategy(strat, config)
            if result is not None:
                all_eval[strat] = result
    else:
        # Load existing eval results
        results_path = os.path.join(config.RESULTS_DIR, "benchmark_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_eval = json.load(f)

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_eval, f, indent=2)

    # Phase 4: Comparison table
    if all_eval:
        print_comparison_table(all_eval, all_train)

    total_time = time.time() - benchmark_start
    print(f"\nTotal benchmark time: {total_time/60:.1f} minutes")

    if all_train:
        print("\nPer-strategy training times:")
        for strat, result in all_train.items():
            t = result.get("total_time_seconds", 0)
            print(f"  {strat:>8}: {t/60:.1f}m")

    print(f"\nResults saved to: {results_path}")
    print("Run visualization: python scripts/visualize_etis.py")


if __name__ == "__main__":
    main()
