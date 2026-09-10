"""
Master Orchestration Script.
Executes:
1. Model training pipeline (ScoreNet, Classifier, PGD-AT, DAE)
2. Benchmark evaluation across corruptions and adversarial perturbations
3. Ablation experiments (step budgets, guidance scale, component breakdown)
4. Figure visualization export into Paper/figs/
"""

import os
import sys
import time
import json
import argparse
from config import Config
from train import run_training
from evaluate import BenchmarkEvaluator
from visualize import (
    generate_fig1_framework,
    generate_fig2_phase_plane,
    generate_fig3_robustness_radar,
    generate_fig4_qualitative_gallery,
    generate_fig5_ablations
)

def main():
    parser = argparse.ArgumentParser(description="SGMP Master Orchestrator")
    parser.add_argument("--profile", choices=["a100", "colab", "cpu", "smoke_test"], default=None, help="Execution profile")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of all models even if checkpoints exist")
    parser.add_argument("--force_eval", action="store_true", help="Force re-running benchmark evaluation even if results exist")
    parser.add_argument("--full_test", action="store_true", help="Evaluate across entire test set")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    if args.profile:
        Config.apply_profile(args.profile)
    if args.device:
        Config.DEVICE = args.device
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.full_test:
        Config.FULL_DATASET = True
        Config.TEST_SAMPLES = None

    print("=" * 80)
    print("SCORE-GUIDED MANIFOLD PROJECTION (SGMP) - FULL EXECUTION PIPELINE")
    print(f"Device: {Config.DEVICE} | Batch Size: {Config.BATCH_SIZE} | Full Dataset: {Config.FULL_DATASET}")
    print("=" * 80)
    start_time = time.time()
    
    # 1. Training
    ckpt_score = os.path.join(Config.CHECKPOINTS_DIR, "score_net.pt")
    ckpt_clf = os.path.join(Config.CHECKPOINTS_DIR, "classifier.pt")
    
    if args.retrain or not (os.path.exists(ckpt_score) and os.path.exists(ckpt_clf)):
        print("\n--- PHASE 1: MODEL TRAINING ---")
        run_training()
    else:
        print("\n--- PHASE 1: CHECKPOINTS DETECTED (use --retrain to force re-training) ---")
        
    # 2. Benchmarking
    print("\n--- PHASE 2: BENCHMARK EVALUATION & ABLATION STUDY ---")
    evaluator = BenchmarkEvaluator(device=Config.DEVICE, batch_size=Config.BATCH_SIZE)
    bench_path = os.path.join(Config.RESULTS_DIR, "benchmark_results.json")
    
    if not args.force_eval and os.path.exists(bench_path) and not args.retrain:
        print(f"Loading existing benchmark results from {bench_path} (use --force_eval to re-evaluate)")
        with open(bench_path, "r") as f:
            benchmark_data = json.load(f)
    else:
        max_b = None if Config.FULL_DATASET else 6
        benchmark_data = evaluator.run_full_benchmark(max_batches=max_b)
        
    ablation_data = evaluator.run_ablation_study(max_batches=None if Config.FULL_DATASET else 3)
    
    # 3. Figure Generation
    print("\n--- PHASE 3: GENERATING PUBLICATION FIGURES FOR PAPER ---")
    generate_fig1_framework()
    generate_fig2_phase_plane()
    generate_fig3_robustness_radar(benchmark_data)
    generate_fig4_qualitative_gallery()
    generate_fig5_ablations(ablation_data)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE SUCCESSFULLY COMPLETED IN {total_time:.2f} SECONDS ({total_time/60:.2f} MINS).")
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print(f"Figures saved to: {Config.FIGS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

