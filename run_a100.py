"""
High-Throughput A100 GPU Execution Orchestrator.
Designed for NVIDIA A100 (40GB / 80GB) SXM4/PCIe instances on Google Colab / Slurm / Cloud.
Executes:
1. Hardware diagnostics & Tensor Core verification.
2. Full dataset training with bfloat16 Automatic Mixed Precision (AMP), TF32, and EMA.
3. 10,000-sample standardized corruption & adversarial benchmarking.
4. Step budget and guidance scale ablation sweeps.
5. Publication figure generation (PDF & PNG) for Elsevier Pattern Recognition.
6. Formatted LaTeX tables ready for direct insertion into Paper/main.tex.
"""

import os
import sys
import time
import json
import torch
import pandas as pd
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

def print_gpu_diagnostics():
    print("=" * 80)
    print("NVIDIA A100 ACCELERATION ENVIRONMENT DIAGNOSTICS")
    print("=" * 80)
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is NOT available! Running on CPU fallback.")
        return
        
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()
    
    print(f"GPU Model:             {device_name}")
    print(f"Total VRAM:            {total_mem:.2f} GB")
    print(f"Compute Capability:    {props.major}.{props.minor}")
    print(f"bfloat16 Support:      {bf16_ok} (Enabled for A100 Tensor Cores)")
    print(f"PyTorch Version:       {torch.__version__}")
    print(f"CUDA Version:          {torch.version.cuda}")
    print(f"TF32 Tensor Cores:     {Config.ALLOW_TF32}")
    print(f"CuDNN Benchmark:       {Config.CUDNN_BENCHMARK}")
    print("=" * 80)

def print_latex_table(results):
    print("\n" + "=" * 80)
    print("LATEX TABLE EXPORT (Table 1: Main Results for Paper/main.tex)")
    print("=" * 80)
    print(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Clean Acc (\%)} & \textbf{Corrupt Acc (\%)} & \textbf{mCE ($\downarrow$)} & \textbf{RRS ($\uparrow$)} & \textbf{PSNR (dB)} & \textbf{Latency (ms)} \\")
    print(r"\midrule")
    for m, d in results.items():
        clean = f"{d.get('clean', 0):.2f}"
        corrupt = f"{d.get('mean_corrupted_acc', 0):.2f}"
        mce = f"{d.get('mCE', 100):.1f}"
        rrs = f"{d.get('relative_robustness', 0):.3f}"
        psnr = f"{d.get('psnr', 0):.2f}" if 'psnr' in d else "--"
        lat = f"{d.get('avg_latency_ms', 0):.2f}"
        bold = (m == "SGMP (Ours)")
        if bold:
            print(f"\\textbf{{{m}}} & \\textbf{{{clean}}} & \\textbf{{{corrupt}}} & \\textbf{{{mce}}} & \\textbf{{{rrs}}} & \\textbf{{{psnr}}} & {lat} \\\\")
        else:
            print(f"{m} & {clean} & {corrupt} & {mce} & {rrs} & {psnr} & {lat} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular*}")
    print("=" * 80 + "\n")

def run_a100_pipeline(smoke_test=False, retrain=True):
    profile = "smoke_test" if smoke_test else "a100"
    Config.apply_profile(profile)
    print_gpu_diagnostics()
    
    t_start = time.time()
    
    # 1. Training Phase
    if retrain:
        print(f"\n[PHASE 1/3] Executing High-Throughput Training (Profile: {profile})...")
        run_training(profile=profile)
    else:
        print("\n[PHASE 1/3] Skipping training (using existing checkpoints in checkpoints/)...")
    
    # 2. Evaluation Phase
    print(f"\n[PHASE 2/3] Running Full Standardized Benchmark Evaluation...")
    evaluator = BenchmarkEvaluator(device=Config.DEVICE, batch_size=Config.BATCH_SIZE)
    max_b = 4 if smoke_test else None
    benchmark_results = evaluator.run_full_benchmark(max_batches=max_b)
    
    print("\n[PHASE 2b] Executing Ablation Sweeps...")
    ablation_results = evaluator.run_ablation_study(max_batches=max_b if smoke_test else 5)
    
    # 3. Figure Generation
    print("\n[PHASE 3/3] Generating High-Resolution Publication Figures...")
    generate_fig1_framework()
    generate_fig2_phase_plane()
    generate_fig3_robustness_radar(benchmark_results)
    generate_fig4_qualitative_gallery()
    generate_fig5_ablations(ablation_results)
    
    print_latex_table(benchmark_results)
    
    total_time = time.time() - t_start
    print("=" * 80)
    print(f"A100 WORKFLOW COMPLETED IN {total_time:.2f}s ({total_time/60:.2f} minutes).")
    print(f"All checkpoints: {Config.CHECKPOINTS_DIR}")
    print(f"All metrics:     {Config.RESULTS_DIR}")
    print(f"All figures:     {Config.FIGS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A100 Master Runner for SGMP")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test validation")
    parser.add_argument("--no_retrain", action="store_true", help="Skip training if checkpoints exist")
    args = parser.parse_args()
    
    run_a100_pipeline(smoke_test=args.smoke_test, retrain=(not args.no_retrain))
