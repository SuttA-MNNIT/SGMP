"""
Comprehensive Evaluation and Benchmarking Suite.
Evaluates:
- Standard Classifier (ERM)
- Adversarial Training (PGD-AT)
- DAE Purifier
- DiffPure (SDEdit / Nie et al. baseline)
- SGMP (Our Proposed Score-Guided Manifold Projection)

Computes Clean Accuracy, Corrupted Accuracy, Mean Corruption Error (mCE),
Relative Robustness Score (RRS), Structural Similarity (SSIM), PSNR, and Inference Latency.
"""

import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from config import Config
from data.corruptions import get_dataloaders, apply_corruption
from models.score_net import ScoreNet
from models.classifier import PatternClassifier
from models.sgmp import SGMPPurifier
from models.baselines import DenoisingAutoencoder, DiffPureBaseline, AdversarialTrainer

def compute_psnr(x_true, x_pred):
    mse = torch.mean((x_true - x_pred) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-8)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()

def compute_ssim_simple(x_true, x_pred):
    # Simplified standard SSIM calculation for 2D tensors
    mu_x = x_true.mean(dim=[2, 3], keepdim=True)
    mu_y = x_pred.mean(dim=[2, 3], keepdim=True)
    var_x = ((x_true - mu_x) ** 2).mean(dim=[2, 3], keepdim=True)
    var_y = ((x_pred - mu_y) ** 2).mean(dim=[2, 3], keepdim=True)
    cov_xy = ((x_true - mu_x) * (x_pred - mu_y)).mean(dim=[2, 3], keepdim=True)
    
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim.mean().item()

def get_amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if Config.AMP_DTYPE == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def autocast_context(use_amp, device_type, amp_dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype)
    return torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)

class BenchmarkEvaluator:
    def __init__(self, device=None, batch_size=None):
        dev = device if device is not None else Config.DEVICE
        self.device = torch.device(dev)
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.use_amp = (self.device.type == "cuda" and Config.USE_AMP)
        self.amp_dtype = get_amp_dtype()
        self.load_models()
        _, self.test_loader = get_dataloaders(batch_size=batch_size)
        
    def load_models(self):
        print(f"Loading trained models onto {self.device} (AMP: {self.use_amp}, dtype: {self.amp_dtype})...")
        self.classifier = PatternClassifier(Config.IN_CHANNELS, Config.NUM_CLASSES).to(self.device)
        self.classifier.load_state_dict(torch.load(os.path.join(Config.CHECKPOINTS_DIR, "classifier.pt"), map_location=self.device))
        self.classifier.eval()
        
        self.classifier_adv = PatternClassifier(Config.IN_CHANNELS, Config.NUM_CLASSES).to(self.device)
        self.classifier_adv.load_state_dict(torch.load(os.path.join(Config.CHECKPOINTS_DIR, "classifier_adv.pt"), map_location=self.device))
        self.classifier_adv.eval()
        
        self.score_net = ScoreNet(Config.IN_CHANNELS).to(self.device)
        self.score_net.load_state_dict(torch.load(os.path.join(Config.CHECKPOINTS_DIR, "score_net.pt"), map_location=self.device))
        self.score_net.eval()
        
        self.dae = DenoisingAutoencoder(Config.IN_CHANNELS).to(self.device)
        self.dae.load_state_dict(torch.load(os.path.join(Config.CHECKPOINTS_DIR, "dae.pt"), map_location=self.device))
        self.dae.eval()
        
        # Pipelines
        self.diffpure = DiffPureBaseline(self.score_net, self.classifier, Config)
        self.sgmp = SGMPPurifier(self.score_net, self.classifier, Config)

    def evaluate_batch(self, x_clean, y, corruption_name="clean", severity=3):
        """
        Runs evaluation on a batch for all 5 methods under a specific corruption.
        Returns accuracy and latency for each method.
        """
        x_clean = x_clean.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        B = x_clean.shape[0]
        
        # Apply corruption
        if corruption_name == "clean":
            x_corrupt = x_clean
        elif corruption_name == "adversarial_pgd":
            x_corrupt = AdversarialTrainer.generate_pgd_adversary(self.classifier, x_clean, y, eps=0.15, iters=8)
        else:
            x_corrupt = apply_corruption(x_clean, corruption_name, severity=severity)
            
        results = {}
        
        # 1. Standard Classifier (Vanilla ERM)
        t0 = time.time()
        with torch.no_grad():
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                logits_erm = self.classifier(x_corrupt)
            acc_erm = (torch.argmax(logits_erm, dim=1) == y).float().mean().item()
        lat_erm = (time.time() - t0) * 1000 / B
        results["Vanilla ERM"] = {"acc": acc_erm * 100.0, "latency_ms": lat_erm}
        
        # 2. Adversarial Training (PGD-AT)
        t0 = time.time()
        with torch.no_grad():
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                logits_adv = self.classifier_adv(x_corrupt)
            acc_adv = (torch.argmax(logits_adv, dim=1) == y).float().mean().item()
        lat_adv = (time.time() - t0) * 1000 / B
        results["PGD-AT"] = {"acc": acc_adv * 100.0, "latency_ms": lat_adv}
        
        # 3. DAE Purifier
        t0 = time.time()
        with torch.no_grad():
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                x_dae = self.dae(x_corrupt)
                logits_dae = self.classifier(x_dae)
            acc_dae = (torch.argmax(logits_dae, dim=1) == y).float().mean().item()
        lat_dae = (time.time() - t0) * 1000 / B
        results["DAE Purifier"] = {
            "acc": acc_dae * 100.0,
            "latency_ms": lat_dae,
            "psnr": compute_psnr(x_clean, x_dae),
            "ssim": compute_ssim_simple(x_clean, x_dae)
        }
        
        # 4. Standard DiffPure
        t0 = time.time()
        with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
            logits_dp, x_dp = self.diffpure(x_corrupt, num_steps=Config.DIFFPURE_STEPS)
        acc_dp = (torch.argmax(logits_dp, dim=1) == y).float().mean().item()
        lat_dp = (time.time() - t0) * 1000 / B
        results["DiffPure"] = {
            "acc": acc_dp * 100.0,
            "latency_ms": lat_dp,
            "psnr": compute_psnr(x_clean, x_dp),
            "ssim": compute_ssim_simple(x_clean, x_dp)
        }
        
        # 5. SGMP (Our method: CMLB + SMC + ATFS)
        t0 = time.time()
        with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
            logits_sgmp, x_sgmp = self.sgmp(x_corrupt, num_steps=Config.SGMP_STEPS)
        acc_sgmp = (torch.argmax(logits_sgmp, dim=1) == y).float().mean().item()
        lat_sgmp = (time.time() - t0) * 1000 / B
        results["SGMP (Ours)"] = {
            "acc": acc_sgmp * 100.0,
            "latency_ms": lat_sgmp,
            "psnr": compute_psnr(x_clean, x_sgmp),
            "ssim": compute_ssim_simple(x_clean, x_sgmp)
        }
        
        return results

    def run_full_benchmark(self, max_batches=None):

        """
        Runs comprehensive evaluation over clean, all 11 corruptions, and adversarial PGD.
        If max_batches is None, evaluates on the entire test set.
        """
        methods = ["Vanilla ERM", "PGD-AT", "DAE Purifier", "DiffPure", "SGMP (Ours)"]
        all_results = {m: {} for m in methods}
        
        print("\n" + "=" * 70)
        print("STARTING FULL BENCHMARK SUITE ACROSS CORRUPTIONS & ADVERSARIAL ATTACK")
        print("=" * 70)
        
        # 1. Clean Benchmark
        print("\n>>> Evaluating Clean Test Data...")
        clean_scores = {m: [] for m in methods}
        latencies = {m: [] for m in methods}
        for b_idx, (x_batch, y_batch) in enumerate(self.test_loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            res = self.evaluate_batch(x_batch, y_batch, corruption_name="clean")
            for m in methods:
                clean_scores[m].append(res[m]["acc"])
                latencies[m].append(res[m]["latency_ms"])
                
        for m in methods:
            all_results[m]["clean"] = float(np.mean(clean_scores[m]))
            all_results[m]["avg_latency_ms"] = float(np.mean(latencies[m]))
            print(f"  {m:16s} | Clean Acc: {all_results[m]['clean']:.2f}% | Latency: {all_results[m]['avg_latency_ms']:.2f} ms/sample")
            
        # 2. Corruptions Benchmark
        corruption_types = Config.CORRUPTION_TYPES
        corrupt_scores = {m: {c: [] for c in corruption_types} for m in methods}
        fidelity_scores = {"DAE Purifier": {"psnr": [], "ssim": []},
                           "DiffPure": {"psnr": [], "ssim": []},
                           "SGMP (Ours)": {"psnr": [], "ssim": []}}
        
        for c_type in corruption_types:
            print(f">>> Evaluating Corruption: {c_type} (Severity 3)...")
            for b_idx, (x_batch, y_batch) in enumerate(self.test_loader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                res = self.evaluate_batch(x_batch, y_batch, corruption_name=c_type, severity=3)
                for m in methods:
                    corrupt_scores[m][c_type].append(res[m]["acc"])
                    if m in fidelity_scores and "psnr" in res[m]:
                        fidelity_scores[m]["psnr"].append(res[m]["psnr"])
                        fidelity_scores[m]["ssim"].append(res[m]["ssim"])
                        
            for m in methods:
                mean_c_acc = float(np.mean(corrupt_scores[m][c_type]))
                all_results[m][c_type] = mean_c_acc
                
        # 3. Aggregate Metrics: mCE (Mean Corruption Error) & Relative Robustness
        # Standard baseline for mCE normalization is Vanilla ERM
        erm_errors = {c: 100.0 - all_results["Vanilla ERM"][c] for c in corruption_types}
        
        for m in methods:
            corrupt_accs = [all_results[m][c] for c in corruption_types]
            all_results[m]["mean_corrupted_acc"] = float(np.mean(corrupt_accs))
            all_results[m]["relative_robustness"] = float(all_results[m]["mean_corrupted_acc"] / all_results[m]["clean"])
            
            # Compute mCE relative to ERM
            ce_ratios = []
            for c in corruption_types:
                err_m = 100.0 - all_results[m][c]
                err_erm = max(1.0, erm_errors[c])
                ce_ratios.append(err_m / err_erm)
            all_results[m]["mCE"] = float(np.mean(ce_ratios) * 100.0)
            
            if m in fidelity_scores:
                all_results[m]["psnr"] = float(np.mean(fidelity_scores[m]["psnr"]))
                all_results[m]["ssim"] = float(np.mean(fidelity_scores[m]["ssim"]))
                
        # Print Summary Table
        print("\n" + "=" * 85)
        print(f"{'Method':16s} | {'Clean Acc':9s} | {'Corrupt Acc':11s} | {'mCE':7s} | {'RRS':6s} | {'Latency (ms)':12s}")
        print("-" * 85)
        for m in methods:
            print(f"{m:16s} | {all_results[m]['clean']:8.2f}% | {all_results[m]['mean_corrupted_acc']:10.2f}% | {all_results[m]['mCE']:6.1f} | {all_results[m]['relative_robustness']:5.3f} | {all_results[m]['avg_latency_ms']:10.2f}")
        print("=" * 85)
        
        # Save JSON
        with open(os.path.join(Config.RESULTS_DIR, "benchmark_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
            
        # Save CSV
        df = pd.DataFrame(all_results).T
        df.to_csv(os.path.join(Config.RESULTS_DIR, "benchmark_summary.csv"))
        print(f"Results written to {Config.RESULTS_DIR}")
        
        return all_results

    def run_ablation_study(self, max_batches=3):
        r"""
        Systematic Ablation Study:
        1. Component isolation:
           - Vanilla Classifier (No purification)
           - DiffPure (Score prior only, no CMLB, no SMC)
           - SGMP w/o SMC (Score prior + CMLB guidance, no contraction)
           - Full SGMP (Score prior + CMLB guidance + SMC)
        2. Solver step budget: K \in [1, 2, 3, 4, 6, 8, 16, 30]
        3. Guidance scale sensitivity: \lambda_0 \in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        """
        print("\n" + "=" * 70)
        print("EXECUTING ABLATION EXPERIMENTS")
        print("=" * 70)
        ablations = {}
        
        # Sample test data (use 256 samples for statistically robust curves)
        x_clean_list, y_list = [], []
        for b_idx, (x_b, y_b) in enumerate(self.test_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x_clean_list.append(x_b)
            y_list.append(y_b)
        x_all = torch.cat(x_clean_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        num_eval = min(256, x_all.shape[0])
        x_clean = x_all[:num_eval].to(self.device)
        y = y_all[:num_eval].to(self.device)
        
        # Corrupted input: Gaussian noise severity 3
        x_corrupt = apply_corruption(x_clean, "gaussian_noise", severity=3)
        
        # --- Experiment 1: Component Breakdown ---
        comp_results = {}
        # A: Vanilla
        with torch.no_grad():
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                acc_vanilla = (torch.argmax(self.classifier(x_corrupt), dim=1) == y).float().mean().item() * 100.0
        comp_results["Vanilla (No Purif)"] = acc_vanilla
        
        # B: Score Prior Only (DiffPure)
        with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
            x_dp4, _ = self.diffpure.purify(x_corrupt, num_steps=Config.SGMP_STEPS)
            acc_dp4 = (torch.argmax(self.classifier(x_dp4.to(self.device)), dim=1) == y).float().mean().item() * 100.0
        comp_results[f"Score Prior Only ({Config.SGMP_STEPS} steps)"] = acc_dp4
        
        # C: Score + CMLB (w/o SMC)
        class SGMP_NoSMC(SGMPPurifier):
            def apply_spectral_manifold_contraction(self, joint_score, score_prior, cls_grad):
                return joint_score  # No contraction
                
        sgmp_no_smc = SGMP_NoSMC(self.score_net, self.classifier, Config)
        with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
            logits_nosmc, _ = sgmp_no_smc(x_corrupt, num_steps=Config.SGMP_STEPS)
            acc_nosmc = (torch.argmax(logits_nosmc, dim=1) == y).float().mean().item() * 100.0
        comp_results["Score + CMLB (No SMC)"] = acc_nosmc
        
        # D: Full SGMP (Score + CMLB + SMC)
        with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
            logits_full, _ = self.sgmp(x_corrupt, num_steps=Config.SGMP_STEPS)
            acc_full = (torch.argmax(logits_full, dim=1) == y).float().mean().item() * 100.0
        comp_results["Full SGMP (Score + CMLB + SMC)"] = acc_full
        
        ablations["component_breakdown"] = comp_results
        print("Component Breakdown Accuracies:", comp_results, flush=True)
        
        # --- Experiment 2: Step Budget Analysis ---
        step_budgets = [1, 2, 3, 4, 6, 8, 12]
        step_accs_sgmp = []
        step_lats_sgmp = []
        step_accs_dp = []
        step_lats_dp = []
        
        for k in step_budgets:
            # SGMP
            t0 = time.time()
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                l_sgmp, _ = self.sgmp(x_corrupt, num_steps=k)
            t_sgmp = (time.time() - t0) * 1000 / x_corrupt.shape[0]
            acc_k = (torch.argmax(l_sgmp, dim=1) == y).float().mean().item() * 100.0
            step_accs_sgmp.append(acc_k)
            step_lats_sgmp.append(t_sgmp)
            
            # DiffPure
            t0 = time.time()
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                l_dp, _ = self.diffpure(x_corrupt, num_steps=k)
            t_dp = (time.time() - t0) * 1000 / x_corrupt.shape[0]
            acc_dp_k = (torch.argmax(l_dp, dim=1) == y).float().mean().item() * 100.0
            step_accs_dp.append(acc_dp_k)
            step_lats_dp.append(t_dp)
            
        ablations["step_budget"] = {
            "steps": step_budgets,
            "sgmp_acc": step_accs_sgmp,
            "sgmp_lat": step_lats_sgmp,
            "diffpure_acc": step_accs_dp,
            "diffpure_lat": step_lats_dp
        }
        
        # --- Experiment 3: Guidance Scale Sensitivity ---
        lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
        lambda_accs = []
        orig_scale = self.sgmp.config.GUIDANCE_SCALE
        for lam in lambdas:
            self.sgmp.config.GUIDANCE_SCALE = lam
            with autocast_context(self.use_amp, self.device_type, self.amp_dtype):
                l_lam, _ = self.sgmp(x_corrupt, num_steps=Config.SGMP_STEPS)
            acc_l = (torch.argmax(l_lam, dim=1) == y).float().mean().item() * 100.0
            lambda_accs.append(acc_l)
        self.sgmp.config.GUIDANCE_SCALE = orig_scale

        
        ablations["guidance_scale"] = {
            "scales": lambdas,
            "accuracies": lambda_accs
        }
        
        # Save ablations
        with open(os.path.join(Config.RESULTS_DIR, "ablation_results.json"), "w") as f:
            json.dump(ablations, f, indent=2)
            
        print("Ablation study finished and saved.")
        return ablations

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SGMP Benchmark Evaluator")
    parser.add_argument("--profile", choices=["a100", "colab", "cpu", "smoke_test"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--full_test", action="store_true")
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
        limit_b = None
    else:
        limit_b = args.max_batches if args.max_batches is not None else (None if Config.FULL_DATASET else 6)
        
    evaluator = BenchmarkEvaluator(device=Config.DEVICE, batch_size=Config.BATCH_SIZE)
    evaluator.run_full_benchmark(max_batches=limit_b)
    evaluator.run_ablation_study(max_batches=limit_b if limit_b is not None else 5)

