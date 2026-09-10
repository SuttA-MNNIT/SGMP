"""
Model Training Pipeline:
Optimized for NVIDIA A100 GPU (bfloat16/float16 AMP, Tensor Cores, Cosine LR, EMA).
Trains:
1. PatternClassifier (Standard ERM)
2. PGD Adversarially-Trained Classifier (PGD-AT)
3. Generative ScoreNet (Continuous-time Score Matching with EMA)
4. Denoising Autoencoder (DAE Purifier)
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config
from data.corruptions import get_dataloaders
from models.score_net import ScoreNet, ScoreNetEMA
from models.classifier import PatternClassifier
from models.baselines import DenoisingAutoencoder, AdversarialTrainer

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

def get_grad_scaler(use_amp, amp_dtype):
    enabled = (use_amp and amp_dtype == torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

def train_score_net(score_net, train_loader, epochs=None, device=None):
    if epochs is None:
        epochs = Config.EPOCHS_SCORE_NET
    if device is None:
        device = torch.device(Config.DEVICE)
        
    print(f"\n>>> [1/4] Training Generative ScoreNet via Multi-Corruption EDM ({epochs} epochs, device={device})...", flush=True)
    score_net.train()
    optimizer = optim.AdamW(score_net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.COSINE_LR_MIN)
    
    use_amp = (device.type == "cuda" and Config.USE_AMP)
    amp_dtype = get_amp_dtype()
    scaler = get_grad_scaler(use_amp, amp_dtype)
    
    ema = ScoreNetEMA(score_net, decay=Config.EMA_DECAY) if Config.USE_EMA else None
    sigma_data = Config.SIGMA_DATA
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        t_epoch = time.time()
        
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            
            # Sample continuous noise levels from log-normal distribution (Karras et al.)
            # P_mean = -1.2, P_std = 1.2
            rnd_normal = torch.randn((B,), device=device)
            sigmas = torch.exp(rnd_normal * 1.2 - 1.2)
            sigmas = torch.clamp(sigmas, Config.SIGMA_MIN, Config.SIGMA_MAX)
            
            noise = torch.randn_like(x)
            
            # Multi-corruption augmentation: train score net on diverse corruption types
            # so it can denoise blur, contrast shifts, fog, impulse noise, and cutout
            aug_choice = torch.rand(1).item()
            if aug_choice < 0.12:
                # Impulse noise blend (salt-and-pepper)
                mask = (torch.rand_like(x) < 0.10).float()
                salt = (torch.rand_like(x) < 0.5).float()
                x_base = x * (1.0 - mask) + salt * mask
            elif aug_choice < 0.24:
                # Random cutout / occlusion
                x_base = x.clone()
                cx = torch.randint(4, 20, (1,)).item()
                cy = torch.randint(4, 20, (1,)).item()
                x_base[:, :, cy:cy+6, cx:cx+6] = 0.0
            elif aug_choice < 0.38:
                # Defocus blur via Gaussian smoothing (σ ∈ [0.8, 2.5])
                x_base = x.clone()
                blur_sigma = 0.8 + torch.rand(1).item() * 1.7
                kernel_size = int(2 * round(2 * blur_sigma) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = max(3, min(kernel_size, 11))
                # Create 1D Gaussian kernel and apply separable convolution
                coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
                gauss_1d = torch.exp(-coords ** 2 / (2 * blur_sigma ** 2))
                gauss_1d = gauss_1d / gauss_1d.sum()
                gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
                gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size).expand(x.shape[1], -1, -1, -1)
                pad_size = kernel_size // 2
                x_padded = F.pad(x, [pad_size] * 4, mode='reflect')
                x_base = F.conv2d(x_padded, gauss_2d, groups=x.shape[1])
                x_base = torch.clamp(x_base, 0.0, 1.0)
            elif aug_choice < 0.50:
                # Contrast reduction (c ∈ [0.25, 0.70])
                c_factor = 0.25 + torch.rand(1).item() * 0.45
                means = x.mean(dim=[2, 3], keepdim=True)
                x_base = (x - means) * c_factor + means
                x_base = torch.clamp(x_base, 0.0, 1.0)
            elif aug_choice < 0.60:
                # Fog-like luminance shift (additive uniform brightness)
                fog_strength = 0.15 + torch.rand(1).item() * 0.35
                # Create a smooth fog gradient
                h, w = x.shape[2], x.shape[3]
                fog_y = torch.linspace(0.3, 0.8, h, device=device).view(1, 1, h, 1)
                fog_x = torch.linspace(0.3, 0.8, w, device=device).view(1, 1, 1, w)
                fog_layer = fog_y * fog_x
                x_base = (1.0 - fog_strength) * x + fog_strength * fog_layer
                x_base = torch.clamp(x_base, 0.0, 1.0)
            else:
                x_base = x
                
            x_perturbed = x_base + sigmas.view(B, 1, 1, 1) * noise
            
            # EDM effective loss weight
            weight = (sigmas ** 2 + sigma_data ** 2) / ((sigmas * sigma_data) ** 2 + 1e-6)
            weight = torch.clamp(weight, 0.1, 100.0)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp, "cuda" if device.type == "cuda" else "cpu", amp_dtype):
                denoised = score_net(x_perturbed, sigmas)
                loss = torch.mean(weight.view(B, 1, 1, 1) * ((denoised - x) ** 2))
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
                optimizer.step()
                
            if ema is not None:
                ema.update()
                
            total_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - t_epoch
        print(f"  [ScoreNet] Epoch {epoch:02d}/{epochs:02d} | EDM Loss: {avg_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.2f}s", flush=True)
            
    if ema is not None:
        print("  Applying EMA weights to ScoreNet...", flush=True)
        ema.apply_shadow()
        
    torch.save(score_net.state_dict(), os.path.join(Config.CHECKPOINTS_DIR, "score_net.pt"))
    print("ScoreNet checkpoint saved successfully.", flush=True)


def train_classifier(classifier, train_loader, adversarial=False, epochs=None, device=None):
    if epochs is None:
        epochs = Config.EPOCHS_ADV if adversarial else Config.EPOCHS_CLASSIFIER
    if device is None:
        device = torch.device(Config.DEVICE)
        
    tag = "PGD-AT Classifier" if adversarial else "Standard Classifier (ERM)"
    step_num = "[3/4]" if adversarial else "[2/4]"
    print(f"\n>>> {step_num} Training {tag} ({epochs} epochs, device={device})...", flush=True)
    classifier.train()
    optimizer = optim.AdamW(classifier.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.COSINE_LR_MIN)
    criterion = nn.CrossEntropyLoss()
    
    use_amp = (device.type == "cuda" and Config.USE_AMP)
    amp_dtype = get_amp_dtype()
    scaler = get_grad_scaler(use_amp, amp_dtype)
    
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        t_epoch = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if adversarial:
                classifier.eval()
                x = AdversarialTrainer.generate_pgd_adversary(classifier, x, y, eps=0.15, iters=5)
                classifier.train()
                
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp, "cuda" if device.type == "cuda" else "cpu", amp_dtype):
                logits = classifier(x)
                loss = criterion(logits, y)
                
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        scheduler.step()
        acc = (correct / total) * 100.0
        elapsed = time.time() - t_epoch
        print(f"  [{tag}] Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Acc: {acc:.2f}% | {elapsed:.2f}s", flush=True)
            
    filename = "classifier_adv.pt" if adversarial else "classifier.pt"
    torch.save(classifier.state_dict(), os.path.join(Config.CHECKPOINTS_DIR, filename))
    print(f"{tag} saved to {filename}.", flush=True)


def train_dae(dae, train_loader, epochs=None, device=None):
    if epochs is None:
        epochs = Config.EPOCHS_DAE
    if device is None:
        device = torch.device(Config.DEVICE)
        
    print(f"\n>>> [4/4] Training Denoising Autoencoder (DAE Purifier) ({epochs} epochs, device={device})...", flush=True)
    dae.train()
    optimizer = optim.AdamW(dae.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.COSINE_LR_MIN)
    criterion = nn.MSELoss()
    
    use_amp = (device.type == "cuda" and Config.USE_AMP)
    amp_dtype = get_amp_dtype()
    scaler = get_grad_scaler(use_amp, amp_dtype)
    
    for epoch in range(1, epochs + 1):
        total_loss, total = 0.0, 0
        t_epoch = time.time()
        
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            noisy_x = torch.clamp(x + 0.25 * torch.randn_like(x), 0.0, 1.0)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp, "cuda" if device.type == "cuda" else "cpu", amp_dtype):
                recon = dae(noisy_x)
                loss = criterion(recon, x)

                
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            
        scheduler.step()
        elapsed = time.time() - t_epoch
        print(f"  [DAE Purifier] Epoch {epoch:02d}/{epochs:02d} | Recon MSE: {total_loss/total:.5f} | {elapsed:.2f}s", flush=True)
            
    torch.save(dae.state_dict(), os.path.join(Config.CHECKPOINTS_DIR, "dae.pt"))
    print("DAE Purifier checkpoint saved.", flush=True)


def run_training(profile=None):
    if profile is not None:
        Config.apply_profile(profile)
        
    torch.manual_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Training initialized on: {device} | Profile: {profile or 'configured'}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"AMP: Enabled ({get_amp_dtype()}) | TF32: {Config.ALLOW_TF32}")
        
    train_loader, _ = get_dataloaders()
    
    t0 = time.time()
    # 1. Standard Classifier
    clf = PatternClassifier(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    train_classifier(clf, train_loader, adversarial=False, epochs=Config.EPOCHS_CLASSIFIER, device=device)
    
    # 2. Adversarial Trained Classifier
    clf_adv = PatternClassifier(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    train_classifier(clf_adv, train_loader, adversarial=True, epochs=Config.EPOCHS_ADV, device=device)
    
    # 3. Score Network
    score_net = ScoreNet(in_channels=Config.IN_CHANNELS).to(device)
    train_score_net(score_net, train_loader, epochs=Config.EPOCHS_SCORE_NET, device=device)
    
    # 4. DAE Purifier
    dae = DenoisingAutoencoder(in_channels=Config.IN_CHANNELS).to(device)
    train_dae(dae, train_loader, epochs=Config.EPOCHS_DAE, device=device)
    
    total_elapsed = time.time() - t0
    print(f"\n>>> All models successfully trained and checkpoints saved in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} mins).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGMP Training Pipeline")
    parser.add_argument("--profile", choices=["a100", "colab", "cpu", "smoke_test"], default=None, help="Execution preset")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs_clf", type=int, default=None, help="Classifier epochs")
    parser.add_argument("--epochs_score", type=int, default=None, help="ScoreNet epochs")
    parser.add_argument("--full_dataset", action="store_true", help="Train on complete 60k dataset")
    args = parser.parse_args()
    
    if args.profile:
        Config.apply_profile(args.profile)
    if args.device:
        Config.DEVICE = args.device
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs_clf:
        Config.EPOCHS_CLASSIFIER = args.epochs_clf
    if args.epochs_score:
        Config.EPOCHS_SCORE_NET = args.epochs_score
    if args.full_dataset:
        Config.FULL_DATASET = True
        Config.TRAIN_SAMPLES = None
        Config.TEST_SAMPLES = None
        
    run_training()

