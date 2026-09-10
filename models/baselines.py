"""
Benchmark Baselines for Pattern Recognition under Corruptions:
1. Standard Classifier (Vanilla ERM)
2. Denoising Autoencoder (DAE Purifier)
3. Standard Diffusion Purification (DiffPure / SDEdit baseline)
4. Adversarial Training (PGD-AT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class DiffPureBaseline(nn.Module):
    """
    Standard Diffusion Purification (Nie et al., ICML 2022 / SDEdit).
    Adds forward noise and executes unconditional reverse SDE for T steps.
    Lacks class likelihood guidance and spectral contraction.
    """
    def __init__(self, score_net, classifier, config=Config):
        super().__init__()
        self.score_net = score_net
        self.classifier = classifier
        self.config = config

    def purify(self, x_corrupt, num_steps=None, return_trajectory=False):
        if num_steps is None:
            num_steps = self.config.DIFFPURE_STEPS

        device = x_corrupt.device
        sigma_start = self.config.PURIFY_NOISE_LEVEL
        noise = torch.randn_like(x_corrupt)
        x_t = torch.clamp(x_corrupt + sigma_start * noise, 0.0, 1.0).detach()

        sigmas = torch.linspace(sigma_start, self.config.SIGMA_MIN, num_steps + 1, device=device)
        trajectory = [x_t.detach().cpu()] if return_trajectory else None

        for i in range(num_steps):
            sigma_curr = sigmas[i].item()
            with torch.no_grad():
                # Unconditional score (lacks CMLB and SMC)
                score, _ = self.score_net.get_score(x_t, sigma_curr)
                score = score.detach()

            # Unconditional Tweedie step
            sig_sq = sigma_curr ** 2
            x_pred = torch.clamp(x_t + sig_sq * score, 0.0, 1.0)
            if i < (num_steps - 1):
                sigma_next = sigmas[i + 1].item()
                x_t = torch.clamp(x_pred + sigma_next * torch.randn_like(x_t) * 0.4, 0.0, 1.0)
            else:
                x_t = x_pred

            if return_trajectory:
                trajectory.append(x_t.detach().cpu())

        return x_t, trajectory

    def forward(self, x_corrupt, num_steps=None):
        purified, _ = self.purify(x_corrupt, num_steps=num_steps, return_trajectory=False)
        logits = self.classifier(purified)
        return logits, purified


class AdversarialTrainer:
    """
    Implements PGD Adversarial Training (Madry et al., ICLR 2018).
    Trains the classifier to minimize worst-case loss within an epsilon ball.
    """
    @staticmethod
    def generate_pgd_adversary(model, x, y, eps=0.15, alpha=0.03, iters=10):
        x_adv = x.clone().detach() + torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(x.device)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(iters):
            x_adv.requires_grad_()
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, min=0.0, max=1.0)

        return x_adv.detach()
