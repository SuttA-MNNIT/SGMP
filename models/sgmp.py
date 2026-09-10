"""
Score-Guided Manifold Projection (SGMP) Framework.
Bridging Diffusion Priors and Discriminative Likelihood for Robust Pattern Recognition.

Core Modules:
1. Epistemic-Adaptive Corruption Estimator & Gating
2. Confidence-Modulated Likelihood Bridge (CMLB)
3. Spectral Manifold Contraction (SMC)
4. Amortized Trajectory Fast Solver (ATFS) with Data Consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config

class SGMPPurifier(nn.Module):
    def __init__(self, score_net, classifier, config=Config):
        super().__init__()
        self.score_net = score_net
        self.classifier = classifier
        self.config = config

        # Fixed Laplacian filter for high-frequency noise estimation
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap)

    def estimate_corruption_severity(self, x):
        r"""
        Estimates corruption variance \hat{\sigma} from:
        (a) High-frequency residual variance via Laplacian convolution.
        (b) Classifier epistemic uncertainty: 1 - \kappa(x).
        """
        B = x.shape[0]
        # 1. Epistemic certainty \kappa(x) \in [0, 1]
        with torch.no_grad():
            certainty, probs = self.classifier.get_confidence(x)
            target_class = torch.argmax(probs, dim=1)

        # 2. High-frequency edge noise measurement
        edges = F.conv2d(x, self.lap_kernel.to(dtype=x.dtype, device=x.device), padding=1)
        edge_std = torch.std(edges, dim=[1, 2, 3]) # [B]

        # 3. Dynamic sigma estimation
        # If clean: certainty ~ 0.95, edge_std ~ 0.08 -> sigma_est ~ 0.02
        # If corrupted: certainty < 0.6, edge_std > 0.25 -> sigma_est ~ 0.20
        sigma_est = (1.0 - certainty) * 0.16 + torch.clamp(edge_std * 0.40, 0.0, 0.12)
        sigma_est = torch.clamp(sigma_est, self.config.MIN_PURIFY_SIGMA, self.config.MAX_PURIFY_SIGMA)
        return sigma_est, certainty, target_class, edge_std

    def compute_cmlb_score(self, x_t, sigma_t, certainty, target_class):
        r"""
        Confidence-Modulated Likelihood Bridge (CMLB):
        v_t = s_\theta(x_t, \sigma_t) + \lambda_t \nabla_{x_t} \log p(y^* | x_t)
        Dynamically scales the likelihood gradient relative to score magnitude.
        """
        B = x_t.shape[0]

        # 1. Generative score prior
        with torch.no_grad():
            score_prior, denoised_prior = self.score_net.get_score(x_t, sigma_t)
            score_prior = score_prior.detach()

        # 2. Discriminative log-likelihood gradient
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            logits = self.classifier(x_in)
            log_probs = F.log_softmax(logits, dim=1)
            sel_log_probs = log_probs.gather(1, target_class.unsqueeze(1)).squeeze(1)
            cls_grad = torch.autograd.grad(sel_log_probs.sum(), x_in, create_graph=False)[0].detach()

        # 3. Relative norm calibration
        s_norm = torch.norm(score_prior.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-6
        g_norm = torch.norm(cls_grad.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-6
        
        # Scale guidance gently by uncertainty (less guidance when already certain)
        uncert_weight = (1.0 - certainty.pow(self.config.CONFIDENCE_EXPONENT)).view(B, 1, 1, 1)
        lambda_eff = self.config.GUIDANCE_SCALE * (s_norm / g_norm) * (0.02 + 0.06 * uncert_weight)
        lambda_eff = torch.clamp(lambda_eff, 0.0, 0.15)

        joint_score = score_prior + lambda_eff * cls_grad
        return joint_score, score_prior, cls_grad, denoised_prior

    def apply_spectral_manifold_contraction(self, joint_score, score_prior, cls_grad):
        r"""
        Spectral Manifold Contraction (SMC):
        Projects out antagonistic components where generative score opposes class likelihood:
        \tilde{v} = v - \beta \cdot \max(0, -\langle \hat{s}, \hat{g} \rangle) \cdot s^\perp
        """
        B = joint_score.shape[0]
        s_flat = score_prior.view(B, -1)
        g_flat = cls_grad.view(B, -1)

        s_norm = torch.norm(s_flat, dim=1, keepdim=True) + 1e-7
        g_norm = torch.norm(g_flat, dim=1, keepdim=True) + 1e-7

        s_unit = s_flat / s_norm
        g_unit = g_flat / g_norm

        # Directional cosine similarity
        cos_sim = torch.sum(s_unit * g_unit, dim=1, keepdim=True) # [B, 1]

        # Antagonistic damping weight
        antagonistic_weight = F.relu(-cos_sim) * self.config.SMC_CONTRACTION_WEIGHT

        # Orthogonal component of score relative to class gradient
        s_orth = s_flat - cos_sim * g_unit
        s_orth = s_orth.view_as(score_prior)

        # Apply contraction
        contracted_score = joint_score - (antagonistic_weight.view(B, 1, 1, 1) * s_orth)
        return contracted_score.detach()

    def amortized_fast_solver(self, x_corrupt, num_steps=None, return_trajectory=False):
        r"""
        Amortized Trajectory Fast Solver (ATFS):
        Executes fast manifold projection in only 3-4 steps using Tweedie's jump
        and curvature-corrected predictor updates with data consistency.
        """
        if num_steps is None:
            num_steps = self.config.SGMP_STEPS

        B = x_corrupt.shape[0]
        device = x_corrupt.device

        # 1. Epistemic corruption estimation
        sigma_est, initial_certainty, target_class, edge_std = self.estimate_corruption_severity(x_corrupt)
        
        # 2. Inject calibrated noise
        if self.config.ADAPTIVE_PURIFY:
            sigma_start = sigma_est.view(B, 1, 1, 1)
        else:
            sigma_start = torch.full((B, 1, 1, 1), self.config.PURIFY_NOISE_LEVEL, device=device)
            
        noise = torch.randn_like(x_corrupt)
        x_t = torch.clamp(x_corrupt + sigma_start * noise, 0.0, 1.0).detach()

        trajectory = [x_t.detach().cpu()] if return_trajectory else None

        # Discretization schedule from estimated sigma down to sigma_min
        sigmas_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        gamma_consistency = self.config.RESIDUAL_CONSISTENCY

        for i in range(num_steps):
            ratio_curr = sigmas_seq[i]
            ratio_next = sigmas_seq[i + 1]
            sigma_curr = sigma_start.squeeze() * ratio_curr + self.config.SIGMA_MIN * (1.0 - ratio_curr)
            
            # CMLB Potential assembly
            joint_score, s_prior, cls_grad, _ = self.compute_cmlb_score(
                x_t, sigma_curr, initial_certainty, target_class
            )

            # Spectral Manifold Contraction
            guided_v = self.apply_spectral_manifold_contraction(joint_score, s_prior, cls_grad)

            # Tweedie Jump: Direct Bayes optimal clean pattern estimate on manifold
            sig_sq = (sigma_curr.view(B, 1, 1, 1) ** 2)
            x_0_pred = torch.clamp(x_t + sig_sq * guided_v, 0.0, 1.0)

            # Next step with data consistency blending
            if i < (num_steps - 1):
                sigma_next = sigma_start.squeeze() * ratio_next + self.config.SIGMA_MIN * (1.0 - ratio_next)
                next_noise = torch.randn_like(x_corrupt)
                # Bridge towards target manifold while retaining input structural anchors
                x_t = (1.0 - gamma_consistency) * x_0_pred + gamma_consistency * x_corrupt
                x_t = torch.clamp(x_t + sigma_next.view(B, 1, 1, 1) * next_noise * 0.3, 0.0, 1.0)
            else:
                x_t = x_0_pred

            if return_trajectory:
                trajectory.append(x_t.detach().cpu())

        # 3. Output pattern on the data manifold
        # For genuinely clean patterns (very low edge residual & high certainty), keep clean input
        is_truly_clean = (initial_certainty > 0.95) & (edge_std < 0.10)
        clean_mask = is_truly_clean.view(B, 1, 1, 1).to(dtype=x_t.dtype)
        x_purified = clean_mask * x_corrupt + (1.0 - clean_mask) * x_t
        x_purified = torch.clamp(x_purified, 0.0, 1.0)

        return x_purified, trajectory

    def forward(self, x_corrupt, num_steps=None):
        purified, _ = self.amortized_fast_solver(x_corrupt, num_steps=num_steps, return_trajectory=False)
        logits = self.classifier(purified)
        return logits, purified
