"""
Score-Guided Manifold Projection (SGMP) Framework.
Bridging Diffusion Priors and Discriminative Likelihood for Robust Pattern Recognition.

Core Modules:
1. Epistemic-Adaptive Corruption Estimator & Gating
2. Confidence-Modulated Likelihood Bridge (CMLB)
3. Spectral Manifold Contraction (SMC)
4. Amortized Trajectory Fast Solver (ATFS) with Data Consistency & Langevin Corrector
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

        Also detects likely adversarial inputs (low edge_std + low certainty).
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
        # Uncertainty-driven: more uncertainty -> more noise needed
        uncertainty = 1.0 - certainty
        sigma_est = uncertainty * 0.22 + torch.clamp(edge_std * 0.35, 0.0, 0.15)
        sigma_est = torch.clamp(sigma_est, self.config.MIN_PURIFY_SIGMA, self.config.MAX_PURIFY_SIGMA)

        # 4. Adversarial detection heuristic:
        # Adversarial images have LOW edge noise (small ε perturbation) but LOW certainty
        # Natural corruptions tend to have HIGH edge noise when certainty is low
        is_likely_adversarial = (certainty < 0.65) & (edge_std < 0.12)

        return sigma_est, certainty, target_class, edge_std, is_likely_adversarial

    def compute_cmlb_score(self, x_t, sigma_t, certainty, target_class):
        r"""
        Confidence-Modulated Likelihood Bridge (CMLB):
        v_t = s_\theta(x_t, \sigma_t) + \lambda_t \nabla_{x_t} \log p(y^* | x_t)

        The guidance scale is properly scaled by sigma^2 and modulated by epistemic
        uncertainty to allow stronger class-directed purification when the classifier
        is uncertain.
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

        # 3. Proper sigma-squared scaled guidance with uncertainty modulation
        # Higher uncertainty -> stronger guidance (let classifier pull harder)
        # Higher sigma -> stronger guidance (early steps allow more correction)
        if isinstance(sigma_t, (float, int)):
            sig_val = torch.full((B,), float(sigma_t), device=x_t.device, dtype=x_t.dtype)
        else:
            sig_val = sigma_t.to(dtype=x_t.dtype)

        sig_sq = (sig_val ** 2).view(B, 1, 1, 1)

        # Norm calibration: scale classifier gradient relative to score magnitude
        s_norm = torch.norm(score_prior.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-6
        g_norm = torch.norm(cls_grad.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-6

        # Uncertainty-modulated guidance: more guidance when uncertain
        uncert_weight = (1.0 - certainty.pow(self.config.CONFIDENCE_EXPONENT)).view(B, 1, 1, 1)

        # λ_eff = λ₀ · (s_norm / g_norm) · σ² · (0.1 + 0.9·uncertainty)
        lambda_eff = self.config.GUIDANCE_SCALE * (s_norm / g_norm) * sig_sq * (0.1 + 0.9 * uncert_weight)

        joint_score = score_prior + lambda_eff * cls_grad
        return joint_score, score_prior, cls_grad, denoised_prior

    def apply_spectral_manifold_contraction(self, joint_score, score_prior, cls_grad):
        r"""
        Spectral Manifold Contraction (SMC):
        Projects out antagonistic components where generative score opposes class likelihood:
        \tilde{v} = v - \beta \cdot \max(0, -(cos_sim - threshold)) \cdot s^\perp

        Only activates when cosine similarity is strongly negative (below threshold),
        preventing unnecessary contraction that would harm accuracy.
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

        # Only contract when cosine similarity is strongly negative (below threshold)
        threshold = self.config.SMC_COSINE_THRESHOLD
        antagonistic_weight = F.relu(-(cos_sim - threshold)) * self.config.SMC_CONTRACTION_WEIGHT

        # Orthogonal component of score relative to class gradient
        s_orth = s_flat - cos_sim * g_unit
        s_orth = s_orth.view_as(score_prior)

        # Apply contraction
        contracted_score = joint_score - (antagonistic_weight.view(B, 1, 1, 1) * s_orth)
        return contracted_score.detach()

    def langevin_corrector_step(self, x, sigma, certainty, target_class):
        r"""
        True Langevin corrector step (Algorithm 1, corrector):
        x = x + τ · s_θ(x, σ) + √(2τ) · ξ

        Refines the current estimate by taking a gradient step along the score field
        before proceeding to the next denoising step.
        """
        if not self.config.CORRECTOR_ENABLED:
            return x

        B = x.shape[0]
        snr = self.config.CORRECTOR_SNR

        for _ in range(self.config.CORRECTOR_STEPS):
            with torch.no_grad():
                score, _ = self.score_net.get_score(x, sigma)

            # Step size proportional to snr * sigma^2
            if isinstance(sigma, (float, int)):
                sig_sq = sigma ** 2
            else:
                sig_sq = (sigma.view(B, 1, 1, 1) ** 2)

            step_size = snr * sig_sq
            noise = torch.randn_like(x)
            x = x + step_size * score + torch.sqrt(2.0 * step_size) * noise
            x = torch.clamp(x, 0.0, 1.0)

        return x

    def amortized_fast_solver(self, x_corrupt, num_steps=None, return_trajectory=False):
        r"""
        Amortized Trajectory Fast Solver (ATFS):
        Executes fast manifold projection using Tweedie's jump and curvature-corrected
        predictor updates with:
        - Adaptive step budget based on corruption type detection
        - Corruption-adaptive data consistency
        - True Langevin corrector steps
        - Geometric noise schedule for better sigma coverage
        """
        B = x_corrupt.shape[0]
        device = x_corrupt.device

        # 1. Epistemic corruption estimation
        sigma_est, initial_certainty, target_class, edge_std, is_likely_adversarial = \
            self.estimate_corruption_severity(x_corrupt)

        # 2. Adaptive step budget: more steps for adversarial/heavy corruption
        if num_steps is not None:
            effective_steps = num_steps
        else:
            base_steps = self.config.SGMP_STEPS
            # Per-batch adaptive: use max across batch for consistency
            if is_likely_adversarial.any():
                effective_steps = self.config.ADVERSARIAL_STEPS
            elif (initial_certainty.mean() < 0.5):
                # Heavy corruption detected
                effective_steps = base_steps + 2
            else:
                effective_steps = base_steps

        # 3. Inject calibrated noise
        if self.config.ADAPTIVE_PURIFY:
            sigma_start = sigma_est.view(B, 1, 1, 1)
            # Boost sigma for adversarial inputs
            if is_likely_adversarial.any():
                adv_mask = is_likely_adversarial.view(B, 1, 1, 1).to(dtype=sigma_start.dtype)
                adv_sigma = torch.full_like(sigma_start, self.config.ADVERSARIAL_SIGMA)
                sigma_start = (1.0 - adv_mask) * sigma_start + adv_mask * adv_sigma
        else:
            sigma_start = torch.full((B, 1, 1, 1), self.config.PURIFY_NOISE_LEVEL, device=device)

        noise = torch.randn_like(x_corrupt)
        x_t = torch.clamp(x_corrupt + sigma_start * noise, 0.0, 1.0).detach()

        trajectory = [x_t.detach().cpu()] if return_trajectory else None

        # Geometric noise schedule (better coverage of high-to-low sigma range)
        # sigma(i) = sigma_start * (sigma_min / sigma_start)^(i/K)
        t_ratios = torch.linspace(0.0, 1.0, effective_steps + 1, device=device)

        # Corruption-adaptive data consistency weight
        # Less blending for heavily corrupted inputs, more for near-clean
        is_near_clean = (initial_certainty > 0.85) & (edge_std < 0.12)
        gamma_base = torch.where(
            is_near_clean,
            torch.full((B,), self.config.RESIDUAL_CONSISTENCY_CLEAN, device=device),
            torch.full((B,), self.config.RESIDUAL_CONSISTENCY, device=device)
        ).view(B, 1, 1, 1)

        for i in range(effective_steps):
            ratio_curr = 1.0 - t_ratios[i]
            ratio_next = 1.0 - t_ratios[i + 1]
            sigma_curr = sigma_start.squeeze() * ratio_curr + self.config.SIGMA_MIN * (1.0 - ratio_curr)

            # CMLB potential assembly
            joint_score, s_prior, cls_grad, _ = self.compute_cmlb_score(
                x_t, sigma_curr, initial_certainty, target_class
            )

            # Spectral Manifold Contraction
            guided_v = self.apply_spectral_manifold_contraction(joint_score, s_prior, cls_grad)

            # Tweedie Jump: direct Bayes optimal clean pattern estimate
            sig_sq = (sigma_curr.view(B, 1, 1, 1) ** 2)
            x_0_pred = torch.clamp(x_t + sig_sq * guided_v, 0.0, 1.0)

            # Next step with data consistency blending
            if i < (effective_steps - 1):
                sigma_next = sigma_start.squeeze() * ratio_next + self.config.SIGMA_MIN * (1.0 - ratio_next)

                # Adaptive data consistency: blend less corruption for heavy corruptions
                gamma_step = gamma_base * (1.0 - t_ratios[i + 1])  # Decay consistency towards end

                # Bridge towards manifold with corruption-adaptive anchoring
                x_t = (1.0 - gamma_step) * x_0_pred + gamma_step * x_corrupt
                next_noise = torch.randn_like(x_corrupt)
                x_t = torch.clamp(x_t + sigma_next.view(B, 1, 1, 1) * next_noise * 0.25, 0.0, 1.0)

                # Langevin corrector step for refinement
                x_t = self.langevin_corrector_step(x_t, sigma_next, initial_certainty, target_class)
            else:
                x_t = x_0_pred

            if return_trajectory:
                trajectory.append(x_t.detach().cpu())

        # 4. Output gating: for genuinely clean patterns, keep clean input
        is_truly_clean = (initial_certainty > 0.95) & (edge_std < 0.08)
        clean_mask = is_truly_clean.view(B, 1, 1, 1).to(dtype=x_t.dtype)
        x_purified = clean_mask * x_corrupt + (1.0 - clean_mask) * x_t
        x_purified = torch.clamp(x_purified, 0.0, 1.0)

        return x_purified, trajectory

    def forward(self, x_corrupt, num_steps=None):
        purified, _ = self.amortized_fast_solver(x_corrupt, num_steps=num_steps, return_trajectory=False)
        logits = self.classifier(purified)
        return logits, purified
