r"""
Continuous-time Score-Based Generative Model (ScoreNet) with Karras EDM Preconditioning.
Supports:
1. High-fidelity U-Net architecture with residual blocks and sinusoidal time embeddings.
2. Karras EDM (Elucidating the Design Space of Diffusion Models) preconditioning.
3. Direct analytical score computation: s_\theta(x, \sigma) = (D_\theta(x, \sigma) - x) / \sigma^2.
4. One-step Tweedie clean manifold projection: \hat{x}_0 = D_\theta(x, \sigma).
5. Exponential Moving Average (EMA) for parameter stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        # t is 1D tensor of timesteps/sigmas
        device = t.device
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.gn1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)

class ScoreNet(nn.Module):
    r"""
    U-Net style Residual Score Network predicting raw score or denoising residual.
    """
    def __init__(self, in_channels=1, base_channels=32, time_dim=64, sigma_data=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.sigma_data = sigma_data
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.block2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        
        # Bottleneck
        self.mid_block = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_proj = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.block3 = ResBlock(base_channels, base_channels, time_dim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward_raw(self, x, t_emb_input):
        t_emb = self.time_mlp(t_emb_input)
        h0 = self.init_conv(x)
        h1 = self.block1(h0, t_emb)
        h2 = self.down1(h1)
        h3 = self.block2(h2, t_emb)
        
        mid = self.mid_block(h3, t_emb)
        
        up = self.up_proj(self.up1(mid))
        if up.shape[-2:] != h1.shape[-2:]:
            up = F.interpolate(up, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        h4 = self.block3(up + h1, t_emb)
        out = self.out_conv(h4)
        return out

    def forward(self, x, sigma):
        r"""
        Karras EDM preconditioned forward pass.
        Given input x and noise level sigma, returns the optimal denoised pattern \hat{x}_0 \in [0, 1].
        """
        B = x.shape[0]
        if isinstance(sigma, (float, int)):
            sigma = torch.full((B,), float(sigma), device=x.device, dtype=x.dtype)
        elif sigma.dim() == 0:
            sigma = sigma.expand(B)
            
        sig = sigma.view(B, 1, 1, 1).to(dtype=x.dtype)
        
        c_skip = (self.sigma_data ** 2) / (sig ** 2 + self.sigma_data ** 2)
        c_out = (sig * self.sigma_data) / torch.sqrt(sig ** 2 + self.sigma_data ** 2)
        c_in = 1.0 / torch.sqrt(sig ** 2 + self.sigma_data ** 2)
        c_noise = torch.log(torch.clamp(sigma, min=1e-5)) * 0.25
        
        f_x = self.forward_raw(c_in * x, c_noise)
        denoised = c_skip * x + c_out * f_x
        return denoised

    def get_score(self, x, sigma):
        r"""
        Computes the continuous-time score vector:
        s_\theta(x, \sigma) = \nabla_x \log p_\sigma(x) \approx (D_\theta(x, \sigma) - x) / \sigma^2
        """
        denoised = self.forward(x, sigma)
        B = x.shape[0]
        if isinstance(sigma, (float, int)):
            sig = torch.full((B, 1, 1, 1), float(sigma), device=x.device, dtype=x.dtype)
        else:
            sig = sigma.view(B, 1, 1, 1).to(dtype=x.dtype)
        score = (denoised - x) / (sig ** 2 + 1e-7)
        return score, denoised


class ScoreNetEMA:
    """
    Exponential Moving Average (EMA) container for ScoreNet parameters.
    Critical for stabilizing continuous-time score matching dynamics.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = {name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
