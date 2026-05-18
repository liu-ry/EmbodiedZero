"""
model/model.py  —  UNet noise prediction backbone

All diffusion model variants (DDPM / DDIM / Stable Diffusion / etc.) use the same
"noise prediction network" to estimate ε_θ. This file provides a reusable SimpleUNet.

Network structure
-----------------
  Encoder (downsampling) → Bottleneck → Decoder (upsampling)
  Sinusoidal timestep embedding injected at every layer; skip connections between encoder and decoder.

Input / Output
--------------
  x_t : (B, C, H, W)   noisy image
  t   : (B,)            integer timestep
  → predicted noise ε_θ : (B, C, H, W)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: adaptive GroupNorm (reduces group count when channel count is small)
# ---------------------------------------------------------------------------
def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(8, ch)
    while ch % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, ch)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# Maps integer t to a continuous vector so the network knows which diffusion stage it is at.
# Identical to Transformer positional encoding.
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t.float()[:, None] * emb[None, :]           # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)      # (B, dim)


# ---------------------------------------------------------------------------
# Residual conv block with timestep embedding
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Conv → Norm → SiLU → Conv, with timestep bias injection and residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)

        self.block1 = nn.Sequential(
            _norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            _norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]  # broadcast time bias across spatial dims
        h = self.block2(h)
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# SimpleUNet (designed for MNIST 28×28, extensible to larger resolutions)
# ---------------------------------------------------------------------------
class SimpleUNet(nn.Module):
    """
    Lightweight UNet for 28×28 grayscale images.

    Channels: 1 → 64 → 128 → 256 (bottleneck) → 128 → 64 → 1
    Spatial:  28 → 14 → 7 (bottleneck) → 14 → 28

    Extension tips
    --------------
    - Increase base_channels for higher generation quality
    - Add Cross-Attention at the bottleneck for text/class conditioning (Stable Diffusion)
    - Add more down/up-sampling layers to support higher resolutions
    """

    def __init__(self,
                 in_channels:   int = 1,
                 base_channels: int = 64,
                 time_emb_dim:  int = 128):
        super().__init__()
        t  = time_emb_dim
        c1 = base_channels        # 64
        c2 = base_channels * 2    # 128
        c3 = base_channels * 4    # 256 (bottleneck)

        # Timestep embedding MLP
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(t),
            nn.Linear(t, t * 2),
            nn.SiLU(),
            nn.Linear(t * 2, t),
        )

        # Encoder
        self.enc1  = ResBlock(in_channels, c1, t)               # 28×28
        self.down1 = nn.Conv2d(c1, c1, 4, stride=2, padding=1)  # → 14×14
        self.enc2  = ResBlock(c1, c2, t)                         # 14×14
        self.down2 = nn.Conv2d(c2, c2, 4, stride=2, padding=1)  # → 7×7

        # Bottleneck
        self.mid1 = ResBlock(c2, c3, t)
        self.mid2 = ResBlock(c3, c2, t)

        # Decoder (skip connections: concat upsampled output with encoder features)
        self.up2  = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)  # → 14×14
        self.dec2 = ResBlock(c2 + c2, c1, t)
        self.up1  = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)  # → 28×28
        self.dec1 = ResBlock(c1 + c1, c1, t)

        # Output projection
        self.out = nn.Sequential(
            _norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        t_emb = self.time_emb(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        m  = self.mid2(self.mid1(self.down2(e2), t_emb), t_emb)

        # bilinear interpolation to align skip-connection spatial dims (supports arbitrary input resolution)
        up2_out = F.interpolate(self.up2(m),  size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2_out, e2], dim=1), t_emb)
        up1_out = F.interpolate(self.up1(d2), size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1_out, e1], dim=1), t_emb)

        return self.out(d1)
