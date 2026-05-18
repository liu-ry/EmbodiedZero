"""
model/vae.py  —  Lightweight VAE (Variational Autoencoder)

The key idea of Stable Diffusion is to first use a VAE to compress pixel space into a
low-dimensional **latent space**, then train a diffusion model in that latent space.
During generation, samples are drawn in latent space and decoded back to images via the VAE.

This file is designed for MNIST 28×28 grayscale images, compressing images to (latent_ch, 7, 7).
    Pixel space  : (B, 1,  28, 28)
    Latent space : (B, 4,   7,  7)   ← 4× spatial downsampling, 4 channels

Mathematical background
-----------------------
  The encoder outputs μ and log σ² (both of latent size), then uses the reparameterization trick:
      z = μ + σ · ε,   ε ~ N(0, I)
  KL divergence as regularization loss (pushes latent distribution toward standard normal):
      KL = -½ · Σ (1 + log σ² - μ² - σ²)
  Reconstruction loss (pixel-space MSE):
      Recon = ||x - x̂||²
  Total loss:
      L_VAE = Recon + kl_weight · KL

Note
----
  Real Stable Diffusion (LDM) uses perceptual loss + discriminator instead of pure MSE;
  here we use MSE only to keep dependencies minimal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: adaptive GroupNorm
# ---------------------------------------------------------------------------
def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(8, ch)
    while ch % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, ch)


# ---------------------------------------------------------------------------
# Encoder: pixel space → μ, log_var (latent space)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Two stride=2 convolutions, mapping 28×28 → 7×7, channels 1 → latent_ch*2 (μ and log σ²).
    """

    def __init__(self, in_channels: int = 1, latent_ch: int = 4, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # 28×28 → 14×14
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            _norm(base_ch),
            nn.SiLU(),
            # 14×14 → 7×7
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            _norm(base_ch * 2),
            nn.SiLU(),
            # keep 7×7, output μ and log σ² (latent_ch*2 channels in total)
            nn.Conv2d(base_ch * 2, latent_ch * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, log_var), each of shape (B, latent_ch, H/4, W/4)."""
        out = self.net(x)
        mu, log_var = out.chunk(2, dim=1)
        return mu, log_var


# ---------------------------------------------------------------------------
# Decoder: latent vector z → reconstructed image
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Two transposed convolution upsamples, mapping 7×7 → 28×28, channels latent_ch → in_channels.
    """

    def __init__(self, in_channels: int = 1, latent_ch: int = 4, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # 7×7 → 7×7（通道映射）
            nn.Conv2d(latent_ch, base_ch * 2, 3, padding=1),
            _norm(base_ch * 2),
            nn.SiLU(),
            # 7×7 → 14×14
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            _norm(base_ch),
            nn.SiLU(),
            # 14×14 → 28×28
            nn.ConvTranspose2d(base_ch, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),   # output [-1, 1], consistent with training data normalization
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """
    Lightweight convolutional VAE.

    Parameters
    ----------
    in_channels : number of input image channels (MNIST=1)
    latent_ch   : number of latent space channels (controls compression ratio)
    base_ch     : base channel count for encoder/decoder
    kl_weight   : KL divergence weight (higher → latent space closer to normal, slightly lower reconstruction quality)
    """

    def __init__(self,
                 in_channels: int   = 1,
                 latent_ch:   int   = 4,
                 base_ch:     int   = 32,
                 kl_weight:   float = 1e-3):
        super().__init__()
        self.encoder    = Encoder(in_channels, latent_ch, base_ch)
        self.decoder    = Decoder(in_channels, latent_ch, base_ch)
        self.kl_weight  = kl_weight
        self.latent_ch  = latent_ch

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, log_var)."""
        return self.encoder(x)

    def reparameterize(self,
                       mu:      torch.Tensor,
                       log_var: torch.Tensor
                       ) -> torch.Tensor:
        """Reparameterization: z = μ + σ · ε. During training ε ~ N(0,I); at inference can use μ directly."""
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → reconstructed image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → reparameterize → decode.

        Returns
        -------
        recon : reconstructed image  (B, C, H, W)
        loss  : scalar, total VAE loss = Recon + kl_weight * KL
        """
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z)

        # Reconstruction loss (MSE, averaged over pixel dimensions)
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recon_loss + self.kl_weight * kl_loss
        return recon, loss

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference stage: uses μ directly (no noise), returns latent variable.
        Used in Stable Diffusion's "encode training images to latent space" step.
        """
        mu, _ = self.encode(x)
        return mu
