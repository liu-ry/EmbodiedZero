"""
stable_diffusion/sampler.py  —  Latent Diffusion (Simplified Stable Diffusion) Sampler

Overview of Stable Diffusion architecture
------------------------------------------
The real Stable Diffusion (Rombach et al., 2022) pipeline:

  Training stage
  ├─ Stage 1: Train VAE (invertible compression from pixel space → latent space)
  └─ Stage 2: Freeze VAE, train diffusion model (UNet predicts noise) in **latent space**

  Inference stage
  ├─ 1. Sample noise in latent space z_T ~ N(0, I)
  ├─ 2. Denoise z_T → z_0 with DDIM/DDPM (in latent space)
  └─ 3. VAE decode: z_0 → image  x = decoder(z_0)

This file implements LatentDDIMSampler, covering the full inference stage logic:
  - Reuses DDIMSampler from ddim/sampler.py (DDIM denoising also runs in latent space)
  - Calls VAE.decode() for decoding

Relationship with ddim/sampler.py
-----------------------------------
  DDIMSampler      : denoises within a single space (pixel or latent), unaware of VAE
  LatentDDIMSampler: wraps DDIMSampler + VAE decoder, performs latent denoising + decoding
"""

import torch
from model.noise_schedule import NoiseSchedule
from model.vae            import VAE
from ddim.sampler         import DDIMSampler


class LatentDDIMSampler:
    """
    Runs DDIM denoising in the VAE latent space, then decodes to pixel space.

    Parameters
    ----------
    vae         : a trained (and frozen) VAE instance
    schedule    : an initialized and .to(device) NoiseSchedule instance
    ddim_steps  : number of DDIM inference steps (default 50)
    eta         : DDIM stochasticity coefficient η (0=purely deterministic)
    """

    def __init__(self,
                 vae:        VAE,
                 schedule:   NoiseSchedule,
                 ddim_steps: int   = 50,
                 eta:        float = 0.0):
        self.vae         = vae
        self.ddim        = DDIMSampler(schedule, ddim_steps=ddim_steps, eta=eta)

    # ------------------------------------------------------------------
    # Full generation: latent space noise → DDIM denoising → VAE decode → pixel image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               model,
               n_samples:  int,
               device:     torch.device,
               save_every: int | None = None
               ) -> tuple[torch.Tensor, list]:
        """
        Generates n_samples images.

        Parameters
        ----------
        model      : noise-predicting UNet ε_θ(z_t, t) trained in latent space
        n_samples  : number of images to generate
        device     : compute device
        save_every : save intermediate frames every this many DDIM steps (optional, for visualization)

        Returns
        -------
        images : decoded pixel-space images  (n_samples, C, H, W), range [-1, 1]
        frames : list of intermediate latent frames (only populated when save_every is not None)
        """
        # latent space shape: (B, latent_ch, latent_H, latent_W)
        latent_ch = self.vae.latent_ch
        # MNIST 28×28 → 7×7 latent space (two stride=2 steps)
        latent_h  = 7
        latent_w  = 7
        shape     = (n_samples, latent_ch, latent_h, latent_w)

        # 1. DDIM denoising in latent space
        z0, frames = self.ddim.sample(model, shape, device, save_every=save_every)

        # 2. VAE decode to pixel space
        images = self.vae.decode(z0)

        return images, frames
