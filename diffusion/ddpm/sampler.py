"""
ddpm/sampler.py  —  DDPM Ancestral Sampler

DDPM's specific reverse sampling method:
  x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t)) + σ_t · z
  where z ~ N(0,I) (when t>0); no noise is added at t=0

Difference from DDIM
---------------------
  DDPM : adds random noise σ_t·z at each step  → stochastic sampling, requires all T steps
  DDIM : deterministic update at each step       → can skip steps, 100 steps gives quality close to 1000

This file contains only DDPM-specific sampling logic; forward noising and training loss are in model/.
"""

import torch
from model.noise_schedule import NoiseSchedule


class DDPMSampler:
    """
    DDPM ancestral sampler based on NoiseSchedule.

    Parameters
    ----------
    schedule : an initialized and .to(device) NoiseSchedule instance
    """

    def __init__(self, schedule: NoiseSchedule):
        self.sch = schedule

    # ------------------------------------------------------------------
    # Single-step reverse denoising: x_t → x_{t-1}
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t_val: int) -> torch.Tensor:
        """Single DDPM denoising step (with random noise)."""
        sch  = self.sch
        B    = x_t.shape[0]
        t    = torch.full((B,), t_val, device=x_t.device, dtype=torch.long)

        eps_pred = model(x_t, t)

        coef  = sch._g(sch.coef_eps,         t, x_t.ndim)
        recip = sch._g(sch.sqrt_recip_alphas, t, x_t.ndim)
        mean  = recip * (x_t - coef * eps_pred)

        if t_val > 0:
            sigma = sch._g(sch.sqrt_betas, t, x_t.ndim)
            return mean + sigma * torch.randn_like(x_t)
        return mean

    # ------------------------------------------------------------------
    # Full reverse sampling: x_T ~ N(0,I) → x_0
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               model,
               shape:      tuple,
               device:     torch.device,
               save_every: int | None = None
               ) -> tuple[torch.Tensor, list]:
        """
        Iteratively denoises from pure Gaussian noise to generate samples.

        Parameters
        ----------
        model      : noise prediction network ε_θ
        shape      : output shape, e.g. (16, 1, 28, 28)
        device     : compute device
        save_every : save a frame every this many steps (for visualizing denoising trajectory)

        Returns
        -------
        x      : final generated sample  shape
        frames : list of intermediate frames (only populated when save_every is not None, ordered T→0)
        """
        x      = torch.randn(shape, device=device)
        frames = []

        for t_val in reversed(range(self.sch.T)):
            x = self.p_sample(model, x, t_val)
            if save_every is not None and (
                    t_val % save_every == 0 or t_val == self.sch.T - 1):
                frames.append(x.clone().cpu())

        return x, frames
