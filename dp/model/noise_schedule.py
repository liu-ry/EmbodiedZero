"""
model/noise_schedule.py  —  Noise schedule + forward noising + training loss

Core component shared by DDPM / DDIM / Stable Diffusion and other diffusion variants:

  1. β-schedule computation (linear or cosine)
  2. Forward noising q_sample  — adds t steps of noise to clean data
  3. Training loss p_losses    — MSE(predicted noise, true noise)

The only difference between variants is the "reverse sampling method";
the forward process and training loss are identical for DDPM / DDIM.

Mathematical notation
---------------------
  β_t            : per-step noise magnitude
  α_t = 1 - β_t
  ᾱ_t = ∏ α_s    : cumulative product (alphas_cumprod)

Forward process
  q(x_t | x_0) = N(x_t; √ᾱ_t · x_0,  (1 - ᾱ_t) · I)
  i.e. x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0,I)

Training objective
  L = E[||ε - ε_θ(x_t, t)||²]
"""

import torch
import torch.nn.functional as F


class NoiseSchedule:
    """
    Pre-computes all coefficients needed for the diffusion process.

    Parameters
    ----------
    num_timesteps : total diffusion steps T
    schedule      : 'linear' (original DDPM) or 'cosine' (improved, better quality)
    beta_start    : starting β for linear schedule
    beta_end      : ending β for linear schedule
    """

    def __init__(self,
                 num_timesteps: int   = 1000,
                 schedule:      str   = 'linear',
                 beta_start:    float = 1e-4,
                 beta_end:      float = 0.02):
        self.T = num_timesteps

        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            # Nichol & Dhariwal 2021: cosine schedule has less noise at low timesteps
            steps = num_timesteps + 1
            t     = torch.linspace(0, num_timesteps, steps) / num_timesteps
            ab    = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            ab    = ab / ab[0]
            betas = (1 - ab[1:] / ab[:-1]).clamp(0, 0.9999)
        else:
            raise ValueError(f'Unknown schedule: {schedule}')

        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)   # ᾱ_t

        # —— Forward process coefficients ——
        self._reg('betas',            betas)
        self._reg('alphas',           alphas)
        self._reg('alphas_cumprod',   alphas_cumprod)
        self._reg('sqrt_ab',          alphas_cumprod.sqrt())           # √ᾱ_t
        self._reg('sqrt_one_minus_ab', (1.0 - alphas_cumprod).sqrt()) # √(1-ᾱ_t)

        # —— DDPM reverse sampling coefficients (not needed by DDIM, kept for compatibility) ——
        self._reg('sqrt_recip_alphas',  (1.0 / alphas).sqrt())
        self._reg('coef_eps',           betas / (1.0 - alphas_cumprod).sqrt())
        self._reg('sqrt_betas',         betas.sqrt())   # 后验方差 σ_t = √β_t

    def _reg(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    def to(self, device):
        for k in vars(self):
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def _g(self, coef: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Index coefficients by timestep and broadcast to arbitrary dimensions (images or vectors)."""
        return coef.gather(0, t).view(t.shape[0], *([1] * (ndim - 1)))

    # ------------------------------------------------------------------
    # Forward noising: q(x_t | x_0)
    # Adds t steps of noise to clean data; shared by DDPM / DDIM training
    # ------------------------------------------------------------------
    def q_sample(self,
                 x0:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x_t, noise).

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab  = self._g(self.sqrt_ab,            t, x0.ndim)
        s_1ab = self._g(self.sqrt_one_minus_ab,  t, x0.ndim)
        return s_ab * x0 + s_1ab * noise, noise

    # ------------------------------------------------------------------
    # Training loss: shared by DDPM / DDIM
    # ------------------------------------------------------------------
    def p_losses(self, model, x0: torch.Tensor) -> torch.Tensor:
        """
        Randomly samples t, adds noise, predicts noise with the network, returns MSE loss.

        Parameters
        ----------
        model : noise prediction network ε_θ, takes (x_t, t) and returns predicted noise
        x0    : clean data (B, ...)
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        return F.mse_loss(model(x_t, t), noise)
