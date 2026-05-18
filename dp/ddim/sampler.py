"""
ddim/sampler.py  —  DDIM Deterministic Sampler

Paper: "Denoising Diffusion Implicit Models" (Song et al., 2020)
       https://arxiv.org/abs/2010.02502

Difference between DDIM and DDPM
----------------------------------
  DDPM : adds random noise σ_t · z at each step (Markov chain), must run all T steps
  DDIM : rewrites the reverse process as non-Markovian, supports **step skipping** (sub-sampling):
           - only S (≪T) steps needed for high-quality samples (e.g. T=1000, S=50)
           - fully deterministic when η=0; degenerates to DDPM stochastic sampling when η=1

Mathematical derivation (single-step update)
--------------------------------------------
Given x_t and network prediction ε_θ(x_t, t):

  1. Predict x_0:
       x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t

  2. Compute variance:
       σ_t = η · √((1-ᾱ_{t-1}) / (1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})

  3. Update x_{t-1}:
       x_{t-1} = √ᾱ_{t-1} · x̂_0
               + √(1 - ᾱ_{t-1} - σ_t²) · ε_θ
               + σ_t · z,    z ~ N(0, I)

  When η=0, σ_t=0 and the update is fully deterministic (original DDIM setting).

Implementation notes
--------------------
  - Uses the same NoiseSchedule as DDPMSampler; no need to retrain the model
  - `ddim_steps` controls number of sampling steps; `eta` controls stochasticity (0=deterministic, 1≈DDPM)
"""

import torch
import numpy as np
from model.noise_schedule import NoiseSchedule


class DDIMSampler:
    """
    DDIM sampler based on NoiseSchedule.

    Parameters
    ----------
    schedule    : an initialized and .to(device) NoiseSchedule instance
    ddim_steps  : number of sampling steps S (≤ schedule.T), default 50
    eta         : DDIM stochasticity coefficient η; 0=purely deterministic, 1=approximately DDPM
    """

    def __init__(self,
                 schedule:   NoiseSchedule,
                 ddim_steps: int   = 50,
                 eta:        float = 0.0):
        self.sch        = schedule
        self.ddim_steps = ddim_steps
        self.eta        = eta

        # 从 T 个时间步均匀选取 S 个（含端点）
        T      = schedule.T
        # 例如 T=1000, S=50 → [980, 960, ..., 20, 0] (倒序，采样时从大到小)
        step_ratio = T // ddim_steps
        # 时间步序列（升序），对应 0,1,...,S-1 → 对应 [step_ratio-1, 2*step_ratio-1, ...]
        timesteps = (np.arange(0, ddim_steps) * step_ratio).round().astype(int)
        timesteps = np.clip(timesteps, 0, T - 1)
        self.timesteps = list(reversed(timesteps.tolist()))   # 采样时从大到小

    # ------------------------------------------------------------------
    # Single-step reverse denoising: x_t → x_{t-prev}
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self,
                 model,
                 x_t:    torch.Tensor,
                 t_val:  int,
                 t_prev: int,
                 ) -> torch.Tensor:
        """
        DDIM single denoising step.

        Parameters
        ----------
        model  : noise prediction network ε_θ
        x_t    : current noisy sample  (B, C, H, W)
        t_val  : current timestep (integer)
        t_prev : previous (smaller) timestep (integer); -1 means already at t=0
        """
        sch = self.sch
        B   = x_t.shape[0]
        t   = torch.full((B,), t_val, device=x_t.device, dtype=torch.long)

        # ε_θ(x_t, t)
        eps_pred = model(x_t, t)

        # ᾱ_t  和  ᾱ_{t-1}
        ab_t    = sch._g(sch.alphas_cumprod, t, x_t.ndim)          # √ᾱ_t²
        sqrt_ab_t    = ab_t.sqrt()
        sqrt_1mab_t  = (1.0 - ab_t).sqrt()

        if t_prev >= 0:
            t_p      = torch.full((B,), t_prev, device=x_t.device, dtype=torch.long)
            ab_prev  = sch._g(sch.alphas_cumprod, t_p, x_t.ndim)  # ᾱ_{t-1}
        else:
            # t_prev = -1 (already at t=0), ᾱ_{-1} defined as 1
            ab_prev = torch.ones_like(ab_t)

        sqrt_ab_prev   = ab_prev.sqrt()
        sqrt_1mab_prev = (1.0 - ab_prev).sqrt()

        # 1. 预测 x_0
        x0_pred = (x_t - sqrt_1mab_t * eps_pred) / sqrt_ab_t

        # 2. DDIM variance σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
        #    when η=0, σ_t=0 (deterministic DDIM)
        if self.eta > 0 and t_prev >= 0:
            sigma = (self.eta
                     * ((1 - ab_prev) / (1 - ab_t)).sqrt()
                     * (1 - ab_t / ab_prev).sqrt())
        else:
            sigma = torch.zeros_like(ab_t)

        # 3. "pointing toward x_t" directional component (noise direction after removing variance)
        dir_xt_coef = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt()

        # 4. x_{t-1}
        x_prev = sqrt_ab_prev * x0_pred + dir_xt_coef * eps_pred
        if self.eta > 0 and t_prev >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

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
        Denoises with DDIM step-skipping from pure Gaussian noise to generate samples.

        Parameters
        ----------
        model      : noise prediction network ε_θ
        shape      : output shape, e.g. (16, 1, 28, 28)
        device     : compute device
        save_every : save a frame every this many steps (in DDIM step sequence) for visualization

        Returns
        -------
        x      : final generated sample
        frames : list of intermediate frames (ordered T→0)
        """
        x      = torch.randn(shape, device=device)
        frames = []

        timesteps = self.timesteps                # 从大到小
        for step_idx, t_val in enumerate(timesteps):
            t_prev = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else -1
            x = self.p_sample(model, x, t_val, t_prev)

            if save_every is not None and (
                    step_idx % save_every == 0 or step_idx == len(timesteps) - 1):
                frames.append(x.clone().cpu())

        return x, frames
