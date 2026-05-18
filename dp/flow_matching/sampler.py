"""
flow_matching/sampler.py  —  Flow Matching ODE Sampler

Corresponding paper: "Flow Matching for Generative Modeling" (Lipman et al., 2022)

Inference algorithm (Euler ODE integration)
--------------------------------------------
Given a trained vector field network v_θ(x_t, t), starting from x_0 ~ N(0,I),
numerically integrate the ODE using Euler's method:

    dx/dt = v_θ(x_t, t),   t: 0 → 1

Discretization (step size h = 1/N):
    x_{t+h} = x_t + h · v_θ(x_t, t)

Higher N gives better accuracy; N=100 is usually sufficient.
Can also be replaced with higher-order methods (Heun, RK4, etc.).

Comparison with DDPM/DDIM
--------------------------
  DDPM   : denoises from x_T (noise) to x_0 (image) step by step, t from large to small
  DDIM   : same, but can skip steps (non-Markovian)
  FM ODE : t from 0 (noise) to 1 (image), forward integration, no "denoising" concept, simpler
"""

import torch
import numpy as np
from model.flow_schedule import FlowSchedule


class FlowMatchingSampler:
    """
    ODE sampler based on FlowSchedule (Euler method).

    Parameters
    ----------
    schedule   : FlowSchedule instance (mainly used to get T for timestep mapping)
    ode_steps  : number of Euler integration steps (default 100)
    """

    def __init__(self,
                 schedule:  FlowSchedule,
                 ode_steps: int = 100):
        self.sch       = schedule
        self.ode_steps = ode_steps

    # ------------------------------------------------------------------
    # Single Euler step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _euler_step(self,
                    model,
                    x_t:   torch.Tensor,
                    t_val: float,
                    dt:    float
                    ) -> torch.Tensor:
        """
        Euler single step: x_{t+dt} = x_t + dt · v_θ(x_t, t).

        Parameters
        ----------
        model  : vector field network v_θ(x, t_int)
        x_t    : current state  (B, ...)
        t_val  : current continuous time  ∈ [0, 1]
        dt     : time step size (positive, since going 0 → 1)
        """
        B      = x_t.shape[0]
        T      = self.sch.T

        # map continuous time t_val ∈ [0,1] to integer timestep ∈ [0, T-1]
        t_int  = torch.full((B,), int(t_val * (T - 1)), device=x_t.device, dtype=torch.long)
        t_int  = t_int.clamp(0, T - 1)

        # predict vector field
        v_pred = model(x_t, t_int)

        return x_t + dt * v_pred

    # ------------------------------------------------------------------
    # Heun second-order correction step (optional, higher accuracy)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _heun_step(self,
                   model,
                   x_t:   torch.Tensor,
                   t_val: float,
                   dt:    float
                   ) -> torch.Tensor:
        """
        Heun method (trapezoidal rule): Euler predictor followed by average vector field correction.
        One order more accurate than Euler; can halve inference steps.
        """
        # predictor (Euler)
        x_pred = self._euler_step(model, x_t, t_val, dt)

        # corrector: average with endpoint vector field
        t_next = min(t_val + dt, 1.0)
        B      = x_t.shape[0]
        T      = self.sch.T
        t_int_next = torch.full(
            (B,), int(t_next * (T - 1)), device=x_t.device, dtype=torch.long
        ).clamp(0, T - 1)
        v_next = model(x_pred, t_int_next)

        t_int_cur  = torch.full(
            (B,), int(t_val * (T - 1)), device=x_t.device, dtype=torch.long
        ).clamp(0, T - 1)
        v_cur = model(x_t, t_int_cur)

        return x_t + dt * 0.5 * (v_cur + v_next)

    # ------------------------------------------------------------------
    # Full sampling: x_0 ~ N(0,I) → x_1 (image)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               model,
               shape:      tuple,
               device:     torch.device,
               method:     str       = 'euler',
               save_every: int | None = None
               ) -> tuple[torch.Tensor, list]:
        """
        Integrates ODE from standard normal noise to data distribution.

        Parameters
        ----------
        model      : vector field network v_θ, same interface as SimpleUNet
        shape      : output shape, e.g. (16, 1, 28, 28) or latent space shape
        device     : compute device
        method     : integration method, 'euler' or 'heun'
        save_every : save intermediate frames every this many steps (optional)

        Returns
        -------
        x      : final generated sample (data at t=1)
        frames : list of intermediate frames (ordered t=0→1)
        """
        N      = self.ode_steps
        dt     = 1.0 / N
        t_vals = np.linspace(0.0, 1.0 - dt, N)  # [0, dt, 2dt, ..., 1-dt]

        x      = torch.randn(shape, device=device)
        frames = []

        step_fn = self._heun_step if method == 'heun' else self._euler_step

        for step_idx, t_val in enumerate(t_vals):
            x = step_fn(model, x, float(t_val), dt)

            if save_every is not None and (
                    step_idx % save_every == 0 or step_idx == N - 1):
                frames.append(x.clone().cpu())

        return x, frames
