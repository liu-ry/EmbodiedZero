"""
model/flow_schedule.py  —  Flow Matching training schedule

Papers: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
        "Improving and Generalizing Flow Matching" (Albergo & Vanden-Eijnden, 2023)

Core idea of Flow Matching
--------------------------
Unlike DDPM/DDIM, Flow Matching does not use a Markov noising process. Instead:
  1. Define a deterministic flow from noise distribution p_0 = N(0,I) to data distribution p_1
  2. Learn this flow by fitting the vector field v_θ(x_t, t)
  3. At inference, start from x_0 ~ N(0,I) and integrate the ODE (Euler etc.) to t=1

Optimal Transport Conditional Flow Matching (OT-CFM)
-----------------------------------------------------
Given noise x_0 ~ N(0,I) and data x_1:

  Conditional flow (linear interpolation):
      x_t = (1 - t) · x_0 + t · x_1,   t ∈ [0, 1]

  Conditional vector field (straight-line direction):
      u_t(x_t | x_1) = x_1 - x_0

  Training objective (MSE fit to vector field):
      L_FM = E_{t,x_0,x_1} [||v_θ(x_t, t) - (x_1 - x_0)||²]

Comparison with DDPM
--------------------
  DDPM    : discrete timesteps (T=1000), learns to predict noise ε, needs Markov chain reverse sampling
  Flow FM : continuous time t ∈ [0,1], learns vector field v, inference uses ODE (arbitrary steps)

Implementation notes
--------------------
  - Timestep t is sampled from Uniform(0, 1) (continuous)
  - The network input timestep must be mapped to an integer or normalized float;
    this implementation scales t to integer [0, T-1] and reuses SimpleUNet's integer timestep embedding (default T=1000)
  - FlowSchedule only handles data preparation and training loss; inference logic is in flow_matching/sampler.py
"""

import torch
import torch.nn.functional as F


class FlowSchedule:
    """
    Training utility class for Flow Matching.

    Parameters
    ----------
    num_timesteps : integer discretization steps (used only for timestep embedding; does not affect the continuous flow math)
                   At inference, any number of Euler integration steps can be chosen
    sigma_min     : minimum noise for the conditional flow (0 = strict straight-line OT; small value for numerical stability)
    """

    def __init__(self,
                 num_timesteps: int   = 1000,
                 sigma_min:     float = 1e-4):
        self.T         = num_timesteps
        self.sigma_min = sigma_min

    def to(self, device):
        # FlowSchedule has no Tensors to migrate (pure computation); kept for API consistency
        self._device = device
        return self

    # ------------------------------------------------------------------
    # Sample intermediate state x_t (called during training)
    # ------------------------------------------------------------------
    def q_sample(self,
                 x1:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        OT-CFM conditional flow sampling: given data x1 and timestep t, returns (x_t, target vector field u_t).

        Parameters
        ----------
        x1    : clean data  (B, ...)
        t     : continuous time, range [0,1]  (B,) or (B, 1, ...)
        noise : base noise x_0 ~ N(0,I); generated automatically if None

        Returns
        -------
        x_t  : interpolated intermediate state  (B, ...)
        u_t  : target vector field (straight-line direction) (B, ...), i.e. x1 - x0
        """
        if noise is None:
            noise = torch.randn_like(x1)

        # broadcast t to same number of dims as x1
        t_bc = t.view(t.shape[0], *([1] * (x1.ndim - 1)))

        # conditional flow (linear interpolation): add sigma_min minimum noise for numerical stability
        x_t = (1 - (1 - self.sigma_min) * t_bc) * noise + t_bc * x1

        # target vector field (straight-line direction)
        u_t = x1 - (1 - self.sigma_min) * noise

        return x_t, u_t

    # ------------------------------------------------------------------
    # Training loss: MSE(v_θ(x_t, t), u_t)
    # ------------------------------------------------------------------
    def p_losses(self,
                 model,
                 x1:    torch.Tensor,
                 noise: torch.Tensor | None = None
                 ) -> torch.Tensor:
        """
        Flow Matching loss for one training iteration.

        Parameters
        ----------
        model : vector field network v_θ(x_t, t_int), same interface as SimpleUNet
                  (takes x_t and integer timestep t_int = round(t * (T-1)))
        x1    : clean data  (B, C, H, W)
        noise : optional base noise (default None → random)

        Returns
        -------
        loss  : scalar, MSE loss
        """
        B      = x1.shape[0]
        device = x1.device

        # sample continuous time from Uniform(0, 1)
        t_cont = torch.rand(B, device=device)                     # (B,) ∈ [0,1]

        # map to integer timestep for timestep embedding
        t_int  = (t_cont * (self.T - 1)).long()                   # (B,) ∈ [0, T-1]

        # generate x_t and target vector field
        x_t, u_t = self.q_sample(x1, t_cont, noise)

        # network predicts vector field
        v_pred = model(x_t, t_int)

        # MSE loss
        return F.mse_loss(v_pred, u_t)
