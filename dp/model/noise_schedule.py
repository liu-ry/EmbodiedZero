"""
model/noise_schedule.py  —  噪声调度表 + 前向加噪 + 训练损失

DDPM / DDIM / Stable Diffusion 等扩散模型变体共用的核心组件：

  1. β-schedule 计算（线性 / cosine 可选）
  2. 前向加噪 q_sample  — 给干净数据添加 t 步噪声
  3. 训练损失 p_losses   — MSE(预测噪声, 真实噪声)

各变体的区别仅在于"反向采样方式"，而非前向过程和训练损失，
因此本文件对 DDPM / DDIM 完全通用。

数学符号
--------
  β_t            : 每步噪声强度
  α_t = 1 - β_t
  ᾱ_t = ∏ α_s    : 累积乘积（alphas_cumprod）

前向过程
  q(x_t | x_0) = N(x_t; √ᾱ_t · x_0,  (1 - ᾱ_t) · I)
  即 x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0,I)

训练目标
  L = E[||ε - ε_θ(x_t, t)||²]
"""

import torch
import torch.nn.functional as F


class NoiseSchedule:
    """
    预计算扩散过程所需的所有系数。

    Parameters
    ----------
    num_timesteps : 扩散总步数 T
    schedule      : 'linear'（DDPM 原始）或 'cosine'（改进版，生成质量更好）
    beta_start    : 线性 schedule 的 β 起点
    beta_end      : 线性 schedule 的 β 终点
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
            # Nichol & Dhariwal 2021: 余弦调度在低时间步噪声更小
            steps = num_timesteps + 1
            t     = torch.linspace(0, num_timesteps, steps) / num_timesteps
            ab    = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            ab    = ab / ab[0]
            betas = (1 - ab[1:] / ab[:-1]).clamp(0, 0.9999)
        else:
            raise ValueError(f'Unknown schedule: {schedule}')

        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)   # ᾱ_t

        # —— 前向过程系数 ——
        self._reg('betas',            betas)
        self._reg('alphas',           alphas)
        self._reg('alphas_cumprod',   alphas_cumprod)
        self._reg('sqrt_ab',          alphas_cumprod.sqrt())           # √ᾱ_t
        self._reg('sqrt_one_minus_ab', (1.0 - alphas_cumprod).sqrt()) # √(1-ᾱ_t)

        # —— DDPM 反向采样系数（DDIM 不需要，但保留以兼容）——
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
        """按时间步索引系数，并广播到任意维度（图像/向量均适用）。"""
        return coef.gather(0, t).view(t.shape[0], *([1] * (ndim - 1)))

    # ------------------------------------------------------------------
    # 前向加噪：q(x_t | x_0)
    # 给干净数据添加 t 步噪声，DDPM / DDIM 训练时共用
    # ------------------------------------------------------------------
    def q_sample(self,
                 x0:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (x_t, noise)。

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab  = self._g(self.sqrt_ab,            t, x0.ndim)
        s_1ab = self._g(self.sqrt_one_minus_ab,  t, x0.ndim)
        return s_ab * x0 + s_1ab * noise, noise

    # ------------------------------------------------------------------
    # 训练损失：DDPM / DDIM 共用
    # ------------------------------------------------------------------
    def p_losses(self, model, x0: torch.Tensor) -> torch.Tensor:
        """
        随机采样 t，加噪，用网络预测噪声，返回 MSE loss。

        Parameters
        ----------
        model : 噪声预测网络 ε_θ，接受 (x_t, t) 返回预测噪声
        x0    : 干净数据 (B, ...)
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        return F.mse_loss(model(x_t, t), noise)
