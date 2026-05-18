"""
ddim/sampler.py  —  DDIM 确定性采样器

论文：《Denoising Diffusion Implicit Models》(Song et al., 2020)
     https://arxiv.org/abs/2010.02502

DDIM 与 DDPM 的区别
--------------------
  DDPM : 每步加随机噪声 σ_t · z（马尔可夫链），必须走完全部 T 步
  DDIM : 将反向过程改写为非马尔可夫形式，支持 **跳步**（sub-sampling）：
           - 只用 S（≪T）步即可生成高质量样本（例如 T=1000, S=50）
           - 当 η=0 时完全确定性；当 η=1 时退化为 DDPM 随机采样

数学推导（单步更新）
--------------------
已知 x_t 和网络预测 ε_θ(x_t, t)：

  1. 预测 x_0：
       x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t

  2. 计算方差：
       σ_t = η · √((1-ᾱ_{t-1}) / (1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})

  3. 更新 x_{t-1}：
       x_{t-1} = √ᾱ_{t-1} · x̂_0
               + √(1 - ᾱ_{t-1} - σ_t²) · ε_θ
               + σ_t · z,    z ~ N(0, I)

  当 η=0 时 σ_t=0，更新完全确定性（DDIM 原始设置）。

实现说明
--------
  - 与 DDPMSampler 使用相同的 NoiseSchedule，无需重新训练模型
  - `ddim_steps` 控制采样步数；`eta` 控制随机性（0=确定，1≈DDPM）
"""

import torch
import numpy as np
from model.noise_schedule import NoiseSchedule


class DDIMSampler:
    """
    基于 NoiseSchedule 的 DDIM 采样器。

    Parameters
    ----------
    schedule    : 已初始化并 .to(device) 的 NoiseSchedule 实例
    ddim_steps  : 采样步数 S（≤ schedule.T），默认 50
    eta         : DDIM 随机系数 η；0=纯确定性，1=近似 DDPM
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
    # 单步反向去噪：x_t → x_{t-prev}
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self,
                 model,
                 x_t:    torch.Tensor,
                 t_val:  int,
                 t_prev: int,
                 ) -> torch.Tensor:
        """
        DDIM 单步去噪。

        Parameters
        ----------
        model  : 噪声预测网络 ε_θ
        x_t    : 当前加噪样本  (B, C, H, W)
        t_val  : 当前时间步（整数）
        t_prev : 上一（更小）时间步（整数），-1 表示已到 t=0
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
            # t_prev = -1（已在 t=0），ᾱ_{-1} 定义为 1
            ab_prev = torch.ones_like(ab_t)

        sqrt_ab_prev   = ab_prev.sqrt()
        sqrt_1mab_prev = (1.0 - ab_prev).sqrt()

        # 1. 预测 x_0
        x0_pred = (x_t - sqrt_1mab_t * eps_pred) / sqrt_ab_t

        # 2. DDIM 方差 σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
        #    当 η=0 时 σ_t=0（确定性 DDIM）
        if self.eta > 0 and t_prev >= 0:
            sigma = (self.eta
                     * ((1 - ab_prev) / (1 - ab_t)).sqrt()
                     * (1 - ab_t / ab_prev).sqrt())
        else:
            sigma = torch.zeros_like(ab_t)

        # 3. "指向 x_t" 的方向分量（去掉方差部分后的噪声方向）
        dir_xt_coef = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt()

        # 4. x_{t-1}
        x_prev = sqrt_ab_prev * x0_pred + dir_xt_coef * eps_pred
        if self.eta > 0 and t_prev >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    # ------------------------------------------------------------------
    # 完整反向采样：x_T ~ N(0,I) → x_0
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               model,
               shape:      tuple,
               device:     torch.device,
               save_every: int | None = None
               ) -> tuple[torch.Tensor, list]:
        """
        从纯高斯噪声用 DDIM 跳步去噪，生成样本。

        Parameters
        ----------
        model      : 噪声预测网络 ε_θ
        shape      : 输出形状，例如 (16, 1, 28, 28)
        device     : 计算设备
        save_every : 每隔该步数（在 DDIM 步序列中）保存一帧，用于可视化

        Returns
        -------
        x      : 最终生成样本
        frames : 中间帧列表（顺序为 T→0）
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
