"""
ddpm/sampler.py  —  DDPM 祖先采样器（Ancestral Sampler）

DDPM 特有的反向采样方式：
  x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t)) + σ_t · z
  其中 z ~ N(0,I)（t>0 时），t=0 时不加噪声

与 DDIM 的区别
--------------
  DDPM : 每步加随机噪声 σ_t·z  → 随机采样，需要全部 T 步
  DDIM : 每步确定性更新          → 可跳步，100 步即可得到与 1000 步相近的质量

本文件仅包含 DDPM 特有的采样逻辑；前向加噪和训练损失在 model/ 中。
"""

import torch
from model.noise_schedule import NoiseSchedule


class DDPMSampler:
    """
    基于 NoiseSchedule 的 DDPM 祖先采样器。

    Parameters
    ----------
    schedule : 已初始化并 .to(device) 的 NoiseSchedule 实例
    """

    def __init__(self, schedule: NoiseSchedule):
        self.sch = schedule

    # ------------------------------------------------------------------
    # 单步反向去噪：x_t → x_{t-1}
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t_val: int) -> torch.Tensor:
        """DDPM 单步去噪（带随机噪声）。"""
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
        从纯高斯噪声逐步去噪，生成样本。

        Parameters
        ----------
        model      : 噪声预测网络 ε_θ
        shape      : 输出形状，例如 (16, 1, 28, 28)
        device     : 计算设备
        save_every : 每隔该步数保存一帧（用于可视化去噪轨迹）

        Returns
        -------
        x      : 最终生成样本  shape
        frames : 中间帧列表（仅 save_every 不为 None 时有内容，顺序为 T→0）
        """
        x      = torch.randn(shape, device=device)
        frames = []

        for t_val in reversed(range(self.sch.T)):
            x = self.p_sample(model, x, t_val)
            if save_every is not None and (
                    t_val % save_every == 0 or t_val == self.sch.T - 1):
                frames.append(x.clone().cpu())

        return x, frames
