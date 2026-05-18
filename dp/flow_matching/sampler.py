"""
flow_matching/sampler.py  —  Flow Matching ODE 采样器

对应论文：《Flow Matching for Generative Modeling》(Lipman et al., 2022)

推理算法（Euler ODE 积分）
--------------------------
给定已训练的向量场网络 v_θ(x_t, t)，从 x_0 ~ N(0,I) 出发，
用 Euler 方法数值积分 ODE：

    dx/dt = v_θ(x_t, t),   t: 0 → 1

离散化（步长 h = 1/N）：
    x_{t+h} = x_t + h · v_θ(x_t, t)

N 越大精度越高，N=100 通常已足够好。
也可替换为更高阶方法（Heun、RK4 等）。

与 DDPM/DDIM 的对比
--------------------
  DDPM   : 从 x_T（噪声）逐步去噪到 x_0（图像），t 从大到小
  DDIM   : 同上，但可跳步（非马尔可夫）
  FM ODE : t 从 0（噪声）到 1（图像），正方向积分，无"去噪"概念，更简洁
"""

import torch
import numpy as np
from model.flow_schedule import FlowSchedule


class FlowMatchingSampler:
    """
    基于 FlowSchedule 的 ODE 采样器（Euler 方法）。

    Parameters
    ----------
    schedule   : FlowSchedule 实例（主要用于获取 T 做时间步映射）
    ode_steps  : Euler 积分步数（默认 100）
    """

    def __init__(self,
                 schedule:  FlowSchedule,
                 ode_steps: int = 100):
        self.sch       = schedule
        self.ode_steps = ode_steps

    # ------------------------------------------------------------------
    # 单步 Euler 更新
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _euler_step(self,
                    model,
                    x_t:   torch.Tensor,
                    t_val: float,
                    dt:    float
                    ) -> torch.Tensor:
        """
        Euler 单步：x_{t+dt} = x_t + dt · v_θ(x_t, t)。

        Parameters
        ----------
        model  : 向量场网络 v_θ(x, t_int)
        x_t    : 当前状态  (B, ...)
        t_val  : 当前连续时间  ∈ [0, 1]
        dt     : 时间步长（正值，因为从 0 → 1）
        """
        B      = x_t.shape[0]
        T      = self.sch.T

        # 映射连续时间 t_val ∈ [0,1] 到整数时间步 ∈ [0, T-1]
        t_int  = torch.full((B,), int(t_val * (T - 1)), device=x_t.device, dtype=torch.long)
        t_int  = t_int.clamp(0, T - 1)

        # 预测向量场
        v_pred = model(x_t, t_int)

        return x_t + dt * v_pred

    # ------------------------------------------------------------------
    # Heun 二阶校正步（可选，精度更高）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _heun_step(self,
                   model,
                   x_t:   torch.Tensor,
                   t_val: float,
                   dt:    float
                   ) -> torch.Tensor:
        """
        Heun 方法（梯形规则）：用 Euler 预估，再用平均向量场校正。
        比 Euler 精度高一阶，推理步数可减半。
        """
        # 预估（Euler）
        x_pred = self._euler_step(model, x_t, t_val, dt)

        # 校正：用终点向量场取平均
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
    # 完整采样：x_0 ~ N(0,I) → x_1（图像）
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
        从标准正态噪声用 ODE 积分到数据分布。

        Parameters
        ----------
        model      : 向量场网络 v_θ，接口与 SimpleUNet 相同
        shape      : 输出形状，例如 (16, 1, 28, 28) 或隐空间形状
        device     : 计算设备
        method     : 积分方法，'euler' 或 'heun'
        save_every : 每隔该步数保存中间帧（可选）

        Returns
        -------
        x      : 最终生成样本（对应 t=1 处的数据）
        frames : 中间帧列表（顺序为 t=0→1）
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
