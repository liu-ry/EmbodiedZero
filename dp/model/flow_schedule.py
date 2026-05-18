"""
model/flow_schedule.py  —  Flow Matching 训练调度

论文：《Flow Matching for Generative Modeling》(Lipman et al., 2022)
     《Improving and Generalizing Flow Matching》(Albergo & Vanden-Eijnden, 2023)

Flow Matching 核心思想
----------------------
与 DDPM/DDIM 不同，Flow Matching 不使用马尔可夫加噪过程，而是：
  1. 定义一条从噪声分布 p_0 = N(0,I) 到数据分布 p_1 的确定性流 (flow)
  2. 通过拟合向量场 v_θ(x_t, t) 来学习这条流
  3. 推理时从 x_0 ~ N(0,I) 出发，用 ODE 求解器（Euler 等）积分到 t=1

最优传输条件流匹配（OT-CFM）
----------------------------
给定噪声 x_0 ~ N(0,I) 和数据 x_1：

  条件流（直线插值）：
      x_t = (1 - t) · x_0 + t · x_1,   t ∈ [0, 1]

  条件向量场（直线方向）：
      u_t(x_t | x_1) = x_1 - x_0

  训练目标（MSE 拟合向量场）：
      L_FM = E_{t,x_0,x_1} [||v_θ(x_t, t) - (x_1 - x_0)||²]

与 DDPM 的对比
--------------
  DDPM    : 离散时间步（T=1000），学习预测噪声 ε，需要马尔可夫链反向采样
  Flow FM : 连续时间 t ∈ [0,1]，学习向量场 v，推理用 ODE（任意步数）

实现说明
--------
  - 时间步 t 从 Uniform(0, 1) 采样（连续）
  - 网络输入时间步需映射到整数或归一化浮点，本实现将 t 缩放到整数 [0, T-1]
    并复用 SimpleUNet 的整数时间步嵌入（T 较小即可，默认 1000）
  - FlowSchedule 只负责数据准备和训练损失；推理逻辑在 flow_matching/sampler.py
"""

import torch
import torch.nn.functional as F


class FlowSchedule:
    """
    Flow Matching 的训练工具类。

    Parameters
    ----------
    num_timesteps : 整数离散化步数（仅用于时间步嵌入，不影响连续流的数学）
                   推理时可选任意步数的 Euler 积分
    sigma_min     : 条件流的最小噪声（为 0 时是严格直线 OT；加小量数值稳定）
    """

    def __init__(self,
                 num_timesteps: int   = 1000,
                 sigma_min:     float = 1e-4):
        self.T         = num_timesteps
        self.sigma_min = sigma_min

    def to(self, device):
        # FlowSchedule 没有需要迁移的 Tensor（纯计算），保持接口统一
        self._device = device
        return self

    # ------------------------------------------------------------------
    # 采样中间状态 x_t（训练时调用）
    # ------------------------------------------------------------------
    def q_sample(self,
                 x1:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        OT-CFM 条件流采样：给定数据 x1 和时间步 t，返回 (x_t, 目标向量场 u_t)。

        Parameters
        ----------
        x1    : 干净数据  (B, ...)
        t     : 连续时间，范围 [0,1]  (B,) 或 (B, 1, ...)
        noise : 基底噪声 x_0 ~ N(0,I)，若 None 则自动生成

        Returns
        -------
        x_t  : 插值后的中间状态  (B, ...)
        u_t  : 目标向量场（直线方向）(B, ...)，即 x1 - x0
        """
        if noise is None:
            noise = torch.randn_like(x1)

        # t 广播到与 x1 相同维度
        t_bc = t.view(t.shape[0], *([1] * (x1.ndim - 1)))

        # 条件流（直线插值）：加入 sigma_min 的最小噪声保证数值稳定
        x_t = (1 - (1 - self.sigma_min) * t_bc) * noise + t_bc * x1

        # 目标向量场（直线方向）
        u_t = x1 - (1 - self.sigma_min) * noise

        return x_t, u_t

    # ------------------------------------------------------------------
    # 训练损失：MSE(v_θ(x_t, t), u_t)
    # ------------------------------------------------------------------
    def p_losses(self,
                 model,
                 x1:    torch.Tensor,
                 noise: torch.Tensor | None = None
                 ) -> torch.Tensor:
        """
        一次训练迭代的 Flow Matching 损失。

        Parameters
        ----------
        model : 向量场网络 v_θ(x_t, t_int)，接口与 SimpleUNet 相同
                  （输入 x_t 和整数时间步 t_int = round(t * (T-1))）
        x1    : 干净数据  (B, C, H, W)
        noise : 可选，指定基底噪声（默认 None → 随机）

        Returns
        -------
        loss  : 标量，MSE 损失
        """
        B      = x1.shape[0]
        device = x1.device

        # 从 Uniform(0, 1) 采样连续时间
        t_cont = torch.rand(B, device=device)                     # (B,) ∈ [0,1]

        # 映射到整数时间步供时间嵌入使用
        t_int  = (t_cont * (self.T - 1)).long()                   # (B,) ∈ [0, T-1]

        # 生成 x_t 和目标向量场
        x_t, u_t = self.q_sample(x1, t_cont, noise)

        # 网络预测向量场
        v_pred = model(x_t, t_int)

        # MSE 损失
        return F.mse_loss(v_pred, u_t)
