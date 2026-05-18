"""
stable_diffusion/sampler.py  —  Latent Diffusion（Stable Diffusion 简化版）采样器

Stable Diffusion 架构简介
--------------------------
真实的 Stable Diffusion（Rombach et al., 2022）流程：

  训练阶段
  ├─ 阶段一：训练 VAE（像素空间 → 隐空间的可逆压缩）
  └─ 阶段二：冻结 VAE，在 **隐空间** 上训练扩散模型（UNet 预测噪声）

  推理阶段
  ├─ 1. 在隐空间采样噪声 z_T ~ N(0, I)
  ├─ 2. 用 DDIM/DDPM 去噪 z_T → z_0（在隐空间进行）
  └─ 3. VAE 解码：z_0 → 图像  x = decoder(z_0)

本文件实现 LatentDDIMSampler，对应推理阶段的完整逻辑：
  - 复用 ddim/sampler.py 中的 DDIMSampler（隐空间中同样是 DDIM 去噪）
  - 解码时调用 VAE.decode()

与 ddim/sampler.py 的关系
--------------------------
  DDIMSampler      : 仅在某一空间（像素或隐空间）内去噪，不关心 VAE
  LatentDDIMSampler: 封装 DDIMSampler + VAE decoder，完成隐空间去噪 + 解码
"""

import torch
from model.noise_schedule import NoiseSchedule
from model.vae            import VAE
from ddim.sampler         import DDIMSampler


class LatentDDIMSampler:
    """
    在 VAE 隐空间上运行 DDIM 去噪，然后解码到像素空间。

    Parameters
    ----------
    vae         : 已训练好（并冻结）的 VAE 实例
    schedule    : 已初始化并 .to(device) 的 NoiseSchedule 实例
    ddim_steps  : DDIM 推理步数（默认 50）
    eta         : DDIM 随机系数 η（0=纯确定性）
    """

    def __init__(self,
                 vae:        VAE,
                 schedule:   NoiseSchedule,
                 ddim_steps: int   = 50,
                 eta:        float = 0.0):
        self.vae         = vae
        self.ddim        = DDIMSampler(schedule, ddim_steps=ddim_steps, eta=eta)

    # ------------------------------------------------------------------
    # 完整生成：隐空间噪声 → DDIM 去噪 → VAE 解码 → 像素图像
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self,
               model,
               n_samples:  int,
               device:     torch.device,
               save_every: int | None = None
               ) -> tuple[torch.Tensor, list]:
        """
        生成 n_samples 张图像。

        Parameters
        ----------
        model      : 在隐空间上训练的噪声预测 UNet ε_θ(z_t, t)
        n_samples  : 生成图像数量
        device     : 计算设备
        save_every : 每隔该 DDIM 步数保存中间帧（可选，用于可视化）

        Returns
        -------
        images : 解码后的像素图像  (n_samples, C, H, W)，值域 [-1, 1]
        frames : 中间隐变量帧列表（仅 save_every 不为 None 时有内容）
        """
        # 隐空间形状：(B, latent_ch, latent_H, latent_W)
        latent_ch = self.vae.latent_ch
        # MNIST 28×28 → 7×7 隐空间（两次 stride=2）
        latent_h  = 7
        latent_w  = 7
        shape     = (n_samples, latent_ch, latent_h, latent_w)

        # 1. 在隐空间用 DDIM 去噪
        z0, frames = self.ddim.sample(model, shape, device, save_every=save_every)

        # 2. VAE 解码到像素空间
        images = self.vae.decode(z0)

        return images, frames
