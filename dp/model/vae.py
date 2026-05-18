"""
model/vae.py  —  轻量级 VAE（变分自编码器）

Stable Diffusion 的核心思路是先用 VAE 把像素空间压缩到低维 **隐空间**，
再在隐空间上训练扩散模型。生成时先在隐空间采样，再用 VAE 解码到图像。

本文件针对 MNIST 28×28 灰度图设计，将图像压缩到 (latent_ch, 7, 7)。
    像素空间  : (B, 1,  28, 28)
    隐空间    : (B, 4,   7,  7)   ← 空间 4× 下采样，通道 4

数学背景
--------
  编码器输出 μ 和 log σ²（均为隐空间大小），然后用重参数化技巧采样：
      z = μ + σ · ε,   ε ~ N(0, I)
  KL 散度作为正则化损失（让隐分布靠近标准正态）：
      KL = -½ · Σ (1 + log σ² - μ² - σ²)
  重构损失（像素空间 MSE）：
      Recon = ||x - x̂||²
  总损失：
      L_VAE = Recon + kl_weight · KL

提示
----
  真实 Stable Diffusion（LDM）使用感知损失 + 判别器替代纯 MSE，
  这里为保持依赖简洁仅使用 MSE。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 工具：自适应 GroupNorm
# ---------------------------------------------------------------------------
def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(8, ch)
    while ch % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, ch)


# ---------------------------------------------------------------------------
# 编码器：像素空间 → μ, log_var（隐空间）
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """
    两次 stride=2 卷积，将 28×28 → 7×7，通道 1 → latent_ch*2（μ 和 log σ²）。
    """

    def __init__(self, in_channels: int = 1, latent_ch: int = 4, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # 28×28 → 14×14
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            _norm(base_ch),
            nn.SiLU(),
            # 14×14 → 7×7
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            _norm(base_ch * 2),
            nn.SiLU(),
            # 保持 7×7，输出 μ 和 log σ²（共 latent_ch*2 个通道）
            nn.Conv2d(base_ch * 2, latent_ch * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (μ, log_var)，各形状 (B, latent_ch, H/4, W/4)。"""
        out = self.net(x)
        mu, log_var = out.chunk(2, dim=1)
        return mu, log_var


# ---------------------------------------------------------------------------
# 解码器：隐向量 z → 重构图像
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    两次转置卷积上采样，将 7×7 → 28×28，通道 latent_ch → in_channels。
    """

    def __init__(self, in_channels: int = 1, latent_ch: int = 4, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # 7×7 → 7×7（通道映射）
            nn.Conv2d(latent_ch, base_ch * 2, 3, padding=1),
            _norm(base_ch * 2),
            nn.SiLU(),
            # 7×7 → 14×14
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            _norm(base_ch),
            nn.SiLU(),
            # 14×14 → 28×28
            nn.ConvTranspose2d(base_ch, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),   # 输出 [-1, 1]，与训练数据归一化一致
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """
    轻量级卷积 VAE。

    Parameters
    ----------
    in_channels : 输入图像通道数（MNIST=1）
    latent_ch   : 隐空间通道数（决定压缩率）
    base_ch     : 编/解码器基础通道数
    kl_weight   : KL 散度权重（越大隐空间越接近正态，但重构质量略降）
    """

    def __init__(self,
                 in_channels: int   = 1,
                 latent_ch:   int   = 4,
                 base_ch:     int   = 32,
                 kl_weight:   float = 1e-3):
        super().__init__()
        self.encoder    = Encoder(in_channels, latent_ch, base_ch)
        self.decoder    = Decoder(in_channels, latent_ch, base_ch)
        self.kl_weight  = kl_weight
        self.latent_ch  = latent_ch

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (μ, log_var)。"""
        return self.encoder(x)

    def reparameterize(self,
                       mu:      torch.Tensor,
                       log_var: torch.Tensor
                       ) -> torch.Tensor:
        """重参数化：z = μ + σ · ε。训练时 ε ~ N(0,I)；推理时可直接用 μ。"""
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → 重构图像。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        完整前向：encode → reparameterize → decode。

        Returns
        -------
        recon : 重构图像  (B, C, H, W)
        loss  : 标量，VAE 总损失 = Recon + kl_weight * KL
        """
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z)

        # 重构损失（MSE，在像素维度求平均）
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL 散度
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recon_loss + self.kl_weight * kl_loss
        return recon, loss

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理阶段：直接使用 μ（不加噪声），返回隐变量。
        用于 Stable Diffusion 的"把训练图像编码到隐空间"步骤。
        """
        mu, _ = self.encode(x)
        return mu
