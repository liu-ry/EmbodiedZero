"""
model/model.py  —  UNet 噪声预测骨干网络

所有扩散模型变体（DDPM / DDIM / Stable Diffusion 等）都使用同一个
"噪声预测网络"来估计 ε_θ，该文件提供可复用的 SimpleUNet 实现。

网络结构
--------
  编码器（下采样）→ 瓶颈 → 解码器（上采样）
  每层注入 Sinusoidal 时间步嵌入；编解码器之间有 skip connection。

输入 / 输出
-----------
  x_t : (B, C, H, W)   加噪图像
  t   : (B,)            整数时间步
  → 预测噪声 ε_θ : (B, C, H, W)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 工具：自适应 GroupNorm（channel 数少时自动减少 group 数）
# ---------------------------------------------------------------------------
def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(8, ch)
    while ch % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, ch)


# ---------------------------------------------------------------------------
# 时间步正弦嵌入
# 将整数 t 映射为连续向量，让网络感知当前所处的扩散阶段。
# 与 Transformer 位置编码完全一致。
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t.float()[:, None] * emb[None, :]           # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)      # (B, dim)


# ---------------------------------------------------------------------------
# 带时间嵌入的残差卷积块
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Conv → Norm → SiLU → Conv，加上时间步偏置和残差捷径。
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)

        self.block1 = nn.Sequential(
            _norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            _norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]  # 时间偏置广播到空间
        h = self.block2(h)
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# SimpleUNet（针对 MNIST 28×28 设计，可扩展到更大分辨率）
# ---------------------------------------------------------------------------
class SimpleUNet(nn.Module):
    """
    轻量级 UNet，适用于灰度 28×28 图像。

    通道: 1 → 64 → 128 → 256(瓶颈) → 128 → 64 → 1
    空间: 28 → 14 → 7(瓶颈) → 14 → 28

    扩展提示
    --------
    - 增大 base_channels 可提升生成质量
    - 为支持文本/类别条件（Stable Diffusion），可在瓶颈层加入 Cross-Attention
    - 为支持更高分辨率，增加更多下采样/上采样层即可
    """

    def __init__(self,
                 in_channels:   int = 1,
                 base_channels: int = 64,
                 time_emb_dim:  int = 128):
        super().__init__()
        t  = time_emb_dim
        c1 = base_channels        # 64
        c2 = base_channels * 2    # 128
        c3 = base_channels * 4    # 256（瓶颈）

        # 时间嵌入 MLP
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(t),
            nn.Linear(t, t * 2),
            nn.SiLU(),
            nn.Linear(t * 2, t),
        )

        # 编码器
        self.enc1  = ResBlock(in_channels, c1, t)               # 28×28
        self.down1 = nn.Conv2d(c1, c1, 4, stride=2, padding=1)  # → 14×14
        self.enc2  = ResBlock(c1, c2, t)                         # 14×14
        self.down2 = nn.Conv2d(c2, c2, 4, stride=2, padding=1)  # → 7×7

        # 瓶颈
        self.mid1 = ResBlock(c2, c3, t)
        self.mid2 = ResBlock(c3, c2, t)

        # 解码器（skip connection：上采样输出 concat 对应编码层特征）
        self.up2  = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)  # → 14×14
        self.dec2 = ResBlock(c2 + c2, c1, t)
        self.up1  = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)  # → 28×28
        self.dec1 = ResBlock(c1 + c1, c1, t)

        # 输出投影
        self.out = nn.Sequential(
            _norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        t_emb = self.time_emb(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        m  = self.mid2(self.mid1(self.down2(e2), t_emb), t_emb)

        # 用双线性插值对齐 skip connection 的空间尺寸（支持任意输入分辨率）
        up2_out = F.interpolate(self.up2(m),  size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2_out, e2], dim=1), t_emb)
        up1_out = F.interpolate(self.up1(d2), size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1_out, e1], dim=1), t_emb)

        return self.out(d1)
