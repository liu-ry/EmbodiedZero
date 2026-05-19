"""
Vision Transformer (ViT) 及变体
================================
本模块实现基于 Transformer 的视觉模型，用于 STL-10 图像分类：

1. PatchEmbedding      - 图像 Patch 切分与线性嵌入
2. VisionTransformer   - 标准 ViT（带 CLS token + 可学习 2D PE）
3. ViTWithRoPE         - RoPE 位置编码版本（无 PE 加法，直接作用在 QK）
4. SwinBlock           - 窗口注意力版（Swin 风格，局部+移位窗口）
5. build_vit           - 工厂函数，按名称构建模型

ViT 核心流程（以标准 ViT 为例）
---------------------------------
  1. 将图像切成 P×P 的 patch，每个 patch 展平后线性投影
     Image: (B, C, H, W) → Patches: (B, N, d_model)  N = (H/P)*(W/P)
  2. 在序列前端插入可学习的 [CLS] token
  3. 加上位置编码
  4. 经过 L 层 Transformer Encoder
  5. 取 [CLS] token 的输出，经分类头得到预测

论文: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
     (Dosovitskiy et al., 2020)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import SinusoidalPE, LearnablePE, Learnable2DPE, RotaryPE
from .attention import (
    MultiHeadAttention,
    WindowAttention,
    RoPEMultiHeadAttention,
    ScaledDotProductAttention,
)
from .transformer_components import FeedForwardNetwork, TransformerEncoder


# ─────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    将图像切分为不重叠的 patch，并线性投影到 d_model 维。

    实现方式: 用步长=patch_size 的卷积代替手动 reshape，更高效。

    参数:
        img_size   - 输入图像边长（正方形）
        patch_size - 每个 patch 的边长
        in_chans   - 输入通道数（RGB = 3）
        d_model    - 嵌入维度
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, d_model: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 必须整除 patch_size"
        self.grid_size  = img_size // patch_size        # 每维的 patch 数
        self.n_patches  = self.grid_size ** 2           # 总 patch 数
        self.patch_size = patch_size

        # 用卷积实现 patch 切分 + 线性投影（等价于 flatten 后做 Linear）
        self.proj = nn.Conv2d(
            in_chans, d_model,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回: (B, N, d_model)，N = grid_size²
        """
        x = self.proj(x)            # (B, d_model, grid_h, grid_w)
        x = x.flatten(2)            # (B, d_model, N)
        x = x.transpose(1, 2)       # (B, N, d_model)
        return self.norm(x)


# ─────────────────────────────────────────────────────────
# 标准 ViT
# ─────────────────────────────────────────────────────────
class VisionTransformer(nn.Module):
    """
    标准 Vision Transformer (ViT-B 风格)。

    位置编码: 可选 'learnable'（默认）或 'sinusoidal'
    CLS token: 可选 True（分类头取 CLS）或 False（全局平均池化）

    参数:
        img_size   - 输入图像边长
        patch_size - Patch 大小（ViT-S: 16, ViT-Ti: 32）
        in_chans   - 输入通道数
        n_classes  - 分类类别数
        d_model    - 嵌入维度
        n_layers   - Transformer 层数
        n_heads    - 注意力头数
        d_ff       - FFN 隐藏维度（默认 4*d_model）
        dropout    - Dropout
        pe_type    - 位置编码类型 ('learnable' | 'sinusoidal' | '2d')
        use_cls    - 是否使用 CLS token
    """

    def __init__(
        self,
        img_size:   int   = 96,
        patch_size: int   = 16,
        in_chans:   int   = 3,
        n_classes:  int   = 10,
        d_model:    int   = 384,
        n_layers:   int   = 6,
        n_heads:    int   = 6,
        d_ff:       int   = None,
        dropout:    float = 0.1,
        pe_type:    str   = "learnable",
        use_cls:    bool  = True,
    ):
        super().__init__()
        self.use_cls = use_cls

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, d_model)
        n_patches = self.patch_embed.n_patches
        grid_size = self.patch_embed.grid_size

        # 2. CLS token
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            seq_len = n_patches + 1
        else:
            seq_len = n_patches

        # 3. 位置编码
        if pe_type == "learnable":
            self.pos_embed = LearnablePE(d_model, max_len=seq_len, dropout=dropout)
        elif pe_type == "sinusoidal":
            self.pos_embed = SinusoidalPE(d_model, max_len=seq_len, dropout=dropout)
        elif pe_type == "2d":
            # 2D PE 只作用在 patch 部分，CLS token 不加
            self.pos_embed = Learnable2DPE(d_model, grid_h=grid_size, grid_w=grid_size, dropout=dropout)
            self.cls_pe_dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"未知位置编码类型: {pe_type}")
        self.pe_type = pe_type

        # 4. Transformer Encoder
        self.encoder = TransformerEncoder(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, pre_norm=True, activation="gelu"
        )

        # 5. 分类头
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回: logits (B, n_classes)
        """
        B = x.size(0)

        # Patch Embedding: (B, N, d_model)
        tokens = self.patch_embed(x)

        # 拼接 CLS token
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            tokens = torch.cat([cls, tokens], dim=1) # (B, N+1, d_model)

        # 位置编码
        if self.pe_type == "2d":
            if self.use_cls:
                # CLS token 不加 2D PE，patch 加 2D PE
                cls_tok  = self.cls_pe_dropout(tokens[:, :1])
                patch_tok = self.pos_embed(tokens[:, 1:])
                tokens = torch.cat([cls_tok, patch_tok], dim=1)
            else:
                tokens = self.pos_embed(tokens)
        else:
            tokens = self.pos_embed(tokens)

        # Transformer Encoder
        features = self.encoder(tokens)  # (B, N+1, d_model)

        # 分类
        if self.use_cls:
            cls_feat = features[:, 0]    # CLS token
        else:
            cls_feat = features.mean(dim=1)  # 全局平均池化

        logits = self.head(self.head_norm(cls_feat))
        return logits

    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1):
        """
        提取指定层的注意力权重（用于可视化 Attention Map）。
        """
        B = x.size(0)
        tokens = self.patch_embed(x)
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        if self.pe_type == "2d":
            if self.use_cls:
                cls_tok = self.cls_pe_dropout(tokens[:, :1])
                patch_tok = self.pos_embed(tokens[:, 1:])
                tokens = torch.cat([cls_tok, patch_tok], dim=1)
            else:
                tokens = self.pos_embed(tokens)
        else:
            tokens = self.pos_embed(tokens)

        attn_maps = []
        for i, layer in enumerate(self.encoder.layers):
            norm_x = layer.norm1(tokens)
            _, attn_w = layer.self_attn(norm_x, norm_x, norm_x)  # (B, H, N, N)
            attn_maps.append(attn_w)
            attn_out, _ = layer.self_attn(norm_x, norm_x, norm_x)
            tokens = tokens + layer.dropout(attn_out)
            tokens = tokens + layer.dropout(layer.ffn(layer.norm2(tokens)))

        return attn_maps[layer_idx]  # (B, H, N+1, N+1)


# ─────────────────────────────────────────────────────────
# Swin 风格的 ViT（窗口注意力）
# ─────────────────────────────────────────────────────────
class SwinEncoderLayer(nn.Module):
    """
    单层 Swin-style 编码器（窗口注意力 + FFN）。
    包含两个连续的 block：
      - 第一个：标准窗口注意力（W-MSA）
      - 第二个：移位窗口注意力（SW-MSA），实现跨窗口信息交流
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.w_attn  = WindowAttention(d_model, n_heads, window_size=window_size, dropout=dropout)
        self.sw_attn = WindowAttention(d_model, n_heads, window_size=window_size, dropout=dropout)
        self.ffn1 = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.ffn2 = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.shift   = window_size // 2

    def _cyclic_shift(self, x: torch.Tensor, shift: int) -> torch.Tensor:
        """循环移位实现移位窗口"""
        return torch.roll(x, shifts=-shift, dims=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: 标准窗口注意力
        attn_out, _ = self.w_attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn1(self.norm2(x)))

        # Block 2: 移位窗口注意力（循环移位后计算，再移回来）
        x_shift = self._cyclic_shift(x, self.shift)
        attn_out, _ = self.sw_attn(self.norm3(x_shift))
        x_shift = x_shift + self.dropout(attn_out)
        x_shift = x_shift + self.dropout(self.ffn2(self.norm4(x_shift)))
        x = self._cyclic_shift(x_shift, -self.shift)  # 移回
        return x


class SwinViT(nn.Module):
    """
    Swin Transformer 风格的 ViT（局部窗口注意力 + 移位窗口）。

    与标准 ViT 不同:
      - 使用窗口注意力替换全局自注意力
      - 通过循环移位窗口实现跨窗口信息交流
      - 复杂度: O(N * w²) 而不是 O(N²)

    参数:
        window_size - 窗口大小（必须整除 n_patches）
    """

    def __init__(
        self,
        img_size:    int   = 96,
        patch_size:  int   = 16,
        in_chans:    int   = 3,
        n_classes:   int   = 10,
        d_model:     int   = 384,
        n_layers:    int   = 4,
        n_heads:     int   = 6,
        d_ff:        int   = None,
        dropout:     float = 0.1,
        window_size: int   = 6,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, d_model)
        n_patches = self.patch_embed.n_patches
        assert n_patches % window_size == 0, \
            f"patch 数 {n_patches} 必须整除 window_size {window_size}"

        self.pos_embed = LearnablePE(d_model, max_len=n_patches, dropout=dropout)

        self.layers = nn.ModuleList([
            SwinEncoderLayer(d_model, n_heads, window_size, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)      # (B, N, d)
        tokens = self.pos_embed(tokens)   # + 位置编码
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        logits = self.head(tokens.mean(dim=1))  # 全局平均池化
        return logits


# ─────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────
def build_vit(
    model_name: str = "vit_small",
    img_size:   int = 96,
    n_classes:  int = 10,
    dropout:    float = 0.1,
) -> nn.Module:
    """
    按名称构建 ViT 模型（适配 STL-10 的轻量配置）。

    可用名称:
        vit_tiny    - d=192, L=4, H=3  （最快）
        vit_small   - d=384, L=6, H=6  （推荐）
        vit_base    - d=512, L=8, H=8
        vit_sinpe   - vit_small + 正弦位置编码
        vit_2dpe    - vit_small + 2D 位置编码
        swin_small  - Swin-style, d=384, L=4, H=6, w=6
    """
    configs = {
        "vit_tiny":  dict(d_model=192, n_layers=4, n_heads=3,  pe_type="learnable"),
        "vit_small": dict(d_model=384, n_layers=6, n_heads=6,  pe_type="learnable"),
        "vit_base":  dict(d_model=512, n_layers=8, n_heads=8,  pe_type="learnable"),
        "vit_sinpe": dict(d_model=384, n_layers=6, n_heads=6,  pe_type="sinusoidal"),
        "vit_2dpe":  dict(d_model=384, n_layers=6, n_heads=6,  pe_type="2d"),
    }

    if model_name == "swin_small":
        return SwinViT(img_size=img_size, patch_size=16, n_classes=n_classes,
                       d_model=384, n_layers=4, n_heads=6, dropout=dropout, window_size=6)

    assert model_name in configs, f"未知模型名: {model_name}，可选: {list(configs.keys()) + ['swin_small']}"
    cfg = configs[model_name]
    return VisionTransformer(
        img_size=img_size, patch_size=16, in_chans=3,
        n_classes=n_classes, dropout=dropout, **cfg
    )


# ─────────────────────────────────────────────────────────
# 功能验证
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 2
    x = torch.randn(B, 3, 96, 96)

    for name in ["vit_tiny", "vit_small", "vit_sinpe", "vit_2dpe", "swin_small"]:
        model = build_vit(name, img_size=96, n_classes=10)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[{name:12s}]  输出: {logits.shape}  参数量: {n_params:.2f}M")

    print("\n所有 ViT 模型测试通过 ✓")
