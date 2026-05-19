"""
Transformer 基础组件
====================
本模块实现 Transformer 的核心构建模块：

1. FeedForwardNetwork       - 前馈神经网络（标准 / SwiGLU / GeGLU 变体）
2. TransformerEncoderLayer  - 编码器层（Pre-Norm / Post-Norm 均支持）
3. TransformerDecoderLayer  - 解码器层（含自注意力 + 交叉注意力）
4. TransformerEncoder       - 编码器堆叠
5. TransformerDecoder       - 解码器堆叠

层归一化位置对比
---------------
  Post-Norm (原始): x = LayerNorm(x + Sublayer(x))
    - 原版 Transformer 设计
    - 训练较不稳定，需要 warmup

  Pre-Norm (现代):  x = x + Sublayer(LayerNorm(x))
    - GPT-2/3、LLaMA 等大模型使用
    - 训练更稳定，梯度流动更好
    - 但输出层需额外 LayerNorm

FFN 激活函数变体
-----------------
  标准 ReLU FFN (原版 Transformer):
    FFN(x) = max(0, xW_1 + b_1) W_2 + b_2

  SwiGLU (PaLM / LLaMA):
    FFN(x) = (Swish(xW_1) ⊙ xW_3) W_2
    参数量更少但效果更好（已成为 LLM 标配）

  GeGLU (GPT-J / T5v1.1):
    FFN(x) = (GELU(xW_1) ⊙ xW_3) W_2
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, CrossAttention


# ─────────────────────────────────────────────────────────
# 1. 前馈网络 (Feed-Forward Network)
# ─────────────────────────────────────────────────────────
class FeedForwardNetwork(nn.Module):
    """
    Transformer 中的前馈网络，支持三种激活变体：
      - 'relu'   : 标准 ReLU FFN（原版 Transformer）
      - 'gelu'   : GELU 激活（BERT / GPT-2）
      - 'swiglu' : SwiGLU 门控 FFN（LLaMA / PaLM，无偏置）

    参数:
        d_model   - 输入/输出维度
        d_ff      - 隐藏层维度（通常 4 * d_model）
        dropout   - Dropout 概率
        activation- 激活函数类型 ('relu' | 'gelu' | 'swiglu')
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.activation_type = activation

        if activation in ("relu", "gelu"):
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            if activation == "relu":
                self.act = nn.ReLU()
            else:
                self.act = nn.GELU()

        elif activation == "swiglu":
            # SwiGLU: 两个并行投影 + 门控，通常将 d_ff 缩小以保持参数量相当
            d_ff_swi = int(d_ff * 2 / 3)  # 约为 2/3 * 4d = 8/3 d
            self.w1 = nn.Linear(d_model, d_ff_swi, bias=False)  # 门
            self.w3 = nn.Linear(d_model, d_ff_swi, bias=False)  # 值
            self.w2 = nn.Linear(d_ff_swi, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"不支持的激活类型: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            # SwiGLU: Swish(xW1) ⊙ (xW3)
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        else:
            return self.dropout(self.w2(self.act(self.w1(x))))


# ─────────────────────────────────────────────────────────
# 2. 编码器层
# ─────────────────────────────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    """
    单个 Transformer 编码器层。

    Pre-Norm (默认):
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    Post-Norm (原版 Transformer):
        x = LN(x + Attention(x))
        x = LN(x + FFN(x))

    参数:
        d_model    - 嵌入维度
        n_heads    - 注意力头数
        d_ff       - FFN 隐藏维度
        dropout    - Dropout 概率
        pre_norm   - True=Pre-Norm（推荐），False=Post-Norm
        activation - FFN 激活函数
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout=dropout, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        src_mask: Optional[torch.Tensor] = None,  # 填充掩码
    ) -> torch.Tensor:
        if self.pre_norm:
            # Pre-Norm: 归一化在残差连接之前
            attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=src_mask)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # Post-Norm: 归一化在残差连接之后
            attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────
# 3. 解码器层
# ─────────────────────────────────────────────────────────
class TransformerDecoderLayer(nn.Module):
    """
    单个 Transformer 解码器层，包含三个子层：
      1. 带因果掩码的自注意力（Masked Self-Attention）
      2. 对编码器输出的交叉注意力（Cross-Attention）
      3. 前馈网络（FFN）

    参数同 TransformerEncoderLayer，增加 cross_attn。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout=dropout)
        self.ffn        = FeedForwardNetwork(d_model, d_ff, dropout=dropout, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt:       torch.Tensor,                       # (B, T_dec, d_model) 解码器输入
        memory:    torch.Tensor,                       # (B, T_enc, d_model) 编码器输出
        tgt_mask:  Optional[torch.Tensor] = None,      # 因果掩码
        mem_mask:  Optional[torch.Tensor] = None,      # 编码器填充掩码
    ) -> torch.Tensor:
        if self.pre_norm:
            # 1. 自注意力
            sa_out, _ = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), mask=tgt_mask)
            tgt = tgt + self.dropout(sa_out)
            # 2. 交叉注意力
            ca_out, _ = self.cross_attn(self.norm2(tgt), memory, src_mask=mem_mask)
            tgt = tgt + self.dropout(ca_out)
            # 3. FFN
            tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        else:
            sa_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
            tgt = self.norm1(tgt + self.dropout(sa_out))
            ca_out, _ = self.cross_attn(tgt, memory, src_mask=mem_mask)
            tgt = self.norm2(tgt + self.dropout(ca_out))
            tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


# ─────────────────────────────────────────────────────────
# 4. 编码器堆叠
# ─────────────────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    """
    多层编码器堆叠（N 个 TransformerEncoderLayer）。

    Pre-Norm 模式下，在最后一层输出额外添加 LayerNorm（GPT-2 风格）。
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, pre_norm, activation)
            for _ in range(n_layers)
        ])
        # Pre-Norm 时在末尾添加最终归一化
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.final_norm(x)


# ─────────────────────────────────────────────────────────
# 5. 解码器堆叠
# ─────────────────────────────────────────────────────────
class TransformerDecoder(nn.Module):
    """
    多层解码器堆叠（N 个 TransformerDecoderLayer）。
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, pre_norm, activation)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask:  Optional[torch.Tensor] = None,
        mem_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return self.final_norm(tgt)


# ─────────────────────────────────────────────────────────
# 功能验证
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T_enc, T_dec, D, H = 2, 32, 16, 128, 4

    # FFN 变体测试
    for act in ["relu", "gelu", "swiglu"]:
        ffn = FeedForwardNetwork(D, activation=act)
        x = torch.randn(B, T_enc, D)
        out = ffn(x)
        print(f"[FFN-{act:6s}]  输入: {x.shape}  输出: {out.shape}")

    # 编码器
    for norm_type, pre_norm in [("Post-Norm", False), ("Pre-Norm", True)]:
        enc = TransformerEncoder(n_layers=3, d_model=D, n_heads=H, pre_norm=pre_norm)
        src = torch.randn(B, T_enc, D)
        out = enc(src)
        print(f"[Encoder-{norm_type}]  输入: {src.shape}  输出: {out.shape}")

    # 解码器
    dec = TransformerDecoder(n_layers=3, d_model=D, n_heads=H)
    memory = torch.randn(B, T_enc, D)
    tgt    = torch.randn(B, T_dec, D)
    out = dec(tgt, memory)
    print(f"[Decoder]           tgt: {tgt.shape}  memory: {memory.shape}  输出: {out.shape}")

    print("\n所有 Transformer 组件测试通过 ✓")
