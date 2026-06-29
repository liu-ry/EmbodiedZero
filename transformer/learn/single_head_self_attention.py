"""
单头自注意力 (Single-Head Self-Attention)
=======================================

本文件用 PyTorch 从零实现一个教学向的单头自注意力层：

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

输入 x 同时生成 Q、K、V，因此它是 self-attention。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadSelfAttention(nn.Module):
    """
    单头自注意力层。

    参数:
        d_model - 输入/输出特征维度
        d_k     - Q/K 的特征维度，默认等于 d_model
        d_v     - V 的特征维度，默认等于 d_model
        dropout - 注意力权重上的 dropout 概率
        bias    - 线性投影层是否使用偏置

    输入:
        x    - (batch_size, seq_len, d_model)
        mask - 可选掩码，True 表示保留，False 表示遮挡
               支持形状: (seq_len, seq_len)、(batch_size, seq_len)、
                       (batch_size, seq_len, seq_len)

    返回:
        output       - (batch_size, seq_len, d_model)
        attn_weights - (batch_size, seq_len, seq_len)
    """

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        self.w_q = nn.Linear(d_model, self.d_k, bias=bias)
        self.w_k = nn.Linear(d_model, self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, self.d_v, bias=bias)
        self.w_o = nn.Linear(self.d_v, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x)  # (B, T, d_k)
        k = self.w_k(x)  # (B, T, d_k)
        v = self.w_v(x)  # (B, T, d_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn_mask = self._build_attention_mask(
            mask=mask,
            causal=causal,
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        output = self.w_o(context)
        return output, attn_weights

    @staticmethod
    def _build_attention_mask(
        mask: Optional[torch.Tensor],
        causal: bool,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        attn_mask = None

        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            if mask.dim() == 2 and mask.shape == (seq_len, seq_len):
                attn_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif mask.dim() == 2 and mask.shape == (batch_size, seq_len):
                attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            elif mask.dim() == 3 and mask.shape == (batch_size, seq_len, seq_len):
                attn_mask = mask
            else:
                raise ValueError(
                    "mask 形状必须是 (T, T)、(B, T) 或 (B, T, T)，"
                    f"当前得到 {tuple(mask.shape)}"
                )

        if causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            ).unsqueeze(0)
            attn_mask = causal_mask if attn_mask is None else attn_mask & causal_mask

        return attn_mask


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    d_model = 8

    x = torch.randn(batch_size, seq_len, d_model)
    padding_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )

    attention = SingleHeadSelfAttention(d_model=d_model, dropout=0.1)
    output, attn_weights = attention(x, mask=padding_mask, causal=True)

    print("input shape:", x.shape)
    print("output shape:", output.shape)
    print("attention weights shape:", attn_weights.shape)
    print("first sample attention weights:")
    print(attn_weights[0])


if __name__ == "__main__":
    main()
