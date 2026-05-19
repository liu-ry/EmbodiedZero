"""
注意力机制 (Attention Mechanisms) 大全
=========================================
本模块从零实现 Transformer 中常见的各种注意力机制：

1. ScaledDotProductAttention  - 缩放点积注意力（基础）
2. MultiHeadAttention         - 多头注意力（原版 Transformer）
3. MultiQueryAttention        - 多查询注意力 (MQA)，K/V 共享
4. GroupedQueryAttention      - 分组查询注意力 (GQA)，LLaMA-2/3 使用
5. LinearAttention            - 线性注意力（近似，O(N) 复杂度）
6. WindowAttention            - 窗口注意力（Swin Transformer 思路）
7. CrossAttention             - 交叉注意力（Encoder-Decoder 通信）

注意力复杂度对比
----------------
| 类型             | 时间复杂度 | KV 缓存大小        | 代表模型          |
|------------------|------------|--------------------|-------------------|
| MultiHead        | O(N²d)     | O(N * n_heads)     | BERT, GPT         |
| MultiQuery       | O(N²d)     | O(N * 1)           | PaLM, Falcon      |
| GroupedQuery     | O(N²d)     | O(N * n_groups)    | LLaMA-2/3         |
| Linear           | O(Nd²)     | -                  | Performer, cosFormer|
| Window (Swin)    | O(N*w²d)   | -                  | Swin Transformer  |
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import RotaryPE, ALiBiPE


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────
def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """生成下三角因果掩码 (1 = 保留, 0 = 遮挡)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask  # (seq_len, seq_len)


# ─────────────────────────────────────────────────────────
# 1. 缩放点积注意力 (Scaled Dot-Product Attention)
# ─────────────────────────────────────────────────────────
class ScaledDotProductAttention(nn.Module):
    """
    原始论文中的缩放点积注意力：

        Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    其中 √d_k 是缩放因子，防止维度较大时内积数值过大导致梯度消失。

    参数:
        dropout - 注意力权重上的 Dropout 概率
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,   # (..., seq_q, d_k)
        k: torch.Tensor,   # (..., seq_k, d_k)
        v: torch.Tensor,   # (..., seq_k, d_v)
        mask: Optional[torch.Tensor] = None,  # (..., seq_q, seq_k) bool, True=保留
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (context, attn_weights)
          context     - (..., seq_q, d_v)
          attn_weights- (..., seq_q, seq_k)
        """
        d_k = q.size(-1)
        # (1) 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq_q, seq_k)

        # (2) 应用掩码（将被遮挡位置填为 -inf）
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # (3) Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)

        # (4) 对 nan 安全处理（全遮挡行会出现 nan）
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # (5) Dropout
        attn_weights = self.dropout(attn_weights)

        # (6) 加权聚合 V
        context = torch.matmul(attn_weights, v)  # (..., seq_q, d_v)
        return context, attn_weights


# ─────────────────────────────────────────────────────────
# 2. 多头注意力 (Multi-Head Attention, MHA)
# ─────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    标准多头注意力（原版 Transformer / BERT / GPT）。

    核心思想: 将 Q/K/V 分别投影到 h 个低维空间，并行执行注意力，
    然后拼接并投影回原始维度。这让模型从多个子空间捕获不同的关系。

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)

    参数:
        d_model   - 总嵌入维度
        n_heads   - 注意力头数（d_model 必须整除 n_heads）
        dropout   - 注意力 Dropout
        bias      - 是否在投影线性层添加偏置
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V 投影（合并为单个线性层以提升效率）
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, n_heads, T, head_dim)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, n_heads, T, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, T, head_dim) -> (B, T, d_model)"""
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,   # (B, T_q, d_model)
        key:   torch.Tensor,   # (B, T_k, d_model)
        value: torch.Tensor,   # (B, T_k, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (output, attn_weights)
          output      - (B, T_q, d_model)
          attn_weights- (B, n_heads, T_q, T_k)
        """
        q = self._split_heads(self.w_q(query))  # (B, H, T_q, hd)
        k = self._split_heads(self.w_k(key))    # (B, H, T_k, hd)
        v = self._split_heads(self.w_v(value))  # (B, H, T_k, hd)

        if mask is not None and mask.dim() == 2:
            # (T_q, T_k) -> (1, 1, T_q, T_k) 广播
            mask = mask.unsqueeze(0).unsqueeze(0)

        context, attn_w = self.attn(q, k, v, mask=mask)  # (B, H, T_q, hd)
        output = self.w_o(self._merge_heads(context))     # (B, T_q, d_model)
        return output, attn_w


# ─────────────────────────────────────────────────────────
# 3. 多查询注意力 (Multi-Query Attention, MQA)
# ─────────────────────────────────────────────────────────
class MultiQueryAttention(nn.Module):
    """
    多查询注意力 (MQA)。
    论文: "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)

    核心思想: Q 仍有 n_heads 个头，但 K/V 只有 1 个头（所有 Q 头共享同一 K/V）。
    大幅降低 KV 缓存显存占用，推理速度更快，精度略有下降。

    被 PaLM、Falcon、Gemini 等大模型采用。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)            # Q: 多头
        self.w_k = nn.Linear(d_model, self.head_dim)      # K: 单头
        self.w_v = nn.Linear(d_model, self.head_dim)      # V: 单头
        self.w_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        # Q: (B, n_heads, T_q, head_dim)
        q = self.w_q(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        # K/V: (B, 1, T_k, head_dim) 然后广播到所有头
        k = self.w_k(key).view(B, T_k, 1, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(B, T_k, 1, self.head_dim).transpose(1, 2)
        # 广播 K/V 到 n_heads
        k = k.expand(-1, self.n_heads, -1, -1)
        v = v.expand(-1, self.n_heads, -1, -1)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        context, attn_w = self.attn(q, k, v, mask=mask)
        output = self.w_o(context.transpose(1, 2).contiguous().view(B, T_q, self.d_model))
        return output, attn_w


# ─────────────────────────────────────────────────────────
# 4. 分组查询注意力 (Grouped-Query Attention, GQA)
# ─────────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (GQA)。
    论文: "GQA: Training Generalized Multi-Query Transformer Models
          from Multi-Head Checkpoints" (Ainslie et al., 2023)

    核心思想: MHA 和 MQA 的折衷方案：
      - 将 n_heads 个 Q 分成 n_kv_heads 组
      - 每组共享一个 K/V 头
      - n_kv_heads == 1  => MQA
      - n_kv_heads == n_heads => MHA

    被 LLaMA-2/3、Mistral 等主流 LLM 广泛采用。

    参数:
        n_kv_heads - KV 头数量（必须整除 n_heads）
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads 必须整除 n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # 每个 KV 头对应的 Q 头数
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout=dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 KV 头重复 n_rep 次以匹配 Q 头数。
        (B, n_kv_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        """
        B, n_kv, T, hd = x.shape
        if self.n_rep == 1:
            return x
        x = x.unsqueeze(2).expand(B, n_kv, self.n_rep, T, hd)
        return x.reshape(B, n_kv * self.n_rep, T, hd)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        q = self.w_q(query).view(B, T_q, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(B,   T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(B,  T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 重复 KV 以匹配 Q 头数
        k = self._repeat_kv(k)  # (B, n_heads, T_k, head_dim)
        v = self._repeat_kv(v)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        context, attn_w = self.attn(q, k, v, mask=mask)
        output = self.w_o(context.transpose(1, 2).contiguous().view(B, T_q, self.d_model))
        return output, attn_w


# ─────────────────────────────────────────────────────────
# 5. 线性注意力 (Linear Attention)
# ─────────────────────────────────────────────────────────
class LinearAttention(nn.Module):
    """
    线性注意力，将 O(N²) 的标准注意力近似为 O(N)。
    参考: "Transformers are RNNs: Fast Autoregressive Transformers
          with Linear Attention" (Katharopoulos et al., 2020)

    核心技巧: 利用核函数 φ 将 Softmax 注意力分解：
        Attention(Q,K,V) ≈ φ(Q) · (φ(K)ᵀ · V) / (φ(Q) · φ(K)ᵀ · 1)

    由于矩阵乘法结合律，先算 φ(K)ᵀ · V（d×d 矩阵），再乘 φ(Q)，
    复杂度从 O(N²d) 降至 O(Nd²)。

    本实现使用 ELU + 1 作为特征函数 φ（保证非负性）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.eps = eps

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """核函数 φ(x) = ELU(x) + 1（保证非负）"""
        return F.elu(x) + 1

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        def reshape(x, T):
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(reshape(self.w_q(query), T_q))  # (B, H, T_q, hd)
        k = self._feature_map(reshape(self.w_k(key),   T_k))  # (B, H, T_k, hd)
        v = reshape(self.w_v(value), T_k)                      # (B, H, T_k, hd)

        # 线性近似: KV = φ(K)ᵀ · V  (B, H, hd, hd)
        kv = torch.matmul(k.transpose(-2, -1), v)

        # 归一化分母: φ(K)ᵀ · 1  (B, H, hd)
        k_sum = k.sum(dim=-2)  # (B, H, hd)

        # 计算输出: φ(Q) · KV / (φ(Q) · k_sum)
        numerator   = torch.matmul(q, kv)                          # (B, H, T_q, hd)
        denominator = (q * k_sum.unsqueeze(-2)).sum(-1, keepdim=True) + self.eps  # (B, H, T_q, 1)
        context = numerator / denominator

        output = self.w_o(context.transpose(1, 2).contiguous().view(B, T_q, self.d_model))
        return output, None  # 线性注意力没有显式的注意力权重矩阵


# ─────────────────────────────────────────────────────────
# 6. 窗口注意力 (Window Attention, Swin-style)
# ─────────────────────────────────────────────────────────
class WindowAttention(nn.Module):
    """
    局部窗口注意力（Swin Transformer 核心思想）。
    论文: "Swin Transformer: Hierarchical Vision Transformer
          using Shifted Windows" (Liu et al., 2021)

    核心思想: 将序列分割为互不重叠的局部窗口，在每个窗口内部独立执行注意力。
      - 将全局 O(N²) 复杂度降至 O(N * w²)，w 为窗口大小
      - 通过移位窗口（shifted window）实现跨窗口的信息交流

    本实现支持:
      - 基础窗口注意力（无移位）
      - 相对位置偏置（Relative Position Bias）

    参数:
        d_model     - 嵌入维度
        n_heads     - 注意力头数
        window_size - 窗口大小 w（序列被分割为 N//w 个窗口）
        dropout     - Dropout 概率
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 可学习的相对位置偏置表，覆盖 [-(w-1), w-1] 范围内的所有相对距离
        # 偏置表大小: (2*w-1, n_heads)
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), n_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # 预计算相对位置索引
        coords = torch.arange(window_size)
        rel_coords = coords.unsqueeze(0) - coords.unsqueeze(1)          # (w, w)
        rel_coords += window_size - 1                                    # 移位到 [0, 2w-2]
        self.register_buffer("rel_pos_index", rel_coords)               # (w, w)

    def forward(
        self,
        x: torch.Tensor,   # (B, T, d_model)，T 必须整除 window_size
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        w = self.window_size
        assert T % w == 0, f"序列长度 {T} 必须整除窗口大小 {w}"
        n_windows = T // w

        # 分窗口: (B * n_windows, w, d_model)
        x_win = x.view(B, n_windows, w, self.d_model).reshape(B * n_windows, w, self.d_model)

        def project_and_split(linear, t):
            return linear(t).view(B * n_windows, w, self.n_heads, self.head_dim).transpose(1, 2)

        q = project_and_split(self.w_q, x_win)  # (B*nw, H, w, hd)
        k = project_and_split(self.w_k, x_win)
        v = project_and_split(self.w_v, x_win)

        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*nw, H, w, w)

        # 添加相对位置偏置: (w, w, H) -> (1, H, w, w)
        rel_bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]  # (w*w, H)
        rel_bias = rel_bias.view(w, w, self.n_heads).permute(2, 0, 1).unsqueeze(0)
        scores = scores + rel_bias

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.dropout(attn_w)
        context = torch.matmul(attn_w, v)  # (B*nw, H, w, hd)

        # 恢复形状
        context = context.transpose(1, 2).contiguous().view(B * n_windows, w, self.d_model)
        output  = self.w_o(context).view(B, T, self.d_model)

        return output, attn_w


# ─────────────────────────────────────────────────────────
# 7. 交叉注意力 (Cross-Attention)
# ─────────────────────────────────────────────────────────
class CrossAttention(nn.Module):
    """
    交叉注意力（Encoder-Decoder Attention）。

    Q 来自解码器，K/V 来自编码器的输出，使解码器能够关注编码器表示的不同部分。
    结构与多头注意力完全相同，仅在使用时传入不同的 query / key / value。

    参数同 MultiHeadAttention。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # 复用 MultiHeadAttention，只是语义上强调 Q ≠ K = V
        self.mha = MultiHeadAttention(d_model, n_heads, dropout=dropout)

    def forward(
        self,
        query:   torch.Tensor,   # (B, T_dec, d_model) 来自解码器
        enc_out: torch.Tensor,   # (B, T_enc, d_model) 来自编码器
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (output, attn_weights)
          output - (B, T_dec, d_model)
        """
        return self.mha(query, enc_out, enc_out, mask=src_mask)


# ─────────────────────────────────────────────────────────
# 8. 带 RoPE 的多头注意力
# ─────────────────────────────────────────────────────────
class RoPEMultiHeadAttention(nn.Module):
    """
    集成旋转位置编码 (RoPE) 的多头注意力。

    RoPE 直接作用在 Q/K 的每个头上，无需在输入嵌入处添加位置编码。
    被 LLaMA 系列模型采用。
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPE(self.head_dim, max_len=max_len)
        self.attn = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        q = self.w_q(query).view(B, T_q, self.n_heads, self.head_dim)
        k = self.w_k(key).view(B,   T_k, self.n_heads, self.head_dim)
        v = self.w_v(value).view(B,  T_k, self.n_heads, self.head_dim)

        # 施加 RoPE（作用在 Q/K 上）
        q, k = self.rope(q, k)

        # 转换为 (B, H, T, hd) 用于注意力计算
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        context, attn_w = self.attn(q, k, v, mask=mask)
        output = self.w_o(context.transpose(1, 2).contiguous().view(B, T_q, self.d_model))
        return output, attn_w


# ─────────────────────────────────────────────────────────
# 功能验证
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, D, H = 2, 32, 256, 8

    x = torch.randn(B, T, D)
    causal_mask = make_causal_mask(T, x.device)

    tests = {
        "MHA":     MultiHeadAttention(D, H),
        "MQA":     MultiQueryAttention(D, H),
        "GQA(4)":  GroupedQueryAttention(D, H, n_kv_heads=4),
        "Linear":  LinearAttention(D, H),
        "Window":  WindowAttention(D, H, window_size=8),
        "Cross":   CrossAttention(D, H),
        "RoPE-MHA":RoPEMultiHeadAttention(D, H),
    }

    for name, module in tests.items():
        module.eval()
        with torch.no_grad():
            if name == "Cross":
                enc = torch.randn(B, 16, D)
                out, _ = module(x, enc)
            elif name == "Window":
                out, _ = module(x)
            else:
                out, _ = module(x, x, x)
        print(f"[{name:12s}]  输入: {x.shape}  输出: {out.shape}")

    print("\n所有注意力机制测试通过 ✓")
