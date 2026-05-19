"""
位置编码 (Positional Encoding) 大全
=====================================
本模块实现了 Transformer 中常见的多种位置编码方式，并提供可视化接口：

1. SinusoidalPE        - 原版 《Attention Is All You Need》 的正弦/余弦编码（1D 序列）
2. LearnablePE         - 可学习的绝对位置嵌入（1D）
3. Learnable2DPE       - 可学习的 2D 位置嵌入（适用于 ViT patch 序列）
4. RotaryPE  (RoPE)    - 旋转位置编码，直接作用在 Q/K 上（相对位置编码）
5. ALiBiPE             - ALiBi 线性偏置位置编码（加在注意力分数上）

工作原理对比
-----------
| 编码类型   | 绝对/相对 | 可学习 | 作用位置     | 代表模型           |
|------------|-----------|--------|--------------|--------------------|
| Sinusoidal | 绝对      | 否     | 输入嵌入     | 原版 Transformer   |
| Learnable  | 绝对      | 是     | 输入嵌入     | BERT, GPT, ViT     |
| RoPE       | 相对      | 否     | Q/K 向量     | LLaMA, GPT-NeoX    |
| ALiBi      | 相对      | 否     | 注意力分数   | BLOOM              |
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. 正弦/余弦绝对位置编码 (Sinusoidal PE)
# ─────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    """
    《Attention Is All You Need》中的绝对正弦位置编码。

    编码公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    特点:
      - 固定，不可学习
      - 天然支持超出训练长度的序列（外推能力有限）
      - 位置差相关：PE(pos+k) 可由 PE(pos) 线性变换得到

    参数:
        d_model  - 嵌入维度
        max_len  - 预计算的最大序列长度
        dropout  - Dropout 概率
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构建 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 分母：10000^(2i/d_model) = exp(2i * ln10000/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度: cos

        # 注册为 buffer（不参与梯度计算，但随模型保存/加载）
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: 加上位置编码后的张量，shape 不变
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

    def visualize(self, seq_len: int = 100, save_path: str = None):
        """可视化位置编码矩阵热图"""
        pe = self.pe[0, :seq_len].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pe, aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Position")
        ax.set_title("Sinusoidal Positional Encoding")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()


# ─────────────────────────────────────────────
# 2. 可学习的 1D 绝对位置嵌入 (Learnable PE)
# ─────────────────────────────────────────────
class LearnablePE(nn.Module):
    """
    可学习的绝对位置嵌入，被 BERT、GPT、ViT 等模型广泛使用。

    每个位置对应一个独立的可训练向量，通过反向传播优化。

    特点:
      - 完全可学习，训练后能捕获任务特定的位置信息
      - 受限于 max_len，无法直接外推到更长序列

    参数:
        d_model  - 嵌入维度
        max_len  - 最大序列长度
        dropout  - Dropout 概率
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # nn.Embedding 的权重即为可学习的位置编码表
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        return self.dropout(x + self.pe(positions))


# ─────────────────────────────────────────────
# 3. 可学习的 2D 位置嵌入 (适用于 ViT)
# ─────────────────────────────────────────────
class Learnable2DPE(nn.Module):
    """
    ViT 风格的 2D 可学习位置嵌入。

    对图像被分割为 H×W 个 patch 的情况，分别学习行位置与列位置嵌入，
    并相加（而非拼接），以减少参数量。

    最终嵌入 = 输入 + row_embed[row_idx] + col_embed[col_idx]

    参数:
        d_model   - 嵌入维度（必须为偶数，行/列各占一半）
        grid_h    - patch 网格高度
        grid_w    - patch 网格宽度
        dropout   - Dropout 概率
    """

    def __init__(self, d_model: int, grid_h: int, grid_w: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model 必须为偶数"
        half_d = d_model // 2
        self.dropout = nn.Dropout(p=dropout)
        self.row_embed = nn.Embedding(grid_h, half_d)
        self.col_embed = nn.Embedding(grid_w, half_d)
        self.grid_h = grid_h
        self.grid_w = grid_w

        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, grid_h * grid_w, d_model)  [patch 序列，不含 CLS token]
        """
        rows = torch.arange(self.grid_h, device=x.device)   # (H,)
        cols = torch.arange(self.grid_w, device=x.device)   # (W,)
        # 广播拼接：每个 patch 位置 = (row, col) 的联合嵌入
        row_emb = self.row_embed(rows).unsqueeze(1).expand(-1, self.grid_w, -1)  # (H,W,half_d)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(self.grid_h, -1, -1)  # (H,W,half_d)
        pos = torch.cat([row_emb, col_emb], dim=-1)           # (H, W, d_model)
        pos = pos.view(-1, pos.size(-1))                       # (H*W, d_model)
        return self.dropout(x + pos.unsqueeze(0))


# ─────────────────────────────────────────────
# 4. 旋转位置编码 RoPE (Rotary Position Embedding)
# ─────────────────────────────────────────────
def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 的复数频率矩阵。

    RoPE 将 Q/K 视为复数向量，通过旋转操作编码位置信息：
        q_rot = q * e^(i * m * theta_j)
    其中 theta_j = 1 / (10000^(2j/dim))，m 为位置索引。

    返回形状: (max_len, dim//2) 的复数张量
    """
    assert dim % 2 == 0
    # 频率：theta_j = 1 / (base^(2j/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
    positions = torch.arange(max_len).float()                           # (max_len,)
    freqs = torch.outer(positions, freqs)                               # (max_len, dim/2)
    # 转为复数（极坐标形式）
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)             # (max_len, dim/2) complex
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    对 Q 或 K 施加旋转位置编码。

    参数:
        x         - (batch, seq_len, n_heads, head_dim)
        freqs_cis - (seq_len, head_dim//2) 复数频率张量

    返回: 与 x 形状相同的旋转后张量
    """
    # 将实数向量重塑为复数：(batch, seq, heads, dim/2) complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # 广播乘法：对每个 head 施加相同的位置旋转
    freqs_cis = freqs_cis[:x.size(1)].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim/2)
    x_rotated = x_complex * freqs_cis
    # 转回实数
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class RotaryPE(nn.Module):
    """
    RoPE 封装模块（LLaMA、GPT-NeoX 使用的相对位置编码）。

    优势:
      - 相对位置编码，内积仅取决于位置差 (m-n)
      - 无额外可学习参数
      - 良好的外推性
      - 与因果 Mask 兼容

    参数:
        head_dim  - 每个注意力头的维度
        max_len   - 预计算的最大序列长度
        theta     - 频率底数，默认 10000
    """

    def __init__(self, head_dim: int, max_len: int = 5000, theta: float = 10000.0):
        super().__init__()
        freqs_cis = precompute_rope_freqs(head_dim, max_len, theta)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (batch, seq_len, n_heads, head_dim)
        返回: 旋转后的 (q, k)
        """
        q_rot = apply_rope(q, self.freqs_cis)
        k_rot = apply_rope(k, self.freqs_cis)
        return q_rot, k_rot


# ─────────────────────────────────────────────
# 5. ALiBi 线性偏置位置编码
# ─────────────────────────────────────────────
class ALiBiPE(nn.Module):
    """
    ALiBi (Attention with Linear Biases) 位置编码。
    论文: "Train Short, Test Long: Attention with Linear Biases Enables Input
          Length Extrapolation" (Press et al., 2022)

    核心思想: 不修改 Q/K，而是在注意力矩阵上加一个与距离成正比的负偏置：
        Attention_score(i, j) = Q_i · K_j / sqrt(d) - m_h * |i - j|

    其中 m_h 是每个头特有的斜率（超参数，不可学习）。

    优势:
      - 极强的外推能力（训练短序列，测试超长序列）
      - 零额外参数

    参数:
        n_heads   - 注意力头数
        max_len   - 预计算的最大序列长度
    """

    def __init__(self, n_heads: int, max_len: int = 4096):
        super().__init__()
        slopes = self._get_slopes(n_heads)               # (n_heads,)
        # 距离矩阵: alibi[h, i, j] = -slope_h * |i - j|
        positions = torch.arange(max_len)
        # 下三角偏置（因果形式），上三角设为 -inf（可选）
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (max_len, max_len)
        # ALiBi 偏置: (n_heads, max_len, max_len)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * rel_dist.unsqueeze(0).float()
        self.register_buffer("alibi", alibi)

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        """
        计算每个头的斜率（ALiBi 论文中的设定）:
          m_h = 2^(-8h/n_heads)，h = 1, 2, ..., n_heads
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            # 非 2 的幂次：用最近的 2 的幂内插
            closest_pow2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_pow2)
            slopes += get_slopes_power_of_2(2 * closest_pow2)[0::2][: n - closest_pow2]
        return torch.tensor(slopes, dtype=torch.float)

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        在注意力分数上叠加 ALiBi 偏置。

        参数:
            attn_scores - (batch, n_heads, seq_len, seq_len)
        返回:
            带偏置的注意力分数，shape 不变
        """
        seq_len = attn_scores.size(-1)
        bias = self.alibi[:, :seq_len, :seq_len].unsqueeze(0)  # (1, n_heads, seq_len, seq_len)
        return attn_scores + bias


# ─────────────────────────────────────────────
# 快速功能验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    B, T, D, H = 2, 16, 128, 4
    head_dim = D // H
    x = torch.randn(B, T, D)

    # 1. SinusoidalPE
    sin_pe = SinusoidalPE(D, max_len=512)
    out = sin_pe(x)
    print(f"[SinusoidalPE]   输入: {x.shape}  输出: {out.shape}")

    # 2. LearnablePE
    learnable_pe = LearnablePE(D, max_len=512)
    out = learnable_pe(x)
    print(f"[LearnablePE]    输入: {x.shape}  输出: {out.shape}")

    # 3. Learnable2DPE (ViT 风格，6x6 patch grid)
    pe_2d = Learnable2DPE(D, grid_h=6, grid_w=6)
    x_2d = torch.randn(B, 36, D)
    out = pe_2d(x_2d)
    print(f"[Learnable2DPE]  输入: {x_2d.shape}  输出: {out.shape}")

    # 4. RoPE
    rope = RotaryPE(head_dim, max_len=512)
    q = torch.randn(B, T, H, head_dim)
    k = torch.randn(B, T, H, head_dim)
    q_r, k_r = rope(q, k)
    print(f"[RoPE]           Q: {q.shape}  Q_rot: {q_r.shape}")

    # 5. ALiBi
    alibi = ALiBiPE(n_heads=H, max_len=512)
    scores = torch.randn(B, H, T, T)
    biased = alibi(scores)
    print(f"[ALiBi]          分数: {scores.shape}  偏置后: {biased.shape}")

    print("\n所有位置编码测试通过 ✓")
