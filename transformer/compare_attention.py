"""
Attention Mechanism Comparison
================================
Three modes:

  Mode 1: demo      - Print tensor shapes and characteristics for each attention type
  Mode 2: visualize - Visualize attention weight heatmaps (random weights)
  Mode 3: compare   - Train multiple models on STL-10 and plot comparison curves

Usage:
  # Mode 1: step-by-step demo (start here)
  python compare_attention.py --mode demo

  # Mode 2: attention heatmap visualization
  python compare_attention.py --mode visualize

  # Mode 3: full comparison training
  python compare_attention.py --mode compare --models mha mqa gqa_4 linear window
  python compare_attention.py --mode compare --models mha mqa          # quick compare
  python compare_attention.py --mode compare --epochs 30 --batch_size 128
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # headless environment - save to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from model.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    LinearAttention,
    WindowAttention,
    CrossAttention,
    RoPEMultiHeadAttention,
    make_causal_mask,
)
from model.positional_encoding import SinusoidalPE, LearnablePE, RotaryPE, ALiBiPE
from model.vit import build_vit, VisionTransformer
from utils.data_loader import build_dataloaders


# ═══════════════════════════════════════════════════════════════
# Console color helpers
# ═══════════════════════════════════════════════════════════════
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def title(s):  print(f"\n{C.BOLD}{C.HEADER}{'═'*60}\n  {s}\n{'═'*60}{C.RESET}")
def section(s):print(f"\n{C.BOLD}{C.CYAN}── {s}{C.RESET}")
def info(s):   print(f"  {C.GREEN}▶{C.RESET} {s}")
def warn(s):   print(f"  {C.YELLOW}⚠{C.RESET}  {s}")
def num(label, val): print(f"  {C.BLUE}{label:<30}{C.RESET} {C.BOLD}{val}{C.RESET}")


# ═══════════════════════════════════════════════════════════════
# Mode 1: Step-by-step demo
# ═══════════════════════════════════════════════════════════════
def demo_attention():
    """
    Print key tensor shapes and characteristics for each attention mechanism.
    """
    B, T, D, H = 1, 8, 64, 4     # batch=1, seq=8, dim=64, heads=4
    head_dim = D // H

    x   = torch.randn(B, T, D)
    enc = torch.randn(B, 12, D)   # encoder output for Cross Attention

    title("Attention Mechanism Step-by-Step Demo")

    # ── 1. Scaled Dot-Product Attention ───────────────────────
    section("1. Scaled Dot-Product Attention")
    info("Formula: softmax(Q·Kt / sqrt(d_k)) · V")

    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)
    v = torch.randn(B, H, T, head_dim)

    # manually demonstrate computation steps
    scores = torch.matmul(q, k.transpose(-2, -1))  # raw scores
    scaled = scores / math.sqrt(head_dim)           # scale
    attn_w = F.softmax(scaled, dim=-1)              # normalize
    out    = torch.matmul(attn_w, v)                # weighted sum

    num("Q shape",            str(q.shape))
    num("K shape",            str(k.shape))
    num("Raw scores Q·Kt",    str(scores.shape))
    num(f"Scaled /sqrt({head_dim})", f"range [{scaled.min():.2f}, {scaled.max():.2f}]  (before: [{scores.min():.2f}, {scores.max():.2f}])")
    num("Attn weights (softmax)", f"{attn_w.shape}  row_sum={attn_w.sum(-1).mean():.4f}")
    num("Output context",        str(out.shape))

    # causal mask effect
    causal_mask = make_causal_mask(T, x.device)
    scores_masked = scaled.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_causal   = F.softmax(scores_masked, dim=-1)
    info(f"After causal mask: pos 0 can only attend to itself (row0 sum={attn_causal[0,0,0].sum():.4f})")

    # ── 2. Multi-Head Attention ────────────────────────────────
    section("2. Multi-Head Attention (MHA) -- why multiple heads?")
    info("Multiple heads = learn attention patterns in h different subspaces in parallel")

    mha = MultiHeadAttention(D, n_heads=H)
    out_mha, attn_mha = mha(x, x, x)

    num("Input x",          str(x.shape))
    num("Head dim",         f"D/H = {D}/{H} = {head_dim}")
    num("Q/K/V after proj", f"{H} groups x {T}x{head_dim}")
    num("Attn weights shape", str(attn_mha.shape))
    num("Output shape",       str(out_mha.shape))

    # print per-head entropy (higher=more spread, lower=more focused)
    entropy = -(attn_mha * (attn_mha + 1e-9).log()).sum(-1).mean(-1)  # (B, H)
    for h in range(H):
        info(f"  Head {h} avg entropy: {entropy[0, h].item():.3f}  "
             f"({'spread' if entropy[0,h]>1.5 else 'focused'})")

    # ── 3. MHA vs MQA vs GQA: KV cache size ──────────────────
    section("3. MHA vs MQA vs GQA -- KV cache size comparison")
    info("KV cache size directly affects GPU memory and inference throughput")

    configs = [
        ("MHA",      MultiHeadAttention(D, H),                  H),
        ("MQA",      MultiQueryAttention(D, H),                 1),
        ("GQA(g=2)", GroupedQueryAttention(D, H, n_kv_heads=2), 2),
        ("GQA(g=4)", GroupedQueryAttention(D, H, n_kv_heads=4), 4),
    ]

    print(f"\n  {'Model':<12} {'KV heads':<10} {'KV cache/token':<22} {'vs MHA':<12} {'Output OK'}")
    print(f"  {'─'*70}")
    for name, module, n_kv in configs:
        kv_size     = 2 * T * n_kv * head_dim * 4  # bytes (float32)
        kv_mha_size = 2 * T * H   * head_dim * 4
        ratio       = n_kv / H
        out_x, _ = module(x, x, x)
        ok = "✓" if out_x.shape == x.shape else "✗"
        print(f"  {name:<12} {n_kv:<8} {kv_size/1024:.2f} KB{'':<12} {ratio:.2f}x{'':<6} {ok}")

    # ── 4. Linear Attention vs Standard: complexity ───────────
    section("4. Linear Attention vs MHA -- complexity comparison")
    info("Standard: O(N^2 d)  |  Linear: O(N d^2)  -- faster for long sequences")

    seq_lengths = [16, 64, 256, 512]
    mha_linear = MultiHeadAttention(D, H)
    lin_attn   = LinearAttention(D, H)

    print(f"\n  {'Seq len':<10} {'MHA (ms)':<16} {'Linear (ms)':<18} {'Speedup'}")
    print(f"  {'─'*55}")
    for T_test in seq_lengths:
        x_test = torch.randn(B, T_test, D)

        # warmup
        for _ in range(3):
            mha_linear(x_test, x_test, x_test)
            lin_attn(x_test, x_test, x_test)

        # timing
        t0 = time.perf_counter()
        for _ in range(50): mha_linear(x_test, x_test, x_test)
        t_mha = (time.perf_counter() - t0) / 50 * 1000

        t0 = time.perf_counter()
        for _ in range(50): lin_attn(x_test, x_test, x_test)
        t_lin = (time.perf_counter() - t0) / 50 * 1000

        speedup = t_mha / t_lin
        print(f"  {T_test:<10} {t_mha:<16.3f} {t_lin:<18.3f} {speedup:.2f}x")

    warn("Note: Linear attention is an approximation of Softmax, slight accuracy loss")

    # ── 5. Window Attention: receptive field ──────────────────
    section("5. Window Attention -- local receptive field")
    info("Global attention: each position attends to all N positions")
    info("Window attention: each position attends only to w positions in its window")

    T_win = 32
    w     = 8
    x_win = torch.randn(B, T_win, D)
    win_attn = WindowAttention(D, H, window_size=w)
    out_win, attn_win = win_attn(x_win)

    num("Sequence length N",  T_win)
    num("Window size w",      w)
    num("Num windows",        T_win // w)
    num("Attn matrix shape",  str(attn_win.shape))
    num("Receptive field",    f"global={T_win}  window={w}  compute saved={(1 - w/T_win)*100:.0f}%")

    # ── 6. Cross-Attention ────────────────────────────────────
    section("6. Cross-Attention -- Encoder-Decoder communication")
    info("Q from decoder (tokens to generate), K/V from encoder (context)")

    ca = CrossAttention(D, H)
    dec_query = torch.randn(B, 5, D)    # decoder: generate 5 tokens
    enc_output = torch.randn(B, 12, D)  # encoder: 12 patches/tokens

    out_ca, attn_ca = ca(dec_query, enc_output)
    num("Decoder Q shape",  str(dec_query.shape))
    num("Encoder K/V shape", str(enc_output.shape))
    num("Attn shape",       str(attn_ca.shape))
    num("Output shape",     str(out_ca.shape))
    info("Per decoder position: top-3 attended encoder positions:")
    for i in range(min(3, dec_query.size(1))):
        top_k = attn_ca[0, 0, i].topk(3)
        info(f"  dec pos {i} -> top enc positions: {top_k.indices.tolist()}  "
             f"weights: {[f'{v:.3f}' for v in top_k.values.tolist()]}")

    # ── 7. Effect of positional encoding on attention ─────────
    section("7. Positional Encoding -- effect on attention")
    info("Comparison of positional encoding types:")

    pe_info = [
        ("SinusoidalPE", "fixed, non-learnable, periodic, limited extrapolation"),
        ("LearnablePE",  "fully learnable, task-adaptive, bounded by max_len"),
        ("RoPE",         "applied to Q/K, relative positions, used in LLaMA/GPT-NeoX"),
        ("ALiBi",        "linear distance penalty on scores, strong length extrapolation"),
    ]
    for name, desc in pe_info:
        info(f"  {name:<16}: {desc}")

    # demonstrate RoPE relative position property
    rope = RotaryPE(head_dim, max_len=64)
    q1 = torch.randn(1, 1, 1, head_dim)
    k1 = torch.randn(1, 1, 1, head_dim)

    # pos 0 vs pos 2 (distance=2)
    q_at_0 = q1.expand(1, 8, 1, head_dim).clone()
    k_at_2 = k1.expand(1, 8, 1, head_dim).clone()
    q_r, k_r = rope(q_at_0, k_at_2)
    # pos 3 vs pos 5 (distance=2, same relative position)
    q_at_3 = q1.expand(1, 8, 1, head_dim).clone()
    k_at_5 = k1.expand(1, 8, 1, head_dim).clone()
    q_r2, k_r2 = rope(q_at_3, k_at_5)

    dot_02 = (q_r[:, 0] * k_r[:, 2]).sum().item()
    dot_35 = (q_r2[:, 3] * k_r2[:, 5]).sum().item()
    info(f"RoPE relative property: dot(pos0,pos2)={dot_02:.4f}  dot(pos3,pos5)={dot_35:.4f}  "
         f"{'similar ✓' if abs(dot_02 - dot_35) < 0.01 else 'different'}")

    title("Demo done! Run --mode visualize to see attention heatmaps")


# ═══════════════════════════════════════════════════════════════
# Mode 2: Attention weight visualization
# ═══════════════════════════════════════════════════════════════
def visualize_attention(save_dir: str = "output/visualize"):
    """
    Generate attention weight heatmaps and save as PNG files.
    """
    os.makedirs(save_dir, exist_ok=True)
    B, T, D, H = 1, 16, 64, 4
    head_dim = D // H
    x = torch.randn(B, T, D)

    title("Attention Weight Visualization")
    info(f"Images will be saved to: {save_dir}/")

    # ── Fig1: Heatmap comparison across attention types ───────
    attention_modules = {
        "MHA\n(Multi-Head)":      MultiHeadAttention(D, H),
        "MQA\n(Multi-Query)":     MultiQueryAttention(D, H),
        "GQA(g=2)\n(Grouped-Query)": GroupedQueryAttention(D, H, n_kv_heads=2),
        "Linear\n(Approx.)":      LinearAttention(D, H),
    }

    fig, axes = plt.subplots(H, len(attention_modules), figsize=(16, 12))
    fig.suptitle("Attention Weight Heatmaps (4 Heads)", fontsize=14, fontweight="bold")

    for col, (name, module) in enumerate(attention_modules.items()):
        module.eval()
        with torch.no_grad():
            _, attn_w = module(x, x, x)
        if attn_w is None:
            attn_w = torch.ones(B, H, T, T) / T  # Linear attention has no explicit weights

        for h in range(H):
            ax = axes[h, col]
            w = attn_w[0, h].detach().numpy()
            im = ax.imshow(w, cmap="Blues", vmin=0, aspect="auto")
            if h == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Head {h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.colorbar(im, ax=axes, shrink=0.6, label="Attention Weight")
    plt.tight_layout()
    path1 = os.path.join(save_dir, "01_attention_heatmaps.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    info(f"Saved: {path1}")

    # ── Fig2: Causal mask vs no mask ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Effect of Masking on Attention Weights", fontsize=13)

    mha = MultiHeadAttention(D, H)
    mha.eval()
    with torch.no_grad():
        # no mask
        _, w_full = mha(x, x, x)
        # causal mask
        causal = make_causal_mask(T, x.device)
        _, w_causal = mha(x, x, x, mask=causal)
        # padding mask (simulate padding)
        pad_mask = torch.ones(T, T, dtype=torch.bool)
        pad_mask[:, T//2:] = False  # second half masked out
        _, w_pad = mha(x, x, x, mask=pad_mask)

    titles  = ["No mask (bidirectional)", "Causal mask (autoregressive)", "Padding mask (2nd half=0)"]
    weights = [w_full[0,0], w_causal[0,0], w_pad[0,0]]

    for ax, t, w in zip(axes, titles, weights):
        im = ax.imshow(w.detach().numpy(), cmap="Reds", vmin=0, aspect="auto")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "02_mask_comparison.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    info(f"Saved: {path2}")

    # ── Fig3: Positional encoding visualization ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("Positional Encoding Visualization", fontsize=13)

    # SinusoidalPE
    from model.positional_encoding import SinusoidalPE, LearnablePE, ALiBiPE
    sin_pe = SinusoidalPE(d_model=128, max_len=64)
    pe_mat = sin_pe.pe[0, :64].detach().numpy()
    im = axes[0].imshow(pe_mat, aspect="auto", cmap="RdBu_r")
    axes[0].set_title("SinusoidalPE\n(fixed, sin/cos)", fontsize=10)
    axes[0].set_xlabel("Dimension"); axes[0].set_ylabel("Position")
    plt.colorbar(im, ax=axes[0])

    # Learnable PE (untrained, near-random)
    learn_pe = LearnablePE(d_model=128, max_len=64)
    lpe_mat = learn_pe.pe.weight.detach().numpy()
    im2 = axes[1].imshow(lpe_mat, aspect="auto", cmap="RdBu_r")
    axes[1].set_title("LearnablePE\n(trainable, after init)", fontsize=10)
    axes[1].set_xlabel("Dimension"); axes[1].set_ylabel("Position")
    plt.colorbar(im2, ax=axes[1])

    # ALiBi bias matrix
    alibi = ALiBiPE(n_heads=4, max_len=64)
    bias_mat = alibi.alibi[0, :64, :64].detach().numpy()
    im3 = axes[2].imshow(bias_mat, aspect="auto", cmap="RdBu_r")
    axes[2].set_title("ALiBi bias\n(head 0, linear distance penalty)", fontsize=10)
    axes[2].set_xlabel("Key position"); axes[2].set_ylabel("Query position")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    path3 = os.path.join(save_dir, "03_positional_encoding.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    info(f"Saved: {path3}")

    # ── Fig4: Window attention vs global attention ────────────
    T_big = 32
    x_big = torch.randn(B, T_big, D)
    win_attn = WindowAttention(D, H, window_size=8)
    mha_big  = MultiHeadAttention(D, H)

    win_attn.eval(); mha_big.eval()
    with torch.no_grad():
        _, w_win = win_attn(x_big)
        _, w_global = mha_big(x_big, x_big, x_big)

    # fill window attention into full N×N matrix (zeros outside windows)
    full_win = torch.zeros(T_big, T_big)
    win_size = 8
    for wi in range(T_big // win_size):
        s, e = wi * win_size, (wi + 1) * win_size
        full_win[s:e, s:e] = w_win[wi * B, 0]   # take first head

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Global Attention vs Window Attention (receptive field)", fontsize=13)

    axes[0].imshow(w_global[0, 0].detach().numpy(), cmap="Blues", aspect="auto")
    axes[0].set_title(f"Global MHA\nreceptive field = all {T_big} positions")
    axes[0].set_xlabel("Key"); axes[0].set_ylabel("Query")

    axes[1].imshow(full_win.numpy(), cmap="Blues", aspect="auto")
    axes[1].set_title(f"Window Attention (w=8)\nreceptive field = 8 positions, {T_big//8} windows")
    axes[1].set_xlabel("Key"); axes[1].set_ylabel("Query")

    # draw window boundaries
    for ax in axes:
        for boundary in range(0, T_big + 1, win_size):
            ax.axhline(boundary - 0.5, color="red", linewidth=1.5, alpha=0.6)
            ax.axvline(boundary - 0.5, color="red", linewidth=1.5, alpha=0.6)

    plt.tight_layout()
    path4 = os.path.join(save_dir, "04_window_vs_global.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    info(f"Saved: {path4}")

    title(f"Visualization done! 4 images saved to {save_dir}/")


# ═══════════════════════════════════════════════════════════════
# Mode 3: Multi-model comparison training
# ═══════════════════════════════════════════════════════════════

# attention type -> vit model name mapping
ATTN_TO_VIT = {
    "mha":     "vit_small",    # standard multi-head attention ViT
    "mqa":     "vit_mqa",      # multi-query attention ViT
    "gqa_2":   "vit_gqa2",
    "gqa_4":   "vit_gqa4",
    "linear":  "vit_linear",
    "window":  "swin_small",   # window attention uses Swin
    "sinpe":   "vit_sinpe",    # test different positional encodings
    "2dpe":    "vit_2dpe",
    "tiny":    "vit_tiny",
}

# display names
MODEL_DISPLAY = {
    "mha":    "MHA (Multi-Head)",
    "mqa":    "MQA (Multi-Query)",
    "gqa_2":  "GQA g=2",
    "gqa_4":  "GQA g=4",
    "linear": "Linear Attn",
    "window": "Window(Swin)",
    "sinpe":  "ViT+SinePE",
    "2dpe":   "ViT+2DPE",
    "tiny":   "ViT-Tiny",
}


def build_model_for_comparison(attn_type: str, img_size: int = 96, n_classes: int = 10,
                                 dropout: float = 0.1) -> nn.Module:
    """Build comparison model (unified d_model=256, n_layers=4 for fair comparison)"""
    from model.vit import VisionTransformer, SwinViT, PatchEmbedding
    from model.transformer_components import TransformerEncoder, FeedForwardNetwork
    from model.attention import (MultiHeadAttention, MultiQueryAttention,
                                GroupedQueryAttention, LinearAttention)
    from model.positional_encoding import LearnablePE, SinusoidalPE, Learnable2DPE

    D, L, H = 256, 4, 4   # unified config: dim=256, layers=4, heads=4

    # custom encoder layer to support different attention types
    class CustomEncoderLayer(nn.Module):
        def __init__(self, attn_module):
            super().__init__()
            self.attn = attn_module
            self.ffn  = FeedForwardNetwork(D, dropout=dropout)
            self.norm1 = nn.LayerNorm(D)
            self.norm2 = nn.LayerNorm(D)
            self.drop  = nn.Dropout(dropout)
            self.is_window = isinstance(attn_module, WindowAttention)

        def forward(self, x):
            nx = self.norm1(x)
            if self.is_window:
                attn_out, _ = self.attn(nx)
            else:
                attn_out, _ = self.attn(nx, nx, nx)
            x = x + self.drop(attn_out)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x

    class CompactViT(nn.Module):
        def __init__(self, attn_type):
            super().__init__()
            from model.vit import PatchEmbedding
            self.patch_embed = PatchEmbedding(img_size, 16, 3, D)
            n_patches = self.patch_embed.n_patches

            # CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            # positional encoding
            if attn_type == "sinpe":
                self.pos_enc = SinusoidalPE(D, max_len=n_patches + 1, dropout=dropout)
            elif attn_type == "2dpe":
                gs = self.patch_embed.grid_size
                self.pos_enc = Learnable2DPE(D, gs, gs, dropout=dropout)
                self.cls_drop = nn.Dropout(dropout)
            else:
                self.pos_enc = LearnablePE(D, max_len=n_patches + 1, dropout=dropout)
            self.attn_type = attn_type

            # attention module
            def make_attn():
                if attn_type in ("mha", "sinpe", "2dpe"):
                    return MultiHeadAttention(D, H, dropout=dropout)
                elif attn_type == "mqa":
                    return MultiQueryAttention(D, H, dropout=dropout)
                elif attn_type == "gqa_2":
                    return GroupedQueryAttention(D, H, n_kv_heads=2, dropout=dropout)
                elif attn_type == "gqa_4":
                    return GroupedQueryAttention(D, H, n_kv_heads=4, dropout=dropout)
                elif attn_type == "linear":
                    return LinearAttention(D, H, dropout=dropout)
                elif attn_type == "window":
                    n_patches_val = self.patch_embed.n_patches
                    w = 6
                    while n_patches_val % w != 0:
                        w -= 1
                    return WindowAttention(D, H, window_size=w, dropout=dropout)
                else:
                    return MultiHeadAttention(D, H, dropout=dropout)

            self.layers = nn.ModuleList([CustomEncoderLayer(make_attn()) for _ in range(L)])
            self.norm   = nn.LayerNorm(D)
            self.head   = nn.Linear(D, n_classes)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x):
            B = x.size(0)
            tokens = self.patch_embed(x)
            cls    = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            if self.attn_type == "2dpe":
                tokens = torch.cat([
                    self.cls_drop(tokens[:, :1]),
                    self.pos_enc(tokens[:, 1:])
                ], dim=1)
            else:
                tokens = self.pos_enc(tokens)

            for layer in self.layers:
                tokens = layer(tokens)

            feat   = self.norm(tokens[:, 0])   # CLS token
            return self.head(feat)

    return CompactViT(attn_type)


def train_model(
    model:       nn.Module,
    train_loader,
    val_loader,
    epochs:      int,
    lr:          float,
    device:      torch.device,
    model_name:  str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train and return (train_losses, val_losses, train_accs, val_accs)"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    total_steps = epochs * len(train_loader)
    warmup      = max(1, epochs // 5) * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=warmup/total_steps, anneal_strategy="cos"
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tl += loss.item() * imgs.size(0)
            tc += logits.max(1)[1].eq(labels).sum().item()
            tt += labels.size(0)

        # validate
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                vl += criterion(logits, labels).item() * imgs.size(0)
                vc += logits.max(1)[1].eq(labels).sum().item()
                vt += labels.size(0)

        train_losses.append(tl / tt)
        train_accs.append(100.0 * tc / tt)
        val_losses.append(vl / vt)
        val_accs.append(100.0 * vc / vt)

        print(f"  [{model_name}] Epoch {epoch:3d}/{epochs}  "
              f"Train: loss={train_losses[-1]:.3f} acc={train_accs[-1]:.1f}%  "
              f"Val: loss={val_losses[-1]:.3f} acc={val_accs[-1]:.1f}%")

    return train_losses, val_losses, train_accs, val_accs


def compare_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, "compare")
    os.makedirs(save_dir, exist_ok=True)

    title(f"Multi-model Comparison Training  [{', '.join(args.models)}]")
    info(f"Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")

    # data
    train_loader, val_loader = build_dataloaders(
        args.data_root, img_size=96,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # train each model in sequence
    results: Dict[str, dict] = {}
    for attn_name in args.models:
        display = MODEL_DISPLAY.get(attn_name, attn_name)
        section(f"Training model: {display} ({attn_name})")

        model = build_model_for_comparison(attn_name, img_size=96, n_classes=10)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        num("Params", f"{n_params:.2f}M")

        t0 = time.time()
        tl, vl, ta, va = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr,
            device=device, model_name=display
        )
        elapsed = time.time() - t0

        results[attn_name] = {
            "display": display, "train_loss": tl, "val_loss": vl,
            "train_acc": ta, "val_acc": va,
            "best_val_acc": max(va), "n_params": n_params, "time": elapsed,
        }
        info(f"Done! Best Val Acc: {max(va):.2f}%  Time: {elapsed:.1f}s")

    # ── plot ─────────────────────────────────────────────────
    epochs = list(range(1, args.epochs + 1))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Attention Mechanism Comparison (STL-10)", fontsize=14, fontweight="bold")

    # subplot 1: val accuracy curves
    for (name, r), c in zip(results.items(), colors):
        axes[0].plot(epochs, r["val_acc"], color=c, linewidth=2,
                     label=f"{r['display']} ({r['best_val_acc']:.1f}%)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Acc (%)")
    axes[0].set_title("Val Accuracy Comparison")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # subplot 2: val loss curves
    for (name, r), c in zip(results.items(), colors):
        axes[1].plot(epochs, r["val_loss"], color=c, linewidth=2, label=r["display"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Val Loss Comparison")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # subplot 3: accuracy vs params bubble chart
    names    = [r["display"]       for r in results.values()]
    accs     = [r["best_val_acc"]  for r in results.values()]
    params   = [r["n_params"]      for r in results.values()]
    times    = [r["time"]          for r in results.values()]

    sc = axes[2].scatter(params, accs, c=colors[:len(results)],
                         s=[t * 20 for t in times], alpha=0.8, edgecolors="black")
    for n, p, a in zip(names, params, accs):
        axes[2].annotate(n, (p, a), textcoords="offset points",
                         xytext=(5, 3), fontsize=8)
    axes[2].set_xlabel("Params (M)"); axes[2].set_ylabel("Best Val Acc (%)")
    axes[2].set_title("Accuracy vs Params\n(bubble size = training time)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "attention_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── print summary table ───────────────────────────────────
    title("Comparison Summary")
    print(f"\n  {'Model':<18} {'Params(M)':<10} {'Best Val Acc':<14} {'Time(s)':<14}")
    print(f"  {'─'*58}")
    for r in sorted(results.values(), key=lambda x: -x["best_val_acc"]):
        print(f"  {r['display']:<18} {r['n_params']:<10.2f} "
              f"{r['best_val_acc']:<14.2f} {r['time']:<14.1f}")

    info(f"Comparison plot saved: {plot_path}")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Attention Mechanism Comparison")

    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "visualize", "compare"],
                        help="mode: demo | visualize | compare")

    # compare mode args
    parser.add_argument("--models", nargs="+",
                        default=["mha", "mqa", "gqa_4"],
                        choices=list(ATTN_TO_VIT.keys()),
                        help="attention models to compare")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--data_root",  type=str,   default="../vae/data/stl10_binary")
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--save_dir",   type=str,   default="output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "demo":
        demo_attention()
    elif args.mode == "visualize":
        visualize_attention(save_dir=os.path.join(args.save_dir, "visualize"))
    elif args.mode == "compare":
        compare_models(args)
