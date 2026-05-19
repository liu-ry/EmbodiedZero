# Attention Mechanism 详解：`compare_attention.py` 算法逻辑

> 本文档对 `compare_attention.py` 的完整算法逻辑进行讲解，涵盖所有 Attention 机制的原理、实现细节与性能对比。

---

## 整体架构

```
compare_attention.py
├── Mode 1: demo       ← 演示和讲解算法（教学模式）
├── Mode 2: visualize  ← 可视化注意力权重（热力图）
└── Mode 3: compare    ← 训练对比（性能评估）
```

运行方式：

```bash
# Mode 1: 教学演示
python compare_attention.py --mode demo

# Mode 2: 注意力热力图
python compare_attention.py --mode visualize

# Mode 3: 完整对比训练
python compare_attention.py --mode compare --models mha mqa gqa_4 linear window
```

---

## Mode 1：演示模式（`demo_attention()`）

逐步讲解 7 个核心算法概念。

### 1. Scaled Dot-Product Attention（基础注意力）

**算法步骤：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
scores = torch.matmul(q, k.transpose(-2, -1))  # Q·K^T: (B,H,T,T)
scaled = scores / math.sqrt(head_dim)           # 按√d_k缩放（防止softmax饱和）
attn_w = F.softmax(scaled, dim=-1)              # Softmax归一化 → [0,1]
out    = torch.matmul(attn_w, v)                # 加权和得到输出
```

**关键问题：为什么除以 $\sqrt{d_k}$？**

当维度很大时，$Q \cdot K^T$ 的方差会随 $d_k$ 增大，导致 softmax 进入梯度消失区域。
除以 $\sqrt{d_k}$ 后，点积方差恢复为 1，梯度更稳定。

**因果掩码（Causal Mask）：**

自回归任务中，未来位置的注意力分数被设为 $-\infty$，softmax 后权重变为 0，防止信息泄露：

```python
causal_mask = make_causal_mask(T, device)
scores_masked = scaled.masked_fill(~causal_mask, float("-inf"))
attn_causal   = F.softmax(scores_masked, dim=-1)
```

---

### 2. Multi-Head Attention（多头注意力）

**核心思想：** 在多个"子空间"中并行学习注意力模式。

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$

**代码逻辑：**

```python
mha = MultiHeadAttention(D, n_heads=H)
out_mha, attn_mha = mha(x, x, x)  # 返回 (B,T,D) 和权重 (B,H,T,T)

# 用信息熵衡量每个头的注意力集中程度
entropy = -(attn_mha * (attn_mha + 1e-9).log()).sum(-1).mean(-1)  # (B, H)
# 高熵 = 注意力分散（全局关注）
# 低熵 = 注意力集中（局部关注）
```

**为什么多头有效？**

- 不同头可以学习不同的语义关系（句法、语义、指代等）
- 增加模型容量而几乎不增加计算量

---

### 3. MHA vs MQA vs GQA：KV 缓存大小对比

这是**推理效率**的核心问题，KV 缓存直接影响 GPU 显存占用和吞吐量。

| 机制 | KV 头数 | KV 缓存大小 | 相对 MHA | 说明 |
|------|--------|------------|---------|------|
| **MHA** | H | $2T \times H \times d_k \times 4$ bytes | 1.0x | 标准方案 |
| **MQA** | 1 | $2T \times 1 \times d_k \times 4$ bytes | 0.125x | 极端压缩，质量有损 |
| **GQA (g=2)** | 2 | $2T \times 2 \times d_k \times 4$ bytes | 0.25x | 平衡方案 |
| **GQA (g=4)** | 4 | $2T \times 4 \times d_k \times 4$ bytes | 0.5x | 性能接近 MHA |

**核心算法区别：**

```python
# MQA: 所有 Query 头共享同一个 K, V
Q: (B, H, T, d_k)   # H 个查询头
K: (B, 1, T, d_k)   # 仅 1 个 Key
V: (B, 1, T, d_k)   # 仅 1 个 Value

# GQA: 分组共享（折中方案）
Q: (B, H, T, d_k)            # H=8 个查询头
K: (B, n_kv_heads, T, d_k)   # 分组，每组多个 Q 共享同一个 K/V
V: (B, n_kv_heads, T, d_k)
```

> LLaMA-70B 使用 GQA，推理速度提升约 25%，KV 缓存减少 8 倍。

---

### 4. Linear Attention vs Standard Attention：时间复杂度

| 方案 | 时间复杂度 | 核心思想 |
|------|-----------|---------|
| **Standard Attention** | $O(N^2 d)$ | 直接计算 $N \times N$ 注意力矩阵 |
| **Linear Attention** | $O(N d^2)$ | 用特征映射替代 softmax，利用结合律 |

**Linear Attention 推导：**

用 $\phi(\cdot)$ 替代 softmax，使点积可分解：

$$\text{out}_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) \cdot v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}$$

由于矩阵乘法满足结合律，可先计算 $\sum_j \phi(k_j) v_j^\top$（一次性完成），从而将复杂度从 $O(N^2)$ 降为 $O(N)$。

**实测性能对比（CPU）：**

```
Seq len=64:   MHA=0.8ms   Linear=0.6ms   加速 1.3x
Seq len=256:  MHA=2.1ms   Linear=1.1ms   加速 1.9x
Seq len=512:  MHA=5.2ms   Linear=1.8ms   加速 2.9x
```

> ⚠️ 注意：Linear Attention 是 softmax 的近似，在精度上略有损失（约 5~10%）。

---

### 5. Window Attention：局部感受野

**算法思想：** 限制每个位置只关注附近固定大小的窗口，大幅降低计算量。

```
标准注意力：每个位置 attend 全部 N 个位置  → O(N²)
窗口注意力：每个位置仅 attend 窗口内 w 个位置 → O(N·w)

示例：N=1024, w=64
计算量：1,048,576 → 65,536，减少 15 倍！
```

```python
T_win = 32
win_attn = WindowAttention(D, H, window_size=8)
out_win, attn_win = win_attn(x_win)
# 注意力矩阵 shape: (num_windows, H, w, w)
# 感受野：全局=32，窗口=8，计算节省 75%
```

**应用：** Swin Transformer（微软），用于图像分类/检测/分割

---

### 6. Cross-Attention：编码器-解码器通信

**应用场景：** 机器翻译、图像生成（Stable Diffusion）、多模态

$$\text{CrossAttn}(Q_{\text{dec}},\ K_{\text{enc}},\ V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^\top}{\sqrt{d_k}}\right) V_{\text{enc}}$$

```python
ca = CrossAttention(D, H)
dec_query  = torch.randn(B, 5, D)    # 解码器：待生成的 5 个 token
enc_output = torch.randn(B, 12, D)  # 编码器：12 个上下文 token

out_ca, attn_ca = ca(dec_query, enc_output)
# attn_ca shape: (B, H, 5, 12) → 每个 decoder token 对 12 个 encoder 位置的权重
```

**矩阵的物理含义：**

- `attn_ca[b, head, i, j]` = decoder 第 i 个 token 对 encoder 第 j 个位置的关注程度
- 每一行 softmax 求和为 1（概率分布）

---

### 7. Positional Encoding 对注意力的影响

Attention 本身是**置换不变**的（无位置感知），需要额外加入位置信息。

| 方案 | 公式/原理 | 优点 | 缺点 |
|------|----------|------|------|
| **SinusoidalPE** | $PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$ | 固定，无需训练，有一定外推能力 | 不可学习 |
| **LearnablePE** | 完全可训练的位置嵌入矩阵 | 自适应任务 | 长度外推差 |
| **RoPE** | 旋转变换作用于 Q, K | 相对位置特性强，LLaMA 标准 | 实现较复杂 |
| **ALiBi** | 直接在注意力分数上加线性距离惩罚 | 极强长度外推性 | 仅编码相对距离 |

**RoPE 关键性质（相对位置独立性）：**

```python
# 无论绝对位置在哪，只要相对距离相同，点积结果相同
dot(q_at_pos0, k_at_pos2) ≈ dot(q_at_pos3, k_at_pos5)
# 相对距离都是 2 → 点积值相似 ✓
```

---

## Mode 2：可视化模式（`visualize_attention()`）

生成 4 张 PNG 可视化图，保存到 `output/visualize/`：

### 图 1：`01_attention_heatmaps.png`
- 对比 MHA、MQA、GQA、Linear 在 4 个头上的权重热力图
- 热力图亮度 = 注意力权重强度
- 观察不同机制在同一输入下的关注模式差异

### 图 2：`02_mask_comparison.png`
| 掩码类型 | 效果 |
|---------|------|
| 无掩码 | 双向注意（BERT 风格） |
| 因果掩码 | 下三角形权重（GPT 风格） |
| 填充掩码 | 后半段权重归零 |

### 图 3：`03_positional_encoding.png`
- SinusoidalPE：规则的正弦波纹图案
- LearnablePE：随机初始化的噪声图案
- ALiBi：以对角线为轴的线性梯度图案

### 图 4：`04_window_vs_global.png`
- 左图：全局 MHA（所有位置均可关注）
- 右图：窗口 Attention（只有对角线块内有权重）
- 红色虚线：窗口边界

---

## Mode 3：对比训练模式（`compare_models()`）

在 **STL-10 数据集**上训练不同注意力机制的 ViT，系统比较各机制的性能。

### 模型架构：`CompactViT`

```
输入图像 (B, 3, 96, 96)
      ↓
PatchEmbedding       # 16×16 patch → (B, N, 256)
      ↓
CLS token concat     # (B, N+1, 256)
      ↓
Positional Encoding  # LearnablePE / SinusoidalPE / 2DPE
      ↓
× 4 TransformerLayer
│   ├── LayerNorm
│   ├── Attention    # MHA / MQA / GQA / Linear / Window
│   ├── Residual + Dropout
│   ├── LayerNorm
│   └── FFN (前馈网络) + Residual
      ↓
取 CLS token → LayerNorm → Linear(256, 10)
      ↓
分类输出 (B, 10)
```

**统一配置（公平对比）：**

```python
D = 256   # 统一嵌入维度
L = 4     # 统一 4 层
H = 4     # 统一 4 个头
```

### 训练流程

```python
optimizer = AdamW(lr=3e-4, weight_decay=0.05)
criterion = CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑防止过拟合
scheduler = OneCycleLR(...)                         # 余弦退火 + warmup

for epoch in range(epochs):
    # 训练阶段
    for imgs, labels in train_loader:
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step(); scheduler.step()

    # 验证阶段
    # 记录 val_loss, val_acc
```

**调度策略说明：**
- `warmup`：前 20% 步数线性升温，避免初期参数剧烈震荡
- `OneCycleLR`：升温后余弦退火，收敛更稳定

### 输出结果

训练完成后输出汇总表和可视化：

```
Model              Params(M)   Best Val Acc   Time(s)
──────────────────────────────────────────────────────
GQA g=4            6.80        78.5%          450
MHA (Multi-Head)   7.10        77.9%          480
Window (Swin)      6.50        77.2%          410
Linear Attn        6.90        76.2%          420
MQA (Multi-Query)  6.20        71.3%          350
```

**三个对比子图：**

1. **Val Accuracy 曲线** — 各模型收敛速度与最终精度
2. **Val Loss 曲线** — 学习稳定性对比
3. **散点气泡图** — 参数量 vs 精度（气泡大小 = 训练时间）

---

## 关键算法指标总结

| 机制 | 时间复杂度 | KV 缓存 | 适用场景 |
|------|-----------|--------|---------|
| **Standard MHA** | $O(N^2 d)$ | $O(N \cdot H)$ | 短序列，精度优先 |
| **MQA** | $O(N^2 d)$ | $O(N)$ | 推理速度优先（LLaMA） |
| **GQA** | $O(N^2 d)$ | $O(N \cdot g)$ | 训练推理平衡（LLaMA-2/3） |
| **Linear Attn** | $O(N d^2)$ | $O(d^2)$ | 长序列，速度优先 |
| **Window Attn** | $O(N w d)$ | $O(N w)$ | 图像任务（Swin-ViT） |
| **Cross Attn** | $O(N_q N_k d)$ | — | 编码器-解码器，多模态 |

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models (GQA)](https://arxiv.org/abs/2305.13245)
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- [Swin Transformer (Window Attention)](https://arxiv.org/abs/2103.14030)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: ALiBi](https://arxiv.org/abs/2108.12409)
