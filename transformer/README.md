# Transformer 知识点全览 —— STL-10 图像分类

> 本目录以 **STL-10** 数据集（96×96 RGB，10 类）为实验平台，
> 从零实现并演示 Transformer 的核心知识点。

---

## 📁 文件结构

```
transformer/
├── data_loader.py            # STL-10 二进制数据加载
├── positional_encoding.py    # 多种位置编码
├── attention.py              # 多种注意力机制
├── transformer_components.py # FFN、Encoder/Decoder 层
├── vit.py                    # Vision Transformer 模型
├── train.py                  # 训练与验证主流程
└── README.md                 # 本文档
```

---

## 🗂️ 知识点导航

### 1. 数据加载 (`data_loader.py`)

STL-10 二进制格式解析：

- 每张图像 = **27648 字节**，通道优先排列 `(C=3, H=96, W=96)`
- 标签文件：每字节一个标签，值域 `1-10`（代码自动转为 `0-9`）
- 提供训练集数据增强（水平翻转、随机裁剪、颜色抖动）与测试集归一化

---

### 2. 位置编码 (`positional_encoding.py`)

| 类名 | 类型 | 可学习 | 作用位置 | 代表模型 |
|------|------|--------|----------|----------|
| `SinusoidalPE` | 绝对 | ✗ | 输入嵌入 | 原版 Transformer |
| `LearnablePE` | 绝对 | ✓ | 输入嵌入 | BERT / GPT / ViT |
| `Learnable2DPE` | 绝对 2D | ✓ | patch 嵌入 | ViT（行+列分离） |
| `RotaryPE (RoPE)` | 相对 | ✗ | Q/K 向量 | LLaMA / GPT-NeoX |
| `ALiBiPE` | 相对 | ✗ | 注意力分数 | BLOOM |

**正弦位置编码公式：**

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

**RoPE 旋转公式：**

$$\mathbf{q}_m' = \mathbf{q}_m \cdot e^{im\theta_j}$$

其中旋转只依赖位置差 $m - n$，天然实现相对位置感知。

---

### 3. 注意力机制 (`attention.py`)

#### 3.1 缩放点积注意力（基础）

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $\sqrt{d_k}$ 缩放防止内积过大导致梯度消失
- 支持任意形状的掩码（填充掩码 / 因果掩码）

#### 3.2 多头注意力 (MHA)

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)$$

- 并行捕获不同子空间的关系
- 复杂度 $O(N^2 d)$，KV 缓存大小 $O(N \cdot h)$

#### 3.3 多查询注意力 (MQA)

- Q 有 $h$ 个头，K/V 只有 **1 个头**（所有 Q 头共享）
- KV 缓存降至 $O(N)$，推理速度提升显著
- 被 PaLM、Falcon 等大模型采用

#### 3.4 分组查询注意力 (GQA)

- MHA 与 MQA 的折衷：K/V 有 $g$ 个头（$1 \le g \le h$）
- 每组 $h/g$ 个 Q 头共享一组 K/V
- $g=1$ ⟹ MQA；$g=h$ ⟹ MHA
- LLaMA-2/3、Mistral 的标准配置

#### 3.5 线性注意力

利用核函数 $\phi$ 分解 Softmax：

$$\text{Attention}(Q,K,V) \approx \frac{\phi(Q)\big(\phi(K)^\top V\big)}{\phi(Q)\big(\phi(K)^\top \mathbf{1}\big)}$$

先算 $\phi(K)^\top V$（$d \times d$ 矩阵）再乘 $\phi(Q)$，
复杂度从 $O(N^2 d)$ 降至 $O(N d^2)$。

#### 3.6 窗口注意力 (Swin)

- 将序列切成大小为 $w$ 的**局部窗口**，窗口内独立注意力
- 复杂度 $O(N w^2)$，相对位置偏置表增强局部感知
- 通过**移位窗口 (SW-MSA)** 实现跨窗口信息交流

#### 3.7 交叉注意力

- Q 来自解码器，K/V 来自编码器输出
- 使解码器每个位置能够关注编码器序列的不同部分

---

### 4. Transformer 组件 (`transformer_components.py`)

#### 前馈网络 (FFN) 变体

| 变体 | 公式 | 代表模型 |
|------|------|----------|
| ReLU FFN | $\max(0,\, xW_1)W_2$ | 原版 Transformer |
| GELU FFN | $\text{GELU}(xW_1)W_2$ | BERT / GPT-2 |
| SwiGLU | $(\text{SiLU}(xW_1) \odot xW_3)W_2$ | LLaMA / PaLM |

#### 归一化位置

**Post-Norm（原版）：**
$$x = \text{LN}\big(x + \text{Sublayer}(x)\big)$$

**Pre-Norm（现代，更稳定）：**
$$x = x + \text{Sublayer}\big(\text{LN}(x)\big)$$

---

### 5. Vision Transformer (`vit.py`)

#### 标准 ViT 流程

```
图像 (B,3,96,96)
    ↓ PatchEmbedding (patch=16 → 6×6=36 个 token)
(B, 36, d_model)
    ↓ 拼接 CLS token → (B, 37, d_model)
    ↓ + 位置编码
    ↓ L × TransformerEncoderLayer
    ↓ 取 CLS token → LayerNorm → Linear
logits (B, 10)
```

#### 可选模型

| 名称 | 维度 | 层数 | 头数 | 位置编码 | 参数量 |
|------|------|------|------|----------|--------|
| `vit_tiny` | 192 | 4 | 3 | 可学习 1D | ~4M |
| `vit_small` | 384 | 6 | 6 | 可学习 1D | ~22M |
| `vit_base` | 512 | 8 | 8 | 可学习 1D | ~48M |
| `vit_sinpe` | 384 | 6 | 6 | 正弦 1D | ~22M |
| `vit_2dpe` | 384 | 6 | 6 | 可学习 2D | ~22M |
| `swin_small` | 384 | 4 | 6 | 可学习 1D + 窗口偏置 | ~15M |

---

### 6. 训练技巧 (`train.py`)

| 技巧 | 说明 |
|------|------|
| **AdamW 优化器** | Transformer 标配，$\beta=(0.9, 0.95)$，weight decay=0.05 |
| **余弦退火 + Warmup** | 前 5 epoch 线性预热，之后余弦衰减至 5% |
| **Label Smoothing** | 软标签防止过拟合，默认 $\epsilon=0.1$ |
| **Mixup** | $\tilde{x}=\lambda x_i+(1-\lambda)x_j$，混合训练对 |
| **梯度裁剪** | `clip_grad_norm=1.0`，防止梯度爆炸 |
| **Checkpoint** | 保存最优模型，支持断点续训 |

---

## 🚀 快速开始

```bash
# 安装依赖
pip install torch torchvision pillow tensorboard

# 快速验证各模块（无需 GPU）
python data_loader.py
python positional_encoding.py
python attention.py
python transformer_components.py
python vit.py

# 训练（默认 vit_small，50 epochs）
cd transformer
python train.py

# 指定模型与参数
python train.py --model vit_tiny --epochs 30 --batch_size 128 --lr 5e-4

# 对比各个attention区别
python compare_attention.py --mode visualize

# 查看 TensorBoard
tensorboard --logdir output/
```

---

## 📊 知识点关系图

```
原始图像
    │
    ▼ PatchEmbedding（卷积切块）
Token 序列
    │
    ▼ + 位置编码
    │   ├─ SinusoidalPE（固定，正弦）
    │   ├─ LearnablePE（可学习，1D）
    │   ├─ Learnable2DPE（可学习，2D）
    │   ├─ RoPE（作用在 Q/K，相对）
    │   └─ ALiBi（加在注意力分数，相对）
    │
    ▼ × L 层 TransformerEncoderLayer
    │   ├─ 注意力（Pre-Norm → Attn → 残差）
    │   │   ├─ MultiHeadAttention（全局）
    │   │   ├─ MultiQueryAttention（KV 共享）
    │   │   ├─ GroupedQueryAttention（分组共享）
    │   │   ├─ LinearAttention（O(N)）
    │   │   └─ WindowAttention（局部窗口）
    │   └─ FFN（Pre-Norm → FFN → 残差）
    │       ├─ ReLU / GELU
    │       └─ SwiGLU（门控）
    │
    ▼ CLS token → LayerNorm → Linear
分类输出 (10类)
```
