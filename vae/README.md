# VAE 学习系列

本目录包含两个从零实现的变分自编码器示例，从最基础的全连接 VAE 到支持高分辨率图像的 VQ-VAE，
帮助理解生成模型中隐变量表示与离散编码的核心思想。

---

## 目录结构

```
vae/
├── run_vae.py          # ✅ 全连接 VAE（MNIST，连续隐变量）
├── run_vq_vae.py       # ✅ VQ-VAE（STL-10 或自定义图片，离散码本）
├── test_vq_vae.py      # VQ-VAE 推理测试脚本
├── requirements.txt
└── README.md
```

---

## 模型说明

### `run_vae.py` — 经典 VAE（全连接，MNIST）

基于论文 [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)（Kingma & Welling, 2014）的改进实现。  
网络结构为全连接层，使用 ReLU 激活和 Adam 优化器（相比原文的 sigmoid + adagrad，收敛更快）。

**网络结构：**
```
输入: (B, 784)  [MNIST 28×28 展平]
  → fc1(784→400, ReLU)
  → fc_μ(400→20) / fc_logvar(400→20)   ← 编码为均值与对数方差
  → 重参数化 z = μ + ε·σ
  → fc3(20→400, ReLU) → fc4(400→784, Sigmoid)
输出: 重建图像 + (μ, logvar)
```

**训练损失：**

$$\mathcal{L}_\text{VAE} = \underbrace{\text{BCE}(x, \hat{x})}_{\text{重构损失}} + \underbrace{D_\text{KL}(q(z|x) \| \mathcal{N}(0,I))}_{\text{KL 正则化}}$$

其中 $D_\text{KL} = -\tfrac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$

---

### `run_vq_vae.py` — VQ-VAE（卷积，高分辨率图像）

向量量化变分自编码器，使用 EMA（指数滑动平均）更新离散码本，支持多分辨率图像。

**网络结构：**
```
输入: (B, 3, H, W)
  → Encoder（多层 stride-2 Conv + ResidualStack）→ (B, num_hiddens, H/k, W/k)
  → pre_vq_conv(1×1) → (B, embedding_dim, H/k, W/k)
  → VectorQuantizerEMA：最近邻查找 → 离散 token → EMA 更新码本
  → Decoder（对称 ConvTranspose + ResidualStack）
输出: 重建图像 + VQ 损失 + codebook 困惑度
```

**下采样倍数 `--downsample`：**

| 值 | stride-2 层数 | 适用分辨率 |
|----|--------------|-----------|
| 4  | 2 层         | 96×96（STL-10）|
| 8  | 3 层         | 256×256   |
| 16 | 4 层         | 512×512+  |

**训练损失：**

$$\mathcal{L}_\text{VQ} = \underbrace{\|x - \hat{x}\|^2 / \sigma^2_\text{data}}_{\text{归一化重构}} + \underbrace{\beta \|z_e - \text{sg}[z_q]\|^2}_{\text{commitment loss}}$$

码本通过 EMA 更新，无需梯度直通技巧。

---

## 快速开始

```bash
pip install -r requirements.txt
```

### 训练 VAE（MNIST）

```bash
# 默认参数（10 epoch，latent dim=20）
python run_vae.py

# 自定义参数
python run_vae.py --epochs 20 --batch-size 64 --seed 42

# 强制使用 CPU
python run_vae.py --no-accel
```

`run_vae.py` 支持的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 128 | 训练批大小 |
| `--epochs` | 10 | 训练轮数 |
| `--no-accel` | — | 禁用加速器，强制 CPU |
| `--seed` | 1 | 随机种子 |
| `--log-interval` | 10 | 每多少 batch 打印一次日志 |

每个 epoch 结束后，结果保存到 `results/`：
- `reconstruction_<epoch>.png`：原图与重建对比
- `sample_<epoch>.png`：从先验 $\mathcal{N}(0,I)$ 随机采样生成的图像

---

### 训练 VQ-VAE（STL-10 / 自定义图片）

```bash
# STL-10（需先手动下载解压到 data/stl10_binary/）
python run_vq_vae.py --dataset stl10 --image-size 96 --downsample 4 --save-results

# 自定义图片目录（256×256）
python run_vq_vae.py --dataset custom --image-dir data/train-images --image-size 256 --downsample 8 --save-results

# 大图（512×512）
python run_vq_vae.py --dataset custom --image-dir data/train-images \
    --image-size 512 --downsample 16 --batch-size 16 --num-hiddens 256
```

`run_vq_vae.py` 常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `stl10` | `stl10` 或 `custom` |
| `--image-dir` | — | custom 模式下的图片目录 |
| `--image-size` | 96 | 训练图像尺寸 |
| `--downsample` | 4 | 编码器下采样倍数（4/8/16）|
| `--num-hiddens` | 128 | 隐层通道数 |
| `--embedding-dim` | 64 | 码本向量维度 |
| `--num-embeddings` | 512 | 码本大小 |
| `--num-epochs` | 500 | 训练轮数 |
| `--batch-size` | 64 | 训练批大小 |
| `--learning-rate` | 2e-4 | Adam 学习率 |
| `--save-results` | — | 保存训练曲线图 |
| `--checkpoint` | `results/vqvae_large_model.pth` | 模型保存路径 |

每隔 `--save-interval`（默认 5）个 epoch，结果保存到 `results/`：
- `large_reconstruction_epoch<N>.png`：原图（上）与重建（下）对比
- `large_training_curve.png`：重构误差 + codebook 困惑度曲线（需 `--save-results`）

---

### VQ-VAE 推理测试

```bash
python test_vq_vae.py \
    --image-dir data/test-image \
    --checkpoint results/vqvae_large_model.pth \
    --output-dir results/test_output
```

---

## 核心概念对比

| 特性 | VAE | VQ-VAE |
|------|-----|--------|
| 隐变量类型 | 连续（高斯） | 离散（码本 token） |
| 正则化方式 | KL 散度 | commitment loss + EMA |
| 网络结构 | 全连接 | 卷积（ResNet 风格）|
| 典型应用 | MNIST 生成/插值 | 图像压缩/生成前置编码器 |
| 与扩散模型关系 | Stable Diffusion 的 VAE 基础 | DALL-E / VQ-Diffusion 的码本基础 |

