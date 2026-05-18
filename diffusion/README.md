# Diffusion Models 学习系列

本目录是扩散模型的系统性学习仓库，各子目录共用 `model/` 中的基础组件，
仅在各自特有的采样/架构部分有所不同。

---

## 目录结构

```
dp/
├── model/                       # ★ 所有变体共用的基础组件
│   ├── model.py                  # SimpleUNet（噪声/向量场预测骨干网络）
│   ├── noise_schedule.py         # β-schedule + 前向加噪 q_sample + 训练损失 p_losses
│   ├── vae.py                    # 轻量级 VAE（Stable Diffusion 隐空间压缩）
│   ├── flow_schedule.py          # OT-CFM 插值 + 向量场训练损失（Flow Matching 专用）
│   └── __init__.py
│
├── ddpm/                         # ✅ DDPM（Denoising Diffusion Probabilistic Models）
│   └── sampler.py                # 祖先采样器（每步加随机噪声，全部 T 步）
│
├── ddim/                         # ✅ DDIM（Denoising Diffusion Implicit Models）
│   └── sampler.py                # 确定性跳步采样器（50 步即可）
│
├── stable_diffusion/             # ✅ Latent Diffusion（Stable Diffusion 简化版）
│   └── sampler.py                # 隐空间 DDIM 采样 + VAE 解码
│
├── flow_matching/                # ✅ Flow Matching（OT-CFM）
│   └── sampler.py                # Euler/Heun ODE 积分采样器
│
├── run_ddpm.py                   # DDPM 训练脚本
├── run_ddim.py                   # DDIM 训练脚本
├── run_stable_diffusion.py       # Stable Diffusion 两阶段训练脚本
├── run_flow_matching.py          # Flow Matching 训练脚本
├── README.md
└── requirements.txt
```

---

## model/ 模块说明

### `model/model.py` — SimpleUNet

所有扩散变体共用的骨干网络，DDPM/DDIM 用它预测噪声 $\varepsilon_\theta$，
Flow Matching 用它预测向量场 $v_\theta$，Stable Diffusion 在隐空间上使用它：

```
输入: (B, C, H, W) 中间状态 + (B,) 整数时间步 t
  → Sinusoidal 时间步嵌入 → MLP
  → 编码器（ResBlock + 下采样）→ 瓶颈 → 解码器（ResBlock + 上采样 + skip）
输出: (B, C, H, W) 预测值（噪声 ε 或向量场 v）
```

### `model/noise_schedule.py` — NoiseSchedule

DDPM / DDIM / Stable Diffusion 共用的扩散前向过程：

| 方法 | 公式 | 用途 |
|------|------|------|
| `q_sample(x0, t)` | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$ | 训练时加噪 |
| `p_losses(model, x0)` | $\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$ | 训练损失 |

支持两种 β 调度：`linear`（原始）、`cosine`（Nichol & Dhariwal 2021）

### `model/vae.py` — VAE（新增）

轻量级卷积 VAE，用于 Stable Diffusion 的隐空间压缩：

```
像素空间 (B, 1, 28, 28) ↔ 隐空间 (B, 4, 7, 7)
```

| 方法 | 说明 |
|------|------|
| `forward(x)` | 编码→重参数化→解码，返回 (重构图, 损失) |
| `encode_to_latent(x)` | 推理用：返回均值 μ，无随机性 |
| `decode(z)` | 隐变量 → 像素图像 |

### `model/flow_schedule.py` — FlowSchedule（新增）

Flow Matching 的训练工具类（OT-CFM 最优传输条件流匹配）：

| 方法 | 公式 | 用途 |
|------|------|------|
| `q_sample(x1, t)` | $x_t = (1-t)x_0 + tx_1$ | 直线插值生成中间状态 |
| `p_losses(model, x1)` | $\|v_\theta(x_t,t) - (x_1-x_0)\|^2$ | 向量场训练损失 |

---

## 各变体的核心区别

| 变体 | 时间域 | 训练目标 | 反向采样 | 推理步数 | 特色 |
|------|--------|---------|---------|---------|------|
| **DDPM** | 离散 $t \in \{0..T\}$ | 预测噪声 $\varepsilon_\theta$ | 随机祖先采样 | T（慢，通常 1000） | 基础版，随机性强 |
| **DDIM** | 离散（跳步） | 预测噪声 $\varepsilon_\theta$（同 DDPM）| 确定性跳步 | 50~100 步 | 复用 DDPM 权重，快速推理 |
| **Stable Diffusion** | 离散（隐空间） | VAE + 隐空间 DDPM | 隐空间 DDIM + VAE 解码 | 50 步 | 隐空间压缩，可扩展到条件生成 |
| **Flow Matching** | 连续 $t \in [0,1]$ | 预测向量场 $v_\theta$ | ODE 积分（Euler/Heun）| 100 步（可更少）| 数学简洁，训练稳定 |

---

## 快速开始

```bash
pip install -r requirements.txt

# 训练 DDPM（像素空间，祖先采样）
python run_ddpm.py --epochs 20

# 训练 DDIM（复用 DDPM 训练目标，50 步推理）
python run_ddim.py --epochs 20 --ddim-steps 50

# 训练 Stable Diffusion（两阶段：VAE + 隐空间扩散）
python run_stable_diffusion.py --vae-epochs 20 --ldm-epochs 20

# 快速测试 SD（各 2 epoch）
python run_stable_diffusion.py --vae-epochs 2 --ldm-epochs 2

# 跳过 VAE 训练，复用已有权重
python run_stable_diffusion.py --vae-epochs 0 --vae-ckpt results_sd/vae_best.pt

# 训练 Flow Matching（OT-CFM，Euler 100 步 ODE）
python run_flow_matching.py --epochs 20

# 使用 Heun 方法（50 步达到更高精度）
python run_flow_matching.py --epochs 20 --ode-steps 50 --ode-method heun
```

---

## 数学速查

### DDPM 前向过程

$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)I)$$

### DDIM 单步更新

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\varepsilon_\theta + \sigma_t z$$

### VAE 损失

$$\mathcal{L}_\text{VAE} = \underbrace{\|x - \hat{x}\|^2}_{\text{重构}} + \beta \cdot \underbrace{D_\text{KL}(q(z|x) \| \mathcal{N}(0,I))}_{\text{正则化}}$$

### Flow Matching（OT-CFM）

条件流（直线插值）：

$$x_t = (1-t)\,x_0 + t\,x_1, \quad x_0\sim\mathcal{N}(0,I),\; x_1\sim p_\text{data}$$

训练目标：

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

推理（Euler ODE）：

$$x_{t+h} = x_t + h\cdot v_\theta(x_t, t), \quad t: 0 \to 1$$

