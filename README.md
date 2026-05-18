# EmbodiedZero

<p align="right">
  <a href="README.md"><b>中文</b></a> | <a href="README_EN.md">English</a>
</p>

> **从零开始学习具身大模型基础知识的实践仓库**

具身大模型（Embodied Large Model）是当前机器人与 AI 领域的前沿方向，涵盖感知、生成、策略学习等多个核心模块。
本仓库以「最小可运行、原理透明」为原则，提供一系列从零实现的基础算法，帮助初学者系统性地打好理论与工程基础。

---

## 🎯 适合谁？

- 想入门具身智能 / 机器人学习，但不知从哪里开始的同学
- 对扩散模型、VAE、Flow Matching 等生成模型有兴趣，希望读懂每一行代码
- 希望在学习扩散策略（Diffusion Policy）、VLA 等具身算法之前，先夯实数学与实现基础

---

## 🚀 快速开始

```bash
git clone https://github.com/liu-ry/EmbodiedZero.git
cd EmbodiedZero

# 例：运行经典 VAE（MNIST，~2 分钟）
cd vae && pip install -r requirements.txt && python run_vae.py

# 例：运行 DDPM（MNIST，~5 分钟）
cd dp && pip install -r requirements.txt && python run_ddpm.py --epochs 10

# 例：运行 Flow Matching
cd dp && python run_flow_matching.py --epochs 10
```

---

## 设计原则

- **最小依赖**：仅使用 PyTorch + torchvision，无多余框架封装
- **代码即文档**：每个模块都有详细的中文注释和数学公式说明
- **循序渐进**：如学习扩散模型，从 DDPM → DDIM → Stable Diffusion → Flow Matching，每一步都有前序铺垫
- **可复现**：所有脚本默认下载公开数据集，一行命令即可运行

---

## 持续更新中
- 欢迎提交Issues、PR
- 欢迎交流