# EmbodiedZero

<p align="right">
  <a href="README.md">中文</a> | <a href="README_EN.md"><b>English</b></a>
</p>

> **A hands-on repository for learning the fundamentals of Embodied Large Models from scratch**

Embodied Large Models sit at the frontier of robotics and AI, spanning perception, generation, and policy learning.
This repository follows a "minimal yet runnable, transparent in principle" philosophy — providing from-scratch implementations of foundational algorithms to help beginners systematically build both theoretical understanding and engineering skills.

---

## 🎯 Who is this for?

- Beginners who want to enter the field of embodied intelligence / robot learning but don't know where to start
- Anyone interested in generative models (diffusion, VAE, Flow Matching) who wants to understand every line of code
- Engineers looking to solidify their math and implementation foundations before tackling Diffusion Policy, VLA, and other embodied algorithms

---

## 🚀 Quick Start

```bash
git clone https://github.com/liu-ry/EmbodiedZero.git
cd EmbodiedZero

# example: Run classic VAE (MNIST, ~2 min)
cd vae && pip install -r requirements.txt && python run_vae.py

# example: Run DDPM (MNIST, ~5 min)
cd dp && pip install -r requirements.txt && python run_ddpm.py --epochs 10

# example: Run Flow Matching
cd dp && python run_flow_matching.py --epochs 10
```

---

## Design Principles

- **Minimal dependencies**: PyTorch + torchvision only — no heavy framework wrappers
- **Code as documentation**: Every module has detailed inline comments and mathematical formulas
- **Progressive learning**: such as Diffusion Model, DDPM → DDIM → Stable Diffusion → Flow Matching, each step building on the last
- **Reproducible**: All scripts download public datasets automatically; one command to run

---

## Continuously updating
- Welcome to submit Issues and PR
- Welcome to exchange ideas