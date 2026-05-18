"""
run_flow_matching.py  —  Flow Matching image generation training script (MNIST)

Algorithm
---------
Flow Matching (Lipman et al., 2022) is a cleaner generative framework than DDPM/DDIM:

  Core idea
  ├─ Define a straight-line flow (OT-CFM) from noise p_0=N(0,I) to data p_1
  ├─ Train a vector-field network v_θ(x_t, t) to fit the tangent direction of the flow
  └─ At inference, integrate x_0 → x_1 with an Euler/Heun ODE solver

  Training objective (OT-CFM: Optimal Transport Conditional Flow Matching)
  ├─ Sample t ~ Uniform(0,1), x_0 ~ N(0,I), x_1 ~ p_data
  ├─ Compute interpolation x_t = (1 - t) · x_0 + t · x_1
  └─ Minimize ||v_θ(x_t, t) - (x_1 - x_0)||²

Comparison with DDPM/DDIM
--------------------------
  DDPM    : Markov noising with T=1000 steps; learns ε_θ; must denoise step-by-step
  DDIM    : Same, but can skip steps (non-Markovian, ~50 steps)
  Flow FM : Continuous time; learns vector field v_θ; inference via ODE (any step count,
            typically 100 steps); simpler math, more stable training, often better quality

Dependencies
------------
  model/model.py          → SimpleUNet (used here to predict v_θ instead of ε_θ, same architecture)
  model/flow_schedule.py  → FlowSchedule (CFM data interpolation + training loss)
  flow_matching/sampler.py → FlowMatchingSampler (Euler/Heun ODE integration)

Usage
-----
  # Standard training (Euler 100 steps)
  python run_flow_matching.py

  # Heun method (50 steps reaches quality comparable to Euler 100 steps)
  python run_flow_matching.py --ode-steps 50 --ode-method heun

  # Quick test
  python run_flow_matching.py --epochs 2 --ode-steps 20

Saved outputs (results_fm/ directory)
────────────────────────────────────────
  samples_epoch_{N}.png      16 generated samples per epoch
  trajectory_epoch_{N}.png   ODE trajectory (~8 frames, pure noise → clear image)
  best_model.pt              model weights at best validation loss
"""

from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model          import SimpleUNet
from model.flow_schedule  import FlowSchedule
from flow_matching.sampler import FlowMatchingSampler


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Flow Matching MNIST Training')
parser.add_argument('--epochs',       type=int,   default=20)
parser.add_argument('--batch-size',   type=int,   default=128)
parser.add_argument('--lr',           type=float, default=2e-4)
parser.add_argument('--timesteps',    type=int,   default=1000,
                    help='integer discretization for time embedding (does not affect the continuous-flow math)')
parser.add_argument('--sigma-min',    type=float, default=1e-4,
                    help='minimum noise in the conditional flow (numerical stability term; usually no need to change)')
parser.add_argument('--ode-steps',    type=int,   default=100,
                    help='ODE integration steps at inference (more = more accurate, typically 50–200)')
parser.add_argument('--ode-method',   type=str,   default='euler',
                    choices=['euler', 'heun'],
                    help='ODE integrator (heun is higher-order; half the steps for similar quality)')
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--log-interval', type=int,   default=100)
parser.add_argument('--no-cuda',      action='store_true')
parser.add_argument('--results-dir',  type=str,   default='results_fm')
parser.add_argument('--load-ckpt',    type=str,   default=None,
                    help='path to existing checkpoint for skipping training and running inference directly')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'Using device  : {device}')
print(f'ODE method    : {args.ode_method}  steps={args.ode_steps}\n')

os.makedirs(args.results_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 数据集（归一化到 [-1, 1]）
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True,  download=True, transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=args.batch_size, shuffle=False, **kwargs)

# ---------------------------------------------------------------------------
# 模型 + 调度 + 采样器
# ---------------------------------------------------------------------------
# SimpleUNet reused here: predicts vector field v_θ instead of noise ε_θ, same architecture
model    = SimpleUNet(in_channels=1, base_channels=64, time_emb_dim=128).to(device)
schedule = FlowSchedule(num_timesteps=args.timesteps, sigma_min=args.sigma_min).to(device)
sampler  = FlowMatchingSampler(schedule, ode_steps=args.ode_steps)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

print(f'Model params  : {sum(p.numel() for p in model.parameters()):,}')
print(f'Timesteps     : {args.timesteps}  sigma_min={args.sigma_min}\n')


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(epoch: int) -> float:
    model.train()
    total = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss = schedule.p_losses(model, imgs)   # CFM loss: MSE(v_θ(x_t,t), u_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg = total / len(train_loader)
    print(f'====> Epoch: {epoch}  Train Avg Loss: {avg:.6f}')
    return avg


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def validate(epoch: int) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            total += schedule.p_losses(model, imgs.to(device)).item()
    avg = total / len(val_loader)
    print(f'====> Epoch: {epoch}  Val   Avg Loss: {avg:.6f}')
    return avg


# ---------------------------------------------------------------------------
# 生成样本 & ODE 轨迹可视化
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_samples(epoch: int):
    model.eval()

    # 生成 16 张样本（Euler/Heun ODE 积分 t: 0→1）
    samples, _ = sampler.sample(
        model, shape=(16, 1, 28, 28), device=device, method=args.ode_method)
    samples = (samples.clamp(-1, 1) + 1) / 2
    path = os.path.join(args.results_dir, f'samples_epoch_{epoch:03d}.png')
    save_image(samples, path, nrow=4)
    print(f'  [Saved] samples      → {path}')

    # ODE 轨迹（约 8 帧，从噪声到图像）
    _, frames = sampler.sample(
        model, shape=(4, 1, 28, 28), device=device,
        method=args.ode_method,
        save_every=max(1, args.ode_steps // 7))
    if frames:
        rows = [make_grid((f.clamp(-1, 1) + 1) / 2, nrow=4) for f in frames]
        grid = torch.cat(rows, dim=1)
        path = os.path.join(args.results_dir, f'trajectory_epoch_{epoch:03d}.png')
        save_image(grid, path)
        print(f'  [Saved] trajectory   → {path}')
    print()


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 加载已有权重（可选）
    if args.load_ckpt is not None:
        print(f'Loading checkpoint from {args.load_ckpt}')
        model.load_state_dict(torch.load(args.load_ckpt, map_location=device))

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        val_loss = validate(epoch)
        save_samples(epoch)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.results_dir, 'best_model.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  >> New best val loss: {best_val:.6f}, saved to {ckpt_path}\n')

    print(f'Training finished.  Best val loss: {best_val:.6f}')
    print(f'All images saved to: {args.results_dir}/')
