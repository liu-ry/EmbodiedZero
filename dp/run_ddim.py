"""
run_ddim.py  —  DDIM image generation training script (MNIST)

Algorithm
---------
  DDIM (Denoising Diffusion Implicit Models, Song et al. 2020)
  Uses the **exact same training objective** as DDPM (predicting noise ε),
  differing only in the **inference (sampling) phase**:
    - DDPM: adds random noise each step, requires all T=1000 steps
    - DDIM: deterministic non-Markovian skipping, 50–100 steps yields comparable quality

Dependencies
------------
  model/model.py          → SimpleUNet (shared with DDPM)
  model/noise_schedule.py → NoiseSchedule (shared with DDPM)
  ddim/sampler.py         → DDIMSampler (DDIM-specific sampling logic)

Key arguments
-------------
  --ddim-steps   : DDIM sampling steps (default 50, range 20–200)
  --eta          : DDIM stochasticity η (0=fully deterministic, 1≈DDPM)
  other args same as run_ddpm.py

Saved outputs (results_ddim/ directory)
-----------------------------------------
  samples_epoch_{N}.png      16 generated samples per epoch
  denoising_epoch_{N}.png    denoising trajectory (~8 frames)
  best_model.pt              model weights at best validation loss

Usage
-----
  # Standard training (50-step deterministic DDIM)
  python run_ddim.py

  # Load a pretrained DDPM checkpoint and run inference only (skip training)
  python run_ddim.py --epochs 0 --load-ckpt results/best_model.pt

  # Stochastic DDIM (η=1 approximates DDPM)
  python run_ddim.py --eta 1.0 --ddim-steps 100
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

from model.model         import SimpleUNet
from model.noise_schedule import NoiseSchedule
from ddim.sampler         import DDIMSampler


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DDIM MNIST Training')
parser.add_argument('--epochs',       type=int,   default=20)
parser.add_argument('--batch-size',   type=int,   default=128)
parser.add_argument('--lr',           type=float, default=2e-4)
parser.add_argument('--timesteps',    type=int,   default=1000,
                    help='total diffusion timesteps T (maximum timestep used during training)')
parser.add_argument('--ddim-steps',   type=int,   default=50,
                    help='DDIM inference steps S (≤ timesteps; fewer = faster)')
parser.add_argument('--eta',          type=float, default=0.0,
                    help='DDIM stochasticity η (0=deterministic, 1≈DDPM)')
parser.add_argument('--schedule',     type=str,   default='linear',
                    choices=['linear', 'cosine'])
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--log-interval', type=int,   default=100)
parser.add_argument('--no-cuda',      action='store_true')
parser.add_argument('--results-dir',  type=str,   default='results_ddim')
parser.add_argument('--load-ckpt',    type=str,   default=None,
                    help='load existing weights and run inference (use with --epochs 0 to skip training)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'Using device  : {device}')
print(f'DDIM steps    : {args.ddim_steps}  (η={args.eta})')

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
# 模型 + 噪声调度 + DDIM 采样器
# ---------------------------------------------------------------------------
model    = SimpleUNet(in_channels=1, base_channels=64, time_emb_dim=128).to(device)
schedule = NoiseSchedule(num_timesteps=args.timesteps, schedule=args.schedule).to(device)
sampler  = DDIMSampler(schedule, ddim_steps=args.ddim_steps, eta=args.eta)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

print(f'Model params  : {sum(p.numel() for p in model.parameters()):,}')
print(f'Timesteps     : {args.timesteps}  schedule={args.schedule}\n')

# 可选：加载预训练权重（例如来自 DDPM 的 best_model.pt）
if args.load_ckpt is not None:
    ckpt = torch.load(args.load_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    print(f'Loaded checkpoint: {args.load_ckpt}\n')


# ---------------------------------------------------------------------------
# 训练（与 DDPM 完全相同的训练目标）
# ---------------------------------------------------------------------------
def train(epoch: int) -> float:
    model.train()
    total = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss = schedule.p_losses(model, imgs)   # MSE(ε_pred, ε_true)
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
# 生成样本 & 去噪轨迹可视化（DDIM 推理）
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_samples(epoch: int):
    model.eval()

    # 生成 16 张样本（DDIM S 步采样）
    samples, _ = sampler.sample(model, shape=(16, 1, 28, 28), device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    path = os.path.join(args.results_dir, f'samples_epoch_{epoch:03d}.png')
    save_image(samples, path, nrow=4)
    print(f'  [Saved] samples      → {path}')

    # 去噪轨迹（每隔 S//7 步存一帧，约 8 帧）
    save_every = max(1, args.ddim_steps // 7)
    _, frames = sampler.sample(
        model, shape=(4, 1, 28, 28), device=device,
        save_every=save_every)
    if frames:
        rows = [make_grid((f.clamp(-1, 1) + 1) / 2, nrow=4) for f in frames]
        grid = torch.cat(rows, dim=1)
        path = os.path.join(args.results_dir, f'denoising_epoch_{epoch:03d}.png')
        save_image(grid, path)
        print(f'  [Saved] denoising    → {path}')
    print()


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    best_val = float('inf')

    # 若 --epochs 0 且已加载权重，直接做一次推理演示
    if args.epochs == 0:
        print('Skipping training (--epochs 0). Running inference demo...')
        save_samples(0)
    else:
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
