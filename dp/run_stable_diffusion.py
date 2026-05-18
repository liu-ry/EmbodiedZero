"""
run_stable_diffusion.py  —  Latent Diffusion (Stable Diffusion simplified) training script (MNIST)

Architecture
------------
Stable Diffusion = VAE (pixel ↔ latent space) + LDM (diffusion model in latent space)

This script has two training stages:

  Stage 1: Train VAE
  ──────────────────
    Goal : learn to compress 28×28 MNIST images into a (4, 7, 7) latent space (16× compression)
    Loss : MSE reconstruction + β·KL divergence
    Saves: vae_best.pt

  Stage 2: Train LDM (Latent Diffusion Model)
  ────────────────────────────────────────────
    Goal : train a diffusion UNet (predicts noise ε_θ) in the frozen VAE latent space
    Steps:
      1. Encode images to latent z using frozen VAE encoder (mean only, no noise)
      2. Apply DDPM forward noising in latent space: z_t = √ᾱ_t · z + √(1-ᾱ_t) · ε
      3. UNet predicts ε_θ(z_t, t)
    Saves: ldm_best.pt

Inference
---------
    Start from z_T ~ N(0,I) in latent space, denoise with DDIM → z_0, decode with VAE

Dependencies
------------
  model/vae.py                → VAE (encoder + decoder)
  model/model.py              → SimpleUNet (predicts noise in latent space)
  model/noise_schedule.py     → NoiseSchedule (diffusion schedule)
  stable_diffusion/sampler.py → LatentDDIMSampler (latent sampling + VAE decoding)

Usage
-----
  # Full two-stage training (default 20 epochs each)
  python run_stable_diffusion.py

  # Quick test (2 epochs each)
  python run_stable_diffusion.py --vae-epochs 2 --ldm-epochs 2

  # Skip VAE training, load existing weights
  python run_stable_diffusion.py --vae-epochs 0 --vae-ckpt results_sd/vae_best.pt

  # DDIM 50-step inference
  python run_stable_diffusion.py --ddim-steps 50

Saved outputs (results_sd/ directory)
──────────────────────────────────────
  vae_recon_epoch_{N}.png    VAE reconstruction comparison (original vs reconstructed)
  vae_best.pt                best VAE weights
  samples_epoch_{N}.png      16 generated samples per LDM epoch
  denoising_epoch_{N}.png    denoising trajectory
  ldm_best.pt                best LDM weights
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

from model.vae             import VAE
from model.model           import SimpleUNet
from model.noise_schedule  import NoiseSchedule
from stable_diffusion.sampler import LatentDDIMSampler


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Latent Diffusion (Stable Diffusion) MNIST')

# Common
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--batch-size',   type=int,   default=128)
parser.add_argument('--no-cuda',      action='store_true')
parser.add_argument('--results-dir',  type=str,   default='results_sd')
parser.add_argument('--log-interval', type=int,   default=100)

# VAE stage
parser.add_argument('--vae-epochs',   type=int,   default=20,
                    help='VAE training epochs (0 = skip; requires --vae-ckpt)')
parser.add_argument('--vae-lr',       type=float, default=1e-3)
parser.add_argument('--vae-kl-weight',type=float, default=1e-3,
                    help='KL divergence weight β (higher = more regular latent space, slight reconstruction drop)')
parser.add_argument('--vae-ckpt',     type=str,   default=None,
                    help='path to existing VAE weights (required when --vae-epochs 0)')

# LDM stage
parser.add_argument('--ldm-epochs',   type=int,   default=20,
                    help='LDM (latent diffusion) training epochs')
parser.add_argument('--ldm-lr',       type=float, default=2e-4)
parser.add_argument('--timesteps',    type=int,   default=1000)
parser.add_argument('--schedule',     type=str,   default='linear',
                    choices=['linear', 'cosine'])
parser.add_argument('--ldm-ckpt',     type=str,   default=None,
                    help='load existing LDM weights (skip training when --ldm-epochs 0)')

# Inference
parser.add_argument('--ddim-steps',   type=int,   default=50,
                    help='DDIM inference steps')
parser.add_argument('--eta',          type=float, default=0.0,
                    help='DDIM η (0=deterministic, 1≈DDPM)')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'Using device  : {device}')

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
# 模型
# ---------------------------------------------------------------------------
# VAE: pixel space ↔ latent space (4, 7, 7)
vae = VAE(in_channels=1, latent_ch=4, base_ch=32,
          kl_weight=args.vae_kl_weight).to(device)

# LDM UNet: predicts noise in latent space (in_channels=4=latent_ch)
ldm_unet = SimpleUNet(in_channels=4, base_channels=64, time_emb_dim=128).to(device)

# Diffusion schedule (fully shared with DDPM/DDIM)
schedule = NoiseSchedule(num_timesteps=args.timesteps, schedule=args.schedule).to(device)

# 推理采样器
sampler = LatentDDIMSampler(vae, schedule,
                            ddim_steps=args.ddim_steps, eta=args.eta)

print(f'VAE params    : {sum(p.numel() for p in vae.parameters()):,}')
print(f'LDM params    : {sum(p.numel() for p in ldm_unet.parameters()):,}')
print(f'Timesteps     : {args.timesteps}  schedule={args.schedule}')
print(f'DDIM steps    : {args.ddim_steps}  η={args.eta}\n')


# ===========================================================================
# 阶段一：训练 VAE
# ===========================================================================
def train_vae_epoch(optimizer, epoch: int) -> float:
    vae.train()
    total = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        _, loss = vae(imgs)
        loss.backward()
        optimizer.step()
        total += loss.item()
        if batch_idx % args.log_interval == 0:
            print('  [VAE] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total / len(train_loader)


def validate_vae(epoch: int) -> float:
    vae.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            _, loss = vae(imgs.to(device))
            total += loss.item()
    avg = total / len(val_loader)
    print(f'  [VAE] Epoch: {epoch}  Val Loss: {avg:.6f}')
    return avg


@torch.no_grad()
def save_vae_recon(epoch: int):
    """Save original vs VAE reconstruction comparison."""
    vae.eval()
    imgs, _ = next(iter(val_loader))
    imgs    = imgs[:8].to(device)
    recon, _ = vae(imgs)
    # 交替排列：原图、重构、原图、重构...
    comparison = torch.cat([imgs, recon], dim=0)
    comparison = (comparison.clamp(-1, 1) + 1) / 2
    path = os.path.join(args.results_dir, f'vae_recon_epoch_{epoch:03d}.png')
    save_image(comparison, path, nrow=8)
    print(f'  [VAE] Saved recon → {path}')


def run_vae_training():
    """Run the VAE training stage."""
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
    best_val = float('inf')
    print('=' * 60)
    print('Stage 1: Train VAE')
    print('=' * 60)
    for epoch in range(1, args.vae_epochs + 1):
        train_vae_epoch(vae_optimizer, epoch)
        val_loss = validate_vae(epoch)
        save_vae_recon(epoch)
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.results_dir, 'vae_best.pt')
            torch.save(vae.state_dict(), ckpt_path)
            print(f'  >> New best VAE val loss: {best_val:.6f}, saved → {ckpt_path}\n')
    print(f'VAE training done.  Best val loss: {best_val:.6f}\n')


# ===========================================================================
# 阶段二：训练 LDM（隐空间扩散）
# ===========================================================================
def train_ldm_epoch(optimizer, epoch: int) -> float:
    """
    Each step:
      1. Encode image to latent z using frozen VAE (mean only, no noise)
      2. Apply NoiseSchedule.p_losses (DDPM training objective) on z
    """
    ldm_unet.train()
    vae.eval()   # freeze VAE
    total = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        # Encode to latent space (deterministic, use mean)
        with torch.no_grad():
            z = vae.encode_to_latent(imgs)  # (B, 4, 7, 7)

        # Compute diffusion loss in latent space
        loss = schedule.p_losses(ldm_unet, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ldm_unet.parameters(), 1.0)
        optimizer.step()
        total += loss.item()

        if batch_idx % args.log_interval == 0:
            print('  [LDM] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total / len(train_loader)


def validate_ldm(epoch: int) -> float:
    ldm_unet.eval()
    vae.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            z = vae.encode_to_latent(imgs.to(device))
            total += schedule.p_losses(ldm_unet, z).item()
    avg = total / len(val_loader)
    print(f'  [LDM] Epoch: {epoch}  Val Loss: {avg:.6f}')
    return avg


@torch.no_grad()
def save_ldm_samples(epoch: int):
    ldm_unet.eval()
    vae.eval()

    # 生成 16 张图像
    images, _ = sampler.sample(ldm_unet, n_samples=16, device=device)
    images = (images.clamp(-1, 1) + 1) / 2
    path = os.path.join(args.results_dir, f'samples_epoch_{epoch:03d}.png')
    save_image(images, path, nrow=4)
    print(f'  [LDM] Saved samples → {path}')

    # 去噪轨迹（取 4 张，约 8 帧）—— 每帧经过 VAE 解码到像素空间
    _, frames = sampler.sample(ldm_unet, n_samples=4, device=device,
                               save_every=max(1, args.ddim_steps // 7))
    if frames:
        decoded_frames = [vae.decode(f.to(device)) for f in frames]
        rows = [make_grid((f.clamp(-1, 1) + 1) / 2, nrow=4) for f in decoded_frames]
        grid = torch.cat(rows, dim=1)
        path = os.path.join(args.results_dir, f'denoising_epoch_{epoch:03d}.png')
        save_image(grid, path)
        print(f'  [LDM] Saved denoising → {path}')
    print()


def run_ldm_training():
    ldm_optimizer = optim.Adam(ldm_unet.parameters(), lr=args.ldm_lr)
    best_val = float('inf')
    print('=' * 60)
    print('Stage 2: Train LDM (latent diffusion)')
    print('=' * 60)
    for epoch in range(1, args.ldm_epochs + 1):
        train_ldm_epoch(ldm_optimizer, epoch)
        val_loss = validate_ldm(epoch)
        save_ldm_samples(epoch)
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.results_dir, 'ldm_best.pt')
            torch.save(ldm_unet.state_dict(), ckpt_path)
            print(f'  >> New best LDM val loss: {best_val:.6f}, saved → {ckpt_path}\n')
    print(f'LDM training done.  Best val loss: {best_val:.6f}\n')


# ===========================================================================
# 主入口
# ===========================================================================
if __name__ == '__main__':

    # —— 阶段一：VAE ——
    if args.vae_epochs > 0:
        run_vae_training()
    elif args.vae_ckpt is not None:
        print(f'Loading VAE weights from {args.vae_ckpt}')
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    else:
        print('[Warning] vae_epochs=0 and --vae-ckpt not specified; latent space quality is not guaranteed.')

    # 冻结 VAE 参数（阶段二无需更新）
    for p in vae.parameters():
        p.requires_grad_(False)

    # —— 阶段二：LDM ——
    if args.ldm_epochs > 0:
        run_ldm_training()
    elif args.ldm_ckpt is not None:
        print(f'Loading LDM weights from {args.ldm_ckpt}')
        ldm_unet.load_state_dict(torch.load(args.ldm_ckpt, map_location=device))
        # 直接推理演示
        print('Running inference demo...')
        save_ldm_samples(epoch=0)

    print('All done.')
    print(f'Results saved to: {args.results_dir}/')
