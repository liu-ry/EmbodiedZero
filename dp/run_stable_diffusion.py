"""
run_stable_diffusion.py  —  Latent Diffusion（Stable Diffusion 简化版）训练脚本（MNIST）

架构说明
--------
Stable Diffusion = VAE（像素 ↔ 隐空间）+ LDM（隐空间上的扩散模型）

本脚本分为两个训练阶段：

  阶段一：训练 VAE
  ─────────────────
    目标：学习将 28×28 MNIST 图像压缩到 (4, 7, 7) 的隐空间（16× 压缩）
    损失：MSE 重构损失 + β·KL 散度
    保存：vae_best.pt

  阶段二：训练 LDM（Latent Diffusion Model）
  ─────────────────────────────────────────
    目标：在冻结的 VAE 隐空间上训练扩散 UNet（预测噪声 ε_θ）
    流程：
      1. 用 VAE 编码器把图像 → 隐变量 z（不加噪声，仅用均值）
      2. 在 z 上做 DDPM 前向加噪：z_t = √ᾱ_t · z + √(1-ᾱ_t) · ε
      3. UNet 预测噪声 ε_θ(z_t, t)
    保存：ldm_best.pt

推理阶段
────────
    从隐空间 z_T ~ N(0,I) 出发，用 DDIM 去噪到 z_0，再用 VAE 解码到图像

依赖关系
────────
  model/vae.py              → VAE（编码器 + 解码器）
  model/model.py            → SimpleUNet（在隐空间上预测噪声）
  model/noise_schedule.py   → NoiseSchedule（扩散调度）
  stable_diffusion/sampler.py → LatentDDIMSampler（隐空间采样 + VAE 解码）

使用示例
────────
  # 完整两阶段训练（默认各 20 epoch）
  python run_stable_diffusion.py

  # 快速测试（各 2 epoch）
  python run_stable_diffusion.py --vae-epochs 2 --ldm-epochs 2

  # 跳过 VAE 训练，使用已有权重
  python run_stable_diffusion.py --vae-epochs 0 --vae-ckpt results_sd/vae_best.pt

  # DDIM 50 步推理
  python run_stable_diffusion.py --ddim-steps 50

保存内容（results_sd/ 目录）
─────────────────────────────
  vae_recon_epoch_{N}.png    VAE 重构对比图（原图 vs 重构）
  vae_best.pt                VAE 最佳权重
  samples_epoch_{N}.png      LDM 每 epoch 生成的 16 张样本
  denoising_epoch_{N}.png    去噪轨迹图
  ldm_best.pt                LDM 最佳权重
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

# 通用
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--batch-size',   type=int,   default=128)
parser.add_argument('--no-cuda',      action='store_true')
parser.add_argument('--results-dir',  type=str,   default='results_sd')
parser.add_argument('--log-interval', type=int,   default=100)

# VAE 阶段
parser.add_argument('--vae-epochs',   type=int,   default=20,
                    help='VAE 训练轮数（0 = 跳过，需要 --vae-ckpt）')
parser.add_argument('--vae-lr',       type=float, default=1e-3)
parser.add_argument('--vae-kl-weight',type=float, default=1e-3,
                    help='KL 散度权重 β（越大隐空间越接近正态，重构质量略降）')
parser.add_argument('--vae-ckpt',     type=str,   default=None,
                    help='加载已有 VAE 权重路径（跳过 VAE 训练时必须指定）')

# LDM 阶段
parser.add_argument('--ldm-epochs',   type=int,   default=20,
                    help='LDM（隐空间扩散）训练轮数')
parser.add_argument('--ldm-lr',       type=float, default=2e-4)
parser.add_argument('--timesteps',    type=int,   default=1000)
parser.add_argument('--schedule',     type=str,   default='linear',
                    choices=['linear', 'cosine'])
parser.add_argument('--ldm-ckpt',     type=str,   default=None,
                    help='加载已有 LDM 权重（--ldm-epochs 0 时跳过训练）')

# 推理
parser.add_argument('--ddim-steps',   type=int,   default=50,
                    help='DDIM 推理步数')
parser.add_argument('--eta',          type=float, default=0.0,
                    help='DDIM η（0=确定性，1≈DDPM）')

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
# VAE：像素空间 ↔ 隐空间 (4, 7, 7)
vae = VAE(in_channels=1, latent_ch=4, base_ch=32,
          kl_weight=args.vae_kl_weight).to(device)

# LDM UNet：在隐空间上预测噪声（输入通道=4=latent_ch）
ldm_unet = SimpleUNet(in_channels=4, base_channels=64, time_emb_dim=128).to(device)

# 扩散调度（与 DDPM/DDIM 完全共用）
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
    """保存原图 vs VAE 重构图对比。"""
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
    """运行 VAE 训练阶段。"""
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
    best_val = float('inf')
    print('=' * 60)
    print('阶段一：训练 VAE')
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
    每步：
      1. 用冻结 VAE 编码图像到隐空间 z
      2. 在 z 上做 NoiseSchedule.p_losses（DDPM 训练目标）
    """
    ldm_unet.train()
    vae.eval()   # VAE 冻结
    total = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        # 编码到隐空间（不加噪，用均值）
        with torch.no_grad():
            z = vae.encode_to_latent(imgs)  # (B, 4, 7, 7)

        # 在隐空间上计算扩散损失
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
    print('阶段二：训练 LDM（隐空间扩散）')
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
        print('[警告] vae_epochs=0 且未指定 --vae-ckpt，隐空间质量无法保证。')

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
