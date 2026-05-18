"""
run_ddpm.py  —  DDPM image generation training script (MNIST)

Dependencies
------------
  model/model.py          → SimpleUNet (noise prediction network, shared with DDIM/SD)
  model/noise_schedule.py → NoiseSchedule (β-schedule + forward noising + training loss)
  ddpm/sampler.py         → DDPMSampler (DDPM ancestral sampling)

Saved outputs (results/ directory)
------------------------------------
  samples_epoch_{N}.png      16 generated samples per epoch
  denoising_epoch_{N}.png    denoising trajectory (~10 frames, pure noise → clear image)
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

# 将 dp/ 根目录加入 path，使 model/ 可被直接 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model         import SimpleUNet
from model.noise_schedule import NoiseSchedule
from ddpm.sampler         import DDPMSampler


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DDPM MNIST Training')
parser.add_argument('--epochs',       type=int,   default=20)
parser.add_argument('--batch-size',   type=int,   default=128)
parser.add_argument('--lr',           type=float, default=2e-4)
parser.add_argument('--timesteps',    type=int,   default=1000)
parser.add_argument('--schedule',     type=str,   default='linear',
                    choices=['linear', 'cosine'],
                    help='beta schedule type (default: linear)')
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--log-interval', type=int,   default=100)
parser.add_argument('--no-cuda',      action='store_true')
parser.add_argument('--results-dir',  type=str,   default='results_ddpm')
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
# 数据集（归一化到 [-1, 1]，与扩散过程一致）
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
# 模型（shared）+ 噪声调度（shared）+ 采样器（DDPM 特有）
# ---------------------------------------------------------------------------
model    = SimpleUNet(in_channels=1, base_channels=64, time_emb_dim=128).to(device)
schedule = NoiseSchedule(num_timesteps=args.timesteps, schedule=args.schedule).to(device)
sampler  = DDPMSampler(schedule)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

print(f'Model params  : {sum(p.numel() for p in model.parameters()):,}')
print(f'Timesteps     : {args.timesteps}  schedule={args.schedule}\n')


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(epoch: int) -> float:
    model.train()
    total = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss = schedule.p_losses(model, imgs)    # ← model
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
            total += schedule.p_losses(model, imgs.to(device)).item()   # ← model
    avg = total / len(val_loader)
    print(f'====> Epoch: {epoch}  Val   Avg Loss: {avg:.6f}')
    return avg


# ---------------------------------------------------------------------------
# 生成样本 & 去噪轨迹可视化
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_samples(epoch: int):
    model.eval()

    # 生成 16 张样本
    samples, _ = sampler.sample(model, shape=(16, 1, 28, 28), device=device)  # ← DDPM
    samples = (samples.clamp(-1, 1) + 1) / 2
    path = os.path.join(args.results_dir, f'samples_epoch_{epoch:03d}.png')
    save_image(samples, path, nrow=4)
    print(f'  [Saved] samples      → {path}')

    # 去噪轨迹（约 10 帧）
    _, frames = sampler.sample(                                                # ← DDPM
        model, shape=(4, 1, 28, 28), device=device,
        save_every=args.timesteps // 9)
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
