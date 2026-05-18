"""
VQ-VAE 大图推理测试脚本
对 --image-dir 中每张图片进行重建，保存左右对比图（原图 | 重建）到 --output-dir

用法：
  python test_vq_vae_large.py \
      --image-dir data/test-image \
      --checkpoint results/vqvae_large_model.pth \
      --output-dir results/test_output_large
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# 从训练脚本导入模型定义
from vae.run_vq_vae import Model

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='VQ-VAE 大图推理测试')
parser.add_argument('--image-dir', type=str, default='data/test-image')
parser.add_argument('--checkpoint', type=str, default='results/vqvae_large_model.pth')
parser.add_argument('--output-dir', type=str, default='results/test_output_large')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────
print(f"加载模型：{args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device)
train_args = checkpoint['args']

model = Model(
    num_hiddens=train_args['num_hiddens'],
    num_residual_layers=train_args['num_residual_layers'],
    num_residual_hiddens=train_args['num_residual_hiddens'],
    num_embeddings=train_args['num_embeddings'],
    embedding_dim=train_args['embedding_dim'],
    commitment_cost=train_args['commitment_cost'],
    decay=train_args['decay'],
    downsample=train_args['downsample'],
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_image_size = train_args['image_size']
print(f"模型训练尺寸: {train_image_size}×{train_image_size}，下采样: {train_args['downsample']}×")

# ─────────────────────────────────────────────
# 预处理（缩放到训练尺寸）
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((train_image_size, train_image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
])

# ─────────────────────────────────────────────
# 读取测试图片
# ─────────────────────────────────────────────
image_paths = sorted([
    p for p in Path(args.image_dir).iterdir()
    if p.suffix.lower() in SUPPORTED
])

if not image_paths:
    print(f"错误：{args.image_dir} 中没有找到支持的图片！")
    exit(1)

print(f"找到 {len(image_paths)} 张图片")
os.makedirs(args.output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 推理 & 保存
# ─────────────────────────────────────────────
with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # 缩放到训练尺寸推理
        x = transform(img).unsqueeze(0).to(device)

        _, x_recon, perplexity = model(x)

        # 反归一化，均保持训练尺寸（train_image_size × train_image_size）
        original_resized = (x.cpu() + 0.5).clamp(0, 1)   # 缩放后的原图
        recon = (x_recon.cpu() + 0.5).clamp(0, 1)        # 重建图

        # 左：缩放后原图，右：重建图（尺寸完全一致，直接横向拼接）
        comparison = torch.cat([original_resized, recon], dim=3)

        out_path = os.path.join(args.output_dir, f'{img_path.stem}_comparison.png')
        save_image(comparison, out_path)
        print(f"  [{img_path.name}]  原始={orig_w}×{orig_h} -> 推理尺寸={train_image_size}×{train_image_size}"
              f"  perplexity={perplexity.item():.1f}"
              f"  -> {out_path}")

print(f"\n完成！共保存 {len(image_paths)} 张对比图到 {args.output_dir}/")
