"""
VQ-VAE inference test script
Reconstructs every image in --image-dir and saves side-by-side comparisons
(original | reconstruction) to --output-dir.

Usage:
  python test_vq_vae.py \
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

# Import model definition from training script
from vae.run_vq_vae import Model

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='VQ-VAE large-image inference test')
parser.add_argument('--image-dir', type=str, default='data/test-image')
parser.add_argument('--checkpoint', type=str, default='results/vqvae_large_model.pth')
parser.add_argument('--output-dir', type=str, default='results/test_output_large')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
print(f"Loading model: {args.checkpoint}")
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
print(f"Model training size: {train_image_size}×{train_image_size}, downsample: {train_args['downsample']}×")

# ─────────────────────────────────────────────
# Preprocessing (resize to training size)
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((train_image_size, train_image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
])

# ─────────────────────────────────────────────
# Load test images
# ─────────────────────────────────────────────
image_paths = sorted([
    p for p in Path(args.image_dir).iterdir()
    if p.suffix.lower() in SUPPORTED
])

if not image_paths:
    print(f"Error: no supported images found in {args.image_dir}!")
    exit(1)

print(f"Found {len(image_paths)} images")
os.makedirs(args.output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Inference & save
# ─────────────────────────────────────────────
with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # resize to training size for inference
        x = transform(img).unsqueeze(0).to(device)

        _, x_recon, perplexity = model(x)

        # de-normalize; both tensors are at training resolution
        original_resized = (x.cpu() + 0.5).clamp(0, 1)   # resized original
        recon = (x_recon.cpu() + 0.5).clamp(0, 1)        # reconstruction

        # left: resized original, right: reconstruction (same size, concat horizontally)
        comparison = torch.cat([original_resized, recon], dim=3)

        out_path = os.path.join(args.output_dir, f'{img_path.stem}_comparison.png')
        save_image(comparison, out_path)
        print(f"  [{img_path.name}]  original={orig_w}×{orig_h} -> inference size={train_image_size}×{train_image_size}"
              f"  perplexity={perplexity.item():.1f}"
              f"  -> {out_path}")

print(f"\nDone! Saved {len(image_paths)} comparison images to {args.output_dir}/")
