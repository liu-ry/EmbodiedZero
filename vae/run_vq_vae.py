"""
VQ-VAE 大图版本
支持 STL-10（96×96）或自定义图片目录，训练任意分辨率图像

编码器下采样倍数可选：
  --downsample 4   -> 4×  缩小（2层 stride-2 Conv），适合 96×96 / 128×128
  --downsample 8   -> 8×  缩小（3层 stride-2 Conv），适合 256×256（默认）
  --downsample 16  -> 16× 缩小（4层 stride-2 Conv），适合 512×512 及以上

用法示例：
  # STL-10（96×96，需手动下载解压到 data/stl10_binary/）
  python vq_vae_large.py --dataset stl10 --image-size 96 --downsample 4 --save-results

  # 自定义图片目录，图像缩放到 256x256
  python vq_vae_large.py --dataset custom --image-dir data/train-images --image-size 256 --save-results

  # 训练 512x512
  python vq_vae_large.py --dataset custom --image-dir data/train-images --image-size 512 \
      --downsample 16 --batch-size 16 --num-hiddens 256
"""

from __future__ import print_function

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下保存图片
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='VQ-VAE 大图训练')
parser.add_argument('--dataset', type=str, default='stl10',
                    choices=['stl10', 'custom'],
                    help='数据集类型：stl10（自动下载 96×96）或 custom（自定义图片目录）')
parser.add_argument('--image-dir', type=str, default=None,
                    help='custom 模式下的训练图片目录（会递归搜索 jpg/png 等格式）')
parser.add_argument('--data-root', type=str, default='data',
                    help='stl10 数据集的下载/缓存根目录')
parser.add_argument('--image-size', type=int, default=96,
                    help='训练时统一缩放到此尺寸（stl10 默认 96）')
parser.add_argument('--downsample', type=int, default=4, choices=[4, 8, 16],
                    help='编码器下采样倍数（stl10/96 推荐 4，256 推荐 8）')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch 大小，stl10/96×96 可用 64，大图适当减小')
parser.add_argument('--num-epochs', type=int, default=500,
                    help='训练 epoch 数')
parser.add_argument('--num-hiddens', type=int, default=128)
parser.add_argument('--num-residual-hiddens', type=int, default=32)
parser.add_argument('--num-residual-layers', type=int, default=2)
parser.add_argument('--embedding-dim', type=int, default=64)
parser.add_argument('--num-embeddings', type=int, default=512)
parser.add_argument('--commitment-cost', type=float, default=0.25)
parser.add_argument('--decay', type=float, default=0.99)
parser.add_argument('--learning-rate', type=float, default=2e-4)
parser.add_argument('--val-ratio', type=float, default=0.1,
                    help='验证集比例')
parser.add_argument('--log-interval', type=int, default=10,
                    help='每多少个 batch 打印一次 loss')
parser.add_argument('--save-interval', type=int, default=5,
                    help='每多少个 epoch 保存一次重建样图')
parser.add_argument('--save-results', action='store_true')
parser.add_argument('--checkpoint', type=str, default='results/vqvae_large_model.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────
# 自定义数据集
# ─────────────────────────────────────────────
SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

class ImageFolderDataset(Dataset):
    """从一个目录（递归）加载所有图片"""
    def __init__(self, image_dir, image_size, transform=None):
        self.paths = sorted([
            p for p in Path(image_dir).rglob('*')
            if p.suffix.lower() in SUPPORTED
        ])
        if not self.paths:
            raise RuntimeError(f"在 {image_dir} 中未找到任何图片！")
        print(f"共找到 {len(self.paths)} 张图片")
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


# ─────────────────────────────────────────────
# 向量量化层（EMA 版）
# ─────────────────────────────────────────────
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        if self.training:
            self._ema_cluster_size = (self._ema_cluster_size * self._decay
                                      + (1 - self._decay) * torch.sum(encodings, 0))
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                      / (n + self._num_embeddings * self._epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


# ─────────────────────────────────────────────
# 残差块 & 残差栈
# ─────────────────────────────────────────────
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 1, 1, bias=False),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


# ─────────────────────────────────────────────
# 编码器（可配置下采样层数）
#   downsample=4  -> 2层 stride-2  (input/4)
#   downsample=8  -> 3层 stride-2  (input/8)
#   downsample=16 -> 4层 stride-2  (input/16)
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, downsample=8):
        super().__init__()
        layers = []
        ch = in_channels

        # 根据 downsample 决定 stride-2 层数
        num_down = {4: 2, 8: 3, 16: 4}[downsample]
        out_channels_seq = []
        for i in range(num_down):
            out_ch = num_hiddens // 2 if i < num_down - 1 else num_hiddens
            # 避免 0 通道
            out_ch = max(out_ch, num_hiddens // (2 ** (num_down - 1 - i)))
            out_channels_seq.append(out_ch)

        # 构建下采样层
        cur_ch = in_channels
        for out_ch in out_channels_seq:
            layers.append(nn.Conv2d(cur_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(True))
            cur_ch = out_ch

        # 最后一个 same-padding conv
        layers.append(nn.Conv2d(cur_ch, num_hiddens, kernel_size=3, stride=1, padding=1))
        self._convs = nn.Sequential(*layers)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens,
                                             num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        return self._residual_stack(self._convs(x))


# ─────────────────────────────────────────────
# 解码器（与编码器对称）
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
                 num_residual_hiddens, downsample=8):
        super().__init__()
        self._conv_in = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens,
                                             num_residual_layers, num_residual_hiddens)

        num_up = {4: 2, 8: 3, 16: 4}[downsample]
        up_layers = []
        cur_ch = num_hiddens
        for i in range(num_up):
            out_ch = 3 if i == num_up - 1 else max(num_hiddens // (2 ** (i + 1)), 16)
            up_layers.append(
                nn.ConvTranspose2d(cur_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            if i < num_up - 1:
                up_layers.append(nn.ReLU(True))
            cur_ch = out_ch

        self._up_convs = nn.Sequential(*up_layers)

    def forward(self, x):
        x = self._conv_in(x)
        x = self._residual_stack(x)
        return self._up_convs(x)


# ─────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0.99,
                 downsample=8):
        super().__init__()
        self._encoder = Encoder(3, num_hiddens, num_residual_layers,
                                num_residual_hiddens, downsample)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, 1, 1)
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                          commitment_cost, decay)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers,
                                num_residual_hiddens, downsample)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity


# ─────────────────────────────────────────────
# 保存对比图（每组：上行原图，下行重建）
# ─────────────────────────────────────────────
def save_comparison(originals, reconstructions, path, nrow=4):
    """originals, reconstructions: tensor [N, C, H, W]，值域 [-0.5, 0.5]"""
    n = min(nrow, originals.shape[0])
    orig = (originals[:n].cpu() + 0.5).clamp(0, 1)
    recon = (reconstructions[:n].cpu() + 0.5).clamp(0, 1)
    # 上行原图，下行重建
    grid = torch.cat([orig, recon], dim=0)  # [2n, C, H, W]
    save_image(grid, path, nrow=n)


# ─────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────
def train():
    # ── 数据集 ──
    norm = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        norm,
    ])

    if args.dataset == 'stl10':
        print(f"使用 STL-10 数据集（{args.image_size}×{args.image_size}），从 {args.data_root}/stl10_binary 加载（不自动下载）")
        stl_dir = Path(args.data_root) / 'stl10_binary'
        expected_files = ['train_X.bin', 'test_X.bin']
        missing = [f for f in expected_files if not (stl_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"STL-10 数据未找到。请手动下载并解压到 {stl_dir}/\n"
                "下载地址: https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz\n"
                "解压后目录结构应包含: train_X.bin, train_y.bin, test_X.bin, test_y.bin, ..."
            )

        # 合并 train（5000张）+ test（8000张）= 13000 张，从中切出少量做验证
        stl_train = tv_datasets.STL10(root=args.data_root, split='train',
                                      download=False, transform=tf)
        stl_test  = tv_datasets.STL10(root=args.data_root, split='test',
                                      download=False, transform=tf)
        full_dataset = torch.utils.data.ConcatDataset([stl_train, stl_test])

        val_size   = max(8, int(len(full_dataset) * args.val_ratio))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"  合并后总计: {len(full_dataset)} 张  训练集: {train_size} 张  验证集: {val_size} 张")
    else:
        if not args.image_dir:
            raise ValueError("custom 模式需要指定 --image-dir")
        full_dataset = ImageFolderDataset(args.image_dir, args.image_size)
        val_size = max(1, int(len(full_dataset) * args.val_ratio))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"  训练集: {train_size} 张  验证集: {val_size} 张")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=min(8, len(val_dataset)),
                              shuffle=False, num_workers=2, pin_memory=True)

    # 计算数据方差（用前 1000 张估算）
    n_sample = min(1000, len(train_dataset))
    sample_imgs = []
    for i in range(n_sample):
        item = train_dataset[i]
        img = item[0] if isinstance(item, (tuple, list)) else item
        sample_imgs.append(img)
    data_variance = torch.stack(sample_imgs).var().item()
    print(f"数据方差: {data_variance:.4f}")

    # ── 模型 ──
    model = Model(
        args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens,
        args.num_embeddings, args.embedding_dim,
        args.commitment_cost, args.decay, args.downsample
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_recon_errors, val_recon_errors = [], []
    train_perplexities = []

    for epoch in range(1, args.num_epochs + 1):
        # ── 训练 ──
        model.train()
        epoch_recon, epoch_perp = [], []

        for batch_idx, batch in enumerate(train_loader):
            # 兼容 (image, label) tuple（STL-10）和纯 image tensor（自定义）
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

            epoch_recon.append(recon_error.item())
            epoch_perp.append(perplexity.item())

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]"
                      f"  recon={recon_error.item():.4f}"
                      f"  perplexity={perplexity.item():.1f}")

        scheduler.step()
        avg_train_recon = np.mean(epoch_recon)
        avg_perp = np.mean(epoch_perp)
        train_recon_errors.append(avg_train_recon)
        train_perplexities.append(avg_perp)

        # ── 验证 ──
        model.eval()
        val_recon_sum = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_data = val_batch[0] if isinstance(val_batch, (tuple, list)) else val_batch
                val_data = val_data.to(device)
                _, val_recon, _ = model(val_data)
                val_recon_sum += (F.mse_loss(val_recon, val_data) / data_variance).item()
        avg_val_recon = val_recon_sum / len(val_loader)
        val_recon_errors.append(avg_val_recon)

        print(f"Epoch {epoch}/{args.num_epochs}"
              f"  train_recon={avg_train_recon:.4f}"
              f"  val_recon={avg_val_recon:.4f}"
              f"  perplexity={avg_perp:.1f}")

        # ── 定期保存重建样图 ──
        if epoch % args.save_interval == 0 or epoch == args.num_epochs:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_data = sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
                sample_data = sample_data.to(device)
                _, sample_recon, _ = model(sample_data)
            save_comparison(
                sample_data, sample_recon,
                f'results/large_reconstruction_epoch{epoch:03d}.png',
                nrow=4
            )
            print(f"  -> 重建样图已保存 epoch {epoch}")

    # ── 绘制训练曲线 ──
    if args.save_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(train_recon_errors, label='train')
        ax1.plot(val_recon_errors, label='val')
        ax1.set_title('Reconstruction Error (NMSE)')
        ax1.set_xlabel('epoch')
        ax1.legend()
        ax2.plot(train_perplexities)
        ax2.set_title('Codebook Perplexity')
        ax2.set_xlabel('epoch')
        plt.tight_layout()
        plt.savefig('results/large_training_curve.png')
        plt.close()
        print("训练曲线已保存到 results/large_training_curve.png")

    # ── 保存模型 ──
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, args.checkpoint)
    print(f"模型已保存到 {args.checkpoint}")

    return model


if __name__ == '__main__':
    train()
