"""
STL-10 数据集加载器
==================
STL-10 数据集: 96x96 RGB 图像，10 个类别
  - 训练集: 5000 张有标签图像
  - 测试集: 8000 张有标签图像
  - 无标签集: 100000 张图像

二进制文件格式:
  每张图像占 27648 字节 (96*96*3)，通道优先排列 (CHW):
    [R 通道: 96*96] [G 通道: 96*96] [B 通道: 96*96]
  标签文件: 每字节一个标签，取值 1-10（本代码转换为 0-9）
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# STL-10 类别名
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


def load_stl10_images(path: str) -> np.ndarray:
    """从 STL-10 二进制文件读取图像，返回 (N, 3, 96, 96) uint8 数组"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # 每张图像: 3*96*96 = 27648 字节
    n = data.shape[0] // (3 * 96 * 96)
    data = data.reshape(n, 3, 96, 96)
    return data


def load_stl10_labels(path: str) -> np.ndarray:
    """从 STL-10 标签二进制文件读取标签，返回 0-indexed int64 数组"""
    with open(path, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64) - 1  # 1-10 -> 0-9


class STL10Dataset(Dataset):
    """
    STL-10 有标签子集的 Dataset 封装。

    参数:
        root      - 数据集根目录（含 train_X.bin, train_y.bin 等）
        split     - 'train' 或 'test'
        transform - 可选的图像变换
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "test"), "split 必须为 'train' 或 'test'"
        self.transform = transform

        img_file = os.path.join(root, f"{split}_X.bin")
        lbl_file = os.path.join(root, f"{split}_y.bin")

        self.images = load_stl10_images(img_file)   # (N, 3, 96, 96) uint8
        self.labels = load_stl10_labels(lbl_file)   # (N,) int64

        assert len(self.images) == len(self.labels), "图像与标签数量不匹配"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (3, 96, 96) uint8

        # 转为 PIL 兼容的 (H, W, C) uint8
        img = np.transpose(img, (1, 2, 0))  # (96, 96, 3)

        if self.transform is not None:
            from PIL import Image
            img = self.transform(Image.fromarray(img))
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def build_transforms(img_size: int = 96, augment: bool = True):
    """构建训练/验证的图像预处理流程"""
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


def build_dataloaders(
    data_root: str,
    img_size: int = 96,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    构建 STL-10 训练 / 测试 DataLoader。

    返回:
        train_loader, val_loader
    """
    train_tf, val_tf = build_transforms(img_size, augment=True)

    train_ds = STL10Dataset(data_root, split="train", transform=train_tf)
    val_ds   = STL10Dataset(data_root, split="test",  transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] 训练集: {len(train_ds)} 张 | 测试集: {len(val_ds)} 张")
    print(f"[DataLoader] 图像尺寸: {img_size}x{img_size} | Batch: {batch_size}")
    return train_loader, val_loader


if __name__ == "__main__":
    data_root = "../vae/data/stl10_binary"
    train_loader, val_loader = build_dataloaders(data_root, img_size=96, batch_size=8, num_workers=0)

    imgs, labels = next(iter(train_loader))
    print(f"图像张量 shape: {imgs.shape}  dtype: {imgs.dtype}")
    print(f"标签张量 shape: {labels.shape} 示例: {labels.tolist()}")
    print(f"类别名: {[STL10_CLASSES[l] for l in labels.tolist()]}")
