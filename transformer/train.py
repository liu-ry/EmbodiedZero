"""
训练与验证主流程
================
基于 STL-10 数据集，对多种 ViT 模型进行训练和评估。

用法示例:
  # 训练 vit_small（默认配置）
  python train.py

  # 指定模型和超参数
  python train.py --model vit_small --epochs 50 --lr 3e-4 --batch_size 64

  # 对比多个模型
  python train.py --model vit_tiny --epochs 30
  python train.py --model swin_small --epochs 30

支持特性:
  - 余弦退火 + Warmup 学习率调度
  - 标签平滑 (Label Smoothing)
  - Mixup 数据增强
  - 梯度裁剪
  - 最佳 checkpoint 保存
  - TensorBoard / 文本日志
"""

import os
import sys
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 将当前目录加入路径（使相对导入可用）
sys.path.insert(0, str(Path(__file__).parent))

from utils import build_dataloaders, STL10_CLASSES
from model import build_vit


# ──────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────
def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", "%Y-%m-%d %H:%M:%S")

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────
# Mixup 数据增强
# ──────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Mixup: 将两个样本线性混合。
    参考: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

    返回: (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 的混合损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────
# 余弦退火 + Warmup 调度器
# ──────────────────────────────────────────
class CosineWarmupScheduler(optim.lr_scheduler.LambdaLR):
    """
    带线性 Warmup 的余弦退火学习率调度。

    - Warmup 阶段: 从 0 线性增长到 base_lr
    - 退火阶段: 按余弦曲线从 base_lr 降至 min_lr

    参数:
        optimizer  - 优化器
        warmup_steps - Warmup 步数
        total_steps  - 总训练步数
        min_lr_ratio - 最小学习率比例 (min_lr = base_lr * ratio)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.05,
    ):
        self.warmup = warmup_steps
        self.total  = total_steps
        self.min_r  = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        super().__init__(optimizer, lr_lambda)


# ──────────────────────────────────────────
# 单 epoch 训练
# ──────────────────────────────────────────
def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    scheduler,
    device:     torch.device,
    epoch:      int,
    args:       argparse.Namespace,
    logger:     logging.Logger,
    writer:     SummaryWriter,
    global_step: int,
) -> Tuple[float, float, int]:
    """
    训练一个 epoch，返回 (avg_loss, top1_acc, global_step)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total   = 0
    t0 = time.time()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup 增强
        if args.mixup_alpha > 0:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, args.mixup_alpha)
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        scheduler.step()

        # 统计（使用原始标签计算准确率）
        total_loss += loss.item() * imgs.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)

        global_step += 1
        if writer:
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch:3d}] Step [{batch_idx:4d}/{len(loader)}] "
                f"Loss: {loss.item():.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.6f}  "
                f"Elapsed: {elapsed:.1f}s"
            )

    avg_loss = total_loss / total
    acc      = 100.0 * correct / total
    return avg_loss, acc, global_step


# ──────────────────────────────────────────
# 验证
# ──────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float, float]:
    """
    在验证集上评估模型。
    返回 (avg_loss, top1_acc, top3_acc)
    """
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss   += loss.item() * imgs.size(0)
        total        += labels.size(0)

        # Top-1
        _, pred1 = logits.max(1)
        top1_correct += pred1.eq(labels).sum().item()

        # Top-3
        _, pred3 = logits.topk(3, dim=1)
        top3_correct += pred3.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss  = total_loss / total
    top1_acc  = 100.0 * top1_correct / total
    top3_acc  = 100.0 * top3_correct / total
    return avg_loss, top1_acc, top3_acc


# ──────────────────────────────────────────
# 每类准确率
# ──────────────────────────────────────────
@torch.no_grad()
def per_class_accuracy(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int = 10,
) -> dict:
    """计算每个类别的准确率"""
    model.eval()
    class_correct = [0] * n_classes
    class_total   = [0] * n_classes

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        _, preds = model(imgs).max(1)
        for c in range(n_classes):
            mask = labels.eq(c)
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    result = {}
    for c in range(n_classes):
        if class_total[c] > 0:
            result[STL10_CLASSES[c]] = 100.0 * class_correct[c] / class_total[c]
        else:
            result[STL10_CLASSES[c]] = 0.0
    return result


# ──────────────────────────────────────────
# 保存 / 加载 Checkpoint
# ──────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_acc", 0.0)


# ──────────────────────────────────────────
# 主训练流程
# ──────────────────────────────────────────
def main(args: argparse.Namespace):
    # ── 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 日志和 TensorBoard
    run_name = f"{args.model}_bs{args.batch_size}_lr{args.lr}_e{args.epochs}"
    log_dir  = os.path.join(args.output_dir, run_name)
    logger   = setup_logger(log_dir)
    writer   = SummaryWriter(log_dir=os.path.join(log_dir, "tb")) if args.tensorboard else None

    logger.info(f"{'='*60}")
    logger.info(f"  模型: {args.model}")
    logger.info(f"  数据集: STL-10  |  图像尺寸: {args.img_size}x{args.img_size}")
    logger.info(f"  设备: {device}")
    logger.info(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    logger.info(f"{'='*60}")

    # ── 数据
    train_loader, val_loader = build_dataloaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── 模型
    model = build_vit(args.model, img_size=args.img_size, n_classes=10, dropout=args.dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数量: {n_params/1e6:.2f}M")

    # ── 损失函数（Label Smoothing）
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── 优化器（AdamW，Transformer 推荐）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── 学习率调度（余弦 Warmup）
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05)

    logger.info(f"总训练步数: {total_steps}  |  Warmup 步数: {warmup_steps}")

    # ── 恢复 checkpoint
    start_epoch = 0
    best_acc    = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"从 checkpoint 恢复: {args.resume}，Epoch {start_epoch}，最优 acc: {best_acc:.2f}%")

    # ── 训练循环
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_t0 = time.time()

        # 训练
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scheduler, device, epoch, args, logger, writer, global_step
        )

        # 验证
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_t0

        logger.info(
            f"\n{'─'*60}\n"
            f"  Epoch {epoch:3d}/{args.epochs}  耗时: {epoch_time:.1f}s\n"
            f"  Train  →  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%\n"
            f"  Val    →  Loss: {val_loss:.4f}  Top-1: {val_top1:.2f}%  Top-3: {val_top3:.2f}%\n"
            f"{'─'*60}"
        )

        if writer:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/train_acc",  train_acc,  epoch)
            writer.add_scalar("epoch/val_loss",   val_loss,   epoch)
            writer.add_scalar("epoch/val_top1",   val_top1,   epoch)
            writer.add_scalar("epoch/val_top3",   val_top3,   epoch)

        # 保存最佳 checkpoint
        is_best = val_top1 > best_acc
        if is_best:
            best_acc = val_top1
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_acc": best_acc},
                path=os.path.join(log_dir, "best.pth")
            )
            logger.info(f"  ★ 最佳模型已保存（val_top1={best_acc:.2f}%）")

        # 定期保存
        if epoch % args.save_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_acc": best_acc},
                path=os.path.join(log_dir, f"epoch_{epoch:03d}.pth")
            )

    # ── 训练结束，汇报每类准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"训练完成！最佳验证 Top-1: {best_acc:.2f}%")

    # 加载最佳模型评估
    best_ckpt = os.path.join(log_dir, "best.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model)
        logger.info("各类别准确率（最佳模型）:")
        cls_acc = per_class_accuracy(model, val_loader, device)
        for cls_name, acc in cls_acc.items():
            logger.info(f"    {cls_name:12s}: {acc:.2f}%")

    if writer:
        writer.close()
    logger.info("Done.")


# ──────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer (ViT) STL-10 训练脚本")

    # 数据
    parser.add_argument("--data_root",    type=str, default="../vae/data/stl10_binary",
                        help="STL-10 二进制数据集根目录")
    parser.add_argument("--img_size",     type=int, default=96,
                        help="输入图像尺寸（正方形）")
    parser.add_argument("--num_workers",  type=int, default=4)

    # 模型
    parser.add_argument("--model",        type=str, default="vit_small",
                        choices=["vit_tiny", "vit_small", "vit_base",
                                 "vit_sinpe", "vit_2dpe", "swin_small"],
                        help="模型架构名称")
    parser.add_argument("--dropout",      type=float, default=0.1)

    # 训练
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--clip_grad",    type=float, default=1.0,
                        help="梯度裁剪最大范数，0=不裁剪")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha",  type=float, default=0.4,
                        help="Mixup alpha，0=不使用 Mixup")

    # 日志
    parser.add_argument("--output_dir",   type=str, default="./output")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval",type=int, default=10)
    parser.add_argument("--tensorboard",  action="store_true", default=True)
    parser.add_argument("--resume",       type=str, default="",
                        help="从 checkpoint 继续训练")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
