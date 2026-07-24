"""
最小稀疏 Mixture-of-Experts（MoE）
=================================

本文件刻意只依赖 PyTorch，并把一个 MoE 层拆成可单独阅读的组件：

    token x
      ├─ Router：为每个 expert 计算概率
      ├─ Top-k：只选择 k 个 expert
      ├─ Dispatch：按 expert 收集 token（可施加容量限制）
      ├─ Expert：各自独立的 FFN
      └─ Combine：按 router 权重加权累加回原 token 位置

输入和输出都是 (batch, sequence, d_model)，因此可以直接替换 Transformer
里的普通 FFN。本实现以可读性优先，使用 Python 循环分发；生产环境通常会
用 fused kernel / all-to-all 通信来加速同样的逻辑。
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """一个专家：结构与 Transformer 中常见的两层 FFN 相同。"""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """把每个 token 映射为 `num_experts` 个路由概率。"""

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            logits: (num_tokens, num_experts)，softmax 前的分数
            probs : (num_tokens, num_experts)，每行和为 1 的路由概率

        为了半精度训练的数值稳定性，softmax 始终在 float32 中计算。
        """
        logits = self.gate(x)
        probs = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        return logits, probs


@dataclass
class RoutingInfo:
    """供教学和监控使用的路由统计信息。"""

    expert_load: torch.Tensor       # 每个 expert 实际接收的 assignment 数
    expert_importance: torch.Tensor # 每个 expert 的平均 router 概率
    dropped_assignments: int        # 因容量限制被丢弃的 assignment 数
    capacity: Optional[int]         # 每个 expert 的最大 assignment 数
    token_experts: Optional[torch.Tensor] = None  # (B, T, k)，仅 return_routing=True 时保存
    token_weights: Optional[torch.Tensor] = None  # (B, T, k)，对应的归一化 gate 权重
    router_probabilities: Optional[torch.Tensor] = None  # (B, T, E)，Router 的完整 softmax 概率

    def as_dict(self) -> Dict[str, object]:
        return {
            "expert_load": self.expert_load.detach(),
            "expert_importance": self.expert_importance.detach(),
            "dropped_assignments": self.dropped_assignments,
            "capacity": self.capacity,
            "token_experts": self.token_experts,
            "token_weights": self.token_weights,
            "router_probabilities": self.router_probabilities,
        }


class SparseMoE(nn.Module):
    """教学向 Top-k 稀疏 MoE 层。

    参数:
        d_model: token 特征维度。
        d_hidden: 每个 expert 的 FFN 隐层维度；默认 4 * d_model。
        num_experts: expert 数量 E。
        top_k: 每个 token 激活的 expert 数量 k（必须不大于 E）。
        capacity_factor: 若不为 None，则启用容量限制。每个 expert 容量为
            ceil(capacity_factor * token_count * top_k / num_experts)。
            容量限制能避免单个 expert 过载，但会造成部分 assignment 被丢弃。
        aux_loss_weight: 负载均衡辅助损失的系数。训练时总损失通常为
            task_loss + aux_loss_weight * aux_loss。

    `forward` 返回 `(output, aux_loss, routing_info)`：
    - output: 与输入形状相同；
    - aux_loss: Switch Transformer 风格的均衡项，越小表示路由越均匀；
    - routing_info: 可用于打印、可视化和排查 expert 塌缩。
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
        capacity_factor: Optional[float] = 1.25,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k 必须满足 1 <= top_k <= num_experts")
        if capacity_factor is not None and capacity_factor <= 0:
            raise ValueError("capacity_factor 必须为正数或 None")

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        d_hidden = d_hidden or 4 * d_model

        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden, dropout) for _ in range(num_experts)]
        )

    def _capacity(self, num_tokens: int) -> Optional[int]:
        if self.capacity_factor is None:
            return None
        # top_k 个选择都会占用容量；不乘 top_k 会让 top-2 默认丢掉约一半路由。
        return math.ceil(
            self.capacity_factor * num_tokens * self.top_k / self.num_experts
        )

    def forward(
        self, x: torch.Tensor, return_routing: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingInfo]:
        if x.ndim != 3:
            raise ValueError("x 的形状应为 (batch, sequence, d_model)")

        batch_size, seq_len, d_model = x.shape
        tokens = x.reshape(-1, d_model)                  # (N, d_model)
        num_tokens = tokens.size(0)
        _, probs = self.router(tokens)                   # (N, E)

        # topk_probs 是原始概率；重新归一化后，每个 token 的 k 个权重和为 1。
        topk_probs, topk_experts = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # f_i：实际 top-k 选择 expert i 的频率；p_i：i 的平均软概率。
        # E * sum_i(f_i * p_i) 最小值为 1；router 全压向一个 expert 时会变大。
        assignment_mask = F.one_hot(topk_experts, self.num_experts).sum(dim=1)
        load_fraction = assignment_mask.float().mean(dim=0) / self.top_k
        importance = probs.float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(load_fraction * importance)

        output = torch.zeros_like(tokens)
        actual_load = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        capacity = self._capacity(num_tokens)
        dropped_assignments = 0

        # 下面就是 Dispatch -> Expert -> Combine。为了看清数据流，按 expert 循环；
        # 真实大规模 MoE 会将这一段替换为 grouped GEMM / 分布式 all-to-all。
        for expert_id, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(topk_experts == expert_id)
            if capacity is not None and token_indices.numel() > capacity:
                dropped_assignments += token_indices.numel() - capacity
                token_indices = token_indices[:capacity]
                topk_slots = topk_slots[:capacity]

            actual_load[expert_id] = token_indices.numel()
            if token_indices.numel() == 0:
                continue

            expert_output = expert(tokens.index_select(0, token_indices))
            gates = topk_weights[token_indices, topk_slots].unsqueeze(-1)
            # index_add_ 可把同一 token 来自 top-k experts 的结果安全地相加。
            output.index_add_(0, token_indices, expert_output * gates)

        # 路由轨迹默认不保存：真实训练中 (B, T, k) 的额外张量会占用显存。
        # 可视化、调试时传入 return_routing=True 即可保留它们。
        info = RoutingInfo(
            expert_load=actual_load,
            expert_importance=importance,
            dropped_assignments=dropped_assignments,
            capacity=capacity,
            token_experts=(
                topk_experts.detach().reshape(batch_size, seq_len, self.top_k)
                if return_routing else None
            ),
            token_weights=(
                topk_weights.detach().reshape(batch_size, seq_len, self.top_k)
                if return_routing else None
            ),
            router_probabilities=(
                probs.detach().reshape(batch_size, seq_len, self.num_experts)
                if return_routing else None
            ),
        )
        return output.reshape(batch_size, seq_len, d_model), aux_loss, info


class MoETransformerBlock(nn.Module):
    """展示 MoE 如何替换 Transformer Block 中的普通 FFN。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SparseMoE(
            d_model, num_experts=num_experts, top_k=top_k, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, return_routing: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingInfo]:
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        moe_output, aux_loss, info = self.moe(
            self.norm2(x), return_routing=return_routing
        )
        return x + moe_output, aux_loss, info


class MoETransformerStack(nn.Module):
    """多层 MoE Transformer，便于观察同一批 token 在每一层的路由变化。"""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerBlock(d_model, num_heads, num_experts, top_k, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self, x: torch.Tensor, return_routing: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, list[RoutingInfo]]:
        total_aux_loss = x.new_zeros(())
        routing_infos = []
        for layer in self.layers:
            x, aux_loss, info = layer(x, return_routing=return_routing)
            total_aux_loss = total_aux_loss + aux_loss
            if return_routing:
                routing_infos.append(info)
        return x, total_aux_loss, routing_infos


def _smoke_test() -> None:
    """无需数据集的最小形状检查：`python moe.py` 可直接运行。"""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    moe = SparseMoE(16, d_hidden=32, num_experts=4, top_k=2)
    y, aux_loss, info = moe(x)
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print(f"aux loss    : {aux_loss.item():.4f}")
    print("expert load :", info.expert_load.tolist())
    print("capacity    :", info.capacity)


if __name__ == "__main__":
    _smoke_test()
