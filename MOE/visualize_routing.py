"""将多层 MoE 的 token -> expert 路由绘制为热力图。"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import torch

try:
    from .moe import MoETransformerStack, RoutingInfo
except ImportError:
    from moe import MoETransformerStack, RoutingInfo


def plot_routing(
    routing_infos: Sequence[RoutingInfo],
    num_experts: int,
    output_path: str,
    batch_index: int = 0,
    token_labels: Optional[Sequence[str]] = None,
) -> None:
    """保存 token × expert 路由热力图；每一个子图代表一个 MoE 层。

    单元格数值是归一化后的 gate 权重。Top-2 时同一 token 行会有两个非零格，
    因而能同时看清“选了谁”与“两个 expert 各占多少权重”。
    """
    if not routing_infos:
        raise ValueError("routing_infos 为空；模型前向时请传入 return_routing=True")
    if any(info.token_experts is None or info.token_weights is None for info in routing_infos):
        raise ValueError("缺少 token 路由轨迹；模型前向时请传入 return_routing=True")

    num_tokens = routing_infos[0].token_experts.size(1)
    if token_labels is None:
        token_labels = [f"token {i}" for i in range(num_tokens)]
    if len(token_labels) != num_tokens:
        raise ValueError("token_labels 的长度必须等于序列长度")

    fig_height = max(3.2 * len(routing_infos), 5.0)
    fig, axes = plt.subplots(len(routing_infos), 1, figsize=(8.0, fig_height), squeeze=False)

    for layer_id, info in enumerate(routing_infos):
        expert_ids = info.token_experts[batch_index].cpu()  # (T, k)
        weights = info.token_weights[batch_index].float().cpu()
        matrix = torch.zeros(num_tokens, num_experts)
        matrix.scatter_add_(1, expert_ids, weights)

        ax = axes[layer_id, 0]
        image = ax.imshow(matrix.numpy(), cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(
            f"Layer {layer_id}: load={info.expert_load.tolist()}, "
            f"dropped={info.dropped_assignments}",
            loc="left",
        )
        ax.set_xlabel("Expert")
        ax.set_ylabel("Token")
        ax.set_xticks(range(num_experts), [f"E{i}" for i in range(num_experts)])
        ax.set_yticks(range(num_tokens), token_labels)

        # 小序列直接把 gate 权重写在格子中，阅读路径会比只看颜色更直观。
        if num_tokens <= 32 and num_experts <= 16:
            for token_id in range(num_tokens):
                for expert_id in range(num_experts):
                    value = matrix[token_id, expert_id].item()
                    if value > 0:
                        ax.text(expert_id, token_id, f"{value:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="normalized gate weight")

    fig.suptitle("Token → Expert routing in every MoE layer", y=1.01, fontsize=14)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_routing_trace(
    routing_infos: Sequence[RoutingInfo],
    num_experts: int,
    output_path: str,
    batch_index: int = 0,
) -> None:
    """导出给 `web/` 交互页面读取的 JSON，不保存梯度或模型权重。"""
    if not routing_infos:
        raise ValueError("routing_infos 为空；模型前向时请传入 return_routing=True")

    layers = []
    for layer_id, info in enumerate(routing_infos):
        if info.token_experts is None or info.token_weights is None:
            raise ValueError("缺少 token 路由轨迹；模型前向时请传入 return_routing=True")
        layer = {
            "layer": layer_id,
            "capacity": info.capacity,
            "dropped_assignments": info.dropped_assignments,
            "expert_load": info.expert_load.cpu().tolist(),
            "expert_importance": info.expert_importance.float().cpu().tolist(),
            "token_experts": info.token_experts[batch_index].cpu().tolist(),
            "token_weights": info.token_weights[batch_index].float().cpu().tolist(),
        }
        if info.router_probabilities is not None:
            layer["router_probabilities"] = (
                info.router_probabilities[batch_index].float().cpu().tolist()
            )
        layers.append(layer)

    trace = {
        "schema_version": 1,
        "source": "EmbodiedZero/MOE",
        "num_experts": num_experts,
        "num_tokens": len(layers[0]["token_experts"]),
        "top_k": len(layers[0]["token_experts"][0]),
        "layers": layers,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制每层 MoE 的 token 路由热力图")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", default="output/routing.png")
    parser.add_argument("--trace-json", default=None, help="可选：导出给 web 可视化页面的 JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = MoETransformerStack(
        args.layers, args.d_model, args.heads, args.experts, args.top_k
    ).eval()
    tokens = torch.randn(1, args.tokens, args.d_model)

    with torch.no_grad():
        _, _, routing_infos = model(tokens, return_routing=True)
    plot_routing(routing_infos, args.experts, args.output)
    print(f"已保存路由图: {args.output}")
    if args.trace_json:
        save_routing_trace(routing_infos, args.experts, args.trace_json)
        print(f"已保存交互页面路由数据: {args.trace_json}")
    print("提示：此脚本默认使用随机初始化模型；接入训练后的模型时，图才反映已学习到的专家分工。")


if __name__ == "__main__":
    main()
