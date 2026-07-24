"""在一个分段非线性 toy task 上训练最小 MoE。"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# 同时支持两种入口：在 MOE/ 内直接执行，或在仓库根目录执行 python -m MOE.demo_train。
try:
    from .moe import SparseMoE
except ImportError:
    from moe import SparseMoE


def make_batch(batch_size: int, seq_len: int, device: torch.device):
    """四种输入区间对应四种函数，鼓励不同 expert 学习不同子任务。"""
    x = torch.rand(batch_size, seq_len, 1, device=device) * 4 - 2  # [-2, 2]
    y = torch.where(
        x < -1,
        x.square() + 0.5,
        torch.where(x < 0, torch.sin(3 * x), torch.where(x < 1, x.pow(3), torch.cos(3 * x))),
    )
    return x, y


class ToyMoEModel(nn.Module):
    """由多个残差 MoE 层组成的 toy 模型，可直接观察逐层路由。"""

    def __init__(self, d_model: int, num_experts: int, top_k: int, num_layers: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.moes = nn.ModuleList([
            SparseMoE(
                d_model=d_model,
                d_hidden=4 * d_model,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=1.25,
                aux_loss_weight=0.01,
            )
            for _ in range(num_layers)
        ])
        self.aux_loss_weight = self.moes[0].aux_loss_weight
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, return_routing: bool = False):
        hidden = self.input_proj(x)
        total_aux_loss = hidden.new_zeros(())
        routing_infos = []
        for moe in self.moes:
            moe_output, aux_loss, info = moe(hidden, return_routing=return_routing)
            hidden = hidden + moe_output
            total_aux_loss = total_aux_loss + aux_loss
            if return_routing:
                routing_infos.append(info)
        return self.output_proj(hidden), total_aux_loss, routing_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="训练教学用最小 MoE")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1, help="堆叠的残差 MoE 层数")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument(
        "--save-routing", default=None,
        help="训练结束后保存每层 token→expert 热力图，例如 output/trained_routing.png",
    )
    parser.add_argument(
        "--save-routing-json", default=None,
        help="训练结束后导出给 web 交互页面的真实路由 JSON",
    )
    args = parser.parse_args()

    if not 1 <= args.top_k <= args.num_experts:
        parser.error("--top-k 必须满足 1 <= top-k <= num-experts")

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = ToyMoEModel(
        args.d_model, args.num_experts, args.top_k, args.num_layers
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"device={device}, layers={args.num_layers}, experts={args.num_experts}, "
        f"top_k={args.top_k}"
    )
    for step in range(1, args.steps + 1):
        x, target = make_batch(args.batch_size, args.seq_len, device)
        prediction, aux_loss, _ = model(x)
        task_loss = F.mse_loss(prediction, target)
        # 注意：aux_loss_weight 放在 MoE 配置中，显式写在这里便于理解总目标。
        loss = task_loss + model.aux_loss_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(
                f"step={step:>3d} task={task_loss.item():.4f} "
                f"aux={aux_loss.item():.4f} total={loss.item():.4f}"
            )

    if args.save_routing or args.save_routing_json:
        try:
            from .visualize_routing import plot_routing, save_routing_trace
        except ImportError:
            from visualize_routing import plot_routing, save_routing_trace

        model.eval()
        x, _ = make_batch(batch_size=1, seq_len=args.seq_len, device=device)
        with torch.no_grad():
            _, _, routing_infos = model(x, return_routing=True)
        if args.save_routing:
            plot_routing(routing_infos, args.num_experts, args.save_routing)
            print(f"已保存训练后路由图: {args.save_routing}")
        if args.save_routing_json:
            save_routing_trace(routing_infos, args.num_experts, args.save_routing_json)
            print(f"已保存交互页面路由数据: {args.save_routing_json}")


if __name__ == "__main__":
    main()
