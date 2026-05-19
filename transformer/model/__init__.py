"""
Transformer 核心模型包
======================
对外暴露最常用的构建接口和类，启动文件只需：

    from model import build_vit
    from model import TransformerEncoder, TransformerDecoder
    from model.attention import MultiHeadAttention, ...
"""

from .vit import build_vit, VisionTransformer, SwinViT
from .transformer_components import (
    FeedForwardNetwork,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    LinearAttention,
    WindowAttention,
    CrossAttention,
    RoPEMultiHeadAttention,
    make_causal_mask,
)
from .positional_encoding import (
    SinusoidalPE,
    LearnablePE,
    Learnable2DPE,
    RotaryPE,
    ALiBiPE,
)

__all__ = [
    # ViT 模型
    "build_vit", "VisionTransformer", "SwinViT",
    # Transformer 组件
    "FeedForwardNetwork",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
    "TransformerEncoder", "TransformerDecoder",
    # 注意力
    "ScaledDotProductAttention",
    "MultiHeadAttention", "MultiQueryAttention",
    "GroupedQueryAttention", "LinearAttention",
    "WindowAttention", "CrossAttention",
    "RoPEMultiHeadAttention", "make_causal_mask",
    # 位置编码
    "SinusoidalPE", "LearnablePE", "Learnable2DPE", "RotaryPE", "ALiBiPE",
]
