import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc


def stochastic_depth(x, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Apply stochastic depth (drop path) to input tensor.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping the path
        training: Whether in training mode
        scale_by_keep: Whether to scale by keep probability

    Returns:
        Tensor with stochastic depth applied
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class StochasticDepth(nn.Module):
    """Stochastic depth layer for regularization."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(StochasticDepth, self).__init__()
        self.drop_probability = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return stochastic_depth(
            x, self.drop_probability, self.training, self.scale_by_keep
        )


def _create_tuple_converter(n):
    """Create a function that converts input to n-tuple."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _create_tuple_converter(2)


class FeatureMerger(nn.Module):
    """
    Fuses two feature maps through concatenation and convolution.

    Args:
        channels: Number of channels in each input feature map
    """

    def __init__(self, channels):
        super(FeatureMerger, self).__init__()
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, feature_map1, feature_map2):
        """
        Fuse two feature maps.

        Args:
            feature_map1: First feature map
            feature_map2: Second feature map

        Returns:
            Fused feature map
        """
        concatenated = torch.cat([feature_map1, feature_map2], dim=1)
        fused = self.fusion_conv(concatenated)
        fused = self.batch_norm(fused)
        fused = self.activation(fused)
        return fused


class FeedForwardNetwork(nn.Module):
    """
    Multi-layer perceptron used in transformer blocks.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden layer dimension (default: same as input)
        out_features: Output feature dimension (default: same as input)
        act_layer: Activation layer class
        drop: Dropout probability
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.dropout1 = nn.Dropout(drop_probs[0])
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """Forward pass through MLP."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class SelfAttentionLayer(nn.Module):
    """
    Standard self-attention mechanism.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Scale factor for attention scores
        attn_drop: Attention dropout probability
        proj_drop: Projection dropout probability
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale_factor = qk_scale or head_dim**-0.5

        self.qkv_projection = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.output_projection = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_drop)

    def forward(self, x):
        """Compute self-attention."""
        batch_size, seq_len, channels = x.shape

        qkv = (
            self.qkv_projection(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_scores = (query @ key.transpose(-2, -1)) * self.scale_factor
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        output = (
            (attention_weights @ value)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, channels)
        )
        output = self.output_projection(output)
        output = self.proj_dropout(output)

        return output


class CrossAttnModule(nn.Module):
    """
    Cross-attention block with self-attention, cross-attention, and FFN.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Scale factor for attention scores
        drop: Dropout probability
        attn_drop: Attention dropout probability
        drop_path: Stochastic depth probability
        act_layer: Activation layer class
        norm_layer: Normalization layer class
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # Self-attention components
        self.norm_self = norm_layer(dim)
        self.self_attention = SelfAttentionLayer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path_self = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Cross-attention components
        self.norm_cross = norm_layer(dim)
        self.cross_attention = FlashCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path_cross = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Feed-forward network components
        self.norm_ffn = norm_layer(dim)
        self.feed_forward = FeedForwardNetwork(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path_ffn = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, query_features, key_value_features):
        """
        Forward pass through cross-attention block.

        Args:
            query_features: Query features from current frame
            key_value_features: Key-value features from reference frame

        Returns:
            Updated query features
        """
        # Self-attention with residual
        query_features = query_features + self.drop_path_self(
            self.self_attention(self.norm_self(query_features))
        )

        # Cross-attention with residual
        query_features = query_features + self.drop_path_cross(
            self.cross_attention(self.norm_cross(query_features), key_value_features)
        )

        # Feed-forward network with residual
        query_features = query_features + self.drop_path_ffn(
            self.feed_forward(self.norm_ffn(query_features))
        )

        return query_features


class FlashCrossAttention(nn.Module):
    """
    Memory-efficient cross-attention using Flash Attention.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in projections
        qk_scale: Scale factor for attention scores
        attn_drop: Attention dropout probability
        proj_drop: Projection dropout probability
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale_factor = qk_scale or head_dim**-0.5

        self.query_projection = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_projection = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_projection = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop_prob = attn_drop
        self.output_projection = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_drop)

    def forward(self, query_input, key_value_input):
        """
        Compute cross-attention between query and key-value inputs.

        Args:
            query_input: Query tensor [B, N_q, C]
            key_value_input: Key-value tensor [B, N_kv, C]

        Returns:
            Attention output [B, N_q, C]
        """
        batch_size, query_len, channels = query_input.shape
        kv_len = key_value_input.shape[1]

        # Project and reshape for multi-head attention
        query = (
            self.query_projection(query_input)
            .reshape(batch_size, query_len, self.num_heads, channels // self.num_heads)
            .transpose(1, 2)
        )

        key = (
            self.key_projection(key_value_input)
            .reshape(batch_size, kv_len, self.num_heads, channels // self.num_heads)
            .transpose(1, 2)
        )

        value = (
            self.value_projection(key_value_input)
            .reshape(batch_size, kv_len, self.num_heads, channels // self.num_heads)
            .transpose(1, 2)
        )

        # Apply Flash Attention (memory-efficient)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
            scale=self.scale_factor,
        )

        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, query_len, channels)
        output = self.output_projection(output)
        output = self.proj_dropout(output)

        return output
