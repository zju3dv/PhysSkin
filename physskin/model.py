# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# Licensed under the Apache License, Version 2.0.
#
# Portions of this file are adapted from Kaolin Simplicits and Michelangelo.
# See THIRD_PARTY_NOTICES.md.

"""Point-shape-handle cross-attention network used by the RigNet model."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlockFC(nn.Module):
    """Fully connected residual block."""

    def __init__(self, size_in: int, size_out: Optional[int] = None,
                 size_h: Optional[int] = None):
        super().__init__()
        size_out = size_in if size_out is None else size_out
        size_h = min(size_in, size_out) if size_h is None else size_h

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.shortcut = (
            None if size_in == size_out
            else nn.Linear(size_in, size_out, bias=False)
        )
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.fc_0(F.elu(x))
        dx = self.fc_1(F.elu(net))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return shortcut + dx


class PointEmbed(nn.Module):
    """Fourier point embedding from 3DShape2VecSet."""

    def __init__(self, hidden_dim: int = 48, dim: int = 128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        frequencies = (
            torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        )
        basis = torch.stack([
            torch.cat([frequencies, torch.zeros_like(frequencies), torch.zeros_like(frequencies)]),
            torch.cat([torch.zeros_like(frequencies), frequencies, torch.zeros_like(frequencies)]),
            torch.cat([torch.zeros_like(frequencies), torch.zeros_like(frequencies), frequencies]),
        ])
        self.register_buffer("basis", basis)
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(inputs: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("bnd,de->bne", inputs, basis)
        return torch.cat([projections.sin(), projections.cos()], dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze_batch = inputs.dim() == 2
        if squeeze_batch:
            inputs = inputs.unsqueeze(0)
        embedded = self.mlp(
            torch.cat([self.embed(inputs, self.basis), inputs], dim=2)
        )
        return embedded.squeeze(0) if squeeze_batch else embedded


class PointwiseCrossAttentionBlock(nn.Module):
    def __init__(self, width: int = 768, heads: int = 12,
                 mlp_ratio: float = 1.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        query = query + attn_out
        return query + self.ffn(self.ffn_norm(query))


class PhysSkinPointShapeHandleModel(nn.Module):
    """PhysSkin architecture used by the RigNet checkpoint.

    The last output channel is a fixed normalized rigid mode. ``num_handles``
    specifies the learned channels, so the output has ``num_handles + 1``
    channels.
    """

    def __init__(
        self,
        spatial_dimensions: int = 3,
        layer_width: int = 64,
        num_handles: int = 12,
        num_layers: int = 8,
        multires: int = 10,
        num_self_attn: int = 1,
        num_handle_tokens: Optional[int] = None,
        device: Optional[torch.device] = torch.device("cuda"),
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        # Kept in the signature for checkpoint/config compatibility.
        del spatial_dimensions, multires, device, dtype
        self.latent_feature = None
        latent_size = 768
        num_handle_tokens = (
            num_handles if num_handle_tokens is None else num_handle_tokens
        )

        self.point_embed = PointEmbed(hidden_dim=48, dim=latent_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=12,
            dim_feedforward=768,
            activation="relu",
            batch_first=True,
        )
        self.latent_self_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=num_self_attn
        )
        self.learnable_query = nn.Parameter(
            torch.randn(1, num_handle_tokens, latent_size)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_size,
            nhead=12,
            dim_feedforward=768,
            activation="relu",
            batch_first=True,
        )
        self.latent_cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.point_shape_cross_attn = PointwiseCrossAttentionBlock(
            width=latent_size, heads=12
        )
        self.point_handle_cross_attn = PointwiseCrossAttentionBlock(
            width=latent_size, heads=12
        )

        self.input_ch = latent_size + latent_size
        self.skips = []
        self.linear = nn.ModuleList(
            [ResnetBlockFC(self.input_ch, layer_width)]
            + [
                ResnetBlockFC(layer_width, layer_width)
                if i not in self.skips
                else ResnetBlockFC(layer_width + self.input_ch, layer_width)
                for i in range(num_layers - 1)
            ]
        )
        self.skin_weights = ResnetBlockFC(layer_width, num_handles)

    def set_latent_feature(self, latent_feature: torch.Tensor) -> None:
        self.latent_feature = latent_feature

    def forward_raw_weights(self, points: torch.Tensor) -> torch.Tensor:
        latent_feature = self.latent_feature
        if latent_feature is None:
            raise RuntimeError("latent_feature must be set before forward")

        squeeze_batch = points.dim() == 2
        if squeeze_batch:
            points = points.unsqueeze(0)
        if latent_feature.dim() == 2:
            latent_feature = latent_feature.unsqueeze(0)

        batch_size = points.shape[0]
        if latent_feature.size(0) == 1 and batch_size > 1:
            latent_feature = latent_feature.expand(batch_size, -1, -1)

        point_embedding = self.point_embed(points)
        shape_context = self.latent_self_attn(latent_feature)
        handle_latent = self.latent_cross_attn(
            tgt=self.learnable_query.expand(batch_size, -1, -1),
            memory=shape_context,
        )
        point_shape_features = self.point_shape_cross_attn(
            query=point_embedding, memory=shape_context
        )
        point_handle_features = self.point_handle_cross_attn(
            query=point_shape_features, memory=handle_latent
        )

        mlp_input = torch.cat([point_embedding, point_handle_features], dim=-1)
        hidden = mlp_input
        for index, layer in enumerate(self.linear):
            hidden = F.elu(layer(hidden))
            if index in self.skips:
                hidden = torch.cat([mlp_input, hidden], -1)
        weights = self.skin_weights(hidden)
        return weights.squeeze(0) if squeeze_batch else weights

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        raw_weights = self.forward_raw_weights(points)
        squeeze_batch = raw_weights.dim() == 2
        weights = raw_weights.unsqueeze(0) if squeeze_batch else raw_weights
        num_points = weights.shape[1]
        weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-8)

        rigid = torch.ones(
            num_points, 1, device=weights.device, dtype=weights.dtype
        )
        rigid = rigid / rigid.norm(dim=0, keepdim=True)
        rigid = rigid.detach().unsqueeze(0).expand(weights.shape[0], num_points, 1)
        output = torch.cat([weights, rigid], dim=2)
        return output.squeeze(0) if squeeze_batch else output


# Compatibility alias used by the training configuration.
PhysSkin_PointShapeHandle_CrossOnly_Model = PhysSkinPointShapeHandleModel
