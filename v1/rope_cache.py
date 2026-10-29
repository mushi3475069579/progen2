# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
# Derived from the contest's models/progen/modeling_progen.py attention forward.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E10: share input-independent FP32 RoPE tables across the attention layers.

Only the positional table lookup changes in attention.forward. Weights,
attention arithmetic, dynamic per-request KV and generation rules are retained.
The public model files on disk and their module globals are left untouched.
"""

from __future__ import annotations

import torch
from models.progen.modeling_progen import ProGenAttention, apply_rotary_pos_emb, fixed_pos_embedding


class CachedRoPEAttention(ProGenAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Same calculation and ordering as the supplied ProGenAttention.forward.
        qkv = self.qkv_proj(hidden_states)
        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0
        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if seq_len > self._e10_rope_capacity:
            raise ValueError("RoPE position exceeds the initialized model context.")

        # This table depends on position and rotary dimension, never on input.
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = (self._e10_rope_sin, self._e10_rope_cos)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = (self._e10_rope_sin, self._e10_rope_cos)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


def install_rope_cache(model) -> dict:
    """Initialize tables before generation and retain every existing Parameter.

    Registers shared nonpersistent buffers. No input, generated suffix or
    effective KV is stored here. Installation is performed once on a fresh model.
    """
    attentions = [module for module in model.modules() if isinstance(module, ProGenAttention)]
    if not attentions or any(type(module) is not ProGenAttention for module in attentions):
        raise RuntimeError("E10 requires the supplied, unmodified ProGenAttention implementation.")
    capacity = int(model.config.max_position_embeddings)
    if capacity <= 0:
        raise ValueError("Invalid context capacity.")

    tables = {}
    before_parameters = tuple(id(parameter) for parameter in model.parameters())
    with torch.inference_mode():
        for attention in attentions:
            device = attention.qkv_proj.weight.device
            if attention.qkv_proj.weight.dtype != torch.float32:
                raise RuntimeError("E10 is an FP32-only comparison; do not combine precision changes.")
            dim = attention.head_dim if attention.rotary_dim is None else min(
                int(attention.rotary_dim), attention.head_dim
            )
            if dim <= 0 or dim % 2:
                raise ValueError("Unsupported rotary dimension.")
            key = (device.type, device.index, dim)
            if key not in tables:
                probe = torch.empty((1, 1, 1, dim), device=device, dtype=torch.float32)
                # Use the original formula, dtype, CPU frequency construction
                # and device sine/cosine, retaining its numerical behavior.
                sin, cos = fixed_pos_embedding(probe, 1, seq_len=capacity)
                if sin.dtype != torch.float32 or cos.dtype != torch.float32:
                    raise RuntimeError("Original RoPE tables must be FP32.")
                # A small input-independent numerical check runs at init only.
                lengths = sorted({1, max(1, capacity // 4), max(1, capacity // 2), capacity - 1, capacity})
                for length in lengths:
                    if length <= 0:
                        continue
                    ref_sin, ref_cos = fixed_pos_embedding(probe, 1, seq_len=length)
                    if not torch.equal(sin[:length], ref_sin) or not torch.equal(cos[:length], ref_cos):
                        raise RuntimeError("Cached RoPE differs from the original positional table.")
                tables[key] = (sin, cos)

            sin, cos = tables[key]
            attention.register_buffer("_e10_rope_sin", sin, persistent=False)
            attention.register_buffer("_e10_rope_cos", cos, persistent=False)
            attention._e10_rope_capacity = capacity
            # Change only the Python implementation of forward on this model's
            # existing objects; parameters and module structure are preserved.
            attention.__class__ = CachedRoPEAttention

    if before_parameters != tuple(id(parameter) for parameter in model.parameters()):
        raise RuntimeError("E10 must not replace or remove any model parameters.")
    return {
        "attention_layers": len(attentions),
        "shared_tables": len(tables),
        "capacity": capacity,
        "dtype": "fp32",
        "table_bytes": sum(
            sin.numel() * sin.element_size() + cos.numel() * cos.element_size()
            for sin, cos in tables.values()
        ),
    }
