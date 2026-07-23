#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors

"""FlashComm v1: prefill-stage TP communication fusion for dense models.

Ports the vllm-ascend FlashComm v1 design (``vllm_ascend/ops/linear_op.py``
``SequenceColumnParallelOp`` / ``SequenceRowParallelOp``) to the FL plugin:

* Row-parallel outputs (attention ``o_proj``, GDN ``out_proj``, MLP
  ``down_proj``) use the matmul-fused reduce-scatter
  ``torch_npu.npu_mm_reduce_scatter_base`` instead of ``GEMM + all_reduce``.
* The residual/hidden stream stays sharded on the token dim (1/TP per rank)
  between row- and column-parallel ops; add+RMSNorm therefore runs on 1/TP
  of the tokens.
* Column-parallel ops restore full tokens with an all-gather at their entry
  (one shared gather per decoder block per norm stage).
* Token counts are padded to a TP multiple at the model entry and unpadded
  at the exit (``Qwen3NextModel.forward`` wrapper).

Activation mirrors vllm-ascend: dense models, TP > 1, PP == 1, and
``num_tokens > 1000`` — i.e. prefill / long chunked-prefill batches only;
decode steps and small batches take the standard path. The decision is made
once per model forward (``_FLASH_STATE``) and additionally suppressed for
embeds-based dummy runs (input_ids=None) and inside dynamo-traced or
graph-captured forwards (``_flash_forbidden_context``), so the flash
collectives never enter traced/captured graphs. Set ``VLLM_FL_FLASHCOMM=0``
to disable. MoE (sparse) MLP blocks are not covered yet and keep the standard
path.
"""

import logging
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer, Qwen3NextModel

logger = logging.getLogger(__name__)

_TOKEN_THRESHOLD = 1000
_HCCL_NAME = None

# Single activation decision per model forward. ``_qwen3next_model_forward``
# computes it once from the real input token count and every patched
# component (decoder layers, MLP, GDN, attention) reads it here — deriving
# the decision per layer from tensor shapes is wrong whenever the standard
# (unsharded) path runs with > threshold tokens (e.g. small chunked-prefill
# steps, large pure-decode batches, or MoE models whose dense MLPs would
# otherwise fire standalone).
_FLASH_STATE = {"active": False}


def flashcomm_enabled() -> bool:
    if os.environ.get("VLLM_FL_FLASHCOMM", "1") == "0":
        return False
    try:
        if get_tensor_model_parallel_world_size() <= 1:
            return False
        if get_pp_group().world_size > 1:
            return False
    except Exception:
        # Parallel groups not initialized yet (e.g. before worker init).
        return False
    return True


def flashcomm_active(num_tokens: int) -> bool:
    """FlashComm only pays off on prefill-sized batches (vllm-ascend rule)."""
    return num_tokens > _TOKEN_THRESHOLD and flashcomm_enabled()


def flash_state_active() -> bool:
    """Per-forward activation flag set by the model wrapper (single
    decision point for all patched components)."""
    return _FLASH_STATE["active"]


def _should_manage_flash(input_ids) -> bool:
    """True only for eager forwards with real token ids. Under dynamo
    tracing or stream capture the flash management must be invisible: a
    global-state mutation (``_FLASH_STATE`` write) inside the compiled region
    is rejected by dynamo's cudagraph-safety check, and HCCL collectives must
    never be baked into captured graphs. vllm-ascend makes the collectives
    graph-safe by wrapping them in custom ops; FL keeps FlashComm eager-only
    instead (decode graph capture sizes are far below the token threshold
    anyway, so nothing is lost)."""
    if torch.compiler.is_compiling():
        return False
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        pass
    # Dummy runs for graph capture / memory profiling pass inputs_embeds
    # with input_ids=None; they must stay on the standard path.
    return isinstance(input_ids, torch.Tensor)


def _hccl_name() -> str:
    global _HCCL_NAME
    if _HCCL_NAME is None:
        rank = get_tensor_model_parallel_rank()
        _HCCL_NAME = (
            get_tp_group()
            .device_group._get_backend(torch.device("npu"))
            .get_hccl_comm_name(rank)
        )
    return _HCCL_NAME


def mm_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-parallel output via matmul-fused reduce-scatter.

    ``x``: ``(tokens, K)``; ``weight``: the linear weight ``(N, K)``.
    Returns the summed output sharded along the token dim,
    ``(tokens / tp, N)``. Falls back to GEMM + reduce_scatter when the
    fused op is unavailable.
    """
    tp_size = get_tensor_model_parallel_world_size()
    try:
        import torch_npu

        out = torch_npu.npu_mm_reduce_scatter_base(
            x,
            weight.t(),
            _hccl_name(),
            tp_size,
            reduce_op="sum",
            bias=None,
            comm_turn=0,
            comm_mode="aiv",
        )
        if bias is not None:
            out = out + bias
        return out
    except Exception as e:
        logger.warning(
            "npu_mm_reduce_scatter_base failed (%s); fallback to GEMM + reduce_scatter", e
        )
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        out = F.linear(x, weight, bias)
        return tensor_model_parallel_reduce_scatter(out, 0)


def _is_dense_mlp(mlp: torch.nn.Module) -> bool:
    return type(mlp).__name__ == "Qwen2MoeMLP" and getattr(mlp, "expert_gate", None) is None


def _pad_to_multiple(hidden_states: torch.Tensor, multiple: int) -> torch.Tensor:
    tokens = hidden_states.shape[0]
    padded = tokens + (-tokens % multiple)
    if padded == tokens:
        return hidden_states
    return F.pad(hidden_states, (0, 0, 0, padded - tokens))


def _token_shard(t: torch.Tensor, tp_size: int, rank: int) -> torch.Tensor:
    per_rank = t.shape[0] // tp_size
    return t[rank * per_rank : (rank + 1) * per_rank]


# ---------------------------------------------------------------------------
# Qwen2MoeMLP (dense MLP) flash forward
# ---------------------------------------------------------------------------

_QWEN2MOEMLP_FORWARD = None


def _qwen2moe_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    if not flash_state_active():
        return _QWEN2MOEMLP_FORWARD(self, x)
    gate_up, _ = self.gate_up_proj(x)
    out = self.act_fn(gate_up)
    return mm_reduce_scatter(out, self.down_proj.weight, self.down_proj.bias)


# ---------------------------------------------------------------------------
# Qwen3_5GatedDeltaNet flash forward (GDN linear-attention layers)
# ---------------------------------------------------------------------------

_QWEN3_5_GDN_FORWARD = None


def _gdn_forward_flash(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Qwen3_5GatedDeltaNet.forward up to ``out_proj``, ending in fused
    matmul-reduce-scatter instead of ``out_proj GEMM + all_reduce``."""
    num_tokens = hidden_states.size(0)

    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
    z_size = self.value_dim // self.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, self.head_v_dim)
    ba, _ = self.in_proj_ba(hidden_states)
    b, a = ba.chunk(2, dim=-1)
    b = b.contiguous()
    a = a.contiguous()

    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)

    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    return mm_reduce_scatter(
        core_attn_out.contiguous(), self.out_proj.weight, self.out_proj.bias
    )


def _gdn_forward(self, hidden_states: torch.Tensor, output: torch.Tensor | None = None):
    if output is None and flash_state_active():
        return _gdn_forward_flash(self, hidden_states)
    return _QWEN3_5_GDN_FORWARD(self, hidden_states, output)


# ---------------------------------------------------------------------------
# Qwen3NextDecoderLayer flash forward
# ---------------------------------------------------------------------------

_QWEN3NEXT_DECODER_FORWARD = None


def _qwen3next_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor = None,
    **kwargs,
):
    tp_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    if (
        not flash_state_active()
        or self.layer_type not in ("linear_attention", "full_attention")
        or not _is_dense_mlp(self.mlp)
    ):
        return _QWEN3NEXT_DECODER_FORWARD(self, hidden_states, residual, positions, **kwargs)

    if residual is None:
        # Model entry: pad to a TP multiple and take this rank's contiguous shard.
        hidden_states = _token_shard(_pad_to_multiple(hidden_states, tp_size), tp_size, rank)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    # Column side: restore full (padded) tokens for the attention core.
    normed_full = tensor_model_parallel_all_gather(hidden_states, 0)
    if self.layer_type == "linear_attention":
        attn_out = self.linear_attn(normed_full, None)
    else:
        attn_out = self.self_attn(positions=positions, output=None, hidden_states=normed_full)
    hidden_states = attn_out
    if self.layer_scale:
        hidden_states = hidden_states * (
            self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
        )
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    post_full = tensor_model_parallel_all_gather(hidden_states, 0)
    mlp_out = self.mlp(post_full)
    hidden_states = mlp_out
    if self.layer_scale:
        hidden_states = hidden_states * (
            self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
        )
    return hidden_states, residual


# ---------------------------------------------------------------------------
# Qwen3NextModel tail: unpad + all-gather before the logits processor
# ---------------------------------------------------------------------------

_QWEN3NEXT_MODEL_FORWARD = None


def _qwen3next_model_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds=None,
):
    # When flash management is skipped (dynamo tracing / stream capture /
    # embeds-based dummy runs), return the plain original forward so nothing
    # flash-related — in particular the ``_FLASH_STATE`` global write — is
    # ever seen by the tracer/capturer (dynamo cudagraph-safety check).
    if not _should_manage_flash(input_ids):
        return _QWEN3NEXT_MODEL_FORWARD(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    # Eager forward with real token ids: single activation decision for the
    # whole pass, consumed by every patched component via ``_FLASH_STATE``.
    num_tokens = input_ids.shape[0]
    active = (
        num_tokens > 0
        and flashcomm_active(num_tokens)
        and _is_dense_mlp(self.layers[0].mlp)
    )
    prev = _FLASH_STATE["active"]
    _FLASH_STATE["active"] = active
    try:
        hidden_states = _QWEN3NEXT_MODEL_FORWARD(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    finally:
        _FLASH_STATE["active"] = prev
    if active and isinstance(hidden_states, torch.Tensor):
        # Exit: restore full tokens from the sharded residual stream and
        # drop the TP-multiple padding rows.
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)[:num_tokens]
    return hidden_states


def patch_flashcomm_v1() -> None:
    """Apply FlashComm v1 (prefill TP communication fusion) for dense models."""
    global _QWEN2MOEMLP_FORWARD, _QWEN3_5_GDN_FORWARD
    global _QWEN3NEXT_DECODER_FORWARD, _QWEN3NEXT_MODEL_FORWARD

    if os.environ.get("VLLM_FL_FLASHCOMM", "1") == "0":
        logger.info("FlashComm v1 disabled (VLLM_FL_FLASHCOMM=0)")
        return

    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    from vllm_fl.models.qwen3_5 import Qwen3_5GatedDeltaNet

    _QWEN2MOEMLP_FORWARD = Qwen2MoeMLP.forward
    Qwen2MoeMLP.forward = _qwen2moe_mlp_forward
    _QWEN3_5_GDN_FORWARD = Qwen3_5GatedDeltaNet.forward
    Qwen3_5GatedDeltaNet.forward = _gdn_forward
    _QWEN3NEXT_DECODER_FORWARD = Qwen3NextDecoderLayer.forward
    Qwen3NextDecoderLayer.forward = _qwen3next_decoder_layer_forward
    _QWEN3NEXT_MODEL_FORWARD = Qwen3NextModel.forward
    Qwen3NextModel.forward = _qwen3next_model_forward
    logger.info(
        "Patched FlashComm v1 for Ascend (matmul-fused reduce-scatter + "
        "sharded residual stream, num_tokens > %d)",
        _TOKEN_THRESHOLD,
    )
