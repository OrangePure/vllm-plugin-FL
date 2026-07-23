# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/acl_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ascend-specific ACL graph extensions for vllm-plugin-FL.

This module is intentionally separated from the generic graph wrapper so that
Ascend behavior (stream sync, graph-param workspaces, capture-error diagnosis,
etc.) is injected at runtime rather than hard-coded into the multi-hardware
framework.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from unittest.mock import patch

import torch

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform

from vllm_fl.compilation.graph import register_graph_wrapper_backend

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stream-resource capture error diagnostics (CANN error 207008)
# --------------------------------------------------------------------------- #
_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)


def _skip_full_graph_replay_sync() -> bool:
    """Skip the per-replay host-side full-stream drain in FULL graph mode
    (vllm-ascend PR #11915 equivalent; see ``ACLGraphBackendMixin.before_replay``
    for why FL does not need the device-side event machinery).

    Default ON (skip the barrier): c64 A/B measured TPOT 103.27 -> 76.55 ms
    (-26%) with no correctness impact. Set
    ``VLLM_FL_SKIP_FULL_GRAPH_REPLAY_SYNC=0`` to restore the legacy barrier."""
    return os.environ.get("VLLM_FL_SKIP_FULL_GRAPH_REPLAY_SYNC", "1") == "1"
_STREAM_RESOURCE_GUIDANCE = (
    "ACL graph capture failed with a known stream-resource exhaustion "
    "signature. Consider upgrading to a newer HDK/CANN stack, reducing "
    "cudagraph_capture_sizes, lowering max_cudagraph_capture_size, preferring "
    "FULL or FULL_DECODE_ONLY for mostly uniform decode workloads, or "
    "temporarily disabling graph mode to confirm the failure is capture-related."
)


def _is_stream_resource_capture_error(exc: RuntimeError) -> bool:
    message = str(exc)
    lowered_message = message.lower()
    has_error_code = _STREAM_RESOURCE_ERROR_CODE in message
    has_stream_resource_marker = any(
        marker in lowered_message for marker in _STREAM_RESOURCE_ERROR_MARKERS)
    return has_stream_resource_marker or (has_error_code
                                          and "stream resource" in lowered_message)


def _raise_stream_resource_capture_error(exc: RuntimeError) -> None:
    raise RuntimeError(
        f"{_STREAM_RESOURCE_GUIDANCE}\nOriginal error:\n{exc}") from exc


# --------------------------------------------------------------------------- #
# Ascend backend mixin for GraphWrapper
# --------------------------------------------------------------------------- #
class ACLGraphBackendMixin:
    """
    Backend-specific mixin that supplies Ascend ACL graph behavior to the
    generic GraphWrapper.

    The mixin is instantiated once per GraphWrapper and receives hook calls
    during capture and replay.  It mirrors the workflow of
    `vllm_ascend.compilation.acl_graph.ACLGraphWrapper` but stays out of the
    generic code path.
    """

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.vllm_config = wrapper.vllm_config
        self.runtime_mode = wrapper.runtime_mode
        self.aclgraph_options = wrapper.graph_options
        self.use_eagle = getattr(wrapper, "use_eagle", False)
        self.enable_enpu = getattr(wrapper, "enable_enpu", False)
        self.is_debugging_mode = wrapper.is_debugging_mode
        self._runnable_str = str(
            wrapper.runnable) if self.is_debugging_mode else None

    def _is_stream_resource_capture_error(self, exc: RuntimeError) -> bool:
        return _is_stream_resource_capture_error(exc)

    def _sync_offloader_before_capture(self) -> None:
        try:
            from vllm.model_executor.offloader.base import get_offloader
            get_offloader().sync_prev_onload()
        except Exception:
            pass

    def _join_offloader_after_forward(self) -> None:
        try:
            from vllm.model_executor.offloader.base import get_offloader
            get_offloader().join_after_forward()
        except Exception:
            pass

    def before_capture(self, entry, args, kwargs) -> None:
        self._sync_offloader_before_capture()

    def wrap_capture_context(self, entry, stack) -> None:
        # For NPU, torch.npu.empty_cache is the function that needs to be
        # disabled when gc_disable is enabled.  The generic wrapper already
        # patches PlatformFL.empty_cache; patch torch.npu.empty_cache as well.
        if self.aclgraph_options.gc_disable:
            stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

    def after_capture(self, entry, output, args, kwargs) -> Any:
        self._join_offloader_after_forward()
        # The generic wrapper will weak-ref the output again; return the
        # original output so PyTorch can manage memory correctly during capture.
        return output

    def capture_error_handler(self, exc: BaseException) -> None:
        if isinstance(exc, RuntimeError) and self._is_stream_resource_capture_error(exc):
            _raise_stream_resource_capture_error(exc)

    def before_replay(self, entry, args, kwargs) -> None:
        # In async scheduling or multi-threaded scenarios, ensure host-side
        # attention-param updates stay ordered with graph execution.
        # When enable_enpu is on, model_runner orders update vs replay; skip.
        # When FULL + EAGLE draft (merge path), replay does not need barrier.
        is_draft_eagle = False
        try:
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        except Exception:
            pass

        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        if not self.enable_enpu and need_sync:
            if _skip_full_graph_replay_sync():
                # FL has no vllm-ascend-style update_stream param-slot race
                # (update_graph_params is unused dead code here): every input /
                # metadata write before replay is issued on the current stream
                # (stream-ordered with replay), and the reused pinned CPU
                # buffers are protected by model_runner.synchronize_input_prep
                # (prepare_inputs_event). The per-replay full-stream drain is
                # therefore pure host-side overhead; skipping it lets the host
                # run ahead into the next step (vllm-ascend PR #11915 achieves
                # the same via device-side events, which FL does not need).
                return
            torch.npu.current_stream().synchronize()

    def weak_ref_tensors(self, tensor: Any) -> Any:
        """Create weak references so the graph pool can reclaim capture-time
        buffers once Python drops its strong refs (mirrors vllm-ascend
        ``utils.weak_ref_tensors``). ``torch_npu._C._weak_ref_tensor`` is
        available in the deployed torch_npu builds; fall back to identity
        when it is missing."""
        if isinstance(tensor, torch.Tensor):
            return _weak_ref_tensor(tensor)
        if isinstance(tensor, list):
            return [self.weak_ref_tensors(t) for t in tensor]
        if isinstance(tensor, tuple):
            return tuple(self.weak_ref_tensors(t) for t in tensor)
        return tensor


def _weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    global _WEAK_REF_TENSOR_FN
    if _WEAK_REF_TENSOR_FN is None:
        try:
            import torch_npu

            _WEAK_REF_TENSOR_FN = torch_npu._C._weak_ref_tensor
        except Exception:
            logger.warning(
                "torch_npu._C._weak_ref_tensor unavailable; graph-capture "
                "buffers will be strongly held (identity fallback)")
            _WEAK_REF_TENSOR_FN = lambda t: t
    return _WEAK_REF_TENSOR_FN(tensor)


_WEAK_REF_TENSOR_FN = None


def patch_graph() -> None:
    """Register the Ascend ACL graph backend mixin."""
    if current_platform.device_type != "npu":
        logger.info(
            "Skipping ACL graph patch: current platform is not NPU (%s)",
            current_platform.device_type)
        return
    register_graph_wrapper_backend("npu", ACLGraphBackendMixin)
    logger.info("Registered Ascend ACL graph backend mixin for GraphWrapper")
