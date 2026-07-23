# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/compiler_interface.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ascend-specific compiler interface for vllm-plugin-FL.

This module provides the CompilerInterface subclass used when Ascend graph
compilation (npugraph_ex / torchair) is enabled.  By default vllm-plugin-FL
falls back to eager mode on NPU, so this class is only instantiated when the
user explicitly enables `ascend_compilation_config.enable_npugraph_ex`.

It also ports vllm-ascend's `enable_static_kernel` feature: when
`ascend_compilation_config.enable_static_kernel` is set (requires
`enable_npugraph_ex`), operator binaries are pre-compiled with fixed shapes
for the decode cudagraph batch sizes during graph capture, removing dynamic
shape/tiling overhead at replay time.  The compiled static kernels are
installed as .run packages under `$ASCEND_HOME_PATH/opp/static_kernel` and
are uninstalled on worker exit via the Ascend patch
`vllm_fl/dispatch/backends/vendor/ascend/patches/patch_static_kernel.py`.
"""

from __future__ import annotations

import copy
import os
from hashlib import sha256
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from torch._inductor.compile_fx import (
    graph_returns_tuple,
    make_graph_return_tuple,
)

from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_ascend_compilation_config(vllm_config: VllmConfig) -> dict:
    """Read the plugin's ``ascend_compilation_config`` dict.

    The plugin deliberately mirrors vllm-ascend's config key so that the
    same ``--additional-config '{"ascend_compilation_config": {...}}'``
    works on both.  Options are plain dict entries here (no dataclass).
    """
    return (vllm_config.additional_config or {}).get(
        "ascend_compilation_config", {}
    ) or {}


def _compute_decode_cudagraph_batch_sizes(vllm_config: VllmConfig) -> list[int]:
    num_spec_tokens = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    uniform_decode_query_len = num_spec_tokens + 1
    max_num_tokens = vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
    return [
        x
        for x in vllm_config.compilation_config.cudagraph_capture_sizes
        if max_num_tokens >= x >= uniform_decode_query_len
    ]


def _configure_backend(
    config: Any,
    ascend_compilation_config: dict,
    vllm_config: VllmConfig,
    process_kwargs_options: Optional[Callable] = None,
) -> None:
    enable_static_kernel = ascend_compilation_config.get(
        "enable_static_kernel", False
    )
    if enable_static_kernel:
        # npugraph_ex's static_kernel requires LOCAL_WORLD_SIZE to determine the
        # physical node topology for creating per-node Gloo groups, which
        # coordinate static kernel compilation and .run package installation.
        # vLLM does not set this env var by default (unlike torchrun), so we
        # compute it from parallel config:
        #   local_world_size: processes per node for one DP replica
        #   data_parallel_size_local: number of DP replicas on this node
        #   actual_local_world_size: total processes on this physical machine
        if "LOCAL_WORLD_SIZE" not in os.environ:
            actual_local_world_size = (
                vllm_config.parallel_config.local_world_size
                * vllm_config.parallel_config.data_parallel_size_local
            )
            os.environ["LOCAL_WORLD_SIZE"] = str(actual_local_world_size)
            logger.info_once(
                "Setting LOCAL_WORLD_SIZE=%d for static kernel (local_world_size=%d * data_parallel_size_local=%d).",
                actual_local_world_size,
                vllm_config.parallel_config.local_world_size,
                vllm_config.parallel_config.data_parallel_size_local,
                scope="global",
            )

    if process_kwargs_options is not None:
        # npugraph_ex (both old and new): build options dict and use _process_kwargs_options.
        # It maps flat option names to nested config paths for old versions,
        # and directly setattr for new versions with flat CompilerConfig.
        # force_eager=True: execute FX graph in eager mode before graph capture.
        # inplace_pass=False: disable reinplace pass to avoid gelu fallback to CPU.
        options: dict[str, Any] = {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        }
        if enable_static_kernel:
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )
            options["static_kernel_compile"] = True
            # Set sym_range to limit static kernel compilation to specified batch sizes.
            options["_vllm_aclnn_static_kernel_sym_range"] = (
                _compute_decode_cudagraph_batch_sizes(vllm_config)
            )
        process_kwargs_options(config, {"options": options})
    else:
        # torchair (reduce-overhead): use nested config structure directly.
        # mode="reduce-overhead": use aclgraph mode, avoid fx graph to Ascend IR transformation.
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        # Disable reinplace pass to avoid gelu fallback to CPU causing host-device copy error.
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        if enable_static_kernel:
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = (
                _compute_decode_cudagraph_batch_sizes(vllm_config)
            )


def npugraph_ex_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    vllm_config: VllmConfig,
    ascend_compilation_config: dict,
    compile_range: Range,
) -> tuple[Optional[Callable], Optional[Any]]:
    # Try npugraph_ex first, fall back to torchair for backward compatibility.
    try:
        import npugraph_ex as nge

        torch.npu.set_compile_mode(jit_compile=False)
        config = nge.CompilerConfig()
        # _process_kwargs_options exists in both old and new npugraph_ex,
        # but in different modules: new -> compiler_config, old -> npugraphex_config.
        try:
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
        except ImportError:
            from npugraph_ex.configs.npugraphex_config import (
                _process_kwargs_options,
            )
        _configure_backend(
            config,
            ascend_compilation_config,
            vllm_config,
            process_kwargs_options=_process_kwargs_options,
        )
        backend = nge.get_npu_backend(compiler_config=config)
    except ImportError:
        try:
            import torchair
        except ImportError as exc:
            raise ImportError(
                "npugraph_ex or torchair is required for AscendCompiler. "
                "Either install it or disable "
                "ascend_compilation_config.enable_npugraph_ex."
            ) from exc

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        _configure_backend(config, ascend_compilation_config, vllm_config)
        backend = torchair.get_npu_backend(compiler_config=config)

    # torch.compile requires the output of the fx graph to be a tuple
    if not graph_returns_tuple(graph):
        compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
    else:
        compiled_fn = backend(graph, example_inputs)
    # No cache handle: vLLM skips load() when the handle is None.
    return compiled_fn, None


class AscendCompiler(CompilerInterface):
    """
    Ascend compiler interface.

    Delegates FX graph compilation to npugraph_ex (preferred) or torchair.
    Instantiated only when `ascend_compilation_config.enable_npugraph_ex`
    is set (see `vllm_fl.platform.PlatformFL.get_compile_backend`).
    """

    name = "AscendCompiler"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        self.vllm_config = vllm_config
        import torch_npu

        ascend_compilation_config = _get_ascend_compilation_config(vllm_config)
        factors = {
            "torch_npu_version": torch_npu.__version__,
            "enable_npugraph_ex": ascend_compilation_config.get(
                "enable_npugraph_ex", False
            ),
            "enable_static_kernel": ascend_compilation_config.get(
                "enable_static_kernel", False
            ),
        }
        logger.info("AscendCompiler hash factors: %s", factors)
        return sha256(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]

    def initialize_cache(self, cache_dir: str, *args, **kwargs) -> None:
        logger.info("AscendCompiler cache dir: %s", cache_dir)

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        from torch._guards import detect_fake_mode

        current_fake_mode = detect_fake_mode()
        if current_fake_mode is not None:
            example_inputs = [
                current_fake_mode.from_tensor(inp)
                if (
                    isinstance(inp, torch.Tensor)
                    and hasattr(inp, "fake_mode")
                    and inp.fake_mode is not current_fake_mode
                )
                else inp
                for inp in example_inputs
            ]

        assert hasattr(self, "vllm_config")
        ascend_compilation_config = _get_ascend_compilation_config(self.vllm_config)
        logger.info_once(
            "enable_npugraph_ex is enabled, which will bring graph compilation optimization.",
            scope="global",
        )
        return npugraph_ex_compile(
            graph,
            example_inputs,
            compiler_config,
            self.vllm_config,
            ascend_compilation_config,
            compile_range,
        )

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable:
        raise NotImplementedError(
            "AscendCompiler.compile returns handle=None, so vLLM never calls "
            "load(); compilation caching is not supported yet."
        )
