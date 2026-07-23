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

"""Weight NZ-layout conversion helpers for Ascend linear layers.

Converting a 2-D BF16/FP16 weight to the Ascend FRACTAL_NZ layout once at
load time lets aclnn matmuls skip the per-call transdata of the weight.
The conversion is numerics-preserving (bit-exact outputs, verified against
``F.linear``).  Note the benefit is shape-dependent: large output dims
(e.g. the vocab projection of ``ParallelLMHead``) gain, while small/medium
output dims are neutral-to-negative on 910B4 — hence the conversion is
applied selectively (see ``patch_linear_nz`` in ``../patch.py``), matching
vllm-ascend's default of enabling NZ only where it pays off
(``weight_nz_mode``).
"""

import logging

import torch

logger = logging.getLogger(__name__)

# aclFormat value of ACL_FORMAT_FRACTAL_NZ (same constant as vllm-ascend).
ACL_FORMAT_FRACTAL_NZ = 29


def convert_weight_nz(weight: torch.Tensor) -> torch.Tensor:
    """Return ``weight`` cast to the FRACTAL_NZ layout when eligible.

    Skips fp32/meta/non-2D tensors (embedding lookups and conv weights must
    not be converted — those are covered by the caller's class filter).
    """
    import torch_npu

    if weight.dtype == torch.float32 or weight.is_meta or weight.dim() != 2:
        return weight
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


def convert_lm_head_weight_nz(layer: torch.nn.Module) -> None:
    """Convert a ``ParallelLMHead`` weight to NZ layout once (idempotent).

    Only the vocab projection (``LogitsProcessor`` → ``F.linear``) consumes
    this weight; the input embedding uses a separate tensor
    (``tie_word_embeddings=False``), so the NZ layout cannot leak into
    ``F.embedding`` lookup paths.
    """
    if getattr(layer, "_vllm_fl_nz_converted", False):
        return
    layer.weight.data = convert_weight_nz(layer.weight.data)
    layer._vllm_fl_nz_converted = True
