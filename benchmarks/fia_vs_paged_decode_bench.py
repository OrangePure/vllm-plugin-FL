"""FIA (TND) vs paged-attention decode microbenchmark (go/no-go for FIA-in-graph).

Mirrors the two decode-attention call paths available to the FL plugin:
  A) torch_npu._npu_paged_attention            (current FL graph decode path)
  B) torch_npu.npu_fused_infer_attention_score.out  (vllm-ascend FIA-in-graph decode path)
at qwen3.6-27B / 35B decode shapes (head_dim=256, q_len=1 per request).

Usage: ASCEND_RT_VISIBLE_DEVICES=<free> python3 benchmarks/fia_vs_paged_decode_bench.py
"""

import torch
import torch_npu  # noqa: F401

DT = torch.bfloat16
BLOCK = 128
DEV = torch.device("npu:0")

# (model, num_heads, num_kv_heads)
MODELS = [("27B", 24, 4), ("35B", 16, 2)]
BATCHES = [16, 64]
CONTEXTS = [1024, 8192, 16384]
HEAD_DIM = 256
ITERS = 100


def bench(fn, iters=ITERS):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters


def run():
    torch.npu.set_device(DEV)
    print(f"{'model':>5} {'B':>3} {'ctx':>6} | {'paged(A)':>9} | {'fia(B)':>9} | {'B vs A':>7}")
    for model, H, KVH in MODELS:
        for B in BATCHES:
            for CTX in CONTEXTS:
                n_blocks_req = (CTX + BLOCK - 1) // BLOCK
                total_blocks = B * n_blocks_req + 8
                q = torch.randn(B, H, HEAD_DIM, dtype=DT, device=DEV)
                kc4 = torch.randn(total_blocks, BLOCK, KVH, HEAD_DIM, dtype=DT, device=DEV)
                vc4 = torch.randn_like(kc4)
                kc = kc4.view(total_blocks, BLOCK, -1)
                vc = vc4.view(total_blocks, BLOCK, -1)
                bt = torch.arange(B * n_blocks_req, dtype=torch.int32, device=DEV).reshape(B, n_blocks_req).contiguous()
                ctx_cpu = torch.full((B,), CTX, dtype=torch.int32, device="cpu")
                out_a = torch.empty_like(q)

                def f_paged():
                    torch_npu._npu_paged_attention(
                        query=q, key_cache=kc4, value_cache=vc4,
                        num_kv_heads=KVH, num_heads=H, scale_value=1.0,
                        block_table=bt, context_lens=ctx_cpu, out=out_a)

                # FIA (TND) decode: cumulative q lens [1..B], per-req kv lens
                aql = list(range(1, B + 1))
                kv_lens = [CTX] * B
                out_b = torch.empty_like(q)
                lse = torch.empty(1, dtype=DT, device=DEV)
                fia_kw = dict(
                    query=q, key=kc, value=vc, atten_mask=None,
                    block_table=bt, input_layout="TND", block_size=BLOCK,
                    actual_seq_lengths=aql, actual_seq_lengths_kv=kv_lens,
                    num_key_value_heads=KVH, num_heads=H, scale=1.0,
                    sparse_mode=0, out=[out_b, lse])
                try:
                    ws = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        **{k: v for k, v in fia_kw.items() if k != "out"})
                except Exception as ex:
                    print(f"{model:>5} {B:>3} {CTX:>6} | {'-':>9} | workspace fail: {str(ex)[:60]}")

                    continue

                def f_fia():
                    torch_npu.npu_fused_infer_attention_score.out(**fia_kw, workspace=ws)

                ta = tb = None
                try:
                    ta = bench(f_paged)
                except Exception as ex:
                    print(f"{model:>5} {B:>3} {CTX:>6} | paged fail: {str(ex)[:80]}")
                    continue
                try:
                    tb = bench(f_fia)
                except Exception as ex:
                    print(f"{model:>5} {B:>3} {CTX:>6} | {ta:>8.3f}m | FIA FAIL: {str(ex)[:60]}")
                    continue
                d = (ta - tb) / ta * 100
                print(f"{model:>5} {B:>3} {CTX:>6} | {ta:>8.3f}m | {tb:>8.3f}m | {d:>6.1f}%")
                del kc4, vc4, ws
                torch.npu.empty_cache()


run()
