import os
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = torch.device("npu", rank)

    hcom = (
        dist.group.WORLD._get_backend(dev).get_hccl_comm_name(rank)
        if hasattr(dist.group.WORLD, "_get_backend")
        else None
    )
    if hcom is None:
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL
        pg = dist.group.WORLD._get_backend(dev)
        hcom = pg.get_hccl_comm_name(rank)

    H = 5120
    dt = torch.bfloat16
    results = {}

    for M in [1024, 2048, 4096, 8192, 16384]:
        x = torch.randn(M, H, dtype=dt, device=dev)
        w = torch.randn(H, H, dtype=dt, device=dev)
        shard = torch.randn(M // world, H, dtype=dt, device=dev)

        def std_row():  # GEMM + all_reduce (standard row-parallel)
            y = x @ w.t()
            dist.all_reduce(y)
            return y

        def flash_row():  # fused mm_reduce_scatter (flash row side)
            return torch_npu.npu_mm_reduce_scatter_base(
                x, w.t(), hcom, world, reduce_op="sum",
                bias=None, comm_turn=0, comm_mode="aiv",
            )

        def flash_ag():  # all_gather (flash column side)
            out = torch.empty(M, H, dtype=dt, device=dev)
            dist.all_gather_into_tensor(out, shard)
            return out

        for name, fn in [("std_row(GEMM+AR)", std_row),
                         ("flash_mmrs", flash_row),
                         ("flash_AG", flash_ag)]:
            for _ in range(10):
                fn()
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            iters = 50
            for _ in range(iters):
                fn()
            end.record()
            torch.npu.synchronize()
            ms = start.elapsed_time(end) / iters
            results.setdefault(M, {})[name] = ms

        # per-layer comparison: standard = 2 x std_row; flash = 2 x mmrs + 2 x AG
        s = results[M]
        s["layer_std(2xGEMM+AR)"] = 2 * s["std_row(GEMM+AR)"]
        s["layer_flash(2mmrs+2AG)"] = 2 * s["flash_mmrs"] + 2 * s["flash_AG"]

    if rank == 0:
        print(f"\n{'M':>6} | {'GEMM+AR':>9} | {'mmrs':>9} | {'AG':>9} | {'layer_std':>10} | {'layer_flash':>11} | {'delta%':>7}")
        for M, s in results.items():
            d = (s["layer_flash(2mmrs+2AG)"] - s["layer_std(2xGEMM+AR)"]) / s["layer_std(2xGEMM+AR)"] * 100
            print(f"{M:>6} | {s['std_row(GEMM+AR)']:>8.3f}m | {s['flash_mmrs']:>8.3f}m | {s['flash_AG']:>8.3f}m | {s['layer_std(2xGEMM+AR)']:>9.3f}m | {s['layer_flash(2mmrs+2AG)']:>10.3f}m | {d:>6.1f}%")

    dist.destroy_process_group()

main()
