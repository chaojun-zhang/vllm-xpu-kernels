"""Direct vs stock DCP A2A LSE reduce latency on XPU.

Usage: bench_a2a.py WORLD [graph|eager]
Mirrors vllm-project/vllm#48897: 10 ops per graph, 100 replays, median of 7,
max over ranks, per-op microseconds.
"""
import os
import statistics
import sys
import time

import multiprocess as mp
import torch
import torch.distributed as dist

import vllm_xpu_kernels._C  # noqa: F401
import vllm.v1.attention.ops.dcp as dcp
from vllm.utils.network_utils import get_open_port

WORLD = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MODE = sys.argv[2] if len(sys.argv) > 2 else "graph"
HPR, HD, MAXT = 16, 512, 128
TOKENS = [int(t) for t in os.environ.get("TOKENS", "1,2,4,8,16,32,64,128").split(",")]
OPS, REPLAYS, REPEATS = 10, 100, 7


class _Group:
    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.device_group = dist.group.WORLD


def _time(fn, device):
    if MODE == "graph":
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            for _ in range(OPS):
                fn()
        run = graph.replay
    else:
        def run():
            for _ in range(OPS):
                fn()
    for _ in range(5):
        run()
    torch.xpu.synchronize(device)
    samples = []
    for _ in range(REPEATS):
        dist.barrier()
        torch.xpu.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            run()
        torch.xpu.synchronize(device)
        us = (time.perf_counter() - t0) * 1e6 / (REPLAYS * OPS)
        t = torch.tensor([us], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(t.item())
    return statistics.median(samples)


def worker(rank, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(WORLD),
                      LOCAL_RANK=str(rank))
    device = torch.device("xpu", rank)
    torch.xpu.set_device(device)
    dist.init_process_group("xccl", rank=rank, world_size=WORLD,
                            device_id=device)
    try:
        dcp.dbo_current_ubatch_id = lambda: 0
        ws = dcp.DirectDCPA2AWorkspace(dist.group.WORLD, device, MAXT, HPR,
                                       HD, torch.bfloat16, num_ubatches=1)
        group = _Group()
        rows = []
        for n in TOKENS:
            torch.manual_seed(n + rank)
            out = torch.randn(n, WORLD * HPR, HD, dtype=torch.bfloat16,
                              device=device)
            lse = torch.randn(n, WORLD * HPR, device=device)
            ref = dcp.dcp_a2a_lse_reduce(out, lse, group,
                                         is_lse_base_on_e=True)
            got = ws.lse_reduce(out, lse, True)
            torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
            stock = _time(lambda: dcp.dcp_a2a_lse_reduce(
                out, lse, group, is_lse_base_on_e=True), device)
            direct = _time(lambda: ws.lse_reduce(out, lse, True), device)
            rows.append((n, stock, direct))
        if rank == 0:
            q.put(rows)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = get_open_port()
    procs = [ctx.Process(target=worker, args=(r, port, q))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    rows = q.get(timeout=900)
    for p in procs:
        p.join()
    print(f"world={WORLD} mode={MODE} heads/rank={HPR} head_dim={HD} bf16")
    print("| tokens | xccl (us) | direct (us) | speedup |")
    print("|---:|---:|---:|---:|")
    for n, s, d in rows:
        print(f"| {n} | {s:.1f} | {d:.1f} | {s / d:.2f}x |")
