"""Shared memory that the coarse kernel needs, by block sizes."""

import itertools

import torch
import triton
import triton_proto as tp

R, D = 4096, 96
X = torch.randn(R, D, device="cuda")
for dtype, GP, B, BD, stages in itertools.product(
    (torch.float32, torch.float16), (256, 512, 1024, 2048), (16, 32), (16, 32), (1, 3)
):
    G = torch.randn(GP, D, device="cuda").to(dtype)
    gn = G.float().square().sum(1)
    probes = torch.empty((R, 16), dtype=torch.int32, device="cuda")
    cnt = torch.zeros(GP, dtype=torch.int32, device="cuda")
    try:
        tp.coarse_kernel[(triton.cdiv(R, B),)](
            X.to(dtype),
            X,
            G,
            gn,
            probes,
            cnt,
            R,
            D,
            D,
            D,
            1.3,
            HAS_IDX=False,
            P=16,
            GP=GP,
            BLOCK_B=B,
            BLOCK_D=BD,
            PRECISION="tf32",
            DYNAMIC=False,
            num_warps=8,
            num_stages=stages,
        )
        torch.cuda.synchronize()
        res = "ok"
    except Exception as exc:
        res = repr(exc)[:80]
    print(str(dtype)[6:], "GP", GP, "B", B, "BD", BD, "stages", stages, res, flush=True)
