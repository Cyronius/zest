# whole_array_bfpq.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array GEMM on quantized weights, unpacked on the array.

npu-prefill-engine design for specs/npu-qgemm/plan.md Phase 2, derived from
the BFP16 design as it then was (B by array column, A by core row, C joined
per column and laid out in DDR by column). In each of the 8 columns:
  * row 2 unpacks: the column's weights arrive as 9,728-byte objects, which
    dq_weights (aie_kernels/aie2p/dq_q4k.cc) turns into 64 x 64 bfp16ebs8 B
    tiles, exactly what the host packer makes. The format is a runtime
    parameter: 0, re-laid Q4_K tiles (src/npu/qtile_pack.h: 64 rows x one
    256-value super-block, four B tiles each); 1, B tiles packed on the host
    (two to an object), for weights of a type without an unpacker;
  * rows 3-5 multiply: each takes every B tile (broadcast from row 2) and its
    own 128-row block of A, and runs mm_bfp_acc.cc's kernels unchanged, so
    output modes 0 (fp32), 1 (bf16) and 2 (SiLU(gate) * up) work as before.
A is split over the 3 multiplying rows, so M is a multiple of 384; N of 512,
K of 256 (format 1: 128). As in whole_array_bfp_colwise.py the core programs
serve every shape and format: the runtime sequence writes them (each
multiplying core's K/64 - 1 and output mode, the unpacking core's objects
and format) before its DMAs, so only insts.bin changes (src/npu/
bfpq_insts.cpp makes it on the host).

Weight layout (qtile_index): column c's row groups c, c + 8, ..., each with
its objects in turn; each column reads its run once per 384-row block.
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNELS = Path(__file__).resolve().parents[1] / "aie_kernels" / "aie2p"
QTILE_BYTES = 9728  # one re-laid Q4_K tile (src/npu/qtile_pack.h)
QTILE_K = 256
DQ_SCRATCH = 4160  # dq_q4k.cc's DQ_SCRATCH


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def whole_array_qmatmul(
    A: In,
    Q: In,
    C: Out,
    *,
    M: CompileTime[int] = 768,
    K: CompileTime[int] = 512,
    N: CompileTime[int] = 512,
    out_mode: CompileTime[int] = 0,
    bf16_sums: CompileTime[bool] = False,
    fmt: CompileTime[int] = 0,
):
    m, k, n = 128, 64, 64
    r = t = 8
    n_cols, n_mul = 8, 3  # columns; multiplying rows (rows 3-5, under row 2)
    fifo_depth = 2
    # weight format (dq_weights in dq_q4k.cc): 0 Q4_K tiles of 64 rows x 256
    # values, four B tiles each; 1 B tiles copied through, two to an object
    assert fmt in (0, 1)
    per_obj = 4 if fmt == 0 else 2
    assert M % (m * n_mul) == 0 and N % (n * n_cols) == 0 and K % (k * per_obj) == 0
    assert out_mode in (0, 1, 2)
    nsb = K // (k * per_obj)  # weight objects per 64 rows

    A_l2_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    Q_ty = np.ndarray[(QTILE_BYTES // 4,), np.dtype[np.uint32]]
    A_l1_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[np.float32]]
    C_l2_ty = np.ndarray[(m * n * n_mul,), np.dtype[np.float32]]
    Scr_ty = np.ndarray[(DQ_SCRATCH,), np.dtype[np.uint8]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]
    kernel_flags += os.environ.get("BFP_KFLAGS", "").split()

    def _kern(name, macro, arg_types):
        return ExternalFunction(
            name,
            source_file=str(_KERNELS / "mm_bfp_acc.cc"),
            arg_types=arg_types,
            compile_flags=kernel_flags + [f"-D{macro}"],
        )

    if bf16_sums:
        k_first = _kern("mm_sums_first", "SUMS_FIRST", [A_l1_ty, B_l1_ty, C_l1_ty, np.int32])
        k_mid = _kern("mm_sums_mid", "SUMS_MID", [A_l1_ty, B_l1_ty, C_l1_ty, np.int32])
        k_out = _kern("out_epilogue_sums", "OUT_EPILOGUE", [C_l1_ty, np.int32])
    else:
        k_first = _kern("mm_f32_first", "F32_FIRST", [A_l1_ty, B_l1_ty, C_l1_ty])
        k_mid = _kern("mm_f32_mid", "F32_MID", [A_l1_ty, B_l1_ty, C_l1_ty])
        k_out = _kern("out_epilogue", "OUT_EPILOGUE", [C_l1_ty, np.int32])
    sums_args = (lambda mode: (mode,)) if bf16_sums else (lambda mode: ())
    # the tile is uint32 words here; the kernel reads bytes
    k_dq = ExternalFunction(
        "dq_weights",
        source_file=str(_KERNELS / "dq_q4k.cc"),
        arg_types=[Q_ty, B_l1_ty, Scr_ty, np.int32],
        compile_flags=os.environ.get("DQ_KFLAGS", "").split(),
    )

    # A: one fifo per multiplying row, through a memtile to that row's 8 cores
    A_l3l2 = [ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth) for i in range(n_mul)]
    A_l2l1 = [f.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{i}") for i, f in enumerate(A_l3l2)]

    Q_l3l2, Q_l2l1, B_fifos, C_l2l3 = [], [], [], []
    C_l1l2 = [[] for _ in range(n_mul)]
    for col in range(n_cols):
        q = ObjectFifo(Q_ty, name=f"Q_L3L2_{col}", depth=fifo_depth)
        Q_l3l2.append(q)
        Q_l2l1.append(q.cons().forward(obj_type=Q_ty, name=f"Q_L2L1_{col}"))
        # unpacked B tiles, from row 2 to rows 3-5
        B_fifos.append(ObjectFifo(B_l1_ty, name=f"B_{col}", depth=fifo_depth))
        # 8x8 blocks -> row-major on the way to DDR
        c_dims: StreamDims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_dims)
        C_l2l3.append(c)
        parts = c.prod().join(
            [m * n * i for i in range(n_mul)],
            obj_types=[C_l1_ty] * n_mul,
            names=[f"C_L1L2_{col}_{i}" for i in range(n_mul)],
            depths=[1] * n_mul,
        )
        for i in range(n_mul):
            C_l1l2[i].append(parts[i])

    def unpack_fn(q_in, b_out, rtp, scr, dq):
        # One output tile's weights (64 rows x K) per pass: this dispatch's
        # objects (rtp[0] + 1 of them) of rtp[2] B tiles each, in format
        # rtp[1]. They are read once the first object has arrived, after the
        # runtime sequence wrote them. The kernel counts the B tiles of an
        # object itself.
        def obj(tq, fmt, per):
            for _ in range_(per):
                o = b_out.acquire(1)
                dq(tq, o, scr, fmt)
                b_out.release(1)
            q_in.release(1)

        tq = q_in.acquire(1)
        n_more = rtp[0]
        fmt = rtp[1]
        per = rtp[2]
        obj(tq, fmt, per)
        for _ in range_(n_more):
            obj(q_in.acquire(1), fmt, per)

    def mul_fn(in_a, in_b, out_c, rtp, first, mid, epilogue):
        # One output tile per pass, as in whole_array_bfp_colwise.py.
        elem_out = out_c.acquire(1)
        a = in_a.acquire(1)
        b = in_b.acquire(1)
        n_mid = rtp[0]
        mode = rtp[1]
        first(a, b, elem_out, *sums_args(mode))
        in_a.release(1)
        in_b.release(1)
        for _ in range_(n_mid):
            a = in_a.acquire(1)
            b = in_b.acquire(1)
            mid(a, b, elem_out, *sums_args(mode))
            in_a.release(1)
            in_b.release(1)
        epilogue(elem_out, mode)
        out_c.release(1)

    def _rtp(name, vals):
        return Buffer(
            np.ndarray[(len(vals),), np.dtype[np.int32]],
            name=name,
            initial_value=np.array(vals, dtype=np.int32),
            use_write_rtp=True,
        )

    # unpacking core: objects - 1, format, B tiles an object; multiplying:
    # K/64 - 1, mode
    rtp_dq = [_rtp(f"dq_{col}", [nsb - 1, fmt, per_obj, 0]) for col in range(n_cols)]
    rtp_mul = [[_rtp(f"nmid_{i}_{col}", [K // k - 1, out_mode]) for col in range(n_cols)] for i in range(n_mul)]
    scratch = [Buffer(Scr_ty, name=f"dq_scr_{col}", initial_value=np.zeros(DQ_SCRATCH, np.uint8))
               for col in range(n_cols)]

    workers = []
    for col in range(n_cols):
        workers.append(
            Worker(
                unpack_fn,
                [Q_l2l1[col].cons(), B_fifos[col].prod(), rtp_dq[col], scratch[col], k_dq],
                tile=Tile(col, 2),
                stack_size=0x1000,
            )
        )
        for i in range(n_mul):
            workers.append(
                Worker(
                    mul_fn,
                    [A_l2l1[i].cons(), B_fifos[col].cons(), C_l1l2[i][col].prod(), rtp_mul[i][col],
                     k_first, k_mid, k_out],
                    tile=Tile(col, 3 + i),
                    stack_size=0x1000,  # out_epilogue's SiLU frame is ~3.2 KB
                )
            )

    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    Qall_ty = np.ndarray[(N // 64 * nsb * QTILE_BYTES // 4,), np.dtype[np.uint32]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    def _lin_tap(total, offset, length, repeat=1, d0=256):
        rows = length // d0
        assert length % d0 == 0
        d1 = max(d for d in range(1, 1024) if rows % d == 0)
        d2 = rows // d1
        assert d2 < 1024 and repeat <= 64
        return TensorAccessPattern((total,), offset, [repeat, d2, d1, d0], [0, d1 * d0, d0, 1])

    row_blocks = M // (m * n_mul)
    per_col = N // n // n_cols
    a_len = m * K // 8
    q_words = per_col * nsb * QTILE_BYTES // 4
    c_per_rb = per_col * m * n_mul * n
    # a drain covering two row blocks would stride past the shim's 2^20 words
    tb_max = 2 if m * n_mul * N > (1 << 20) else 4

    def sequence(a, qw, c, A_prods, Q_prods, C_conses):
        for col in range(n_cols):
            rtp_dq[col][0] = nsb - 1
            rtp_dq[col][1] = fmt
            rtp_dq[col][2] = per_obj
            for i in range(n_mul):
                rtp_mul[i][col][0] = K // k - 1
                rtp_mul[i][col][1] = out_mode
        tg = TaskGroup()
        for tb in range((row_blocks + tb_max - 1) // tb_max):
            for pp in (0, 1):
                row_base = tb * tb_max + pp * tb_max // 2
                if row_base >= row_blocks:
                    break
                rows = min(tb_max // 2, row_blocks - row_base)
                for col in range(n_cols):
                    C_conses[col].drain(
                        c,
                        tap=_lin_tap(M * N, col * (M * N // n_cols) + row_base * c_per_rb, rows * c_per_rb),
                        wait=True,
                        group=tg,
                    )
                    for rr in range(rows):
                        if col < n_mul:
                            blk = (row_base + rr) * n_mul + col
                            A_prods[col].fill(a, tap=_lin_tap(M * K // 8, blk * a_len, a_len, per_col), group=tg)
                        Q_prods[col].fill(
                            qw, tap=_lin_tap(N // 64 * nsb * QTILE_BYTES // 4, col * q_words, q_words, d0=608),
                            group=tg,
                        )
                if tb > 0 or pp > 0:
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(
        sequence,
        [A_ty, Qall_ty, C_ty, [f.prod() for f in A_l3l2], [f.prod() for f in Q_l3l2], [f.cons() for f in C_l2l3]],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="AIE whole-array GEMM on Q4_K weights unpacked on the array")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=768)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("--out-mode", dest="out_mode", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--bf16-sums", dest="bf16_sums", action="store_true")
    p.add_argument("--fmt", type=int, choices=[0, 1], default=0)
    opts = p.parse_args()
    run_design_cli(
        whole_array_qmatmul,
        opts,
        compile_kwargs=lambda o: dict(M=o.M, K=o.K, N=o.N, out_mode=o.out_mode, bf16_sums=o.bf16_sums, fmt=o.fmt),
        device=lambda o: device_from_args(o, n_cols=8),
    )


if __name__ == "__main__":
    main()
