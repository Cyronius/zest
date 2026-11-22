# whole_array_bfp_colwise.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array bfp16 x bfp16 GEMM, activations by column, whose core program
serves every shape.

Derived from mlir-aie's whole_array matmul design. C[M x N] = A[M x K] * B^T
on 4 x 8 cores, each computing one m x n output tile at a time over K in
chunks of k (m x k x n = 64 x 64 x 128):
  * A: each array column takes one block of m rows, through its memtile to
    the column's four cores, once for every one of the cores' output tiles.
    So a pass covers 8m = 512 rows.
  * B: each core row takes one group of n weight rows, straight from a shim
    and broadcast along the row: 4n = 512 output columns per round of tiles.
  * C: each column's memtile joins its four cores' tiles and sends them on.
    A column's tiles are consecutive output columns of the same m rows, so C
    lands in DDR row block by row block.
Every column is laid out alike (its A and C in its own memtile, on its own
shim), which the runtime sequence relies on (PLACEMENT below).
The core program does not bake in the shape (aie_kernels/aie2p/mm_bfp_acc.cc,
F32_FIRST/MID, OUT_EPILOGUE):
  * K: the runtime sequence writes K/k - 1 into a per-core runtime parameter,
    read once per output tile, after that tile's first A/B chunk arrives.
  * M, N: the core body is one output tile; IRON's implicit while(true)
    repeats it for however many tiles the runtime sequence feeds.
  * the output step: the sequence also writes an output mode, and after the
    last K chunk the core runs out_epilogue on the finished tile: 0 leaves it
    fp32, 1 packs it as bf16, 2 packs SiLU(gate) * up as bf16, 3 packs the
    same as the bfp16 A tile of the multiply that takes it next.
  * how much of a tile reaches DDR: every mode packs its results at the
    tile's front, and the sequence sets how many words of each tile the
    memtiles pass on (c_words). So C is each mode's results and nothing
    else, and mode 3's C is, byte for byte, the A operand of a multiply with
    K = N / 2: a feed-forward block's gate/up and down run back to back with
    no host step between.
  * resident A: when a column's row block of A fits its memtile (K up to
    5,440; and N at least 1,536), the sequence has the memtile take it once
    per pass and play it to the cores once per output tile, instead of the
    shim reading it from DDR every time. A is then read once, not N / 512 times: about half a call's
    DDR traffic, which is what holds this kernel. Longer K is relayed tile
    by tile as the fifo was built.
One xclbin serves any (M, K, N) with M % 8m == 0, N % 4n == 0, K % k == 0, in
any output mode; only the instruction stream changes, and the host makes it
(src/npu/bfp16_insts.cpp, word for word what this design's sequence lowers to).
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import aiex
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
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL_SRC = (
    Path(__file__).resolve().parents[1] / "aie_kernels" / "aie2p" / "mm_bfp_acc.cc"
)

ROWS, COLS = 4, 8

# PLACEMENT. The runtime sequence writes memtile registers itself, so it has to
# know what the toolchain made of each column's memtile (column c's is memtile
# c: the fifos are pinned there). Every column comes out alike, and
# check_placement.py compares each build with this:
#   A in:   S2MM channel 0, the fifo's descriptors 2 and 3
#   A out:  MM2S channel 0, descriptors 0 and 1
#           its locks: 8 counts free tiles (2 to start), 9 full ones
#   C out:  MM2S channel 1, descriptors 26 to 29, one per core row
#   memory: the C join's buffer in the first two 64 KB banks; A's two tiles at
#           the start of the next two; the rest unused
# Resident A uses what the fifos leave free: descriptors 10 to 12, locks 16
# and 17, and the memory from the third bank on (over A's own two tiles,
# which a resident call doesn't use).
A_IN_CH, A_IN_BD, A_OUT_CH, A_OUT_BD = 0, 2, 0, 0
A_FREE_LOCK, A_FULL_LOCK = 8, 9
C_OUT_CH, C_OUT_BD = 1, 26
RES_BASE, RES_BYTES = 131072, 393216
RES_FILL, RES_PLAY, RES_LAST = 10, 11, 12
RES_FREE_LOCK, RES_FULL_LOCK = 16, 17

# Memtile registers (AIE2P). A buffer descriptor is eight words: the first is
# its length in words (low 17 bits), the second its address in words (the
# tile's own memory starts at 0x20000), the last its locks (the tile's own are
# 64 up). A channel's control register resets it (bit 1); its start queue
# takes a descriptor and a repeat count. A lock's value is a register.
MEM_BD, MEM_BD_STRIDE = 0xA0000, 0x20
MEM_S2MM_CTRL, MEM_S2MM_QUEUE = 0xA0600, 0xA0604
MEM_MM2S_CTRL, MEM_MM2S_QUEUE = 0xA0630, 0xA0634
MEM_CH_STRIDE = 8
MEM_LOCK, MEM_LOCK_STRIDE = 0xC0000, 0x10
MEM_OWN_MEMORY, MEM_OWN_LOCKS = 0x20000, 64


def resident(M, K, N, m=64, n=128):
    """Whether a call keeps A in the memtiles: its row block fits, its
    passes' transfers fit the channels' queues, and there are output tiles
    enough to gain by it (a memtile plays a block only once it has all of
    it, which costs a narrow call more than its second read from DDR)."""
    return m * (K // 8) * 9 <= RES_BYTES and M // (m * COLS) <= 2 and N // (n * ROWS) >= 3


def c_words(m, n, out_mode):
    """Words of each core tile that reach DDR."""
    return [m * n, m * n // 2, m * n // 4, m * (n // 16) * 9 // 4][out_mode]


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def whole_array_matmul(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 512,
    K: CompileTime[int] = 512,
    N: CompileTime[int] = 512,
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 128,
    out_mode: CompileTime[int] = 0,
    bf16_sums: CompileTime[bool] = False,
):
    fifo_depth = 2
    assert K // k >= 2
    assert M % (m * COLS) == 0, "M must be a multiple of m * 8"
    assert K % k == 0
    assert N % (n * ROWS) == 0, "N must be a multiple of n * 4"
    # the bfp16ebs8 MAC unit is 8x8x8, and the kernel takes its tiles in pairs
    assert m % 16 == 0 and k % 16 == 0 and n % 16 == 0
    assert out_mode in (0, 1, 2, 3)
    # mode 3 writes the A tile of a multiply with K = N / 2
    assert out_mode != 3 or n == 2 * k

    A_l2_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(m * n * ROWS,), np.dtype[np.float32]]
    A_l1_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[np.float32]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]
    # extra kernel defines, e.g. BFP_KFLAGS="-DROUND_CONV_EVEN"
    kernel_flags += os.environ.get("BFP_KFLAGS", "").split()

    def _kern(name, macro, arg_types):
        return ExternalFunction(
            name,
            source_file=str(_KERNEL_SRC),
            arg_types=arg_types,
            compile_flags=kernel_flags + [f"-D{macro}"],
        )

    if bf16_sums:
        # modes 1 to 3 keep bf16 running sums (mm_bfp_acc.cc, mm_sums_*); the
        # kernels take the output mode
        k_first = _kern("mm_sums_first", "SUMS_FIRST", [A_l1_ty, B_l1_ty, C_l1_ty, np.int32])
        k_mid = _kern("mm_sums_mid", "SUMS_MID", [A_l1_ty, B_l1_ty, C_l1_ty, np.int32])
        k_out = _kern("out_epilogue_sums", "OUT_EPILOGUE", [C_l1_ty, np.int32])
    else:
        k_first = _kern("mm_f32_first", "F32_FIRST", [A_l1_ty, B_l1_ty, C_l1_ty])
        k_mid = _kern("mm_f32_mid", "F32_MID", [A_l1_ty, B_l1_ty, C_l1_ty])
        k_out = _kern("out_epilogue", "OUT_EPILOGUE", [C_l1_ty, np.int32])
    sums_args = (lambda mode: (mode,)) if bf16_sums else (lambda mode: ())

    A_l3l2, A_l2l1, B_l3l1, C_l2l3 = [], [], [], []
    C_l1l2 = [[] for _ in range(ROWS)]
    for col in range(COLS):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=fifo_depth)
        A_l3l2.append(a_l3l2)
        A_l2l1.append(a_l3l2.cons().forward(tile=Tile(col, 1), obj_type=A_l1_ty, name=f"A_L2L1_{col}"))
        # One buffer for the join, so its four descriptors go round exactly
        # once per join and the channel stands at the first between calls
        # (the sequence restarts it there). Tiles are many thousands of cycles
        # apart, so a second buffer would buy nothing.
        c_l2l3 = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=1)
        C_l2l3.append(c_l2l3)
        parts = c_l2l3.prod().join(
            [m * n * i for i in range(ROWS)],
            tile=Tile(col, 1),
            obj_types=[C_l1_ty] * ROWS,
            names=[f"C_L1L2_{col}_{row}" for row in range(ROWS)],
            depths=[1] * ROWS,
        )
        for row in range(ROWS):
            C_l1l2[row].append(parts[row])
    for row in range(ROWS):
        # straight from the shim: a memtile relay would take two banks of the
        # memory resident A needs, and is no faster
        B_l3l1.append(ObjectFifo(B_l1_ty, name=f"B_L3L1_{row}", depth=fifo_depth))

    def core_fn(in_a, in_b, out_c, rtp, first, mid, epilogue):
        # One output tile per pass of the implicit while(true).
        elem_out = out_c.acquire(1)
        elem_in_a = in_a.acquire(1)
        elem_in_b = in_b.acquire(1)
        # this dispatch's K/k - 1 and output mode
        n_mid = rtp[0]
        mode = rtp[1]
        first(elem_in_a, elem_in_b, elem_out, *sums_args(mode))
        in_a.release(1)
        in_b.release(1)
        for _ in range_(n_mid):
            elem_in_a = in_a.acquire(1)
            elem_in_b = in_b.acquire(1)
            mid(elem_in_a, elem_in_b, elem_out, *sums_args(mode))
            in_a.release(1)
            in_b.release(1)
        epilogue(elem_out, mode)
        out_c.release(1)

    rtps = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"nmid_{row}_{col}",
                initial_value=np.array([K // k - 1, out_mode], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(COLS)
        ]
        for row in range(ROWS)
    ]

    workers = Worker.grid(
        ROWS,
        COLS,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1[col].cons(),
                B_l3l1[row].cons(),
                C_l1l2[row][col].prod(),
                rtps[row][col],
                k_first,
                k_mid,
                k_out,
            ],
            stack_size=0x1000,  # out_epilogue's SiLU frame is ~3.2 KB
        ),
    )

    cw = c_words(m, n, out_mode)
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(N * K // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M // m * (N // n) * cw,), np.dtype[np.float32]]

    # A, B and C are laid out in DDR tile by tile in the order the array
    # reads and writes them (src/npu/bfp16_pack.h), so every shim transfer is
    # one contiguous run:
    #   A: row block rb (m rows), then each K chunk       -> rb * m * K/8
    #   B: core row r's groups (rows (r + 4t) * n ..), each K chunk
    #   C: row block rb, then each n-wide output tile, cw words each
    def _lin_tap(total, offset, length, repeat=1):
        d0 = 256  # elements; every length here is a multiple of it
        assert length % d0 == 0
        runs = length // d0
        d1 = max(d for d in range(1, 1024) if runs % d == 0)
        d2 = runs // d1
        assert d2 < 1024 and repeat <= 64
        return TensorAccessPattern((total,), offset, [repeat, d2, d1, d0], [0, d1 * d0, d0, 1])

    passes, per = M // (m * COLS), N // (n * ROWS)
    a_run, b_run, c_run = m * (K // 8), (N // ROWS) * (K // 8), (N // n) * cw
    keep_a = resident(M, K, N, m, n)

    def reset(mt, ctrl, ch):
        aiex.npu_maskwrite32(ctrl + MEM_CH_STRIDE * ch, 2, 2, column=mt, row=1)
        aiex.npu_maskwrite32(ctrl + MEM_CH_STRIDE * ch, 0, 2, column=mt, row=1)

    def set_lock(mt, lock, value):
        aiex.npu_write32(MEM_LOCK + MEM_LOCK_STRIDE * lock, value, column=mt, row=1)

    def sequence(a, b, c, A_prods, B_prods, C_conses):
        # This dispatch's K-chunk count and output mode, written before any of
        # its DMAs start.
        for row in rtps:
            for rtp in row:
                rtp[0] = K // k - 1
                rtp[1] = out_mode
        for mt in range(COLS):
            # The memtile passes on cw words of each tile. A waiting channel
            # has already loaded its next descriptor, so the channel is reset
            # and started again at its first, where it stands between calls.
            reset(mt, MEM_MM2S_CTRL, C_OUT_CH)
            for i in range(ROWS):
                aiex.npu_maskwrite32(MEM_BD + MEM_BD_STRIDE * (C_OUT_BD + i), cw, 0x1FFFF, column=mt, row=1)
            aiex.npu_write32(MEM_MM2S_QUEUE + MEM_CH_STRIDE * C_OUT_CH, C_OUT_BD, column=mt, row=1)
            # A's two channels, whatever the call before left them doing
            reset(mt, MEM_S2MM_CTRL, A_IN_CH)
            reset(mt, MEM_MM2S_CTRL, A_OUT_CH)
            if not keep_a:
                # relayed tile by tile: the fifo's own descriptors, going
                # round, and its locks as they start
                set_lock(mt, A_FREE_LOCK, 2)
                set_lock(mt, A_FULL_LOCK, 0)
                aiex.npu_write32(MEM_S2MM_QUEUE + MEM_CH_STRIDE * A_IN_CH, A_IN_BD, column=mt, row=1)
                aiex.npu_write32(MEM_MM2S_QUEUE + MEM_CH_STRIDE * A_OUT_CH, A_OUT_BD, column=mt, row=1)
                continue
            # Resident: three descriptors of our own over the whole row
            # block. Fill waits for the buffer to be free and marks it full;
            # play needs it full and leaves it so; a pass's last play frees
            # it for the next pass's fill.
            set_lock(mt, RES_FREE_LOCK, 1)
            set_lock(mt, RES_FULL_LOCK, 0)
            for bd, acq, rel in (
                (RES_FILL, RES_FREE_LOCK, RES_FULL_LOCK),
                (RES_PLAY, RES_FULL_LOCK, RES_FULL_LOCK),
                (RES_LAST, RES_FULL_LOCK, RES_FREE_LOCK),
            ):
                aiex.npu_writebd(
                    column=mt, row=1, bd_id=bd, buffer_length=a_run * 9 // 4,
                    buffer_offset=MEM_OWN_MEMORY + RES_BASE // 4,
                    enable_packet=0, out_of_order_id=0, packet_id=0, packet_type=0,
                    d0_size=0, d0_stride=0, d1_size=0, d1_stride=0, d2_size=0, d2_stride=0,
                    iteration_current=0, iteration_size=0, iteration_stride=0,
                    next_bd=0, use_next_bd=0, valid_bd=1,
                    lock_rel_val=1, lock_rel_id=MEM_OWN_LOCKS + rel,
                    lock_acq_enable=1, lock_acq_val=-1, lock_acq_id=MEM_OWN_LOCKS + acq,
                    d0_zero_before=0, d1_zero_before=0, d2_zero_before=0,
                    d0_zero_after=0, d1_zero_after=0, d2_zero_after=0,
                )
            for _ in range(passes):
                aiex.npu_push_queue(column=mt, row=1, direction=0, channel=A_IN_CH, issue_token=False,
                                    repeat_count=0, bd_id=RES_FILL)
                if per > 1:
                    aiex.npu_push_queue(column=mt, row=1, direction=1, channel=A_OUT_CH, issue_token=False,
                                        repeat_count=per - 2, bd_id=RES_PLAY)
                aiex.npu_push_queue(column=mt, row=1, direction=1, channel=A_OUT_CH, issue_token=False,
                                    repeat_count=0, bd_id=RES_LAST)
        tg = TaskGroup()
        for p in range(passes):
            for col in range(COLS):
                rb = p * COLS + col
                C_conses[col].drain(c, tap=_lin_tap(M // m * c_run, rb * c_run, c_run), wait=True, group=tg)
                # the shim reads the row block once for a resident call, else
                # once per output tile
                A_prods[col].fill(a, tap=_lin_tap(M * K // 8, rb * a_run, a_run, 1 if keep_a else per), group=tg)
            for row in range(ROWS):
                B_prods[row].fill(b, tap=_lin_tap(N * K // 8, row * b_run, b_run), group=tg)
            # two passes' transfers queue on a channel before the wait
            if p % 2 == 1 or p == passes - 1:
                tg.finish()
                if p != passes - 1:
                    tg = TaskGroup()

    rt = Runtime(
        sequence,
        [
            A_ty,
            B_ty,
            C_ty,
            [f.prod() for f in A_l3l2],
            [f.prod() for f in B_l3l1],
            [f.cons() for f in C_l2l3],
        ],
    )

    return Program(
        iron.get_current_device(),
        rt,
        workers=[w for row in workers for w in row],
    ).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE whole-array bfp16ebs8 matmul, activations by column")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=128)
    p.add_argument("--out-mode", dest="out_mode", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--bf16-sums", dest="bf16_sums", action="store_true")
    return p


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_matmul,
        opts,
        compile_kwargs=lambda o: dict(
            M=o.M, K=o.K, N=o.N, m=o.m, k=o.k, n=o.n, out_mode=o.out_mode, bf16_sums=o.bf16_sums
        ),
        device=lambda o: device_from_args(o, n_cols=COLS),
    )


if __name__ == "__main__":
    main()
