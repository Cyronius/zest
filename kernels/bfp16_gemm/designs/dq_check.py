"""One core running dq_q4k (aie_kernels/aie2p/dq_q4k.cc) over NT re-laid Q4_K
tiles: each tile (64 rows x one 256-value super-block, 9,728 bytes) becomes
four 64 x 64 bfp16ebs8 B tiles, in the kernel's vector order
(specs/npu-qgemm/plan.md, Phase 1). For checking the unpacker against the
host packer and timing it (two tile counts, host clock).

    python dq_check.py --xclbin-path build/final.xclbin --insts-path build/insts.bin
    (NT=<tiles>, default 20)
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import Buffer, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

KERNEL_SRC = Path(__file__).resolve().parents[1] / "aie_kernels" / "aie2p" / "dq_q4k.cc"
NT = int(os.environ.get("NT", "20"))
TILE_BYTES = 9728
OUT_BYTES = 64 * 64 * 9 // 8
SCRATCH_BYTES = 4096  # DQ_Q4K_SCRATCH
TILE_TY = np.ndarray[(TILE_BYTES,), np.dtype[np.uint8]]
OUT_TY = np.ndarray[(OUT_BYTES,), np.dtype[np.uint8]]
SCRATCH_TY = np.ndarray[(SCRATCH_BYTES,), np.dtype[np.uint8]]


@iron.jit
def dq_check(tiles: In, out: Out):
    tile_fifo = ObjectFifo(TILE_TY, name="q4k_tile", depth=2)
    out_fifo = ObjectFifo(OUT_TY, name="b_tile", depth=2)
    unpack = ExternalFunction("dq_q4k", source_file=str(KERNEL_SRC), arg_types=[TILE_TY, OUT_TY, SCRATCH_TY, np.int32])
    scratch = Buffer(SCRATCH_TY, name="dq_scratch")

    def core(tile_in, b_out, scr, fn):
        for _ in range_(NT):
            t = tile_in.acquire(1)
            for j in range(4):  # k-tile 0 first: it expands the tile's d and dmin
                o = b_out.acquire(1)
                fn(t, o, scr, j)
                b_out.release(1)
            tile_in.release(1)

    worker = Worker(core, [tile_fifo.cons(), out_fifo.prod(), scratch, unpack], stack_size=0x1000)

    def sequence(tiles_arg, out_arg, tile_prod, out_cons):
        tile_prod.fill(tiles_arg)
        out_cons.drain(out_arg, wait=True)

    rt = Runtime(
        sequence,
        [np.ndarray[(NT * TILE_BYTES,), np.dtype[np.uint8]], np.ndarray[(NT * 4 * OUT_BYTES,), np.dtype[np.uint8]],
         tile_fifo.prod(), out_fifo.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    parser = argparse.ArgumentParser(prog="Q4_K tiles to GEMM-ordered BFP16, one core")
    add_compile_args(parser, default_dev="npu2")
    run_design_cli(dq_check, parser.parse_args(), compile_kwargs={}, device=lambda o: device_from_args(o, n_cols=1))


if __name__ == "__main__":
    main()
