"""Prototype IQ3_S GGUF superblock decoder on one AIE2P core.

One dispatch turns 64 raw GGUF rows, each with one 256-value IQ3_S block,
into four 64x64 BFP weight tiles. This verifies the on-device dequantization
and the existing GEMM's B layout before joining the two programs.
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

ROWS = 64
K = 256
RAW_BYTES = ROWS * 110
OUT_BYTES = ROWS * K * 9 // 8
RAW_TY = np.ndarray[(RAW_BYTES,), np.dtype[np.uint8]]
OUT_TY = np.ndarray[(OUT_BYTES,), np.dtype[np.uint8]]
KERNEL_SRC = Path(__file__).resolve().parents[1] / "aie_kernels" / "aie2p" / "iq3s_to_bfp.cc"


@iron.jit
def iq3s_decode(raw: In, out: Out):
    raw_fifo = ObjectFifo(RAW_TY, name="iq3s_raw", depth=2)
    out_fifo = ObjectFifo(OUT_TY, name="bfp_weight", depth=2)
    decode = ExternalFunction(
        "iq3s_to_bfp", source_file=str(KERNEL_SRC), arg_types=[RAW_TY, OUT_TY]
    )

    def core(raw_in, bfp_out, fn):
        src = raw_in.acquire(1)
        dst = bfp_out.acquire(1)
        fn(src, dst)
        raw_in.release(1)
        bfp_out.release(1)

    worker = Worker(core, [raw_fifo.cons(), out_fifo.prod(), decode])

    def sequence(raw_arg, out_arg, raw_prod, out_cons):
        raw_prod.fill(raw_arg)
        out_cons.drain(out_arg, wait=True)

    rt = Runtime(sequence, [RAW_TY, OUT_TY, raw_fifo.prod(), out_fifo.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    parser = argparse.ArgumentParser(prog="IQ3_S to GEMM-ordered BFP16")
    add_compile_args(parser, default_dev="npu2")
    args = parser.parse_args()
    run_design_cli(iq3s_decode, args, compile_kwargs={}, device=lambda o: device_from_args(o, n_cols=1))


if __name__ == "__main__":
    main()
