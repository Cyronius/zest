#!/usr/bin/env python3
"""Check the experimental NPU IQ3_S decoder against real GGUF weights.

Run under Dibs with IRON 1.4.3 and NPU access. The test reads just 110 bytes
from each of 64 weight rows; it does not load the whole model into memory.
"""

import argparse
import ctypes
import os
import re
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "kernels/bfp16_gemm/aie_kernels/aie2p/iq3s_grid.inc"
DEFAULT_BUILD = ROOT / "third_party/iq3s-kernel-build"


def gguf_rows(path):
    with path.open("rb") as f:
        def u32(): return struct.unpack("<I", f.read(4))[0]
        def u64(): return struct.unpack("<Q", f.read(8))[0]
        def string(): return f.read(u64()).decode(errors="replace")

        def skip(t):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if t in sizes:
                f.seek(sizes[t], 1)
            elif t == 8:
                f.seek(u64(), 1)
            elif t == 9:
                element, count = u32(), u64()
                if element in sizes:
                    f.seek(sizes[element] * count, 1)
                else:
                    for _ in range(count): skip(element)
            else:
                raise ValueError(f"unknown GGUF metadata type {t}")

        if f.read(4) != b"GGUF" or u32() != 3:
            raise ValueError("expected GGUF v3")
        tensor_count, kv_count = u64(), u64()
        for _ in range(kv_count):
            string()
            skip(u32())
        chosen = None
        for _ in range(tensor_count):
            name = string()
            dims = [u64() for _ in range(u32())]
            typ, offset = u32(), u64()
            if chosen is None and typ == 21 and len(dims) == 2 and dims[0] % 256 == 0 and dims[1] >= 64:
                chosen = name, dims, offset
        if chosen is None:
            raise ValueError("no 64-row IQ3_S matrix found")
        data_start = (f.tell() + 31) & ~31
        name, dims, offset = chosen
        row_bytes = dims[0] // 256 * 110
        raw = np.empty((64, 110), dtype=np.uint8)
        for r in range(64):
            f.seek(data_start + offset + r * row_bytes)
            raw[r] = np.frombuffer(f.read(110), dtype=np.uint8)
        return name, raw


def bfp_block(x):
    bits = x.view(np.uint32)
    exponent = int(((bits >> 23) & 255).max())
    scaled = np.ldexp(x.astype(np.float64), 133 - exponent)
    q = np.clip(np.rint(scaled), -128, 127).astype(np.int8)
    return bytes([exponent]) + q.tobytes()


def oracle(raw):
    rows = re.findall(r"^\s*\{([^{}]+)\},$", GRID.read_text(), re.M)
    grid = np.array([[int(n) for n in row.split(",")] for row in rows], dtype=np.float32)
    assert grid.shape == (512, 4)
    values = np.empty((64, 256), dtype=np.float32)
    for r in range(64):
        row = raw[r]
        d = np.frombuffer(row[:2].tobytes(), dtype="<f2")[0].astype(np.float32)
        g = np.arange(64)
        indexes = row[2:66].astype(np.int32) | (((row[66 + g // 8] >> (g % 8)) & 1) << 8)
        codes = grid[indexes].reshape(256)
        k = np.arange(256)
        scale_bytes = row[106 + k // 64]
        scales = 1 + 2 * ((scale_bytes >> np.where((k & 32) != 0, 4, 0)) & 15).astype(np.int32)
        signs = 1 - 2 * ((row[74 + k // 8] >> (k % 8)) & 1).astype(np.int32)
        values[r] = ((d * scales).astype(np.float32) * codes) * signs
    out = bytearray()
    for kk in range(4):
        for sp in range(0, 64, 16):
            for sx in range(8):
                for half in range(2):
                    for i in range(8):
                        r = sp + half * 8 + i
                        k = kk * 64 + sx * 8
                        out.extend(bfp_block(values[r, k:k + 8]))
    return np.frombuffer(out, dtype=np.uint8)


def run_npu(raw, xclbin, instructions, repeat):
    try:
        ctypes.CDLL("libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    shim = ctypes.CDLL(str(ROOT / "build/libggml-zest-xrt.so"))
    ptr = ctypes.c_void_p
    size = ctypes.c_size_t
    signatures = {
        "xrtsh_last_error": (ctypes.c_char_p, []),
        "xrtsh_device_open": (ptr, [ctypes.c_int]),
        "xrtsh_device_free": (None, [ptr]),
        "xrtsh_hwctx_create": (ptr, [ptr, ctypes.c_char_p]),
        "xrtsh_hwctx_free": (None, [ptr]),
        "xrtsh_kernel_create_xclbin": (ptr, [ptr, ctypes.c_char_p]),
        "xrtsh_kernel_free": (None, [ptr]),
        "xrtsh_bo_create_instr": (ptr, [ptr, ptr, size]),
        "xrtsh_bo_create": (ptr, [ptr, size]),
        "xrtsh_bo_free": (None, [ptr]),
        "xrtsh_bo_write": (ctypes.c_int, [ptr, ptr, size, size]),
        "xrtsh_bo_read": (ctypes.c_int, [ptr, ptr, size, size]),
        "xrtsh_bo_sync": (ctypes.c_int, [ptr, ctypes.c_int]),
        "xrtsh_run_create": (ptr, [ptr]),
        "xrtsh_run_free": (None, [ptr]),
        "xrtsh_run_set_arg_int": (ctypes.c_int, [ptr, ctypes.c_int, ctypes.c_int]),
        "xrtsh_run_set_arg_bo": (ctypes.c_int, [ptr, ctypes.c_int, ptr]),
        "xrtsh_run_start": (ctypes.c_int, [ptr]),
        "xrtsh_run_wait_ms": (ctypes.c_int, [ptr, ctypes.c_uint]),
    }
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(shim, name)
        fn.restype, fn.argtypes = restype, argtypes

    def check(value, what):
        if value is None or (isinstance(value, int) and value < 0):
            raise RuntimeError(f"{what}: {shim.xrtsh_last_error().decode(errors='replace')}")
        return value

    dev = ctx = kernel = ibo = raw_bo = out_bo = run = None
    try:
        dev = check(shim.xrtsh_device_open(0), "open NPU")
        ctx = check(shim.xrtsh_hwctx_create(dev, os.fsencode(xclbin)), "load xclbin")
        kernel = check(shim.xrtsh_kernel_create_xclbin(ctx, b"MLIR_AIE"), "create kernel")
        inst = instructions.read_bytes()
        ibo = check(shim.xrtsh_bo_create_instr(dev, kernel, len(inst)), "allocate instructions")
        raw_bo = check(shim.xrtsh_bo_create(dev, raw.size), "allocate raw weights")
        out_bo = check(shim.xrtsh_bo_create(dev, 64 * 256 * 9 // 8), "allocate BFP output")
        for bo, data in ((ibo, inst), (raw_bo, raw.tobytes())):
            src = ctypes.create_string_buffer(data)
            check(shim.xrtsh_bo_write(bo, src, len(data), 0), "write input")
            check(shim.xrtsh_bo_sync(bo, 1), "sync input")
        run = check(shim.xrtsh_run_create(kernel), "create run")
        for index, value in ((0, 3), (2, len(inst) // 4)):
            check(shim.xrtsh_run_set_arg_int(run, index, value), "set integer argument")
        for index, bo in ((1, ibo), (3, raw_bo), (4, out_bo)):
            check(shim.xrtsh_run_set_arg_bo(run, index, bo), "set buffer argument")
        def dispatch():
            check(shim.xrtsh_run_start(run), "start run")
            state = shim.xrtsh_run_wait_ms(run, 30000)
            if state == 8:
                os._exit(3)  # a timed-out NPU may still be using its buffers
            if state != 4:
                raise RuntimeError(f"NPU run state {state}: {shim.xrtsh_last_error().decode(errors='replace')}")

        dispatch()
        duration_ms = None
        if repeat > 1:
            start = time.perf_counter()
            for _ in range(repeat): dispatch()
            duration_ms = (time.perf_counter() - start) * 1000 / repeat
        check(shim.xrtsh_bo_sync(out_bo, 0), "sync output")
        dst = ctypes.create_string_buffer(64 * 256 * 9 // 8)
        check(shim.xrtsh_bo_read(out_bo, dst, len(dst), 0), "read output")
        return np.frombuffer(dst.raw, dtype=np.uint8).copy(), duration_ms
    finally:
        for handle, free in ((run, shim.xrtsh_run_free), (out_bo, shim.xrtsh_bo_free),
                             (raw_bo, shim.xrtsh_bo_free), (ibo, shim.xrtsh_bo_free),
                             (kernel, shim.xrtsh_kernel_free), (ctx, shim.xrtsh_hwctx_free),
                             (dev, shim.xrtsh_device_free)):
            if handle: free(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    name, rows = gguf_rows(args.model)
    expected = oracle(rows)
    actual, duration_ms = run_npu(rows.reshape(-1), args.build / "final.xclbin",
                                  args.build / "insts.bin", args.repeat)
    bad = np.flatnonzero(actual != expected)
    if len(bad):
        first = bad[:8].tolist()
        raise AssertionError(f"{name}: {len(bad)} wrong bytes; first {first}; "
                             f"actual {actual[first].tolist()}, expected {expected[first].tolist()}")
    print(f"PASS IQ3_S NPU dequant: {name}, {len(expected)} BFP bytes exact")
    if duration_ms is not None:
        print(f"NPU decode submission plus wait: {duration_ms:.3f} ms per 64x256 tile over {args.repeat} repetitions")


if __name__ == "__main__":
    main()
