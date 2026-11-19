// The NPU instruction stream (insts.bin) for one size of the bfp16 matmul,
// made on the host instead of by the IRON toolchain.
//
// kernels/bfp16_gemm/designs/whole_array_bfp_colwise.py builds one core
// program that serves every size: its xclbin is the same whatever M, K, N and
// output mode it was built for (only a timestamp, an ID and starting values
// the stream overwrites differ). What changes per size is the runtime
// sequence: each core's K and output mode, how much of each output tile the
// memtiles pass on, whether the memtiles keep A for the call or relay it,
// then the shim DMA transfers that stream A and B in and C out. This
// reproduces that sequence, as the design lowers it at m = 64, k = 64,
// n = 128 on 4 x 8 cores, word for word.
//
// Traces: XDNA-INSTS-GEN
#pragma once

#include <cstdint>
#include <vector>

// Whether the design takes this size: M a multiple of 512, N a multiple of
// 512 and at most 34,816, K a multiple of 64 and at least 128.
bool bfp16_insts_fits(int64_t M, int64_t K, int64_t N);

// The stream for C[M x N] = A[M x K] * B^T in output mode 0 (float32), 1
// (bf16), 2 (SiLU(gate) * up in bf16) or 3 (the same as the bfp16 A operand
// of the next multiply; bfp16_pack.h has C's layout in each). Empty if the
// size doesn't fit.
std::vector<uint32_t> bfp16_insts(int64_t M, int64_t K, int64_t N, int mode);

// Whether a call of this size keeps A resident in the memtiles (each column's
// row block read from DDR once per pass and played to the cores from the
// memtile's own memory) instead of reading it once per output tile: K up to
// 5,440, M up to 1,024 and N at least 1,536. The results are the same either
// way.
bool bfp16_insts_resident(int64_t M, int64_t K, int64_t N);
