// The NPU instruction stream (insts.bin) for one size of the GEMM on
// quantized weights, made on the host like bfp16_insts.h.
//
// kernels/bfp16_gemm/designs/whole_array_bfpq.py: in each column row 2
// unpacks re-laid Q4_K tiles (qtile_pack.h) into B tiles for rows 3-5, which
// multiply. One xclbin serves every size; the stream writes each core's K
// (and the output mode), then streams A and the weights in and C out. This
// reproduces the design's runtime sequence word for word.
//
// Traces: XDNA-INSTS-GEN
#pragma once

#include <cstdint>
#include <vector>

// The weights' format, as the unpacking core takes it (dq_weights):
// re-laid Q4_K tiles (qtile_pack_q4k), or B tiles packed on the host two to
// a 9,728-byte object (qtile_pack_bfp16), for the other types.
enum bfpq_format { BFPQ_Q4K = 0, BFPQ_BFP16 = 1 };

// Whether the design takes this size: M a multiple of 384, N of 512 and at
// most 34,816, K of 256 (BFPQ_BFP16: 128).
bool bfpq_insts_fits(int64_t M, int64_t K, int64_t N, int fmt = BFPQ_Q4K);

// The stream for C[M x N] = A[M x K] * W^T in output mode 0 (float32), 1
// (bf16) or 2 (SiLU(gate) * up in bf16). C is laid out as bfp16_insts' is,
// but in row blocks of 384. Empty if the size doesn't fit.
std::vector<uint32_t> bfpq_insts(int64_t M, int64_t K, int64_t N, int mode, int fmt = BFPQ_Q4K);
