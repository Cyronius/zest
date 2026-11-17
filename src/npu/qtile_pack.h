// Q4_K weights re-laid for the NPU's unpacker (kernels/bfp16_gemm/
// aie_kernels/aie2p/dq_q4k.cc), which turns them into the GEMM's bfp16ebs8
// B tiles bit for bit what bfp16_pack_b makes of ggml's dequantization.
//
// One tile is 64 weight rows x one 256-value super-block (QTILE_Q4K_BYTES):
//   dh[64] dl[64] mh[64] ml[64]   bf16 bits: each row's d and dmin as hi + lo
//                                 (hi + lo == the fp16 value exactly)
//   sc[8][64] m[8][64]            each row's 6-bit sub-block scales and mins
//   qs[4 j][8 g][2 hs][4 x'][32]  per k-tile j (64 values), row group g (rows
//                                 8g..8g+7) and sub-block half hs: the 64
//                                 nibbles of output vector x = x' + 4 hs,
//                                 lane = row * 8 + k, two lanes a byte, low
//                                 nibble first
// 9,728 bytes for 9,216 of GGUF data: bit splits and byte moves, no rounding.
//
// The unpacker makes each k-tile as 64 vectors of 8 rows x 8 values (72
// bytes, 8 bfp16ebs8 blocks), vector n = (g * 2 + hs) * 4 + x' in its order;
// in the B tile (bfp16_pack.h) that is chunk qtile_b_chunk(n).
//
// Traces: HYBRID-Q4-PACK
#pragma once

#include "bfp16_pack.h"

#include <cstddef>
#include <cstdint>

constexpr size_t Q4K_BLOCK_BYTES = 144;  // ggml block_q4_K: d, dmin, scales[12], qs[128]
constexpr size_t QTILE_Q4K_BYTES = 9728;
constexpr int64_t QTILE_ROWS = 64, QTILE_K = 256;

// N a multiple of 64, K of 256.
bool qtile_q4k_fits(int64_t N, int64_t K);

// N rows of K values as Q4_K (row r's K/256 blocks at raw + r * row_stride)
// -> (N/64) x (K/256) tiles, tile (row group rg, super-block sb) at
// (rg * K/256 + sb) * QTILE_Q4K_BYTES.
void qtile_pack_q4k(const uint8_t * raw, size_t row_stride, int64_t N, int64_t K, uint8_t * out,
                    const bfp16_runner & run);

// One tile from 64 rows' blocks of one super-block.
void qtile_pack_q4k_tile(const uint8_t * const * blocks, uint8_t * tile);

// What the unpacker makes of one tile: its four k-tiles' B tiles, 64 x 64
// bfp16ebs8 (4,608 bytes) each, every one in the unpacker's vector order.
// A host model of dq_q4k.cc, for tests.
void qtile_unpack_q4k(const uint8_t * tile, uint8_t * out);

// The unpacker's vector n -> its 72-byte chunk in the B tile.
inline int qtile_b_chunk(int n) {
    const int g = n >> 3, hs = (n >> 2) & 1, x = (n & 3) + 4 * hs;
    return (g >> 1) * 16 + x * 2 + (g & 1);
}
