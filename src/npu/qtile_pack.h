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
// bytes, 8 bfp16ebs8 blocks) and stores each where bfp16_pack_b's B tile
// keeps it. The tiles go in the order the NPU GEMM's columns read B
// (bfp16_pack.h): column c's row groups c, c + 8, ..., each with its
// super-blocks in turn. So the unpacked tiles, one after another, are
// bfp16_pack_b's B.
//
// Traces: HYBRID-Q4-PACK
#pragma once

#include "bfp16_pack.h"

#include <cstddef>
#include <cstdint>

constexpr size_t Q4K_BLOCK_BYTES = 144;  // ggml block_q4_K: d, dmin, scales[12], qs[128]
constexpr size_t QTILE_Q4K_BYTES = 9728;
constexpr int64_t QTILE_ROWS = 64, QTILE_K = 256, QTILE_COLS = 8;

// N a multiple of 512 (64 rows for each of the GEMM's 8 columns), K of 256.
bool qtile_q4k_fits(int64_t N, int64_t K);

// N rows of K values as Q4_K (row r's K/256 blocks at raw + r * row_stride)
// -> (N/64) x (K/256) tiles, tile (row group rg, super-block sb) at
// qtile_index(rg, sb, N, K) * QTILE_Q4K_BYTES.
void qtile_pack_q4k(const uint8_t * raw, size_t row_stride, int64_t N, int64_t K, uint8_t * out,
                    const bfp16_runner & run);

// One tile from 64 rows' blocks of one super-block.
void qtile_pack_q4k_tile(const uint8_t * const * blocks, uint8_t * tile);

// Where tile (row group rg, super-block sb) goes: column rg % 8's run, its
// group rg / 8, super-block sb.
inline int64_t qtile_index(int64_t rg, int64_t sb, int64_t N, int64_t K) {
    const int64_t per_col = N / (QTILE_ROWS * QTILE_COLS);
    return ((rg % QTILE_COLS) * per_col + rg / QTILE_COLS) * (K / QTILE_K) + sb;
}

// What the unpacker makes of one tile: its four k-tiles' B tiles, 64 x 64
// bfp16ebs8 (4,608 bytes) each. A host model of dq_q4k.cc, for tests.
void qtile_unpack_q4k(const uint8_t * tile, uint8_t * out);
