// Operands for the NPU's bfp16ebs8 matmul (kernels/bfp16_gemm): blocks of
// eight values sharing one exponent byte, each value a signed 8-bit mantissa,
// nine bytes a block, rounded to nearest on the host.
//
// The kernel computes C[M x N] = A[M x K] * B[K x N]. A is the activations,
// one token per row; B is given as its transpose, which is exactly a GGUF
// weight's layout (one output per row, K along the row). Blocks run along K
// in both. Each operand is laid out tile by tile in the order the cores read
// it, so every NPU transfer is one contiguous run:
//
//   A: for each block of m rows, for each chunk of k along K: one m x k tile
//   B: for each of the design's `cols` B streams c, for each of that stream's
//      n-row groups (rows (c + cols*t)*n onward), for each chunk of k: one
//      n x k tile
//
// Inside a tile the 8x8 subtiles go row pair by row pair (16 rows), block
// column by block column, the pair's first eight rows then its second. The
// kernel reads rows z and z+1 from one stream that way.
//
// Traces: HYBRID-BFP16-NUMERICS (the rounding), HYBRID-Q4-PACK (the weight path)
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

constexpr size_t BFP16_BLOCK_BYTES = 9;

// One block of eight: exponent byte = the largest of the eight float
// exponents, mantissa q = round-to-nearest-even(x * 2^(133 - exponent)).
// If a value rounds up to +128, the whole block is rounded again from x at
// one exponent higher, as the NPU's own accumulator -> bfp16ebs8 conversion
// does, so weights unpacked on the NPU equal these bit for bit
// (specs/npu-qgemm/plan.md). -128 stays as it is.
void bfp16_encode_block(const float * x, uint8_t * out);
void bfp16_decode_block(const uint8_t * in, float * x);

// A design's core tile and how its array shares the operands out: one pass
// covers m * rows prompt rows, and one round of output tiles n * cols outputs.
struct bfp16_tiling {
    int m, k, n;  // the core tile
    int cols;     // B streams: groups of n weight rows are dealt to them in turn
    int rows;     // A row blocks a pass
};
// The BFP16 kernel (kernels/bfp16_gemm/designs/whole_array_bfp_colwise.py):
// a row block of A to each of the 8 array columns, a group of B to each of
// the 4 core rows.
constexpr bfp16_tiling BFP16_TILE = { 64, 64, 128, 4, 8 };
// The kernel on quantized weights (designs/whole_array_bfpq.py): B by array
// column, A to each of its 3 multiplying rows.
constexpr bfp16_tiling BFPQ_TILE = { 128, 64, 64, 8, 3 };

// Activations, M x K row-major float -> A in the kernel's order. M a multiple
// of t.m, K of t.k. `out` is resized to M*K/8*9.
void bfp16_pack_a(const float * x, int64_t M, int64_t K, const bfp16_tiling & t, std::vector<uint8_t> & out,
                  int n_threads);

// Weights, N x K row-major float (a GGUF tensor as dequantised) -> B in the
// kernel's order. N a multiple of t.n * t.cols, K of t.k.
void bfp16_pack_b(const float * w, int64_t N, int64_t K, const bfp16_tiling & t, std::vector<uint8_t> & out,
                  int n_threads);

// Runs fn(begin, end) over [0, n), in parallel however the caller likes
// (thread_pool.h).
using bfp16_runner = std::function<void(int64_t n, const std::function<void(int64_t, int64_t)> & fn)>;

// bfp16_pack_a straight into `out` (M*K/8*9 bytes, e.g. a mapped NPU buffer).
void bfp16_pack_a_into(const float * x, int64_t M, int64_t K, const bfp16_tiling & t, uint8_t * out,
                       const bfp16_runner & run);

// bfp16_pack_b straight into `out` (N*K/8*9 bytes).
void bfp16_pack_b_into(const float * w, int64_t N, int64_t K, const bfp16_tiling & t, uint8_t * out,
                       const bfp16_runner & run);

// Pack one dequantized row at a time. The loader fills K floats for row r;
// it can decode a quantized GGUF row or supply zeros for padding. Keeping
// only one row per worker avoids a full float32 copy of the weight.
using bfp16_row_loader = std::function<void(int64_t r, float * row)>;
void bfp16_pack_b_rows_into(int64_t N, int64_t K, const bfp16_tiling & t, uint8_t * out,
                            const bfp16_runner & run, const bfp16_row_loader & load_row);

// Where the block holding columns [8*cb, 8*cb + 8) of row r goes: its byte
// offset in an A operand of K columns, or a B operand of N rows and K
// columns. For producers that encode blocks straight into place, in whatever
// order they compute them.
size_t bfp16_a_offset(int64_t r, int64_t cb, int64_t K, const bfp16_tiling & t);
size_t bfp16_b_offset(int64_t r, int64_t cb, int64_t N, int64_t K, const bfp16_tiling & t);

// The same offsets for a whole row, without a division per block: calls
// fn(cb, byte offset) for cb = 0 .. K/8-1 in order.
template <typename F>
inline void bfp16_row_blocks(int64_t first_tile, int64_t rr, int64_t K, int64_t tile_rows, const bfp16_tiling & t, F && fn) {
    const int64_t tkb = t.k / 8, nk = K / t.k, tile = tile_rows * tkb;
    int64_t at = first_tile * nk * tile + (rr / 16) * tkb * 16 + rr % 16;
    for (int64_t kk = 0, cb = 0; kk < nk; kk++, at += tile)
        for (int64_t x = 0; x < tkb; x++, cb++) fn(cb, (size_t) (at + x * 16) * BFP16_BLOCK_BYTES);
}
template <typename F>
inline void bfp16_a_row(int64_t r, int64_t K, const bfp16_tiling & t, F && fn) {
    bfp16_row_blocks(r / t.m, r % t.m, K, t.m, t, fn);
}

// Encode up to eight neighboring activation rows together. Within each
// K/8 block their BFP records are adjacent in the NPU's A layout, so this
// writes one contiguous run instead of independent strided row streams.
// r0 is a multiple of eight; count <= 8 and cannot cross a 16-row group.
void bfp16_pack_a_tile_rows(const float * const * rows, int64_t r0, int64_t count,
                           int64_t K, const bfp16_tiling & t, uint8_t * out);
template <typename F>
inline void bfp16_b_row(int64_t r, int64_t N, int64_t K, const bfp16_tiling & t, F && fn) {
    const int64_t g = r / t.n, per_col = N / t.n / t.cols;
    bfp16_row_blocks((g % t.cols) * per_col + g / t.cols, r % t.n, K, t.n, t, fn);
}

// The BFP16 kernel's output C (whole_array_bfp_colwise.py): row block (t.m
// rows) after row block, in each its n-wide output tiles in order, and of
// each tile only the words its output mode makes (bfp16_tile_words):
//   0  the tile as float32
//   1  its values as bf16
//   2  SiLU(gate) * up as bf16, n/2 values a tile row, from weights ordered
//      by bfp16_interleave_gate_up
//   3  mode 2's values as bfp16 blocks in A's layout: C is then, byte for
//      byte, the A operand (N/2 columns) of the multiply that takes it next,
//      and the host never reads it
// Modes 0 to 2 keep the tile's own block order: 8x8 block (z, j) at element
// (z * (blocks across) + j) * 64, row-major inside.
size_t bfp16_tile_words(const bfp16_tiling & t, int mode);
inline size_t bfp16_c_bytes(int64_t M, int64_t N, const bfp16_tiling & t, int mode) {
    return (size_t) (M / t.m) * (size_t) (N / t.n) * bfp16_tile_words(t, mode) * 4;
}
// Row r's results (N in modes 0 and 1, N/2 in mode 2) into `row` as floats.
void bfp16_tile_row(const void * c, int64_t N, const bfp16_tiling & t, int mode, int64_t r, float * row);

// The quantized-weight kernel's output C (whole_array_bfpq.py) is laid out by
// array column instead. Each column's results land as one contiguous run:
// column c owns floats [c*M*N/cols, (c+1)*M*N/cols); in it come blocks of
// m*rows x n, row block by row block, then the column's N tiles c, c+cols,
// ... in order, each block row-major. This copies the first `n_cols` columns
// (a multiple of t.n) of row r into `row`.
void bfp16_c_row(const float * c, int64_t M, int64_t N, const bfp16_tiling & t, int64_t r, int64_t n_cols, float * row);

// That kernel's 16-bit output modes: each core packs its m x n tile's bf16
// results at the tile's front, 8x8 block after 8x8 block in the tile's own
// block order, and the tile then travels as fp32, so the bf16 pairs land
// wherever those fp32 words do. Mode 1 is every value; mode 2 is SiLU(gate) *
// up, n/2 values per tile. This copies row r's results (N of them in mode 1,
// N/2 in mode 2) into `row` as floats.
void bfp16_c16_row(const float * c, int64_t M, int64_t N, const bfp16_tiling & t, int mode, int64_t r, float * row);

// gate and up (F x K each, row-major) as one 2F x K weight for output mode
// 2: 8 gate rows, then the 8 up rows that pair with them, and so on.
void bfp16_interleave_gate_up(const float * gate, const float * up, int64_t F, int64_t K, float * out);

// Inverse of bfp16_pack_a / bfp16_pack_b, back to row-major floats (tests).
void bfp16_unpack_a(const uint8_t * in, int64_t M, int64_t K, const bfp16_tiling & t, float * x);
void bfp16_unpack_b(const uint8_t * in, int64_t N, int64_t K, const bfp16_tiling & t, float * w);
