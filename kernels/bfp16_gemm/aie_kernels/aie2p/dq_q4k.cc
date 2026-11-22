//===- dq_q4k.cc - Q4_K weights -> the GEMM's bfp16ebs8 B tiles -------------===//
//
// One call turns k-tile j (64 values) of a tile of 64 Q4_K rows x one
// 256-value super-block into the 64 x 64 B tile mm_bfp_acc.cc reads, bit for
// bit what the host makes today (bfp16_encode_block of ggml's to_float).
//
// Input tile (re-laid by the host, byte moves and bit splits only):
//   dh[64] dl[64] mh[64] ml[64]   bf16 bits: each row's d and dmin as hi + lo
//   sc[8][64] m[8][64]            each row's 6-bit sub-block scales and mins
//   qs[4 j][8 g][2 hs][4 x'][32 bytes]  the 64 nibbles of output vector
//                                 (g, x = x' + 4 hs):
//                                 rows 8g..8g+7 x values 8x..8x+7 of k-tile
//                                 j, lane = row * 8 + k, two lanes a byte,
//                                 low nibble first (GGUF bytes 32j + 8x' ..,
//                                 x = x' low nibbles, x = x' + 4 high)
//
// ggml computes a weight as d1 * q - m1, d1 = d * sc and m1 = dmin * m in
// fp32: the products are exact, the subtraction rounds once. XDNA2 has no
// fp16 vectors, so here every product is bf16 x bf16 into fp32 accumulators:
//   D1 = dh * sc + dl * sc,  M1 = mh * m + ml * m      (exact, 17 bits)
//   D1 = D1h + D1m + D1l, three bf16 pieces
//   w  = (D1h q + D1m q + D1l q) - M1                  (exact, then one RNE)
// and the accumulator -> bfp16ebs8 conversion equals bfp16_encode_block
// (specs/npu-qgemm/plan.md has the measurements).
//
// Output order: a 64-lane vector is 8 rows x 8 consecutive values, which the
// conversion turns into the 8 consecutive blocks (72 bytes) the B tile keeps
// for them (bfp16_pack.h). They are made in the order the scales are shared
// (row group g, sub-block half hs, byte group x') and stored where the B tile
// keeps vector (g, x = x' + 4 hs): chunk (g / 2) * 16 + x * 2 + g % 2, so
// every other chunk, the stream skipping one after each store.
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

using vb = aie::vector<bfloat16, 64>;
using acc_t = aie::accum<accfloat, 64>;

constexpr int HDR = 0, SC = 512, MN = 1024, QS = 1536;

// Scratch (DQ_Q4K_SCRATCH bytes, the caller's, kept between calls): each
// row's d and dmin pieces expanded to 64 lanes per row group (lane = row *
// 8 + k), made by the call for k-tile 0 of a tile, which must come first.
struct scratch {
  alignas(64) bfloat16 eR[8][4][64];  // per row group: dh, dl, mh, ml
};
constexpr int DQ_Q4K_SCRATCH = sizeof(scratch);
// dq_weights' state, after it: the next k-tile of the current tile
struct state {
  alignas(64) int32_t j;
};
constexpr int DQ_SCRATCH = DQ_Q4K_SCRATCH + sizeof(state);

// lanes 0..7 of v, each repeated 8 times
template <typename T> __attribute__((always_inline)) inline aie::vector<T, 64> expand8(aie::vector<T, 64> v) {
  v = aie::interleave_zip(v, v, 1).first;
  v = aie::interleave_zip(v, v, 1).first;
  return aie::interleave_zip(v, v, 1).first;
}

// bytes v (< 128) -> bf16 128 + v: each byte under a 0x43 byte
__attribute__((always_inline)) inline vb plus128(aie::vector<uint8_t, 64> v) {
  const auto zz = aie::interleave_zip(v, aie::broadcast<uint8_t, 64>(0x43), 1);
  return aie::concat(zz.first, zz.second).cast_to<bfloat16>();
}

} // namespace

// k-tile j of a tile -> its B tile
__attribute__((always_inline)) inline void q4k_ktile(const uint8_t *__restrict tile, bfp16ebs8 *__restrict out,
                                                     uint8_t *__restrict scr, int32_t j) {
#ifdef DQ_NOP  // (measurements: data movement alone)
  return;
#endif
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  scratch &S = *(scratch *)scr;
  const vb k128 = aie::broadcast<bfloat16, 64>(128.0f);

  if (j == 0) {  // a new tile: its rows' d and dmin pieces, expanded
    const bfloat16 *hdr = (const bfloat16 *)(tile + HDR);
    for (int g = 0; g < 8; g++)
      for (int a = 0; a < 4; a++)
        aie::store_v(S.eR[g][a], expand8(aie::load_unaligned_v<16>(hdr + a * 64 + 8 * g, 8).grow<64>()));
  }

  // per row group g and sub-block 2j + hs: D1 = d * sc and M1 = dmin * m,
  // expanded, from 128 + sc as d * (128 + sc) - 128 * d piece by piece
  // (exact); D1 as three bf16 pieces; -128 * D1, every weight's start. Then
  // the group's four vectors: w = -128 D1 + sum of D1p * (128 + q), every
  // partial sum exact (at most 21 bits), M1 off, the one rounding. (One loop
  // for both: split into a scale loop and a weight loop it ran 12% slower.)
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os(out);
  const uint8_t *qs = tile + QS + j * 2048;
  const uint8_t *sc = tile + SC + 2 * j * 64, *mn = tile + MN + 2 * j * 64;
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int gh = 0; gh < 16; gh++) {
    const int g = gh >> 1, hs = gh & 1;
    const bfloat16 *r = S.eR[g][0];
    const vb dh = aie::load_v<64>(r), dl = aie::load_v<64>(r + 64);
    const vb mh = aie::load_v<64>(r + 128), ml = aie::load_v<64>(r + 192);
    const vb s1 = plus128(expand8(aie::load_unaligned_v<64>(sc + hs * 64 + 8 * g, 8)));
    const vb m1 = plus128(expand8(aie::load_unaligned_v<64>(mn + hs * 64 + 8 * g, 8)));
    const acc_t D1 = aie::msc(aie::mac(aie::msc(aie::mul(dh, s1), k128, dh), dl, s1), k128, dl);
    const acc_t M1 = aie::msc(aie::mac(aie::msc(aie::mul(mh, m1), k128, mh), ml, m1), k128, ml);
    const vb p0 = D1.to_vector<bfloat16>();
    acc_t a0;
    a0.from_vector(p0);
    const acc_t r1 = aie::sub(D1, a0);
    const vb p1 = r1.to_vector<bfloat16>();
    acc_t a1;
    a1.from_vector(p1);
    const vb p2 = aie::sub(r1, a1).to_vector<bfloat16>();
    const acc_t c0 = aie::msc(aie::msc(aie::negmul(k128, p0), k128, p1), k128, p2);
    AIE_LOOP_UNROLL_FULL
    for (int x = 0; x < 4; x++) {
      const vb q1 = plus128(aie::load_v<64>((const uint4 *)(qs + gh * 128 + x * 32)).unpack());  // 128 + q
      const acc_t w = aie::mac(aie::mac(aie::mac(c0, q1, p0), q1, p1), q1, p2);
      // on two chunks; after a group's last vector, to the next group's first
      // chunk: (g, 1)'s is (g, 0)'s + 8, (g + 1, 0)'s (g, 0)'s + 1 for g even
      // and + 15 for g odd
      const int skip = x < 3 || hs == 0 ? 1 : (g & 1) ? 0 : -14;
      os.push_seek(aie::sub(w, M1).to_vector<bfp16ebs8>(), skip);
    }
  }
}

extern "C" {

// One k-tile (j = 0..3) of a Q4_K tile; k-tile 0 of a tile first.
void dq_q4k(const uint8_t *__restrict tile, bfp16ebs8 *__restrict out, uint8_t *__restrict scr, int32_t j) {
  q4k_ktile(tile, out, scr, j);
}

// The GEMM's unpacking core: the next B tile from the current 9,728-byte
// weight object, in this dispatch's format, which the call counts itself
// (scratch: DQ_SCRATCH bytes, zero at the start):
//   0  a Q4_K tile (src/npu/qtile_pack.h): four k-tiles
//   1  two B tiles already in bfp16ebs8 (4,608 bytes each, then 512 unused):
//      weights of a type without an unpacker, copied through
void dq_weights(const uint8_t *__restrict tile, bfp16ebs8 *__restrict out, uint8_t *__restrict scr, int32_t fmt) {
  state &T = *(state *)(scr + DQ_Q4K_SCRATCH);
  const int32_t j = T.j;
  if (fmt == 0) {
    q4k_ktile(tile, out, scr, j);
    T.j = (j + 1) & 3;
  } else {
    const aie::vector<uint8_t, 64> *src = (const aie::vector<uint8_t, 64> *)(tile + j * 4608);
    aie::vector<uint8_t, 64> *dst = (aie::vector<uint8_t, 64> *)out;
    AIE_LOOP_MIN_ITERATION_COUNT(72)
    for (int i = 0; i < 4608 / 64; i++) dst[i] = src[i];
    T.j = (j + 1) & 1;
  }
}

} // extern "C"
