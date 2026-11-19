//===- mm_bfp_acc.cc - bfp16 x bfp16 GEMM with fp32 or bf16 partial sums ---===//
//
// npu-prefill-engine addition, derived from mlir-aie v1.4.2's
// aie_kernels/aie2p/mm_bfp.cc (Apache-2.0 WITH LLVM-exception).
//
// The stock kernel keeps C as bfp16ebs8 in local memory and re-quantises the
// partial sums after every K chunk (1.8-3.0% error at K=2048..6144). These
// entry points keep the partial sums in fp32 (or bf16) instead:
//
//   fp32: first(A,B,Cacc)  mid(A,B,Cacc)  last(A,B,Cacc,Cout)   Cout is bf16
//   bf16: bf16_first(A,B,C)  bf16_mid(A,B,C)                      C is bf16
//
// A and B arrive pair-interleaved (see src/npu/bfp16_pack.h):
// one block stream per operand (see mm_paired). C blocks are 8x8,
// block (z, j) at element (z * n/8 + j) * 64, each block row-major (the layout
// whole_array_mixed's C path untiles). Every accum -> bf16 conversion rounds
// to nearest even.
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

namespace {

constexpr unsigned rowA = DIM_M / 8, colA = DIM_K / 8, colB = DIM_N / 8;
static_assert(DIM_M % 16 == 0 && DIM_N % 16 == 0 && DIM_K % 16 == 0);

using acc_t = aie::accum<accfloat, 64>;

enum class Mode { First, Mid, Last };

// One pass over the output tile in 2x2 blocks of 8x8 sub-blocks (rows z, z+1;
// columns j, j+1), in one flat loop: band by band (16 rows), each band's
// blocks left to right. The A and B streams persist across blocks and are
// moved on by 2-D seeks; each block's K steps are written out. Each block
// pops 2 k/8 panels per stream, a multiple of 8, so a stream is drained
// exactly at every seek.
//
// The loop is laid out so the compiler's pipeliner can overlap one block with
// the next: C's loads and its stores walk their own pointers, each with its
// own memory resource, and nothing but those pointers is carried from block
// to block; the operand streams are the restrict kind, each with its own
// resource; and on the K steps that refill a stream, B pops ahead of A. The
// MACs on each accumulator are the same, in the same order, as the plain
// loop's, so results are bit-identical. Measured 2026-11-18 at the
// 128 x 64 x 64 tile (docs/results/halo-npu-qgemm-2026-11-18.json): the
// quantized-weight GEMM's multiply falls from about 14.5 to 11.9 ms per
// 1,536 x 17,408 x 5,120 (the GEMM from 15.5 to 13.3 ms, where its unpacking
// core then holds it), and the BFP16 GEMM, which DRAM holds, gains 3%.
//
// With fp32 C, C itself is the next limit: a block's four accumulators fill
// the core's accumulator registers, so every block pays its C loads and
// stores (bf16 sums, half the bytes, bring the multiply to about 8 ms).
//
// TC: type C is read from (ignored for First); TO: type written.
template <Mode mode, typename TC, typename TO>
__attribute__((always_inline)) inline void
mm_paired(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB,
          const TC __aie_dm_resource_c *__restrict pCld,
          TO __aie_dm_resource_c *__restrict pCst) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  using AStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::a>;
  using BStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b>;
  using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;
  constexpr int NJ = colB / 2;        // blocks per band
  constexpr int NBR = rowA / 2;       // bands
  constexpr int PANELS = 2 * colA;    // panels a block pops per stream
  static_assert(PANELS % 8 == 0 && colA == 8);
  constexpr int BAND = (colB + 2) * 64;  // from a band's last block to the next band's first

  AStream sA(pA);
  BStream sB(pB);
  dims_2d_t walkA(NJ - 1, -PANELS, 0);
  dims_2d_t walkB(NJ - 1, 0, -(int)(PANELS * NJ));
  int cL = 0, cS = 0;

#define ZEST_KSTEP_B                                                           \
  {                                                                            \
    aie::block_vector<bfp16ebs8, 64> A0, A1, B0, B1;                           \
    B0 = sB.pop(); B1 = sB.pop(); A0 = sA.pop(); A1 = sA.pop();                \
    c00.mac(A0, aie::op_transpose(B0)); c01.mac(A0, aie::op_transpose(B1));    \
    c10.mac(A1, aie::op_transpose(B0)); c11.mac(A1, aie::op_transpose(B1));    \
  }
#define ZEST_KSTEP_A                                                           \
  {                                                                            \
    aie::block_vector<bfp16ebs8, 64> A0, A1, B0, B1;                           \
    A0 = sA.pop(); A1 = sA.pop(); B0 = sB.pop(); B1 = sB.pop();                \
    c00.mac(A0, aie::op_transpose(B0)); c01.mac(A0, aie::op_transpose(B1));    \
    c10.mac(A1, aie::op_transpose(B0)); c11.mac(A1, aie::op_transpose(B1));    \
  }

  for (int it = 0; it < NBR * NJ; it++) {
#define ZEST_BODY                                                              \
    ZEST_KSTEP_B ZEST_KSTEP_A ZEST_KSTEP_A ZEST_KSTEP_A                        \
    ZEST_KSTEP_B ZEST_KSTEP_A ZEST_KSTEP_A ZEST_KSTEP_A                        \
    aie::store_v(pCst, c00.template to_vector<TO>());                          \
    aie::store_v(pCst + 64, c01.template to_vector<TO>());                     \
    aie::store_v(pCst + colB * 64, c10.template to_vector<TO>());              \
    aie::store_v(pCst + colB * 64 + 64, c11.template to_vector<TO>());
    if constexpr (mode == Mode::First) {
      MMUL c00(aie::zeros<accfloat, 64>()), c01(aie::zeros<accfloat, 64>()),
          c10(aie::zeros<accfloat, 64>()), c11(aie::zeros<accfloat, 64>());
      ZEST_BODY
    } else {
      MMUL c00((acc_t(aie::load_v<64>(pCld))));
      MMUL c01((acc_t(aie::load_v<64>(pCld + 64))));
      MMUL c10((acc_t(aie::load_v<64>(pCld + colB * 64))));
      MMUL c11((acc_t(aie::load_v<64>(pCld + colB * 64 + 64))));
      ZEST_BODY
    }
    sA.seek(walkA);
    sB.seek(walkB);
    if constexpr (mode != Mode::First)
      pCld = add_2d_ptr(pCld, BAND, NJ - 1, cL, 128);
    pCst = add_2d_ptr(pCst, BAND, NJ - 1, cS, 128);
  }
#undef ZEST_BODY
#undef ZEST_KSTEP_A
#undef ZEST_KSTEP_B
}

template <typename T> static inline const T __aie_dm_resource_c *cld(const T *p) {
  return (const T __aie_dm_resource_c *)p;
}
template <typename T> static inline T __aie_dm_resource_c *cst(T *p) {
  return (T __aie_dm_resource_c *)p;
}

} // namespace

extern "C" {

#ifdef F32_FIRST
void mm_f32_first(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC) {
  mm_paired<Mode::First, float, float>(pA, pB, cld(pC), cst(pC));
}
#endif

#ifdef F32_MID
void mm_f32_mid(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC) {
  mm_paired<Mode::Mid, float, float>(pA, pB, cld(pC), cst(pC));
}
#endif

#ifdef F32_LAST
void mm_f32_last(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *__restrict pC,
                 bfloat16 *__restrict pOut) {
  mm_paired<Mode::Last, float, bfloat16>(pA, pB, cld(pC), cst(pOut));
}
#endif

// bf16 running sums for the 16-bit output modes (whole_array_bfp_rtp.py
// --bf16-sums): mode 0 keeps fp32 sums; modes 1 and 2 keep the tile's sums as
// bf16 at its front, 8x8 block s at bf16 element s * 64, rounded to nearest
// even after every K chunk. That is already mode 1's output layout
// (out_epilogue), so mode 1 needs no epilogue, and mode 2's reads bf16. The
// sums' rounding adds about as much error as bfp16's operands (specs/npu-qgemm).
#ifdef SUMS_FIRST
void mm_sums_first(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC, int32_t mode) {
  if (mode == 0)
    mm_paired<Mode::First, float, float>(pA, pB, cld(pC), cst(pC));
  else
    mm_paired<Mode::First, bfloat16, bfloat16>(pA, pB, cld((bfloat16 *)pC), cst((bfloat16 *)pC));
}
#endif

#ifdef SUMS_MID
void mm_sums_mid(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *pC, int32_t mode) {
  if (mode == 0)
    mm_paired<Mode::Mid, float, float>(pA, pB, cld(pC), cst(pC));
  else
    mm_paired<Mode::Mid, bfloat16, bfloat16>(pA, pB, cld((bfloat16 *)pC), cst((bfloat16 *)pC));
}
#endif

#ifdef BF16_FIRST
void mm_bf16_first(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, bfloat16 *pC) {
  mm_paired<Mode::First, bfloat16, bfloat16>(pA, pB, cld(pC), cst(pC));
}
#endif

#ifdef BF16_MID
void mm_bf16_mid(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, bfloat16 *pC) {
  mm_paired<Mode::Mid, bfloat16, bfloat16>(pA, pB, cld(pC), cst(pC));
}
#endif
}

#ifdef OUT_EPILOGUE
// The output step whole_array_bfp_rtp runs on a finished fp32 tile, chosen per
// call by a runtime parameter:
//   0: nothing, the tile goes out as fp32
//   1: every value rounded to bf16 (to nearest even)
//   2: SiLU(gate) * up in fp32, rounded to bf16. The weights are ordered so
//      that the tile's 8x8 blocks alternate: gate block, then the up block
//      that pairs with it (bfp16_pack.h, bfp16_interleave_gate_up)
// The bf16 results are packed from the front of the tile, in the tile's own
// block order: 8x8 block s (mode 1: block s; mode 2: pair s) at bf16 element
// s * 64, row-major inside. The rest of the tile is left as it was. Each step
// reads a stretch of the tile before writing a stretch no further along than
// what it has read, so the packing is safe in place.
//
// Every access goes through float pointers, so the compiler sees the reads and
// writes as aliasing and keeps their order.

namespace {

constexpr unsigned kTile = DIM_M * DIM_N;
static_assert(kTile % 128 == 0);

// fp32 multiplies are emulated on this core, nine bf16 passes each, which
// made an all-fp32 SiLU cost 3x the matmul. So the arithmetic below is bf16 x
// bf16 products summed in fp32 accumulators (native, exact products), and
// where more than bf16's 8 bits matter a value is carried as a pair of bf16,
// hi + lo, good to about 2^-17. Every step's error is kept below 2^-15 of the
// result, which then rounds once to bf16 (2^-9).
using vf = aie::vector<float, 32>;
using vb = aie::vector<bfloat16, 32>;
using vi = aie::vector<int32_t, 32>;
using va = aie::accum<accfloat, 32>;

struct hl {
  vb h, l;
};

__attribute__((always_inline)) inline va acc(vf v) {
  va a;
  a.from_vector(v);
  return a;
}
__attribute__((always_inline)) inline va accb(vb v) {
  va a;
  a.from_vector(v);
  return a;
}
__attribute__((always_inline)) inline va accc(float c) {
  return acc(aie::broadcast<float, 32>(c));
}
__attribute__((always_inline)) inline vb bc(float c) {
  return aie::broadcast<bfloat16, 32>(c);
}
__attribute__((always_inline)) inline vi ic(int32_t c) {
  return aie::broadcast<int32_t, 32>(c);
}
// a as hi + lo
__attribute__((always_inline)) inline hl split(va a) {
  vb h = a.template to_vector<bfloat16>();
  return {h, aie::msc(a, h, bc(1.0f)).template to_vector<bfloat16>()};
}
// c + a * b, dropping lo * lo (2^-16 of the product)
__attribute__((always_inline)) inline va mac3(va c, hl a, hl b) {
  return aie::mac(aie::mac(aie::mac(c, a.h, b.h), a.h, b.l), a.l, b.h);
}
__attribute__((always_inline)) inline va mul3(hl a, hl b) {
  return aie::mac(aie::mac(aie::mul(a.h, b.h), a.h, b.l), a.l, b.h);
}

// SiLU(g) * u = g * u * s, s = 1 / (1 + e^-g). With e = e^-|g| in (0, 1]:
// s = 1 / (1 + e) for g >= 0 and e / (1 + e) for g < 0, which never subtracts
// nearly equal numbers. |g| is capped at 44 for e, keeping 2^(-|g| log2 e) a
// normal float; past it s is 1, or far under anything bf16 can add to a sum.
__attribute__((always_inline)) inline vb swiglu32(vf g, vf u) {
  const hl gs = split(acc(g)), us = split(acc(u));
  // tn = -|g| * log2(e), log2(e) as hi + lo, g's hi capped at +-44: e = 2^tn.
  // Both signs are computed and the negative one kept (g's hi has g's sign).
  const aie::mask<32> neg = aie::lt(gs.h, bc(0.0f));
  const vb ch = aie::max(aie::min(gs.h, bc(44.0f)), bc(-44.0f));
  const vf tp = aie::mac(aie::mac(aie::mul(ch, bc(1.4453125f)), ch, bc(-0.0026174591f)), gs.l, bc(1.4453125f))
                    .template to_vector<float>();
  const vf tm = aie::msc(aie::msc(aie::negmul(ch, bc(1.4453125f)), ch, bc(-0.0026174591f)), gs.l, bc(1.4453125f))
                    .template to_vector<float>();
  const vf tn = aie::select(tm, tp, neg);
  // tn = k + f, k an integer, |f| <= 1/2
  const vi k = aie::to_fixed<int32_t>(tn); // rounds to nearest
  const hl f = split(aie::sub(acc(tn), acc(aie::to_float<float>(k))));
  // 2^f, degree 4 (2.7e-6). The inner terms are small enough for plain bf16.
  const vb r3 = aie::mac(accc(0.05591704f), f.h, bc(0.00956051f)).template to_vector<bfloat16>();
  const hl r2 = split(aie::mac(accc(0.24024981f), r3, f.h));
  const hl r1 = split(mac3(accc(0.69312197f), r2, f));
  const va p = mac3(accc(0.99999919f), r1, f);
  // e = 2^f * 2^k, k added straight into the exponent
  const vf e = aie::add(p.template to_vector<float>().template cast_to<int32_t>(), aie::upshift(k, 23))
                   .template cast_to<float>();
  const hl es = split(acc(e));
  // y = 1 / d, d = 1 + e in (1, 2]: a linear first guess (5.9%), then two
  // Newton steps, y += y * (1 - d * y), to about 2^-16
  const hl d = split(aie::add(acc(e), accc(1.0f)));
  const vb y0 = aie::msc(accc(1.4115f), d.h, bc(0.4705f)).template to_vector<bfloat16>();
  const vb e1 = aie::msc(aie::msc(accc(1.0f), d.h, y0), d.l, y0).template to_vector<bfloat16>();
  const va y1 = aie::mac(accb(y0), y0, e1);
  const hl y1s = split(y1);
  const vb e2 = aie::sub(accc(1.0f), mul3(d, y1s)).template to_vector<bfloat16>();
  const hl y = split(aie::mac(y1, y1s.h, e2));
  // s = y, or e * y where g < 0
  const hl num = {aie::select(bc(1.0f), es.h, neg), aie::select(bc(0.0f), es.l, neg)};
  const hl s = split(mul3(num, y));
  return mul3(split(mul3(gs, us)), s).template to_vector<bfloat16>();
}

// 32 floats -> 32 bf16, returned as the 16 floats' worth of bits they occupy
aie::vector<float, 16> to_bf16_bits(aie::vector<float, 32> v) {
  aie::accum<accfloat, 32> a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>().template cast_to<float>();
}

} // namespace

extern "C" void out_epilogue(float *pC, int32_t mode) {
  if (mode == 1) {
    ::aie::set_rounding(::aie::rounding_mode::conv_even);
    // 32 values in, 16 floats' worth out
    AIE_LOOP_MIN_ITERATION_COUNT(kTile / 32)
    for (unsigned i = 0; i < kTile / 32; i++)
      aie::store_v(pC + i * 16, to_bf16_bits(aie::load_v<32>(pC + i * 32)));
  } else if (mode == 2) {
    ::aie::set_rounding(::aie::rounding_mode::conv_even);
    // a gate block and its up block (128 values) in, 64 results out: the
    // first 32 of each and the second, two independent chains to interleave
    AIE_LOOP_MIN_ITERATION_COUNT(kTile / 128)
    for (unsigned s = 0; s < kTile / 128; s++) {
      const float *in = pC + s * 128;
      const vb lo = swiglu32(aie::load_v<32>(in), aie::load_v<32>(in + 64));
      const vb hi = swiglu32(aie::load_v<32>(in + 32), aie::load_v<32>(in + 96));
      aie::store_v(pC + s * 32, lo.template cast_to<float>());
      aie::store_v(pC + s * 32 + 16, hi.template cast_to<float>());
    }
  }
}

// out_epilogue for a bf16 C tile (whole_array_bfp_rtp.py --c16: mm_bf16_first
// / mm_bf16_mid sums, the tile itself bf16): the same steps as
// out_epilogue_sums.
extern "C" void out_epilogue_sums(float *pC, int32_t mode);
extern "C" void out_epilogue16(bfloat16 *pC, int32_t mode) {
  out_epilogue_sums((float *)pC, mode);
}

// out_epilogue for a tile from mm_sums_first / mm_sums_mid: mode 1 is
// already done; mode 2 reads each pair's gate and up blocks as bf16 (bf16
// element s * 128 on) and writes its 64 results at bf16 element s * 64,
// behind what it has read.
namespace {
__attribute__((always_inline)) inline vf bf16_at(const float *p) {
  return accb(aie::load_v<16>(p).template cast_to<bfloat16>()).template to_vector<float>();
}
} // namespace

extern "C" void out_epilogue_sums(float *pC, int32_t mode) {
  if (mode == 2) {
    ::aie::set_rounding(::aie::rounding_mode::conv_even);
    AIE_LOOP_MIN_ITERATION_COUNT(kTile / 128)
    for (unsigned s = 0; s < kTile / 128; s++) {
      const float *in = pC + s * 64;  // the pair's 128 bf16
      const vb lo = swiglu32(bf16_at(in), bf16_at(in + 32));
      const vb hi = swiglu32(bf16_at(in + 16), bf16_at(in + 48));
      aie::store_v(pC + s * 32, lo.template cast_to<float>());
      aie::store_v(pC + s * 32 + 16, hi.template cast_to<float>());
    }
  }
}
#endif
