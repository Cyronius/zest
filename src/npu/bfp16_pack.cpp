#include "bfp16_pack.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <functional>
#include <thread>

namespace {

void parallel_for(int64_t n, int nth, const std::function<void(int64_t, int64_t)> & fn) {
    nth = (int) std::max<int64_t>(1, std::min<int64_t>(nth, n));
    if (nth == 1) { fn(0, n); return; }
    const int64_t chunk = (n + nth - 1) / nth;
    std::vector<std::thread> pool;
    for (int t = 1; t < nth; t++) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b < e) pool.emplace_back(fn, b, e);
    }
    fn(0, std::min(n, chunk));
    for (auto & t : pool) t.join();
}

// Visits one tile's blocks in the order the core reads them, calling
// fn(row, block column) for each; the caller advances its own byte offset.
template <typename F>
void walk_tile(int64_t r0, int64_t kb0, int th, int tkb, F && fn) {
    for (int sp = 0; sp < th; sp += 16)
        for (int sx = 0; sx < tkb; sx++)
            for (int half = 0; half < 2; half++)
                for (int i = 0; i < 8; i++) fn(r0 + sp + half * 8 + i, kb0 + sx);
}

// A: for each block of m rows, K/k tiles, one after another. Split by tile,
// not by row block: a 512-row call has only four row blocks.
template <typename F>
void walk_a_run(int64_t M, int64_t K, const bfp16_tiling & t, const bfp16_runner & run, F && fn) {
    const int64_t nk = K / t.k, tile = (int64_t) t.m * (t.k / 8) * BFP16_BLOCK_BYTES;
    run((M / t.m) * nk, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; i++) {
            const int64_t rb = i / nk, kk = i % nk;
            size_t off = (size_t) (i * tile);
            walk_tile(rb * t.m, kk * (t.k / 8), t.m, t.k / 8, [&](int64_t r, int64_t c) {
                fn(r, c, off);
                off += BFP16_BLOCK_BYTES;
            });
        }
    });
}

template <typename F>
void walk_a(int64_t M, int64_t K, const bfp16_tiling & t, int nth, F && fn) {
    walk_a_run(M, K, t, [nth](int64_t n, const std::function<void(int64_t, int64_t)> & f) { parallel_for(n, nth, f); },
               std::forward<F>(fn));
}

// B: for each (array column, n-row group) pair, K/k tiles, one after
// another. Split by tile.
template <typename F>
void walk_b_run(int64_t N, int64_t K, const bfp16_tiling & t, const bfp16_runner & run, F && fn) {
    const int64_t per_col = N / t.n / t.cols, nk = K / t.k, tile = (int64_t) t.n * (t.k / 8) * BFP16_BLOCK_BYTES;
    run(t.cols * per_col * nk, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; i++) {
            const int64_t g = i / nk, kk = i % nk, c = g / per_col, tt = g % per_col;
            size_t off = (size_t) (i * tile);
            walk_tile((c + t.cols * tt) * t.n, kk * (t.k / 8), t.n, t.k / 8, [&](int64_t r, int64_t cb) {
                fn(r, cb, off);
                off += BFP16_BLOCK_BYTES;
            });
        }
    });
}

template <typename F>
void walk_b(int64_t N, int64_t K, const bfp16_tiling & t, int nth, F && fn) {
    walk_b_run(N, K, t, [nth](int64_t n, const std::function<void(int64_t, int64_t)> & f) { parallel_for(n, nth, f); },
               std::forward<F>(fn));
}

} // namespace

void bfp16_encode_block(const float * x, uint8_t * out) {
#ifdef __AVX2__
    // The same arithmetic eight lanes at a time: the scale 2^(133 - maxe) is
    // built from its bits, x * scale is exact (a power of two), and
    // cvtps_epi32 rounds to nearest even under the default rounding mode.
    const __m256  v    = _mm256_loadu_ps(x);
    const __m256i ex   = _mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(v), 23), _mm256_set1_epi32(0xff));
    __m128i m4 = _mm_max_epi32(_mm256_castsi256_si128(ex), _mm256_extracti128_si256(ex, 1));
    m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, _MM_SHUFFLE(1, 0, 3, 2)));
    m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, _MM_SHUFFLE(2, 3, 0, 1)));
    const int mx = _mm_cvtsi128_si32(m4);
    const int sexp = 133 - mx + 127;  // biased exponent of the scale
    if (sexp >= 2 && sexp <= 254) {
        out[0] = (uint8_t) mx;
        __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(sexp << 23))));
        // A value that rounds up to +128: the block again, one exponent up
        if (!_mm256_testz_si256(_mm256_cmpgt_epi32(q, _mm256_set1_epi32(127)), _mm256_set1_epi32(-1))) {
            out[0] = (uint8_t) (mx + 1);
            q = _mm256_cvtps_epi32(_mm256_mul_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32((sexp - 1) << 23))));
        }
        q = _mm256_min_epi32(_mm256_max_epi32(q, _mm256_set1_epi32(-128)), _mm256_set1_epi32(127));
        const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        _mm_storel_epi64((__m128i *) (out + 1), _mm_packs_epi16(q16, q16));
        return;
    }
    // Eight zeros (or negative zeros): common, as the padding rows that fill
    // out a call, and too slow on the scalar path below.
    const __m256i mag = _mm256_and_si256(_mm256_castps_si256(v), _mm256_set1_epi32(0x7fffffff));
    if (_mm256_testz_si256(mag, mag)) {
        memset(out, 0, BFP16_BLOCK_BYTES);
        return;
    }
    // exponents so small the scale is not a normal float: the scalar path
#endif
    uint32_t maxe = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t u;
        memcpy(&u, &x[i], 4);
        maxe = std::max(maxe, (u >> 23) & 0xffu);
    }
    int sh = 133 - (int) maxe;
    float q[8];
    bool over = false;
    for (int i = 0; i < 8; i++) {
        q[i] = std::nearbyint(std::ldexp(x[i], sh));
        over |= q[i] > 127.0f;
    }
    if (over) {  // a value rounds up to +128: the block again, one exponent up
        maxe++;
        sh--;
        for (int i = 0; i < 8; i++) q[i] = std::nearbyint(std::ldexp(x[i], sh));
    }
    out[0] = (uint8_t) maxe;
    for (int i = 0; i < 8; i++) out[1 + i] = (uint8_t) (int8_t) std::min(127.0f, std::max(-128.0f, q[i]));
}

#if defined(__AVX2__) && defined(__GNUC__) && defined(__x86_64__)
// The activation tile offers pairs of neighboring rows at each K block. Zen 5
// can convert both blocks with one 512-bit multiply, conversion, clamp and
// saturating byte pack. The scalar exponent reduction remains separate because
// each eight-value block has its own shared exponent.
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
static void bfp16_encode_pair_avx512(const float * x0, const float * x1, uint8_t * out0, uint8_t * out1) {
    const __m256 v0 = _mm256_loadu_ps(x0), v1 = _mm256_loadu_ps(x1);
    auto max_exp = [](__m256 v) {
        const __m256i e = _mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(v), 23), _mm256_set1_epi32(0xff));
        __m128i m = _mm_max_epi32(_mm256_castsi256_si128(e), _mm256_extracti128_si256(e, 1));
        m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
        m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2, 3, 0, 1)));
        return _mm_cvtsi128_si32(m);
    };
    const int e0 = max_exp(v0), e1 = max_exp(v1);
    const int s0 = 133 - e0 + 127, s1 = 133 - e1 + 127;
    if (s0 < 2 || s0 > 254 || s1 < 2 || s1 > 254) {
        bfp16_encode_block(x0, out0);
        bfp16_encode_block(x1, out1);
        return;
    }
    const __m512 v = _mm512_insertf32x8(_mm512_castps256_ps512(v0), v1, 1);
    const __m512i scales = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(s0 << 23), _mm512_set1_epi32(s1 << 23));
    __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v, _mm512_castsi512_ps(scales)));
    if (_mm512_cmpgt_epi32_mask(q, _mm512_set1_epi32(127))) {  // a block steps its exponent up (rare)
        bfp16_encode_block(x0, out0);
        bfp16_encode_block(x1, out1);
        return;
    }
    q = _mm512_min_epi32(_mm512_max_epi32(q, _mm512_set1_epi32(-128)), _mm512_set1_epi32(127));
    const __m128i bytes = _mm512_cvtsepi32_epi8(q);
    out0[0] = (uint8_t) e0;
    out1[0] = (uint8_t) e1;
    _mm_storel_epi64((__m128i *) (out0 + 1), bytes);
    _mm_storel_epi64((__m128i *) (out1 + 1), _mm_srli_si128(bytes, 8));
}

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
static void bfp16_pack_a_tile_rows_avx512(const float * const * rows, int64_t r0, int64_t count,
                                           int64_t K, const bfp16_tiling & t, uint8_t * out) {
    const int64_t tkb = t.k / 8, nk = K / t.k, tile_blocks = t.m * tkb;
    const int64_t base = (r0 / t.m) * nk * tile_blocks + ((r0 % t.m) / 16) * tkb * 16 + r0 % 16;
    for (int64_t kk = 0; kk < nk; kk++) {
        const int64_t first_cb = kk * tkb;
        for (int64_t sx = 0; sx < tkb; sx++) {
            uint8_t * dst = out + (base + kk * tile_blocks + sx * 16) * BFP16_BLOCK_BYTES;
            const int64_t c = (first_cb + sx) * 8;
            int64_t i = 0;
            for (; i + 1 < count; i += 2)
                bfp16_encode_pair_avx512(rows[i] + c, rows[i + 1] + c,
                                         dst + i * BFP16_BLOCK_BYTES, dst + (i + 1) * BFP16_BLOCK_BYTES);
            if (i < count) bfp16_encode_block(rows[i] + c, dst + i * BFP16_BLOCK_BYTES);
        }
    }
}
#endif

void bfp16_pack_a_tile_rows(const float * const * rows, int64_t r0, int64_t count,
                           int64_t K, const bfp16_tiling & t, uint8_t * out) {
#if defined(__AVX2__) && defined(__GNUC__) && defined(__x86_64__)
    static const bool avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                               __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
    if (avx512) {
        bfp16_pack_a_tile_rows_avx512(rows, r0, count, K, t, out);
        return;
    }
#endif
    const int64_t tkb = t.k / 8, nk = K / t.k, tile_blocks = t.m * tkb;
    const int64_t base = (r0 / t.m) * nk * tile_blocks + ((r0 % t.m) / 16) * tkb * 16 + r0 % 16;
    for (int64_t kk = 0; kk < nk; kk++) {
        const int64_t first_cb = kk * tkb;
        for (int64_t sx = 0; sx < tkb; sx++) {
            uint8_t * dst = out + (base + kk * tile_blocks + sx * 16) * BFP16_BLOCK_BYTES;
            const int64_t c = (first_cb + sx) * 8;
            for (int64_t i = 0; i < count; i++)
                bfp16_encode_block(rows[i] + c, dst + i * BFP16_BLOCK_BYTES);
        }
    }
}

void bfp16_decode_block(const uint8_t * in, float * x) {
#ifdef __AVX2__
    // mantissa * 2^(exponent - 133), with the scale built from its bits when
    // it is a normal float (exponent byte >= 7)
    if (in[0] >= 7) {
        int64_t m;
        memcpy(&m, in + 1, 8);
        const __m256 q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(m)));
        const __m256 scale = _mm256_castsi256_ps(_mm256_set1_epi32(((int) in[0] - 133 + 127) << 23));
        _mm256_storeu_ps(x, _mm256_mul_ps(q, scale));
        return;
    }
#endif
    const int sh = (int) in[0] - 133;
    for (int i = 0; i < 8; i++) x[i] = std::ldexp((float) (int8_t) in[1 + i], sh);
}

void bfp16_pack_a(const float * x, int64_t M, int64_t K, const bfp16_tiling & t, std::vector<uint8_t> & out,
                  int n_threads) {
    out.resize((size_t) (M * K / 8 * BFP16_BLOCK_BYTES));
    uint8_t * o = out.data();
    walk_a(M, K, t, n_threads, [&](int64_t r, int64_t c, size_t off) { bfp16_encode_block(x + r * K + c * 8, o + off); });
}

void bfp16_pack_a_into(const float * x, int64_t M, int64_t K, const bfp16_tiling & t, uint8_t * out,
                       const bfp16_runner & run) {
    walk_a_run(M, K, t, run, [&](int64_t r, int64_t c, size_t off) { bfp16_encode_block(x + r * K + c * 8, out + off); });
}

void bfp16_pack_b(const float * w, int64_t N, int64_t K, const bfp16_tiling & t, std::vector<uint8_t> & out,
                  int n_threads) {
    out.resize((size_t) (N * K / 8 * BFP16_BLOCK_BYTES));
    uint8_t * o = out.data();
    walk_b(N, K, t, n_threads, [&](int64_t r, int64_t c, size_t off) { bfp16_encode_block(w + r * K + c * 8, o + off); });
}

void bfp16_pack_b_into(const float * w, int64_t N, int64_t K, const bfp16_tiling & t, uint8_t * out,
                       const bfp16_runner & run) {
    walk_b_run(N, K, t, run, [&](int64_t r, int64_t c, size_t off) { bfp16_encode_block(w + r * K + c * 8, out + off); });
}

void bfp16_pack_b_rows_into(int64_t N, int64_t K, const bfp16_tiling & t, uint8_t * out,
                            const bfp16_runner & run, const bfp16_row_loader & load_row) {
    run(N, [&](int64_t r0, int64_t r1) {
        std::vector<float> row((size_t) K);
        for (int64_t r = r0; r < r1; r++) {
            load_row(r, row.data());
            bfp16_b_row(r, N, K, t, [&](int64_t cb, size_t off) {
                bfp16_encode_block(row.data() + cb * 8, out + off);
            });
        }
    });
}

// Inside a tile, walk_tile's order: row pair (16 rows) by row pair, block
// column by block column, then the pair's 16 rows.
static int64_t in_tile(int64_t rr, int64_t cb, const bfp16_tiling & t) {
    const int64_t tkb = t.k / 8;
    return ((rr / 16) * tkb + cb % tkb) * 16 + rr % 16;
}

size_t bfp16_a_offset(int64_t r, int64_t cb, int64_t K, const bfp16_tiling & t) {
    const int64_t nk = K / t.k, tile = (r / t.m) * nk + cb / (t.k / 8);
    return (size_t) (tile * t.m * (t.k / 8) + in_tile(r % t.m, cb, t)) * BFP16_BLOCK_BYTES;
}

size_t bfp16_b_offset(int64_t r, int64_t cb, int64_t N, int64_t K, const bfp16_tiling & t) {
    const int64_t per_col = N / t.n / t.cols, nk = K / t.k, g = r / t.n;
    const int64_t tile = ((g % t.cols) * per_col + g / t.cols) * nk + cb / (t.k / 8);
    return (size_t) (tile * t.n * (t.k / 8) + in_tile(r % t.n, cb, t)) * BFP16_BLOCK_BYTES;
}

void bfp16_c_row(const float * c, int64_t M, int64_t N, const bfp16_tiling & t, int64_t r, int64_t n_cols, float * row) {
    const int64_t rb = (int64_t) t.m * t.rows, blk = rb * t.n, J = N / ((int64_t) t.n * t.cols);
    const float * base = c + (r / rb) * J * blk + (r % rb) * t.n;
    for (int64_t ct = 0; ct < n_cols / t.n; ct++)
        memcpy(row + ct * t.n, base + (ct % t.cols) * (M * N / t.cols) + (ct / t.cols) * blk, (size_t) t.n * sizeof(float));
}

void bfp16_c16_row(const float * c, int64_t M, int64_t N, const bfp16_tiling & t, int mode, int64_t r, float * row) {
    const int64_t rb = (int64_t) t.m * t.rows, blk = rb * t.n, J = N / ((int64_t) t.n * t.cols);
    const int64_t cb = t.n / 8;                        // 8x8 blocks across a tile
    const int64_t groups = mode == 2 ? cb / 2 : cb;    // 8-value results a tile row holds
    const int64_t zr = (r % t.m) / 8, rr = r % 8;
    // Where each of the row's groups sits inside its tile, as a float offset
    // from the tile's first row. Group g is 8x8 block s = zr * groups + g of
    // the bf16 results, whose row rr is 16 bytes at fp32 word s * 32 + rr * 4
    // of the core's blocked tile; that word's place in the tile's row-major
    // rows is (block row, block column) of fp32 block w / 64.
    int64_t at[32];  // n <= 256
    for (int64_t g = 0; g < groups; g++) {
        const int64_t s = zr * groups + g, w = s * 32 + rr * 4, fb = w / 64, wb = w % 64;
        at[g] = ((fb / cb) * 8 + wb / 8) * t.n + (fb % cb) * 8 + wb % 8;
    }
    // the tile's first row: the core tile holding row r, in array column ct % cols
    const int64_t r0 = r - r % t.m;
    const float * base = c + (r0 / rb) * J * blk + (r0 % rb) * t.n;
    for (int64_t ct = 0; ct < N / t.n; ct++) {
        const uint16_t * tile = (const uint16_t *) (base + (ct % t.cols) * (M * N / t.cols) + (ct / t.cols) * blk);
        float * out = row + ct * groups * 8;
        for (int64_t g = 0; g < groups; g++) {
            const uint16_t * h = (const uint16_t *) ((const float *) tile + at[g]);
#ifdef __AVX2__
            const __m256i x = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) h));
            _mm256_storeu_ps(out + g * 8, _mm256_castsi256_ps(_mm256_slli_epi32(x, 16)));
#else
            for (int i = 0; i < 8; i++) {
                const uint32_t bits = (uint32_t) h[i] << 16;
                memcpy(out + g * 8 + i, &bits, 4);
            }
#endif
        }
    }
}

void bfp16_interleave_gate_up(const float * gate, const float * up, int64_t F, int64_t K, float * out) {
    for (int64_t i = 0; i < F; i++) {
        const int64_t at = (i / 8) * 16 + i % 8;
        std::copy(gate + i * K, gate + (i + 1) * K, out + at * K);
        std::copy(up + i * K, up + (i + 1) * K, out + (at + 8) * K);
    }
}

void bfp16_unpack_a(const uint8_t * in, int64_t M, int64_t K, const bfp16_tiling & t, float * x) {
    walk_a(M, K, t, 1, [&](int64_t r, int64_t c, size_t off) { bfp16_decode_block(in + off, x + r * K + c * 8); });
}

void bfp16_unpack_b(const uint8_t * in, int64_t N, int64_t K, const bfp16_tiling & t, float * w) {
    walk_b(N, K, t, 1, [&](int64_t r, int64_t c, size_t off) { bfp16_decode_block(in + off, w + r * K + c * 8); });
}
