// The Q4_K re-layout for the NPU's unpacker (src/npu/qtile_pack.h): what the
// unpacker makes of the tiles (qtile_unpack_q4k, the host model of
// dq_q4k.cc) must equal bfp16_encode_block of ggml's own dequantization,
// block for block, and land where bfp16_pack_b puts it. Data: ggml-quantized
// weights, and raw blocks of random bytes (negative and subnormal scales,
// zero scales and mins, every nibble).
//
// Traces: HYBRID-Q4-PACK
#include "ggml.h"
#include "bfp16_pack.h"
#include "qtile_pack.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const bfp16_runner serial = [](int64_t n, const std::function<void(int64_t, int64_t)> & fn) { fn(0, n); };

// rows x K of raw Q4_K -> the B layout bfp16_pack_b makes of ggml's dequantization
static std::vector<uint8_t> reference_b(const std::vector<uint8_t> & raw, int64_t rows, int64_t K) {
    std::vector<float> f((size_t) rows * K);
    ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(raw.data(), f.data(), rows * K);
    std::vector<uint8_t> b;
    bfp16_pack_b(f.data(), rows, K, bfp16_tiling{}, b, 1);
    return b;
}

// tiles -> what the NPU would put in B, rearranged into bfp16_pack_b's order
// (column c owns row groups c, c + 8, ...; per row group its K/64 k-tiles)
static std::vector<uint8_t> unpacked_b(const std::vector<uint8_t> & tiles, int64_t rows, int64_t K) {
    const int64_t nsb = K / 256, groups = rows / 64, per_col = groups / 8;
    std::vector<uint8_t> b((size_t) rows * K / 8 * 9), kt(4 * 4608);
    for (int64_t rg = 0; rg < groups; rg++)
        for (int64_t sb = 0; sb < nsb; sb++) {
            qtile_unpack_q4k(tiles.data() + (rg * nsb + sb) * QTILE_Q4K_BYTES, kt.data());
            for (int j = 0; j < 4; j++) {
                const int64_t pos = ((rg % 8) * per_col + rg / 8) * (K / 64) + sb * 4 + j;
                for (int n = 0; n < 64; n++)
                    memcpy(b.data() + pos * 4608 + qtile_b_chunk(n) * 72, kt.data() + j * 4608 + n * 72, 72);
            }
        }
    return b;
}

static int check_rows(const char * what, const std::vector<uint8_t> & raw, int64_t rows, int64_t K) {
    std::vector<uint8_t> tiles((size_t) rows / 64 * (K / 256) * QTILE_Q4K_BYTES);
    qtile_pack_q4k(raw.data(), (size_t) K / 256 * Q4K_BLOCK_BYTES, rows, K, tiles.data(), serial);
    const std::vector<uint8_t> want = reference_b(raw, rows, K), got = unpacked_b(tiles, rows, K);
    int64_t bad = 0;
    for (size_t i = 0; i < want.size(); i += 9) bad += memcmp(&want[i], &got[i], 9) != 0;
    const bool ok = bad == 0 && tiles.size() == (size_t) rows / 64 * (K / 256) * QTILE_Q4K_BYTES;
    printf("%s %s: %lld of %zu blocks differ\n", ok ? "PASS" : "FAIL", what, (long long) bad, want.size() / 9);
    return ok ? 0 : 1;
}

int main() {
    int failures = 0;
    std::mt19937 rng(7);

    // weights as ggml quantizes them: rows with different spreads
    {
        const int64_t rows = 512, K = 1024;
        std::vector<float> w((size_t) rows * K);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int64_t r = 0; r < rows; r++) {
            const float s = std::ldexp(1.0f, (int) (r % 23) - 16);
            for (int64_t k = 0; k < K; k++) w[r * K + k] = nd(rng) * s + (r % 5 == 0 ? 0.5f * s : 0.0f);
        }
        std::vector<uint8_t> raw(ggml_row_size(GGML_TYPE_Q4_K, K) * rows);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, w.data(), raw.data(), 0, rows, K, nullptr);
        failures += check_rows("ggml-quantized rows", raw, rows, K);
    }

    // random raw blocks: any finite d and dmin (negative, subnormal, zero),
    // any scales, mins and nibbles
    {
        const int64_t rows = 512, K = 512;
        std::vector<uint8_t> raw((size_t) rows * K / 256 * Q4K_BLOCK_BYTES);
        for (auto & b : raw) b = (uint8_t) rng();
        for (size_t i = 0; i < raw.size(); i += Q4K_BLOCK_BYTES)
            for (int h = 0; h < 2; h++) {
                uint8_t * p = &raw[i + 2 * h];
                if (((p[1] >> 2) & 31) == 31) p[1] &= 0x83;  // no inf / NaN: exponent 31 -> 0 (subnormal)
                if (rng() % 8 == 0) p[0] = p[1] = 0;          // zero
            }
        failures += check_rows("random raw blocks", raw, rows, K);
    }

    // the pack is byte moves and bit splits: same bytes however it is split up
    {
        const int64_t rows = 128, K = 512;
        std::vector<uint8_t> raw((size_t) rows * K / 256 * Q4K_BLOCK_BYTES);
        for (auto & b : raw) b = (uint8_t) (rng() & 0x7b);  // (keeps d finite)
        std::vector<uint8_t> a((size_t) rows / 64 * 2 * QTILE_Q4K_BYTES), b2(a.size());
        qtile_pack_q4k(raw.data(), K / 256 * Q4K_BLOCK_BYTES, rows, K, a.data(), serial);
        const bfp16_runner chunks = [](int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
            for (int64_t i = n; i > 0; i--) fn(i - 1, i);
        };
        qtile_pack_q4k(raw.data(), K / 256 * Q4K_BLOCK_BYTES, rows, K, b2.data(), chunks);
        const bool ok = a == b2;
        printf("%s packing in any order of tiles gives the same bytes\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    // vector order -> B chunks: a permutation of the 64 chunks
    {
        bool seen[64] = {};
        bool ok = true;
        for (int n = 0; n < 64; n++) {
            const int c = qtile_b_chunk(n);
            ok = ok && c >= 0 && c < 64 && !seen[c];
            if (c >= 0 && c < 64) seen[c] = true;
        }
        printf("%s the unpacker's vector order maps onto the B tile's 64 chunks once each\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    printf(failures ? "FAILED\n" : "ALL PASS\n");
    return failures ? 1 : 0;
}
