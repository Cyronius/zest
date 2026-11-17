#include "qtile_pack.h"

#include <cstring>

namespace {

// fp16 bits -> fp32 bits (exact; subnormals normalised, inf / NaN kept)
uint32_t fp16_to_fp32_bits(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 31, man = h & 1023;
    if (exp == 31) return sign | 0x7f800000u | (man << 13);
    if (exp == 0) {
        if (man == 0) return sign;
        exp = 113;  // 127 - 15 + 1, then normalise
        while (!(man & 1024)) {
            man <<= 1;
            exp--;
        }
        return sign | (exp << 23) | ((man & 1023) << 13);
    }
    return sign | ((exp + 112) << 23) | (man << 13);
}

float bits_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
uint32_t float_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// an fp16 value as bf16 hi + lo: hi its upper 16 bits, lo the rest (at most
// 3 significant bits, so a bf16 exactly)
void split_fp16(const uint8_t * p, uint16_t & hi, uint16_t & lo) {
    const uint32_t f = fp16_to_fp32_bits((uint16_t) (p[0] | (p[1] << 8)));
    const uint32_t h = f & 0xffff0000u;
    hi = (uint16_t) (h >> 16);
    lo = (uint16_t) (float_to_bits(bits_to_float(f) - bits_to_float(h)) >> 16);
}

// ggml's get_scale_min_k4
void scale_min(int j, const uint8_t * q, uint8_t & sc, uint8_t & m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

constexpr int HDR = 0, SC = 512, MN = 1024, QS = 1536;

} // namespace

bool qtile_q4k_fits(int64_t N, int64_t K) {
    return N > 0 && K > 0 && N % (QTILE_ROWS * QTILE_COLS) == 0 && K % QTILE_K == 0;
}

void qtile_pack_q4k_tile(const uint8_t * const * blocks, uint8_t * tile) {
    uint16_t * hdr = (uint16_t *) (tile + HDR);
    for (int r = 0; r < 64; r++) {
        const uint8_t * b = blocks[r];
        split_fp16(b, hdr[r], hdr[64 + r]);
        split_fp16(b + 2, hdr[128 + r], hdr[192 + r]);
        for (int s = 0; s < 8; s++) scale_min(s, b + 4, tile[SC + s * 64 + r], tile[MN + s * 64 + r]);
    }
    // nibbles: GGUF byte 32j + l of a row holds value 64j + l (low nibble)
    // and 64j + 32 + l (high); vector (j, g, x = x' + 4 hs) takes values
    // 64j + 8x .. +7 of rows 8g..8g+7
    uint8_t * q = tile + QS;
    for (int j = 0; j < 4; j++)
        for (int g = 0; g < 8; g++)
            for (int hs = 0; hs < 2; hs++)
                for (int xp = 0; xp < 4; xp++, q += 32)
                    for (int i = 0; i < 8; i++) {  // row 8g + i: lanes 8i .. 8i + 7, bytes 4i .. 4i + 3
                        const uint8_t * src = blocks[8 * g + i] + 16 + 32 * j + 8 * xp;
                        for (int k = 0; k < 4; k++) {
                            const uint8_t a = src[2 * k], b = src[2 * k + 1];
                            q[4 * i + k] = hs ? (uint8_t) ((a >> 4) | (b & 0xf0)) : (uint8_t) ((a & 15) | (b << 4));
                        }
                    }
}

void qtile_pack_q4k(const uint8_t * raw, size_t row_stride, int64_t N, int64_t K, uint8_t * out,
                    const bfp16_runner & run) {
    const int64_t nsb = K / QTILE_K, ntiles = N / QTILE_ROWS * nsb;
    run(ntiles, [&](int64_t t0, int64_t t1) {
        const uint8_t * blocks[64];
        for (int64_t t = t0; t < t1; t++) {
            const int64_t rg = t / nsb, sb = t % nsb;
            for (int r = 0; r < 64; r++) blocks[r] = raw + (rg * 64 + r) * row_stride + sb * Q4K_BLOCK_BYTES;
            qtile_pack_q4k_tile(blocks, out + qtile_index(rg, sb, N, K) * QTILE_Q4K_BYTES);
        }
    });
}

void qtile_unpack_q4k(const uint8_t * tile, uint8_t * out) {
    // the kernel's arithmetic gives ggml's: w = d1 * q - m1 in fp32, d1 =
    // d * sc and m1 = dmin * m exact, one rounding
    const uint16_t * hdr = (const uint16_t *) (tile + HDR);
    float d[64], dmin[64];
    for (int r = 0; r < 64; r++) {
        d[r] = bits_to_float((uint32_t) hdr[r] << 16) + bits_to_float((uint32_t) hdr[64 + r] << 16);
        dmin[r] = bits_to_float((uint32_t) hdr[128 + r] << 16) + bits_to_float((uint32_t) hdr[192 + r] << 16);
    }
    const uint8_t * q = tile + QS;
    for (int j = 0; j < 4; j++)
        for (int n = 0; n < 64; n++, q += 32) {
            // vector n = (g * 2 + hs) * 4 + x' of the nibbles, values
            // 8x .. 8x + 7 (x = x' + 4 hs) of rows 8g ..: the B tile keeps
            // it at chunk (g / 2) * 16 + x * 2 + g % 2
            const int g = n >> 3, hs = (n >> 2) & 1, s = 2 * j + hs, x = (n & 3) + 4 * hs;
            uint8_t * dst = out + (size_t) j * 4608 + (size_t) ((g >> 1) * 16 + x * 2 + (g & 1)) * 72;
            for (int i = 0; i < 8; i++) {  // one row's block
                const int r = 8 * g + i;
                const float d1 = d[r] * (float) tile[SC + s * 64 + r], m1 = dmin[r] * (float) tile[MN + s * 64 + r];
                float w[8];
                for (int k = 0; k < 8; k++) {
                    const int lane = i * 8 + k;
                    const int v = (q[lane / 2] >> (4 * (lane & 1))) & 15;
                    w[k] = d1 * (float) v - m1;  // the product is exact, so an FMA gives the same
                }
                bfp16_encode_block(w, dst + i * 9);
            }
        }
}
