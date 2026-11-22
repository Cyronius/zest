// IQ3_S GGUF superblocks -> Zest's GEMM-ordered BFP16 weight tile.
// A single prototype tile is 64 output rows by 256 K values. Each raw row
// is one 110-byte IQ3_S superblock; the 18,432-byte output is four 64x64
// BFP tiles in the order whole_array_bfp_rtp.py consumes them.
//
// This decoder establishes the exact packing contract before a vector
// decoder and a streamed GEMM share the array. It does not replace the
// production BFP16 weight path yet.

#include <stdint.h>

#include "iq3s_grid.inc"

namespace {

void encode_block(const uint8_t *row, int k0, uint8_t *dst,
                  uint32_t dm, int de, bool dneg) {
  const int g = k0 >> 2;
  const uint8_t high = row[66 + (g >> 3)];
  const unsigned index0 = row[2 + g] | (((high >> (g & 7)) & 1u) << 8);
  const unsigned index1 = row[3 + g] | (((high >> ((g + 1) & 7)) & 1u) << 8);
  const int sc = (row[106 + (k0 >> 6)] >> ((k0 & 32) ? 4 : 0)) & 15;
  const uint32_t factor = dm * (uint32_t)(1 + 2 * sc);
  const uint8_t sign_bits = row[74 + (k0 >> 3)];
  uint32_t products[8];
  uint32_t largest = 0;
  for (int j = 0; j < 8; j++) {
    const uint32_t p = factor * (uint32_t)iq3s_grid[j < 4 ? index0 : index1][j & 3];
    products[j] = p;
    if (p > largest) largest = p;
  }
  if (!largest) {
    for (int j = 0; j < 9; j++) dst[j] = 0;
    return;
  }
  const int top = 31 - __builtin_clz(largest);
  dst[0] = (uint8_t)(127 + de + top);
  const int shift = top - 6;
  for (int j = 0; j < 8; j++) {
    uint32_t q = products[j];
    if (shift > 0) {
      const uint32_t mask = (1u << shift) - 1u;
      const uint32_t rem = q & mask;
      q >>= shift;
      const uint32_t halfway = 1u << (shift - 1);
      if (rem > halfway || (rem == halfway && (q & 1u))) q++;
    } else if (shift < 0) {
      q <<= -shift;
    }
    const bool neg = dneg != (((sign_bits >> j) & 1u) != 0);
    if (q > (neg ? 128u : 127u)) q = neg ? 128u : 127u;
    dst[1 + j] = (uint8_t)(neg ? -(int)q : (int)q);
  }
}

} // namespace

extern "C" void iq3s_to_bfp(const uint8_t *__restrict raw, uint8_t *__restrict out) {
  // Keep each 110-byte superblock and its scale in the core's working set
  // while visiting its 32 BFP blocks. Scatter to the GEMM tile order.
  for (int r = 0; r < 64; r++) {
    const uint8_t *row = raw + r * 110;
    const uint16_t h = (uint16_t)row[0] | ((uint16_t)row[1] << 8);
    const uint32_t he = (h >> 10) & 31u;
    const uint32_t dm = (h & 1023u) | (he ? 1024u : 0u);
    const int de = (he ? (int)he : 1) - 25;
    const bool dneg = (h & 0x8000u) != 0;
    const int sp = r >> 4, half = (r >> 3) & 1, i = r & 7;
    const int row_slot = (sp * 8 * 16 + half * 8 + i) * 9;
    for (int kk = 0; kk < 4; kk++) {
      uint8_t *dst = out + kk * 4608 + row_slot;
      for (int sx = 0; sx < 8; sx++) {
        encode_block(row, kk * 64 + sx * 8, dst, dm, de, dneg);
        dst += 16 * 9;
      }
    }
  }
}
