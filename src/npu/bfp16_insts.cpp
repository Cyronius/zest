#include "bfp16_insts.h"
#include "insts_stream.h"

#include <algorithm>

namespace {

constexpr int64_t TM = 128, TK = 64, TN = 64, ROWS = 4, COLS = 8;
constexpr int64_t BLOCK_BYTES = 9;  // eight values sharing an exponent

// Where the toolchain placed each column's streams: the shim column that
// drains C (on its S2MM channel 0), and the shim column and MM2S channel that
// feed A (columns 0-3 only, one per core row) and B. Fixed by the design, not
// by the size.
constexpr int C_SHIM[COLS] = { 3, 2, 4, 1, 5, 6, 7, 0 };
constexpr int A_SHIM[ROWS][2] = { { 0, 0 }, { 2, 0 }, { 4, 0 }, { 6, 0 } };
constexpr int B_SHIM[COLS][2] = { { 3, 0 }, { 3, 1 }, { 2, 1 }, { 4, 1 }, { 5, 0 }, { 5, 1 }, { 6, 1 }, { 1, 0 } };

} // namespace

using namespace npu_insts;

bool bfp16_insts_fits(int64_t M, int64_t K, int64_t N) {
    // 68 output tile groups cover Qwen3.8-27B's 17,408-wide gate and up
    // weights in one fused 34,816-wide kernel invocation.
    return M > 0 && M % (TM * ROWS) == 0 && N > 0 && N % (TN * COLS) == 0 && N / (TN * COLS) <= 68 && K % TK == 0 &&
           K >= 2 * TK;
}

std::vector<uint32_t> bfp16_insts(int64_t M, int64_t K, int64_t N, int mode) {
    if (!bfp16_insts_fits(M, K, N) || mode < 0 || mode > 2) return {};
    stream s;
    for (int col = 0; col < COLS; col++)
        for (int row = 2; row < 2 + ROWS; row++) {
            s.write32(tile_addr(col, row, RTP_NMID), (uint32_t) (K / TK - 1));
            s.write32(tile_addr(col, row, RTP_MODE), (uint32_t) mode);
        }

    const int64_t row_blocks = M / (TM * ROWS);                 // blocks of 512 rows
    const int64_t per_col = N / TN / COLS;                      // n-wide output tiles per column
    const int64_t c_per_rb = per_col * TM * ROWS * TN;          // floats one column drains per row block
    const int64_t a_blocks = TM * (K / 8), b_blocks = (N / COLS) * (K / 8);
    // A drain covering two row blocks would stride past the shim's 2^20-word
    // limit when N > 2048, so the design drains one row block at a time there.
    const int64_t tb_max = TM * ROWS * N > (1 << 20) ? 2 : 4;

    for (int64_t tb = 0; tb < (row_blocks + tb_max - 1) / tb_max; tb++) {
        for (int pp = 0; pp < 2; pp++) {
            const int64_t row_base = tb * tb_max + pp * tb_max / 2;
            if (row_base >= row_blocks) break;
            const int64_t rows = std::min(tb_max / 2, row_blocks - row_base);
            for (int col = 0; col < COLS; col++) {
                // C: this column's run of the --c-tiled output
                s.task(C_SHIM[col], ARG_C, (uint32_t) ((col * (M * N / COLS) + row_base * c_per_rb) * 4),
                       (uint32_t) (rows * c_per_rb), S2MM0_QUEUE, 1, true);
                for (int64_t r = 0; r < rows; r++) {
                    // A: a block of m rows, once per output tile of the column
                    if (col < ROWS) {
                        const int64_t t = ((row_base + r) * ROWS + col) % (M / TM);
                        s.task(A_SHIM[col][0], ARG_A, (uint32_t) (t * a_blocks * BLOCK_BYTES),
                               (uint32_t) (a_blocks * BLOCK_BYTES / 4), A_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE,
                               (uint32_t) per_col, false);
                    }
                    // B: the column's weight rows
                    s.task(B_SHIM[col][0], ARG_B, (uint32_t) (col * b_blocks * BLOCK_BYTES),
                           (uint32_t) (b_blocks * BLOCK_BYTES / 4), B_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false);
                }
            }
            if (tb > 0 || pp > 0) s.finish();
        }
    }
    s.finish();

    return finish_stream(s);
}
