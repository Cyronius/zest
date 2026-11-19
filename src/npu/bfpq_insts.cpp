#include "bfpq_insts.h"
#include "insts_stream.h"
#include "qtile_pack.h"

#include <algorithm>

namespace {

constexpr int64_t TM = 128, TN = 64, ROWS = 3, COLS = 8;  // multiplying rows
constexpr int64_t BLOCK_BYTES = 9, A_TK = 64;

// Where the toolchain placed the streams: the shim column draining each
// column's C (S2MM 0), the shim column and MM2S channel feeding each
// multiplying row's A and each column's weights. Fixed by the design.
constexpr int C_SHIM[COLS] = { 0, 1, 2, 3, 4, 5, 6, 7 };
constexpr int A_SHIM[ROWS][2] = { { 3, 0 }, { 3, 1 }, { 2, 0 } };
constexpr int Q_SHIM[COLS][2] = { { 0, 0 }, { 1, 0 }, { 2, 1 }, { 4, 0 }, { 4, 1 }, { 5, 0 }, { 6, 0 }, { 7, 0 } };
// Unpacking core tile (row 2): this dispatch's weight objects per 64 rows,
// less one, their format, and B tiles an object
constexpr uint32_t RTP_DQ_OBJ = 0x9200, RTP_DQ_FMT = 0x9204, RTP_DQ_PER = 0x9208;

int64_t per_object(int fmt) { return fmt == BFPQ_Q4K ? 4 : 2; }

} // namespace

using namespace npu_insts;

bool bfpq_insts_fits(int64_t M, int64_t K, int64_t N, int fmt) {
    return (fmt == BFPQ_Q4K || fmt == BFPQ_BFP16) && M > 0 && M % (TM * ROWS) == 0 && N > 0 && N % (TN * COLS) == 0 &&
           N / (TN * COLS) <= 68 && K > 0 && K % (A_TK * per_object(fmt)) == 0;
}

std::vector<uint32_t> bfpq_insts(int64_t M, int64_t K, int64_t N, int mode, int fmt) {
    if (!bfpq_insts_fits(M, K, N, fmt) || mode < 0 || mode > 2) return {};
    const int64_t n_obj = K / (A_TK * per_object(fmt));  // weight objects per 64 rows
    stream s;
    for (int col = 0; col < COLS; col++) {
        s.write32(tile_addr(col, 2, RTP_DQ_OBJ), (uint32_t) (n_obj - 1));
        s.write32(tile_addr(col, 2, RTP_DQ_FMT), (uint32_t) fmt);
        s.write32(tile_addr(col, 2, RTP_DQ_PER), (uint32_t) per_object(fmt));
        for (int row = 3; row < 3 + ROWS; row++) {
            s.write32(tile_addr(col, row, RTP_NMID), (uint32_t) (K / A_TK - 1));
            s.write32(tile_addr(col, row, RTP_MODE), (uint32_t) mode);
        }
    }

    const int64_t row_blocks = M / (TM * ROWS);             // blocks of 384 rows
    const int64_t per_col = N / TN / COLS;                  // n-wide output tiles per column
    const int64_t c_per_rb = per_col * TM * ROWS * TN;      // floats one column drains per row block
    const int64_t a_bytes = TM * (K / 8) * BLOCK_BYTES;     // one 128-row block of A
    const int64_t q_bytes = per_col * n_obj * (int64_t) QTILE_Q4K_BYTES;  // one column's weights
    const int64_t tb_max = TM * ROWS * N > (1 << 20) ? 2 : 4;

    for (int64_t tb = 0; tb < (row_blocks + tb_max - 1) / tb_max; tb++) {
        for (int pp = 0; pp < 2; pp++) {
            const int64_t row_base = tb * tb_max + pp * tb_max / 2;
            if (row_base >= row_blocks) break;
            const int64_t rows = std::min(tb_max / 2, row_blocks - row_base);
            for (int col = 0; col < COLS; col++) {
                s.task(C_SHIM[col], ARG_C, (uint32_t) ((col * (M * N / COLS) + row_base * c_per_rb) * 4),
                       (uint32_t) (rows * c_per_rb), S2MM0_QUEUE, 1, true);
                for (int64_t r = 0; r < rows; r++) {
                    if (col < ROWS) {
                        const int64_t blk = (row_base + r) * ROWS + col;
                        s.task(A_SHIM[col][0], ARG_A, (uint32_t) (blk * a_bytes), (uint32_t) (a_bytes / 4),
                               A_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, (uint32_t) per_col, false);
                    }
                    s.task(Q_SHIM[col][0], ARG_B, (uint32_t) (col * q_bytes), (uint32_t) (q_bytes / 4),
                           Q_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false);
                }
            }
            if (tb > 0 || pp > 0) s.finish();
        }
    }
    s.finish();
    return finish_stream(s);
}
