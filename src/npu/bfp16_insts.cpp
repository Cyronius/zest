#include "bfp16_insts.h"
#include "bfp16_pack.h"
#include "insts_stream.h"

namespace {

constexpr int64_t TM = BFP16_TILE.m, TK = BFP16_TILE.k, TN = BFP16_TILE.n;
constexpr int ROWS = 4, COLS = 8;  // core rows (each takes a group of B) and columns (each a row block of A)
static_assert(BFP16_TILE.cols == ROWS && BFP16_TILE.rows == COLS, "bfp16_pack.h's tiling is this design's");

// Where the toolchain placed things, the same in every column and for every
// size (the design pins each column's fifos to its own memtile);
// kernels/bfp16_gemm/designs/check_placement.py checks every build against
// the design's copy of these.
//   shims: column c's A comes in on shim c's MM2S channel 0 and its C leaves
//          on that shim's S2MM channel 0; core row r's B comes in on shim
//          2r's MM2S channel 1
//   memtile c: A in on S2MM channel 0 (the fifo's descriptors from 2), A out
//          on MM2S channel 0 (from 0), A's locks 8 (free tiles, 2 to start)
//          and 9 (full ones); C out on MM2S channel 1, descriptors 26 to 29,
//          one per core row
constexpr uint32_t A_IN_CH = 0, A_IN_BD = 2, A_OUT_CH = 0, A_OUT_BD = 0, A_FREE_LOCK = 8, A_FULL_LOCK = 9;
constexpr uint32_t C_OUT_CH = 1, C_OUT_BD = 26;
// Resident A's own: descriptors and locks the fifos leave unused, and the
// memtile's memory after the C join's two 64 KB banks.
constexpr uint32_t RES_BASE = 131072, RES_BYTES = 393216, RES_FILL = 10, RES_PLAY = 11, RES_LAST = 12;
constexpr uint32_t RES_FREE_LOCK = 16, RES_FULL_LOCK = 17;

// Memtile registers. A buffer descriptor is eight words: the first its length
// in words (low 17 bits), the second its address in words (the tile's own
// memory starts at 0x20000), the last its locks (the tile's own are 64 up). A
// channel's control register resets it (bit 1); its start queue takes a
// descriptor and a repeat count. A lock's value is a register.
constexpr uint32_t MEM_BD = 0xa0000, MEM_BD_STRIDE = 0x20, MEM_BD_LENGTH = 0x1ffff;
constexpr uint32_t MEM_S2MM_CTRL = 0xa0600, MEM_S2MM_QUEUE = 0xa0604, MEM_MM2S_CTRL = 0xa0630, MEM_MM2S_QUEUE = 0xa0634;
constexpr uint32_t MEM_CH_STRIDE = 8, MEM_CH_RESET = 2;
constexpr uint32_t MEM_LOCK = 0xc0000, MEM_LOCK_STRIDE = 0x10, MEM_OWN_MEMORY = 0x20000, MEM_OWN_LOCKS = 64;

// A descriptor's last word: valid, releases `rel` by one when done, and
// first needs `acq` to be at least one (and takes one).
constexpr uint32_t bd_locks(uint32_t acq, uint32_t rel) {
    return 0x80000000u | (1u << 24) | ((MEM_OWN_LOCKS + rel) << 16) | (1u << 15) | (0x7fu << 8) | (MEM_OWN_LOCKS + acq);
}

} // namespace

using namespace npu_insts;

bool bfp16_insts_resident(int64_t M, int64_t K, int64_t N) {
    // The row block fits the memtile's free memory, two passes' fills and
    // plays fit its channels' queues, and there are output tiles enough to
    // gain by it: a memtile plays a block only once it has all of it, which
    // costs a narrow call more than its second read from DDR (the 27B's
    // 1,024-wide k projection ran 10% slower resident).
    return TM * (K / 8) * (int64_t) BFP16_BLOCK_BYTES <= RES_BYTES && M / (TM * COLS) <= 2 && N / (TN * ROWS) >= 3;
}

bool bfp16_insts_fits(int64_t M, int64_t K, int64_t N) {
    // 68 rounds of output tiles cover Qwen3.8-27B's 17,408-wide gate and up
    // weights in one fused 34,816-wide kernel invocation.
    return M > 0 && M % (TM * COLS) == 0 && N > 0 && N % (TN * ROWS) == 0 && N / (TN * ROWS) <= 68 && K % TK == 0 &&
           K >= 2 * TK;
}

std::vector<uint32_t> bfp16_insts(int64_t M, int64_t K, int64_t N, int mode) {
    if (!bfp16_insts_fits(M, K, N) || mode < 0 || mode > 3) return {};
    stream s;
    for (int col = 0; col < COLS; col++)
        for (int row = 2; row < 2 + ROWS; row++) {
            s.write32(tile_addr(col, row, RTP_NMID), (uint32_t) (K / TK - 1));
            s.write32(tile_addr(col, row, RTP_MODE), (uint32_t) mode);
        }

    const int64_t passes = M / (TM * COLS);  // 512 rows each
    const int64_t per = N / (TN * ROWS);     // output tiles per core
    const uint32_t cw = (uint32_t) bfp16_tile_words(BFP16_TILE, mode);
    // blocks of A one column reads per pass and of B one core row reads;
    // words of C one column writes per pass
    const int64_t a_run = TM * (K / 8), b_run = (N / ROWS) * (K / 8), c_run = (N / TN) * cw;
    const bool keep_a = bfp16_insts_resident(M, K, N);
    const auto reset = [&](int mt, uint32_t ctrl, uint32_t ch) {
        s.maskwrite32(tile_addr(mt, 1, ctrl + MEM_CH_STRIDE * ch), MEM_CH_RESET, MEM_CH_RESET);
        s.maskwrite32(tile_addr(mt, 1, ctrl + MEM_CH_STRIDE * ch), 0, MEM_CH_RESET);
    };
    const auto set_lock = [&](int mt, uint32_t lock, uint32_t v) { s.write32(tile_addr(mt, 1, MEM_LOCK + MEM_LOCK_STRIDE * lock), v); };
    for (int mt = 0; mt < COLS; mt++) {
        // The memtile passes on only the words of each tile this mode makes.
        // A waiting channel has already loaded its next descriptor (with the
        // last call's length), so it is reset and started again at its first
        // descriptor, where it stands between calls.
        reset(mt, MEM_MM2S_CTRL, C_OUT_CH);
        for (uint32_t i = 0; i < ROWS; i++) s.maskwrite32(tile_addr(mt, 1, MEM_BD + MEM_BD_STRIDE * (C_OUT_BD + i)), cw, MEM_BD_LENGTH);
        s.write32(tile_addr(mt, 1, MEM_MM2S_QUEUE + MEM_CH_STRIDE * C_OUT_CH), C_OUT_BD);
        // A's two channels, whatever the call before left them doing
        reset(mt, MEM_S2MM_CTRL, A_IN_CH);
        reset(mt, MEM_MM2S_CTRL, A_OUT_CH);
        if (!keep_a) {
            // relayed tile by tile: the fifo's own descriptors, going round,
            // and its locks as they start
            set_lock(mt, A_FREE_LOCK, 2);
            set_lock(mt, A_FULL_LOCK, 0);
            s.write32(tile_addr(mt, 1, MEM_S2MM_QUEUE + MEM_CH_STRIDE * A_IN_CH), A_IN_BD);
            s.write32(tile_addr(mt, 1, MEM_MM2S_QUEUE + MEM_CH_STRIDE * A_OUT_CH), A_OUT_BD);
            continue;
        }
        // Resident: three descriptors of our own over the whole row block.
        // Fill waits for the buffer to be free and marks it full; play needs
        // it full and leaves it so; a pass's last play frees it for the next
        // pass's fill.
        set_lock(mt, RES_FREE_LOCK, 1);
        set_lock(mt, RES_FULL_LOCK, 0);
        const uint32_t bds[3][3] = { { RES_FILL, RES_FREE_LOCK, RES_FULL_LOCK },
                                     { RES_PLAY, RES_FULL_LOCK, RES_FULL_LOCK },
                                     { RES_LAST, RES_FULL_LOCK, RES_FREE_LOCK } };
        for (const auto & bd : bds) {
            const uint32_t w[8] = { (uint32_t) (a_run * BFP16_BLOCK_BYTES / 4), MEM_OWN_MEMORY + RES_BASE / 4, 0, 0, 0, 0, 0,
                                    bd_locks(bd[1], bd[2]) };
            s.write_bd(tile_addr(mt, 1, MEM_BD + MEM_BD_STRIDE * bd[0]), w);
        }
        for (int64_t p = 0; p < passes; p++) {
            s.write32(tile_addr(mt, 1, MEM_S2MM_QUEUE + MEM_CH_STRIDE * A_IN_CH), RES_FILL);
            if (per > 1) s.write32(tile_addr(mt, 1, MEM_MM2S_QUEUE + MEM_CH_STRIDE * A_OUT_CH), (uint32_t) ((per - 2) << 16) | RES_PLAY);
            s.write32(tile_addr(mt, 1, MEM_MM2S_QUEUE + MEM_CH_STRIDE * A_OUT_CH), RES_LAST);
        }
    }

    for (int64_t p = 0; p < passes; p++) {
        for (int col = 0; col < COLS; col++) {
            const int64_t rb = p * COLS + col;  // the column's row block
            s.task(col, ARG_C, (uint32_t) (rb * c_run * 4), (uint32_t) c_run, S2MM0_QUEUE, 1, true);
            // A: the row block, once for a resident call, else once per output tile
            s.task(col, ARG_A, (uint32_t) (rb * a_run * BFP16_BLOCK_BYTES), (uint32_t) (a_run * BFP16_BLOCK_BYTES / 4), MM2S0_QUEUE,
                   (uint32_t) (keep_a ? 1 : per), false);
        }
        for (int row = 0; row < ROWS; row++)
            s.task(2 * row, ARG_B, (uint32_t) (row * b_run * BFP16_BLOCK_BYTES), (uint32_t) (b_run * BFP16_BLOCK_BYTES / 4),
                   MM2S1_QUEUE, 1, false);
        // two passes' transfers queue on a channel before the wait
        if (p % 2 == 1 || p == passes - 1) s.finish();
    }
    return finish_stream(s);
}
