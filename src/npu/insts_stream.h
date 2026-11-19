// The pieces of an NPU instruction stream (insts.bin) the host-made GEMM
// streams share (bfp16_insts.cpp, bfpq_insts.cpp): register writes, linear
// shim DMA transfers between a kernel argument and a channel, and the waits
// on drains. Word for word what the IRON toolchain emits for them.
//
// Traces: XDNA-INSTS-GEN
#pragma once

#include <cstdint>
#include <initializer_list>
#include <set>
#include <utility>
#include <vector>

namespace npu_insts {

constexpr int SHIM_COLS = 8;

// Kernel arguments: A, the weights (B), C
constexpr uint32_t ARG_A = 0, ARG_B = 1, ARG_C = 2;

// Shim tile registers
constexpr uint32_t BD_BASE = 0x1d000, BD_STRIDE = 0x20;       // buffer descriptors
constexpr uint32_t S2MM0_CTRL = 0x1d200, S2MM0_QUEUE = 0x1d204;
constexpr uint32_t MM2S0_QUEUE = 0x1d214, MM2S1_QUEUE = 0x1d21c;
// Multiplying core tile: this dispatch's K/k - 1 and output mode (the
// designs' rtp buffer)
constexpr uint32_t RTP_NMID = 0xfc00, RTP_MODE = 0xfc04;

inline uint32_t tile_addr(int col, int row, uint32_t off) { return ((uint32_t) col << 25) | ((uint32_t) row << 20) | off; }

struct stream {
    std::vector<uint32_t> w;
    uint32_t ops = 0;
    std::set<int> used[SHIM_COLS];           // buffer descriptor IDs in use, per shim column
    std::vector<std::pair<int, int>> group;  // IDs taken since the last wait
    std::vector<int> drains;                 // shim columns of the drains since the last wait

    void put(std::initializer_list<uint32_t> words) { w.insert(w.end(), words); ops++; }
    void write32(uint32_t addr, uint32_t v) { put({ 0, 0, addr, 0, v, 24 }); }

    // One transfer between a kernel argument and a shim DMA channel: a linear
    // buffer descriptor (every transfer this design makes is contiguous), the
    // patch that points it at the argument, and the push onto the queue.
    // A drain also has the channel issue a completion token.
    void task(int col, uint32_t arg, uint32_t off, uint32_t words, uint32_t queue, uint32_t repeat, bool drain) {
        int bd = 0;
        while (used[col].count(bd)) bd++;  // the lowest free ID
        used[col].insert(bd);
        group.push_back({ col, bd });
        const uint32_t a = tile_addr(col, 0, BD_BASE + BD_STRIDE * (uint32_t) bd);
        put({ 1, 0, a, 48, words, off, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000 });
        put({ 0x81, 48, 0, 0, 0, 0, a + 4, 0, arg, 0, off, 0 });
        if (drain) {
            put({ 3, 0, tile_addr(col, 0, S2MM0_CTRL), 0, 0xf00, 0x1f00, 28 });
            drains.push_back(col);
        }
        write32(tile_addr(col, 0, queue), (drain ? 0x80000000u : 0u) | ((repeat - 1) << 16) | (uint32_t) bd);
    }

    // Waits for the group's drains, then frees its descriptor IDs.
    void finish() {
        for (int col : drains) put({ 0x80, 16, (uint32_t) col << 16, 0x00010100 });
        for (auto [col, bd] : group) used[col].erase(bd);
        group.clear();
        drains.clear();
    }
};


// The stream's header and words.
inline std::vector<uint32_t> finish_stream(const stream & s) {
    std::vector<uint32_t> out = { 0x06040100, 0x108, s.ops, (uint32_t) ((s.w.size() + 4) * 4) };
    out.insert(out.end(), s.w.begin(), s.w.end());
    return out;
}

} // namespace npu_insts
