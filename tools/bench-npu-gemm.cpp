// Times the NPU's bfp16 GEMM at Qwen3.8-27B's projection shapes, and sets
// each time against the math it does and the DRAM bytes it moves.
//
// The kernel (kernels/bfp16_gemm/designs/whole_array_bfp_colwise.py) streams
// A once per n * cols output columns, or just once when the call keeps it
// resident in the memtiles (bfp16_insts_resident), and B once per m * rows
// rows, each 9 bytes per 8 values; C is written once: 4 bytes a value (mode
// 0), 2 (mode 1), 2 per SiLU(gate) * up value (mode 2), or that value as 9
// bytes per 8 (mode 3, the next multiply's activations). So a GEMM moves
//   A: M*K*9/8 * (1 or N/(n*cols))   B: N*K*9/8 * M/(m*rows)   C: by its mode
// and its time is bounded by the larger of that over the NPU's bandwidth and
// its math over the array's rate.
//
// usage: bench-npu-gemm <xclbin> [M=1024] [repetitions=10]
#include "bfp16_insts.h"
#include "bfp16_pack.h"
#include "npu_bfp16.h"
#include "xrt_load.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench-npu-gemm <xclbin> [M=1024] [repetitions=10]\n"); return 2; }
    const int64_t M = argc > 2 ? atoll(argv[2]) : 1024;
    const int reps = argc > 3 ? atoi(argv[3]) : 10;
    const std::string why = xrt_load();
    if (!why.empty()) { fprintf(stderr, "XRT: %s\n", why.c_str()); return 2; }

    std::string err;
    npu_bfp16 npu;
    if (!npu.open(argv[1], err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 2; }
    npu.wait_ms = 30000;
    const bfp16_tiling & tile = BFP16_TILE;

    struct shape { const char * name; int64_t K, N; int mode; };
    // Qwen3.8-27B: hidden 5120, FFN 17408, GDN qkv 10240 and z 6144, FA q (+ gate) 12288, k / v 1024
    const shape shapes[] = {
        { "gdn qkv", 5120, 10240, 1 },
        { "gdn z", 5120, 6144, 1 },
        { "fa q+gate", 5120, 12288, 1 },
        { "fa k", 5120, 1024, 1 },
        { "o_proj (gdn)", 6144, 5120, 1 },
        { "gate_up fused", 5120, 34816, 2 },
        { "gate_up chain", 5120, 34816, 3 },
        { "down", 17408, 5120, 1 },
    };

    printf("M = %lld, median of %d\n", (long long) M, reps);
    printf("%-14s %6s %6s %4s %8s %8s %8s %8s %8s\n", "projection", "K", "N", "mode", "ms", "TFLOPS", "MB", "GB/s", "B share");
    // (a resident call's A share of the bytes is small: the memtiles replay it)
    double total_ms = 0, total_flop = 0, total_mb = 0;
    for (const shape & s : shapes) {
        std::vector<float> a((size_t) M * s.K, 1.0f / 64), w((size_t) s.N * s.K, 1.0f / 64);
        std::vector<uint8_t> pa, pw;
        bfp16_pack_a(a.data(), M, s.K, tile, pa, 8);
        bfp16_pack_b(w.data(), s.N, s.K, tile, pw, 8);
        const int wi = npu.add_weights(pw, err);
        const auto insts = bfp16_insts(M, s.K, s.N, s.mode);
        if (wi < 0 || insts.empty()) { fprintf(stderr, "%s: %s\n", s.name, insts.empty() ? "shape refused" : err.c_str()); return 2; }
        const size_t c_bytes = bfp16_c_bytes(M, s.N, tile, s.mode);
        const int sh = npu.add_shape(insts, M, s.K, s.N, err, 0, 0, c_bytes);
        if (sh < 0 || !npu.set_a(sh, pa, err)) { fprintf(stderr, "%s: %s\n", s.name, err.c_str()); return 2; }
        std::vector<float> c((size_t) M * s.N);
        std::vector<double> t;
        for (int i = 0; i < reps + 1; i++) {
            double ms = 0;
            if (!npu.run(sh, wi, c.data(), err, &ms)) {
                fprintf(stderr, "%s: %s\n", s.name, err.c_str());
                if (npu.timed_out) std::_Exit(3);
                return 3;
            }
            if (i) t.push_back(ms);  // the first run warms up
        }
        std::sort(t.begin(), t.end());
        const double ms = t[t.size() / 2];
        const double flop = 2.0 * M * s.K * s.N;
        const double a_mb = M * s.K * 9.0 / 8 * (bfp16_insts_resident(M, s.K, s.N) ? 1 : s.N / (tile.n * tile.cols)) / 1e6;
        const double b_mb = s.N * s.K * 9.0 / 8 * ((double) M / (tile.m * tile.rows)) / 1e6;
        const double c_mb = c_bytes / 1e6;
        const double mb = a_mb + b_mb + c_mb;
        printf("%-14s %6lld %6lld %4d %8.2f %8.2f %8.0f %8.1f %7.0f%%\n", s.name, (long long) s.K, (long long) s.N, s.mode,
               ms, flop / ms / 1e9, mb, mb / ms, 100 * b_mb / mb);
        total_ms += ms;
        total_flop += flop;
        total_mb += mb;
    }
    printf("%-14s %6s %6s %4s %8.2f %8.2f %8.0f %8.1f\n", "all", "", "", "", total_ms, total_flop / total_ms / 1e9, total_mb,
           total_mb / total_ms);
    return 0;
}
