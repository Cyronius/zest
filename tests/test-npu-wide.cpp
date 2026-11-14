// Hardware check for the 34,816-column fused Qwen3.8-27B gate/up shape.
// The operands are ones, so every GEMM output is exactly K and every fused
// SwiGLU output is exactly K*K after BF16 rounding.
#include "bfp16_insts.h"
#include "bfp16_pack.h"
#include "npu_bfp16.h"
#include "xrt_load.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    constexpr int64_t M = 512, K = 128, N = 34816;
    const bfp16_tiling tile;
    if (argc != 2) { fprintf(stderr, "usage: test-npu-wide <xclbin>\n"); return 2; }
    const std::string why = xrt_load();
    if (!why.empty()) { fprintf(stderr, "XRT: %s\n", why.c_str()); return 2; }

    std::string err;
    npu_bfp16 npu;
    if (!npu.open(argv[1], err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 2; }
    npu.wait_ms = 30000;

    std::vector<float> a((size_t) M * K, 1.0f), w((size_t) N * K, 1.0f);
    std::vector<uint8_t> pa, pw;
    bfp16_pack_a(a.data(), M, K, tile, pa, 8);
    bfp16_pack_b(w.data(), N, K, tile, pw, 8);
    const int wi = npu.add_weights(pw, err);
    if (wi < 0) { fprintf(stderr, "weights: %s\n", err.c_str()); return 2; }
    std::vector<float> c((size_t) M * N), row(N);

    for (int mode : {0, 2}) {
        const auto insts = bfp16_insts(M, K, N, mode);
        if (insts.empty()) { fprintf(stderr, "wide shape refused\n"); return 2; }
        const int shape = npu.add_shape(insts, M, K, N, err);
        if (shape < 0 || !npu.set_a(shape, pa, err)) { fprintf(stderr, "setup: %s\n", err.c_str()); return 2; }
        double ms = 0;
        if (!npu.run(shape, wi, c.data(), err, &ms)) {
            fprintf(stderr, "mode %d: %s\n", mode, err.c_str());
            if (npu.timed_out) std::_Exit(3);  // a timed-out run may still use its buffers
            return 3;
        }
        const int64_t width = mode == 2 ? N / 2 : N;
        for (int64_t r : {int64_t(0), M / 2, M - 1}) {
            if (mode == 2) bfp16_c16_row(c.data(), M, N, tile, mode, r, row.data());
            else bfp16_c_row(c.data(), M, N, tile, r, width, row.data());
            for (int64_t j = 0; j < width; j++) {
                const float expected = mode == 2 ? float(K * K) : float(K);
                if (row[j] != expected) {
                    fprintf(stderr, "mode %d row %lld column %lld: %g, expected %g\n", mode,
                            (long long) r, (long long) j, row[j], expected);
                    return 1;
                }
            }
        }
        printf("PASS mode %d, %lld x %lld x %lld, %.1f ms\n", mode,
               (long long) M, (long long) K, (long long) N, ms);
    }

    return 0;
}
