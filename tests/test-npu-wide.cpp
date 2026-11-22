// Hardware checks of the BFP16 kernel's shapes and output modes that the
// backend's own tests don't reach at their sizes.
//
// The 34,816-column fused Qwen3.8-27B gate/up shape: the operands are ones,
// so every GEMM output is exactly K and every fused SwiGLU output is exactly
// K*K after BF16 rounding.
//
// A feed-forward chain: output mode 3 writes SiLU(gate) * up as the bfp16
// activations of the multiply that takes it. It must be, byte for byte, what
// the host packs from mode 2's output, and the two multiplies run back to
// back as one submission (the second reading the first's output on the NPU)
// must give what the second gives on the host-packed activations.
#include "bfp16_insts.h"
#include "bfp16_pack.h"
#include "npu_bfp16.h"
#include "xrt_load.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

const bfp16_tiling & tile = BFP16_TILE;

int add_shape(npu_bfp16 & npu, int64_t M, int64_t K, int64_t N, int mode, std::string & err) {
    const auto insts = bfp16_insts(M, K, N, mode);
    if (insts.empty()) { err = "shape refused"; return -1; }
    return npu.add_shape(insts, M, K, N, err, 0, 0, bfp16_c_bytes(M, N, tile, mode));
}

// 1 after a run that never answered: its buffers may still be in use
[[noreturn]] void give_up(const npu_bfp16 & npu, const char * what, const std::string & err) {
    fprintf(stderr, "%s: %s\n", what, err.c_str());
    if (npu.timed_out) std::_Exit(3);
    exit(3);
}

bool wide(npu_bfp16 & npu) {
    constexpr int64_t M = 512, K = 128, N = 34816;
    std::string err;
    std::vector<float> a((size_t) M * K, 1.0f), w((size_t) N * K, 1.0f);
    std::vector<uint8_t> pa, pw;
    bfp16_pack_a(a.data(), M, K, tile, pa, 8);
    bfp16_pack_b(w.data(), N, K, tile, pw, 8);
    const int wi = npu.add_weights(pw, err);
    if (wi < 0) give_up(npu, "weights", err);
    std::vector<float> c((size_t) M * N), row(N);

    for (int mode : { 0, 2 }) {
        const int shape = add_shape(npu, M, K, N, mode, err);
        if (shape < 0 || !npu.set_a(shape, pa, err)) give_up(npu, "setup", err);
        double ms = 0;
        if (!npu.run(shape, wi, c.data(), err, &ms)) give_up(npu, "wide shape", err);
        const int64_t width = mode == 2 ? N / 2 : N;
        for (int64_t r : { int64_t(0), M / 2, M - 1 }) {
            bfp16_tile_row(c.data(), N, tile, mode, r, row.data());
            for (int64_t j = 0; j < width; j++) {
                const float expected = mode == 2 ? float(K * K) : float(K);
                if (row[j] != expected) {
                    printf("FAIL mode %d row %lld column %lld: %g, expected %g\n", mode, (long long) r, (long long) j, row[j], expected);
                    return false;
                }
            }
        }
        printf("PASS mode %d, %lld x %lld x %lld, %.1f ms\n", mode, (long long) M, (long long) K, (long long) N, ms);
    }
    return true;
}

bool chain(npu_bfp16 & npu) {
    // two passes of rows, several rounds of output tiles, gate values of a few units on both sides of zero
    constexpr int64_t M = 1024, K = 512, F = 1024, N = 512;
    std::string err;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> a((size_t) M * K), g((size_t) F * K), u((size_t) F * K), gu((size_t) 2 * F * K), d((size_t) N * F);
    for (float & v : a) v = nd(rng);
    for (float & v : g) v = nd(rng) * 2 / std::sqrt((float) K);
    for (float & v : u) v = nd(rng) * 2 / std::sqrt((float) K);
    for (float & v : d) v = nd(rng) / std::sqrt((float) F);
    bfp16_interleave_gate_up(g.data(), u.data(), F, K, gu.data());
    std::vector<uint8_t> pa, pgu, pd;
    bfp16_pack_a(a.data(), M, K, tile, pa, 8);
    bfp16_pack_b(gu.data(), 2 * F, K, tile, pgu, 8);
    bfp16_pack_b(d.data(), N, F, tile, pd, 8);
    const int wgu = npu.add_weights(pgu, err), wd = wgu < 0 ? -1 : npu.add_weights(pd, err);
    if (wd < 0) give_up(npu, "weights", err);
    const int s2 = add_shape(npu, M, K, 2 * F, 2, err), s3 = s2 < 0 ? -1 : add_shape(npu, M, K, 2 * F, 3, err);
    const int sd = s3 < 0 ? -1 : add_shape(npu, M, F, N, 1, err);
    if (sd < 0) give_up(npu, "setup", err);

    // mode 2, and the host's packing of it
    std::vector<uint8_t> c2(bfp16_c_bytes(M, 2 * F, tile, 2));
    if (!npu.set_a(s2, pa, err) || !npu.run(s2, wgu, (float *) c2.data(), err)) give_up(npu, "gate/up, mode 2", err);
    std::vector<float> x((size_t) M * F);
    double sum = 0;
    for (int64_t r = 0; r < M; r++) bfp16_tile_row(c2.data(), 2 * F, tile, 2, r, x.data() + r * F);
    for (float v : x) sum += std::fabs(v);
    std::vector<uint8_t> want;
    bfp16_pack_a(x.data(), M, F, tile, want, 8);
    bool ok = std::isfinite(sum) && sum / (double) x.size() > 0.1;
    printf("%s mode 2: SiLU(gate) * up finite, mean magnitude %.3f\n", ok ? "PASS" : "FAIL", sum / (double) x.size());

    // mode 3 against it
    std::vector<uint8_t> c3(bfp16_c_bytes(M, 2 * F, tile, 3));
    if (c3.size() != want.size()) { printf("FAIL mode 3's output is %zu bytes, the next A %zu\n", c3.size(), want.size()); return false; }
    if (!npu.set_a(s3, pa, err) || !npu.run(s3, wgu, (float *) c3.data(), err)) give_up(npu, "gate/up, mode 3", err);
    size_t differ = 0;
    for (size_t i = 0; i < c3.size(); i++) differ += c3[i] != want[i];
    printf("%s mode 3: %zu of %zu bytes differ from the host's packing of mode 2\n", differ ? "FAIL" : "PASS", differ, c3.size());
    ok = ok && !differ;

    // the down multiply on the host-packed activations, then the two as one submission
    const size_t y_bytes = bfp16_c_bytes(M, N, tile, 1);
    std::vector<uint8_t> y_host(y_bytes);
    if (!npu.set_a(sd, want, err) || !npu.run(sd, wd, (float *) y_host.data(), err)) give_up(npu, "down", err);
    npu_bfp16::batch b(npu);
    for (int round = 0; round < 2; round++) {
        if (!b.prepare({ s3, sd }, err, { wgu, wd }, { -1, 0 })) give_up(npu, "chain", err);
        memcpy(b.a(0), pa.data(), pa.size());
        double ms = 0;
        if (!b.run(err, &ms)) give_up(npu, "chain", err);
        const bool same = memcmp(b.c(1), y_host.data(), y_bytes) == 0;
        printf("%s chain, round %d: gate/up and down as one submission give down's result on the host-packed activations (%.1f ms)\n",
               same ? "PASS" : "FAIL", round, ms);
        ok = ok && same;
    }
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: test-npu-wide <xclbin>\n"); return 2; }
    const std::string why = xrt_load();
    if (!why.empty()) { fprintf(stderr, "XRT: %s\n", why.c_str()); return 2; }

    std::string err;
    npu_bfp16 npu;
    if (!npu.open(argv[1], err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 2; }
    npu.wait_ms = 30000;

    bool ok = wide(npu);
    ok = chain(npu) && ok;
    printf("%s\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
