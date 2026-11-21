// The second half of a transformer block as llama.cpp hands it to the
// backend, on the NPU: attention's output multiply and its residual add, a
// norm, the gate and up multiplies, their SwiGLU, the down multiply and the
// residual add, the weights in a weight buffer. The backend fuses gate and up
// and chains the down multiply behind them (one NPU submission, the SwiGLU
// never coming to the host); with GGML_XDNA_CHAIN=0 a host pass sits between.
// As shipped the GPU runs half the rows at the same time (GGML_XDNA_SPLIT, 0
// for none; those tests also look for the trace line that says the piece was
// shared).
// Whichever way, the result has to stay close to the CPU backend's on the
// same bytes, and the NPU must not have failed on the way (when it does the
// GPU finishes the work, with the right answer: GGML_XDNA_FAIL_AFTER tests
// that).
//
// Prints a checksum of the result, so runs that should give the same bits can
// be compared.
//
// Traces: XDNA-FFN-CHAIN, XDNA-BLOCK-AGREES

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool g_npu_failed = false;
static void on_log(ggml_log_level, const char * text, void *) {
    if (strstr(text, "the NPU failed")) g_npu_failed = true;
    fputs(text, stderr);
}

constexpr int64_t E = 2048, F = 4096, T = 2048;

struct block {
    ggml_context * wctx = nullptr, * ctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, buf = nullptr;
    ggml_tensor * x = nullptr, * scale = nullptr, * wo = nullptr, * wg = nullptr, * wu = nullptr, * wd = nullptr, * out = nullptr;
    ggml_cgraph * gf = nullptr;
    ~block() {
        if (buf) ggml_backend_buffer_free(buf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ctx) ggml_free(ctx);
        if (wctx) ggml_free(wctx);
    }
};

// the block on `backend`, its weights and input set from the given bytes
static bool build(block & b, ggml_backend_t backend, const std::vector<uint8_t> & o, const std::vector<uint8_t> & g,
                  const std::vector<uint8_t> & u, const std::vector<uint8_t> & d, const std::vector<float> & scale,
                  const std::vector<float> & x) {
    ggml_init_params wp = { ggml_tensor_overhead() * 8, NULL, true };
    b.wctx = ggml_init(wp);
    b.wo = ggml_new_tensor_2d(b.wctx, GGML_TYPE_Q4_0, E, E);
    ggml_set_name(b.wo, "blk.0.attn_output.weight");
    b.wg = ggml_new_tensor_2d(b.wctx, GGML_TYPE_Q4_0, E, F);
    b.wu = ggml_new_tensor_2d(b.wctx, GGML_TYPE_Q4_0, E, F);
    b.wd = ggml_new_tensor_2d(b.wctx, GGML_TYPE_Q4_0, F, E);
    b.scale = ggml_new_tensor_1d(b.wctx, GGML_TYPE_F32, E);
    ggml_set_name(b.wg, "blk.0.ffn_gate.weight");
    ggml_set_name(b.wu, "blk.0.ffn_up.weight");
    ggml_set_name(b.wd, "blk.0.ffn_down.weight");
    ggml_set_name(b.scale, "blk.0.ffn_norm.weight");
    b.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(b.wctx, ggml_backend_get_default_buffer_type(backend));
    if (!b.wbuf) return false;
    ggml_backend_buffer_set_usage(b.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(b.wo, o.data(), 0, o.size());
    ggml_backend_tensor_set(b.wg, g.data(), 0, g.size());
    ggml_backend_tensor_set(b.wu, u.data(), 0, u.size());
    ggml_backend_tensor_set(b.wd, d.data(), 0, d.size());
    ggml_backend_tensor_set(b.scale, scale.data(), 0, scale.size() * sizeof(float));

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
    b.ctx = ggml_init(ip);
    // the residual stream as it reaches the block: not a graph input (the backend leaves an add onto one of those,
    // which may sit in host memory, to the GPU)
    b.x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, E, T);
    // the backend takes a small op only behind a multiply it takes, so the block starts with one
    ggml_tensor * r = ggml_add(b.ctx, ggml_mul_mat(b.ctx, b.wo, b.x), b.x);
    ggml_tensor * h = ggml_mul(b.ctx, ggml_rms_norm(b.ctx, r, 1e-6f), b.scale);
    ggml_tensor * act = ggml_swiglu_split(b.ctx, ggml_mul_mat(b.ctx, b.wg, h), ggml_mul_mat(b.ctx, b.wu, h));
    // (in place, as a graph may have it: the result then lives in the multiply's memory)
    b.out = ggml_add_inplace(b.ctx, ggml_mul_mat(b.ctx, b.wd, act), r);
    ggml_set_output(b.out);
    b.gf = ggml_new_graph(b.ctx);
    ggml_build_forward_expand(b.gf, b.out);
    // Views that follow the block's last op, as llama.cpp's scheduler leaves them in a piece: the result turned for
    // an op outside it, and a reshape of something with no token rows at all (another layer's state). They compute
    // nothing, and must not keep the GPU from taking its share of the rows.
    ggml_build_forward_expand(b.gf, ggml_transpose(b.ctx, b.out));
    ggml_build_forward_expand(b.gf, ggml_reshape_2d(b.ctx, ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 24), 6, 4));
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (!b.buf) return false;
    ggml_backend_tensor_set(b.x, x.data(), 0, x.size() * sizeof(float));
    return true;
}

int main() {
    ggml_log_set(on_log, NULL);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("XDNA0");
    if (dev == NULL) {
        fprintf(stderr, "XDNA0 not registered - is GGML_BACKEND_PATH set?\n");
        return 2;
    }
    ggml_backend_t xdna = ggml_backend_dev_init(dev, NULL);
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), NULL);
    if (!xdna || !cpu) {
        fprintf(stderr, "cannot start the backends\n");
        return 2;
    }

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto weight = [&](int64_t rows, int64_t k, float size) {
        std::vector<float> w((size_t) (rows * k));
        for (float & v : w) v = nd(rng) * size;
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_0, k) * (size_t) rows);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), q.data(), 0, rows, k, NULL);
        return q;
    };
    // gate values of a few units on both sides of zero: across SiLU's curve
    const std::vector<uint8_t> g = weight(F, E, 2.0f / std::sqrt((float) E)), u = weight(F, E, 1.0f / std::sqrt((float) E));
    const std::vector<uint8_t> d = weight(E, F, 1.0f / std::sqrt((float) F)), o = weight(E, E, 1.0f / std::sqrt((float) E));
    std::vector<float> scale((size_t) E), x((size_t) (E * T));
    for (float & v : scale) v = 1.0f + 0.25f * nd(rng);
    for (float & v : x) v = nd(rng) * 1.5f;

    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        failures += !ok;
    };

    block on_npu, on_cpu;
    if (!build(on_npu, xdna, o, g, u, d, scale, x) || !build(on_cpu, cpu, o, g, u, d, scale, x)) {
        fprintf(stderr, "cannot allocate the block\n");
        return 2;
    }
    // as llama.cpp's scheduler would: every op offered first (that is also how the backend learns who reads what)
    // (in host-only mode it takes only the multiplies; handed the whole block it still runs it)
    const char * host_only = getenv("GGML_XDNA_HOST_ONLY");
    bool all = true;
    for (int i = 0; i < ggml_graph_n_nodes(on_npu.gf); i++) {
        ggml_tensor * n = ggml_graph_node(on_npu.gf, i);
        if (n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_RESHAPE) continue;  // a view goes where its source went
        if (ggml_backend_supports_op(xdna, n) || (host_only && *host_only && n->op != GGML_OP_MUL_MAT)) continue;
        printf("not taken: %s\n", ggml_op_name(n->op));
        all = false;
    }
    check(all, "the backend takes every op of the block");

    const bool ran = ggml_backend_graph_compute(xdna, on_npu.gf) == GGML_STATUS_SUCCESS &&
                     ggml_backend_graph_compute(cpu, on_cpu.gf) == GGML_STATUS_SUCCESS;
    check(ran, "both backends compute it");
    if (ran) {
        std::vector<float> got((size_t) (E * T)), want(got.size());
        ggml_backend_tensor_get(on_npu.out, got.data(), 0, got.size() * sizeof(float));
        ggml_backend_tensor_get(on_cpu.out, want.data(), 0, want.size() * sizeof(float));
        // against the CPU, on what the block adds to its input
        double se = 0, ss = 0;
        uint64_t sum = 1469598103934665603ull;
        for (size_t i = 0; i < got.size(); i++) {
            const double a = (double) got[i] - x[i], b = (double) want[i] - x[i];
            se += (a - b) * (a - b);
            ss += b * b;
            uint32_t bits;
            memcpy(&bits, &got[i], 4);
            sum = (sum ^ bits) * 1099511628211ull;
        }
        const double nrmse = std::sqrt(se / std::max(ss, 1e-300));
        printf("block against the CPU's: nrmse %.2e, checksum %016llx\n", nrmse, (unsigned long long) sum);
        // three multiplies on 8-bit blocks, two of them in a row: about 2e-2
        check(nrmse < 3e-2 && std::isfinite(nrmse), "the block's result is close to the CPU's");
    }
    // (GGML_XDNA_FAIL_AFTER makes it fail, to test that the GPU then finishes the block)
    const char * fail = getenv("GGML_XDNA_FAIL_AFTER");
    const bool injected = fail && *fail;
    check(g_npu_failed == injected, injected ? "the NPU was made to fail, and the backend said so" : "the NPU did not fail");

    ggml_backend_free(xdna);
    ggml_backend_free(cpu);
    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
