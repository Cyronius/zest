// The NPU side of the backend: each claimed weight's private BFP16 copy, and
// the matmul itself, on the bfp16 kernel (npu/npu_bfp16.h, kernels/bfp16_gemm).
//
// The kernel is one core program (an xclbin of whole_array_bfp_colwise) that
// serves every size; each size's instruction stream is made here
// (npu/bfp16_insts.h).
// ZEST_KERNELS names the xclbin: the file itself, a directory holding
// final.xclbin, or a directory of per-size builds (any one's final.xclbin).
// A call covers 1024 prompt rows, or 512 for a short remainder.
//
// A fused gate/up can write SiLU(gate) * up as the activations of the
// multiply that takes it next (the kernel's output mode 3), and the two then
// run back to back on the NPU as one submission: a chain (begin).
//
// A weight's copy can instead live under the kernel on quantized weights
// (whole_array_bfpq, npu/bfpq_insts.h), at about half the bytes for Q4_K:
// Q4_K is re-laid for the NPU to unpack (npu/qtile_pack.h), other types are
// packed as BFP16 and copied through. Its calls run 384, 768 or 1,152 rows.
// It is its own xclbin and hardware context (bfpq_gemm.xclbin, or
// ZEST_QKERNELS), and switching contexts costs ~2.4 ms, so a model's
// weights all go under one kernel (ggml-zest.cpp, ZEST_WEIGHTS).
#pragma once

#include "ggml.h"
#include "npu_bfp16.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

class thread_pool;

// The kernel's xclbin: ZEST_KERNELS if set (the file, or a directory
// holding final.xclbin or per-size builds), else bfp16_gemm.xclbin next to
// this DLL. Empty when there's none.
const std::string & xdna_xclbin();
// Whether the NPU can run the backend: it opens, loads the xclbin, and gets a
// small test multiply right. Checked once, when llama.cpp registers the
// backend; `why` says what failed.
bool xdna_npu_usable(std::string & why);
// Whether the kernel takes a weight of K x N at both row counts.
bool xdna_npu_has_shape(int64_t K, int64_t N);
// The quantized-weight kernel's xclbin, as xdna_xclbin; empty when there's none.
const std::string & xdna_qxclbin();
// Whether a weight of this type and size can have a quantized-kernel copy,
// and its bytes (N as the kernel runs it).
bool xdna_npu_has_qshape(int type, int64_t K, int64_t N);
double xdna_npu_qcopy_bytes(int type, int64_t K, int64_t N);
// Bytes the NPU's buffers may lock in all (Linux's memlock limit, when XRT's buffers count against it); negative: no limit.
double xdna_npu_lock_limit();
// Kept back from that limit for the NPU's buffers other than weight copies.
constexpr double XDNA_LOCK_KEPT = 2.0 * 1024 * 1024 * 1024;

class xdna_npu {
public:
    xdna_npu();
    ~xdna_npu();
    bool init(int n_threads, std::string & err);

    // Identifies a weight across calls: where it lives and what it is.
    struct wkey {
        const void * buffer, * data;
        int type;
        int64_t K, N;
        bool operator<(const wkey & o) const;
    };
    // Whether the weight's BFP16 copy exists already.
    bool has_weight(const wkey & key) const { return weights_.count(key) != 0; }
    // Builds it from the weight's raw ggml bytes: each row is dequantized with
    // ggml and encoded directly into its mapped NPU buffer. It runs in the
    // 16-bit output mode where that's built (half the bytes to read back;
    // ZEST_OUT16=0 keeps float32). `quant`: under the quantized-weight
    // kernel instead (xdna_npu_has_qshape must hold).
    bool add_weight(const wkey & key, const void * raw, std::string & err, bool quant = false);

    // A gate and an up weight fused, for SiLU(gate) * up computed on the NPU
    // (the kernel's output mode 2): rows interleaved so each output tile pairs
    // a gate block with its up block. Its output has F = N values a row, the
    // SwiGLU result. Keyed by fused_key(gate). Needs the 2F-wide mode-2 builds.
    static wkey fused_key(const wkey & gate) { return { gate.buffer, gate.data, 1000 + gate.type, gate.K, 2 * gate.N }; }
    // The gate's key; up == gate is one weight holding the gate's rows and then the up's, so its first half.
    static wkey gate_key(const ggml_tensor * gate, const ggml_tensor * up) {
        return { gate->buffer, gate->data, (int) gate->type, gate->ne[0], gate == up ? gate->ne[1] / 2 : gate->ne[1] };
    }
    static bool has_fused_shape(int64_t K, int64_t F);
    bool add_fused(const wkey & gate, const void * gate_raw, const void * up_raw, std::string & err,
                   bool quant = false);
    // Whether a weight added above lives under the quantized-weight kernel.
    bool is_quant(const wkey & key) const { return weights_.at(key).q; }
    // Whether a fused gate/up's result can go straight into `next`'s multiply
    // on the NPU, as a chain: both under the BFP16 kernel, and next's K the
    // fused output's width as the kernel runs it.
    bool can_chain(const wkey & fused, const wkey & next) const;

    // Output values a row for a weight added above (N, or N/2 fused).
    int64_t out_width(const wkey & key) const { return weights_.at(key).N_out; }

    // y[T x N] = x[T x K] * W^T, float32 row-major, for a weight added above.
    bool mul_mat(const wkey & key, const float * x, int64_t T, float * y, std::string & err);

    // The same in steps, for block claiming. `stream` picks a set of buffers,
    // with one group of calls in flight per stream: one call per weight, all
    // sharing one input (q, k and v; gate and up). The steps let the host read
    // one group's outputs and write the next group's input in the same pass
    // over the rows:
    //   reserve     sizes the stream's buffers for every group it will run,
    //               before the first, so none moves while outputs are unread
    //   begin       readies the next group, for `rows` rows (at most 1,024)
    //   encode_row  writes input row r (K values); any thread, one per row
    //   submit      starts the group
    //   wait        waits for it; its outputs can then be read until the
    //               next submit
    //   decode_row  reads row r of output k (out_width values); any thread.
    //               False if a value is NaN or infinite: then the NPU's
    //               results can't be trusted (not_finite_error says so)
    // A chain is a group of two: a fused gate/up, and the weight whose
    // multiply takes its result (can_chain). It has one output, the second
    // weight's; the result between them stays on the NPU.
    struct group {
        std::vector<wkey> keys;
        bool chain = false;
    };
    bool reserve(int stream, const std::vector<group> & groups, std::string & err);
    bool begin(int stream, const group & g, int64_t rows, std::string & err);
    void encode_row(int stream, int64_t r, const float * row);
    void encode_tile(int stream, int64_t r0, int64_t count, const float * const * rows);
    bool submit(int stream, std::string & err);
    bool wait(int stream, std::string & err);
    bool decode_row(int stream, int k, int64_t r, float * row) const;
    static const char * not_finite_error();

    thread_pool & pool() { return *pool_; }

    // Time spent, since the last call: encoding activations, the NPU
    // (submit to done), reading its output, building weight copies.
    struct times {
        double encode_ms = 0, npu_ms = 0, decode_ms = 0, pack_ms = 0;
        double weight_shape_ms = 0, weight_alloc_ms = 0, weight_encode_ms = 0, weight_upload_ms = 0;
        int calls = 0;
    };
    times take_times();

private:
    struct weight {
        int handle = -1;
        // per call size: M = 512 and 1,024; quantized, M = 384, 768, 1,152
        int shapes[3] = { -1, -1, -1 };
        int64_t K = 0, N = 0, N_out = 0;       // N as the kernel runs it (padded); N_out real values a row
        int mode = 0;                          // output mode: 0 float32, 1 bf16, 2 SiLU(gate) * up in bf16
        bool q = false;                        // under the quantized-weight kernel
        // a fused weight under the BFP16 kernel: the same sizes in output
        // mode 3, for a chain
        int chain_shapes[2] = { -1, -1 };
    };
    // The rows a call on `w` runs for `rows` prompt rows, and its shape.
    static int64_t call_rows(const weight & w, int64_t rows);
    static int call_shape(const weight & w, int64_t rows);
    static const bfp16_tiling & tiling(const weight & w);
    // Bytes of a call's output: `rows` call rows of w in `mode`.
    static size_t c_bytes(const weight & w, int64_t rows, int mode);
    int shape(int64_t M, int64_t K, int64_t N, int mode, std::string & err, int fmt = -1);
    bool open(std::string & err, bool quant);
    // `quant_src`: for a Q4_K weight under the quantized kernel, row r's
    // GGUF bytes (nullptr: zeros); else rows come through `load_row`.
    bool add_packed(const wkey & key, int64_t N, int64_t n_out, int mode, const bfp16_row_loader & load_row,
                    std::string & err, bool quant = false, const std::function<const uint8_t *(int64_t)> & quant_src = {});
    static void unpack_row(const void * raw, ggml_type type, int64_t r, int64_t K, float * dst);

    std::unique_ptr<npu_bfp16> npu_;
    std::unique_ptr<npu_bfp16::batch> batch_;  // one matmul's calls; buffers only grow
    struct flight {
        std::unique_ptr<npu_bfp16::batch> batch;
        std::vector<const weight *> w;         // the group being filled: each call's weight
        std::vector<int64_t> M;                // each call's rows
        bool chain = false;                    // call 1 reads call 0's output
        // the outputs readable now: calls out_first on, their weights and rows
        std::vector<const weight *> w_out;
        std::vector<int64_t> M_out;
        int out_first = 0;
        int64_t rows = 0, M_max = 0;
        std::vector<size_t> c_cap;             // output bytes reserved per call
        bool readable = false;
    };
    flight & flight_for(int stream);
    std::vector<flight> flights_;  // by stream
    std::unique_ptr<thread_pool> pool_;
    std::map<wkey, weight> weights_;
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int>, int> shapes_;  // (M, K, N, mode, format) -> handle
    bool opened_ = false;
    int qkernel_ = -1;  // npu_'s quantized-weight kernel, once opened
    times times_;
};
