// Block claiming: runs one graph piece the scheduler handed us, the matmuls on
// the NPU and the small ops between them on the host.
//
// Every op in a claimed piece works row by row (one row per prompt token), so
// the piece's rows are split into streams and the streams take turns: the host
// runs one stream's small ops while the NPU runs another's matmuls. Between
// two matmuls the host makes one
// pass over the rows, a few at a time: it reads the NPU's output, runs the
// small ops, and encodes the next matmul's input. Results used only inside
// that pass never leave the cache; the rest stay in host memory, and only
// those something outside the piece reads go back to Vulkan.
#pragma once

#include "ggml.h"

#include <atomic>
#include <functional>
#include <string>
#include <utility>
#include <vector>

class xdna_npu;
class thread_pool;

// What the executor needs from the backend.
struct xdna_io {
    // a tensor's first bytes, from wherever it lives, into host memory
    std::function<void(const ggml_tensor *, void *, size_t)> read;
    // host memory into a tensor's first bytes
    std::function<void(ggml_tensor *, const void *, size_t)> write;
    // a weight-like leaf (a norm's scale), read once and kept
    std::function<const float *(const ggml_tensor *)> param;
    // host memory for the piece, freed when it ends; `pinned` memory is where
    // the GPU copies inputs fastest
    std::function<void *(size_t)> alloc;
    // whether a result must go back to Vulkan: something outside the piece
    // (or something we never saw) reads it
    std::function<bool(const ggml_tensor *, const ggml_cgraph *)> needed_outside;
    // an NPU matmul's weight: made ready (8-bit copy built) and identified
    std::function<bool(const ggml_tensor *, std::string &)> ensure_weight;
    // a gate and an up weight, fused for SiLU(gate) * up on the NPU
    std::function<bool(const ggml_tensor *, const ggml_tensor *, std::string &)> ensure_fused;
    // a weight's raw bytes in host memory, for matmuls on the host; null if
    // they can't be read
    std::function<const void *(const ggml_tensor *)> weight_bytes;
    // Optional. Between these two the piece touches none of the tensors'
    // memory: every input has been read, and no result is written back until
    // before_write returns (false, with the error set, fails the piece).
    // With a row limit, that is when the GPU can run the other rows.
    std::function<void()> inputs_read;
    std::function<bool(std::string &)> before_write;
};

// Host time in those passes, and how many, since the last reset, for
// ZEST_TRACE.
extern double xdna_pass_ms;
extern int64_t xdna_passes;
// Worker CPU time (summed across threads), enabled only by ZEST_TRACE.
extern std::atomic<int64_t> xdna_decode_cpu_ns;
extern std::atomic<int64_t> xdna_encode_cpu_ns;
extern std::atomic<int64_t> xdna_op_cpu_ns[GGML_OP_COUNT];

// Whether the executor implements this op, in this variant. The policy asks
// this before claiming anything but a matmul.
bool xdna_exec_supports(const ggml_tensor * op);

// The input an op keeps as a parameter, not per row (a scale or bias row, rope's frequency factors); null for none.
const ggml_tensor * xdna_exec_param(const ggml_tensor * op);

// The NPU weight copies the piece will use, as running it would make them:
// (weight, null) for a plain one, (gate, up) for a fused pair, (w, w) for
// one weight holding both. Only
// io.needed_outside is used.
void xdna_exec_weights(const ggml_cgraph * g, const xdna_io & io,
                       std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> & out);

// Along which dimension a piece's tensor holds its rows (one per prompt
// token), and the same for input j of a piece's op: -1 for an input that
// isn't per row (a matmul's weight, a scale or bias row, rotary factors).
int xdna_exec_row_dim(const ggml_tensor * t);
int xdna_exec_row_dim(const ggml_tensor * op, int j);

// Runs the piece: its matmuls on `npu`, or with `npu` null on the host (the
// fallback when the NPU fails; slow, but the same results within rounding).
// With `rows`, only the piece's first `rows` rows: it reads and writes only
// those of each tensor. Returns false with `err` set on failure. A failed
// piece has written nothing back, so it can be run again.
bool xdna_exec_piece(const ggml_cgraph * g, xdna_npu * npu, thread_pool & pool, const xdna_io & io, int n_streams,
                     std::string & err, int64_t rows = 0);
