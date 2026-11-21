// ggml backend for the AMD XDNA2 NPU, working next to the Vulkan GPU.
//
// The device registers as an integrated GPU named XDNA0 and is opted in with
// `-dev XDNA0,Vulkan0`: listed first, it has the highest priority. It reports
// no memory of its own, so llama.cpp gives every layer's weights to Vulkan.
// Its buffer type *is* Vulkan's, so it can read every weight and activation
// where they already are, and the scheduler inserts no copies between the two
// backends. That makes supports_op the whole policy: the scheduler hands us
// any op we accept, and Vulkan runs everything else exactly as it would alone.
// supports_op accepts big prompt work and declines the rest (reply generation,
// short prompts).
//
// Without -dev naming XDNA0, llama.cpp keeps only the first integrated GPU it
// finds (Vulkan's), and runs as if this backend weren't there.
//
// Tensors in Vulkan's buffers have no CPU pointer (their `data` is a device
// offset). We read them with an async get through a Vulkan backend of our own,
// into pinned host memory: the GPU does the copy, ~20 GB/s on this machine,
// where the plain get copies out of memory the CPU doesn't cache at 0.2 GB/s.
// We write results with the plain set, a copy into mapped memory, which is
// fast.

#include "ggml-zest.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "xdna-env.h"
#include "xdna-platform.h"
#include "xdna-ref.h"
#ifdef XDNA_HAVE_NPU
#include "xdna-npu.h"
#include "xdna-exec.h"
#include "thread_pool.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define XDNA_DESCRIPTION "AMD XDNA2 NPU (prompt work, next to the Vulkan GPU)"

// dst->ne[1] tokens for a plain matmul - the convention the Vulkan backend
// uses for its offload threshold.
static int64_t xdna_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT: return op->ne[1];
        default:              return ggml_nrows(op);
    }
}

static int64_t env_int(const char * name, int64_t def) { return xdna_env_int(name, def); }

// Batch size at or above which a matmul is ours. At llama.cpp's default
// prompt chunk (-ub 512) the NPU loses to the GPU in every NPU power mode
// tried (median 0.80-0.87x over 20+ paired runs, 2026-10-02); at 2,048 it
// leads (1.06-1.15x). The kernel works in blocks of 512 rows, so a chunk
// between 512 and 1,024 costs about what 1,024 does. `npu --min-chunk`
// sets it.
static int64_t xdna_min_batch() {
    static int64_t v = env_int("GGML_XDNA_MIN_BATCH", 1024);
    return v;
}

// A dispatch costs something whatever it computes, so tiny matmuls stay on
// the GPU even at a large batch.
static int64_t xdna_min_mflop() {
    static int64_t v = env_int("GGML_XDNA_MIN_MFLOP", 256);
    return v;
}

static int xdna_n_threads() {
    static int v = (int) env_int("GGML_XDNA_N_THREADS", [] {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc ? (int) hc : 4;
    }());
    return v;
}

// How the backend runs, decided once, when llama.cpp registers it:
//   npu   the NPU opens and runs the kernel (xdna_npu_usable)
//   host  GGML_XDNA_HOST_ONLY=1: claimed matmuls run on a CPU reference,
//         for tests and CI on machines without an NPU
//   off   neither: no device is offered, and llama.cpp runs as if the backend
//         weren't there
enum class xdna_mode { off, npu, host };

static xdna_mode xdna_run_mode() {
    static const xdna_mode m = [] {
        if (env_int("GGML_XDNA_HOST_ONLY", 0) != 0) {
            GGML_LOG_INFO("xdna: host-only mode (GGML_XDNA_HOST_ONLY): claimed matmuls run on the CPU\n");
            return xdna_mode::host;
        }
#ifdef XDNA_HAVE_NPU
        std::string why;
        if (xdna_npu_usable(why)) {
            GGML_LOG_INFO("xdna: %s\n", why.c_str());
            return xdna_mode::npu;
        }
        GGML_LOG_WARN("xdna: not offering " GGML_XDNA_DEVICE_NAME ": %s\n", why.c_str());
#else
        GGML_LOG_WARN("xdna: not offering " GGML_XDNA_DEVICE_NAME ": built without the NPU\n");
#endif
        return xdna_mode::off;
    }();
    return m;
}

// Set when the NPU fails during a prompt. The piece that failed is finished
// on the GPU, and from then on the backend claims nothing, so everything goes
// to the GPU. Never set in host-only mode.
static std::atomic<bool> g_npu_broken{ false };

static void xdna_npu_failed(const std::string & err) {
    if (!g_npu_broken.exchange(true))
        GGML_LOG_ERROR("xdna: the NPU failed (%s); finishing this step on the GPU and handing everything to the GPU "
                       "from now on\n",
                       err.c_str());
}

//
// the Vulkan device we work next to
//

struct xdna_vk {
    ggml_backend_dev_t         dev       = nullptr;
    ggml_backend_buffer_type_t buft      = nullptr;  // its device memory: ours too
    ggml_backend_buffer_type_t host_buft = nullptr;  // its pinned host memory
};

// Looked up on first use rather than at registration: our DLL loads after
// Vulkan's, but the registry is still registering us while we'd look.
static const xdna_vk & xdna_vulkan() {
    static xdna_vk vk;
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_backend_dev_t d = ggml_backend_dev_by_name("Vulkan0");
        if (!d) {
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t c = ggml_backend_dev_get(i);
                const enum ggml_backend_dev_type t = ggml_backend_dev_type(c);
                if ((t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                    strcmp(ggml_backend_dev_name(c), GGML_XDNA_DEVICE_NAME) != 0) {
                    d = c;
                    break;
                }
            }
        }
        if (!d) {
            GGML_LOG_ERROR("%s: no Vulkan device registered; " GGML_XDNA_DEVICE_NAME " works next to one\n", __func__);
            return;
        }
        vk.dev = d;
        vk.buft = ggml_backend_dev_buffer_type(d);
        vk.host_buft = ggml_backend_dev_host_buffer_type(d);
    });
    return vk;
}

//
// backend
//

// Host memory we read Vulkan's tensors into is Vulkan's pinned memory, where
// the GPU copies fastest. GGML_XDNA_PINNED=0 uses ordinary memory instead: a
// diagnostic, and for software Vulkan devices (CI's lavapipe on Windows hands
// out pinned memory ggml then rejects as misaligned).
static bool xdna_pinned_mem() {
    static const bool v = env_int("GGML_XDNA_PINNED", 1) != 0;
    return v;
}

// Whether a weight's NPU copy goes under the quantized-weight kernel (the budget's choice, below).
static bool xdna_weight_quant(const ggml_tensor * w);

struct xdna_context {
    ggml_backend_t vk = nullptr;  // our own Vulkan backend, for async reads, and pieces after an NPU failure

    // host memory the GPU copies our inputs into; grows as needed
    ggml_backend_buffer_t pin = nullptr;
    size_t pin_size = 0;

    // Vulkan memory for a piece's inputs that live in host memory, when the GPU runs the piece; grows as needed
    ggml_backend_buffer_t gpu_in = nullptr;
    size_t gpu_in_size = 0;

    // Each claimed weight's raw bytes, read once, for matmuls run on the
    // host (the NPU keeps its own BFP16 copy). Only tensors in buffers
    // llama.cpp marks as weights are kept (their contents never change after loading),
    // keyed on where they live and what they are rather than on the tensor
    // struct, whose address can be reused.
    struct wkey {
        const void * buffer, * data;
        int type;
        int64_t ne0, ne1;
        bool operator==(const wkey & o) const {
            return buffer == o.buffer && data == o.data && type == o.type && ne0 == o.ne0 && ne1 == o.ne1;
        }
    };
    struct wkey_hash {
        size_t operator()(const wkey & k) const {
            return std::hash<const void *>()(k.buffer) ^ (std::hash<const void *>()(k.data) * 31) ^ (size_t) k.ne1;
        }
    };
    std::unordered_map<wkey, std::vector<uint8_t>, wkey_hash> weights;
    std::vector<uint8_t> scratch_weight;  // a weight outside a weight buffer, read fresh each time

    std::vector<uint8_t> out;  // a result, before it goes back to Vulkan

    // where the time goes, printed with GGML_XDNA_TRACE=1 every `trace_every`
    // matmuls: reading inputs from Vulkan, writing results back, everything
    // inside graph_compute
    double read_ms = 0, write_ms = 0, compute_ms = 0;
    int64_t matmuls = 0, pieces = 0;
    int64_t late_copies = 0;  // NPU weight copies built inside a piece, not at load
    int64_t gpu_pieces = 0;   // pieces the GPU ran after an NPU failure
    int64_t bare_pieces = 0;  // pieces with no multiply in them, which the GPU ran
    // split pieces (GGML_XDNA_SPLIT): how many, the GPU's time on its rows,
    // and how long the NPU's side then waited for it
    int64_t split_pieces = 0;
    double split_gpu_ms = 0, split_wait_ms = 0;

#ifdef XDNA_HAVE_NPU
    // the NPU, started on first use; null in host-only mode or once it failed
    std::unique_ptr<xdna_npu> npu;
    xdna_npu * get_npu() {
        if (g_npu_broken || xdna_run_mode() != xdna_mode::npu) return nullptr;
        if (npu) return npu.get();
        npu = std::make_unique<xdna_npu>();
        std::string err;
        if (!npu->init(xdna_n_threads(), err)) {
            xdna_npu_failed(err);
            npu.reset();
        }
        return npu.get();
    }
    // threads for pieces run on the host when the NPU has failed
    std::unique_ptr<thread_pool> host_pool;
    thread_pool & host_threads() {
        if (!host_pool) host_pool = std::make_unique<thread_pool>(xdna_n_threads());
        return *host_pool;
    }

    // Weight copies built while the model loads: graph_optimize lists the
    // copies each piece will use as llama.cpp plans it. The load callback
    // waits for the builder before returning, so the first request does not
    // pay for the copies.
    struct build_job { const ggml_tensor * w, * up; };  // up: the up weight of a fused gate/up
    std::thread builder;
    std::mutex build_mu;
    std::deque<build_job> build_todo;
    std::set<xdna_npu::wkey> build_queued;  // ever queued, as the NPU keys them
    bool build_running = false, build_stop = false;
    std::string build_err;
    double build_wait_ms = 0;
    int64_t built_at_load = 0;

    void queue_copies(const std::vector<build_job> & jobs) {
        std::lock_guard<std::mutex> lk(build_mu);
        bool added = false;
        for (const build_job & j : jobs) {
            const xdna_npu::wkey key = j.up ? xdna_npu::fused_key(xdna_npu::gate_key(j.w, j.up))
                                            : xdna_npu::wkey{ j.w->buffer, j.w->data, (int) j.w->type, j.w->ne[0], j.w->ne[1] };
            if (!build_queued.insert(key).second) continue;
            build_todo.push_back(j);
            added = true;
        }
        if (!added || build_running) return;
        if (builder.joinable()) builder.join();  // finished: it cleared build_running
        build_running = true;
        builder = std::thread([this] { build_copies(); });
    }

    void build_copies() {
        std::vector<uint8_t> raw, raw_up;
        for (;;) {
            build_job j;
            {
                std::lock_guard<std::mutex> lk(build_mu);
                if (build_stop || build_todo.empty() || !build_err.empty()) {
                    build_running = false;
                    return;
                }
                j = build_todo.front();
                build_todo.pop_front();
            }
            const xdna_npu::wkey key = { j.w->buffer, j.w->data, (int) j.w->type, j.w->ne[0], j.w->ne[1] };
            if (npu->has_weight(j.up ? xdna_npu::fused_key(xdna_npu::gate_key(j.w, j.up)) : key)) continue;
            std::string err;
            bool ok;
            if (j.up) {
                ok = add_fused(j.w, j.up, raw, raw_up, err);
            } else {
                raw.resize(ggml_nbytes(j.w));
                read(j.w, raw.data());
                ok = npu->add_weight(key, raw.data(), err, xdna_weight_quant(j.w));
            }
            std::lock_guard<std::mutex> lk(build_mu);
            if (ok) built_at_load++;
            else build_err = std::string(j.w->name) + ": " + err;
        }
    }

    // A gate and an up weight's fused NPU copy; up == gate is one weight holding the gate's rows, then the up's.
    bool add_fused(const ggml_tensor * gw, const ggml_tensor * uw, std::vector<uint8_t> & g, std::vector<uint8_t> & u,
                   std::string & err) {
        g.resize(ggml_nbytes(gw));
        read(gw, g.data());
        const void * up = g.data() + ggml_row_size(gw->type, gw->ne[0]) * (gw->ne[1] / 2);
        if (uw != gw) {
            u.resize(ggml_nbytes(uw));
            read(uw, u.data());
            up = u.data();
        }
        return npu->add_fused(xdna_npu::gate_key(gw, uw), g.data(), up, err, xdna_weight_quant(gw));
    }

    // Waits for the copies queued so far. A copy that failed to build fails
    // the NPU, as it would have failed the piece.
    void finish_copies() {
        if (!builder.joinable()) return;
        const auto t0 = std::chrono::steady_clock::now();
        builder.join();
        build_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!build_err.empty()) xdna_npu_failed("building a weight copy at load: " + build_err);
        build_err.clear();
    }
#endif

    ~xdna_context() {
#ifdef XDNA_HAVE_NPU
        {
            std::lock_guard<std::mutex> lk(build_mu);
            build_stop = true;
        }
        if (builder.joinable()) builder.join();
#endif
        for (pin_buf & p : pool) {
            if (p.buf) ggml_backend_buffer_free(p.buf);
            else ggml_aligned_free(p.mem, p.size);
        }
        if (pin) ggml_backend_buffer_free(pin);
        if (gpu_in) ggml_backend_buffer_free(gpu_in);
        if (vk) ggml_backend_free(vk);
    }

    void * pinned(size_t size) {
        if (size > pin_size) {
            if (pin) ggml_backend_buffer_free(pin);
            pin = ggml_backend_buft_alloc_buffer(xdna_pinned_mem() ? xdna_vulkan().host_buft : ggml_backend_cpu_buffer_type(),
                                                 size);
            pin_size = pin ? size : 0;
        }
        return pin ? ggml_backend_buffer_get_base(pin) : nullptr;
    }

    // a tensor (or its first `bytes`) from Vulkan memory into host memory, the GPU doing the copy
    void read(const ggml_tensor * t, void * dst, size_t bytes = 0) {
        if (!bytes) bytes = ggml_nbytes(t);
        if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {  // a graph input the CPU holds
            memcpy(dst, t->data, bytes);
            return;
        }
        ggml_backend_tensor_get_async(vk, t, dst, 0, bytes);
        ggml_backend_synchronize(vk);
    }

    // A result (or its first `bytes`) from the piece's host memory into Vulkan
    // memory: the GPU copies it from our pinned memory, when finish_writes
    // says so. ggml's own write is a memcpy into the mapped buffer, and what
    // a memcpy of under 24 MB leaves there the GPU then reads slowly (glibc
    // uses non-temporal stores only from that size): the next op to read it
    // ran at less than half speed, which on a mixture-of-experts model was
    // every layer's expert multiplies (8.4 ms each against 3.8). The GPU's
    // copy is also three times faster than the memcpy.
    bool writes_pending = false;
    void write(ggml_tensor * t, const void * src, size_t bytes) {
        if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {  // a result the CPU holds
            ggml_backend_tensor_set(t, src, 0, bytes);
            return;
        }
        ggml_backend_tensor_set_async(vk, t, src, 0, bytes);
        writes_pending = true;
    }
    // Waits for those copies. The memory they read has to stay as it is until then.
    void finish_writes() {
        if (!writes_pending) return;
        ggml_backend_synchronize(vk);
        writes_pending = false;
    }

    // Block claiming's host memory: pinned buffers (xdna_pinned_mem), handed
    // out per piece and all returned when it ends, reused by the next.
    struct pin_buf { ggml_backend_buffer_t buf; void * mem; size_t size; bool used; };
    std::vector<pin_buf> pool;
    void * pool_alloc(size_t n) {
        const bool pinned_mem = xdna_pinned_mem();
        pin_buf * best = nullptr;
        for (pin_buf & p : pool)
            if (!p.used && p.size >= n && (!best || p.size < best->size)) best = &p;
        if (!best) {
            pin_buf nb = { nullptr, nullptr, n, false };
            if (pinned_mem) {
                nb.buf = ggml_backend_buft_alloc_buffer(xdna_vulkan().host_buft, n);
                if (!nb.buf) return nullptr;
                nb.mem = ggml_backend_buffer_get_base(nb.buf);
            } else {
                nb.mem = ggml_aligned_malloc(n);
                if (!nb.mem) return nullptr;
            }
            pool.push_back(nb);
            best = &pool.back();
        }
        best->used = true;
        return best->mem;
    }
    void pool_release() { for (pin_buf & p : pool) p.used = false; }

    // small weight-like leaves (norm scales), read once
    std::unordered_map<wkey, std::vector<uint8_t>, wkey_hash> params;
    const float * param(const ggml_tensor * t) {
        const wkey key = { t->buffer, t->data, (int) t->type, t->ne[0], t->ne[1] };
        auto it = params.find(key);
        if (it != params.end()) return (const float *) it->second.data();
        std::vector<uint8_t> & v = params[key];
        v.resize(ggml_nbytes(t));
        read(t, v.data());
        return (const float *) v.data();
    }

    const std::vector<uint8_t> & weight(const ggml_tensor * w) {
        if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            scratch_weight.resize(ggml_nbytes(w));
            read(w, scratch_weight.data());
            return scratch_weight;
        }
        const wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
        auto it = weights.find(key);
        if (it != weights.end()) return it->second;
        std::vector<uint8_t> & bytes = weights[key];
        bytes.resize(ggml_nbytes(w));
        read(w, bytes.data());
        return bytes;
    }
};

//
// the memory budget for the NPU's weight copies
//

// What the NPU's copy of a K x N weight takes: 9 bytes for every 8 values,
// with N padded as the kernel runs it (pad_n in xdna-npu.cpp). A fused
// gate/up pair takes no more than its two weights apart. Host-only mode
// budgets the same, so it decides as the NPU would.
static double xdna_copy_bytes(int64_t K, int64_t N) {
    return (double) ((N + 511) / 512 * 512) * (double) K / 8 * 9;
}

// Which kernel a weight's copy is for (GGML_XDNA_WEIGHTS):
//   bfp16  (default) the BFP16 kernel, every copy 9 bits a weight
//   quant  the kernel on quantized weights (xdna-npu.h): Q4_K at 4.75 bits a
//          weight, unpacked on the NPU, other types copied through as
//          BFP16; ~30% slower a GEMM (24 of the 32 cores multiply), but
//          about half the memory for a Q4_K model
// One kernel for the whole model: switching the NPU between the two costs
// ~2.4 ms, and a mix chosen as the copies fill the limit gains nothing (by
// the time BFP16 copies stop fitting, a quantized one doesn't fit either;
// Qwen3.8-27B, 2026-11-18).
enum class xdna_wmode { bfp16, quant };
static xdna_wmode xdna_weights_mode() {
    static const xdna_wmode m = [] {
        const char * s = xdna_env("GGML_XDNA_WEIGHTS");
        if (!s || !*s || !strcmp(s, "bfp16")) return xdna_wmode::bfp16;
        if (!strcmp(s, "quant")) return xdna_wmode::quant;
        GGML_LOG_WARN("xdna: GGML_XDNA_WEIGHTS=%s is not bfp16 or quant; using bfp16\n", s);
        return xdna_wmode::bfp16;
    }();
    return m;
}

// A quantized-kernel copy's bytes, or negative when the weight can't have one.
static double xdna_qcopy_bytes(const ggml_tensor * w) {
#ifdef XDNA_HAVE_NPU
    if (xdna_npu_has_qshape((int) w->type, w->ne[0], w->ne[1])) return xdna_npu_qcopy_bytes((int) w->type, w->ne[0], w->ne[1]);
#endif
    (void) w;
    return -1;
}

// The layer a weight belongs to, from llama.cpp's names ("blk.12.attn_q.weight");
// -1 for the others (the output matrix).
static int xdna_layer_of(const ggml_tensor * w) {
    int layer = -1;
    return sscanf(w->name, "blk.%d.", &layer) == 1 ? layer : -1;
}

static constexpr double GB = 1024.0 * 1024 * 1024;

// The copies are built at load (graph_optimize) or on first use, and freed
// with the context, so without a limit they could take more memory than the
// machine has. Weights are taken in the order llama.cpp asks about them (layer by
// layer, while it plans the first prompt at load) until the next would pass
// the limit; from then on that layer and every new weight stay on the GPU.
// What's taken stays taken, so the split never changes between prompts. When
// the last context goes, its copies go with it, and the next starts afresh.
//
// The limit is the memory free when the first weight is asked about (the
// model and its context are loaded by then), less what's kept back for
// llama.cpp's work buffers and the rest of the machine: the larger of 4 GB and
// a tenth of memory. On Windows it's also at most 20 GB: the NPU holds only so
// much there (about 25.9 GiB on the 88 GB test machine), and past that Windows
// moves its buffers out and they come back damaged, silently (Qwen3.8-27B,
// 2026-10-03; the NPU's results are checked for that too, xdna-npu.cpp). Linux
// has no such ceiling where we looked (Strix Halo, 2026-11-20), so there the
// memory rule is the only limit. GGML_XDNA_MAX_COPY_GB sets it instead.
//
// supports_op is asked with no context, so there's one budget per process:
// two contexts alive at once on the same model would each build the copies it
// counted once.
struct xdna_budget {
    std::mutex mu;
    int contexts = 0;  // backends alive
#ifdef _WIN32
    static constexpr double DEFAULT_MAX = 20 * GB;
#else
    static constexpr double DEFAULT_MAX = 0;  // none
#endif
    bool started = false, cut = false, running = false, overridden = false, capped = false, locked = false;
    double limit = 0, free = 0, kept_back = 0, taken = 0;
    struct entry {
        int layer;
        double bytes;  // of the copy it gets
        bool quant;    // under the quantized kernel
    };
    std::unordered_map<xdna_context::wkey, entry, xdna_context::wkey_hash> in, out;  // taken; left to the GPU

    // where the limit came from, for the log
    std::string source() const {
        char s[112];
        if (locked) snprintf(s, sizeof(s), "the locked-memory limit, less 2 GB for other NPU buffers; ulimit -l raises it");
        else if (overridden) snprintf(s, sizeof(s), "set by GGML_XDNA_MAX_COPY_GB");
        else if (capped) snprintf(s, sizeof(s), "the most by default; zest --memory-gb raises it");
        else snprintf(s, sizeof(s), "%.1f GB free, less %.1f GB kept back", free / GB, kept_back / GB);
        return s;
    }

    void reset() {
        started = cut = running = overridden = capped = locked = false;
        limit = free = kept_back = taken = 0;
        in.clear();
        out.clear();
    }
};
static xdna_budget g_budget;

// Whether the weight's copy fits the budget. Weights with no data (llama.cpp's
// memory-fitting trial and load-time checks), weights outside a weight buffer
// (nothing is kept for them) and questions asked with no context alive build
// nothing, so they pass.
static bool xdna_budget_takes(const ggml_tensor * w) {
    if (!w->buffer || !w->data || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        return true;
    xdna_budget & b = g_budget;
    std::lock_guard<std::mutex> lk(b.mu);
    if (b.contexts == 0) return true;
    const xdna_context::wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
    if (b.in.count(key)) return true;
    if (b.out.count(key)) return false;
    if (!b.started) {
        b.started = true;
        const xdna_memory m = xdna_system_memory();
        b.free = m.available;
        b.kept_back = std::max(4 * GB, m.total / 10);
        const char * s = xdna_env("GGML_XDNA_MAX_COPY_GB");
        b.overridden = s != nullptr;
        const double by_memory = b.free - b.kept_back;
        b.capped = !s && xdna_budget::DEFAULT_MAX > 0 && by_memory > xdna_budget::DEFAULT_MAX;
        b.limit = s ? atof(s) * GB : b.capped ? xdna_budget::DEFAULT_MAX : by_memory;
#ifdef XDNA_HAVE_NPU
        const double lock = xdna_npu_lock_limit();
        if (lock >= 0 && lock - XDNA_LOCK_KEPT < b.limit) {
            b.limit = lock - XDNA_LOCK_KEPT;
            b.locked = true;
        }
#endif
    }
    const double qbytes = xdna_weights_mode() == xdna_wmode::quant ? xdna_qcopy_bytes(w) : -1;
    const xdna_budget::entry e = { xdna_layer_of(w), qbytes >= 0 ? qbytes : xdna_copy_bytes(w->ne[0], w->ne[1]), qbytes >= 0 };
    if (!b.cut && b.taken + e.bytes <= b.limit) {
        b.in[key] = e;
        b.taken += e.bytes;
        return true;
    }
    if (!b.cut) {
        b.cut = true;
        // The layer's weights taken so far go back to the GPU with it, so no
        // layer is split between the two. Once a prompt has run, their copies
        // may exist, and they stay.
        if (e.layer >= 0 && !b.running)
            for (auto it = b.in.begin(); it != b.in.end();) {
                if (it->second.layer != e.layer) { ++it; continue; }
                b.taken -= it->second.bytes;
                b.out.insert(*it);
                it = b.in.erase(it);
            }
        std::unordered_set<int> layers;
        for (const auto & kv : b.in)
            if (kv.second.layer >= 0) layers.insert(kv.second.layer);
        char split[96] = "no layer fits, so all run on the GPU";
        if (!layers.empty()) snprintf(split, sizeof(split), "the first %zu layers on the NPU, the rest on the GPU", layers.size());
        GGML_LOG_WARN("xdna: NPU weight copies limited to %.1f GB (%s): %s\n", std::max(0.0, b.limit) / GB,
                      b.source().c_str(), split);
    }
    b.out[key] = e;
    return false;
}

// When nothing was left to the GPU, one line on what the copies take, as a
// context first runs a piece. (A limit reached says so where it's reached.)
static void xdna_budget_report() {
    xdna_budget & b = g_budget;
    std::lock_guard<std::mutex> lk(b.mu);
    if (b.running) return;
    b.running = true;
    if (!b.started || b.cut) return;
    size_t quant = 0;
    for (const auto & kv : b.in) quant += kv.second.quant;
    GGML_LOG_INFO("xdna: NPU weight copies: %.1f GB for %zu weights, %zu under the quantized kernel (limit %.1f GB: %s)\n",
                  b.taken / GB, b.in.size(), quant, std::max(0.0, b.limit) / GB, b.source().c_str());
}

// Whether a weight's copy goes under the quantized kernel: as the budget
// decided, or, for a weight it never counted, as GGML_XDNA_WEIGHTS says
// without a budget (quant: when it can).
static bool xdna_weight_quant(const ggml_tensor * w) {
    xdna_budget & b = g_budget;
    {
        std::lock_guard<std::mutex> lk(b.mu);
        const xdna_context::wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
        const auto it = b.in.find(key);
        if (it != b.in.end()) return it->second.quant;
    }
    return xdna_weights_mode() == xdna_wmode::quant && xdna_qcopy_bytes(w) >= 0;
}

//
// what we noticed about the model, said once as its first prompt runs
//

// If the user asked for the NPU, the NPU runs the model. A model the NPU
// handles badly gets a warning, never a refusal. Two so far:
//
//   mixture of experts   The expert step is a MUL_MAT_ID, which we don't
//                        take, so only the attention and shared weights are
//                        left for us. In the sweep these models' answers also
//                        drifted further from the GPU's than any dense
//                        model's (LFM2.5 8B-A1B 0.020, gpt-oss-20b 0.037,
//                        against a 0.01 bar), most likely because rounding
//                        each layer's router multiply to 8 bits flips which
//                        experts a token uses.
//   a narrow model       Below a width of 2,048 the GPU alone was faster in
//                        every test (0.34-0.84x). 2,048 ties.
//
// Both are noticed while llama.cpp plans the graph and said from
// graph_compute: llama.cpp's memory-fitting trial plans graphs with warnings
// hidden, but never computes one.
struct xdna_notes {
    std::atomic<bool>    moe{ false };
    std::atomic<int64_t> width{ 0 };
    std::atomic<bool>    said{ false };
    void reset() {
        moe   = false;
        width = 0;
        said  = false;
    }
};
static xdna_notes g_notes;

static constexpr int64_t XDNA_NARROW_WIDTH = 2048;

// Whether a name is "blk.<layer>.<tail>", as llama.cpp names a layer's weights.
static bool xdna_name_in_layer(const char * name, const char * tail) {
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * p = name + 4;
    while (*p >= '0' && *p <= '9') p++;
    return *p == '.' && strcmp(p + 1, tail) == 0;
}

static void xdna_note_model(const ggml_tensor * op) {
    if (op->op == GGML_OP_MUL_MAT_ID && !g_notes.moe.load(std::memory_order_relaxed))
        g_notes.moe.store(true, std::memory_order_relaxed);
    if (g_notes.width.load(std::memory_order_relaxed) != 0) return;
    // The width is the first dimension of the token embedding, or of a
    // layer's q weight if we're never shown the embedding.
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        const ggml_tensor * s = op->src[j];
        if (!s) continue;
        if (strcmp(s->name, "token_embd.weight") == 0 || xdna_name_in_layer(s->name, "attn_q.weight") ||
            xdna_name_in_layer(s->name, "attn_qkv.weight")) {
            g_notes.width.store(s->ne[0], std::memory_order_relaxed);
            return;
        }
    }
}

static void xdna_say_notes() {
    if (g_notes.said.exchange(true)) return;
    if (g_notes.moe.load())
        GGML_LOG_WARN("xdna: mixture-of-experts model: the NPU cannot take the expert step, so it gets only part of "
                      "the work, and in our tests these models' answers drifted further from the GPU's than any dense "
                      "model's. Running it on the NPU as asked; -dev Vulkan0 would use the GPU alone\n");
    const int64_t width = g_notes.width.load();
    if (width != 0 && width < XDNA_NARROW_WIDTH)
        GGML_LOG_WARN("xdna: narrow model (width %lld): the GPU alone was faster than the NPU on every model this "
                      "size we tested. Running it on the NPU as asked\n",
                      (long long) width);
}

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return GGML_XDNA_DEVICE_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    delete (xdna_context *) backend->context;
    delete backend;
    std::lock_guard<std::mutex> lk(g_budget.mu);
    if (--g_budget.contexts == 0) {
        g_budget.reset();
        g_notes.reset();  // the next model gets its own warnings
    }
}

static bool xdna_mul_mat(xdna_context & ctx, ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0], * src1 = node->src[1];
    void * x = ctx.pinned(ggml_nbytes(src1));
    if (!x) {
        GGML_LOG_ERROR("%s: cannot allocate %zu bytes of pinned memory\n", __func__, ggml_nbytes(src1));
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    ctx.read(src1, x);
    ctx.read_ms += since(t0);
    ctx.out.resize(ggml_nbytes(node));
    ctx.matmuls++;

#ifdef XDNA_HAVE_NPU
    // On the NPU, for weights llama.cpp loaded (a weight buffer) in a built
    // shape: its BFP16 copy is made the first time it's used.
    xdna_npu * npu = ctx.get_npu();
    if (npu && src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        xdna_npu_has_shape(src0->ne[0], src0->ne[1])) {
        const xdna_npu::wkey key = { src0->buffer, src0->data, (int) src0->type, src0->ne[0], src0->ne[1] };
        std::string err;
        bool ok = npu->has_weight(key);
        if (!ok) {
            ctx.late_copies++;
            ctx.scratch_weight.resize(ggml_nbytes(src0));
            ctx.read(src0, ctx.scratch_weight.data());
            ok = npu->add_weight(key, ctx.scratch_weight.data(), err, xdna_weight_quant(src0));
        }
        if (ok) ok = npu->mul_mat(key, (const float *) x, src1->ne[1], (float *) ctx.out.data(), err);
        if (ok) {
            t0 = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(node, ctx.out.data(), 0, ggml_nbytes(node));
            ctx.write_ms += since(t0);
            return true;
        }
        xdna_npu_failed(std::string(node->name) + ": " + err);
        return false;  // the caller hands the graph to the GPU
    }
#endif
    const std::vector<uint8_t> & w = ctx.weight(src0);

    // host-memory stand-ins for the three tensors, for the reference matmul
    ggml_tensor s0 = *src0, s1 = *src1, d = *node;
    s0.data = (void *) w.data();
    s1.data = x;
    d.data = ctx.out.data();
    xdna_ref_mul_mat(&s0, &s1, &d, xdna_n_threads());

    ggml_backend_tensor_set(node, ctx.out.data(), 0, ggml_nbytes(node));
    return true;
}

// A piece on the GPU (after an NPU failure, or one with nothing for the NPU); inputs in host memory (rotary
// positions) move into Vulkan's for the call.
static bool xdna_gpu_compute(xdna_context & ctx, ggml_cgraph * g) {
    std::vector<ggml_tensor *> owners, views;
    std::unordered_set<ggml_tensor *> seen;
    for (int i = 0; i < g->n_nodes; i++)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * s = g->nodes[i]->src[j];
            if (!s || !s->buffer || !ggml_backend_buffer_is_host(s->buffer) || !seen.insert(s).second) continue;
            if (!s->view_src) owners.push_back(s);
            else {
                views.push_back(s);
                if (seen.insert(s->view_src).second) owners.push_back(s->view_src);
            }
        }
    const xdna_vk & vk = xdna_vulkan();
    const size_t align = ggml_backend_buft_get_alignment(vk.buft);
    size_t need = 0;
    for (const ggml_tensor * o : owners) need += GGML_PAD(ggml_nbytes(o), align);
    if (need > ctx.gpu_in_size) {
        if (ctx.gpu_in) ggml_backend_buffer_free(ctx.gpu_in);
        ctx.gpu_in = ggml_backend_buft_alloc_buffer(vk.buft, need);
        ctx.gpu_in_size = ctx.gpu_in ? need : 0;
        if (!ctx.gpu_in) return false;
    }
    struct place { ggml_tensor * t; ggml_backend_buffer_t buffer; void * data; };
    std::vector<place> moved;
    size_t off = 0;
    for (ggml_tensor * o : owners) {
        moved.push_back({ o, o->buffer, o->data });
        o->buffer = ctx.gpu_in;
        o->data = (char *) ggml_backend_buffer_get_base(ctx.gpu_in) + off;
        ggml_backend_tensor_set(o, moved.back().data, 0, ggml_nbytes(o));
        off += GGML_PAD(ggml_nbytes(o), align);
    }
    for (ggml_tensor * v : views) {
        moved.push_back({ v, v->buffer, v->data });
        v->buffer = v->view_src->buffer;
        v->data = (char *) v->view_src->data + v->view_offs;
    }
    const bool ok = ggml_backend_graph_compute(ctx.vk, g) == GGML_STATUS_SUCCESS;
    for (auto it = moved.rbegin(); it != moved.rend(); ++it) {
        it->t->buffer = it->buffer;
        it->t->data = it->data;
    }
    return ok;
}

// The row split. The NPU takes GGML_XDNA_SPLIT percent of a piece's rows (in
// whole 512s, default half) and the GPU the rest of them at the same time, so
// both devices multiply at once instead of taking turns: the NPU ties the GPU
// on the multiplies, and a piece it took whole cost what the GPU would have
// spent on it plus the host's work. 0: the NPU takes whole pieces.
static int64_t xdna_split_percent() {
    static const int64_t v = env_int("GGML_XDNA_SPLIT", 50);
    return v;
}

// An op that computes nothing: its result is its input's memory, seen another way.
static bool xdna_is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// A piece's graph over rows [r0, r1) of its per-row tensors, for the GPU: the
// same ops in the same order (the memory llama.cpp planned for the piece is
// reused between its nodes in that order), each tensor a view of those rows of
// the original's memory, so the GPU reads the piece's inputs and writes its
// results in place.
struct xdna_rows_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * g = nullptr;
    xdna_rows_graph() = default;
    xdna_rows_graph(const xdna_rows_graph &) = delete;
    ~xdna_rows_graph() { if (ctx) ggml_free(ctx); }
};

// False if the piece has something this doesn't split (then it isn't split;
// `why` says what).
static bool xdna_rows_view(xdna_context & ctx, const ggml_cgraph * g, int64_t T, int64_t r0, int64_t r1, xdna_rows_graph & out,
                           std::string & why) {
    const size_t max_tensors = (size_t) g->n_nodes * (1 + GGML_MAX_SRC) + 8;
    ggml_init_params ip = { ggml_tensor_overhead() * max_tensors + ggml_graph_overhead_custom(max_tensors, false), nullptr, true };
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    std::unordered_set<const ggml_tensor *> in_piece;
    for (int i = 0; i < g->n_nodes; i++)
        if (g->nodes[i]->op != GGML_OP_NONE) in_piece.insert(g->nodes[i]);
    std::unordered_map<const ggml_tensor *, ggml_tensor *> view;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> host_inputs;  // (original, its view) for inputs in host memory
    bool ok = true;
    // t's rows [r0, r1), along dimension d: one of the piece's ops with the
    // same inputs' rows, or for an input from outside, a leaf
    std::function<ggml_tensor *(const ggml_tensor *, int)> rows = [&](const ggml_tensor * t, int d) -> ggml_tensor * {
        auto it = view.find(t);
        if (it != view.end()) return it->second;
        const bool node = in_piece.count(t) != 0;
        // Rows along the last dimension in use, so a row range is one stretch
        // of memory whatever the strides inside a row. An op that shares
        // another tensor's memory (a reshape, a slice, an op done in place)
        // shares the same rows of it.
        if (t->ne[d] != T || (node && t->view_src && t->view_src->ne[xdna_exec_row_dim(t->view_src)] != T)) {
            if (ok) why = std::string(ggml_op_name(t->op)) + " " + t->name + ": its rows don't run along its last dimension";
            ok = false;
            return nullptr;
        }
        int64_t ne[GGML_MAX_DIMS];
        for (int i = 0; i < GGML_MAX_DIMS; i++) ne[i] = i == d ? r1 - r0 : t->ne[i];
        ggml_tensor * v = ggml_new_tensor(out.ctx, t->type, GGML_MAX_DIMS, ne);
        view[t] = v;
        for (int i = 0; i <= d; i++) v->nb[i] = t->nb[i];  // the original's strides (a slice's are its source's)
        for (int i = d + 1; i < GGML_MAX_DIMS; i++) v->nb[i] = v->nb[i - 1] * (size_t) v->ne[i - 1];
        ggml_set_name(v, t->name);
        v->buffer = t->buffer;
        if (!node) {
            // wherever it comes from, its memory holds it whole
            if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) host_inputs.push_back({ t, v });  // placed below
            else v->data = (char *) t->data + r0 * (int64_t) t->nb[d];
            return v;
        }
        v->op = t->op;
        memcpy(v->op_params, t->op_params, sizeof(v->op_params));
        v->flags = t->flags;
        if (t->view_src) {
            ggml_tensor * o = rows(t->view_src, xdna_exec_row_dim(t->view_src));
            if (!o) return nullptr;
            v->view_src = o;
            v->view_offs = t->view_offs;
            v->buffer = o->buffer;
            v->data = (char *) o->data + t->view_offs;
        } else {
            v->data = (char *) t->data + r0 * (int64_t) t->nb[d];
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * s = t->src[j];
            if (!s) continue;
            const int sd = xdna_exec_row_dim(t, j);
            if (sd < 0 && s->op != GGML_OP_NONE) {  // a weight or a parameter row is a leaf
                if (ok) why = std::string(ggml_op_name(t->op)) + " " + t->name + ": a computed weight or parameter row";
                ok = false;
            }
            v->src[j] = sd < 0 ? (ggml_tensor *) s : rows(s, sd);
            if (!v->src[j] || !ok) return nullptr;
        }
        return v;
    };
    out.g = ggml_new_graph_custom(out.ctx, max_tensors, false);
    for (int i = 0; i < g->n_nodes && ok; i++) {
        const ggml_tensor * n = g->nodes[i];
        // A reshape or a transpose computes nothing, and llama.cpp's
        // scheduler leaves any that follow a piece's last op in the piece,
        // whatever they are of (another layer's state, or a result of the
        // piece turned for an op outside it). Only one that an op of the
        // piece reads matters here, and that op brings it in.
        if (xdna_is_view_op(n->op)) continue;
        ggml_tensor * v = rows(n, xdna_exec_row_dim(n));
        // one at a time, so the nodes keep the piece's order
        if (v) ggml_build_forward_expand(out.g, v);
    }
    if (!ok) return false;
    // inputs the CPU holds (rotary positions): their rows, copied into the GPU's memory
    const xdna_vk & vk = xdna_vulkan();
    const size_t align = ggml_backend_buft_get_alignment(vk.buft);
    size_t need = 0;
    for (const auto & h : host_inputs) need += GGML_PAD(ggml_nbytes(h.second), align);
    if (need > ctx.gpu_in_size) {
        if (ctx.gpu_in) ggml_backend_buffer_free(ctx.gpu_in);
        ctx.gpu_in = ggml_backend_buft_alloc_buffer(vk.buft, need);
        ctx.gpu_in_size = ctx.gpu_in ? need : 0;
        if (!ctx.gpu_in) return false;
    }
    size_t off = 0;
    for (const auto & h : host_inputs) {
        ggml_tensor * v = h.second;
        const int d = xdna_exec_row_dim(h.first);
        v->buffer = ctx.gpu_in;
        v->data = (char *) ggml_backend_buffer_get_base(ctx.gpu_in) + off;
        ggml_backend_tensor_set(v, (const char *) h.first->data + r0 * (int64_t) h.first->nb[d], 0, ggml_nbytes(v));
        off += GGML_PAD(ggml_nbytes(v), align);
    }
    // the piece's views of those inputs were made before their memory was known
    for (auto & kv : view)
        if (kv.second->view_src && kv.second->view_src->buffer == ctx.gpu_in) {
            kv.second->buffer = ctx.gpu_in;
            kv.second->data = (char *) kv.second->view_src->data + kv.second->view_offs;
        }
    return true;
}

static bool xdna_trace();
static void xdna_trace_piece(const ggml_cgraph * cgraph);
static bool xdna_blocks();
static bool xdna_read_outside(const ggml_tensor * t, const std::unordered_set<const ggml_tensor *> & in_piece);

// How many streams block claiming splits a piece's rows into.
static int xdna_streams() {
    static int v = (int) env_int("GGML_XDNA_STREAMS", 2);
    return v;
}

static ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    xdna_context & ctx = *(xdna_context *) backend->context;
    xdna_budget_report();
    xdna_say_notes();
#ifdef XDNA_HAVE_NPU
    ctx.finish_copies();
#endif
    if (xdna_trace()) xdna_trace_piece(cgraph);
    const auto t_start = std::chrono::steady_clock::now();
    struct timer {
        xdna_context & c;
        std::chrono::steady_clock::time_point t;
        ~timer() {
            c.compute_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
            c.pieces++;
            // A single-projection benchmark needs the same breakdown as a
            // full prompt, without waiting for 196 accumulated matmuls.
            if (!xdna_trace() || (c.matmuls < 196 && !xdna_env("GGML_XDNA_TRACE_EVERY"))) return;
#ifdef XDNA_HAVE_NPU
            const xdna_npu::times n = c.npu ? c.npu->take_times() : xdna_npu::times{};
#else
            const struct {
                double encode_ms = 0, npu_ms = 0, decode_ms = 0, pack_ms = 0;
                double weight_shape_ms = 0, weight_alloc_ms = 0, weight_encode_ms = 0, weight_upload_ms = 0;
                int calls = 0;
            } n;
#endif
            fprintf(stderr,
                    "xdna times, %lld matmuls in %lld pieces: all %.1f ms | read inputs %.1f, encode %.1f, NPU %.1f "
                    "(%d calls), decode %.1f, write results %.1f, weight copies %.1f ms\n",
                    (long long) c.matmuls, (long long) c.pieces, c.compute_ms, c.read_ms, n.encode_ms, n.npu_ms, n.calls,
                    n.decode_ms, c.write_ms, n.pack_ms);
#ifdef XDNA_HAVE_NPU
            // block claiming reads NPU outputs and encodes inputs inside the
            // passes, so there "encode" is only padding and copies, and
            // "decode" is 0
            fprintf(stderr, "xdna host passes: %.1f ms in %lld\n", xdna_pass_ms, (long long) xdna_passes);
            fprintf(stderr, "xdna host pass worker CPU: decode %.1f, encode %.1f ms",
                    xdna_decode_cpu_ns.exchange(0) / 1e6, xdna_encode_cpu_ns.exchange(0) / 1e6);
            for (int op = 0; op < GGML_OP_COUNT; op++) {
                const double ms = xdna_op_cpu_ns[op].exchange(0) / 1e6;
                if (ms >= 1.0) fprintf(stderr, ", %s %.1f", ggml_op_name((ggml_op) op), ms);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "xdna weight copies: %lld built at load, %lld inside pieces; %.1f ms waiting for those "
                            "built at load\n",
                    (long long) c.built_at_load, (long long) c.late_copies, c.build_wait_ms);
            fprintf(stderr, "xdna weight preparation: shapes %.1f, allocate %.1f, dequant+encode %.1f, sync %.1f ms\n",
                    n.weight_shape_ms, n.weight_alloc_ms, n.weight_encode_ms, n.weight_upload_ms);
            if (c.gpu_pieces) fprintf(stderr, "xdna: %lld pieces on the GPU after the NPU failed\n", (long long) c.gpu_pieces);
            if (c.bare_pieces) fprintf(stderr, "xdna: %lld pieces with no multiply, run by the GPU\n", (long long) c.bare_pieces);
            if (c.split_pieces)
                fprintf(stderr, "xdna split: %lld pieces shared with the GPU, %.1f ms handing it its rows, %.1f ms waiting for "
                                "them after the NPU's\n",
                        (long long) c.split_pieces, c.split_gpu_ms, c.split_wait_ms);
            c.split_pieces = 0;
            c.split_gpu_ms = c.split_wait_ms = 0;
            xdna_pass_ms = 0;
            xdna_passes = 0;
            c.build_wait_ms = 0;
            c.built_at_load = 0;
#endif
            c.read_ms = c.write_ms = c.compute_ms = 0;
            c.matmuls = c.pieces = c.late_copies = c.gpu_pieces = c.bare_pieces = 0;
        }
    } tm{ ctx, t_start };

#ifdef XDNA_HAVE_NPU
    // after an NPU failure the GPU runs the rest of the prompt, which llama.cpp keeps sending here
    if (g_npu_broken && xdna_gpu_compute(ctx, cgraph)) {
        ctx.gpu_pieces++;
        return GGML_STATUS_SUCCESS;
    }

    // Block claiming runs every matmul on the NPU. The policy only hands us
    // matmuls it can run, but a caller computing a graph directly (a test)
    // can pass others; a graph of only such matmuls goes op by op below,
    // which falls back to the CPU. A graph with small ops in it is a claimed
    // block: it runs here, on the GPU if the NPU has failed. So does one a
    // test hands over in host-only mode, which is how the host's small ops
    // are tested without an NPU.
    bool npu_takes_all = true, small_ops = false, multiplies = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT) {
            small_ops |= n->op != GGML_OP_NONE && n->op != GGML_OP_RESHAPE;
            continue;
        }
        multiplies = true;
        const ggml_tensor * w = n->src[0];
        npu_takes_all &= w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                         xdna_npu_has_shape(w->ne[0], w->ne[1]);
    }
    // Small ops are claimed behind a multiply, but llama.cpp's scheduler can
    // cut them off from it (an op we don't take falls between them in the
    // graph's order). Such a piece has nothing for the NPU, and on the host
    // it would cost a read and a write of every tensor for a few cheap ops:
    // the GPU runs it where the tensors are.
    if (xdna_blocks() && small_ops && !multiplies) {
        bool gpu_takes = true;  // (an op only the host has stays here)
        for (int i = 0; i < cgraph->n_nodes && gpu_takes; i++)
            gpu_takes = ggml_backend_supports_op(ctx.vk, cgraph->nodes[i]);
        if (gpu_takes && xdna_gpu_compute(ctx, cgraph)) {
            ctx.bare_pieces++;
            return GGML_STATUS_SUCCESS;
        }
    }
    if ((xdna_blocks() && npu_takes_all) || small_ops) {
        xdna_npu * npu = npu_takes_all ? ctx.get_npu() : nullptr;
        std::unordered_set<const ggml_tensor *> in_piece;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            in_piece.insert(cgraph->nodes[i]);
            ctx.matmuls += cgraph->nodes[i]->op == GGML_OP_MUL_MAT;
        }
        auto since = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        };
        xdna_io io;
        io.read = [&](const ggml_tensor * t, void * dst, size_t bytes) {
            const auto t0 = std::chrono::steady_clock::now();
            ctx.read(t, dst, bytes);
            ctx.read_ms += since(t0);
        };
        io.write = [&](ggml_tensor * t, const void * src, size_t bytes) {
            const auto t0 = std::chrono::steady_clock::now();
            ctx.write(t, src, bytes);
            ctx.write_ms += since(t0);
        };
        const auto finish_writes = [&] {
            const auto t0 = std::chrono::steady_clock::now();
            ctx.finish_writes();
            ctx.write_ms += since(t0);
        };
        io.param = [&](const ggml_tensor * t) { return ctx.param(t); };
        io.alloc = [&](size_t n) { return ctx.pool_alloc(n); };
        io.needed_outside = [&](const ggml_tensor * t, const ggml_cgraph *) { return xdna_read_outside(t, in_piece); };
        io.ensure_weight = [&](const ggml_tensor * w, std::string & err) {
            if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                err = std::string(w->name) + " is not in a weight buffer";
                return false;
            }
            const xdna_npu::wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
            if (npu->has_weight(key)) return true;
            ctx.late_copies++;
            ctx.scratch_weight.resize(ggml_nbytes(w));
            ctx.read(w, ctx.scratch_weight.data());
            return npu->add_weight(key, ctx.scratch_weight.data(), err, xdna_weight_quant(w));
        };
        io.ensure_fused = [&](const ggml_tensor * gw, const ggml_tensor * uw, std::string & err) {
            for (const ggml_tensor * w : { gw, uw })
                if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    err = std::string(w->name) + " is not in a weight buffer";
                    return false;
                }
            if (npu->has_weight(xdna_npu::fused_key(xdna_npu::gate_key(gw, uw)))) return true;
            ctx.late_copies++;
            std::vector<uint8_t> g, u;
            return ctx.add_fused(gw, uw, g, u, err);
        };
        io.weight_bytes = [&](const ggml_tensor * w) -> const void * { return ctx.weight(w).data(); };
        std::string err;
        bool ok = false;
        // The split (GGML_XDNA_SPLIT): the NPU's rows come first, the GPU
        // runs the rest while the NPU works. The GPU's share reads and writes
        // the tensors' own memory, so it runs only between the piece's reads
        // and its writes.
        int64_t T = 0, npu_rows = 0;
        for (int i = 0; i < cgraph->n_nodes && !T; i++)
            if (!xdna_is_view_op(cgraph->nodes[i]->op)) T = cgraph->nodes[i]->ne[xdna_exec_row_dim(cgraph->nodes[i])];
        xdna_rows_graph gpu_rows;
        std::thread gpu_thread;
        bool gpu_started = false, gpu_ok = true;
        double gpu_ms = 0;
        if (npu && xdna_split_percent() > 0 && T >= 1024) {
            npu_rows = std::min(T - 512, std::max<int64_t>(512, (T * xdna_split_percent() / 100 + 256) / 512 * 512));
            std::string why;
            if (!xdna_rows_view(ctx, cgraph, T, npu_rows, T, gpu_rows, why)) {
                if (xdna_trace()) fprintf(stderr, "xdna split: a piece not shared (%s)\n", why.empty() ? "no memory" : why.c_str());
                npu_rows = 0;
            }
        }
        if (npu_rows) {
            io.inputs_read = [&] {
                gpu_started = true;
                gpu_thread = std::thread([&] {
                    const auto t0 = std::chrono::steady_clock::now();
                    gpu_ok = ggml_backend_graph_compute(ctx.vk, gpu_rows.g) == GGML_STATUS_SUCCESS;
                    gpu_ms = since(t0);
                });
            };
            io.before_write = [&](std::string & e) {
                const auto t0 = std::chrono::steady_clock::now();
                if (gpu_thread.joinable()) gpu_thread.join();
                // Vulkan's compute may return with work still queued: wait for the rows themselves
                ggml_backend_synchronize(ctx.vk);
                ctx.split_wait_ms += since(t0);
                ctx.split_gpu_ms += gpu_ms;
                ctx.split_pieces++;
                if (!gpu_ok) e = "the GPU's share of a piece failed";
                return gpu_ok;
            };
        }
        if (npu) {
            ok = xdna_exec_piece(cgraph, npu, npu->pool(), io, xdna_streams(), err, npu_rows);
            finish_writes();
            ctx.pool_release();
            if (gpu_thread.joinable()) {  // the NPU's side failed before its writes
                gpu_thread.join();
                ggml_backend_synchronize(ctx.vk);
            }
            if (!ok && gpu_ok) xdna_npu_failed(err);
        }
        io.inputs_read = nullptr;
        io.before_write = nullptr;
        if (!gpu_ok) {
            GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
            return GGML_STATUS_FAILED;
        }
        // A failed piece wrote nothing back: the GPU runs it again, or the
        // host if the GPU can't. After a split whose GPU share ran, the GPU
        // has done its rows (in place, so they can't be run again): only the
        // NPU's are left.
        if (!gpu_started) npu_rows = 0;
        if (!ok && g_npu_broken) {
            if (npu_rows) {
                xdna_rows_graph rest;
                std::string why;
                ok = xdna_rows_view(ctx, cgraph, T, 0, npu_rows, rest, why) &&
                     ggml_backend_graph_compute(ctx.vk, rest.g) == GGML_STATUS_SUCCESS;
                ctx.gpu_pieces++;
            } else {
                ok = xdna_gpu_compute(ctx, cgraph);
                ctx.gpu_pieces++;
            }
        }
        if (!ok) {
            ok = xdna_exec_piece(cgraph, nullptr, ctx.host_threads(), io, 1, err, npu_rows);
            finish_writes();
            ctx.pool_release();
        }
        if (!ok) {
            GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }
#endif

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_MUL_MAT:
                if (xdna_mul_mat(ctx, node)) break;
                if (!g_npu_broken) return GGML_STATUS_FAILED;
                // the NPU failed just now: the whole graph on the GPU, or this matmul on the CPU if the GPU can't
                if (xdna_gpu_compute(ctx, cgraph)) return GGML_STATUS_SUCCESS;
                if (!xdna_mul_mat(ctx, node)) return GGML_STATUS_FAILED;
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// llama.cpp shows each backend the pieces it plans for it: at load, when it
// plans the first prompt, and again whenever a prompt's pieces change. The
// first time a piece's weights appear, their NPU copies are built before the
// load callback returns. GGML_XDNA_COPY_AT_LOAD=0 leaves them to the first
// prompt.
static void ggml_backend_xdna_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph,
                                             ggml_backend_graph_optimize_params * params) {
#ifdef XDNA_HAVE_NPU
    static const bool at_load = env_int("GGML_XDNA_COPY_AT_LOAD", 1) != 0;
    if (!at_load || !xdna_blocks()) return;
    xdna_context & ctx = *(xdna_context *) backend->context;
    std::unordered_set<const ggml_tensor *> in_piece;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        in_piece.insert(n);
        if (n->op != GGML_OP_MUL_MAT) continue;
        // As graph_compute decides: the NPU runs a piece only if it takes
        // every matmul in it. llama.cpp's memory-fitting trial loads no
        // weight data, so its pieces build nothing.
        const ggml_tensor * w = n->src[0];
        if (!w->data || !w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !xdna_npu_has_shape(w->ne[0], w->ne[1]))
            return;
    }
    xdna_io io;
    io.needed_outside = [&](const ggml_tensor * t, const ggml_cgraph *) { return xdna_read_outside(t, in_piece); };
    std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> ws;
    xdna_exec_weights(cgraph, io, ws);
    if (ws.empty() || !ctx.get_npu()) return;
    std::vector<xdna_context::build_job> jobs;
    for (const auto & w : ws) jobs.push_back({ w.first, w.second });
    ctx.queue_copies(jobs);
    ctx.finish_copies();
#endif
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(params);
}

static const ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name             = */ ggml_backend_xdna_get_name,
    /* .free                 = */ ggml_backend_xdna_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .set_tensor_2d_async  = */ NULL,
    /* .get_tensor_2d_async  = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ NULL,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ ggml_backend_xdna_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ ggml_backend_xdna_graph_optimize,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x9d, 0x3a, 0x1c, 0x74, 0x2e, 0x58, 0x41, 0xb6,
                              0x8f, 0x0d, 0xc5, 0x93, 0x6a, 0x27, 0xe1, 0x40 };
    return &guid;
}

//
// device
//

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return GGML_XDNA_DEVICE_NAME;
    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    return XDNA_DESCRIPTION;
    GGML_UNUSED(dev);
}

// No free memory, but some in total: llama.cpp then gives this device no
// layers (all weights go to Vulkan). Zero for both would make it fall back to
// counting system memory, and hand us layers.
static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 1;
    GGML_UNUSED(dev);
}

// An integrated GPU, not an accelerator: llama.cpp ranks accelerators below
// every GPU, where the scheduler would never hand us anything.
static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    props->device_id   = NULL;
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);

    // no async and no events: llama.cpp then keeps pipeline parallelism off,
    // and hands over between us and Vulkan with plain synchronizes
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init(ggml_backend_dev_t dev, const char * params) {
    const xdna_vk & vk = xdna_vulkan();
    if (!vk.dev) return nullptr;
    auto * ctx = new xdna_context;
    ctx->vk = ggml_backend_dev_init(vk.dev, nullptr);
    if (!ctx->vk) {
        GGML_LOG_ERROR("%s: cannot start a backend on %s\n", __func__, ggml_backend_dev_name(vk.dev));
        delete ctx;
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_budget.mu);
        g_budget.contexts++;
    }
    return new ggml_backend{
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    GGML_UNUSED(params);
}

// The weight side of a matmul we can take: a plain 2D weight tensor (not a
// view, not computed), of a type ggml can unpack. Keys and values from the
// cache are views, so attention's own matmuls never match.
static bool xdna_is_weight(const ggml_tensor * w) {
    if (w->op != GGML_OP_NONE || w->view_src != nullptr) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w)) return false;
    if (w->type == GGML_TYPE_F32) return true;
    const ggml_type_traits * tr = ggml_get_type_traits(w->type);
    return tr && tr->to_float != nullptr;
}

// Who reads what, as the scheduler shows us. It asks supports_op about every
// op it places: the ones we take, and every one it assigns to Vulkan (we rank
// above Vulkan and share its buffer type, so its upgrade pass checks us
// first). Each question carries the op's inputs, so this records every reader
// of every tensor. Block claiming writes a result back to Vulkan only when a
// reader outside the piece exists.
// Readers from earlier graphs whose tensors ggml reused can linger; that only
// ever adds a write, never skips one.
static bool xdna_trace() {
    static bool v = env_int("GGML_XDNA_TRACE", 0) != 0;
    return v;
}
static std::mutex g_read_mu;
static std::unordered_map<const ggml_tensor *, std::unordered_set<const ggml_tensor *>> g_readers;

static void xdna_note_readers(const ggml_tensor * op) {
    std::lock_guard<std::mutex> lk(g_read_mu);
    for (int j = 0; j < GGML_MAX_SRC; j++)
        if (const ggml_tensor * s = op->src[j]) {
            g_readers[s].insert(op);
            for (const ggml_tensor * v = s->view_src; v; v = v->view_src) g_readers[v].insert(op);
        }
}

// Whether something outside `in_piece` reads `t`, or might: a graph output, a
// tensor never seen, or one with a reader elsewhere.
static bool xdna_read_outside(const ggml_tensor * t, const std::unordered_set<const ggml_tensor *> & in_piece) {
    static const bool all = env_int("GGML_XDNA_WRITE_ALL", 0) != 0;  // diagnostic: write every result back
    if (all || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) return true;
    std::lock_guard<std::mutex> lk(g_read_mu);
    auto it = g_readers.find(t);
    if (it == g_readers.end()) return true;
    for (const ggml_tensor * r : it->second)
        if (!in_piece.count(r)) return true;
    return false;
}

static void xdna_trace_piece(const ggml_cgraph * cgraph) {
    static int64_t pieces = 0, results = 0, unseen = 0;
    pieces++;
    std::unordered_set<const ggml_tensor *> in_piece;
    for (int i = 0; i < cgraph->n_nodes; i++) in_piece.insert(cgraph->nodes[i]);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT) continue;
        results++;
        std::lock_guard<std::mutex> lk(g_read_mu);
        if (!g_readers.count(n)) {
            unseen++;
            fprintf(stderr, "xdna trace: %s (%s) has no recorded reader\n", n->name, ggml_op_name(n->op));
        }
    }
    if (pieces % 100 == 0)
        fprintf(stderr, "xdna trace: %lld pieces, %lld matmul results, %lld with no recorded reader\n", (long long) pieces,
                (long long) results, (long long) unseen);
}

// Block claiming: with the NPU configured, also take the small
// ops next to our matmuls. GGML_XDNA_BLOCKS=0 claims matmuls only.
static bool xdna_blocks() {
    static bool v = env_int("GGML_XDNA_BLOCKS", 1) != 0;
#ifdef XDNA_HAVE_NPU
    return v && xdna_run_mode() == xdna_mode::npu;
#else
    return false;
#endif
}

// Rows are prompt tokens, along a tensor's last dimension in use.
static int64_t xdna_tokens(const ggml_tensor * t) {
    return t->ne[3] > 1 ? t->ne[3] : t->ne[2] > 1 ? t->ne[2] : t->ne[1];
}

static bool xdna_supports_op_policy(const ggml_tensor * op, int depth = 0);

// Whether a tensor may live in host memory once allocated: a graph input, an
// embedding lookup (llama.cpp keeps the token embeddings on the CPU), or one
// already in a host buffer.
static bool xdna_maybe_host(const ggml_tensor * t) {
    for (; t; t = t->view_src) {
        if ((t->flags & GGML_TENSOR_FLAG_INPUT) || t->op == GGML_OP_GET_ROWS) return true;
        if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) return true;
    }
    return false;
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (xdna_blocks() || xdna_trace()) xdna_note_readers(op);
    xdna_note_model(op);
    return xdna_supports_op_policy(op);
    GGML_UNUSED(dev);
}

static bool xdna_claim_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0], * src1 = op->src[1];
    if (xdna_op_batch_size(op) < xdna_min_batch()) return false;
    if (!xdna_is_weight(src0)) return false;
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src1) || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
#ifdef XDNA_HAVE_NPU
    // on the NPU, only sizes the kernel takes
    if (xdna_run_mode() == xdna_mode::npu && !xdna_npu_has_shape(src0->ne[0], src0->ne[1])) return false;
#endif
    const int64_t mflop = 2 * src0->ne[0] * src0->ne[1] * xdna_op_batch_size(op) / 1000000;
    return mflop >= xdna_min_mflop() && xdna_budget_takes(src0);
}

// The whole policy. Anything accepted here is taken from Vulkan, so it has to
// be strict. It must also cope with llama.cpp's load-time probe, where the
// weight has a placeholder buffer and no data: nothing here reads data.
//
// A small op is claimed only when the executor implements it, it covers a big
// batch, and one of its inputs comes from an op we'd claim: that grows pieces
// out from our matmuls (norm -> q/k/v; o -> add -> norm -> gate/up -> SiLU ->
// down -> add) without pulling in unrelated work, which would split Vulkan's
// graph into pieces for nothing.
static bool xdna_supports_op_policy(const ggml_tensor * op, int depth) {
    if (!xdna_vulkan().dev || g_npu_broken) return false;
    if (op->op == GGML_OP_MUL_MAT) return xdna_claim_mul_mat(op);
#ifdef XDNA_HAVE_NPU
    if (!xdna_blocks() || depth > 8) return false;
    if (op->op != GGML_OP_RESHAPE && xdna_tokens(op) < xdna_min_batch()) return false;
    if (!xdna_exec_supports(op)) return false;
    {
        // GGML_XDNA_BLOCK_OPS: a comma list of the small ops block claiming
        // may take (ggml's op names, e.g. "ADD,RMS_NORM"), for tracking down
        // a wrong result op by op. Unset: all of them.
        static const std::string allow = [] {
            const char * s = xdna_env("GGML_XDNA_BLOCK_OPS");
            return s ? "," + std::string(s) + "," : std::string();
        }();
        if (!allow.empty() && allow.find("," + std::string(ggml_op_name(op->op)) + ",") == std::string::npos) return false;
    }
    // a scale or bias row, or rotary factors: a plain leaf, kept once
    if (const ggml_tensor * w = xdna_exec_param(op))
        if (w->op != GGML_OP_NONE || w->view_src) return false;
    // ggml's allocator may put an op's result in the memory of an input the
    // same shape (in place), without checking that the input is in the same
    // kind of memory. An input in host memory (the token embeddings, or a
    // graph input) would then take our result there, and Vulkan, reading it
    // next, crashes. LFM2's first residual add is one: add(embeddings, x).
    for (int j = 0; j < GGML_MAX_SRC; j++)
        if (const ggml_tensor * s = op->src[j])
            if (xdna_maybe_host(s) && ggml_are_same_shape(op, s)) return false;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        const ggml_tensor * s = op->src[j];
        if (s && s->op != GGML_OP_NONE && xdna_supports_op_policy(s, depth + 1)) return true;
    }
#endif
    return false;
    GGML_UNUSED(depth);
}

// Vulkan's buffers, and host memory too (the CPU's graph inputs: the token
// embeddings, positions). Reading host tensors directly also keeps our pieces
// free of scheduler-copied inputs, and that matters for correctness: before a
// piece with no inputs the scheduler waits for the previous backend (Vulkan),
// but before a piece with inputs it waits only for the backend each input
// came from. A piece whose one copied input came from the CPU could then
// start reading Vulkan results before Vulkan had finished them.
static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    const xdna_vk & vk = xdna_vulkan();
    return (vk.buft != nullptr && buft == vk.buft) || ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

// Vulkan's own buffer type: sharing it is what lets the scheduler hand ops
// between us and Vulkan with no copies.
static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    const xdna_vk & vk = xdna_vulkan();
    if (!vk.buft) GGML_ABORT(GGML_XDNA_DEVICE_NAME " needs the Vulkan backend loaded");
    return vk.buft;
    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    // deliberately no offload_op: that path copies the weights per op
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// reg
//

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_XDNA_NAME;
    GGML_UNUSED(reg);
}

// No device when the NPU can't run the backend (and host-only mode is off):
// llama.cpp then never sees XDNA0, and runs as if the backend weren't there.
static size_t ggml_backend_xdna_reg_device_count(ggml_backend_reg_t reg) {
    return xdna_run_mode() == xdna_mode::off ? 0 : 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_device_get(ggml_backend_reg_t reg, size_t index);

static const ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ ggml_backend_xdna_reg(),
        /* .context = */ NULL,
    };
    return &dev;
    GGML_UNUSED(reg);
}

static int ggml_backend_xdna_score(void) {
    return 1;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_xdna_score)
