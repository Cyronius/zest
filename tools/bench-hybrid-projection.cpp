// Isolated output-column split of one real GGUF projection between Vulkan and
// XDNA. This measures the split without the host graph passes that currently
// surround NPU matmuls in Zest. It is an experiment, not the inference path.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct matrix {
    ggml_type type;
    int64_t k, n;
    std::vector<uint8_t> raw;
};

matrix load_matrix(const char * path, const char * name) {
    gguf_context * gguf = gguf_init_from_file(path, { true, nullptr });
    if (!gguf) throw std::runtime_error("cannot read GGUF metadata");
    int64_t id = -1;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++)
        if (strcmp(gguf_get_tensor_name(gguf, i), name) == 0) { id = i; break; }
    if (id < 0) { gguf_free(gguf); throw std::runtime_error("matrix not found in GGUF"); }
    const int64_t * ne = gguf_get_tensor_ne(gguf, id);
    matrix m{ gguf_get_tensor_type(gguf, id), ne[0], ne[1], {} };
    const size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id);
    m.raw.resize(gguf_get_tensor_size(gguf, id));
    gguf_free(gguf);
    FILE * f = fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open GGUF weights");
    if (fseek(f, (long) offset, SEEK_SET) != 0 || fread(m.raw.data(), 1, m.raw.size(), f) != m.raw.size()) {
        fclose(f);
        throw std::runtime_error("cannot read GGUF weight matrix");
    }
    fclose(f);
    if (m.n <= 0 || m.k <= 0 || m.raw.size() != ggml_row_size(m.type, m.k) * (size_t) m.n)
        throw std::runtime_error("bad GGUF matrix dimensions");
    return m;
}

class projection {
public:
    projection(ggml_backend_dev_t device, const matrix & mat, const std::vector<float> & x,
               int64_t first, int64_t count)
        : n_(count), m_((int64_t) x.size() / mat.k) {
        backend_ = ggml_backend_dev_init(device, nullptr);
        if (!backend_) throw std::runtime_error("backend init failed");
        ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
        ctx_ = ggml_init(params);
        if (!ctx_) throw std::runtime_error("ggml context init failed");
        w_ = ggml_new_tensor_2d(ctx_, mat.type, mat.k, count);
        x_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, mat.k, m_);
        y_ = ggml_mul_mat(ctx_, w_, x_);
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, ggml_backend_get_default_buffer_type(backend_));
        if (!buffer_) throw std::runtime_error("backend buffer allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const size_t row_bytes = ggml_row_size(mat.type, mat.k);
        ggml_backend_tensor_set(w_, mat.raw.data() + first * row_bytes, 0, count * row_bytes);
        ggml_backend_tensor_set(x_, x.data(), 0, x.size() * sizeof(float));
        graph_ = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph_, y_);
    }

    projection(const projection &) = delete;
    projection & operator=(const projection &) = delete;

    ~projection() {
        if (backend_) ggml_backend_synchronize(backend_);
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_) ggml_free(ctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    void compute() {
        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
        ggml_backend_synchronize(backend_);
    }

    std::vector<float> output() {
        std::vector<float> values((size_t) (m_ * n_));
        ggml_backend_tensor_get(y_, values.data(), 0, values.size() * sizeof(float));
        return values;
    }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor * w_ = nullptr, * x_ = nullptr, * y_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    int64_t n_, m_;
};

double millis(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

double run_one(projection & p, int repetitions) {
    double total = 0;
    for (int i = 0; i < repetitions; i++) {
        const auto start = clock_type::now();
        p.compute();
        total += millis(start);
    }
    return total / repetitions;
}

double run_pair(projection & gpu, projection & npu, int repetitions) {
    double total = 0;
    for (int i = 0; i < repetitions; i++) {
        const auto start = clock_type::now();
        std::exception_ptr ep_gpu, ep_npu;
        std::thread tg([&] { try { gpu.compute(); } catch (...) { ep_gpu = std::current_exception(); } });
        std::thread tn([&] { try { npu.compute(); } catch (...) { ep_npu = std::current_exception(); } });
        tg.join();
        tn.join();
        if (ep_gpu) std::rethrow_exception(ep_gpu);
        if (ep_npu) std::rethrow_exception(ep_npu);
        total += millis(start);
    }
    return total / repetitions;
}

void compare(const std::vector<float> & reference, const std::vector<float> & gpu,
             const std::vector<float> & npu, int64_t width) {
    const int64_t gpu_width = (int64_t) gpu.size() / (reference.size() / width);
    const int64_t npu_width = width - gpu_width;
    double sq_error = 0, sq_ref = 0;
    for (size_t i = 0; i < reference.size(); i++) {
        const size_t row = i / width, col = i % width;
        const float value = col < (size_t) gpu_width ? gpu[row * gpu_width + col]
                                                        : npu[row * npu_width + col - gpu_width];
        if (!std::isfinite(value)) throw std::runtime_error("hybrid result contains NaN or infinity");
        const double delta = (double) value - reference[i];
        sq_error += delta * delta;
        sq_ref += (double) reference[i] * reference[i];
    }
    const double nrmse = std::sqrt(sq_error / std::max(sq_ref, 1e-300));
    printf("  combined output NRMSE vs GPU: %.6f\n", nrmse);
    if (nrmse > 0.03) throw std::runtime_error("hybrid output differs too much from GPU");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "usage: %s model.gguf [tensor=blk.2.ffn_down.weight] [rows=512] [repetitions=5]\n", argv[0]);
        return 2;
    }
    try {
        const char * tensor = argc > 2 ? argv[2] : "blk.2.ffn_down.weight";
        const int64_t rows = argc > 3 ? atoll(argv[3]) : 512;
        const int reps = argc > 4 ? atoi(argv[4]) : 5;
        if (rows <= 0 || reps <= 0) throw std::runtime_error("rows and repetitions must be positive");
        ggml_backend_load_all();
        ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_name("Vulkan0");
        ggml_backend_dev_t npu_dev = ggml_backend_dev_by_name("XDNA0");
        if (!gpu_dev || !npu_dev) throw std::runtime_error("Vulkan0 or XDNA0 unavailable");
        matrix mat = load_matrix(argv[1], tensor);
        if (mat.n % 512 != 0 || mat.n < 1024) throw std::runtime_error("weight width must be >=1024 and divisible by 512");
        printf("%s: K=%lld N=%lld type=%s, %lld activation rows\n", tensor,
               (long long) mat.k, (long long) mat.n, ggml_type_name(mat.type), (long long) rows);
        std::vector<float> x((size_t) (rows * mat.k));
        for (size_t i = 0; i < x.size(); i++) x[i] = 0.25f * std::sin((double) i * 0.013);

        projection gpu_full(gpu_dev, mat, x, 0, mat.n);
        gpu_full.compute();
        const auto reference = gpu_full.output();
        const double initial_gpu_ms = run_one(gpu_full, reps);
        printf("GPU full initial: %.3f ms\n", initial_gpu_ms);

        for (int64_t npu_width : { mat.n / 4 / 512 * 512, mat.n / 2 / 512 * 512, mat.n * 3 / 4 / 512 * 512 }) {
            if (npu_width < 512 || npu_width >= mat.n) continue;
            const int64_t gpu_width = mat.n - npu_width;
            projection gpu(gpu_dev, mat, x, 0, gpu_width);
            projection npu(npu_dev, mat, x, gpu_width, npu_width);
            // Warm both paths and build the NPU's weight copy before timing.
            gpu.compute();
            npu.compute();
            compare(reference, gpu.output(), npu.output(), mat.n);
            const double gpu_part_ms = run_one(gpu, reps);
            const double npu_part_ms = run_one(npu, reps);
            const double pair_ms = run_pair(gpu, npu, reps);
            const double gpu_control_ms = run_one(gpu_full, reps);
            printf("GPU %lld + NPU %lld columns: GPU %.3f ms, NPU %.3f ms, overlap %.3f ms, "
                   "GPU control %.3f ms, %.3fx control\n",
                   (long long) gpu_width, (long long) npu_width, gpu_part_ms, npu_part_ms,
                   pair_ms, gpu_control_ms, gpu_control_ms / pair_ms);
        }
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "bench-hybrid-projection: %s\n", e.what());
        return 1;
    }
}
