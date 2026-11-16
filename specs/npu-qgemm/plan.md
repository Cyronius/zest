# NPU GEMM on quantized weights: plan

Status: planned 2026-11-15. Phase 0 (tools and baseline) is done; the
baseline reordered the work (see Baseline findings).

## Why

Zest exists to put the NPU to work on prefill. On Strix Halo it currently
loses to the GPU, and other NPU engines show the cause is zest's design, not
the NPU.

| Qwen3.8-27B prefill on Strix Halo | tok/s | conditions |
|---|---|---|
| artlite, NPU only, Q4_K_M, 1,024 tokens | 455 | published, quiet machine |
| artlite, NPU only, Q4_K_M, 1,024 / 2,769 tokens | 355 / 340 | measured here 2026-11-15, machine busy |
| zest, NPU + GPU, UD-IQ3_S, 2,769 tokens | 231-255 | 2026-11-13, quiet |
| llama.cpp b10944 Vulkan, GPU only, UD-IQ3_S | 258-300 | 2026-11-13, quiet |

The 2026-11-13 trace and the code point at three costs, in this order of
attack:

1. **Weights.** Every weight gets a private BFP16 copy (9 bits a value),
   built on the host at load (`xdna_npu::add_packed`, `src/xdna-npu.cpp`).
   That makes about 1.1 GB per billion parameters, capped at 20 GB, so only
   49 of the 27B's 64 layers fit on the NPU. The kernel streams those copies
   from DRAM once per 512 tokens: roughly 400 MB of weights per 2,048-token
   gate/up GEMM. That is twice the bytes of the GGUF weights.
2. **Host passes.** The small ops between matmuls run on the CPU, which
   decodes every NPU output and encodes every input: 1.3-1.5 s a prompt.
3. **Submit and wait.** The host starts each NPU job and waits for it:
   1.2-1.3 s a prompt. XRT sits in that path.

This plan covers cost 1: **a GEMM whose weight operand is the quantized
weights, unpacked on the NPU.** Costs 2 and 3 are the next steps (see Later).

## Reference designs

- **artlite** (`artlite-project/artlite`, LGPL-2.1; zest is MIT, so we
  follow ideas and write our own code). Per array column, one core unpacks
  each 64-value k-tile of weights into BFP16 and broadcasts it to the
  column's other cores, which only multiply (`src/artlite/gemm/gemm_qd.py`).
  It unpacks about 5 weights a cycle; IQ3_S takes about 12 cycles per row
  per k-tile, using the matrix unit to compute table offsets and 16-wide
  table lookups. Weights are re-laid out at load into NPU tiles at their
  quantized size. Q4_K scales are rounded to bf16, so it is not bit-exact
  with ggml.
- **yet-another-halo-engine** (`hisamiii/yet-another-halo-engine`). The NPU
  reads raw GGUF rows in place and decodes them on its own cores. This is
  the later, zero-copy refinement here.
- Zest's own 11-13 prototype (`kernels/bfp16_gemm/aie_kernels/aie2p/iq3s_to_bfp.cc`):
  a scalar IQ3_S decoder on one core. It is bit-exact but runs at about one
  weight every 60 cycles, about 300x short of what the GEMM needs.

## Design

### Array layout

Today's kernel (`kernels/bfp16_gemm/designs/whole_array_bfp_rtp.py`) uses all
4 rows x 8 columns as multiply cores. The new design `whole_array_bfpq.py`:

- **Row 2 of each column unpacks.** Quantized k-tiles go shim → memtile →
  row-2 core, which writes BFP16 k-tiles into an ObjectFifo **broadcast to
  rows 3-5**.
- **Rows 3-5 multiply.** They run the existing `mm_f32_first`,
  `mm_f32_mid` and `out_epilogue` from `mm_bfp_acc.cc` unchanged, so output
  modes 0 (fp32), 1 (bf16) and 2 (SiLU(gate)·up) keep working.
- A is split over the 3 multiply rows and C joined from them. Rows per call
  become a multiple of 3·m (m=128 → 384); the call size (1,152, or 768 /
  1,536) is chosen by measurement.
- Runtime parameters: K chunks and output mode as today, plus a **format id**.
  One xclbin then serves several weight types in **one hardware context**,
  because a context switch costs about 2.4 ms against about 0.1 ms per
  dispatch. Formats are added while the unpacking core's program memory
  allows.
- Throughput budget: three multiply cores finish a 128×64×64 tile each in
  about 1,000 cycles, so the unpacking core must turn out a 64×64 k-tile
  (4,096 weights) in about 1,000 cycles: **≥ 4 weights per cycle**.

### Weight layout in DRAM

`src/npu/qtile_pack.{h,cpp}` re-lays each GGUF tensor once at load into
k-tile order: per column, per 64-row group, per 256-value super-block, with
the scales grouped ahead of the values. Same byte count as the GGUF data,
byte moves only, no float math. The weight-copy budget then counts real
sizes, about 16 GB for the 27B Q4_K_M, so every layer fits.

### Numerics: bit-exact with today's copies

Today a weight becomes `bfp16_encode_block(ggml to_float(row))`:

- ggml dequantizes Q4_K in fp32 as `d·sc·q − dmin·m`. The products are
  exact; the subtraction rounds once.
- The encoder takes the block's largest fp32 exponent `e` and rounds each
  value to `rne(x·2^(133−e))`, saturated to [−128, 127].

The AIE2P vector units multiply **fp16 × fp16 natively into fp32
accumulators** (`aie_api/detail/aie2p/mul_acc32_fp.hpp`), and Q4_K's scales
`d` and `dmin` are fp16. So the unpacker computes

    w = d·(1024 + sc·q) − d·1024 − dmin·m

- `sc·q ≤ 945`, so `1024 + sc·q` is an exact fp16 value, built with an
  OR of the bits `0x6400 | sc·q`.
- Each product is exact in fp32, and the first two terms cancel exactly.
  The only rounding is the final subtraction, the same place ggml rounds.

If the accumulator rounds that sum once, round-to-nearest-even, and the
hardware's accumulator → bfp16ebs8 conversion matches our encoder (largest
exponent, round-to-nearest-even, saturate), the NPU's BFP16 weights equal
today's bit for bit. Phase 1 tests both conditions first. If either fails,
the fallback is to define the NPU's arithmetic as the reference: a host model
of it, a bit-exact test against that model, and an accuracy check against
ggml (max error per block, KL divergence on the model).

Q6_K (`d` fp16, int8 scales, 6-bit values) fits the same scheme. The IQ
formats need table lookups: artlite's 16-wide lookups are the model.

## Baseline findings (2026-11-15) and the revised order

Measured on the test machine with other work running (`tools/bench-npu-gemm`,
artlite-probe, a one-core cycle probe of `mm_f32_mid`):

| current BFP16 kernel | result |
|---|---|
| multiply kernel alone, one core, no data movement | 2,579 cycles per 64-value K chunk, ideal 1,024: **40% of the MMUL peak** |
| whole GEMM at the 27B's shapes, M = 1,024 | **12.1 TFLOPS, ~21% of the ~58 TFLOPS peak** |
| DRAM traffic | 29 GB/s, against 46.5 GB/s available (57 quiet); ≤ 3.6 GB/s on a channel, against 11.4 |

The kernel is **compute-bound, not bandwidth-bound**: the multiply loop pays
about 48 cycles per 32 MMULs to reload, store and restart its accumulators,
and the whole GEMM loses about half of the rest to hand-offs, waits, the
epilogue and dispatch. A dedicated unpacking row would leave 3 of 4 rows
multiplying, so with today's multiply loop the quantized GEMM would run at
about ¾ of today's speed. The order is therefore:

1. **Multiply loop** (`mm_bfp_acc.cc`): toward one MMUL per cycle, measured
   with the cycle probe. This speeds up the existing BFP16 path on its own.
2. **Whole-GEMM losses:** per-core timing inside the 32-core GEMM, then fix
   the stalls.
3. **Quantized weights unpacked on the NPU** (Phases 1-3 below).

### Step 1 results (2026-11-15)

- **Flat multiply loop, shipped.** `mm_paired` in `mm_bfp_acc.cc` now runs
  the 2×2 blocks in one loop: persistent A/B streams moved by 2-D seeks,
  K steps fully unrolled, separate `__restrict` C pointers. Bit-identical.
  One core: 2,579 → 2,496 cycles per chunk. Whole GEMM at the 27B's shapes:
  **12.1 → 13.3 TFLOPS (+9%)**.
- **The limit is C, not the schedule.** Every 64-value K chunk reloads the
  fp32 running sums into accumulators, and a 2×2 block's four accumulators
  fill the core's ~5 accumulator registers, so the next block's sums can't be
  loaded ahead (the compiler even spills one). Starting chains from zero and
  adding C afterwards spills more (31%). bf16 sums would reach artlite's
  ~97%, but in a host simulation they add ~1.06% error on top of BFP16's
  0.97% at K = 5,120: ruled out.
- **Longer K chunks help the core, not yet the GEMM.** One core reaches 59%
  of the peak at m = 64, k = 128, n = 64 (72% at k = 256, which doesn't fit
  local memory double-buffered). In the whole GEMM that tile ran at **10.2
  TFLOPS**: cores busy ~30% of the time, waiting on data, while DRAM moved 35
  of 46.5 GB/s and no channel passed 3.5 GB/s. Its smaller row blocks double
  the transfers and the C-drain waits in the runtime sequence. It goes back on
  the shelf until step 2 fixes data movement; `build-linux.sh` builds it with
  `TILE_M=64 TILE_K=128` (RTP words then at `0xe000`/`0xe004`, shim placement
  unchanged).

Step 2 is now the bigger lever: at the shipped tile the cores are busy about
55% of the GEMM's time, at the 64×128×64 tile about 30%.

### Which weights get which copy

BFP16 copies stay wherever they fit; quantized copies are for the rest.
`GGML_XDNA_WEIGHTS` (and a `zest` flag beside `--memory-gb`):

- `auto` (default): BFP16 copies within the memory budget, quantized copies
  for the weights past it, which today stay on the GPU (15 of the 27B's 64
  layers). Small models never reach the fallback and are unchanged.
- `bfp16`: today's behaviour.
- `quant`: quantized copies for every supported weight. Less memory (the
  27B: ~16 GB instead of ~30) and faster loading, at the speed the
  measurements give.

Which kernel is faster can flip once the multiply loop is fixed: at 2.5× its
speed the BFP16 kernel would need ~72 GB/s, past the NPU's port, while the
quantized kernel moves half the weight bytes. `auto`'s rule is set by
measurement then.

## Phases

### Phase 0: tools and baseline

Done: tools, the rebuilt kernel passing the NPU tests, and the baseline above.

Done 2026-11-15: `third_party/ironenv` (mlir_aie 1.4.3, the version the
shipped xclbin was built with, plus its pinned Peano
`llvm-aie 22.0.0.2026090701`), and `third_party/mlir-aie` at v1.4.3 (both
gitignored).

To do:
- Install XRT (needs root): `sudo pacman -S xrt xrt-plugin-amdxdna`.
  mlir-aie's `env_setup.sh` refuses to run without `xrt-smi`, and zest
  dlopens `libxrt_coreutil.so.2`.
- Rebuild the shipped kernel (`kernels/bfp16_gemm/build-linux.sh 1024 5120 10240`)
  and run `ctest -L npu`, `test-npu-wide` and `test-mul-mat`.
- **Baseline:** time the current kernel per GEMM at the 27B's shapes
  (`GGML_XDNA_TRACE`, `build/npu`, `bench-hybrid-projection`) and compare
  with the bytes it moves.

**Stop point:** if the current kernel is compute-bound, not bandwidth-bound,
the new GEMM mostly buys memory and layer coverage rather than speed per
GEMM. Report before Phase 2.

### Phase 1: Q4_K unpacker on one core

- `kernels/bfp16_gemm/aie_kernels/aie2p/dq_q4k.cc`: one 64-row × 64-value
  k-tile of re-laid Q4_K → the BFP16 tile order `mm_bfp_acc.cc` reads. It
  is vectorized: nibble unpack, the fp16 trick above, then native
  conversion to bfp16ebs8.
- `kernels/bfp16_gemm/designs/dq_check.py`: a single-core harness like
  `designs/iq3s_to_bfp.py`. It reads real rows of
  `Qwen3.8-27B-Q4_K_M.gguf`, compares the output with the host packer, and
  reads the core's cycle counter.
- First experiment: the rounding questions above, on a handful of blocks
  that include ties.
- Then Q6_K (`dq_q6k.cc`), which Q4_K_M uses for attn_v and half of
  ffn_down.

**Exit:** bit-exact (or the fallback, documented), ≥ 4 weights per cycle.

**Stop point:** below about 2 weights per cycle, the dedicated-unpacking-core
design needs rethinking before Phase 2.

### Phase 2: whole-array GEMM

- `designs/whole_array_bfpq.py` as described above; built by a
  `build-linux.sh` variant into `prebuilt/bfpq_gemm.xclbin`.
- `src/npu/bfpq_insts.{h,cpp}`: host-made instruction streams, following
  `src/npu/bfp16_insts.cpp`, with this design's shim placement.
  Checked against the toolchain's `insts.bin` for a few shapes (as
  XDNA-INSTS-GEN does today).
- Decide one xclbin with both designs (preferred), or separate contexts,
  from the measured switch cost.

**Exit:** for Q4_K and Q6_K weights at the 27B's shapes and output modes
0/1/2, the output equals the BFP16 kernel's bit for bit (same BFP16 weights,
same multiply kernel); per-GEMM time measured against the BFP16 kernel.

### Phase 3: backend

- `src/xdna-npu.{h,cpp}`:
  - `add_qweight(key, raw)` packs via `qtile_pack` into an NPU buffer at
    quantized size;
  - `mul_mat` / `begin` / `submit` use the quantized kernel when the weight
    has one;
  - a quantized `add_fused` for gate/up.
- `src/npu/npu_bfp16.{h,cpp}`: the second kernel in the same context.
- Weight types without an unpacker keep BFP16 copies. The memory budget
  counts each copy at its real size.
- Policy (`src/ggml-zest.cpp`) is unchanged apart from capacity.
- Spec updates in `specs/xdna-backend/spec.md` (XDNA-BFP16-PACK,
  XDNA-MEMORY-BUDGET, XDNA-INSTS-GEN, XDNA-MUL-MAT-AGREES); docs in
  `docs/technical-guide.md`.

### Phase 4: measure

On the busy machine, with paired runs in one `llama-bench` call:

    zest llama-bench -m Qwen3.8-27B-Q4_K_M.gguf -dev Vulkan0,XDNA0/Vulkan0 -p 1024,2769 -n 0

- before and after, with per-GEMM NPU ms from `GGML_XDNA_TRACE`;
- layers on the NPU (expect 64/64), load time, and memory;
- `tools/linux-npu-check.sh`'s KL check (mean KL < 0.01 against the GPU)
  and 24-token reply match.

Results go in `docs/results/halo-npu-qgemm-<date>.json`. Artlite's
340-355 tok/s under similar load is the NPU-only reference.

## Later (separate plans)

In order of the trace's costs:

1. **Host passes (1.3-1.5 s a prompt).** Have Vulkan write activations
   straight in the NPU's BFP16 layout and read the NPU's outputs, so the
   CPU drops out of the data path (yahe does this). This needs Vulkan
   shaders for encode and decode and a shared, exportable buffer.
2. **Submit and wait (1.2-1.3 s a prompt).** The NPU waits on ready flags
   the GPU writes, with no host relay (yahe, HRX patch 0015). Drop XRT and
   talk to `amdxdna` directly, as artlite, hipfire and yahe do; this also
   removes the XRT install step for users.
3. **More formats:** IQ3_S, IQ3_XXS and IQ4_XS for unsloth's UD files; IQ1_S
   and IQ2 if the files people use need them.
4. **Raw GGUF rows in place** (no re-laid copy at all).
5. **GPU/NPU split of single projections** by output columns, once the
   handoff is cheap. yahe gets 1.18x over its GPU alone this way.

## Risks

- **Rounding:** if the accumulator or the hardware conversion doesn't round
  as assumed, the result is the documented fallback, not bit-exact.
- **Program memory:** each format's unpacker competes for the unpacking
  core's program memory; several formats in one xclbin may not fit.
- **Rows per call:** the 3-multiply-row shape changes the call size and pads
  some prompts; pick by measurement.
- **Bandwidth:** the NPU reads about 57 GB/s of the bus on Halo, shared with
  the busy machine. Gains measured under load will be noisy; always pair
  runs.
- **Toolchain:** mlir_aie 1.4.3 and its Peano pin are kept; a newer mlir_aie
  is a separate change.
