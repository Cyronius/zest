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
  value to `rne(x·2^(133−e))` (with the NPU's +128 rule, below).

**Measured 2026-11-16, and how it is done instead.** XDNA2 has no fp16
vectors (`aie_api`'s type table lists float16 only for AIE-MLv2, and
`mul_acc32_fp` takes bf16 alone at this architecture), so the fp16 trick
first planned here does not exist. Everything is bf16 × bf16 into fp32
accumulators, which multiply exactly (8 × 8 bits) and add in IEEE fp32,
round to nearest even:

    D1 = d·sc, M1 = dmin·m        exact (17 bits): d = dh + dl as two bf16,
                                  sc and m as bf16 128 + v, then
                                  dh·(128+sc) − 128·dh + dl·(128+sc) − 128·dl
    D1 = p0 + p1 + p2             three bf16 pieces (round, subtract, twice)
    w  = −128·D1 + Σ pᵢ·(128 + q) − M1

Every partial sum stays within 21 bits, so the sum is exact until the final
subtraction of M1: the same single rounding as ggml's `d1·q − m1`. Small
integers become bf16 128 + v by interleaving their bytes with 0x43 bytes. On
the NPU this matched ggml bit for bit on 6,144 random lanes (about 160 per
run needing rounding).

The accumulator → bfp16ebs8 conversion matched `bfp16_encode_block` on
every case but one: when a block's largest value rounds up to +128, the
hardware does not saturate at 127, it rounds the whole block again from the
original values one exponent up. `bfp16_encode_block` now does the same
(src/npu/bfp16_pack.cpp, all paths; tests/test-bfp16-pack.cpp), which
changes only those rare blocks; −128 stays as it was. After that, 512 of 512
probe blocks matched, ties, spreads, zeros and the +128 case included.

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
  adding C afterwards spills more (31%).
- **bf16 running sums: kept as an option** (`BF16_SUMS=1 build-linux.sh`,
  design `--bf16-sums`; modes 1 and 2 only, mode 0 stays fp32). artlite ships
  bf16 C. The host side is unchanged: bf16 sums at the tile's front are
  already mode 1's output layout, and the toolchain's streams still match the
  host's word for word. Measured 2026-11-16:
  - one core, 128×64×64: **2,502 → 1,822 cycles per chunk (41% → 56% of
    peak)**. At 64×128×64 no gain (1,734 vs 1,813). artlite's own MAC helper
    compiles to the same 57-bundle, unpipelined loop here (the post-pipeliner
    tries II 32–51 and fails on one latency edge), not its II 33.
  - error (test-mul-mat, K = 2,048, vs fp64): NPU nrmse **0.87–0.90% →
    1.04–1.11%**; the CPU's q8 activations give 0.54–0.70%. One case lands
    just past the test's 2× CPU bar. More K chunks mean more roundings, so
    K = 17,408 (`down`) will be worse; k = 128 chunks halve them. Model KL
    not measured yet (GPU memory full).
  - whole GEMM at the 27B's shapes, 6 alternating pairs: 57.0 ms vs 55.3 ms
    with fp32 sums, no gain. The GEMM moves ~40 GB/s against ~46 GB/s
    available under this load: it waits on DRAM, not on the cores. bf16 sums
    pay off once traffic drops (quantized B, fewer A/B re-streams).
- **Longer K chunks help the core, not yet the GEMM.** One core reaches 59%
  of the peak at m = 64, k = 128, n = 64 (72% at k = 256, which doesn't fit
  local memory double-buffered). In the whole GEMM that tile ran at **10.2
  TFLOPS**: cores busy ~30% of the time, waiting on data, while DRAM moved 35
  of 46.5 GB/s and no channel passed 3.5 GB/s. Its smaller row blocks double
  the transfers and the C-drain waits in the runtime sequence. It goes back on
  the shelf until step 2 fixes data movement; `build-linux.sh` builds it with
  `TILE_M=64 TILE_K=128` (RTP words then at `0xe000`/`0xe004`, shim placement
  unchanged).

### Step 2 results (2026-11-16): DRAM-bound, so weight bytes are the lever

On a quieter machine the shipped GEMM runs at 20 TFLOPS (45 ms at the 27B's
shapes, M = 1,024), and it is **bandwidth-bound**. Counting what really
moves, its C tiles travel as whole fp32 tiles even in modes 1 and 2 (the bf16
results are packed at the front, `bfp16_c16_row`), so six of the shapes (all
but the fused gate/up) move ~1,390 MB in 27.0 ms: **~51 GB/s against ~57 GB/s
available quiet**. The 64×128×64 tile lost for the same reason: its 256-row
blocks re-stream B twice as often.

Tried, all as `build-linux.sh` options, with toolchain streams:

- **bf16 C** (`C16=1`, design `--c16`): bf16 sums in a bf16 tile carried as
  half-width fp32 words, sent to DRAM as it is (each core tile in 8x8 block
  order: the host would need a new reader and `bfp16_insts.cpp` the half C
  sizes). 6% fewer bytes, **27.0 → 25.2 ms (−7%)**: time tracks bytes. But
  bf16 sums **stall on long K**: with every chunk adding 64/4,096, `down`
  (K = 17,408) stops at 4.0 instead of 4.25, because each addition is half a
  bf16 step past 4.0. Real activations are noisier; the model's KL decides.
- **Wide tile 128×32×128 with bf16 C** (`TILE_K=32 TILE_N=128 C16=1`): halves
  A's re-streaming, but at k = 32 the core loop (47% of peak, 2,202 cycles per
  chunk) becomes the limit: 25.4 ms, no better than bf16 C alone. k = 32 also
  doubles the sums' roundings (host simulation: 2.9% nrmse at K = 17,408).
- **fp32 sums, only the packed half sent to DRAM**: not expressible here. The
  memtile can be told to send half of each core tile (`dims_to_stream`), but
  the toolchain keeps the transfer length at the whole tile and the GEMM
  hangs; a half-size output object needs a separate fp32 work buffer, which
  does not fit beside A and B in a core's 64 KB.
- **Fixed on the way:** the design's `--c-tiled` runtime sequence stopped
  after two row groups (it counted `C_tiles`, one per tile group, against a
  per-column index), so toolchain streams for M ≥ 2,048, or m = 64 at
  M = 1,024, left half the rows unscheduled and hung. It now walks every row
  group like `bfp16_insts.cpp`, and matches it word for word at M = 2,048.
  The backend was never affected: it uses the host generator.

So the remaining big lever is **B's bytes**, about 44% of the traffic: step 3,
quantized weights, cuts them about in half at Q4_K and should take ~20% off the
GEMM at today's bandwidth, more with bf16 C on top.

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

**Phase 1 results (2026-11-16): Q4_K, bit-exact, 2.4 weights per cycle.**
`kernels/bfp16_gemm/aie_kernels/aie2p/dq_q4k.cc` turns one 64-row × 64-value
k-tile of a re-laid Q4_K tile into the GEMM's B tile. On one core it equals
zest's packing of ggml's dequantization **bit for bit** on all 512 rows of
`blk.3.ffn_gate` (K = 5,120: 327,680 blocks) and 3 of 8 row groups of
`blk.40.ffn_down` (K = 17,408: 417,792 blocks).

- Speed: **1,683 cycles per k-tile (2.43 weights/cycle)**, from host timing
  of 40 against 400 tiles (it agrees with the core's cycle counter, 1,656).
  Below the 4/cycle target, which assumed a multiply loop near one MMUL per
  cycle; today a multiplying core takes ~2,500 cycles per k-tile (~1,820 with
  bf16 sums), so one unpacking core keeps its three multiplying cores fed.
  Getting there: the bf16 arithmetic above; nibbles as `uint4` loads, no
  shifts; one loop over (row group, half) computing that group's scales in
  registers and then its four output vectors (a separate scale loop was 12%
  slower; unrolling further spilled).
- Re-laid tile (`src/npu/qtile_pack.{h,cpp}`, 2026-11-17): per
  64 rows × one super-block, d and dmin as bf16 hi + lo (bit splits of the
  fp16 values, exact), the 6-bit scales and mins unpacked to bytes, then the
  nibbles of each 8-row × 8-value output vector together. 9,728 bytes for
  9,216 of GGUF data (+5.6%): still half of BFP16's 18,432. Byte for byte the
  tiles the kernel was checked with; 2.4 GB/s on one thread (the 27B's Q4_K in
  ~5 s, well under 1 s threaded). `tests/test-qtile-pack.cpp` checks its host
  model of the kernel (`qtile_unpack_q4k`) against `bfp16_pack_b` of ggml's
  `to_float`, on ggml-quantized rows and on raw blocks of random bytes
  (negative, subnormal and zero scales, every nibble).
- Output order: the unpacker makes vectors in the order the scales are
  shared ((row group, half, byte group)) and stores each straight into its
  B-tile chunk (`push_seek`, skipping every other chunk), so no DMA reorder
  is needed (a core tile's DMA has too few dimensions for it). Bit-exact on
  hardware as before.
- This file's Q4_K_M stores the attention projections as Q8_0 (288 tensors)
  and the FFN as Q4_K (193). Q8_0 is 8.5 bits a weight against BFP16's 9, so
  unpacking it on the NPU saves nothing; the FFN's Q4_K is where the bytes go.
- Tooling notes: the pinned Peano cannot read the cycle counter
  (`G_READCYCLECOUNTER` does not legalize), and the newer one asserts
  (`AIE::Region`) on this kernel; `load_v` silently assumes alignment
  (`load_unaligned_v` for row-group offsets, at most 1,024 bits).

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

**Phase 2 first results (2026-11-17): bit-exact, but compute-bound at 24
multiplying cores.** `designs/whole_array_bfpq.py` (row 2 unpacks, rows 3-5
multiply, one xclbin for every shape) gives the shipped BFP16 kernel's
output bit for bit: 1536 x 1024 x 1024 in modes 0, 1 and 2, and the 27B's
gate (1536 x 5120 x 17408) and down (1536 x 17408 x 5120) shapes in mode 0.
The host lays the tiles out in the order the columns read them
(`qtile_index`). Times, NPU shared with another process (min of 15 runs,
interleaved):

| shape (M x K x N)   | BFP16 (32 cores) | Q4_K (24 + 8) | Q4_K, no-op unpack | Q4_K, bf16 sums |
|---------------------|------------------|---------------|--------------------|-----------------|
| 1536 x 5120 x 17408 | 17.1 ms          | 19.5 ms       | 19.7 ms            | 17.1 ms (mode 1) |
| 1536 x 17408 x 5120 | 11.7 ms          | 15.7 ms       |                    |                 |

- The unpacker is not the limit (a no-op unpack is no faster).
- Down is compute-bound in both: 261,120 k-tile products at ~2,500
  cycles each is 11.4 ms on 32 cores and 15.2 ms on 24, as measured. Gate
  moves more bytes (A is re-read for every 512 output columns: 300 MB, as
  much as BFP16's weights), so BFP16 there is slower than its compute.
- So with the multiply loop as it is, a quarter fewer multiplying cores
  costs more than the halved weight bytes save. What the design does buy
  today is memory: the weight copy at 9,728 bytes per 64 x 256 instead of
  18,432.

**Phase 2 completed (2026-11-17).**

- Host streams: `src/npu/bfpq_insts.{h,cpp}` (the stream pieces shared with
  `bfp16_insts.cpp` moved to `src/npu/insts_stream.h`). Word for word the
  toolchain's on 10 shapes and both formats (`tests/insts/q4k_*`,
  `qcopy_*`; `test-insts-gen`).
- A runtime weight format on the unpacking core (`dq_weights`): 0 is Q4_K;
  1 copies B tiles packed on the host straight through (two to a 9,728-byte
  object, `qtile_spread_bfp16`; 5.6% more bytes than BFP16), so weights of
  any type run under this xclbin. One xclbin
  (`kernels/bfp16_gemm/prebuilt/bfpq_gemm.xclbin`, `build-linux.sh` with
  `DESIGN=q4k`) with host streams equals the BFP16 kernel bit for bit in both
  formats, modes 0-2, at the gate, down and 6,144-wide shapes.
- Times on a quiet machine (min of 15): gate 1536 x 5120 x 17408: BFP16
  12.1 ms, Q4_K 15.7, copy-through 15.7; down 1536 x 17408 x 5120: 11.8,
  15.5, 15.5. All three are compute-bound now (the earlier gate figures were
  under load); 24 multiplying cores against 32 is the whole difference.
- Switching hardware contexts measured here too: 0.16 ms a call on one
  context, 2.55 ms a call alternating between the two xclbins. So the
  backend must not interleave them per matmul.

**Backend plan from this.** `GGML_XDNA_WEIGHTS=auto|bfp16|quant`:
- `bfp16`: today's path only.
- `quant`: every NPU weight under the quantized xclbin, Q4_K unpacked and
  the rest copied through. One context, the smallest copies, ~30% slower
  GEMMs.
- `auto` (default): BFP16 copies while they fit the memory budget; from the
  layer where they stop fitting, every weight of that layer and the later
  ones goes under the quantized xclbin. (Removed 2026-11-18: see the Phase 3
  result below. `bfp16` is the default.)

**Phase 3 status (2026-11-17): in the backend.**
`npu_bfp16` holds a second kernel (its own hardware context; a batch runs
each kernel's calls as their own runlist). `xdna_npu` builds quantized-kernel
copies (`add_weight` / `add_fused` with `quant`: Q4_K re-laid straight from
the GGUF rows, fused gate/up interleaved the same way; other types packed and
spread), runs them at 384 / 768 / 1,152 rows a call, and reads their C with
the 3-row tiling. `ggml-zest.cpp`'s budget picks the kernel per weight
(`GGML_XDNA_WEIGHTS`). Tests: `mul-mat-npu-quant` (every weight-buffer case
under the quantized kernel: same errors as BFP16's to the printed digit),
`memory-budget-npu-quant`.

**Phase 3 result (2026-11-18, quiet machine,
`docs/results/halo-npu-qgemm-2026-11-18.json`).** Qwen3.8-27B Q4_K_M, prefill
tokens/s at 1,024 / 2,048 / 2,769 tokens, micro-batch 2,048:

| | 1,024 | 2,048 | 2,769 |
|---|---|---|---|
| GPU alone | 367 | 349 | 347 |
| `bfp16` (49 layers on the NPU) | 301 | 298 | 310 |
| `quant` (64 layers) | 169 | 217 | 239 |
| artlite upstream, NPU only | 440 | 443 | 432 |

- `quant` is correct on the whole model: mean KL divergence against the GPU
  0.0042 (BFP16 on 52 layers 0.0034), identical to BFP16 on all 64 layers to
  the last printed digit, and the first 24 greedy tokens match the GPU's.
- It is slower than `bfp16`, so `bfp16` is the default. More layers on the
  NPU do not help while the two devices take turns: BFP16 on all 64 layers
  (28 GB limit) gave 294 at 2,048 tokens against 298 on 49.
- `auto` (BFP16 until the copies stop fitting, then quantized) never used
  the quantized kernel: when a BFP16 copy stops fitting, a quantized copy of
  the same layer doesn't fit either. Removed.
- `quant` pads rows: each call rounds up to a multiple of 384, so a
  1,024-token chunk runs as 1,536 rows.
- `mul-mat-npu-quant` passed when the quantized kernel could not run (the
  NPU failed and the GPU finished the work). The test now fails on an NPU
  failure the backend reports, unless one was injected.
- The quantized kernel's limit is the unpacking core, 12.8 ms per 1,536-row
  feed-forward multiply (repeating the unpack 2 and 3 times a tile: 26.4 and
  39.2 ms). The multiply loop needs 11.9 ms, or about 8 with bf16 sums.
- The multiply loop restructured for the compiler's pipeliner
  (`mm_bfp_acc.cc`, same MACs in the same order) is in both shipped kernels:
  bit-identical results (the 27B's KL figures unchanged to the last digit),
  the quantized GEMM 15.5 -> 13.3 ms per 1,536-row feed-forward multiply
  (`quant` prefill +9%), the BFP16 GEMM +3%.
- The NPU and the GPU tie on the multiplies (BFP16 kernel 20.3-21.0 TFLOPS,
  the GPU about 21); the host's work between the NPU's multiplies is the
  loss. Next: the feed-forward block as one NPU submission.

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
