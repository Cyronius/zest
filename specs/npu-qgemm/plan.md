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

### After Phase 3: the feed-forward block as one NPU submission (2026-11-18)

A feed-forward block was three NPU calls with the host between them: the
fused gate/up call wrote SiLU(gate) * up as bf16, the host read all of it
(the block's widest tensor, 17,408 values a token on the 27B), packed it
again as bfp16 activations, and submitted the down call. Now the first call
writes that result as the second's activation operand and the two go to the
NPU together (XDNA-FFN-CHAIN in `specs/xdna-backend/spec.md`). Three things
made that possible, and all three are in the BFP16 kernel that ships
(`designs/whole_array_bfp_colwise.py`, which replaces
`whole_array_bfp_rtp.py`):

- **Activations by column.** Before, each core row took a block of A's rows
  and each column a group of B's, and a column's memtile joined its four
  cores' output tiles: four different row blocks of the same output columns.
  C therefore landed in DDR output-column-major, the transpose of what a
  multiply reads as A (row block by row block, every K chunk of it
  together). Now each column takes a row block of A and each core row a
  group of B (core tile 64 x 64 x 128, so a core's fused tile is 64 rows x 64
  SiLU(gate) * up values: exactly one A tile of the next multiply), and a
  column's join is four consecutive K tiles of one row block. C is row-major
  in tiles, and in mode 3 it is the next A byte for byte, in K's natural
  order. Same traffic per FLOP as before (A once per 512 outputs, B once per
  512 rows); the multiply kernel is the same source at other tile sizes.
- **Only a mode's bytes reach DDR.** Every output mode packs its results at
  the tile's front, but the whole 32 KB fp32 tile used to travel to DDR
  whatever the mode. The instruction stream now sets how many words of each
  tile the memtiles pass on (the length field of the join's four buffer
  descriptors): all of it (mode 0), half (bf16), a quarter (fused), or 4,608
  bytes (mode 3). A fused gate/up call's C falls from 143 MB to 20 MB per
  1,024 rows, 13% of that call's DDR traffic.
  A memtile channel waiting for its lock has **already loaded its next
  descriptor**: writing a new length alone left the first tile after every
  change at the old length (found by where the tiles landed). So the stream
  resets the channel and starts it again at its first descriptor. For that
  to be where the channel stands between calls, the join has one buffer, not
  two: its four descriptors go round once per join (a second buffer buys
  nothing, tiles being tens of thousands of cycles apart).
  The stream therefore writes registers the toolchain placed. The design
  holds that placement as a table, emits the same writes in its own runtime
  sequence (so the host's streams are still checked word for word against
  the toolchain's, 35 of 35), and `check_placement.py` fails a build whose
  placement differs.
- **Mode 3 in the epilogue** (`mm_bfp_acc.cc`, `pack_chain`): the SwiGLU
  values, rounded to bf16 as mode 2 sends them, converted on the core to
  bfp16ebs8 (the conversion the Q4_K unpacker uses, equal to the host's
  `bfp16_encode_block`) and stored where the next A tile has them.

Checks: mode 3's output equals the host's packing of mode 2's, byte for
byte, at 1,024 x 512 x 2,048 and at the 27B's 1,024 x 5,120 x 34,816; the two
calls as one runlist give the down call's output on host-packed activations
(`test-npu-wide`); a whole block through the backend gives the same bits
with and without chains and under the quantized kernel
(`ffn-chain-same-bits`); a NaN made in the first call comes out as a NaN
row of the second. On the 27B the KL figures are those of the kernel before
(0.003445 mean, 0.251934 maximum, 96.275% same top token), chains on or
off.

Speed, with another session's LLM server keeping the GPU at 95% throughout
(so DRAM, which this kernel is bound by, was shared: the old kernel ran at
15.4-16.5 TFLOPS in the same sitting, against 20.3-21.0 that morning):

| call, 1,024 rows | kernel before | by column |
|---|---|---|
| gate/up fused, bf16 out (5,120 -> 34,816) | 23.65 ms | 22.93 ms |
| the same, written as the next A | (none) | 22.81 ms |
| down (17,408 -> 5,120) | 11.04 ms | 10.87 ms |

Qwen3.8-27B Q4_K_M, 2,048 tokens, ten samples each in turn: 190.8
tokens/s before (163-197), 196.2 with the new kernel and chains (194-202).
Host passes per 196 multiplies fall from 576 to 520, and the pass workers'
decode and encode time by 30% and 24%, but inside the pieces the NPU's own
time is now what the host waits on, so most of that saving shows as
waiting. With the GPU to itself later that evening (controls matching the
morning's quiet figures): 300.0 before, 305.6 with the new kernel and
chains (`docs/results/halo-npu-chain-2026-11-18.json` has both sittings).

**The row split (`GGML_XDNA_SPLIT`; the default at 50% since 2026-11-19,
an off-by-default experiment when this was written).** Taking
turns, the two devices cannot beat the faster of them by much: the NPU and
the GPU tie on the multiplies, so the pieces the NPU takes cost what the GPU
would have spent on them plus the host's work. The split gives the NPU a
share of each claimed piece's rows and runs the piece's own graph over the
other rows on the GPU at the same time, in place in the tensors' memory
(XDNA-ROW-SPLIT). An earlier token-row split (2026-11-13) lost, when the
NPU kernel ran at 12-13 TFLOPS and the host passes cost twice what they do
now. On paper, at today's speeds, half the rows each takes the claimed
pieces from about 4.6 s to about 2.4 s per 2,048 tokens (298 tokens/s to
about 430), more with every layer on the NPU.

It passes its tests (a block with half its rows on the GPU; the same with
every NPU submission failing, where the GPU then runs the NPU's rows too)
and on the 27B it moves the answers toward the GPU's. With one
2,048-token prompt per batch, as in prefill, and every piece shared: mean
KL 0.0025, maximum 0.031 (0.0034 and 0.15 without the split). With four
512-token sequences per batch: 0.0028 and 0.029 (0.0034 and 0.25). After a
2,887-token prompt that asks for a passage again word for word, zest writes
the GPU's 24 tokens with the split and without it.

Measured with the GPU to itself, 2,048 tokens (1,024 in brackets), on the
kernel with resident activations (below):

| | tokens/s |
|---|---|
| GPU alone | 346.0 (364.9) |
| zest, no split | 323.8 (325.9) |
| split 50% | 353.8 (360.2) |
| split 25% (NPU 512 rows) | 343.4 |
| split 75% (NPU 1,536 rows) | 339.7 |
| all 64 layers on the NPU (28 GB), no split | 318.3 |
| all 64 layers, split 50% | 359.4 |

The first zest figures above the GPU alone at the same micro-batch, but far
from the paper figure, and the trace says why: only 142 of 218 pieces a
prompt were shared, and the ones left out were the largest. In the 27B's
linear-attention layers (39 of the 52 on the NPU), the piece with the qkv
multiply and the piece with the feed-forward block each end with a view
that computes nothing: a transpose of the piece's own result for an op
outside it, or a reshape of the next layer's recurrent state. llama.cpp's
scheduler leaves a view where it falls in the node order, which puts these
in the piece before them, and the row view refused the whole piece for a
tensor whose rows don't run along its last dimension. Those pieces were
2.6 s of the 3.5 s a prompt spent in pieces. (A first guess, an op done in
place, was wrong; the row view takes those now too, and the block test has
one.) A view is now part of the GPU's share only if an op of the piece
reads it. The block test ends with both kinds of view, and its split test
fails unless the trace says the piece was shared: before, it passed either
way.

With every piece shared (2026-11-19, 02:22-02:35, the GPU to itself, the
controls as before), tokens/s:

| | 1,024 tokens | 2,048 | 4,096 |
|---|---|---|---|
| GPU alone | 366.9 | 343.6 | 331.5 |
| zest, no split | 324.1 | 322.5 | 313.1 |
| split 50% | 440.5 | 431.5 | 414.5 |
| split 25% (NPU 512 of 2,048 rows) | | 374.1 | |
| split 75% (NPU 1,536) | | 375.4 | |
| all 64 layers on the NPU (28 GB), no split | | 317.4 | |
| all 64 layers, split 50% | | 469.1 | |
| all 64 layers, split 25%, 75% | | 383.2, 388.2 | |

That is the paper figure: 1.26x the GPU alone at 2,048 tokens with the
default 20 GB of weight copies and 1.37x with every layer on the NPU, where
without the split more layers on the NPU lose. Per 2,048-token prompt, in
seconds:

| | no split | split 50% | split 50%, 64 layers |
|---|---|---|---|
| whole prompt | 6.35 | 4.75 | 4.37 |
| in zest's pieces | 4.03 | 2.55 | 3.28 |
| of that: waiting on the NPU | 2.90 | 1.62 | 2.11 |
| host passes, reads, writes | 1.11 | 0.75 | 0.92 |
| waiting for the GPU's rows after the NPU's | | 0.16 | 0.21 |
| the GPU alone on the rest | 2.32 | 2.20 | 1.09 |

Half each is close to balanced (the pieces wait 0.16 s for the GPU's rows;
at 25% they wait 1.88 s), and a micro-batch of 2,048 allows only quarters.

It also holds when the GPU has other work. With a second job of this
session's own generating tokens from the 27B on the GPU throughout (04:53,
2,048 tokens, one round): GPU alone 237.0, zest without the split 199.7,
split 50% 290.7 (282.7-298.7), NPU 25% of the rows 270.1, NPU 75% 249.2.
Right after, with the GPU to itself again: 342.6 and 430.6. So 1.23x the
GPU alone there, against 1.26x quiet. The evening's figure that had the
split losing on a busy GPU (229 against 196-260 without, in different
windows) came from the build that left the largest pieces whole.

On three other models (05:22, 2,048 tokens, n=3, the GPU and the NPU held;
KL against the GPU with one 2,048-token prompt per batch), every piece
shared and no NPU failure in each:

| | GPU alone | zest, no split | split 50% | mean KL, no split | with the split |
|---|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M | 3,456 | 3,094 | 4,060 | 0.0028 | 0.0017 |
| Qwen3.5-9B Q4_K_M | 1,081 | 976 | 1,341 | 0.0041 | 0.0024 |
| Qwen3-Coder-30B-A3B Q4_K_M (mixture of experts) | 1,718 | 1,048 | 1,370 | 0.0045 | 0.0035 |

The dense models go from behind the GPU alone to 1.17x and 1.24x of it.
The mixture-of-experts model stays behind it (0.80x, from 0.61x); the next
section has why.

The quantized-weight kernel (`GGML_XDNA_WEIGHTS=quant`, all 64 layers in
16 GB of copies) gains from the split too, but stays behind (04:19, 2,048
tokens, GPU alone 342.5 and the default kernel's split 431.2 in the same
turn): 234.2 without the split, 308.4 at 50%, 370.9 at 25%, where the NPU
has so few rows that the kernel matters little. Its answers with the split:
mean KL 0.0025, maximum 0.037 (0.0041 and 0.14 without), every piece
shared. So it is not the way to every layer on the NPU at speed: that costs
28 GB of copies with the default kernel.

**Activations resident in the memtile (2026-11-18, in the shipped kernel).**
The BFP16 kernel is held by DRAM at about 50 GB/s of the port's 57 (21
TFLOPS quiet), and nearly half of a call's traffic is A, read from DDR once
for every 512 output columns. Now a call whose row block fits (K up to
5,440) has each column's memtile take its 64-row block of A once per pass
and play it to its cores once per output tile, as artlite does: A crosses
DDR once. That is 368,640 bytes of a memtile's 524,288 at K = 5,120, beside
the C join's 131,072, which the by-column layout makes possible (one row
block per memtile) and which needed three things in the design:

- every column's fifos pinned to its own memtile and shim (the toolchain had
  put two columns' A in one memtile), which also makes the placement the
  same in every column;
- B straight from the shims to the core rows, with no memtile relay, whose
  two tiles took two of the banks the row block needs (no slower: 4.06
  against 4.15 ms on 1,024 x 5,120 x 5,120);
- the runtime sequence programming the memtile itself, per call: it resets
  A's two channels and gives them either the fifo's own descriptors again
  (A relayed tile by tile, for a longer K) or three of its own over the
  whole row block: *fill* (waits on a "free" lock, releases "full"), *play*
  (needs "full", leaves it; queued with a repeat count of the tiles less
  one) and a pass's *last play* (releases "free", so the next pass's fill
  can overwrite the buffer). The register words are the toolchain's own
  (`npu_writebd`, `npu_push_queue`), and the host makes the same stream,
  checked word for word (45 of 45 reference streams, both kinds).

Results are the same bits as relayed (mode 3's output still equals the
host's packing; calls of either kind in any order in one context; the 27B's
KL figures unchanged). A narrow call loses (the 27B's 1,024-wide k
projection ran 10% slower: a memtile plays a block only once it has all of
it), so calls under three output tiles stay relayed, as do the down multiply
(K = 17,408) and the 6,144-long o projection.

Speed, the same loaded sitting (GPU at 100% for another session), relayed
and resident kernels in turn, 1,024 rows:

| call | relayed | resident |
|---|---|---|
| gdn qkv (5,120 -> 10,240) | 7.11, 7.49 ms | 6.27, 6.59 ms |
| gdn z (5,120 -> 6,144) | 4.54, 4.65 | 4.01, 3.95 |
| fa q+gate (5,120 -> 12,288) | 9.04, 9.42 | 7.66, 7.89 |
| gate/up fused, as the next A (5,120 -> 34,816) | 25.67, 27.15 | 22.09, 21.73 |
| down (17,408 -> 5,120), relayed in both | 12.03, 12.01 | 11.78, 11.60 |

13-25% on the calls it applies to, which are about two thirds of the 27B's
FLOPs. With the GPU to itself later that evening, in turn twice:

| call | relayed | resident |
|---|---|---|
| gdn qkv (5,120 -> 10,240) | 4.71, 4.65 ms (22.8-23.1 TFLOPS) | 4.06, 4.01 (26.5-26.8) |
| fa q+gate (5,120 -> 12,288) | 5.64, 5.55 | 4.80, 4.76 (26.9-27.1) |
| gate/up fused, as the next A | 17.38, 16.93 (21.0-21.6) | 14.68, 14.68 (24.9) |
| down, relayed in both | 7.76, 7.62 (23.5-24.0) | 7.62, 7.64 |
| all seven shapes | 59.1, 57.9 (21.8-22.2) | 51.7, 51.7 (24.9) |

A resident gate/up call moves 427 MB instead of 822, at 29 GB/s: DRAM no
longer holds it. What does is a core's own streams: each core takes its
operands on two 32-bit streams, and B's tile (9,216 bytes per K chunk) needs
at least 2,304 cycles, 26 TFLOPS for the array at 1.81 GHz, which is where
these calls now run. Prefill at 2,048 tokens: 305.6 to 323.8 tokens/s.

**Kernel work that remains.**
- *A balanced tile* (80 x 64 x 80) would lift the stream limit to the
  multiply loop's own (about 30 TFLOPS), at awkward sizes (M in 640s, N in
  320s).
- *The o projection resident* (K = 6,144 needs 442,368 bytes): room for it
  if the C join took only the bytes a 16-bit mode sends (65,536), which
  means shortening the cores' own transfers, whose receiving channels wait
  with a lock already taken.
- *The down multiply* (a quarter of the FLOPs) cannot be resident at
  K = 17,408; its traffic is A and B in equal parts.

### Mixture-of-experts models, and how results get to the GPU (2026-11-19)

On Qwen3-Coder-30B-A3B (width 2,048, 128 experts of 768, 8 a token) zest
read a prompt at 0.63x the GPU alone, and on Qwen3.6-35B-A3B at 0.51x.
Vulkan's own per-op timer (`GGML_VK_PERF_LOGGER=1`) says where the GPU
alone spends a 2,048-token prompt on the 30B:

| | ms | share |
|---|---|---|
| expert multiplies (`MUL_MAT_ID`) | 561 | 47% |
| attention | 209 | 17% |
| attention projections and the router, which zest takes | 232 | 19% |
| everything else | 195 | 16% |

So the NPU has a fifth of the work to take. But more than half of what zest
lost was not its own time: **the GPU's expert gate and up multiplies took
8.4 ms each under zest against 3.8 ms alone**, 430 ms a prompt. They are the
first ops to read the normed activations that zest's second piece of each
layer writes. What it was not: the GPU's clock (sampled at 2 kHz, it only
follows how busy the GPU is), the host's threads (7.97 ms with two), the
router multiply (8.37 ms without it). What it was: the copy. zest wrote its
results with `ggml_backend_tensor_set`, a `memcpy` into Vulkan's mapped
memory, and glibc's `memcpy` uses ordinary stores under 24 MB on this
machine and non-temporal stores from there. This model's results are 17 MB
tensors; the 27B's are 42 MB, which is why it never showed. Three things
each put those multiplies back at 4.0-4.2 ms: leaving the norm to the GPU,
lowering glibc's threshold to 1 MB for the process (`GLIBC_TUNABLES`), and
having the GPU copy the results from the piece's pinned memory
(`ggml_backend_tensor_set_async`), which is what zest now does
(XDNA-RESULT-COPY). The GPU's copy is also three times as fast as the
memcpy: 0.19 s to 0.06 s a prompt on the 30B. Flushing the source from the
CPU's caches before the GPU's copy gained nothing.

Prefill at 2,048 tokens, one build with a switch between the copies (08:45,
the GPU and the NPU held; the GPU alone from ten minutes earlier):

| | GPU alone | memcpy | GPU copy | GPU copy, with the split |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 1,718 | 1,059 | 1,447 | 1,620 (NPU a quarter of the rows) |
| Qwen3.6-35B-A3B | 1,364 | 744 | 1,082 | 1,257 (half) |
| Qwen3.5-9B | 1,079 | 986 | 1,047 | 1,408 (half) |
| Qwen3.8-27B | 344 | 323 | 341 | 442 (half) |

The 27B without the split is now within 1% of the GPU alone.

The build that keeps the GPU copy (09:27-09:45, each model's figures within
two minutes of each other):

| | GPU alone | zest | NPU a quarter of the rows | half |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 1,712 | 1,446 | 1,631 | 1,645 |
| Qwen3.6-35B-A3B | 980-987 | 903 | 1,028 | 1,030 |
| Qwen3.5-9B | 907 | 871 | 976 | 1,120 |
| Qwen3.5-2B | 3,440 | 3,310 | 3,750 | 4,317 |
| Qwen3.8-27B | 344 | 340 | 381 | 449 |

On the 35B and the 9B the GPU alone itself read 28% and 16% slower from
09:28 on than at 08:40, and zest with it; the other three models did not
move. It is not zest (at 10:18 the 9B gave 915, 915 and 919 tokens/s with
the build before, this one and no zest loaded) and it is not explained, so
for those two models only one sitting's figures compare. Answers: mean KL
against the GPU 0.0045 on the 30B (0.0035 with the split), 0.0090 on the 35B
(0.0060), 0.0041 on the 9B (0.0024), 0.0033 on the 27B (0.0025), no NPU
failure.

A second, smaller thing: 29 of the 35B's 250 pieces a prompt, and 8 of the
27B's 218, held no multiply at all. A small op is claimed behind a multiply, llama.cpp's
scheduler then orders the graph its own way, and an op zest doesn't take
can fall between the two. Such a piece cost 2.7 ms on the host for a few
cheap ops; the GPU runs it now, in 0.4 ms (XDNA-BARE-PIECE).

**What is left on these models is structure.** zest's pieces on the 30B
take 0.44 s a prompt (97 pieces of 4.5 ms: NPU 0.12 s, host passes 0.18,
reads and writes 0.11) for work the GPU does in 0.24 s. At this width a
multiply is about a millisecond on either device (the NPU's q projection on
1,024 rows: 1.12 ms, 15 TFLOPS, against 25 on the 27B's shapes), so the
kernel's and the host's fixed costs per piece are what is being compared.
Sharing rows helps (0.96x with the split at 50%), and no more than that:
with a fifth of the work to share, a free NPU would give 1.1x.

**The expert step on the NPU does not pay with this kernel.** An
expert-sized call on 512 rows takes 0.40 ms (gate/up fused, 2,048 to 1,536)
and 0.26 ms (down), so a layer's 16,384 (token, expert) rows cost at least
21 ms on the NPU against the GPU's 12 ms, and that is if every pass were
full: the kernel gives all eight columns the same weights, and an expert
has 128 rows on average, so each would need a pass of its own, or the
instruction streams a new kind of call with a weight block per column. The
experts' BFP16 copies would be 33 GB on the 30B.
`docs/results/halo-moe-2026-11-19.json` has the figures.

### The page's Strix Halo figures (2026-11-19)

Measured for the project's page by `bench.ps1`'s method (a Python port, now
`tools/bench.py`: no PowerShell on this machine), each model in a turn of
its own with the machine reserved, on one build throughout (the row split on
by default). `docs/results/halo-page-2026-11-19.json` has every round.

| model | GPU alone, tokens/s | as shipped | split off | mean KL as shipped / split off |
|---|---|---|---|---|
| LFM2.5-2.6B Q8_0 | 2,980 | 1.56x (1.55–1.56) | | 0.0036 / 0.0066 |
| LFM2.5-1.2B Q8_0 | 6,478 | 1.52x (1.30–1.54) | | 0.0020 / 0.0041 |
| Qwen3.8-27B Q4_K_M, 30 GB limit (64 layers) | 343 | 1.44x (1.44–1.44) | 0.99x | 0.0031 / |
| Qwen3-4B Q4_K_M | 1,861 | 1.39x (1.39–1.39) | | 0.0103 / 0.0134 |
| Qwen3.5-9B Q4_K_M | 1,080 | 1.31x (1.31–1.31) | 0.97x | 0.0018 / 0.0037 |
| Qwen3.8-27B Q4_K_M | 343 | 1.31x (1.30–1.31) | 0.99x | 0.0028 / 0.0034 |
| Qwen3.8-27B UD-IQ3_S | 353 | 1.27x (1.26–1.27) | 0.98x | 0.0035 / 0.0040 |
| LFM2.5-350M Q8_0 | 18,772 | 1.27x (1.26–1.28) | | 0.0083 / 0.0138 |
| Qwen3.5-2B Q4_K_M | 3,455 | 1.24x (1.24–1.25) | 0.95x | 0.0014 / 0.0028 |
| Qwen3-Coder-30B-A3B Q4_K_M | 1,715 | 0.96x (0.96–0.96) | 0.84x | 0.0049 / 0.0059 |
| Qwen3.6-35B-A3B UD-Q4_K_XL | 1,356 | 0.95x (0.95–0.96) | 0.81x | 0.0072 / 0.0095 |

- Every dense model gains with the split. Without it, of the five measured
  that way, only LFM2.5-2.6B does (1.10–1.11x, measured 2026-11-20): on
  this machine the split is nearly all of the gain.
- All 64 of the 27B's layers on the NPU (25.8 GB of copies against the
  default 20 GB's 52 layers) give 494 tokens/s, 1.44x. The default limit is
  what it is because of the damage seen past 26 GiB on Windows
  (XDNA-MEMORY-BUDGET). That ceiling is not here: `npu-ceiling-repro` ran 16
  to 64 GiB of NPU buffers with nothing damaged (2026-11-20, 02:10 and
  05:08). The
  default limit on Linux has no 20 GB cap since then (the owner's decision);
  Windows keeps it. On that build (2026-11-20, 27 of 27 tests) the 27B
  reads 1.44x by default (1.43–1.45, Q4_K_M) and 1.38x (UD-IQ3_S), and
  1.31x with the limit set back to 20 GB; its mean KL is 0.0031 and 0.0037.
  The other models' accuracy figures are the same to the last digit on both
  builds, and LFM2.5-2.6B measured again gave 1.55x.
- Two things about measuring here. The machine has two schedulers in use
  (arbiter and the older dibs) and a job under one is invisible to the
  other: three rows of the 2026-11-20 series ran with another project's
  job on the GPU and were thrown out (the GPU alone at a third of its
  speed). A timing now holds both and refuses to start on a busy GPU. And
  the per-round files of that day's runs were in /tmp and went with a
  restart; the results file says which figures were recorded from
  summaries.
- The narrow-model warning ("the GPU alone was faster than the NPU on every
  model this size we tested") no longer holds here: LFM2.5-350M, width
  1,024, gains 1.27x. It was measured on Strix Point before the split.
- The split's share is fixed at half. Strix Point's GPU is far slower
  against the same NPU (the published 1.9x on the 27B UD-IQ3_S says a layer
  on the NPU path takes about 0.4 of the GPU's time), so half the rows on
  its GPU would take about as long as all of them on the NPU: the right
  share there is probably nearer three quarters, and differs by model.
  Not measured yet; if it is so, the share should follow the two devices'
  measured piece times rather than be a setting.

**One token on Qwen3-4B.** Its mean KL is one token: 14.0 by
llama-perplexity's count at piece 1, position 449 of the text (the
repository's README: after `https://cyron`, the third time that address
appears; next is `ius`), and 0.0035 without it. Each run's logits saved and
compared token by token over the first four pieces:

| | p(`ius`) | KL there against the GPU |
|---|---|---|
| GPU alone, and again | 0.165, 0.165 | 0 |
| CPU alone | 0.061 | 0.55 |
| zest as shipped; split off; chain off; quantized-weight kernel; build a87d2ae | 1.000 | 8.26 |
| zest, the NPU's share on the CPU (`GGML_XDNA_HOST_ONLY=1`) | 0.812 | 1.44 |
| zest, multiplies only (`GGML_XDNA_BLOCKS=0`) | 0.010 | 1.18 |

(8.26 and not 14.0 because the saved log-probabilities stop 16 below the
largest logit.) The CPU and the GPU differ most from each other at this
same token (0.80 against the CPU, the next largest 0.32, then under 0.1),
so the position is one where the model amplifies whatever rounding it is
given, and zest's 8-bit blocks are the largest rounding of the three. It is
not something this session's changes brought, and not the split, the chain
or the kernel variant. Which side a higher-precision file of the model
takes was not checked.

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
