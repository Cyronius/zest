# Turn the row split off by default on Strix Point

**Branch:** `fix/strix-point-split-default`, cut from `origin/perf/strix-halo-prefill-2026-11-13` (3b0c3fb)

## Why

The row split (`ZEST_SPLIT`, default 50) was measured only on Strix Halo. On Strix Point it costs about a fifth of the NPU's gain. Lemonade 2026.43.3, three alternating warm pairs, the published prose prompt, 2026-11-21, Ryzen AI 9 HX PRO 370 / Radeon 890M:

| Model | split 50 (current default) | split 0 | website (Oct 21) |
|---|---|---|---|
| Qwen3-4B Q4_K_M | 1.31x | **1.48x** | 1.46x |
| LFM2.5-2.6B Q8_0 | 1.24x | **1.58x** | 1.54x |
| LFM2.5-1.2B-Instruct Q8_0 | 1.26x | **1.55x** | 1.46x |

The likely reason is that the 890M has 16 compute units, against 40 on Halo's 8060S. When the GPU is that small, giving it half the rows makes it the slower device, and the NPU ends up waiting for it.

## Spec impact

Modifies **XDNA-ROW-SPLIT**: the default becomes "50 on other GPUs, 0 on Strix Point and Krackan Point graphics". An explicit `ZEST_SPLIT` still wins either way.

New requirement **XDNA-SPLIT-DEFAULT** (Verification: `test`). It is deterministic, nobody would notice if it broke, and it's cheap to check.

### XDNA-SPLIT-DEFAULT: The split's default follows the GPU
**Applies to:** Zest
**Verification:** test

When `ZEST_SPLIT` is unset or empty, the backend shall use 0 if the Vulkan GPU's description names a Radeon 800M-series laptop GPU (`8x0M`: 890M, 880M, 860M, 840M, 820M), and 50 otherwise.

**Acceptance criteria:**
- `"AMD Radeon(TM) 890M Graphics"` (Windows driver) → 0
- `"AMD Radeon 890M Graphics (RADV STRIX1)"` (Linux) → 0
- `"AMD Radeon(TM) 860M Graphics"` → 0
- `"AMD Radeon(TM) 8060S Graphics"` (Strix Halo) → 50
- `"AMD Radeon RX 7900 XTX"`, `""`, a null description → 50
- `ZEST_SPLIT=50` set on an 890M → 50; `ZEST_SPLIT=0` on an 8060S → 0

## Changes

1. `src/xdna-env.h`: add a small pure function, `xdna_split_default(const char * gpu_description)`, that applies the rule above.
2. `src/ggml-zest.cpp` `xdna_split_percent()`: when the variable isn't set, take the default from the description of `xdna_vulkan().dev`, and print the choice once under `ZEST_TRACE`.
3. `tests/test-split-default.cpp` (host only, no NPU): checks the acceptance criteria against the pure function. Register it in `CMakeLists.txt` with the other host tests.
4. `CMakeLists.txt`: set `ZEST_SPLIT=50` explicitly on `ffn-block-npu` and `ffn-block-npu-split-fail`. They test the split itself and would otherwise stop splitting on this laptop.
5. Spec: update XDNA-ROW-SPLIT's default and status, add XDNA-SPLIT-DEFAULT, and add the Strix Point measurements.
6. Check: all tests pass here, then rerun the Lemonade A/B for the three models with nothing set. The numbers should match the "split 0" column above.

## Choice to confirm

The rule matches by GPU **name** (Radeon 8x0M) rather than by compute-unit count. ggml's Vulkan device doesn't report compute units through the backend interface, and the name is the same on Windows and Linux. As a result:

- A discrete GPU, or any other GPU, keeps the split at 50, the Halo-tuned default. Neither case has been measured.
- Krackan Point (860M/840M) gets 0. Its GPU is smaller than the 890M, so it should behave like Strix Point, but that hasn't been measured either.

Not in scope: README and website numbers. They can be updated after the step 6 rerun if you want the Strix Point table refreshed.
