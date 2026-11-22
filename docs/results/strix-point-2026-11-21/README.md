# Strix Point through Lemonade, 2026-11-21

Ryzen AI 9 HX PRO 370 / Radeon 890M, Windows 11 26200, Lemonade 2026.43.3 portable, llama.cpp b10944 Vulkan. Source: `origin/perf/strix-halo-prefill-2026-11-13` at 3b0c3fb, built locally.

`lemonade_ab.py` repeats `tools/start-lemonade.ps1` and `tools/bench-lemonade.ps1` with the same settings: context 8192, batch and micro-batch 2048, flash attention, one slot, temperature 0, 16 output tokens, prompt caching off, eight copies of `tests/prompts/prose.txt`. Three alternating warm pairs per model; one warmup request after each load.

- `split-off.json`: `GGML_XDNA_SPLIT=0` (now `ZEST_SPLIT`, and the default on this GPU)
- `split-on.json`: the 50% row split, the default at the time
- `v0.2.0.json`: the v0.2.0 release package, same day
- `smoke-2026-11-22-*`: one pair on Qwen3-4B with nothing set, after the default changed, plus the answer checks (3/3 in both modes). The machine was busy, so its speed isn't a measurement.

Every 2026-11-21 series was taken with `dibs run --measure` and reported a quiet machine before and after.
