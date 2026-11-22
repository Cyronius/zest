# v0.3.0 release checks, 2026-11-22

The XDNA-MODELS checks for a release, on the published `zest-v0.3.0-llama-b12109-win-x64.zip` (checksum verified), Ryzen AI 9 HX PRO 370 / Radeon 890M, Windows 11. Nothing set, so the row split was off (the default on this GPU).

1. Runs: `zest llama-completion`, a 5,600-token prompt (16 copies of `tests/prompts/prose.txt`), 32 greedy tokens, `ZEST_TRACE=1`. Both finished with NPU calls and no NPU failure (`*-run.log`).
2. and 3. `tools/bench.py --kl`: a paired `llama-bench` at 2,048 tokens, and llama-perplexity over the same text.

| Model | GPU alone | With the add-on | Median speedup (range) | Rounds | Mean KL | Same top token |
|---|---:|---:|---:|---:|---:|---:|
| LFM2.5-2.6B Q8_0 | 857 | 1,923 | 2.23x (1.81-4.17) | 9 | 0.0018 | 98.7% |
| Qwen3-4B Q4_K_M | 508 | 876 | 1.72x (1.62-1.84) | 5 | 0.0018 | 99.0% |

The 4.17x round is one where the GPU-alone side ran at 466 tokens/s against about 860 in the others. Both series reported a quiet machine before and after. These are direct llama.cpp figures, not Lemonade ones, so they don't compare with the Lemonade tables.
