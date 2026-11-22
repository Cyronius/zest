# Move Zest from llama.cpp b10944 to a current build

**Branch:** separate from `fix/strix-point-split-default`, cut after it lands.

## Where things stand (2026-11-22)

- Zest is built against **b10944**, the build bundled with Lemonade 2026.43.3.
- The newest llama.cpp is **b12631** (2026-11-22). **v1.0.0** (2026-11-13) is a tagged milestone that ships no binaries, so the Windows/Linux Vulkan zips we build against only exist for the `bNNNNN` builds.
- Lemonade's newest stable release, **2026.47.1**, pins llama.cpp **b12109** for Vulkan.
- The add-on interface changed between b10944 and v1.0.0. `GGML_BACKEND_API_VERSION` went from 2 to 3. The buffer-type and buffer structs gained new members in the middle, ggml.h has about 350 changed lines (op list included), and v1.0.0 removed 35 deprecated llama.h functions. Zest fills those structs by position, so it won't even load without a port.

## Choice to make: which build

| Option | For | Against |
|---|---|---|
| **b12109 (recommended)** | Same build as Lemonade 2026.47.1, so Zest drops into a current, unmodified Lemonade the way v0.2.0 did with 2026.43.3 | About 500 builds behind the newest |
| b12631 (newest today) | Newest fixes and models | No Lemonade release runs it yet, so Lemonade users still need Zest's own llama-server, and it will be stale within days |

Zest's add-on must match the llama.cpp build exactly, so "the newest" is only worth it if we're shipping our own llama-server anyway. Matching Lemonade's pin keeps the Lemonade story simple.

## Steps

1. Change the tag in `tools/fetch-llama.ps1` / `.sh`, `CMakeLists.txt` (`GGML_DIST`, `GGML_HEADERS`), the CI and release workflows, and the package scripts. Today b10944 appears in about 20 files outside the results folders.
2. Port `src/ggml-zest.cpp` to backend interface version 3: the new struct members (`alloc_buffer_n`, `get_alloc_size_n`, `set_tensor_n`, `get_tensor_n`, `cpy_tensor_n`), and anything else the compiler finds. Switch the struct initializers from positional to named members where the compiler allows, so the next interface change breaks loudly.
3. Check the op handling in `src/xdna-exec.cpp` and the claim rules against the new ggml.h op list, especially any renumbered or new ops next to the multiplies (norms, SwiGLU, adds).
4. Check `tools/zest.cpp` against the new llama-server, llama-bench and llama-completion options (`-dev`, `-ts`, `-ub`, `-b`, `--models-preset`, removed `--defrag-thold`).
5. Rebuild, run all tests, and run the model checks the spec lists (XDNA-MODELS, XDNA-BLOCK-AGREES, KL divergence) on Strix Point.
6. Rerun the Lemonade A/B (five models) with the matching Lemonade release, and check for a speed change caused by the new llama.cpp alone: GPU-only numbers before and after.
7. Update the README, the setup guide, the technical guide, the site's "tested with" lines, and the spec's pinned versions, then cut a release.

## Spec impact

Modifies **XDNA-BACKEND-LOAD**, **XDNA-SERVER-USE**, **XDNA-RELEASE** and **XDNA-LINUX** (pinned build and Lemonade version). No new requirements expected unless the port turns up new behavior.

## Risks

- GPU-only speed may change with the new Vulkan backend, which moves every ratio we publish. Step 6 measures that before anything is published.
- Lemonade 2026.47.1 may pass llama-server options that b10944-era scripts don't expect. The launcher script needs checking against it.
- Strix Halo numbers would need a rerun on that machine.
