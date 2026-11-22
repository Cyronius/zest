Zest accelerates long prompts for many GGUF models through Lemonade's
llama.cpp backend, using an AMD Ryzen AI NPU. The GPU handles attention and
writes the reply. No ONNX or Q4NX export or per-model NPU kernel compilation
is required. Coverage follows the bundled build and supported operations;
speedups vary by model.

This is an independent community add-on, tested with Lemonade 2026.47.1
and the bundled b12109 backend.

## New in v0.3.0

- **Faster on Strix Point.** Through Lemonade on a Radeon 890M laptop, prompt
  reading ran 1.48-1.58x the GPU alone on three models, up from 1.29-1.35x
  for v0.2.0 measured the same day.
- **Strix Halo support.** On its larger GPU, the NPU and the GPU now share
  each big prompt chunk (the row split). On Strix Point and other Radeon 800M
  laptops that was slower, so the split is off there by default.
- **llama.cpp b12109**, the build Lemonade 2026.47.1 bundles (was b10944).
- **Settings renamed to `ZEST_*`** (`ZEST_TRACE`, `ZEST_SPLIT`, ...). The old
  `GGML_XDNA_*` names still work for now, with a warning. Every setting is
  listed in the technical guide.

Windows and Linux packages include both `zest` and `npu`. Either command runs
the same launcher with the same options.

## Use it with Lemonade on Windows

1. Download the Windows ZIP below and extract it. Keep the files together.
2. Quit your existing Lemonade tray server and double-click `Start-Lemonade.cmd`.
   Choose Lemonade's server executable if asked, then your local GGUF model.
3. In Lemonade App, set **Settings > Base URL** to `http://127.0.0.1:13315`
   and save. Select the imported model and send a long prompt.

The launcher keeps its configuration separate and uses the GPU if the NPU
can't start. It doesn't include Lemonade, model weights, or AMD drivers.
The ZIP has an internal checksum manifest; its `.sha256` file is attached.

[Website](https://cyronius.github.io/zest/)
· [Lemonade setup and troubleshooting](https://github.com/Cyronius/zest/blob/main/docs/lemonade.md)
· [Test report](https://github.com/Cyronius/zest/blob/main/docs/lemonade-validation.md)

## Before you use it

We tested Windows 11 on a Ryzen AI 9 HX PRO 370 with Radeon 890M graphics,
and direct llama.cpp use on Linux on a Ryzen AI MAX+ 395 (Strix Halo). On
llama.cpp b12109 so far, one model has been checked end to end on Strix
Point. Lemonade on Linux hasn't been verified by this project.

The NPU keeps extra converted weight copies in RAM and changes numerical
results through block rounding. Small prompt chunks stay on the GPU.
Speed depends on the model. If the NPU fails during a request, failed work
is retried on the GPU and later requests use the GPU. The CPU reference
remains available if the GPU cannot run a failed piece.

## Direct llama.cpp use

On Windows, run `npu llama-server -m model.gguf` from the extracted ZIP.
On Linux, unpack the Ubuntu archive, enter its directory, and run
`./npu llama-server -m model.gguf`. You can also use `zest` in either command.
See the [technical guide](https://github.com/Cyronius/zest/blob/main/docs/technical-guide.md)
for commands, build instructions, accuracy checks, and standalone benchmarks.
