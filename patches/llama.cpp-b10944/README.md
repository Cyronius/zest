# Patches to llama.cpp b10944

- `vulkan-rdna3-prefill.patch`: the Vulkan backend changes behind Zest's Strix Halo prompt speed
  (see [Strix Halo](../../docs/technical-guide.md#strix-halo)). `tools/build-vulkan.sh` applies it to the
  pinned b10944 source and builds a drop-in `libggml-vulkan.so`.
- `test-backend-ops-cases.patch`: extra `test-backend-ops` cases used to check it: shapes that pick the new
  matmul tile for every quantized type (with ragged edges), a concat with a transposed input, and
  benchmark shapes from Qwen3.8-27B. For checking only; the build doesn't apply it.

Check the backend after applying both to a b10944 checkout, building with `-DGGML_VULKAN=ON`, and running
`test-backend-ops test -b Vulkan0` (19,943 of 19,943 passed on a Radeon 8060S, 2026-11-14).
