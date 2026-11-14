#!/usr/bin/env bash
# Builds llama.cpp b10944's Vulkan backend with patches/llama.cpp-b10944/vulkan-rdna3-prefill.patch,
# giving build/vulkan/libggml-vulkan.so: a drop-in for the release's libggml-vulkan.so that reads prompts
# faster on RDNA3 GPUs such as Strix Halo's Radeon 8060S. Other GPUs run the release's code paths.
#
# Needs cmake, a C++ compiler, curl, patch, glslc with GL_KHR_cooperative_matrix (shaderc 2023 or newer,
# or the LunarG Vulkan SDK), and the Vulkan and SPIR-V headers. If the headers aren't installed system-wide,
# set VULKAN_PREFIX to a directory holding include/vulkan, include/spirv and SPIR-V-Headers' cmake config.
#   tools/build-vulkan.sh            then copy build/vulkan/libggml-vulkan.so next to llama-server
set -euo pipefail

tag=b10944
src_sha256=c080edc3128934166cbc228890ff3edd0398723661a7698ab5a1a48d057017fc
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
patch_file="$root/patches/llama.cpp-$tag/vulkan-rdna3-prefill.patch"
work="$root/third_party/llama-$tag-vulkan-src"
out=${OUT:-$root/build/vulkan}

for tool in cmake curl patch glslc; do
    command -v "$tool" >/dev/null || { echo "build-vulkan: $tool not found" >&2; exit 1; }
done

archive="$root/third_party/llama.cpp-$tag-src.tar.gz"
if [[ ! -f "$archive" ]]; then
    mkdir -p "$root/third_party"
    curl --fail --location --retry 3 --silent --show-error \
        "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$tag.tar.gz" -o "$archive.part"
    mv "$archive.part" "$archive"
fi
echo "$src_sha256  $archive" | sha256sum --check --status ||
    { echo "build-vulkan: the llama.cpp $tag source archive's checksum doesn't match" >&2; exit 1; }

# a fresh tree every time, so the patch always applies to the pristine release source
rm -rf "$work"
mkdir -p "$work"
tar -xzf "$archive" -C "$work" --strip-components=1
patch -d "$work" -p1 --quiet < "$patch_file"

generator=()
command -v ninja >/dev/null && generator=(-G Ninja)
prefix=()
if [[ -n "${VULKAN_PREFIX:-}" ]]; then
    prefix=(-DCMAKE_PREFIX_PATH="$VULKAN_PREFIX" -DVulkan_INCLUDE_DIR="$VULKAN_PREFIX/include")
fi
cmake -S "$work" -B "$work/build" "${generator[@]}" "${prefix[@]}" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF > "$work/configure.log"
# without cooperative matrices the backend would build, but slower than llama.cpp's own release
grep -q "GL_KHR_cooperative_matrix supported by glslc" "$work/configure.log" ||
    { echo "build-vulkan: $(command -v glslc) lacks GL_KHR_cooperative_matrix; use shaderc 2023 or newer, or the Vulkan SDK" >&2; exit 1; }
cmake --build "$work/build" -j"$(nproc)" --target ggml-vulkan

mkdir -p "$out"
cp "$work/build/bin/libggml-vulkan.so" "$out/"
echo "build-vulkan: $out/libggml-vulkan.so"
