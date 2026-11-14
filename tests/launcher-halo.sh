#!/bin/sh
# zest on Strix Halo: the GPU alone, in 256-token pieces. zest runs from a folder whose llama-server is a
# stand-in that lists one GPU and prints the arguments it gets.
#   sh tests/launcher-halo.sh path/to/zest
set -eu
zest=$1
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cp "$zest" "$dir/zest"

stand_in() {  # $1: the GPU's name in --list-devices
    cat > "$dir/llama-server" <<EOF
#!/bin/sh
if [ "\$1" = --list-devices ]; then
    printf 'Available devices:\n  Vulkan0: $1 (65259 MiB, 63564 MiB free)\n'
    exit 0
fi
echo "args: \$*"
EOF
    chmod +x "$dir/llama-server"
}

expect() {  # $1: the line llama-server must print; the rest: zest's arguments
    want=$1
    shift
    out=$(GGML_BACKEND_PATH= "$dir/zest" "$@" </dev/null 2>&1)
    if ! printf '%s\n' "$out" | grep -qxF -- "$want"; then
        printf 'expected: %s\ngot:\n%s\n' "$want" "$out"
        exit 1
    fi
}

stand_in "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
expect "args: -dev Vulkan0 -ub 256 -b 2048 -m model.gguf" llama-server -m model.gguf
expect "args: -dev Vulkan0 -m model.gguf -ub 512" llama-server -m model.gguf -ub 512
expect "args: -m model.gguf -dev Vulkan0" llama-server -m model.gguf -dev Vulkan0

stand_in "AMD Radeon 8050S Graphics (RADV STRIX_HALO)"
expect "args: -dev Vulkan0 -ub 256 -b 2048 -m model.gguf" llama-server -m model.gguf

# AMD's Windows driver wasn't measured: the usual path, and with no add-on here nothing is added
stand_in "AMD Radeon(TM) 8060S Graphics"
expect "args: -m model.gguf" llama-server -m model.gguf

# the 16-CU 8040S stays on the usual path: with no add-on here, nothing is added
stand_in "AMD Radeon 8040S Graphics (RADV STRIX_HALO)"
expect "args: -m model.gguf" llama-server -m model.gguf

echo "launcher-halo: ok"
