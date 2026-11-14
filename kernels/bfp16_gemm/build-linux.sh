#!/usr/bin/env bash
# Rebuild the shipped 8-column BFP16 kernel on Linux. Run this under
# `dibs run build -- ...` on a shared machine.
#
# IRON_ENV points to a Python venv with mlir_aie 1.4.3 and Peano installed;
# MLIR_AIE_ROOT points to the matching mlir-aie checkout. Their defaults are
# the ignored local development copies under third_party/.
# Usage: build-linux.sh M K N [out-mode]
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
iron_env=${IRON_ENV:-$root/third_party/ironenv}
mlir_aie_root=${MLIR_AIE_ROOT:-$root/third_party/mlir-aie}

if (( $# < 3 || $# > 4 )); then
    echo "usage: $0 M K N [out-mode: 0, 1, or 2]" >&2
    exit 2
fi
M=$1 K=$2 N=$3 mode=${4:-0}
set --  # env_setup.sh must see no inherited shape arguments
if [[ ! $M =~ ^[1-9][0-9]*$ || ! $K =~ ^[1-9][0-9]*$ || ! $N =~ ^[1-9][0-9]*$ || ! $mode =~ ^[012]$ ]]; then
    echo "M, K, N must be positive integers and out-mode must be 0, 1, or 2" >&2
    exit 2
fi
if [[ ! -f $iron_env/bin/activate || ! -f $mlir_aie_root/utils/env_setup.sh ]]; then
    echo "Set IRON_ENV and MLIR_AIE_ROOT to installed IRON and mlir-aie paths" >&2
    exit 2
fi

shape="${M}x${K}x${N}"
if (( mode != 0 )); then shape="${shape}_m${mode}"; fi
out="$here/build/$shape"
mkdir -p -- "$out"
rm -rf -- "$out/final.prj"
source "$iron_env/bin/activate"
# mlir-aie's environment script appends to optional path variables that may
# be unset in a clean shell.
set +u
source "$mlir_aie_root/utils/env_setup.sh"
set -u
export BFP_KFLAGS=''
python "$here/designs/whole_array_bfp_rtp.py" -d npu2 --n-aie-cols 8 \
    -M "$M" -K "$K" -N "$N" -m 128 -k 64 -n 64 --c-tiled --out-mode "$mode" \
    --xclbin-path "$out/final.xclbin" --insts-path "$out/insts.bin"
echo "BUILD_OK $shape -> $out"
