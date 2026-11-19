#!/usr/bin/env bash
# Rebuild the shipped 8-column BFP16 kernel on Linux. Run this under
# `dibs run build -- ...` on a shared machine.
#
# IRON_ENV points to a Python venv with mlir_aie 1.4.3 and Peano installed;
# MLIR_AIE_ROOT points to the matching mlir-aie checkout. Their defaults are
# the ignored local development copies under third_party/.
# The design is designs/whole_array_bfp_colwise.py: core tile m x k x n =
# 64 x 64 x 128, M a multiple of 512, N of 512, K of 64 (src/npu/bfp16_pack.h's
# BFP16_TILE and src/npu/bfp16_insts.cpp must match it). One build's xclbin
# serves every shape and output mode; the shape and mode given here only set
# the instruction stream written beside it (tests/insts keeps some, to check
# the host's streams against). TILE_M, TILE_K and TILE_N build another tile,
# for measurements. BF16_SUMS=1 keeps bf16 running sums in output modes 1 to 3
# (--bf16-sums). DESIGN=q4k builds the GEMM on Q4_K weights unpacked on the
# array instead (designs/whole_array_bfpq.py: tile 128 x 64 x 64, M a multiple
# of 384, K of 256; src/npu/bfpq_insts.cpp makes its streams), into
# build/q4k_<shape>; FMT=1 gives its streams for B tiles copied through (the
# xclbin is the same).
# Usage: build-linux.sh M K N [out-mode]
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
iron_env=${IRON_ENV:-$root/third_party/ironenv}
mlir_aie_root=${MLIR_AIE_ROOT:-$root/third_party/mlir-aie}

if (( $# < 3 || $# > 4 )); then
    echo "usage: $0 M K N [out-mode: 0 to 3]" >&2
    exit 2
fi
M=$1 K=$2 N=$3 mode=${4:-0}
set --  # env_setup.sh must see no inherited shape arguments
if [[ ! $M =~ ^[1-9][0-9]*$ || ! $K =~ ^[1-9][0-9]*$ || ! $N =~ ^[1-9][0-9]*$ || ! $mode =~ ^[0123]$ ]]; then
    echo "M, K, N must be positive integers and out-mode must be 0 to 3" >&2
    exit 2
fi
if [[ ! -f $iron_env/bin/activate || ! -f $mlir_aie_root/utils/env_setup.sh ]]; then
    echo "Set IRON_ENV and MLIR_AIE_ROOT to installed IRON and mlir-aie paths" >&2
    exit 2
fi

design=${DESIGN:-bfp16}
if [[ $design == bfp16 ]]; then
    tm=${TILE_M:-64} tk=${TILE_K:-64} tn=${TILE_N:-128}
    default_tile=64/64/128
elif [[ $design == q4k ]]; then
    tm=${TILE_M:-128} tk=${TILE_K:-64} tn=${TILE_N:-64}
    default_tile=128/64/64
    if [[ $tm/$tk/$tn != "$default_tile" || $mode == 3 ]]; then
        echo "DESIGN=q4k has the 128 x 64 x 64 tile and output modes 0 to 2 only" >&2
        exit 2
    fi
else
    echo "DESIGN must be bfp16 or q4k" >&2
    exit 2
fi
shape="${M}x${K}x${N}"
if (( mode != 0 )); then shape="${shape}_m${mode}"; fi
if [[ $tm/$tk/$tn != "$default_tile" ]]; then shape="${shape}_t${tm}x${tk}x${tn}"; fi
sums=()
if [[ ${BF16_SUMS:-0} == 1 ]]; then shape="${shape}_s16"; sums=(--bf16-sums); fi
if [[ $design == q4k ]]; then
    if [[ ${FMT:-0} == 1 ]]; then shape="qcopy_$shape"; else shape="q4k_$shape"; fi
fi
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
if [[ $design == q4k ]]; then
    python "$here/designs/whole_array_bfpq.py" -d npu2 \
        -M "$M" -K "$K" -N "$N" --out-mode "$mode" --fmt "${FMT:-0}" "${sums[@]}" \
        --xclbin-path "$out/final.xclbin" --insts-path "$out/insts.bin"
else
    python "$here/designs/whole_array_bfp_colwise.py" -d npu2 \
        -M "$M" -K "$K" -N "$N" -m "$tm" -k "$tk" -n "$tn" --out-mode "$mode" "${sums[@]}" \
        --xclbin-path "$out/final.xclbin" --insts-path "$out/insts.bin"
    # the streams' register writes and the host's streams assume a placement
    python "$here/designs/check_placement.py" "$out/final.prj"
fi
echo "BUILD_OK $shape -> $out"
