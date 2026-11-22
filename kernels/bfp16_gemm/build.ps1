# Builds the bfp16 GEMM kernel at one size into
# build\<M>x<K>x<N>[_m<mode>]\{final.xclbin,insts.bin}.
#   .\build.ps1 -M 512 -K 2048 -N 4096 [-OutMode 1|2|3] [-IronEnv <path>]
#
# The backend doesn't need this: final.xclbin is the same core program at
# every size (prebuilt\bfp16_gemm.xclbin), and the backend makes each size's
# insts.bin itself. It's here to rebuild that program, and to make reference
# insts.bin files for tests\insts (XDNA-INSTS-GEN).
#
# Needs AMD's IRON toolchain (mlir-aie with the Peano compiler), set up by its
# iron_env.ps1. The design is designs\whole_array_bfp_colwise.py, tiled 64 x 64
# x 128: the settings the shipped xclbin was built with. After the build,
# check_placement.py checks that the toolchain put the design's streams where
# the instruction streams take them to be.
param(
    [Parameter(Mandatory)][int]$M, [Parameter(Mandatory)][int]$K, [Parameter(Mandatory)][int]$N,
    [ValidateSet(0, 1, 2, 3)][int]$OutMode = 0,
    [string]$IronEnv = 'C:\dev\mlir-aie\iron_env.ps1'
)
$ErrorActionPreference = 'Stop'
. $IronEnv
$env:BFP_KFLAGS = ''  # the design adds these to the kernel's defines; the shipped build has none
$here = $PSScriptRoot
$shape = "${M}x${K}x${N}"
if ($OutMode) { $shape += "_m$OutMode" }
$out = Join-Path $here "build\$shape"
New-Item -ItemType Directory -Force -Path $out | Out-Null
# aiecc keeps kernel objects in <xclbin>.prj and skips recompiling them; start clean.
Remove-Item -Recurse -Force (Join-Path $out 'final.prj') -ErrorAction SilentlyContinue
python (Join-Path $here 'designs\whole_array_bfp_colwise.py') -d npu2 `
    -M $M -K $K -N $N -m 64 -k 64 -n 128 --out-mode $OutMode `
    --xclbin-path (Join-Path $out 'final.xclbin') --insts-path (Join-Path $out 'insts.bin')
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD_FAIL $shape"; exit 1 }
python (Join-Path $here 'designs\check_placement.py') (Join-Path $out 'final.prj')
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD_FAIL $shape (placement)"; exit 1 }
Write-Host "BUILD_OK $shape -> $out"
