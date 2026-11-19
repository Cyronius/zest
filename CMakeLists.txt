cmake_minimum_required(VERSION 3.20)
project(zest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prebuilt llama.cpp release we build against and run with. The headers come
# from the matching source tag; on Windows the import libs are generated from
# the release DLLs by tools/fetch-llama.ps1.
set(GGML_DIST "${CMAKE_SOURCE_DIR}/third_party/llama-b10944" CACHE PATH "llama.cpp release binaries")
set(GGML_HEADERS "${CMAKE_SOURCE_DIR}/third_party/ggml-b10944" CACHE PATH "matching ggml headers")
set(GGML_IMPLIB_DIR "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH "generated ggml import libs")

# Without the NPU path the backend runs its claimed matmuls on a CPU
# reference only, like GGML_XDNA_HOST_ONLY=1.
option(GGML_XDNA_NPU "build the NPU path" ON)

if(WIN32)
    set(GGML_LIBS "${GGML_IMPLIB_DIR}/ggml-base.lib" "${GGML_IMPLIB_DIR}/ggml.lib")
    set(EXE ".exe")
    set(XDNA_LIB "ggml-zest.dll")
    set(AVX2 /arch:AVX2)
else()
    find_library(GGML_BASE_LIB NAMES ggml-base PATHS "${GGML_DIST}" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIB NAMES ggml PATHS "${GGML_DIST}" NO_DEFAULT_PATH REQUIRED)
    set(GGML_LIBS "${GGML_BASE_LIB}" "${GGML_LIB}")
    set(EXE "")
    set(XDNA_LIB "libggml-zest.so")
    set(AVX2 -mavx2 -mfma)
    # everything we build runs next to llama.cpp's libraries and finds them there, as llama.cpp's own programs do
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    find_package(Threads REQUIRED)
endif()

add_library(ggml-zest SHARED
    src/ggml-zest.cpp
    src/xdna-ref.cpp
    src/xdna-platform.cpp
    include/ggml-zest.h
)

target_include_directories(ggml-zest PRIVATE
    include
    src
    "${GGML_HEADERS}/include"
    "${GGML_HEADERS}/src"
)

target_compile_definitions(ggml-zest PRIVATE
    GGML_SHARED            # import ggml core from ggml-base
    GGML_BACKEND_SHARED
    GGML_BACKEND_BUILD     # export ggml_backend_init / ggml_backend_score
    GGML_BACKEND_DL
    _CRT_SECURE_NO_WARNINGS
)

# ggml-base for the core, ggml for the device registry, to find the Vulkan device we work next to
target_link_libraries(ggml-zest PRIVATE ${GGML_LIBS})

if(WIN32)
    set_target_properties(ggml-zest PROPERTIES PREFIX "")
else()
    target_link_libraries(ggml-zest PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    # only ggml's entry points leave the library, so our stand-ins for the XRT wrapper's functions stay inside
    set_target_properties(ggml-zest PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# Drop the library next to the llama.cpp binaries so GGML_BACKEND_PATH is a
# one-liner and ggml-base resolves from the same directory. The NPU
# kernel goes next to the library in both places: the backend looks for it there.
set(XDNA_XCLBIN "${CMAKE_SOURCE_DIR}/kernels/bfp16_gemm/prebuilt/bfp16_gemm.xclbin")
set(XDNA_QXCLBIN "${CMAKE_SOURCE_DIR}/kernels/bfp16_gemm/prebuilt/bfpq_gemm.xclbin")
add_custom_command(TARGET ggml-zest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-zest> "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" $<TARGET_FILE_DIR:ggml-zest>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_QXCLBIN}" "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_QXCLBIN}" $<TARGET_FILE_DIR:ggml-zest>
)

# The NPU path. XRT is vendored: headers under vendor/xrt (Apache-2.0, see
# vendor/xrt/LICENSE) and our extern "C" shim over XRT's C++ API under
# vendor/xrt-shim. The build needs no NPU driver on either system, and the
# backend loads without one and says so.
if(GGML_XDNA_NPU)
    target_sources(ggml-zest PRIVATE
        src/xdna-npu.cpp
        src/xdna-exec.cpp
        src/npu/npu_bfp16.cpp
        src/npu/bfp16_pack.cpp
        src/npu/bfp16_insts.cpp
        src/npu/bfpq_insts.cpp
        src/npu/qtile_pack.cpp
    )
    target_include_directories(ggml-zest PRIVATE src/npu vendor/xrt-shim)
    target_compile_definitions(ggml-zest PRIVATE XDNA_HAVE_NPU)
    if(WIN32)
        # XRT ships no import lib on Windows, so one is built here from the
        # list of functions the backend imports (vendor/xrt-implib). The
        # driver's DLL is delay-loaded.
        set(XRT_DEF "${CMAKE_SOURCE_DIR}/vendor/xrt-implib/xrt_coreutil.def")
        set(XRT_LIB "${CMAKE_BINARY_DIR}/xrt_coreutil.lib")
        add_custom_command(OUTPUT "${XRT_LIB}"
            COMMAND "${CMAKE_AR}" /nologo "/def:${XRT_DEF}" /machine:x64 "/out:${XRT_LIB}"
            DEPENDS "${XRT_DEF}"
            VERBATIM
        )
        add_custom_target(xrt-implib DEPENDS "${XRT_LIB}")
        add_dependencies(ggml-zest xrt-implib)
        target_sources(ggml-zest PRIVATE vendor/xrt-shim/xrt_shim.cpp)
        target_include_directories(ggml-zest PRIVATE vendor/xrt/include)
        # /Zc:__cplusplus: XRT's headers pick std::any over boost::any only when
        # __cplusplus is reported honestly. /arch:AVX2 /fp:fast: the host's small
        # ops vectorize.
        target_compile_options(ggml-zest PRIVATE /Zc:__cplusplus /arch:AVX2 /fp:fast)
        target_link_libraries(ggml-zest PRIVATE "${XRT_LIB}" delayimp)
        target_link_options(ggml-zest PRIVATE /DELAYLOAD:xrt_coreutil.dll)
    else()
        target_sources(ggml-zest PRIVATE src/npu/xrt_load.cpp)
        # fast math that still has NaN and infinity: the check on the NPU's results looks for them
        target_compile_options(ggml-zest PRIVATE ${AVX2} -ffast-math -fno-finite-math-only)
        # The shim as a library of its own, unlinked to XRT: the backend loads XRT, then this (src/npu/xrt_load.cpp).
        add_library(ggml-zest-xrt SHARED vendor/xrt-shim/xrt_shim.cpp)
        target_include_directories(ggml-zest-xrt PRIVATE vendor/xrt-shim vendor/xrt/include)
        find_library(UUID_LIB uuid REQUIRED)  # XRT's headers use libuuid on Linux (uuid-dev)
        target_link_libraries(ggml-zest-xrt PRIVATE "${UUID_LIB}")
        add_dependencies(ggml-zest ggml-zest-xrt)
        add_custom_command(TARGET ggml-zest-xrt POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-zest-xrt> "${GGML_DIST}"
        )
    endif()

    # A standalone repro of the NPU's buffers going bad past its memory ceiling: the NPU layer and the kernel only. Not shipped.
    add_executable(npu-ceiling-repro
        tools/npu-ceiling-repro.cpp
        src/xdna-platform.cpp
        src/npu/npu_bfp16.cpp
        src/npu/bfp16_pack.cpp
        src/npu/bfp16_insts.cpp
    )
    target_include_directories(npu-ceiling-repro PRIVATE src src/npu vendor/xrt-shim)
    if(WIN32)
        target_sources(npu-ceiling-repro PRIVATE vendor/xrt-shim/xrt_shim.cpp)
        target_include_directories(npu-ceiling-repro PRIVATE vendor/xrt/include)
        target_compile_options(npu-ceiling-repro PRIVATE /Zc:__cplusplus ${AVX2})
        target_link_libraries(npu-ceiling-repro PRIVATE "${XRT_LIB}" pdh)
        add_dependencies(npu-ceiling-repro xrt-implib)
    else()
        target_sources(npu-ceiling-repro PRIVATE src/npu/xrt_load.cpp)
        target_compile_options(npu-ceiling-repro PRIVATE ${AVX2})
        target_link_libraries(npu-ceiling-repro PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(npu-ceiling-repro ggml-zest-xrt)
    endif()
    # next to the kernel, and on Linux to the XRT wrapper
    add_custom_command(TARGET npu-ceiling-repro POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:npu-ceiling-repro> "${GGML_DIST}"
    )
    if(NOT WIN32)
        add_executable(test-npu-wide
            tests/test-npu-wide.cpp
            src/xdna-platform.cpp
            src/npu/npu_bfp16.cpp
            src/npu/bfp16_pack.cpp
            src/npu/bfp16_insts.cpp
            src/npu/xrt_load.cpp
        )
        target_include_directories(test-npu-wide PRIVATE src src/npu vendor/xrt-shim)
        target_compile_options(test-npu-wide PRIVATE ${AVX2})
        target_link_libraries(test-npu-wide PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-npu-wide ggml-zest-xrt)

        # GEMM times at Qwen3.8-27B's shapes against math and bytes moved (not a test)
        add_executable(bench-npu-gemm
            tools/bench-npu-gemm.cpp
            src/xdna-platform.cpp
            src/npu/npu_bfp16.cpp
            src/npu/bfp16_pack.cpp
            src/npu/bfp16_insts.cpp
            src/npu/xrt_load.cpp
        )
        target_include_directories(bench-npu-gemm PRIVATE src src/npu vendor/xrt-shim)
        target_compile_options(bench-npu-gemm PRIVATE ${AVX2})
        target_link_libraries(bench-npu-gemm PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(bench-npu-gemm ggml-zest-xrt)

        # One real GGUF projection split by output columns across Vulkan and
        # XDNA. Diagnostic benchmark only; it does not change inference.
        add_executable(bench-hybrid-projection tools/bench-hybrid-projection.cpp)
        target_include_directories(bench-hybrid-projection PRIVATE "${GGML_HEADERS}/include")
        target_compile_definitions(bench-hybrid-projection PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
        target_link_libraries(bench-hybrid-projection PRIVATE ${GGML_LIBS} Threads::Threads)
        add_custom_command(TARGET bench-hybrid-projection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:bench-hybrid-projection> "${GGML_DIST}"
        )
    endif()
endif()

# The launcher: `zest llama-server ...` with nothing to set (XDNA-LAUNCHER).
# It goes next to the llama.cpp programs it runs.
add_executable(zest tools/zest.cpp)
if(WIN32)
    target_compile_definitions(zest PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(zest PRIVATE /utf-8)
    target_link_libraries(zest PRIVATE shell32)
endif()
add_custom_command(TARGET zest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:zest> "${GGML_DIST}"
    # Both public command names run the same launcher.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:zest> "$<TARGET_FILE_DIR:zest>/npu${EXE}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:zest> "${GGML_DIST}/npu${EXE}"
)

#
# tests
#

enable_testing()

foreach(t test-mul-mat test-dispatch-gate test-memory-budget test-safe-start test-host-ops test-ffn-block)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE "${GGML_HEADERS}/include")
    target_compile_definitions(${t} PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${t} PRIVATE ${GGML_LIBS})
    # they run next to the llama.cpp libraries
    add_custom_command(TARGET ${t} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${t}> "${GGML_DIST}"
    )
endforeach()

# The NPU instruction streams made on the host, against the toolchain's (no
# NPU or XRT needed).
add_executable(test-insts-gen tests/test-insts-gen.cpp src/npu/bfp16_insts.cpp src/npu/bfpq_insts.cpp src/npu/bfp16_pack.cpp)
target_include_directories(test-insts-gen PRIVATE src/npu)
target_compile_definitions(test-insts-gen PRIVATE XDNA_INSTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/insts")

# The weight packer against the kernel's verified layout (pure host code).
add_executable(test-bfp16-pack tests/test-bfp16-pack.cpp src/npu/bfp16_pack.cpp)
target_include_directories(test-bfp16-pack PRIVATE src/npu)
target_compile_options(test-bfp16-pack PRIVATE ${AVX2})

# The Q4_K re-layout for the NPU's unpacker, against ggml's dequantization
# (host code; runs next to the llama.cpp libraries for ggml).
add_executable(test-qtile-pack tests/test-qtile-pack.cpp src/npu/qtile_pack.cpp src/npu/bfp16_pack.cpp)
target_include_directories(test-qtile-pack PRIVATE src/npu "${GGML_HEADERS}/include")
target_compile_definitions(test-qtile-pack PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
target_compile_options(test-qtile-pack PRIVATE ${AVX2})
target_link_libraries(test-qtile-pack PRIVATE ${GGML_LIBS})
add_custom_command(TARGET test-qtile-pack POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:test-qtile-pack> "${GGML_DIST}"
)

# ctest labels: "host" needs no NPU (the backend-facing ones do need a Vulkan
# device), "npu" needs the NPU, "nodriver" expects a machine without the NPU
# driver (CI).
file(TO_NATIVE_PATH "${GGML_DIST}/${XDNA_LIB}" XDNA_PATH)
set(XDNA_ENV "GGML_BACKEND_PATH=${XDNA_PATH}" "GGML_XDNA_KERNELS=")
add_test(NAME insts-gen COMMAND test-insts-gen)
add_test(NAME bfp16-pack COMMAND test-bfp16-pack)
add_test(NAME qtile-pack COMMAND "${GGML_DIST}/test-qtile-pack${EXE}")
if(TARGET test-npu-wide)
    add_test(NAME npu-wide COMMAND test-npu-wide "${XDNA_XCLBIN}")
    set_tests_properties(npu-wide PROPERTIES LABELS npu)
endif()
add_test(NAME safe-start COMMAND "${GGML_DIST}/test-safe-start${EXE}")
add_test(NAME safe-start-no-driver COMMAND "${GGML_DIST}/test-safe-start${EXE}" --no-driver)
foreach(t dispatch-gate mul-mat)
    add_test(NAME ${t}-host COMMAND "${GGML_DIST}/test-${t}${EXE}")
    add_test(NAME ${t}-npu COMMAND "${GGML_DIST}/test-${t}${EXE}")
    set_tests_properties(${t}-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=1")
    set_tests_properties(${t}-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=")
endforeach()
# every NPU result comes back as NaN: the backend has to catch it and still
# give the right answer
add_test(NAME mul-mat-npu-nan COMMAND "${GGML_DIST}/test-mul-mat${EXE}")
set_tests_properties(mul-mat-npu-nan PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_NAN_AFTER=0")
# every weight under the quantized-weight kernel (Q4_K unpacked on the NPU,
# the rest copied through): the same answers as the BFP16 kernel's
add_test(NAME mul-mat-npu-quant COMMAND "${GGML_DIST}/test-mul-mat${EXE}")
set_tests_properties(mul-mat-npu-quant PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_WEIGHTS=quant")
# every NPU submission fails: the same, through the other way the NPU fails
add_test(NAME mul-mat-npu-fail COMMAND "${GGML_DIST}/test-mul-mat${EXE}")
set_tests_properties(mul-mat-npu-fail PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_FAIL_AFTER=0")
add_test(NAME memory-budget-host COMMAND "${GGML_DIST}/test-memory-budget${EXE}")
add_test(NAME memory-budget-npu COMMAND "${GGML_DIST}/test-memory-budget${EXE}")
set(XDNA_BUDGET_ENV "${XDNA_ENV};GGML_XDNA_MAX_COPY_GB=0.05")
set_tests_properties(memory-budget-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=1")
set_tests_properties(memory-budget-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=")
# the same limit with every copy for the quantized-weight kernel
add_test(NAME memory-budget-npu-quant COMMAND "${GGML_DIST}/test-memory-budget${EXE}")
set_tests_properties(memory-budget-npu-quant PROPERTIES LABELS npu ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_WEIGHTS=quant")
# A whole feed-forward block on the NPU: gate/up fused and the down multiply
# chained behind them; the same with a host pass between (and the same bits);
# under the quantized-weight kernel, which doesn't chain; and on the host.
add_test(NAME ffn-block-host COMMAND "${GGML_DIST}/test-ffn-block${EXE}")
set_tests_properties(ffn-block-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=1")
add_test(NAME ffn-block-npu COMMAND "${GGML_DIST}/test-ffn-block${EXE}")
set_tests_properties(ffn-block-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=")
add_test(NAME ffn-block-npu-quant COMMAND "${GGML_DIST}/test-ffn-block${EXE}")
set_tests_properties(ffn-block-npu-quant PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_WEIGHTS=quant")
add_test(NAME ffn-chain-same-bits COMMAND ${CMAKE_COMMAND} "-DEXE=${GGML_DIST}/test-ffn-block${EXE}"
    "-DENV_BOTH=${XDNA_ENV};GGML_XDNA_HOST_ONLY=" "-DENV_A=GGML_XDNA_CHAIN=1" "-DENV_B=GGML_XDNA_CHAIN=0"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/same-checksum.cmake")
set_tests_properties(ffn-chain-same-bits PROPERTIES LABELS npu)
# the host's small steps between NPU matmuls, against ggml's CPU backend
add_test(NAME host-ops-host COMMAND "${GGML_DIST}/test-host-ops${EXE}")
set_tests_properties(host-ops-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=1")
set_tests_properties(insts-gen bfp16-pack qtile-pack PROPERTIES LABELS host)
set_tests_properties(safe-start PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV}")
set_tests_properties(safe-start-no-driver PROPERTIES LABELS nodriver ENVIRONMENT "${XDNA_ENV}")

# Exercise the public commands with their own backend discovery, including the
# npu command. Direct backend tests above set GGML_BACKEND_PATH.
add_test(NAME launcher-devices COMMAND "${GGML_DIST}/zest${EXE}" llama-server --list-devices)
add_test(NAME launcher-npu-devices COMMAND "${GGML_DIST}/npu${EXE}" llama-server --list-devices)
set_tests_properties(launcher-devices launcher-npu-devices PROPERTIES
    LABELS host
    ENVIRONMENT "GGML_BACKEND_PATH=;GGML_XDNA_KERNELS="
    PASS_REGULAR_EXPRESSION "Vulkan0:"
)
