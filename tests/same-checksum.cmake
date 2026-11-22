# Runs a test program twice, under two settings, and fails unless both pass
# and print the same "checksum <hex>": for settings that must not change a
# result's bits.
#   cmake -DEXE=<program> "-DENV_BOTH=<A=1;B=2>" "-DENV_A=<...>" "-DENV_B=<...>" -P same-checksum.cmake
foreach(run A B)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${ENV_BOTH} ${ENV_${run}} ${EXE}
        OUTPUT_VARIABLE out
        RESULT_VARIABLE code
    )
    message("${out}")
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "the run under \"${ENV_${run}}\" failed (${code})")
    endif()
    if(NOT out MATCHES "checksum ([0-9a-f]+)")
        message(FATAL_ERROR "the run under \"${ENV_${run}}\" printed no checksum")
    endif()
    set(sum_${run} ${CMAKE_MATCH_1})
endforeach()
if(NOT sum_A STREQUAL sum_B)
    message(FATAL_ERROR "the results differ: ${sum_A} under \"${ENV_A}\", ${sum_B} under \"${ENV_B}\"")
endif()
message("same result under both: ${sum_A}")
