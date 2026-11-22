# Runs a test program and fails unless it passes and its output (stdout and
# stderr together) has EXPECT in it: for a path the program can't tell from its
# results, such as work shared between two devices, which gives the same
# answer when it isn't shared.
#   cmake -DEXE=<program> "-DRUN_ENV=<A=1;B=2>" "-DEXPECT=<regex>" -P expect-output.cmake
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ${RUN_ENV} ${EXE}
    OUTPUT_VARIABLE out
    ERROR_VARIABLE out
    RESULT_VARIABLE code
)
message("${out}")
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the run failed (${code})")
endif()
if(NOT out MATCHES "${EXPECT}")
    message(FATAL_ERROR "the output has no \"${EXPECT}\"")
endif()
