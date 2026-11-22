// The row split's default follows the GPU, and ZEST_SPLIT overrides it.
// Traces: XDNA-SPLIT-DEFAULT, XDNA-ENV-NAMES

#include "xdna-env.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;

static void expect(const char * value, const char * gpu, int64_t want) {
    const int64_t got = xdna_split_choice(value, gpu);
    if (got != want) {
        printf("FAIL: ZEST_SPLIT=%s on \"%s\": %lld, expected %lld\n", value ? value : "(unset)", gpu ? gpu : "(null)",
               (long long) got, (long long) want);
        failures++;
    }
}

int main() {
    expect(nullptr, "AMD Radeon(TM) 890M Graphics", 0);
    expect(nullptr, "AMD Radeon 890M Graphics (RADV STRIX1)", 0);
    expect(nullptr, "AMD Radeon(TM) 880M Graphics", 0);
    expect(nullptr, "AMD Radeon(TM) 860M Graphics", 0);
    expect("", "AMD Radeon(TM) 840M Graphics", 0);
    expect(nullptr, "AMD Radeon(TM) 8060S Graphics", 50);
    expect(nullptr, "AMD Radeon 8060S Graphics (RADV GFX1151)", 50);
    expect(nullptr, "AMD Radeon RX 7900 XTX", 50);
    expect(nullptr, "AMD Radeon(TM) 780M Graphics", 50);
    expect(nullptr, "", 50);
    expect(nullptr, nullptr, 50);

    expect("50", "AMD Radeon(TM) 890M Graphics", 50);
    expect("on", "AMD Radeon(TM) 890M Graphics", 50);
    expect("ON", "AMD Radeon(TM) 890M Graphics", 50);
    expect("25", "AMD Radeon(TM) 890M Graphics", 25);
    expect("0", "AMD Radeon(TM) 8060S Graphics", 0);
    expect("off", "AMD Radeon(TM) 8060S Graphics", 0);
    expect("150", nullptr, 100);
    expect("-5", nullptr, 0);

    // an old GGML_XDNA_* name still reads, the new name winning when both are set
#ifdef _WIN32
    _putenv("ZEST_TEST_OLD=");
    _putenv("GGML_XDNA_TEST_OLD=7");
#else
    unsetenv("ZEST_TEST_OLD");
    setenv("GGML_XDNA_TEST_OLD", "7", 1);
#endif
    if (xdna_env_int("ZEST_TEST_OLD", 0) != 7) { printf("FAIL: the old name was not read\n"); failures++; }
#ifdef _WIN32
    _putenv("ZEST_TEST_OLD=9");
#else
    setenv("ZEST_TEST_OLD", "9", 1);
#endif
    if (xdna_env_int("ZEST_TEST_OLD", 0) != 9) { printf("FAIL: the new name did not win\n"); failures++; }

    if (failures) return 1;
    printf("split-default: all passed\n");
    return 0;
}
