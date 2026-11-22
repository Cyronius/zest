// The backend's settings, from ZEST_* environment variables. An empty value
// counts as unset: `set ZEST_FOO=` on Windows, or `FOO= cmd` in a shell, is how
// people clear one.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

inline const char * xdna_env_raw(const char * name) {
    const char * s = getenv(name);
    return s && *s ? s : nullptr;
}

// The settings were named GGML_XDNA_* before 2026-11-22: an old name still works, with a warning.
inline const char * xdna_env(const char * name) {
    if (const char * s = xdna_env_raw(name)) return s;
    if (strncmp(name, "ZEST_", 5) != 0) return nullptr;
    const std::string old = std::string("GGML_XDNA_") + (name + 5);
    const char * s = xdna_env_raw(old.c_str());
    if (s) {
        static std::mutex mu;
        static std::set<std::string> warned;
        std::lock_guard<std::mutex> lk(mu);
        if (warned.insert(old).second)
            fprintf(stderr, "zest: %s is now named %s; the old name still works for now\n", old.c_str(), name);
    }
    return s;
}

inline int64_t xdna_env_int(const char * name, int64_t def) {
    const char * s = xdna_env(name);
    return s ? atoll(s) : def;
}

// The row split's default from the GPU's description. Radeon 800M laptop
// graphics (Strix Point, Krackan Point) are small enough that the NPU ends up
// waiting on the GPU's half, so the split is off there.
inline int64_t xdna_split_default(const char * gpu_description) {
    if (!gpu_description) return 50;
    for (const char * p = gpu_description; p[0] && p[1] && p[2] && p[3]; p++)
        if (p[0] == '8' && isdigit((unsigned char) p[1]) && p[2] == '0' && p[3] == 'M' &&
            (p == gpu_description || !isalnum((unsigned char) p[-1])) && !isalnum((unsigned char) p[4]))
            return 0;
    return 50;
}

// ZEST_SPLIT: "off" or 0 turns the split off, "on" gives the NPU half the rows,
// a number is the NPU's share in percent; unset follows the GPU.
inline int64_t xdna_split_choice(const char * value, const char * gpu_description) {
    if (!value || !*value) return xdna_split_default(gpu_description);
    std::string v(value);
    for (auto & c : v) c = (char) tolower((unsigned char) c);
    if (v == "off" || v == "no" || v == "false") return 0;
    if (v == "on" || v == "yes" || v == "true") return 50;
    const long long n = atoll(value);
    return n < 0 ? 0 : n > 100 ? 100 : n;
}
