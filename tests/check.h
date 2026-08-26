// Test assertion macro that survives -DNDEBUG (unlike the standard <cassert>).
// Use this in tests in place of `assert(...)`. Failing a check prints a clear
// diagnostic and increments a per-binary failure counter; the binary returns
// non-zero exit on any failure.
//
// Usage:
//   INSPIRE_CHECK(cond);                   // bare boolean
//   INSPIRE_CHECK(cond, "expected X = Y"); // with message
//   return inspire_test_status();          // at end of main(): non-zero if any failed
#pragma once

#include <cstdio>
#include <string>
#include <atomic>

inline std::atomic<int>& inspire_test_failures() {
    static std::atomic<int> count{0};
    return count;
}

inline void inspire_test_record_fail(const char* file, int line,
                                     const char* expr, const std::string& msg) {
    int n = ++inspire_test_failures();
    std::fprintf(stderr, "  [CHECK FAIL #%d] %s:%d   %s%s%s\n",
                 n, file, line, expr,
                 msg.empty() ? "" : "   -- ",
                 msg.c_str());
}

// One-shot status: returns 0 if every INSPIRE_CHECK passed in this process,
// non-zero otherwise. Call at the end of `main()` and return its result.
inline int inspire_test_status() {
    int n = inspire_test_failures().load();
    if (n == 0) {
        std::fprintf(stderr, "  [OK] all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "  [FAIL] %d check(s) failed\n", n);
    return 1;
}

#define INSPIRE_CHECK_1(cond)              \
    do {                                   \
        if (!(cond)) {                     \
            inspire_test_record_fail(      \
                __FILE__, __LINE__, #cond, std::string()); \
        }                                  \
    } while (0)

#define INSPIRE_CHECK_2(cond, msg)         \
    do {                                   \
        if (!(cond)) {                     \
            inspire_test_record_fail(      \
                __FILE__, __LINE__, #cond, std::string(msg)); \
        }                                  \
    } while (0)

// Overload-style dispatch on argument count
#define INSPIRE_CHECK_GET(_1, _2, NAME, ...) NAME
#define INSPIRE_CHECK(...) \
    INSPIRE_CHECK_GET(__VA_ARGS__, INSPIRE_CHECK_2, INSPIRE_CHECK_1)(__VA_ARGS__)
