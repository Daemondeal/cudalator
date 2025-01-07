#pragma once

#include <cstdlib>
#include <spdlog/spdlog.h>

#if defined(__GNUC__) || defined(__clang__)
#define CD_CRASH() __builtin_trap()
#else
#define CD_CRASH() ::std::abort()
#endif

#define CD_UNREACHABLE(...)                                                    \
    do {                                                                       \
        spdlog::error("Unreachable path reached: ({}, line {})", __FILE__,     \
                      __LINE__);                                               \
        spdlog::error(__VA_ARGS__);                                            \
        CD_CRASH();                                                            \
    } while (0);

#define CD_ASSERT_MSG(condition, ...)                                          \
    do {                                                                       \
        if (!(condition)) {                                                    \
            spdlog::error("Assertion Failed: \"{}\" ({}, line {})",            \
                          #condition, __FILE__, __LINE__);                     \
            spdlog::error(__VA_ARGS__);                                        \
            CD_CRASH();                                                        \
        }                                                                      \
    } while (0);

#define CD_ASSERT(condition)                                                   \
    do {                                                                       \
        if (!(condition)) {                                                    \
            spdlog::error("Assertion Failed: \"{}\" ({}, line {})",            \
                          #condition, __FILE__, __LINE__);                     \
            CD_CRASH();                                                        \
        }                                                                      \
    } while (0);

#define CD_ASSERT_NONNULL(ptr)                                                 \
    do {                                                                       \
        if ((ptr) == nullptr) {                                                \
            spdlog::error("Assertion Failed: \"{} is null\" ({}, line {})",    \
                          #ptr, __FILE__, __LINE__);                           \
            CD_CRASH();                                                        \
        }                                                                      \
    } while (0);
