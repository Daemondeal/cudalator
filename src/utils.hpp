#pragma once

#include <spdlog/spdlog.h>

// TODO: This needs to be implemented better
#define CD_ASSERT(condition, message)                                          \
    do {                                                                       \
        if (!(condition)) {                                                    \
            spdlog::error("{}", message);                                      \
            exit(-1);                                                          \
        }                                                                      \
    } while (0);
