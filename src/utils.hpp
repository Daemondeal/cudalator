#pragma once

#include <cstdlib>
#include <spdlog/spdlog.h>

#if defined(__GNUC__) || defined(__clang__)
#define CD_CRASH() __builtin_trap()
#else
#define CD_CRASH() ::std::abort()
#endif

#define CD_ASSERT_MSG(condition, ...)                                          \
    do {                                                                       \
        if (!(condition)) {                                                    \
            spdlog::error("Assertion Failed: \"{}\" (%s, line %d)",            \
                          #condition, __FILE__, __LINE__);                     \
            spdlog::error(__VA_ARGS__);                                        \
            CD_CRASH();                                                        \
        }                                                                      \
    } while (0);

#define CD_ASSERT(condition)                                                   \
    do {                                                                       \
        if (!(condition)) {                                                    \
            spdlog::error("Assertion Failed: \"{}\" (%s, line %d)",            \
                          #condition, __FILE__, __LINE__);                     \
            CD_CRASH();                                                        \
        }                                                                      \
    } while (0);

#include <memory>
#include <string>
#include <stdexcept>

template<typename ... Args>
static std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}
