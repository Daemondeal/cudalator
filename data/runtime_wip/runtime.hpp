#include <array>
#include <cstdint>
#include <limits>
#include <string>

/**
 * List of operators we must implement:
 * - ASSIGNMENT OPERATORS
 *   =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=, <<<=, >>>=
 * - CONDITIONAL OPERATORS
 *   cond_predicate ? {attribute_instance} expression : expression
 * - UNARY OPERATORS
 *   +, -, !, ~, &, ~&, |, ~|, ^, ~^, ^~
 * - BINARY OPERATORS
 *   +, -, *, /, %, ==, !=, ===, !==, ==?, !=?, &&, ||. **, <, <=, >, >=, &, |,
 * ^, ^~, ~^, >>, <<. >>>, <<<, ->, <->
 * - INCREMENT OR DECREMENT OPERATORS
 *   ++, --
 * - STREAM OPERATORS
 *   >>, <<
 */

template <int N>
class Bit {
    uint32_t ciao;
    Bit() = default;
};
