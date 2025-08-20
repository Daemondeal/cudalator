#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include <algorithm> // For std::max
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iomanip>   // For std::hex, std::setw, std::setfill
#include <sstream>   // For std::stringstream
#include <stdexcept> // For std::invalid_argument
#include <string>
#include <type_traits> // For std::is_integral_v and std::enable_if_t

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

/**
 * operator bool() fa funzionare il tipo come se fosse un booleano quindi se
 * scrivi if(oggetto) ti restituisce true o false
 */
template <int N>
class Bit {
    static_assert(N > 0 && N <= 128, "The maximum supported bit width is 128");

    static constexpr int max(int a, int b) { return a > b ? a : b; }

    template <int M>
    friend class Bit;

public:
    /**
     * @brief Default constructor.
     * Creates a Bit vector zero-initialized
     */
    Bit() = default;

    /**
     * @brief Constructor from a single integral value.
     */
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit Bit(T value) {
        set_value(value);
    }

    /**
     * @brief Constructor from a list of 32-bit chunks.
     * For chunks > 64 bits. Chunks from least significant to most
     * significant.
     * Example: Bit<96> a = {0xFFFFFFFF, 0x0000FFFF,
     * 0x00000001};
     */
    Bit(std::initializer_list<uint32_t> init) { copy_from_init(init); }

    /**
     * @brief Constructor from a hexadecimal string literal
     */
    explicit Bit(const char* hex_string) { parse_hex_string(hex_string); }

    /**
     * @brief Converting constructor from another Bit vector
     * This allows initialization from a Bit vectorof a different width, like
     * Bit<8> result = Bit<9>(...)
     */
    template <int M>
    Bit(const Bit<M>& rhs) {
        // reusing the assignment operator
        *this = rhs;
    }

    /**
     * @brief Assignment from another Bit vector.
     * Handles assignment from both same-sized and different-sized Bit
     * vectors. Truncates if the source is larger, zero-extends if it is
     * smaller.
     */
    template <int M>
    Bit& operator=(const Bit<M>& rhs) {
        // Clear old data since we don't need it anymore
        chunks.fill(0);

        // computing the smaller of the two chunk counts
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int chunks_to_copy =
            num_chunks < rhs_chunks ? num_chunks : rhs_chunks;

        for (int i = 0; i < chunks_to_copy; ++i) {
            chunks[i] = rhs.chunks[i];
        }

        apply_mask();
        // returning a reference to this object to allow chaining like a = b = c
        return *this;
    }

    /**
     * @brief Assignment from a single integral value.
     */
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    Bit& operator=(T value) {
        set_value(value);
        return *this;
    }

    /**
     * @brief Equality comparison operator.
     */
    template <int M>
    Bit<1> operator==(const Bit<M>& rhs) const {
        constexpr int lhs_chunks = num_chunks;
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int max_chunks =
            (lhs_chunks > rhs_chunks) ? lhs_chunks : rhs_chunks;

        for (int i = 0; i < max_chunks; ++i) {
            uint32_t lhs_chunk = (i < lhs_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            if (lhs_chunk != rhs_chunk) {
                return Bit<1>(0);
            }
        }
        return Bit<1>(1); // all chunks were identical, return true
    }

    /**
     * @brief Inequality comparison operator
     */
    template <int M>
    Bit<1> operator!=(const Bit<M>& rhs) const {
        return !(*this == rhs);
    }

    /**
     * @brief Addition operator.
     * Adds two Bit vectors, returning a new vector with the result.
     * The result is one bit wider to accommodate a carry-out, UNLESS
     * that would exceed the 128-bit limit, in which case the carry is
     * discarded.
     */
    template <int M>
    auto operator+(const Bit<M>& rhs) const
        -> Bit<max(N, M) < 128 ? max(N, M) + 1 : 128> {
        // computing the result width
        constexpr int RESULT_BITS = (max(N, M) < 128) ? (max(N, M) + 1) : 128;
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        uint64_t carry = 0;
        // Looping through the maximum number of chunks needed for the result
        for (int i = 0; i < result_chunks; ++i) {
            uint64_t lhs_val = (i < num_chunks) ? chunks[i] : 0;
            uint64_t rhs_val = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            uint64_t sum = lhs_val + rhs_val + carry;
            // the lower 32 bits of the sum are the chunk for our result
            result.chunks[i] = static_cast<uint32_t>(sum);
            // right shift to get the eventual carry bit
            carry = sum >> 32;
        }
        result.apply_mask();
        return result;
    }

    /**
     * @brief Subtraction operator.
     * Subtracts two Bit vectors, returning a new vector with the result.
     * The result width is the same as the wider of the two operands.
     */
    template <int M>
    auto operator-(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        uint64_t borrow = 0;

        for (int i = 0; i < result_chunks; ++i) {
            uint64_t lhs_val = (i < num_chunks) ? chunks[i] : 0;
            uint64_t rhs_val = (i < rhs_chunks) ? rhs.chunks[i] : 0;

            // compute the subtraction including the borrow
            uint64_t diff = lhs_val - rhs_val - borrow;

            // lower 32 bits = result for this chunk
            result.chunks[i] = static_cast<uint32_t>(diff);

            // compute the borrow for the next iteration.
            // a borrow is needed if the subtraction underflowed & this happens
            // if the subtrahend (rhs_val + borrow) was larger than the minuend
            // (lhs_val).
            borrow = (lhs_val < rhs_val + borrow) ? 1 : 0;
        }

        result.apply_mask();
        return result;
    }

    /**
     * @brief Multiplication operator.
     * The result width is N + M, capped at 128 bits to avoid violating
     * the class assertion. Any overflow beyond 128 bits is discarded.
     */
    template <int M>
    auto operator*(const Bit<M>& rhs) const
        -> Bit<(N + M > 128) ? 128 : (N + M)> {
        constexpr int IDEAL_BITS = N + M;
        constexpr int RESULT_BITS = (IDEAL_BITS > 128) ? 128 : IDEAL_BITS;

        // zero initialized
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;

        // multiplication chunk by chunk
        for (int j = 0; j < rhs_chunks; ++j) {
            uint64_t carry = 0;
            for (int i = 0; i < num_chunks; ++i) {
                // BOUNDS CHECK: Only proceed if the result chunk is within our
                // capped Bit object
                if (i + j < result.num_chunks) {
                    uint64_t product =
                        static_cast<uint64_t>(chunks[i]) * rhs.chunks[j];
                    uint64_t sum = static_cast<uint64_t>(result.chunks[i + j]) +
                                   product + carry;

                    result.chunks[i + j] = static_cast<uint32_t>(sum);
                    carry = sum >> 32;
                }
            }
            // propagating the final carry if it fits in the result
            if (j + num_chunks < result.num_chunks) {
                result.chunks[j + num_chunks] += static_cast<uint32_t>(carry);
            }
        }

        result.apply_mask();
        return result;
    }

    /**
     * @brief Less-than comparison operator
     */
    template <int M>
    Bit<1> operator<(const Bit<M>& rhs) const {
        constexpr int lhs_chunks = num_chunks;
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int max_chunks =
            (lhs_chunks > rhs_chunks) ? lhs_chunks : rhs_chunks;

        // comparison starting from the msb chunk
        for (int i = max_chunks - 1; i >= 0; --i) {
            // missing chunks in shorter numbers = zero
            uint32_t lhs_chunk = (i < lhs_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;

            if (lhs_chunk < rhs_chunk) {
                return Bit<1>(1);
            }
            if (lhs_chunk > rhs_chunk) {
                return Bit<1>(0);
            }
        }
        // all chunks =
        return Bit<1>(0);
    }

    /**
     * @brief Greater-than comparison operator
     */
    template <int M>
    Bit<1> operator>(const Bit<M>& rhs) const {
        return rhs < *this;
    }

    /**
     * @brief Less-than-or-equal-to comparison operator
     */
    template <int M>
    Bit<1> operator<=(const Bit<M>& rhs) const {
        return !(*this > rhs);
    }

    /**
     * @brief greater-than-or-equal-to comparison operator
     */
    template <int M>
    Bit<1> operator>=(const Bit<M>& rhs) const {
        return !(*this < rhs);
    }

    /**
     * @brief logical left shift operator, filling with zeros.
     */
    Bit<N> operator<<(int shift) const {
        Bit<N> result;

        if (shift <= 0) {
            return *this; // no shift
        }
        if (shift >= N) {
            return result; // shifting by N or more results in zero
        }

        int chunk_shift = shift / 32;
        int bit_shift = shift % 32;

        for (int i = num_chunks - 1; i >= chunk_shift; --i) {
            // the main part of the new chunk comes from the source chunk
            // shifted left
            result.chunks[i] = chunks[i - chunk_shift] << bit_shift;

            // if there's a bit_shift, we also need to bring in the high bits
            // from the previous source chunk.
            if (bit_shift > 0 && i > chunk_shift) {
                result.chunks[i] |=
                    chunks[i - chunk_shift - 1] >> (32 - bit_shift);
            }
        }

        result.apply_mask();
        return result;
    }

    /**
     * @brief Logical left shift by another Bit vector (like the Verilog shift
     * of a variable amount)
     */
    template <int M>
    Bit<N> operator<<(const Bit<M>& shift_amount) const {
        uint64_t shift_val = static_cast<uint64_t>(shift_amount);
        return *this << shift_val;
    }

    /**
     * @brief Logical right shift operator, filling with zeros
     */
    Bit<N> operator>>(int shift) const {
        Bit<N> result; // zero-initialized by default

        if (shift <= 0) {
            return *this; // no shift or invalid shift
        }
        if (shift >= N) {
            return result; // shifting by N or more results in zero
        }

        int chunk_shift = shift / 32;
        int bit_shift = shift % 32;

        for (int i = 0; i < num_chunks - chunk_shift; ++i) {
            // the main part of the new chunk comes from the source chunk,
            // shifted right
            result.chunks[i] = chunks[i + chunk_shift] >> bit_shift;

            // ff there's a bit_shift, we also need to bring in the low bits
            // from the next source chunk
            if (bit_shift > 0 && i + chunk_shift + 1 < num_chunks) {
                result.chunks[i] |= chunks[i + chunk_shift + 1]
                                    << (32 - bit_shift);
            }
        }

        result.apply_mask();
        return result;
    }

    template <int M>
    Bit<N> operator>>(const Bit<M>& shift_amount) const {
        uint64_t shift_val = static_cast<uint64_t>(shift_amount);
        return *this >> shift_val;
    }

    /**
     * @brief Logical NOT operator
     * Returns Bit<1>(1) if the entire vector is zero, Bit<1>(0) otherwise
     */
    Bit<1> operator!() const {
        // Checking if any bit in any chunk is non-zero
        for (int i = 0; i < num_chunks; ++i) {
            if (chunks[i] != 0) {
                return Bit<1>(0); // Vector is not zero, so logical NOT is false
            }
        }
        // If all chunks are zero, logical NOT is true
        return Bit<1>(1);
    }

    /**
     * @brief Bitwise AND operator
     */
    template <int M>
    auto operator&(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        for (int i = 0; i < result_chunks; ++i) {
            // Treat missing chunks in shorter numbers as zero for the AND
            // operation
            uint32_t lhs_chunk = (i < num_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            result.chunks[i] = lhs_chunk & rhs_chunk;
        }

        result.apply_mask();
        return result;
    }

    /**
     * @brief Bitwise OR operator
     */
    template <int M>
    auto operator|(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        for (int i = 0; i < result_chunks; ++i) {
            // Treat missing chunks in shorter numbers as zero for the OR
            // operation
            uint32_t lhs_chunk = (i < num_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            result.chunks[i] = lhs_chunk | rhs_chunk;
        }

        result.apply_mask();
        return result;
    }

    /**
     * @brief Bitwise XOR operator
     */
    template <int M>
    auto operator^(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        for (int i = 0; i < result_chunks; ++i) {
            // Treat missing chunks in shorter numbers as zero for the XOR
            // operation
            uint32_t lhs_chunk = (i < num_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            result.chunks[i] = lhs_chunk ^ rhs_chunk;
        }

        result.apply_mask();
        return result;
    }

    /**
     * @brief Logical AND operator
     * Treats the entire vector as a single boolean value (true if non-zero)
     */
    template <int M>
    Bit<1> operator&&(const Bit<M>& rhs) const {
        // i'm using the explicit bool conversion to determine if each vector
        // isnon-zero.
        bool lhs_is_true = static_cast<bool>(*this);
        bool rhs_is_true = static_cast<bool>(rhs);

        return Bit<1>(lhs_is_true && rhs_is_true);
    }

    /**
     * @brief Logical OR operator
     * Treats the entire vector as a single boolean value (true if non-zero)
     */
    template <int M>
    Bit<1> operator||(const Bit<M>& rhs) const {
        bool lhs_is_true = static_cast<bool>(*this);
        bool rhs_is_true = static_cast<bool>(rhs);

        return Bit<1>(lhs_is_true || rhs_is_true);
    }

    /**
     * @brief Unary bitwise NOT operator
     */
    Bit<N> operator~() const {
        Bit<N> result;
        for (int i = 0; i < num_chunks; ++i) {
            result.chunks[i] = ~chunks[i];
        }
        result.apply_mask();
        return result;
    }

    /**
     * @brief Performs a reduction AND on the vector
     * @return Bit<1>(1) if all bits are 1, Bit<1>(0) otherwise
     */
    Bit<1> reduce_and() const {
        // Create a temporary vector with all bits set to 1
        Bit<N> all_ones;
        all_ones.chunks.fill(0xFFFFFFFF);
        all_ones.apply_mask(); // mask it to the correct width N

        // The reduction AND is 1 sse the number is equal to all ones.
        return (*this == all_ones);
    }

    /**
     * @brief Performs a reduction OR on the vector.
     * @return Bit<1>(1) if any bit is 1, Bit<1>(0) otherwise.
     */
    Bit<1> reduce_or() const {
        // The reduction OR is 1 if the number is non-zero.
        return Bit<1>(static_cast<bool>(*this));
    }

    /**
     * @brief Performs a reduction XOR on the vector
     * @return Bit<1>(1) if there is an odd number of set bits, Bit<1>(0)
     * otherwise.
     */
    Bit<1> reduce_xor() const {
        int total_set_bits = 0;
        for (int i = 0; i < num_chunks; ++i) {
            uint32_t chunk = chunks[i];
            // there is the __popcount intrisic for clang
            // TODO: search for the gpu equivalent
            while (chunk > 0) {
                chunk &= (chunk - 1);
                total_set_bits++;
            }
        }
        // result is 1 if the total number of set bits is odd
        return Bit<1>(total_set_bits % 2);
    }

    /**
     * @brief Performs a reduction NAND on the vector
     */
    Bit<1> reduce_nand() const { return !this->reduce_and(); }

    /**
     * @brief Performs a reduction NOR on the vector
     */
    Bit<1> reduce_nor() const { return !this->reduce_or(); }

    /**
     * @brief Performs a reduction XNOR on the vector
     */
    Bit<1> reduce_xnor() const { return !this->reduce_xor(); }

    /**
     * @brief Addition assignment operator
     */
    template <int M>
    Bit<N>& operator+=(const Bit<M>& rhs) {
        *this = *this + rhs;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator
     */
    template <int M>
    Bit<N>& operator-=(const Bit<M>& rhs) {
        *this = *this - rhs;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator
     */
    template <int M>
    Bit<N>& operator*=(const Bit<M>& rhs) {
        *this = *this * rhs;
        return *this;
    }

    /**
     * @brief Division assignment operator
     */
    template <int M>
    Bit<N>& operator/=(const Bit<M>& rhs) {
        *this = *this / rhs;
        return *this;
    }

    explicit operator uint64_t() const {
        uint64_t value = 0;
        if (num_chunks > 0) {
            value = chunks[0];
        }
        if (num_chunks > 1) {
            value |= static_cast<uint64_t>(chunks[1]) << 32;
        }
        // For simplicity, we just return the lower 64 bits.
        // A more complex implementation could handle overflow.
        return value;
    }

    /**
     * @brief Explicit conversion to bool
     * Allows a Bit object to be used in a boolean context (e.g., if
     * statements). Returns true if the vector is non-zero, false otherwise
     */
    explicit operator bool() const {
        for (int i = 0; i < num_chunks; ++i) {
            if (chunks[i] != 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Division operator.
     * The result width is the same as the WIDER of the two operands.
     */
    template <int M>
    auto operator/(const Bit<M>& divisor) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        // Division by zero is undefined. In Verilog, it results in all 'X's.
        // Since we don't have an 'X' state, i'm choosing to return all ones but
        // you can change this
        // TODO: confrontati
        if (divisor == Bit<M>(0)) {
            Bit<RESULT_BITS> all_ones;
            all_ones.chunks.fill(0xFFFFFFFF);
            all_ones.apply_mask();
            return all_ones;
        }

        // The dividend must be promoted to the result width for the algorithm
        // to work correctly.
        const Bit<RESULT_BITS> dividend(*this);
        Bit<RESULT_BITS> quotient;
        Bit<RESULT_BITS> remainder;

        // iterate from the most significant bit to the least significant bit.
        for (int i = RESULT_BITS - 1; i >= 0; --i) {
            // 1. shift the remainder left by 1.
            remainder = remainder << 1;
            // 2. "bring down" the next bit from the dividend and set it as the
            // LSB of the remainder. Check if the i-th bit of the dividend is 1
            if ((dividend.chunks[i / 32] >> (i % 32)) & 1) {
                remainder.chunks[0] |= 1;
            }
            // 3. If the remainder is now greater than or equal to the divisor,
            // we can subtract.
            if (remainder >= divisor) {
                remainder = remainder - divisor;
                // set the corresponding bit in the quotient to 1
                quotient.chunks[i / 32] |= (1U << (i % 32));
            }
        }
        quotient.apply_mask();
        return quotient;
    }

    /**
     * @brief Modulo operator.
     * The result width is the same as the DIVISOR (rhs).
     */
    template <int M>
    auto operator%(const Bit<M>& divisor) const -> Bit<M> {
        if (divisor == Bit<M>(0)) {
            Bit<M> all_ones;
            all_ones.chunks.fill(0xFFFFFFFF);
            all_ones.apply_mask();
            return all_ones;
        }

        // i'm reusing the division logic to find the remainder.
        Bit<max(N, M)> dividend(*this);
        Bit<max(N, M)> remainder;

        for (int i = max(N, M) - 1; i >= 0; --i) {
            remainder = remainder << 1;
            if ((dividend.chunks[i / 32] >> (i % 32)) & 1) {
                remainder.chunks[0] |= 1;
            }
            if (remainder >= divisor) {
                remainder = remainder - divisor;
            }
        }

        // The final remainder is cast to the width of the divisor.
        return Bit<M>(remainder);
    }

    /**
     * @brief Converts the Bit vector to a hexadecimal string.
     * Mimics the behavior of Verilog's '$display("%h", ...)' for comparison.
     * @return A std::string containing the hexadecimal representation.
     */
    std::string to_string() const {
        std::stringstream ss;
        ss << std::hex; // setting the stream to output in hexadecimal format

        // We can start by handling the most significant chunk first, as it may
        // not be a full 8 hex characters
        int msb_chunk_idx = num_chunks - 1;

        // computing how many bits are in the last chunk
        int bits_in_msb = (N % 32 == 0) ? 32 : (N % 32);

        // computing the number of hex characters needed for those bits
        int hex_chars_in_msb = (bits_in_msb + 3) / 4;

        // printing the most significant chunk with the calculated width
        ss << std::setw(hex_chars_in_msb) << std::setfill('0')
           << chunks[msb_chunk_idx];

        // printing the rest of the chunks (if any) from most to least
        // significant
        for (int i = msb_chunk_idx - 1; i >= 0; --i) {
            // all lower chunks are full, so they are 8 hex characters (32 bits)
            ss << std::setw(8) << std::setfill('0') << chunks[i];
        }

        return ss.str();
    }

private:
    /**
     * ============ Private helper functions ============
     */
    // constructors, assignments or arithmetic operators MUST call
    // apply_mask() before they finish to clean up the result to
    // preserve the class invariant
    void apply_mask() {
        for (int i = 0; i < num_chunks; ++i)
            chunks[i] &= mask[i];
    }

    // Helper for the initializer_list constructor
    void copy_from_init(std::initializer_list<uint32_t> init) {
        chunks.fill(0);
        int i = 0;
        // Copy values from the list, ensuring we don't overflow our
        // chunks array
        for (uint32_t val : init) {
            if (i >= num_chunks) {
                break;
            }
            chunks[i++] = val;
        }
        apply_mask();
    }

    // Helper function to parse a hex string and populate the chunks.
    void parse_hex_string(const char* hex_string) {
        chunks.fill(0); // starting with a clean slate.

        const std::string str(hex_string);
        size_t start_pos = 0;

        // x or X prefix
        if (str.length() > 2 && str[0] == '0' &&
            (str[1] == 'x' || str[1] == 'X')) {
            start_pos = 2;
        }

        int chunk_idx = 0;
        int bits_in_chunk = 0;

        // parsing from lsb to msb
        for (int i = str.length() - 1; i >= static_cast<int>(start_pos); --i) {
            char c = str[i];
            uint32_t val = 0;

            // Convert hex character to its 4-bit integer value
            if (c >= '0' && c <= '9') {
                val = c - '0';
            } else if (c >= 'a' && c <= 'f') {
                val = 10 + (c - 'a');
            } else if (c >= 'A' && c <= 'F') {
                val = 10 + (c - 'A');
            } else {
                // ignoring invalid chars
                // TODO: some exception?
                continue;
            }

            // Place the 4-bit value into the current chunk at the correct
            // position
            chunks[chunk_idx] |= (val << bits_in_chunk);
            bits_in_chunk += 4;

            // If the current chunk is full (32 bits), move to the next
            // one
            if (bits_in_chunk == 32) {
                chunk_idx++;
                bits_in_chunk = 0;
                // Stop if we have filled all the chunks our Bit<N> can
                // hold
                if (chunk_idx >= num_chunks) {
                    break;
                }
            }
        }

        // ensuring the final chunk is properly masked
        apply_mask();
    }

    // Helper to set the value from a integer type
    template <typename T>
    void set_value(T value) {
        static_assert(std::is_integral_v<T>,
                      "Input value must be an integral type.");

        // cast of the input to 64 to make the shift safe
        uint64_t temp_val = value;

        chunks.fill(0);
        for (int i = 0; i < num_chunks && temp_val != 0; ++i) {
            chunks[i] = static_cast<uint32_t>(temp_val);
            temp_val >>= 32;
        }
        apply_mask();
    }

    /**
     * ============ Private constants & mask generation ============
     */
    // Number of chunks for the bit vector storage
    static constexpr int num_chunks = (N + 31) / 32;

    // The mask computation is static since the mask is shared by all
    // the objects with the same Bit<N> width
    static constexpr std::array<uint32_t, num_chunks> compute_mask() {
        // result accumulator
        std::array<uint32_t, num_chunks> mask{};
        // chunk by chunk
        for (int i = 0; i < num_chunks; ++i) {
            // msb chunk discriminator
            if (i == num_chunks - 1 && N % 32 != 0) {
                // N % 32 returns how many bits are used in the final
                // chunk 1U << n_bits moves a 1 left by n_bits, then by
                // doing -1 we flip all the bits to the rhs and obtain
                // the mask
                mask[i] = (1U << (N % 32)) - 1;
            } else {
                // full chunk case
                mask[i] = 0xFFFFFFFF;
            }
        }
        return mask;
    }
    static constexpr std::array<uint32_t, num_chunks> mask = compute_mask();

    // ============ Data storage ============
    std::array<uint32_t, num_chunks> chunks{};
};

#endif // RUNTIME_HPP
