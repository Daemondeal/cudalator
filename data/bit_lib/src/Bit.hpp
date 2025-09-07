#pragma once

#include "cuda_compat.hpp"
#include <cstdint>
#include <fmt/format.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

/**
 * (operators), (name), (implemented)
 * (=), (Binary assignment operator), (yes)
 * (+= -= *= /= %=), (Binary arithmetic assignment operators), (yes)
 * (&= |= ^=), (Binary bitwise assignment operators), (yes)
 * (<<= >>=), (Binary logical shift assignment operators), (yes)
 * (<<<= >>>=), (Binary arithmetic shift assignment operators), (no but i'm
 * supporting only unsigned ops)
 * (?:), (Conditional operator), (yes)
 * (+ -), (Unary arithmetic operators), (yes)
 * (~), (Unary bitwise negation operator), (yes)
 * (!), (Unary logical negation operator), (yes)
 * (& ~& | ~| ^ ~^), (Unary reduction operators), (yes)
 * (+ - * /), (Binary arithmetic operators), (yes)
 * (%), (Binary arithmetic modulus operator), (yes)
 * (& | ^ ^~ ~^), (Binary bitwise operators), (yes)
 * (>> <<), (Binary logical shift operators), (yes)
 * (>>> <<<), (Binary arithmetic shift operators), (no but again it's only
 * unsigned)
 * (&& || -> <->), (Binary logical operators), (yes)
 * (< <= > >=), (Binary relational operators), (yes)
 * (== !=), (Binary case equality operators), (yes)
 * (=== !== ==? !=?), (Binary logical equality operators), (not applicable
 * without X,Z)
 * (==? !=?), (Binary wildcard equality operators), (not applicable without X,Z)
 * (++ --), (Unary increment, decrement operators), (yes)
 * (inside), (Binary set membership operator), (yes)
 * (dist), (Binary distribution operator), (nope, absolutely not)
 * ({} {{}}), (Concatenation, replication operators), (yes)
 * (<<{} >>{}), (Stream operators), (i would like not to implement also these)
 */

template <int N>
class Bit {
    static_assert(N > 0 && N <= 128, "The maximum supported bit width is 128");

    static constexpr int max(int a, int b) { return a > b ? a : b; }

    template <int M>
    friend class Bit;

public:
    static constexpr int width = N;
    /**
     * @brief Default constructor.
     * Creates a Bit vector zero-initialized
     */
    HOST_DEVICE Bit() {
        for (int i = 0; i < num_chunks; ++i) {
            chunks[i] = 0;
        }
    }

    /**
     * @brief Constructor from a single integral value.
     */
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    HOST_DEVICE explicit Bit(T value) {
        set_value(value);
    }

#ifndef __CUDACC__
    /**
     * @brief Constructor from a list of 32-bit chunks
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
#endif
    /**
     * @brief Converting constructor from another Bit vector
     * This allows initialization from a Bit vectorof a different width, like
     * Bit<8> result = Bit<9>(...)
     */
    template <int M>
    HOST_DEVICE Bit(const Bit<M>& rhs) {
        *this = rhs;
    }

    /**
     * @brief Assignment from another Bit vector
     * Handles assignment from both same-sized and different-sized Bit
     * vectors. Truncates if the source is larger, zero-extends if it is
     * smaller.
     */
    template <int M>
    HOST_DEVICE Bit& operator=(const Bit<M>& rhs) {
        // Clear old data since we don't need it anymore
        for (int i = 0; i < num_chunks; ++i) {
            chunks[i] = 0;
        }

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
     * @brief Assignment from a single integral value
     */
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    HOST_DEVICE Bit& operator=(T value) {
        set_value(value);
        return *this;
    }

    /**
     * @brief Equality comparison operator
     */
    template <int M>
    HOST_DEVICE Bit<1> operator==(const Bit<M>& rhs) const {
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
    HOST_DEVICE Bit<1> operator!=(const Bit<M>& rhs) const {
        return !(*this == rhs);
    }

    /**
     * @brief Prefix increment (++a)
     */
    HOST_DEVICE Bit<N>& operator++() {
        *this += Bit<1>(1);
        return *this; // return the new incremented value
    }

    /**
     * @brief postfix increment (a++)
     * I'm using int as a dummy parameter to distinguish the signature
     */
    HOST_DEVICE Bit<N> operator++(int) {
        Bit<N> temp = *this;
        *this += Bit<1>(1);
        return temp;
    }

    /**
     * @brief prefix decrement (--a)
     */
    HOST_DEVICE Bit<N>& operator--() {
        *this -= Bit<1>(1);
        return *this;
    }

    /**
     * @brief postfix decrement (a--)
     */
    HOST_DEVICE Bit<N> operator--(int) {
        Bit<N> temp = *this;
        *this -= Bit<1>(1);
        return temp;
    }

    /**
     * @brief mimics the Verilog 'inside' operator to check for set membership
     * @param value_to_check The value to look for
     * @param values_in_set A variable number of values that form the set
     * @return Bit<1>(1) if the value is in the set, Bit<1>(0) otherwise
     */
    template <int N_Check, typename... T_Set>
    HOST_DEVICE static Bit<1> inside(const Bit<N_Check>& value_to_check,
                                     const T_Set&... values_in_set) {
        bool is_inside = false;
        // è una fold expression, in teoria solo c++17, potenzialmente da
        // cambiare
        // OR sul result di ogni equality check
        ((is_inside = is_inside || (value_to_check == values_in_set)), ...);
        return Bit<1>(is_inside);
    }

    /**
     * @brief Concatenates multiple Bit vectors into a single, wider Bit vector
     * Verilog equivalent: {arg1, arg2, ...}
     */
    template <typename... Args>
    HOST_DEVICE static auto concat(const Args&... args) {
        constexpr int FINAL_WIDTH = (0 + ... + std::decay_t<Args>::width);
        static_assert(FINAL_WIDTH <= 128,
                      "Concatenation result exceeds 128 bits");

        Bit<FINAL_WIDTH> result;
        int current_shift = FINAL_WIDTH; // Start shift from the top (MSB)

        // lambda to place each argument
        auto process = [&](const auto& arg) {
            constexpr int arg_width = std::decay_t<decltype(arg)>::width;
            current_shift -= arg_width; // Move the shift position down
            Bit<FINAL_WIDTH> temp(arg);
            result |= (temp << current_shift); // Place the argument
        };

        (process(args), ...); // Process from left-to-right (MSB to LSB)
        return result;
    }

    /**
     * @brief Replicates a Bit vector a specified number of times
     * Verilog equivalent: {N{val}}
     */
    template <int REPLICATION_COUNT, int M>
    HOST_DEVICE static auto replicate(const Bit<M>& val) {
        constexpr int FINAL_WIDTH = REPLICATION_COUNT * M;
        static_assert(FINAL_WIDTH > 0,
                      "Replication must result in a positive width.");
        static_assert(FINAL_WIDTH <= 128,
                      "Replication result exceeds 128 bits");

        Bit<FINAL_WIDTH> result;

        for (int i = 0; i < REPLICATION_COUNT; ++i) {
            result |= (Bit<FINAL_WIDTH>(val) << (i * M));
        }
        return result;
    }

    /**
     * @brief Addition operator
     * Adds two Bit vectors, returning a new vector with the result.
     * The result is one bit wider to accommodate a carry-out, UNLESS
     * that would exceed the 128-bit limit, in which case the carry is
     * discarded.
     */
    template <int M>
    HOST_DEVICE auto operator+(const Bit<M>& rhs) const
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
        return result;
    }

    /**
     * @brief Subtraction operator.
     * Subtracts two Bit vectors, returning a new vector with the result.
     * The result width is the same as the wider of the two operands.
     */
    template <int M>
    HOST_DEVICE auto operator-(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        // temporary operands with the final result width
        const Bit<RESULT_BITS> lhs_promoted(*this);
        const Bit<RESULT_BITS> rhs_promoted(rhs);

        constexpr int result_chunks = (RESULT_BITS + 31) / 32;
        uint64_t borrow = 0;

        for (int i = 0; i < result_chunks; ++i) {
            uint64_t lhs_val = lhs_promoted.chunks[i];
            uint64_t rhs_val = rhs_promoted.chunks[i];

            uint64_t diff = lhs_val - rhs_val - borrow;
            result.chunks[i] = static_cast<uint32_t>(diff);
            borrow = (lhs_val < rhs_val + borrow) ? 1 : 0;
        }

        return result;
    }

    /**
     * @brief Multiplication operator.
     * The result width is N + M, capped at 128 bits to avoid violating
     * the class assertion. Any overflow beyond 128 bits is discarded.
     */
    template <int M>
    HOST_DEVICE auto operator*(const Bit<M>& rhs) const
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
                // only proceed if the result chunk is within our
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
    HOST_DEVICE Bit<1> operator<(const Bit<M>& rhs) const {
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
    HOST_DEVICE Bit<1> operator>(const Bit<M>& rhs) const {
        return rhs < *this;
    }

    /**
     * @brief Less-than-or-equal-to comparison operator
     */
    template <int M>
    HOST_DEVICE Bit<1> operator<=(const Bit<M>& rhs) const {
        return !(*this > rhs);
    }

    /**
     * @brief greater-than-or-equal-to comparison operator
     */
    template <int M>
    HOST_DEVICE Bit<1> operator>=(const Bit<M>& rhs) const {
        return !(*this < rhs);
    }

    /**
     * @brief logical left shift operator, filling with zeros
     */
    HOST_DEVICE Bit<N> operator<<(int shift) const {
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
     * @brief logical left shift by another Bit vector (aka Verilog shift
     * of a variable amount)
     */
    template <int M>
    HOST_DEVICE Bit<N> operator<<(const Bit<M>& shift_amount) const {
        uint64_t shift_val = static_cast<uint64_t>(shift_amount);
        return *this << shift_val;
    }

    /**
     * @brief Logical right shift operator, filling with zeros
     */
    HOST_DEVICE Bit<N> operator>>(int shift) const {
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
    HOST_DEVICE Bit<N> operator>>(const Bit<M>& shift_amount) const {
        uint64_t shift_val = static_cast<uint64_t>(shift_amount);
        return *this >> shift_val;
    }

    /**
     * @brief Logical NOT operator
     * Returns Bit<1>(1) if the entire vector is zero, Bit<1>(0) otherwise
     */
    HOST_DEVICE Bit<1> operator!() const {
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
    HOST_DEVICE auto operator&(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        for (int i = 0; i < result_chunks; ++i) {
            // treating missing chunks in shorter numbers as zero for the AND
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
    HOST_DEVICE auto operator|(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        for (int i = 0; i < result_chunks; ++i) {
            // treating missing chunks in shorter numbers as zero for the OR
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
    HOST_DEVICE auto operator^(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        Bit<RESULT_BITS> result;

        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int result_chunks = (RESULT_BITS + 31) / 32;

        for (int i = 0; i < result_chunks; ++i) {
            // treating missing chunks in shorter numbers as zero for the XOR
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
    HOST_DEVICE Bit<1> operator&&(const Bit<M>& rhs) const {
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
    HOST_DEVICE Bit<1> operator||(const Bit<M>& rhs) const {
        bool lhs_is_true = static_cast<bool>(*this);
        bool rhs_is_true = static_cast<bool>(rhs);

        return Bit<1>(lhs_is_true || rhs_is_true);
    }

    /**
     * @brief Performs a logical implication (if-then)
     * Treats the entire vector as a single boolean value.
     * Logically equivalent to !(*this) || rhs.
     * @return Bit<1>(1) for true, Bit<1>(0) for false.
     */
    template <int M>
    HOST_DEVICE Bit<1> logical_implication(const Bit<M>& rhs) const {
        return !(*this) || rhs;
    }

    /**
     * @brief Performs a logical equivalence (if-and-only-if).
     * Treats the entire vector as a single boolean value.
     * @return Bit<1>(1) for true, Bit<1>(0) for false.
     */
    template <int M>
    HOST_DEVICE Bit<1> logical_equivalence(const Bit<M>& rhs) const {
        // implementing (A -> B) && (B -> A)
        return this->logical_implication(rhs) && rhs.logical_implication(*this);
    }

    /**
     * @brief Mimics Verilog conditional -ternary operator.
     */
    template <int Cond_N, int W1, int W2>
    HOST_DEVICE static Bit<max(W1, W2)> conditional(const Bit<Cond_N>& cond,
                                                    const Bit<W1>& true_val,
                                                    const Bit<W2>& false_val) {
        if (static_cast<bool>(cond)) {
            return true_val;
        } else {
            return false_val;
        }
    }

    /**
     * @brief unary plus (+a)
     */
    HOST_DEVICE Bit<N> operator+() const {
        return *this; // returns a copy of the object unchanged
    }

    /**
     * @brief unary minus (-a)
     */
    HOST_DEVICE Bit<N> operator-() const {
        // 2s complement is (~a + 1)
        return ~(*this) + Bit<1>(1);
    }

    /**
     * @brief Unary bitwise NOT operator
     */
    HOST_DEVICE Bit<N> operator~() const {
        Bit<N> result;
        for (int i = 0; i < num_chunks; ++i) {
            result.chunks[i] = ~chunks[i];
        }
        result.apply_mask();
        return result;
    }

    /**
     * @brief Bitwise XNOR operator function
     */
    template <int M>
    HOST_DEVICE auto xnor(const Bit<M>& rhs) const -> Bit<max(N, M)> {
        return ~(*this ^ rhs);
    }

    /**
     * @brief Performs a reduction AND on the vector
     * @return Bit<1>(1) if all bits are 1, Bit<1>(0) otherwise
     */
    HOST_DEVICE Bit<1> reduce_and() const {
        // creating a temporary vector with all 1s
        Bit<N> all_ones;
        for (int i = 0; i < num_chunks; ++i) {
            all_ones.chunks[i] = 0xFFFFFFFF;
        }
        all_ones.apply_mask(); // mask it to the correct width N

        // The reduction AND is 1 sse the number is equal to all ones.
        return (*this == all_ones);
    }

    /**
     * @brief Performs a reduction OR on the vector
     * @return Bit<1>(1) if any bit is 1, Bit<1>(0) otherwise
     */
    HOST_DEVICE Bit<1> reduce_or() const {
        // The reduction OR is 1 if the number is non-zero
        return Bit<1>(static_cast<bool>(*this));
    }

    /**
     * @brief Performs a reduction XOR on the vector
     * @return Bit<1>(1) if there is an odd number of set bits, Bit<1>(0)
     * otherwise
     */
    HOST_DEVICE Bit<1> reduce_xor() const {
        int total_set_bits = 0;
#ifdef __CUDACC__
        for (int i = 0; i < num_chunks; ++i) {
            total_set_bits += __popc(chunks[i]);
        }
#else
        for (int i = 0; i < num_chunks; ++i) {
            uint32_t chunk = chunks[i];
            while (chunk > 0) {
                chunk &= (chunk - 1);
                total_set_bits++;
            }
        }
#endif
        // result is 1 if the total number of set bits is odd
        return Bit<1>(total_set_bits % 2);
    }

    /**
     * @brief Performs a reduction NAND on the vector
     */
    HOST_DEVICE Bit<1> reduce_nand() const { return !this->reduce_and(); }

    /**
     * @brief Performs a reduction NOR on the vector
     */
    HOST_DEVICE Bit<1> reduce_nor() const { return !this->reduce_or(); }

    /**
     * @brief Performs a reduction XNOR on the vector
     */
    HOST_DEVICE Bit<1> reduce_xnor() const { return !this->reduce_xor(); }

    /**
     * @brief Addition assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator+=(const Bit<M>& rhs) {
        *this = *this + rhs;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator-=(const Bit<M>& rhs) {
        *this = *this - rhs;
        return *this;
    }

    /**
     * @brief Multiplication assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator*=(const Bit<M>& rhs) {
        *this = *this * rhs;
        return *this;
    }

    /**
     * @brief Division assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator/=(const Bit<M>& rhs) {
        *this = *this / rhs;
        return *this;
    }

    /**
     * @brief Modulo assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator%=(const Bit<M>& rhs) {
        *this = *this % rhs;
        return *this;
    }

    /**
     * @brief Bitwise AND assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator&=(const Bit<M>& rhs) {
        *this = *this & rhs;
        return *this;
    }

    /**
     * @brief Bitwise OR assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator|=(const Bit<M>& rhs) {
        *this = *this | rhs;
        return *this;
    }

    /**
     * @brief Bitwise XOR assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator^=(const Bit<M>& rhs) {
        *this = *this ^ rhs;
        return *this;
    }

    /**
     * @brief Logical right shift assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator>>=(const Bit<M>& rhs) {
        *this = *this >> rhs;
        return *this;
    }

    /**
     * @brief Logical left shift assignment operator
     */
    template <int M>
    HOST_DEVICE Bit<N>& operator<<=(const Bit<M>& rhs) {
        *this = *this << rhs;
        return *this;
    }

    HOST_DEVICE explicit operator uint64_t() const {
        // static_assert(N <= 64, "Bit<N> is too wide to be cast to uint64_t
        // without data loss.");
        uint64_t value = 0;
        if (num_chunks > 0) {
            value = chunks[0];
        }
        if (num_chunks > 1) {
            value |= static_cast<uint64_t>(chunks[1]) << 32;
        }
        // For simplicity, i'm returning the lower 64 bits
        // A more complex implementation could handle overflow
        return value;
    }

    /**
     * @brief Explicit conversion to bool
     * returns true if the vector is non-zero, false otherwise
     */
    HOST_DEVICE explicit operator bool() const {
        for (int i = 0; i < num_chunks; ++i) {
            if (chunks[i] != 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Division operator
     * The result width is the same as the WIDER of the two operands
     */
    template <int M>
    HOST_DEVICE auto operator/(const Bit<M>& divisor) const -> Bit<max(N, M)> {
        constexpr int RESULT_BITS = max(N, M);
        // Division by zero is undefined. In Verilog, it results in all 'X's.
        // Since we don't have an 'X' state, i'm choosing to return all ones but
        // you can change this
        // TODO: confrontati
        if (divisor == Bit<M>(0)) {
            Bit<RESULT_BITS> all_ones;
            for (int i = 0; i < all_ones.num_chunks; ++i) {
                all_ones.chunks[i] = 0xFFFFFFFF;
            }
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
            // we can subtract
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
     * @brief Modulo operator
     * The result width is the same as the DIVISOR (rhs)
     */
    template <int M>
    HOST_DEVICE auto operator%(const Bit<M>& divisor) const
        -> Bit<N> { // Return Bit<N>
        if (divisor == Bit<M>(0)) {
            Bit<N> all_ones; // Create a Bit<N>
            for (int i = 0; i < all_ones.num_chunks; ++i) {
                all_ones.chunks[i] = 0xFFFFFFFF;
            }
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

        // The final remainder is cast to the width of the divisor
        return Bit<M>(remainder);
    }

    /**
     * @brief Performs bit selection
     * @param index  the bit position to access (0 is the LSB)
     */
    HOST_DEVICE Bit<1> select_bit(int index) const {
        // if (index < 0 || index >= N) { throw std::out_of_range("Index out of
        // bounds"); }

        int chunk_index = index / 32;
        int bit_in_chunk = index % 32;

        // we can right shift & maks
        uint32_t bit_value = (chunks[chunk_index] >> bit_in_chunk) & 1;

        return Bit<1>(bit_value);
    }

    /**
     * @brief part selection
     * @tparam MSB most significant bit of the slice (inclusive)
     * @tparam LSB least significant bit of the slice (inclusive)
     */
    template <int MSB, int LSB>
    HOST_DEVICE auto select_part() const -> Bit<MSB - LSB + 1> {
        static_assert(MSB >= LSB, "MSB must be greater than or equal to LSB");
        static_assert(MSB < N, "MSB is out of bounds for this Bit vector");

        constexpr int SLICE_WIDTH = MSB - LSB + 1;

        // right-shift so to move the lsb of the slice to position 0
        auto shifted_val = (*this) >> LSB;

        // = will truncate and the result is already of the desired length
        return Bit<SLICE_WIDTH>(shifted_val);
    }

    /**
     * @brief same thing of select_part to be used in cases where there is
     * variable position
     */
    template <int SLICE_WIDTH>
    HOST_DEVICE auto select_part_indexed(int lsb_start_index) const
        -> Bit<SLICE_WIDTH> {
        static_assert(SLICE_WIDTH <= N,
                      "Slice width cannot be larger than the vector");
        auto shifted_val = (*this) >> lsb_start_index;
        return Bit<SLICE_WIDTH>(shifted_val);
    }

    /**
     * @brief part-select assignment (writing to a slice with compile-time
     * bounds) aka Verilog's a[MSB:LSB] = value
     * @tparam MSB of the slice (inclusive)
     * @tparam LSB of the slice (inclusive)
     * @tparam M width of the value being assigned. Must match the slice width.
     * @param value bit vector to assign to the slice
     */
    template <int MSB, int LSB, int M>
    HOST_DEVICE void assign_part(const Bit<M>& value) {
        constexpr int SLICE_WIDTH = MSB - LSB + 1;
        static_assert(MSB >= LSB, "MSB must be greater than or equal to LSB");
        static_assert(MSB < N, "MSB is out of bounds for this Bit vector");
        static_assert(M == SLICE_WIDTH,
                      "Value width must match the slice width");

        // create the mask
        Bit<SLICE_WIDTH> slice_ones =
            ~Bit<SLICE_WIDTH>(0); // 1s for the slice width
        Bit<N> mask = Bit<N>(slice_ones) << LSB;

        // shift the new value into the correct position
        Bit<N> shifted_value = Bit<N>(value) << LSB;

        *this &= ~mask;         // "hole" in the vector
        *this |= shifted_value; // fills the "hole" with the new value
    }

    /**
     * @brief Part-select assignment (writing to a slice with a runtime start
     * index).
     * @tparam M The width of the value being assigned.
     * @param lsb_start_index The starting LSB position for the assignment.
     * @param value The Bit vector to assign to the slice.
     */
    template <int M>
    HOST_DEVICE void assign_part_indexed(int lsb_start_index,
                                         const Bit<M>& value) {
        // Runtime check for bounds to prevent errors.
        if (lsb_start_index < 0 || lsb_start_index + M > N) {
            // In production code, you might throw an exception like
            // std::out_of_range.
            return;
        }

        // 1. Create the mask at runtime.
        Bit<M> slice_ones = ~Bit<M>(0);
        Bit<N> mask = Bit<N>(slice_ones) << lsb_start_index;

        // 2. Prepare the value at runtime.
        Bit<N> shifted_value = Bit<N>(value) << lsb_start_index;

        // 3. Clear the target bits and then set them.
        *this &= ~mask;
        *this |= shifted_value;
    }

    // #ifndef __CUDACC__
    /**
     * @brief Converts the Bit vector to a hexadecimal string.
     * Mimics the behavior of Verilog's '$display("%h", ...)' for comparison
     * @return A std::string containing the hexadecimal representation
     */
    HOST std::string to_string() const {
        std::stringstream ss;
        ss << std::hex; // stream output in hexadecimal format

        // most significant chunk first, as it may not be full
        int msb_chunk_idx = num_chunks - 1;

        // apply the mask for the most significant chunk to ensure we only
        // consider the valid bits for this Bit<N> object
        uint32_t msb_val =
            // chunks[msb_chunk_idx] & mask_holder.data[msb_chunk_idx];
            chunks[msb_chunk_idx] & mask_at(msb_chunk_idx);

        // compute how many bits are in the last chunk
        int bits_in_msb = (N % 32 == 0) ? 32 : (N % 32);
        // compute the number of hex characters needed for those bits
        int hex_chars_in_msb = (bits_in_msb + 3) / 4;

        // print the correctly masked and sized most significant chunk
        if (hex_chars_in_msb > 0) {
            ss << std::setw(hex_chars_in_msb) << std::setfill('0') << msb_val;
        }

        // print the rest of the chunks (if any) from most to least significant
        for (int i = msb_chunk_idx - 1; i >= 0; --i) {
            // Lower chunks are always full, so they are 8 hex characters (32
            // bits)
            ss << std::setw(8) << std::setfill('0') << chunks[i];
        }

        // Handle the case of a zero-width stringstream (e.g. for Bit<0>)
        if (ss.str().empty()) {
            return "0";
        }

        return ss.str();
    }

    HOST std::string to_binary_string() {
        if (N == 0)
            return "0";

        std::stringstream ss;

        int msb_idx = num_chunks - 1;
        int bits_in_msb = (N % 32 == 0) ? 32 : (N % 32);

        // Mask the most significant chunk
        // uint32_t msb_val = chunks[msb_idx] & mask[msb_idx];
        uint32_t msb_val = chunks[msb_idx] & mask_at(msb_idx);
        // Print MSB chunk
        for (int i = bits_in_msb - 1; i >= 0; --i) {
            ss << ((msb_val >> i) & 1);
        }

        // Print remaining chunks
        for (int i = msb_idx - 1; i >= 0; --i) {
            for (int j = 31; j >= 0; --j) {
                ss << ((chunks[i] >> j) & 1);
            }
        }

        return ss.str();
    }

    template <typename FormatContext>
    HOST static auto format(const Bit<N>& n, FormatContext& ctx) {
        if (n.num_chunks == 0) {
            return fmt::format_to(ctx.out(), "{}'h0", N);
        }

        auto out = fmt::format_to(ctx.out(), "{}'h{:X}", N,
                                  n.chunks[n.num_chunks - 1]);

        for (ssize_t j = n.num_chunks - 2; j >= 0; j--) {
            out = fmt::format_to(out, "{:08X}", n.chunks[j]);
        }

        return out;
    }
    // #endif

private:
    /**
     * ============ Private helper functions ============
     */
    // constructors, assignments or arithmetic operators MUST call
    // apply_mask() before they finish to clean up the result to
    // preserve the class invariant
    HOST_DEVICE void apply_mask() {
        for (int i = 0; i < num_chunks; ++i)
            // chunks[i] &= mask_holder.data[i]
            chunks[i] &= mask_at(i);
    }

// Helper for the initializer_list constructor
#ifndef __CUDACC__
    void copy_from_init(std::initializer_list<uint32_t> init) {
        for (int i = 0; i < num_chunks; ++i) {
            chunks[i] = 0;
        }
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
        for (int i = 0; i < num_chunks; ++i) {
            chunks[i] = 0;
        }

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
#endif

    // Helper to set the value from a integer type
    template <typename T>
    HOST_DEVICE void set_value(T value) {
        static_assert(std::is_integral_v<T>,
                      "Input value must be an integral type.");

        // cast of the input to 64 to make the shift safe
        uint64_t temp_val = value;

        for (int i = 0; i < num_chunks; ++i)
            chunks[i] = 0;
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

    // To return a C-style array from a constexpr function, i need to wrap it in
    // a struct
    struct MaskArray {
        uint32_t data[num_chunks];
    };

    // The mask computation is static since the mask is shared by all
    // the objects with the same Bit<N> width
    HOST_DEVICE static constexpr MaskArray compute_mask() {
        // result accumulator
        MaskArray mask_struct{};
        // chunk by chunk
        for (int i = 0; i < num_chunks; ++i) {
            // msb chunk discriminator
            if (i == num_chunks - 1 && N % 32 != 0) {
                // N % 32 returns how many bits are used in the final
                // chunk 1U << n_bits moves a 1 left by n_bits, then by
                // doing -1 we flip all the bits to the rhs and obtain
                // the mask
                mask_struct.data[i] = (1U << (N % 32)) - 1;
            } else {
                // full chunk case
                mask_struct.data[i] = 0xFFFFFFFF;
            }
        }
        return mask_struct;
    }
    // permanent owner for the whole program lifetime
    // inline static constexpr MaskArray mask_holder = compute_mask();
    // pointer, così non scrivo mask_holder ma solo mask[]
    // static constexpr uint32_t const* mask = mask_holder.data;

    HOST_DEVICE static constexpr uint32_t mask_at(int i) {
        return (i == num_chunks - 1 && (N % 32) != 0) ? ((1u << (N % 32)) - 1u)
                                                      : 0xFFFFFFFFu;
    }

    // ============ Data storage ============
    uint32_t chunks[num_chunks];
};

template <int N>
struct fmt::formatter<Bit<N>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const Bit<N>& n, FormatContext& ctx) const {
        return Bit<N>::format(n, ctx);
    }
};
