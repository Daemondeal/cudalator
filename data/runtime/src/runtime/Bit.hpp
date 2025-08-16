#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>

#include <fmt/core.h>

/**
 * -- non ho idea di cosa volessi dire con questo commento
Quindi praticamente l'idea è che ci arriva un vettore
Bit<128> a = [{0, 0, 0, 1}];
2^96
Bit<96> a = Bit<128>({qua i valori})
E per i signed scusa? Non stiamo facendo nulla, cosa cambia?

// TODO: ti mancano i reduction operators che ti collassano il vettore ad
// uno scalare
*/

template <int N>
class Bit {
    static_assert(N > 0 && N <= 128, "Bit width must be between 1 and 128");

    static constexpr int num_chunks = (N + 31) / 32;

    static constexpr std::array<uint32_t, num_chunks> compute_mask() {
        std::array<uint32_t, num_chunks> mask{};
        for (int i = 0; i < num_chunks; ++i) {
            if (i == num_chunks - 1 && N % 32 != 0) {
                mask[i] = (1U << (N % 32)) - 1;
            } else {
                mask[i] = 0xFFFFFFFF;
            }
        }
        return mask;
    }

    static constexpr std::array<uint32_t, num_chunks> mask = compute_mask();

    std::array<uint32_t, num_chunks> chunks{};

    void apply_mask() {
        for (int i = 0; i < num_chunks; ++i)
            chunks[i] &= mask[i];
    }

    void copy_from_init(std::initializer_list<uint32_t> init) {
        chunks.fill(0);
        int i = 0;
        for (uint32_t val : init) {
            if (i >= num_chunks)
                break;
            chunks[i++] = val;
        }
        apply_mask();
    }

public:
    Bit() = default;

    static constexpr int max(uint32_t a, uint32_t b) { return (a > b) ? a : b; }

    // Usual constructor by {}
    Bit(std::initializer_list<uint32_t> init) { copy_from_init(init); }

    // Constructor from single value (for now just for debugging purposes, up to
    // 64 bits)
    explicit Bit(uint64_t value) {
        chunks.fill(0);
        if (num_chunks > 0) {
            chunks[0] = static_cast<uint32_t>(value);
        }
        if (num_chunks > 1) {
            chunks[1] = static_cast<uint32_t>(value >> 32);
        }
        apply_mask();
    }

    // Assignment for the usual {}, used to modify the value of a bit vector
    Bit& operator=(std::initializer_list<uint32_t> init) {
        copy_from_init(init);
        return *this;
    }

    // Assignment like Bit<N> = Bit<M>. If M<N then the MSB's are zeroed,
    // otherwise they are lost.
    template <int M>
    Bit& operator=(const Bit<M>& rhs) {
        chunks.fill(0); // MSB at zero

        // Copy as many chunks as fit
        constexpr int thisChunks = num_chunks;
        constexpr int rhsChunks = (M + 31) / 32;
        constexpr int copyCount =
            (thisChunks < rhsChunks ? thisChunks : rhsChunks);

        for (int i = 0; i < copyCount; ++i)
            chunks[i] = rhs.chunks[i];

        apply_mask();
        return *this;
    }

    // Arithmetic operators
    template <int M>
    Bit<max(N, M) + 1> operator+(const Bit<M>& rhs) const {
        // We need to know the number of chunks
        constexpr int lhs_chunks = num_chunks;
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int res_chunks = ((max(N, M) + 1) + 31) / 32;

        Bit<max(N, M) + 1> result;
        uint64_t carry = 0;

        // Now we can loop over the maximum number of chunks in the two
        // operands.
        for (int i = 0; i < res_chunks; ++i) {
            uint64_t lhs_val = (i < lhs_chunks) ? chunks[i] : 0;
            uint64_t rhs_val = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            uint64_t sum = lhs_val + rhs_val + carry;
            result.chunks[i] = static_cast<uint32_t>(sum);
            carry = sum >> 32;
        }

        result.apply_mask();
        return result;
    }

    template <int M>
    Bit<max(N, M)> operator-(const Bit<M>& rhs) const {
        // I'm assuming Bit<max(N, M)> as the output length because
        // in the worst case i'm doing like x-0 so the result is x
        Bit<max(N, M)> result;

        // Just as before, we start by extracting the number of chunks
        constexpr int lhs_chunks = num_chunks;            // N
        constexpr int rhs_chunks = (M + 31) / 32;         // M
        constexpr int res_chunks = (max(N, M) + 31) / 32; // res

        uint64_t borrow = 0;

        // Chunk by chunk, up to res_chunks
        for (int i = 0; i < res_chunks; ++i) {
            uint64_t lhs_val = (i < lhs_chunks) ? chunks[i] : 0;
            uint64_t rhs_val = (i < rhs_chunks) ? rhs.chunks[i] : 0;

            uint64_t diff = lhs_val - rhs_val - borrow;
            result.chunks[i] = static_cast<uint32_t>(diff);

            // Se l'MSB è 1 quando shifto 64 >> 32 allora diff ha borrow out
            borrow = (diff >> 32) & 1;
        }

        // Qua voglio forzare il risultato ad avere max(N,M) bits
        result.apply_mask();
        return result;
    }

    template <int M>
    Bit<N + M> operator*(const Bit<M>& rhs) const {
        Bit<N + M> result;
        for (int i = 0; i < num_chunks; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j < rhs.num_chunks; ++j) {
                if (i + j >= result.num_chunks)
                    break;
                uint64_t product =
                    static_cast<uint64_t>(chunks[i]) * rhs.chunks[j];
                uint64_t temp = static_cast<uint64_t>(result.chunks[i + j]) +
                                (product & 0xFFFFFFFF) + carry;
                result.chunks[i + j] = static_cast<uint32_t>(temp);
                carry = (product >> 32) + (temp >> 32);
            }
            if (i + rhs.num_chunks < result.num_chunks) {
                result.chunks[i + rhs.num_chunks] +=
                    static_cast<uint32_t>(carry);
            }
        }
        result.apply_mask();
        return result;
    }

    // Power function with result capped at 128 bits
    // TODO: parlane con Pietro, ok che è O(log n) però jesus, rimuovendo la
    // static_assert potrei usare l'operatore * che già c'è
    template <int M>
    Bit<((N * M) > 128 ? 128 : (N * M))> pow(const Bit<M>& exponent) const {
        constexpr int ResultBits = ((N * M) > 128) ? 128 : (N * M);

        if (*this == Bit(0) && exponent == Bit<M>(0)) {
            return Bit<ResultBits>(1);
        }

        Bit<ResultBits> result(1);
        Bit<ResultBits> base;
        base = *this; // Use assignment operator for conversion
        Bit<M> exp = exponent;

        // Early exit if base is zero
        if (base == Bit<ResultBits>(0)) {
            return Bit<ResultBits>(0);
        }

        while (exp != Bit<M>(0)) {
            if ((exp.chunks[0] & 1) != 0) {
                // Manual multiplication: result = result * base
                Bit<ResultBits> temp;
                for (int i = 0; i < result.num_chunks; ++i) {
                    uint64_t carry = 0;
                    for (int j = 0; j < base.num_chunks; ++j) {
                        if (i + j >= temp.num_chunks)
                            break;
                        uint64_t product =
                            static_cast<uint64_t>(result.chunks[i]) *
                            base.chunks[j];
                        uint64_t sum =
                            static_cast<uint64_t>(temp.chunks[i + j]) +
                            (product & 0xFFFFFFFF) + carry;
                        temp.chunks[i + j] = static_cast<uint32_t>(sum);
                        carry = (product >> 32) + (sum >> 32);
                    }
                    if (i + base.num_chunks < temp.num_chunks && carry > 0) {
                        temp.chunks[i + base.num_chunks] +=
                            static_cast<uint32_t>(carry);
                    }
                }
                temp.apply_mask();
                result = temp;
            }

            // Manual multiplication: base = base * base
            Bit<ResultBits> squared;
            for (int i = 0; i < base.num_chunks; ++i) {
                uint64_t carry = 0;
                for (int j = 0; j < base.num_chunks; ++j) {
                    if (i + j >= squared.num_chunks)
                        break;
                    uint64_t product =
                        static_cast<uint64_t>(base.chunks[i]) * base.chunks[j];
                    uint64_t sum =
                        static_cast<uint64_t>(squared.chunks[i + j]) +
                        (product & 0xFFFFFFFF) + carry;
                    squared.chunks[i + j] = static_cast<uint32_t>(sum);
                    carry = (product >> 32) + (sum >> 32);
                }
                if (i + base.num_chunks < squared.num_chunks && carry > 0) {
                    squared.chunks[i + base.num_chunks] +=
                        static_cast<uint32_t>(carry);
                }
            }
            squared.apply_mask();
            base = squared;

            exp = exp >> 1;

            // Short-circuit if base becomes zero
            if (base == Bit<ResultBits>(0))
                break;
        }
        return result;
    }

    // Division --> return X for division by zero to match Verilog behavior
    Bit operator/(const Bit& divisor) const {
        // return X for division by zero
        if (divisor == Bit(0)) {
            Bit result;
            for (int i = 0; i < num_chunks; ++i) {
                result.chunks[i] = mask[i];
            }
            return result;
        }

        Bit dividend = *this;
        Bit quotient;
        Bit remainder;

        for (int i = N - 1; i >= 0; --i) {
            remainder = remainder << 1;
            remainder.chunks[0] |= (dividend.chunks[i / 32] >> (i % 32)) & 1;

            if (remainder >= divisor) {
                remainder = remainder - divisor;
                quotient.chunks[i / 32] |= 1 << (i % 32);
            }
        }
        quotient.apply_mask();
        return quotient;
    }

    Bit operator%(const Bit& divisor) const {
        // Return X (all bits set) for modulo by zero
        if (divisor == Bit(0)) {
            Bit result;
            for (int i = 0; i < num_chunks; ++i) {
                result.chunks[i] = mask[i];
            }
            return result;
        }

        Bit remainder;

        for (int i = N - 1; i >= 0; --i) {
            remainder = remainder << 1;
            remainder.chunks[0] |= (chunks[i / 32] >> (i % 32)) & 1;
            if (remainder >= divisor) {
                remainder = remainder - divisor;
            }
        }
        remainder.apply_mask();
        return remainder;
    }

    bool operator<(const Bit& rhs) const {
        // Da MSB a LSB
        for (int i = num_chunks - 1; i >= 0; --i) {
            if (chunks[i] < rhs.chunks[i])
                return true;
            if (chunks[i] > rhs.chunks[i])
                return false;
        }
        return false; // equal
    }

    template <int M>
    bool operator<(const Bit<M>& rhs) const {
        constexpr int lhs_chunks = num_chunks;
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int max_chunks =
            (lhs_chunks > rhs_chunks) ? lhs_chunks : rhs_chunks;

        for (int i = max_chunks - 1; i >= 0; --i) {
            uint32_t lhs_chunk = (i < lhs_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;

            if (lhs_chunk < rhs_chunk)
                return true;
            if (lhs_chunk > rhs_chunk)
                return false;
        }
        return false; // Equal
    }

    bool operator>(const Bit& rhs) const { return rhs < *this; }
    bool operator<=(const Bit& rhs) const { return !(*this > rhs); }
    bool operator>=(const Bit& rhs) const { return !(*this < rhs); }

    // And
    Bit operator&(const Bit& rhs) const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i)
            result.chunks[i] = chunks[i] & rhs.chunks[i];
        result.apply_mask();
        return result;
    }

    // Or
    Bit operator|(const Bit& rhs) const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i)
            result.chunks[i] = chunks[i] | rhs.chunks[i];
        result.apply_mask();
        return result;
    }

    // Xor
    Bit operator^(const Bit& rhs) const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i)
            result.chunks[i] = chunks[i] ^ rhs.chunks[i];
        result.apply_mask();
        return result;
    }

    // Not
    Bit operator~() const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i)
            result.chunks[i] = ~chunks[i];
        result.apply_mask();
        return result;
    }

    // Shift operators
    Bit operator<<(int shift) const {
        Bit result;
        if (shift >= N)
            return result;

        int chunk_shift = shift / 32;
        int bit_shift = shift % 32;

        for (int i = num_chunks - 1; i >= 0; --i) {
            if (i - chunk_shift >= 0) {
                result.chunks[i] = chunks[i - chunk_shift] << bit_shift;
                if (bit_shift > 0 && i - chunk_shift - 1 >= 0)
                    result.chunks[i] |=
                        chunks[i - chunk_shift - 1] >> (32 - bit_shift);
            }
        }
        result.apply_mask();
        return result;
    }

    Bit operator>>(int shift) const {
        Bit result;
        if (shift >= N)
            return result;

        int chunk_shift = shift / 32;
        int bit_shift = shift % 32;

        for (int i = 0; i < num_chunks; ++i) {
            if (i + chunk_shift < num_chunks) {
                result.chunks[i] = chunks[i + chunk_shift] >> bit_shift;
                if (bit_shift > 0 && i + chunk_shift + 1 < num_chunks)
                    result.chunks[i] |= chunks[i + chunk_shift + 1]
                                        << (32 - bit_shift);
            }
        }
        result.apply_mask();
        return result;
    }

    // Comparison operators
    bool operator==(const Bit& rhs) const {
        for (int i = 0; i < num_chunks; ++i)
            if (chunks[i] != rhs.chunks[i])
                return false;
        return true;
    }

    template <int M>
    bool operator==(const Bit<M>& rhs) const {
        constexpr int lhs_chunks = num_chunks;
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int max_chunks =
            (lhs_chunks > rhs_chunks) ? lhs_chunks : rhs_chunks;

        for (int i = 0; i < max_chunks; ++i) {
            uint32_t lhs_chunk = (i < lhs_chunks) ? chunks[i] : 0;
            uint32_t rhs_chunk = (i < rhs_chunks) ? rhs.chunks[i] : 0;
            if (lhs_chunk != rhs_chunk)
                return false;
        }
        return true;
    }

    bool operator!=(const Bit& rhs) const { return !(*this == rhs); }

    template <int M>
    bool operator!=(const Bit<M>& rhs) const {
        return !(*this == rhs);
    }

    /**
        Despite working with chunks of uint32_t, this conversion operator
        is meant to mimic the behavior of the "$display("%d", vec)" semantics
        that implicitly convert a vector to a 32/64-bit integer (depending on
        the specific simulator)
    */
    explicit operator uint64_t() const {
        uint64_t value = 0;

        // Handle up to 64 bits properly
        if constexpr (N <= 32) {
            value = chunks[0];
            // Mask to N bits
            value &= ((1ULL << N) - 1);
        } else if constexpr (N < 64) {
            if (num_chunks >= 2) {
                value = static_cast<uint64_t>(chunks[1]) << 32;
                value |= chunks[0];
            } else {
                value = chunks[0];
            }
            // Mask to N bits
            value &= ((1ULL << N) - 1);
        } else {
            // N >= 64 -> we can just return the lower 64 bits
            if (num_chunks >= 2) {
                value = static_cast<uint64_t>(chunks[1]) << 32;
                value |= chunks[0];
            } else {
                value = chunks[0];
            }
        }

        return value;
    }

    // Just for debugging purposes
    std::string to_string() const {
        std::string str;
        for (int i = num_chunks - 1; i >= 0; --i) {
            char buf[9];
            snprintf(buf, sizeof(buf), "%08X", chunks[i]);
            str += buf;
        }
        return "0x" + str;
    }

    template <typename FormatContext>
    static auto format(const Bit<N>& n, FormatContext& ctx) {
        if (n.chunks.size() == 0) {
            return fmt::format_to(ctx.out(), "{}'h0", N);
        }

        auto out = fmt::format_to(
            ctx.out(),
            "{}'h{:X}",
            N,
            n.chunks[n.chunks.size()-1]
        );

        for (ssize_t j = n.chunks.size() - 2; j >= 0; j--) {
            out = fmt::format_to(out, "{:08X}", n.chunks[j]);
        }

        return out;
    }

    // Per accedere ai private fields ad es con =
    template <int M>
    friend class Bit;
};

template <int N>
struct fmt::formatter<Bit<N>> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Bit<N>& n, FormatContext& ctx) const {
        return Bit<N>::format(n, ctx);
    }
};
