#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>

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
    Bit(std::initializer_list<uint32_t> init) {
        copy_from_init(init);
        return *this;
    }

    // Assignement like Bit<N> = Bit<M>. If M<N then the MSB's are zeroed,
    // otherwire they are lost. I'm returning a reference because that's what
    // usually happens, if you want to change it in theory there should be no
    // problems
    // https://stackoverflow.com/questions/15292892/what-is-the-return-type-of-the-built-in-assignment-operator
    template <int M>
    Bit& operator=(const Bit<M>& rhs) {
        chunks.fill(0);

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

    // Assignement for the usual {}
    Bit& operator=(std::initializer_list<uint32_t> init) {
        copy_from_init(init);
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
        // I'm assuming Bit<max(N, M)> as the output lenght
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
                result.chunks[i + rhs.num_chunks] =
                    static_cast<uint32_t>(carry);
            }
        }
        result.apply_mask();
        return result;
    }

    // TODO: la dimensione? La stiamo nascondendo sotto il tappeto?
    Bit operator/(const Bit& divisor) const {
        // Technically X but it's not implemented
        if (divisor == Bit(0))
            return Bit(0);

        Bit dividend = *this;
        Bit quotient;
        Bit remainder;

        // data-dependance, divergence?
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
        if (divisor == Bit(0))
            return Bit(0);

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

    //
    Bit pow(const Bit& exponent) const {
        if (*this == Bit(0) && exponent == Bit(0)) {
            return Bit(1);
        }

        Bit result(1);
        Bit base = *this;
        Bit exp = exponent;

        // Early exit se zero base
        if (base == Bit(0)) {
            return Bit(0);
        }

        while (exp != Bit(0)) {
            if ((exp.chunks[0] & 1) != 0) {
                result = result * base;
            }
            base = base * base;
            exp = exp >> 1;

            // Short-circuit se base becomes zero
            if (base == Bit(0))
                break;
        }
        return result;
    }

    bool operator<(const Bit& rhs) const {
        // Da MSB a LSB
        for (int i = num_chunks - 1; i >= 0; --i) {
            if (chunks[i] < rhs.chunks[i])
                return true;
            if (chunks[i] > rhs.chunks[i])
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

    // TODO: ti mancano i reduction operators che ti collassano il vettore ad
    // uno scalare

    // Shift operators
    // TODO: da controllare, prob sto scrivendo spaghetti code
    Bit operator<<(int shift) const {
        Bit result;
        if (shift >= N)
            return result;

        int chunk_shift = shift / 32;
        int bit_shift = shift % 32;

        // O(n) ma ho un'idea
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

    // TODO: controlla
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

    bool operator!=(const Bit& rhs) const { return !(*this == rhs); }

    /**
        Despite working with chunks of uint32_t, this conversion operator
        is meant to mimic the behavior of the "$display("%d", vec)" semantics
        that implicitly convert a vector to a 32/64-bit integer (depending on
        the specific simulator)
    */
    explicit operator uint64_t() const {
        uint64_t value = 0;
        for (int i = num_chunks - 1; i >= 0; --i) {
            if (i == 1)
                value <<= 32;
            value |= chunks[i];
        }
        return value & ((1ULL << N) - 1);
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
};
