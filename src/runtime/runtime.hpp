#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

template <int N>
class Bit {
    static_assert(N > 0 && N <= 128, "Bit width must be between 1 and 128");

    // Sincero potrei tornare alla funzione align_size
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

    template <typename T>
    void set_value(T value) {
        chunks.fill(0);
        for (int i = 0; i < num_chunks && value != 0; ++i) {
            chunks[i] = static_cast<uint32_t>(value & 0xFFFFFFFF);
            value >>= 32;
        }
        apply_mask();
    }

public:
    // Constructors
    Bit() = default;
    // Sto assumendo che T sia un tipo intero, confrontati con Pietro
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    Bit(T value) {
        set_value(value);
    }

    // Assignment operators
    // Anche qua parlane, non è detto sia il modo più efficiente
    template <typename T>
    Bit& operator=(T value) {
        set_value(value);
        return *this;
    }

    // Arithmetic operators
    Bit operator+(const Bit& rhs) const {
        Bit result;
        uint64_t carry = 0;
        for (int i = 0; i < num_chunks; ++i) {
            uint64_t sum =
                static_cast<uint64_t>(chunks[i]) + rhs.chunks[i] + carry;
            result.chunks[i] = static_cast<uint32_t>(sum);
            carry = sum >> 32;
        }
        result.apply_mask();
        return result;
    }

    Bit operator-(const Bit& rhs) const {
        Bit result;
        uint64_t borrow = 0;
        for (int i = 0; i < num_chunks; ++i) {
            uint64_t diff =
                static_cast<uint64_t>(chunks[i]) - rhs.chunks[i] - borrow;
            result.chunks[i] = static_cast<uint32_t>(diff);
            borrow = (diff >> 32) ? 1 : 0;
        }
        result.apply_mask();
        return result;
    }

    // TODO: da debuggare, sono troppo stanco
    Bit operator*(const Bit& rhs) const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j < num_chunks; ++j) {
                if (i + j >= num_chunks)
                    break;
                uint64_t product =
                    static_cast<uint64_t>(chunks[i]) * rhs.chunks[j];
                uint64_t temp =
                    result.chunks[i + j] + (product & 0xFFFFFFFF) + carry;
                result.chunks[i + j] = static_cast<uint32_t>(temp);
                carry = (product >> 32) + (temp >> 32);
            }
        }
        result.apply_mask();
        return result;
    }

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

    Bit operator^(const Bit& rhs) const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i)
            result.chunks[i] = chunks[i] ^ rhs.chunks[i];
        result.apply_mask();
        return result;
    }

    Bit operator~() const {
        Bit result;
        for (int i = 0; i < num_chunks; ++i)
            result.chunks[i] = ~chunks[i];
        result.apply_mask();
        return result;
    }

    // Shift operators
    // TODO: da controllare, prob sto scrivendo spaghetti code
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

    bool operator!=(const Bit& rhs) const {
        return !(*this == rhs);
    }

    // Conversion operator
    explicit operator uint64_t() const {
        uint64_t value = 0;
        for (int i = num_chunks - 1; i >= 0; --i) {
            if (i == 1)
                value <<= 32;
            value |= chunks[i];
        }
        return value & ((1ULL << N) - 1);
    }

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
