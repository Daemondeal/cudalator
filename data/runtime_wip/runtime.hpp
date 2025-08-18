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

/**
 * operator bool() fa funzionare il tipo come se fosse un booleano quindi se
 * scrivi if(oggetto) ti restituisce true o false
 */
template <int N>
class Bit {
    static_assert(N > 0 && N <= 128, "The maximum supported bit width is 128");

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
     * @brief Constructor from a hexadecimal string literal..
     */
    explicit Bit(const char* hex_string) { parse_hex_string(hex_string); }

private:
    /**
     * ============ Private helper functions ============
     */
    // constructors, assignments or arithmetic operators MUST call
    // apply_mask() before they finish to clean up the result to preserve the
    // class invariant
    void apply_mask() {
        for (int i = 0; i < num_chunks; ++i)
            chunks[i] &= mask[i];
    }

    // Helper for the initializer_list constructor
    void copy_from_init(std::initializer_list<uint32_t> init) {
        chunks.fill(0);
        int i = 0;
        // Copy values from the list, ensuring we don't overflow our chunks
        // array
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

            // If the current chunk is full (32 bits), move to the next one
            if (bits_in_chunk == 32) {
                chunk_idx++;
                bits_in_chunk = 0;
                // Stop if we have filled all the chunks our Bit<N> can hold
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
        chunks.fill(0);
        for (int i = 0; i < num_chunks && value != 0; ++i) {
            chunks[i] = static_cast<uint32_t>(value);
            value >>= 32;
        }
        apply_mask();
    }

    /**
     * ============ Private constants & mask generation ============
     */
    // Number of chunks for the bit vector storage
    static constexpr int num_chunks = (N + 31) / 32;

    // The mask computation is static since the mask is shared by all the
    // objects with the same Bit<N> width
    static constexpr std::array<uint32_t, num_chunks> compute_mask() {
        // result accumulator
        std::array<uint32_t, num_chunks> mask{};
        // chunk by chunk
        for (int i = 0; i < num_chunks; ++i) {
            // msb chunk discriminator
            if (i == num_chunks - 1 && N % 32 != 0) {
                // N % 32 returns how many bits are used in the final chunk
                // 1U << n_bits moves a 1 left by n_bits, then by doing -1
                // we flip all the bits to the rhs and obtain the mask
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
