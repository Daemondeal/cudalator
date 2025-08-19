#include <array>
#include <cstdint>
#include <iomanip> // for std::hex, std::setw, std::setfill
#include <limits>
#include <sstream> // for std::stringstream
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
        chunks.fill(0);
        constexpr int rhs_chunks = (M + 31) / 32;
        constexpr int chunks_to_copy =
            num_chunks < rhs_chunks ? num_chunks : rhs_chunks;
        for (int i = 0; i < chunks_to_copy; ++i) {
            chunks[i] = rhs.chunks[i];
        }
        apply_mask();
    }

    /**
     * @brief Assignment from another Bit vector.
     * Handles assignment from both same-sized and different-sized Bit vectors.
     * Truncates if the source is larger, zero-extends if it is smaller.
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
