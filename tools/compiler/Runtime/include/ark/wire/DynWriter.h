#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <span>
#include <bit>

namespace ark::wire {

/// \brief A dynamically growing buffer writer (std::vector backend).
///
/// Unlike BufWriter (which is fixed-size/non-owning), DynWriter manages its
/// own memory and grows automatically. Useful for building packets/artifacts
/// where the final size is not known upfront.
class DynWriter {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    DynWriter() = default;

    /// \brief Construct with initial capacity reservation.
    explicit DynWriter(size_t initial_capacity) {
        buffer_.reserve(initial_capacity);
    }

    // -------------------------------------------------------------------------
    // State / Access
    // -------------------------------------------------------------------------

    /// \brief Reserve memory to prevent reallocations.
    void reserve(size_t capacity) {
        buffer_.reserve(capacity);
    }

    size_t size() const { return buffer_.size(); }
    size_t capacity() const { return buffer_.capacity(); }
    const uint8_t* data() const { return buffer_.data(); }

    /// \brief Access the constructed buffer.
    const std::vector<uint8_t>& output() const { return buffer_; }

    /// \brief Extract the buffer (move ownership).
    std::vector<uint8_t> take_output() { return std::move(buffer_); }

    // -------------------------------------------------------------------------
    // Primitive Writes (Little Endian)
    // -------------------------------------------------------------------------

    void write_u8(uint8_t v) {
        buffer_.push_back(v);
    }

    void write_u16(uint16_t v) {
        buffer_.push_back(static_cast<uint8_t>(v));
        buffer_.push_back(static_cast<uint8_t>(v >> 8));
    }

    void write_u32(uint32_t v) {
        buffer_.push_back(static_cast<uint8_t>(v));
        buffer_.push_back(static_cast<uint8_t>(v >> 8));
        buffer_.push_back(static_cast<uint8_t>(v >> 16));
        buffer_.push_back(static_cast<uint8_t>(v >> 24));
    }

    void write_u64(uint64_t v) {
        buffer_.push_back(static_cast<uint8_t>(v));
        buffer_.push_back(static_cast<uint8_t>(v >> 8));
        buffer_.push_back(static_cast<uint8_t>(v >> 16));
        buffer_.push_back(static_cast<uint8_t>(v >> 24));
        buffer_.push_back(static_cast<uint8_t>(v >> 32));
        buffer_.push_back(static_cast<uint8_t>(v >> 40));
        buffer_.push_back(static_cast<uint8_t>(v >> 48));
        buffer_.push_back(static_cast<uint8_t>(v >> 56));
    }

    // Signed Types
    void write_i8(int8_t v)   { write_u8(static_cast<uint8_t>(v)); }
    void write_i16(int16_t v) { write_u16(static_cast<uint16_t>(v)); }
    void write_i32(int32_t v) { write_u32(static_cast<uint32_t>(v)); }
    void write_i64(int64_t v) { write_u64(static_cast<uint64_t>(v)); }

    // Floating Point (IEEE-754)
    void write_f32(float v) {
        static_assert(sizeof(float) == 4, "float is not 32-bit");
        write_u32(std::bit_cast<uint32_t>(v));
    }

    void write_f64(double v) {
        static_assert(sizeof(double) == 8, "double is not 64-bit");
        write_u64(std::bit_cast<uint64_t>(v));
    }

    void write_bool(bool v) {
        write_u8(v ? 1 : 0);
    }

    // -------------------------------------------------------------------------
    // Bulk Writes
    // -------------------------------------------------------------------------

    void write_bytes(const void* src, size_t len) {
        if (len == 0) return;
        const uint8_t* p = static_cast<const uint8_t*>(src);
        buffer_.insert(buffer_.end(), p, p + len);
    }

    void write_span(std::span<const uint8_t> data) {
        buffer_.insert(buffer_.end(), data.begin(), data.end());
    }

    void write_string(std::string_view str) {
        buffer_.insert(buffer_.end(), str.begin(), str.end());
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /// \brief Zero-fill `len` bytes.
    void pad(size_t len) {
        if (len == 0) return;
        buffer_.resize(buffer_.size() + len, 0);
    }

    /// \brief Align the cursor to a power of 2, zero-filling if necessary.
    void align(size_t alignment) {
        if (alignment == 0) return;
        size_t mask = alignment - 1;
        size_t current_offset = buffer_.size();
        size_t padding = (alignment - (current_offset & mask)) & mask;
        pad(padding);
    }

private:
    std::vector<uint8_t> buffer_;
};

} // namespace ark::wire