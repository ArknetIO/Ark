#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <bit>
#include <span>
#include <string_view>
#include <limits>
#include <algorithm>

namespace ark::wire {

/// \brief A high-performance, safe, non-owning buffer writer.
///
/// Wraps a fixed-size memory region and provides sequential write operations
/// with strict bounds checking. All multi-byte integers are written in
/// Little Endian format to ensure ABI stability across platforms.
class BufWriter {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// \brief Construct from a raw pointer and capacity.
    /// \warning The caller must ensure [buf, buf + cap) is valid memory.
    BufWriter(uint8_t* buf, size_t cap) 
        : start_(buf), curr_(buf), end_(buf + cap) {}

    /// \brief Construct from a std::span (Safe).
    explicit BufWriter(std::span<uint8_t> buffer)
        : BufWriter(buffer.data(), buffer.size()) {}

    BufWriter(const BufWriter&) = default;
    BufWriter& operator=(const BufWriter&) = default;

    // -------------------------------------------------------------------------
    // State Query
    // -------------------------------------------------------------------------

    /// \brief Returns the number of bytes written so far.
    [[nodiscard]] size_t size() const { return static_cast<size_t>(curr_ - start_); }

    /// \brief Returns the total capacity of the underlying buffer.
    [[nodiscard]] size_t capacity() const { return static_cast<size_t>(end_ - start_); }

    /// \brief Returns the number of bytes remaining in the buffer.
    [[nodiscard]] size_t remaining() const { return static_cast<size_t>(end_ - curr_); }

    /// \brief Returns the pointer to the start of the buffer.
    [[nodiscard]] const uint8_t* data() const { return start_; }

    /// \brief Returns the pointer to the current write position.
    [[nodiscard]] uint8_t* cursor() { return curr_; }

    // -------------------------------------------------------------------------
    // Cursor Management
    // -------------------------------------------------------------------------

    /// \brief Advance the cursor forward by `len` bytes without writing.
    /// \return true if successful, false if out of bounds.
    [[nodiscard]] bool skip(size_t len) {
        if (remaining() < len) return false;
        curr_ += len;
        return true;
    }

    /// \brief Move the cursor to an absolute offset.
    /// \return true if `offset` is within [0, capacity].
    [[nodiscard]] bool seek(size_t offset) {
        if (offset > capacity()) return false;
        curr_ = start_ + offset;
        return true;
    }

    /// \brief Align the cursor to the next multiple of `alignment`.
    /// \param alignment Must be a power of 2.
    /// \return true if padding was successful (or unnecessary), false if OOB.
    [[nodiscard]] bool align(size_t alignment) {
        if (alignment == 0) return false;
        size_t mask = alignment - 1;
        size_t current_offset = size();
        size_t padding = (alignment - (current_offset & mask)) & mask;
        return pad(padding);
    }

    // -------------------------------------------------------------------------
    // Primitive Writes (Little Endian)
    // -------------------------------------------------------------------------

    [[nodiscard]] bool write_u8(uint8_t v) {
        if (remaining() < 1) return false;
        *curr_++ = v;
        return true;
    }

    [[nodiscard]] bool write_u16(uint16_t v) {
        if (remaining() < 2) return false;
        curr_[0] = static_cast<uint8_t>(v);
        curr_[1] = static_cast<uint8_t>(v >> 8);
        curr_ += 2;
        return true;
    }

    [[nodiscard]] bool write_u32(uint32_t v) {
        if (remaining() < 4) return false;
        curr_[0] = static_cast<uint8_t>(v);
        curr_[1] = static_cast<uint8_t>(v >> 8);
        curr_[2] = static_cast<uint8_t>(v >> 16);
        curr_[3] = static_cast<uint8_t>(v >> 24);
        curr_ += 4;
        return true;
    }

    [[nodiscard]] bool write_u64(uint64_t v) {
        if (remaining() < 8) return false;
        curr_[0] = static_cast<uint8_t>(v);
        curr_[1] = static_cast<uint8_t>(v >> 8);
        curr_[2] = static_cast<uint8_t>(v >> 16);
        curr_[3] = static_cast<uint8_t>(v >> 24);
        curr_[4] = static_cast<uint8_t>(v >> 32);
        curr_[5] = static_cast<uint8_t>(v >> 40);
        curr_[6] = static_cast<uint8_t>(v >> 48);
        curr_[7] = static_cast<uint8_t>(v >> 56);
        curr_ += 8;
        return true;
    }

    // Signed Integers (Cast to unsigned)
    [[nodiscard]] bool write_i8(int8_t v)   { return write_u8(static_cast<uint8_t>(v)); }
    [[nodiscard]] bool write_i16(int16_t v) { return write_u16(static_cast<uint16_t>(v)); }
    [[nodiscard]] bool write_i32(int32_t v) { return write_u32(static_cast<uint32_t>(v)); }
    [[nodiscard]] bool write_i64(int64_t v) { return write_u64(static_cast<uint64_t>(v)); }

    // Floating Point (IEEE-754 assumed via memcpy/bit_cast)
    [[nodiscard]] bool write_f32(float v) {
        static_assert(sizeof(float) == 4, "float is not 32-bit");
        return write_u32(std::bit_cast<uint32_t>(v));
    }

    [[nodiscard]] bool write_f64(double v) {
        static_assert(sizeof(double) == 8, "double is not 64-bit");
        return write_u64(std::bit_cast<uint64_t>(v));
    }

    [[nodiscard]] bool write_bool(bool v) {
        return write_u8(v ? 1 : 0);
    }

    // -------------------------------------------------------------------------
    // Bulk Writes
    // -------------------------------------------------------------------------

    /// \brief Write raw bytes from a pointer.
    [[nodiscard]] bool write_bytes(const void* src, size_t len) {
        if (len == 0) return true;
        if (remaining() < len) return false;
        std::memcpy(curr_, src, len);
        curr_ += len;
        return true;
    }

    /// \brief Write bytes from a span.
    [[nodiscard]] bool write_span(std::span<const uint8_t> data) {
        return write_bytes(data.data(), data.size());
    }

    /// \brief Write a string (without null terminator).
    [[nodiscard]] bool write_string(std::string_view str) {
        return write_bytes(str.data(), str.size());
    }

    /// \brief Write a C-string (without null terminator).
    [[nodiscard]] bool write_cstr(const char* str) {
        return write_bytes(str, std::strlen(str));
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /// \brief Zero-fill the next `len` bytes.
    [[nodiscard]] bool pad(size_t len) {
        if (len == 0) return true;
        if (remaining() < len) return false;
        std::memset(curr_, 0, len);
        curr_ += len;
        return true;
    }

private:
    uint8_t* start_;
    uint8_t* curr_;
    uint8_t* end_;
};

// Helper for expressive return checking (e.g. if (is_ok(w.write_u8(...))))
constexpr bool is_ok(bool status) { return status; }

} // namespace ark::wire