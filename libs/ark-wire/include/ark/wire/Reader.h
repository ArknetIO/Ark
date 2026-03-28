#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <bit>
#include <span>
#include <string_view>
#include <limits>

namespace ark::wire {

/// \brief Status codes for Reader operations.
enum class ReaderStatus {
    Ok = 0,
    Eof,        ///< Not enough bytes remaining
    Overflow,   ///< Value too large for destination type
    Invalid     ///< Malformed data (e.g. invalid bool byte)
};

/// \brief Helper to check status in conditions (e.g. if (is_ok(r.read(...))))
constexpr bool is_ok(ReaderStatus s) { return s == ReaderStatus::Ok; }

/// \brief A high-performance, safe, non-owning buffer reader.
///
/// Provides sequential read operations from a memory block.
/// Guaranteed safe unaligned reads on all architectures via byte shifting.
class Reader {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// \brief Construct from pointer and size.
    Reader(const uint8_t* data, size_t size) 
        : start_(data), curr_(data), end_(data + size) {}

    /// \brief Construct from std::span (Safe).
    explicit Reader(std::span<const uint8_t> buffer)
        : Reader(buffer.data(), buffer.size()) {}

    Reader(const Reader&) = default;
    Reader& operator=(const Reader&) = default;

    // -------------------------------------------------------------------------
    // State Query
    // -------------------------------------------------------------------------

    [[nodiscard]] size_t offset() const { return static_cast<size_t>(curr_ - start_); }
    [[nodiscard]] size_t remaining() const { return static_cast<size_t>(end_ - curr_); }
    [[nodiscard]] const uint8_t* data() const { return start_; }
    [[nodiscard]] const uint8_t* cursor() const { return curr_; }

    // -------------------------------------------------------------------------
    // Cursor Management
    // -------------------------------------------------------------------------

    [[nodiscard]] ReaderStatus skip(size_t len) {
        if (remaining() < len) return ReaderStatus::Eof;
        curr_ += len;
        return ReaderStatus::Ok;
    }

    [[nodiscard]] ReaderStatus seek(size_t offset) {
        if (offset > static_cast<size_t>(end_ - start_)) return ReaderStatus::Eof;
        curr_ = start_ + offset;
        return ReaderStatus::Ok;
    }

    // -------------------------------------------------------------------------
    // Primitive Reads (Little Endian)
    // -------------------------------------------------------------------------

    [[nodiscard]] ReaderStatus read_u8(uint8_t& out) {
        if (remaining() < 1) return ReaderStatus::Eof;
        out = *curr_++;
        return ReaderStatus::Ok;
    }

    [[nodiscard]] ReaderStatus read_u16(uint16_t& out) {
        if (remaining() < 2) return ReaderStatus::Eof;
        uint16_t v = curr_[0];
        v |= (static_cast<uint16_t>(curr_[1]) << 8);
        out = v;
        curr_ += 2;
        return ReaderStatus::Ok;
    }

    [[nodiscard]] ReaderStatus read_u32(uint32_t& out) {
        if (remaining() < 4) return ReaderStatus::Eof;
        uint32_t v = curr_[0];
        v |= (static_cast<uint32_t>(curr_[1]) << 8);
        v |= (static_cast<uint32_t>(curr_[2]) << 16);
        v |= (static_cast<uint32_t>(curr_[3]) << 24);
        out = v;
        curr_ += 4;
        return ReaderStatus::Ok;
    }

    [[nodiscard]] ReaderStatus read_u64(uint64_t& out) {
        if (remaining() < 8) return ReaderStatus::Eof;
        uint64_t v = curr_[0];
        v |= (static_cast<uint64_t>(curr_[1]) << 8);
        v |= (static_cast<uint64_t>(curr_[2]) << 16);
        v |= (static_cast<uint64_t>(curr_[3]) << 24);
        v |= (static_cast<uint64_t>(curr_[4]) << 32);
        v |= (static_cast<uint64_t>(curr_[5]) << 40);
        v |= (static_cast<uint64_t>(curr_[6]) << 48);
        v |= (static_cast<uint64_t>(curr_[7]) << 56);
        out = v;
        curr_ += 8;
        return ReaderStatus::Ok;
    }

    // Signed Integers (Bit-cast)
    [[nodiscard]] ReaderStatus read_i8(int8_t& out) {
        uint8_t v;
        auto s = read_u8(v);
        if (is_ok(s)) out = static_cast<int8_t>(v);
        return s;
    }

    [[nodiscard]] ReaderStatus read_i16(int16_t& out) {
        uint16_t v;
        auto s = read_u16(v);
        if (is_ok(s)) out = static_cast<int16_t>(v);
        return s;
    }

    [[nodiscard]] ReaderStatus read_i32(int32_t& out) {
        uint32_t v;
        auto s = read_u32(v);
        if (is_ok(s)) out = static_cast<int32_t>(v);
        return s;
    }

    [[nodiscard]] ReaderStatus read_i64(int64_t& out) {
        uint64_t v;
        auto s = read_u64(v);
        if (is_ok(s)) out = static_cast<int64_t>(v);
        return s;
    }

    // Floating Point
    [[nodiscard]] ReaderStatus read_f32(float& out) {
        uint32_t v;
        auto s = read_u32(v);
        if (is_ok(s)) out = std::bit_cast<float>(v);
        return s;
    }

    [[nodiscard]] ReaderStatus read_f64(double& out) {
        uint64_t v;
        auto s = read_u64(v);
        if (is_ok(s)) out = std::bit_cast<double>(v);
        return s;
    }

    [[nodiscard]] ReaderStatus read_bool(bool& out) {
        uint8_t v;
        auto s = read_u8(v);
        if (!is_ok(s)) return s;
        if (v > 1) return ReaderStatus::Invalid; // Strict bool check
        out = (v != 0);
        return ReaderStatus::Ok;
    }

    // -------------------------------------------------------------------------
    // Buffer Reads
    // -------------------------------------------------------------------------

    /// \brief Zero-copy view. Pointer remains valid only as long as source buffer exists.
    [[nodiscard]] ReaderStatus read_bytes_view(const uint8_t*& out_ptr, size_t len) {
        if (remaining() < len) return ReaderStatus::Eof;
        out_ptr = curr_;
        curr_ += len;
        return ReaderStatus::Ok;
    }

    /// \brief Zero-copy span view.
    [[nodiscard]] ReaderStatus read_span(std::span<const uint8_t>& out_span, size_t len) {
        if (remaining() < len) return ReaderStatus::Eof;
        out_span = {curr_, len};
        curr_ += len;
        return ReaderStatus::Ok;
    }

    /// \brief Zero-copy string view.
    [[nodiscard]] ReaderStatus read_string_view(std::string_view& out_sv, size_t len) {
        if (remaining() < len) return ReaderStatus::Eof;
        out_sv = {reinterpret_cast<const char*>(curr_), len};
        curr_ += len;
        return ReaderStatus::Ok;
    }

    /// \brief Copy into destination buffer.
    [[nodiscard]] ReaderStatus read_bytes_copy(void* dst, size_t len) {
        if (remaining() < len) return ReaderStatus::Eof;
        std::memcpy(dst, curr_, len);
        curr_ += len;
        return ReaderStatus::Ok;
    }

private:
    const uint8_t* start_;
    const uint8_t* curr_;
    const uint8_t* end_;
};

} // namespace ark::wire