#include <ark_protocol.h>

#include <cstdint>
#include <cstddef>
#include <stddef.h>   // ptrdiff_t
#include <cerrno>     // errno, EINTR
#include <climits>    // INT_MAX
#include <cstdio>
#include <cstring>    // std::memcpy
#include <cinttypes>  // PRIu64

#if defined(_WIN32)
#include <io.h>       // _write, _fileno
#else
#include <unistd.h>   // write, STDOUT_FILENO, STDERR_FILENO
#endif

// -----------------------------------------------------------------------------
// CONFIG: Output Stream Selection
// -----------------------------------------------------------------------------

#if defined(_WIN32)
static inline int ark_stdout_fd() { return _fileno(stdout); }
static inline int ark_stderr_fd() { return _fileno(stderr); }
#else
static inline int ark_stdout_fd() { return STDOUT_FILENO; }
static inline int ark_stderr_fd() { return STDERR_FILENO; }
#endif

#ifdef ARK_PRINT_TO_STDERR
static inline int ark_output_fd() { return ark_stderr_fd(); }
#else
static inline int ark_output_fd() { return ark_stdout_fd(); }
#endif

// -----------------------------------------------------------------------------
// Constants & Fallbacks
// -----------------------------------------------------------------------------

#ifndef ARK_IO_RETRY_BUDGET
#define ARK_IO_RETRY_BUDGET 64
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX INT_MAX
#endif

// -----------------------------------------------------------------------------
// Platform Write Shim
// Windows: _write
// POSIX:   write
// -----------------------------------------------------------------------------

#if defined(_WIN32)
using ark_write_result_t = int;

static inline ark_write_result_t ark_os_write(int fd, const char* s, size_t n) {
    if (n > static_cast<size_t>(INT_MAX)) {
        n = static_cast<size_t>(INT_MAX);
    }
    return ::_write(fd, s, static_cast<unsigned int>(n));
}
#else
using ark_write_result_t = ssize_t;

static inline ark_write_result_t ark_os_write(int fd, const char* s, size_t n) {
    return ::write(fd, s, n);
}
#endif

static inline void ark_memcpy(void* dst, const void* src, size_t n) {
    std::memcpy(dst, src, n);
}

static inline const char* ark_nn(const char* s, const char* fb) {
    return s ? s : fb;
}

// -----------------------------------------------------------------------------
// Helpers: Robust I/O (Lock-Free, 0-Spin, Bounded)
// -----------------------------------------------------------------------------

static size_t ark_strlen_bounded(const char* s, size_t cap) {
    size_t n = 0;
    while (n < cap && s[n]) {
        ++n;
    }
    return n;
}

static void ark_write_all_fd(int fd, const char* s, size_t n) {
    int eintr_budget = ARK_IO_RETRY_BUDGET;

    while (n > 0) {
        size_t chunk = n;
        if (chunk > static_cast<size_t>(SSIZE_MAX)) {
            chunk = static_cast<size_t>(SSIZE_MAX);
        }

        const ark_write_result_t w = ark_os_write(fd, s, chunk);

        if (w > 0) {
            s += static_cast<size_t>(w);
            n -= static_cast<size_t>(w);
            eintr_budget = ARK_IO_RETRY_BUDGET;
            continue;
        }

        if (w < 0 && errno == EINTR && eintr_budget > 0) {
            --eintr_budget;
            continue;
        }

        break;
    }
}

template <size_t N>
static inline void ark_wlit(int fd, const char (&lit)[N]) {
    ark_write_all_fd(fd, lit, N - 1);
}

static inline void ark_wch(int fd, char c) {
    ark_write_all_fd(fd, &c, 1);
}

// -----------------------------------------------------------------------------
// Helpers: Integer Formatting
// -----------------------------------------------------------------------------

static size_t ark_fmt_u64(char* buf, uint64_t v) {
    char tmp[32];
    size_t pos = 0;

    if (v == 0) {
        buf[0] = '0';
        return 1;
    }

    while (v > 0) {
        tmp[pos++] = static_cast<char>('0' + (v % 10));
        v /= 10;
    }

    for (size_t i = 0; i < pos; ++i) {
        buf[i] = tmp[pos - 1 - i];
    }

    return pos;
}

static size_t ark_fmt_i64(char* buf, int64_t v) {
    if (v == 0) {
        buf[0] = '0';
        return 1;
    }

    const bool neg = (v < 0);
    const uint64_t uv = neg ? (static_cast<uint64_t>(-(v + 1)) + 1u) : static_cast<uint64_t>(v);

    size_t n = ark_fmt_u64(buf, uv);
    if (!neg) {
        return n;
    }

    for (size_t i = n; i > 0; --i) {
        buf[i] = buf[i - 1];
    }
    buf[0] = '-';
    return n + 1;
}

static void ark_write_i64_fd(int fd, int64_t v) {
    char num[32];
    ark_write_all_fd(fd, num, ark_fmt_i64(num, v));
}

static void ark_write_u64_fd(int fd, uint64_t v) {
    char num[32];
    ark_write_all_fd(fd, num, ark_fmt_u64(num, v));
}

// -----------------------------------------------------------------------------
// Manual Float Formatting (Lock-Free, No stdio)
// -----------------------------------------------------------------------------

static inline bool ark_is_special_f64(uint64_t bits) {
    return ((bits >> 52) & 0x7FFu) == 0x7FFu;
}

static inline bool ark_is_zero_f64(uint64_t bits) {
    return (bits & 0x7FFFFFFFFFFFFFFFULL) == 0;
}

static bool ark_write_special_f64(int fd, uint64_t bits) {
    if (!ark_is_special_f64(bits)) {
        return false;
    }

    const bool sign = (bits >> 63) != 0;
    const uint64_t mant = bits & 0xFFFFFFFFFFFFFULL;

    if (mant == 0) {
        if (sign) {
            ark_wlit(fd, "-Infinity");
        } else {
            ark_wlit(fd, "Infinity");
        }
    } else {
        ark_wlit(fd, "NaN");
    }

    return true;
}

static inline uint64_t ark_f64_bits(double v) {
    uint64_t bits = 0;
    ark_memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static inline double ark_pow10_i(int e) {
    static const double p10_pos[] = {
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
        1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18
    };

    static const double p10_neg[] = {
        1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
        1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18
    };

    if (e >= 0) {
        if (e <= 18) {
            return p10_pos[e];
        }

        double x = p10_pos[18];
        for (int i = 18; i < e; ++i) {
            x *= 10.0;
        }
        return x;
    }

    const int ne = -e;
    if (ne <= 18) {
        return p10_neg[ne];
    }

    double x = p10_neg[18];
    for (int i = 18; i < ne; ++i) {
        x *= 0.1;
    }
    return x;
}

static inline int ark_floor_log10_from_exp2(int exp2) {
    // log10(2) ~= 0.30102999566
    // floor(exp2 * log10(2))
    // Good enough as a starting point; corrected by one step later.
    const double k = 0.30102999566398119521;
    const double x = static_cast<double>(exp2) * k;

    int e10 = static_cast<int>(x);
    if (x < 0.0 && static_cast<double>(e10) != x) {
        --e10;
    }

    return e10;
}

static void ark_write_scientific_rounded(int fd, double v, int mant_digits) {
    // v > 0 finite
    // Print: d.dddd e±NN (mant_digits after decimal, rounded)
    // Find base-10 exponent cheaply from base-2 exponent, then correct.

    uint64_t bits = ark_f64_bits(v);
    int exp2 = static_cast<int>((bits >> 52) & 0x7FFu) - 1023;

    if (((bits >> 52) & 0x7FFu) == 0) {
        // subnormal: normalize by scaling
        // shift into normal range: multiply by 2^54
        const double scale = 18014398509481984.0;
        v *= scale;
        bits = ark_f64_bits(v);
        exp2 = static_cast<int>((bits >> 52) & 0x7FFu) - 1023 - 54;
    }

    int e10 = ark_floor_log10_from_exp2(exp2);

    double s = v / ark_pow10_i(e10);
    if (s >= 10.0) {
        s *= 0.1;
        ++e10;
    } else if (s < 1.0) {
        s *= 10.0;
        --e10;
    }

    int n = mant_digits;
    if (n < 0) {
        n = 0;
    }
    if (n > 16) {
        n = 16;
    }

    // Generate digits: one before '.', n after '.', plus guard for rounding.
    char digs[32];
    double x = s;

    int d0 = static_cast<int>(x);
    if (static_cast<unsigned>(d0) > 9u) {
        d0 = 1;
    }
    digs[0] = static_cast<char>('0' + d0);
    x -= static_cast<double>(d0);

    for (int i = 0; i < n + 1; ++i) {
        x *= 10.0;
        int d = static_cast<int>(x);
        if (static_cast<unsigned>(d) > 9u) {
            d = 0;
        }
        digs[1 + i] = static_cast<char>('0' + d);
        x -= static_cast<double>(d);
    }

    // Round on guard digit.
    int carry = (n >= 0 && digs[1 + n] >= '5') ? 1 : 0;
    for (int i = n; i >= 1 && carry; --i) {
        const int d = (digs[i] - '0') + carry;
        if (d >= 10) {
            digs[i] = '0';
            carry = 1;
        } else {
            digs[i] = static_cast<char>('0' + d);
            carry = 0;
        }
    }

    if (carry) {
        // 9.999.. rounded -> 10.000.. => shift exponent.
        digs[0] = '1';
        for (int i = 1; i <= n; ++i) {
            digs[i] = '0';
        }
        ++e10;
    }

    ark_wch(fd, digs[0]);
    if (n > 0) {
        ark_wch(fd, '.');
        ark_write_all_fd(fd, &digs[1], static_cast<size_t>(n));
    } else {
        ark_wlit(fd, ".0");
    }

    ark_wch(fd, 'e');
    if (e10 >= 0) {
        ark_wch(fd, '+');
        ark_write_i64_fd(fd, static_cast<int64_t>(e10));
    } else {
        ark_write_i64_fd(fd, static_cast<int64_t>(e10));
    }
}

static void ark_write_f64_fd(int fd, double val) {
    const uint64_t bits = ark_f64_bits(val);

    if (ark_write_special_f64(fd, bits)) {
        return;
    }

    if (ark_is_zero_f64(bits)) {
        if (bits >> 63) {
            ark_wch(fd, '-');
        }
        ark_wlit(fd, "0.0");
        return;
    }

    if (bits >> 63) {
        ark_wch(fd, '-');
        val = -val;
    }

    // Policy: fixed when 1e-4 <= v < 1e15, otherwise scientific.
    const double kMinFixed = 1e-4;
    const double kMaxFixed = 1e15;

    if (val >= kMinFixed && val < kMaxFixed) {
        // 6 digits after decimal, rounded.
        // Near carry boundaries, fall back to scientific for correctness.
        const double ip_d = static_cast<double>(static_cast<uint64_t>(val));
        const double frac = val - ip_d;

        if (frac > 0.9999995) {
            ark_write_scientific_rounded(fd, val, 6);
            return;
        }

        const uint64_t ip = static_cast<uint64_t>(val);
        ark_write_u64_fd(fd, ip);
        ark_wch(fd, '.');

        double f = val - static_cast<double>(ip);

        // digits + guard
        char digits[8];
        for (int i = 0; i < 7; ++i) {
            f *= 10.0;
            int d = static_cast<int>(f);
            if (static_cast<unsigned>(d) > 9u) {
                d = 0;
            }
            digits[i] = static_cast<char>('0' + d);
            f -= static_cast<double>(d);
        }

        // round guard digits[6] into digits[0..5]
        int carry = (digits[6] >= '5') ? 1 : 0;
        for (int i = 5; i >= 0 && carry; --i) {
            const int d = (digits[i] - '0') + carry;
            if (d >= 10) {
                digits[i] = '0';
                carry = 1;
            } else {
                digits[i] = static_cast<char>('0' + d);
                carry = 0;
            }
        }

        if (carry) {
            ark_write_scientific_rounded(fd, val, 6);
            return;
        }

        ark_write_all_fd(fd, digits, 6);
        return;
    }

    ark_write_scientific_rounded(fd, val, 6);
}

// -----------------------------------------------------------------------------
// Vector Debug Printing
// -----------------------------------------------------------------------------

typedef void (*ArkElemPrinter)(int fd, const void* elem);

static void ark_vec_print_bounded(const ark_vec_t* v,
                                  int64_t elem_size,
                                  int64_t max_elems,
                                  ArkElemPrinter ep) {
    if (!v) {
        ark_wlit(ark_output_fd(), "vec(null)\n");
        return;
    }

    if (!ep) {
        ark_wlit(ark_output_fd(), "vec(<null_printer>)\n");
        return;
    }

    if (elem_size <= 0) {
        ark_wlit(ark_output_fd(), "vec(<bad_elem_size>)\n");
        return;
    }

    if (elem_size > static_cast<int64_t>(PTRDIFF_MAX)) {
        ark_wlit(ark_output_fd(), "vec(<elem_size_too_large>)\n");
        return;
    }

    if (v->len < 0 || v->cap < 0) {
        ark_wlit(ark_output_fd(), "vec(<corrupt_header>)\n");
        return;
    }

    if (v->len > v->cap) {
        ark_wlit(ark_output_fd(), "vec(<len_gt_cap>)\n");
        return;
    }

    if (v->len > 0 && !v->ptr) {
        ark_wlit(ark_output_fd(), "vec(<null_data>)\n");
        return;
    }

    const int fd = ark_output_fd();
    const int64_t n = v->len;
    const int64_t lim = (max_elems < 0) ? 0 : ((n < max_elems) ? n : max_elems);

    ark_wlit(fd, "vec{ len=");
    ark_write_i64_fd(fd, v->len);
    ark_wlit(fd, " cap=");
    ark_write_i64_fd(fd, v->cap);
    ark_wlit(fd, " data=[ ");

    const uint8_t* p = static_cast<const uint8_t*>(v->ptr);
    const uint64_t elem_size_u = static_cast<uint64_t>(elem_size);
    const uint64_t max_index = static_cast<uint64_t>(PTRDIFF_MAX) / elem_size_u;

    for (int64_t i = 0; i < lim; ++i) {
        if (i != 0) {
            ark_wlit(fd, " ");
        }

        const uint64_t idx = static_cast<uint64_t>(i);
        if (idx > max_index) {
            ark_wlit(fd, "<offset_overflow>");
            break;
        }

        const ptrdiff_t off = static_cast<ptrdiff_t>(idx * elem_size_u);
        ep(fd, static_cast<const void*>(p + off));
    }

    if (lim < n) {
        ark_wlit(fd, " ... ");
        ark_write_u64_fd(fd, static_cast<uint64_t>(n - lim));
        ark_wlit(fd, " more");
    }

    ark_wlit(fd, " ] }\n");
}

static void dbg_i32(int fd, const void* e) {
    ark_write_i64_fd(fd, static_cast<int64_t>(*static_cast<const int32_t*>(e)));
}

static void dbg_i64(int fd, const void* e) {
    ark_write_i64_fd(fd, *static_cast<const int64_t*>(e));
}

static void dbg_bool(int fd, const void* e) {
    (*static_cast<const uint8_t*>(e)) ? ark_wlit(fd, "true") : ark_wlit(fd, "false");
}

static void dbg_f32(int fd, const void* e) {
    float f = 0.0f;
    ark_memcpy(&f, e, sizeof(f));
    ark_write_f64_fd(fd, static_cast<double>(f));
}

static void dbg_f64(int fd, const void* e) {
    double d = 0.0;
    ark_memcpy(&d, e, sizeof(d));
    ark_write_f64_fd(fd, d);
}

static void dbg_str(int fd, const void* e) {
    const ArkStr* s = static_cast<const ArkStr*>(e);
    ark_wlit(fd, "\"");

    if (!s || !s->ptr) {
        ark_wlit(fd, "(null)");
    } else if (s->len < 0) {
        ark_wlit(fd, "(bad_len)");
    } else {
        uint64_t u = static_cast<uint64_t>(s->len);
        if (u > 1024) {
            u = 1024;
        }

        ark_write_all_fd(fd, s->ptr, static_cast<size_t>(u));
        if (static_cast<uint64_t>(s->len) > u) {
            ark_wlit(fd, "...");
        }
    }

    ark_wlit(fd, "\"");
}

// -----------------------------------------------------------------------------
// Hex Formatting (for *_raw exports)
// -----------------------------------------------------------------------------

static size_t ark_fmt_hex(char* buf, uint64_t v, int width) {
    static const char hex[] = "0123456789abcdef";

    if (width < 0) {
        width = 0;
    }
    if (width > 16) {
        width = 16;
    }

    for (int i = width - 1; i >= 0; --i) {
        buf[i] = hex[v & 0xFULL];
        v >>= 4;
    }

    return static_cast<size_t>(width);
}

static void dbg_f32_raw(int fd, const void* e) {
    uint32_t u = 0;
    ark_memcpy(&u, e, sizeof(u));

    char buf[8];
    ark_wlit(fd, "0x");
    ark_fmt_hex(buf, static_cast<uint64_t>(u), 8);
    ark_write_all_fd(fd, buf, 8);
}

static void dbg_f64_raw(int fd, const void* e) {
    uint64_t u = 0;
    ark_memcpy(&u, e, sizeof(u));

    char buf[16];
    ark_wlit(fd, "0x");
    ark_fmt_hex(buf, u, 16);
    ark_write_all_fd(fd, buf, 16);
}

// -----------------------------------------------------------------------------
// Exported C ABI
// -----------------------------------------------------------------------------

extern "C" {

void printNone() {
    ark_wlit(ark_output_fd(), "(none)");
}

void __ark_print_str(const char* ptr, int64_t len) {
    const int fd = ark_output_fd();

    if (!ptr) {
        return;
    }

    if (len < 0) {
        ark_wlit(fd, "(bad_len)");
        return;
    }

    if (len == 0) {
        return;
    }

    const uint64_t limit = static_cast<uint64_t>(1) << 20;
    uint64_t u = static_cast<uint64_t>(len);
    if (u > limit) {
        u = limit;
    }

    ark_write_all_fd(fd, ptr, static_cast<size_t>(u));
    if (static_cast<uint64_t>(len) > u) {
        ark_wlit(fd, "...");
    }
}

void printStr(const char* s) {
    const int fd = ark_output_fd();
    s = ark_nn(s, "(null)");
    ark_write_all_fd(fd, s, ark_strlen_bounded(s, static_cast<size_t>(1) << 20));
}

void printSpace(void) {
    ark_wlit(ark_output_fd(), " ");
}

void printNewline(void) {
    ark_wlit(ark_output_fd(), "\n");
}

void printI32(int32_t i) {
    ark_write_i64_fd(ark_output_fd(), static_cast<int64_t>(i));
}

void printI64(int64_t i) {
    ark_write_i64_fd(ark_output_fd(), i);
}

void printBool(uint8_t b) {
    b ? ark_wlit(ark_output_fd(), "true") : ark_wlit(ark_output_fd(), "false");
}

void printF32(float f) {
    ark_write_f64_fd(ark_output_fd(), static_cast<double>(f));
}

void printF64(double d) {
    ark_write_f64_fd(ark_output_fd(), d);
}

#ifndef ARK_PRINT_MAX_ELEMS
#define ARK_PRINT_MAX_ELEMS 32
#endif

#define ARK_VEC_PRINT_IMPL(name, type, printer) \
    void name(ark_vec_t* v) { \
        ark_vec_print_bounded(v, static_cast<int64_t>(sizeof(type)), ARK_PRINT_MAX_ELEMS, printer); \
    }

ARK_VEC_PRINT_IMPL(ark_vec_print_i32, int32_t, dbg_i32)
ARK_VEC_PRINT_IMPL(ark_vec_print_i64, int64_t, dbg_i64)
ARK_VEC_PRINT_IMPL(ark_vec_print_bool, uint8_t, dbg_bool)
ARK_VEC_PRINT_IMPL(ark_vec_print_f32, float, dbg_f32)
ARK_VEC_PRINT_IMPL(ark_vec_print_f64, double, dbg_f64)
ARK_VEC_PRINT_IMPL(ark_vec_print_str, ArkStr, dbg_str)
ARK_VEC_PRINT_IMPL(ark_vec_print_f32_raw, float, dbg_f32_raw)
ARK_VEC_PRINT_IMPL(ark_vec_print_f64_raw, double, dbg_f64_raw)

void printUnknown() {
    ark_wlit(ark_output_fd(), "<unknown type>");
}

// [FIX] Hex printer for raw pointers (addr(A))
void printRawPtr(void* ptr) {
    const int fd = ark_output_fd();

    if (!ptr) {
        ark_wlit(fd, "null");
        return;
    }

    ark_wlit(fd, "0x");

    char buf[sizeof(uintptr_t) * 2];
    const int width = static_cast<int>(sizeof(uintptr_t) * 2);
    ark_fmt_hex(buf, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)), width);
    ark_write_all_fd(fd, buf, static_cast<size_t>(width));
}

} // extern "C"