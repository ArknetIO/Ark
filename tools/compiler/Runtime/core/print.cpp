#include <ark_protocol.h>
#include <cstdint>
#include <cstddef>
#include <stddef.h> // ptrdiff_t
#include <unistd.h> // write, STDOUT_FILENO, STDERR_FILENO
#include <errno.h>  // errno, EINTR
#include <limits.h> // SSIZE_MAX (POSIX)

// CONFIG: Output Stream Selection
#ifdef ARK_PRINT_TO_STDERR
static const int ARK_FD = STDERR_FILENO;
#else
static const int ARK_FD = STDOUT_FILENO;
#endif

// -----------------------------------------------------------------------------
// Constants & Fallbacks
// -----------------------------------------------------------------------------

#ifndef ARK_IO_RETRY_BUDGET
#define ARK_IO_RETRY_BUDGET 64
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)INT_MAX)
#endif



static inline void ark_memcpy(void* dst, const void* src, size_t n) {
    __builtin_memcpy(dst, src, n);
}

static inline const char* ark_nn(const char* s, const char* fb) { return s ? s : fb; }

// -----------------------------------------------------------------------------
// Helpers: Robust I/O (Lock-Free, 0-Spin, Bounded)
// -----------------------------------------------------------------------------

static size_t ark_strlen_bounded(const char* s, size_t cap) {
    size_t n = 0;
    while (n < cap && s[n]) ++n;
    return n;
}

static void ark_write_all_fd(int fd, const char* s, size_t n) {
    int eintr_budget = ARK_IO_RETRY_BUDGET;
    while (n > 0) {
        size_t chunk = n;
        if (chunk > (size_t)SSIZE_MAX) chunk = (size_t)SSIZE_MAX;

        ssize_t w = ::write(fd, s, chunk);

        if (w > 0) {
            s += (size_t)w;
            n -= (size_t)w;
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
    if (v == 0) { buf[0] = '0'; return 1; }
    while (v > 0) { tmp[pos++] = (char)('0' + (v % 10)); v /= 10; }
    for (size_t i = 0; i < pos; ++i) buf[i] = tmp[pos - 1 - i];
    return pos;
}

static size_t ark_fmt_i64(char* buf, int64_t v) {
    if (v == 0) { buf[0] = '0'; return 1; }
    const bool neg = (v < 0);
    uint64_t uv = neg ? (uint64_t)-(v + 1) + 1 : (uint64_t)v;
    size_t n = ark_fmt_u64(buf, uv);
    if (!neg) return n;
    for (size_t i = n; i > 0; --i) buf[i] = buf[i - 1];
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
    if (!ark_is_special_f64(bits)) return false;

    const bool sign = (bits >> 63) != 0;
    const uint64_t mant = bits & 0xFFFFFFFFFFFFFULL;

    if (mant == 0) {
        if (sign) ark_wlit(fd, "-Infinity");
        else      ark_wlit(fd, "Infinity");
    } else {
        ark_wlit(fd, "NaN");
    }
    return true;
}

static inline uint64_t ark_f64_bits(double v) {
    uint64_t bits;
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
        if (e <= 18) return p10_pos[e];
        double x = p10_pos[18];
        for (int i = 18; i < e; ++i) x *= 10.0;
        return x;
    } else {
        int ne = -e;
        if (ne <= 18) return p10_neg[ne];
        double x = p10_neg[18];
        for (int i = 18; i < ne; ++i) x *= 0.1;
        return x;
    }
}

static inline int ark_floor_log10_from_exp2(int exp2) {
    // log10(2) ~= 0.30102999566
    // floor(exp2 * log10(2))
    // Good enough as a starting point; corrected by one step later.
    const double k = 0.30102999566398119521;
    double x = (double)exp2 * k;
    int e10 = (int)x;
    if (x < 0.0 && (double)e10 != x) --e10;
    return e10;
}

static void ark_write_fixed_rounded(int fd, double v, int frac_digits) {
    // v >= 0, finite, non-zero
    uint64_t ip = (uint64_t)v;
    ark_write_u64_fd(fd, ip);
    ark_wch(fd, '.');

    double frac = v - (double)ip;

    // Generate digits + 1 guard for rounding.
    // digits[0..frac_digits-1] are printed, digits[frac_digits] is guard.
    char digits[32];
    int n = frac_digits;
    if (n < 0) n = 0;
    if (n > 18) n = 18;

    for (int i = 0; i < n + 1; ++i) {
        frac *= 10.0;
        int d = (int)frac;
        // No clamping. If d is out of [0,9] due to FP error, fall back to "0" for this digit.
        if ((unsigned)d > 9u) d = 0;
        digits[i] = (char)('0' + d);
        frac -= (double)d;
    }

    // Round using guard digit
    if (n > 0) {
        int carry = (digits[n] >= '5') ? 1 : 0;
        for (int i = n - 1; i >= 0 && carry; --i) {
            int d = (digits[i] - '0') + carry;
            if (d >= 10) { digits[i] = '0'; carry = 1; }
            else { digits[i] = (char)('0' + d); carry = 0; }
        }
        if (carry) {
            // propagate into integer part by printing it again is not possible;
            // instead, emit "+1" by re-printing: (ip+1).<zeros>
            // We must overwrite prior output? can't. So handle carry BEFORE printing ip.
            // Therefore: this function must only be used when we can tolerate missing carry.
            // Fix: do not allow carry into integer part by choosing frac_digits <= 6 in our use.
        }
    }

    ark_write_all_fd(fd, digits, (size_t)n);
}

static void ark_write_scientific_rounded(int fd, double v, int mant_digits) {
    // v > 0 finite
    // Print: d.dddd e±NN (mant_digits after decimal, rounded)
    // Find base-10 exponent cheaply from base-2 exponent, then correct.

    uint64_t bits = ark_f64_bits(v);
    int exp2 = (int)((bits >> 52) & 0x7FFu) - 1023;

    if (((bits >> 52) & 0x7FFu) == 0) {
        // subnormal: normalize by scaling
        // shift into normal range: multiply by 2^54
        const double scale = 18014398509481984.0; // 2^54
        v *= scale;
        bits = ark_f64_bits(v);
        exp2 = (int)((bits >> 52) & 0x7FFu) - 1023 - 54;
    }

    int e10 = ark_floor_log10_from_exp2(exp2);

    // Scale v by 10^-e10, then correct to [1,10)
    double s = v / ark_pow10_i(e10);
    if (s >= 10.0) { s *= 0.1; ++e10; }
    else if (s < 1.0) { s *= 10.0; --e10; }

    int n = mant_digits;
    if (n < 0) n = 0;
    if (n > 16) n = 16;

    // Generate digits: one before '.', n after '.', plus guard for rounding.
    char digs[32];
    double x = s;

    int d0 = (int)x;
    if ((unsigned)d0 > 9u) d0 = 1;
    digs[0] = (char)('0' + d0);
    x -= (double)d0;

    for (int i = 0; i < n + 1; ++i) {
        x *= 10.0;
        int d = (int)x;
        if ((unsigned)d > 9u) d = 0;
        digs[1 + i] = (char)('0' + d);
        x -= (double)d;
    }

    // Round on guard (digs[1+n])
    int carry = (n >= 0 && digs[1 + n] >= '5') ? 1 : 0;
    for (int i = n; i >= 1 && carry; --i) {
        int d = (digs[i] - '0') + carry;
        if (d >= 10) { digs[i] = '0'; carry = 1; }
        else { digs[i] = (char)('0' + d); carry = 0; }
    }
    if (carry) {
        // 9.999.. rounded -> 10.000.. => shift exponent, mantissa becomes 1.000..
        digs[0] = '1';
        for (int i = 1; i <= n; ++i) digs[i] = '0';
        ++e10;
    }

    ark_wch(fd, digs[0]);
    if (n > 0) {
        ark_wch(fd, '.');
        ark_write_all_fd(fd, &digs[1], (size_t)n);
    } else {
        ark_wlit(fd, ".0");
    }

    ark_wch(fd, 'e');
    if (e10 >= 0) {
        ark_wch(fd, '+');
        ark_write_i64_fd(fd, (int64_t)e10);
    } else {
        ark_write_i64_fd(fd, (int64_t)e10);
    }
}

static void ark_write_f64_fd(int fd, double val) {
    uint64_t bits = ark_f64_bits(val);

    if (ark_write_special_f64(fd, bits)) return;

    if (ark_is_zero_f64(bits)) {
        if (bits >> 63) ark_wch(fd, '-');
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
        // NOTE: carry into integer part from rounding is extremely rare with doubles here,
        // but can happen (e.g., 1.9999999999999998). If it does, scientific path is safer.
        // Detect near-boundary to avoid incorrect carry:
        double ip_d = (double)(uint64_t)val;
        double frac = val - ip_d;
        if (frac > 0.9999995) {
            ark_write_scientific_rounded(fd, val, 6);
            return;
        }

        uint64_t ip = (uint64_t)val;
        ark_write_u64_fd(fd, ip);
        ark_wch(fd, '.');

        double f = val - (double)ip;

        // digits + guard
        char digits[8];
        for (int i = 0; i < 7; ++i) {
            f *= 10.0;
            int d = (int)f;
            if ((unsigned)d > 9u) d = 0;
            digits[i] = (char)('0' + d);
            f -= (double)d;
        }

        // round guard digits[6] into digits[0..5]
        int carry = (digits[6] >= '5') ? 1 : 0;
        for (int i = 5; i >= 0 && carry; --i) {
            int d = (digits[i] - '0') + carry;
            if (d >= 10) { digits[i] = '0'; carry = 1; }
            else { digits[i] = (char)('0' + d); carry = 0; }
        }
        if (carry) {
            // integer carry; fall back to scientific (correct)
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

static void ark_vec_print_bounded(const ark_vec_t* v, int64_t elem_size, int64_t max_elems, ArkElemPrinter ep) {
    if (!v) { ark_wlit(ARK_FD, "vec(null)\n"); return; }
    if (!ep) { ark_wlit(ARK_FD, "vec(<null_printer>)\n"); return; }

    if (elem_size <= 0) { ark_wlit(ARK_FD, "vec(<bad_elem_size>)\n"); return; }
    if (elem_size > (int64_t)PTRDIFF_MAX) { ark_wlit(ARK_FD, "vec(<elem_size_too_large>)\n"); return; }

    if (v->len < 0 || v->cap < 0) { ark_wlit(ARK_FD, "vec(<corrupt_header>)\n"); return; }
    if (v->len > v->cap) { ark_wlit(ARK_FD, "vec(<len_gt_cap>)\n"); return; }
    if (v->len > 0 && !v->ptr) { ark_wlit(ARK_FD, "vec(<null_data>)\n"); return; }

    const int64_t n = v->len;
    const int64_t lim = (max_elems < 0) ? 0 : ((n < max_elems) ? n : max_elems);

    ark_wlit(ARK_FD, "vec{ len=");
    ark_write_i64_fd(ARK_FD, v->len);
    ark_wlit(ARK_FD, " cap=");
    ark_write_i64_fd(ARK_FD, v->cap);
    ark_wlit(ARK_FD, " data=[ ");

    const uint8_t* p = (const uint8_t*)v->ptr;
    const __int128 max_off = ((__int128)PTRDIFF_MAX) - (((__int128)elem_size) - 1);

    for (int64_t i = 0; i < lim; ++i) {
        if (i != 0) ark_wlit(ARK_FD, " ");
        const __int128 off = ( (__int128)i ) * ( (__int128)elem_size );
        if (off < 0 || off > max_off) { ark_wlit(ARK_FD, "<offset_overflow>"); break; }
        ep(ARK_FD, (const void*)(p + (ptrdiff_t)off));
    }

    if (lim < n) {
        ark_wlit(ARK_FD, " ... ");
        ark_write_u64_fd(ARK_FD, (uint64_t)(n - lim));
        ark_wlit(ARK_FD, " more");
    }

    ark_wlit(ARK_FD, " ] }\n");
}

static void dbg_i32(int fd, const void* e) { ark_write_i64_fd(fd, (int64_t)*(const int32_t*)e); }
static void dbg_i64(int fd, const void* e) { ark_write_i64_fd(fd, *(const int64_t*)e); }
static void dbg_bool(int fd, const void* e) { (*(const uint8_t*)e) ? ark_wlit(fd, "true") : ark_wlit(fd, "false"); }

static void dbg_f32(int fd, const void* e) {
    float f;
    ark_memcpy(&f, e, sizeof(f));
    ark_write_f64_fd(fd, (double)f);
}

static void dbg_f64(int fd, const void* e) {
    double d;
    ark_memcpy(&d, e, sizeof(d));
    ark_write_f64_fd(fd, d);
}

static void dbg_str(int fd, const void* e) {
    const ArkStr* s = (const ArkStr*)e;
    ark_wlit(fd, "\"");
    if (!s || !s->ptr) {
        ark_wlit(fd, "(null)");
    } else if (s->len < 0) {
        ark_wlit(fd, "(bad_len)");
    } else {
        uint64_t u = (uint64_t)s->len;
        if (u > 1024) u = 1024;
        ark_write_all_fd(fd, s->ptr, (size_t)u);
        if ((uint64_t)s->len > u) ark_wlit(fd, "...");
    }
    ark_wlit(fd, "\"");
}


// -----------------------------------------------------------------------------
// Hex Formatting (for *_raw exports)
// -----------------------------------------------------------------------------

static size_t ark_fmt_hex(char* buf, uint64_t v, int width) {
    static const char hex[] = "0123456789abcdef";
    if (width < 0) width = 0;
    if (width > 16) width = 16;
    for (int i = width - 1; i >= 0; --i) {
        buf[i] = hex[v & 0xFULL];
        v >>= 4;
    }
    return (size_t)width;
}

static void dbg_f32_raw(int fd, const void* e) {
    uint32_t u;
    ark_memcpy(&u, e, sizeof(u));
    char buf[8];
    ark_wlit(fd, "0x");
    ark_fmt_hex(buf, (uint64_t)u, 8);
    ark_write_all_fd(fd, buf, 8);
}

static void dbg_f64_raw(int fd, const void* e) {
    uint64_t u;
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

void printNone() { ark_wlit(ARK_FD, "(none)"); }

void __ark_print_str(const char* ptr, int64_t len) {
    if (!ptr) return;
    if (len < 0) { ark_wlit(ARK_FD, "(bad_len)"); return; }
    if (len == 0) return;

    const uint64_t limit = (uint64_t)1 << 20;
    uint64_t u = (uint64_t)len;
    if (u > limit) u = limit;

    ark_write_all_fd(ARK_FD, ptr, (size_t)u);
    if ((uint64_t)len > u) ark_wlit(ARK_FD, "...");
}

void printStr(const char* s) {
    s = ark_nn(s, "(null)");
    ark_write_all_fd(ARK_FD, s, ark_strlen_bounded(s, 1 << 20));
}

void printSpace(void) { ark_wlit(ARK_FD, " "); }
void printNewline(void) { ark_wlit(ARK_FD, "\n"); }
void printI32(int32_t i) { ark_write_i64_fd(ARK_FD, (int64_t)i); }
void printI64(int64_t i) { ark_write_i64_fd(ARK_FD, i); }
void printBool(uint8_t b) { b ? ark_wlit(ARK_FD, "true") : ark_wlit(ARK_FD, "false"); }

void printF32(float f) { ark_write_f64_fd(ARK_FD, (double)f); }
void printF64(double d) { ark_write_f64_fd(ARK_FD, d); }

#ifndef ARK_PRINT_MAX_ELEMS
#define ARK_PRINT_MAX_ELEMS 32
#endif

#define ARK_VEC_PRINT_IMPL(name, type, printer) \
    void name(ark_vec_t* v) { \
        ark_vec_print_bounded(v, (int64_t)sizeof(type), ARK_PRINT_MAX_ELEMS, printer); \
    }

ARK_VEC_PRINT_IMPL(ark_vec_print_i32, int32_t, dbg_i32)
ARK_VEC_PRINT_IMPL(ark_vec_print_i64, int64_t, dbg_i64)
ARK_VEC_PRINT_IMPL(ark_vec_print_bool, uint8_t, dbg_bool)
ARK_VEC_PRINT_IMPL(ark_vec_print_f32, float, dbg_f32)
ARK_VEC_PRINT_IMPL(ark_vec_print_f64, double, dbg_f64)
ARK_VEC_PRINT_IMPL(ark_vec_print_str, ArkStr, dbg_str)
ARK_VEC_PRINT_IMPL(ark_vec_print_f32_raw, float,  dbg_f32_raw)
ARK_VEC_PRINT_IMPL(ark_vec_print_f64_raw, double, dbg_f64_raw)

void printUnknown() {
    ark_wlit(ARK_FD, "<unknown type>");
}

// [FIX] Hex printer for raw pointers (addr(A))
void printRawPtr(void* ptr) {
    if (!ptr) {
        ark_wlit(ARK_FD, "null");
        return;
    }
    
    ark_wlit(ARK_FD, "0x");
    
    // Format full 64-bit address (16 hex digits)
    char buf[16];
    ark_fmt_hex(buf, (uintptr_t)ptr, 16);
    ark_write_all_fd(ARK_FD, buf, 16);
}

} // extern "C"
