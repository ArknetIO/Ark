#include <ark_protocol.h>
#include <cstdint>
#include <cstddef>
#include <unistd.h> // write, _exit, STDERR_FILENO
#include <errno.h>  // errno, EINTR
#include <signal.h> // signal, SIGSEGV
#include <execinfo.h> // backtrace
#include <stdlib.h> // free (for demangling)
#include <cxxabi.h> // __cxa_demangle

// -----------------------------------------------------------------------------
// Helpers: Signal-Safe Logic (C++ Linkage / Internal)
// -----------------------------------------------------------------------------
// MOVED OUTSIDE extern "C" because templates are not allowed inside.

static inline const char* ark_nn(const char* s, const char* fallback) {
    return s ? s : fallback;
}

static size_t ark_strlen(const char* s) {
    const char* p = s;
    while (*p) ++p;
    return (size_t)(p - s);
}

// Retrying write wrapper
static void ark_write_all(const char* s, size_t n) {
    int eintr_budget = 64;
    
    while (n > 0) {
        ssize_t w = ::write(STDERR_FILENO, s, n);
        
        if (w > 0) {
            s += (size_t)w;
            n -= (size_t)w;
            continue;
        }
        
        if (w < 0 && errno == EINTR) {
            if (--eintr_budget > 0) continue;
        }
        
        break; 
    }
}

// Compile-time size inference for string literals.
template <size_t N>
static inline void ark_write_lit(const char (&lit)[N]) {
    ark_write_all(lit, N - 1);
}

// Signal-safe integer formatter.
static size_t ark_fmt_i64(char* buf, int64_t v) {
    char temp[32];
    size_t pos = 0;
    if (v == 0) {
        buf[0] = '0';
        return 1;
    }
    const bool neg = (v < 0);
    uint64_t uv = neg ? (uint64_t)-(v + 1) + 1 : (uint64_t)v;
    
    while (uv > 0) {
        temp[pos++] = (char)('0' + (uv % 10));
        uv /= 10;
    }
    
    if (neg) temp[pos++] = '-';
    
    for (size_t i = 0; i < pos; ++i) {
        buf[i] = temp[pos - 1 - i];
    }
    return pos;
}

// -----------------------------------------------------------------------------
// Panic Implementations (Exported C ABI)
// -----------------------------------------------------------------------------
extern "C" {

void arkPanicAt(const char* msg, const char* file, int32_t line, int32_t col) {
    msg = ark_nn(msg, "panic");
    file = ark_nn(file, "<unknown>");

    ark_write_lit("\n\033[1;31m🚨 PANIC: ");
    ark_write_all(msg, ark_strlen(msg));
    ark_write_lit("\033[0m\n  at ");
    ark_write_all(file, ark_strlen(file));
    ark_write_lit(":");
    
    char num[32];
    ark_write_all(num, ark_fmt_i64(num, (int64_t)line));
    
    ark_write_lit(":");
    ark_write_all(num, ark_fmt_i64(num, (int64_t)col));
    ark_write_lit("\n\n");

    // Print stack trace if possible
    void* array[64];
    size_t size = backtrace(array, 64);
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    _exit(134);
}

void arkPanicBounds(const char* file, int32_t line, int32_t col, int64_t idx, int64_t len) {
    file = ark_nn(file, "<unknown>");
    
    ark_write_lit("\n\033[1;31m🚨 PANIC: Index Out of Bounds\033[0m\n");
    ark_write_lit("  index=");
    
    char num[32];
    ark_write_all(num, ark_fmt_i64(num, idx));
    
    ark_write_lit(", length=");
    ark_write_all(num, ark_fmt_i64(num, len));
    
    ark_write_lit("\n  at ");
    ark_write_all(file, ark_strlen(file));
    ark_write_lit(":");
    ark_write_all(num, ark_fmt_i64(num, (int64_t)line));
    ark_write_lit(":");
    ark_write_all(num, ark_fmt_i64(num, (int64_t)col));
    ark_write_lit("\n");

    _exit(134);
}

// Generic Panic
void panic(const char* msg) {
    arkPanicAt(msg, "<runtime>", 0, 0);
}

// -----------------------------------------------------------------------------
// Segfault Handler
// -----------------------------------------------------------------------------
void __ark_signal_handler(int sig) {
    ark_write_lit("\n\033[1;31m🚨 RUNTIME PANIC: Signal ");
    char num[32];
    ark_write_all(num, ark_fmt_i64(num, (int64_t)sig));
    ark_write_lit(" (Segmentation Fault)\033[0m\n");
    ark_write_lit("Stack Trace:\n");

    void* array[64];
    size_t size = backtrace(array, 64);
    
    // We try to be fancy, but if heap is corrupted, backtrace_symbols might fail.
    // backtrace_symbols_fd is safer as it writes directly to FD without malloc.
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    _exit(139);
}

// Auto-register the handler on load
struct SignalInstaller {
    SignalInstaller() {
        signal(SIGSEGV, __ark_signal_handler);
        signal(SIGABRT, __ark_signal_handler);
        signal(SIGFPE,  __ark_signal_handler);
    }
};

static SignalInstaller _installer;

static inline bool ark_memeq_u8(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint64_t wa = 0, wb = 0;
        __builtin_memcpy(&wa, a + i, 8);
        __builtin_memcpy(&wb, b + i, 8);
        if (wa != wb) return false;
    }

    for (; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }

    return true;
}

} // extern "C"