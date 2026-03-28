#include <ark_protocol.h>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "Dbghelp.lib")
#else
#include <unistd.h>
#include <execinfo.h>
#endif

// -----------------------------------------------------------------------------
// Platform I/O
// -----------------------------------------------------------------------------

#if defined(_WIN32)
using ark_write_result_t = int;

static inline int ark_stderr_fd() {
    return _fileno(stderr);
}

static inline ark_write_result_t ark_os_write(int fd, const char* s, size_t n) {
    if (n > static_cast<size_t>(INT_MAX)) {
        n = static_cast<size_t>(INT_MAX);
    }
    return ::_write(fd, s, static_cast<unsigned int>(n));
}
#else
using ark_write_result_t = ssize_t;

static inline int ark_stderr_fd() {
    return STDERR_FILENO;
}

static inline ark_write_result_t ark_os_write(int fd, const char* s, size_t n) {
    return ::write(fd, s, n);
}
#endif

// -----------------------------------------------------------------------------
// Helpers: Low-Level Output
// -----------------------------------------------------------------------------

static inline const char* ark_nn(const char* s, const char* fallback) {
    return s ? s : fallback;
}

static size_t ark_strlen(const char* s) {
    const char* p = s;
    while (*p) {
        ++p;
    }
    return static_cast<size_t>(p - s);
}

// Retrying write wrapper
static void ark_write_all(const char* s, size_t n) {
    int eintr_budget = 64;
    const int fd = ark_stderr_fd();

    while (n > 0) {
        ark_write_result_t w = ark_os_write(fd, s, n);

        if (w > 0) {
            s += static_cast<size_t>(w);
            n -= static_cast<size_t>(w);
            eintr_budget = 64;
            continue;
        }

        if (w < 0 && errno == EINTR && eintr_budget > 0) {
            --eintr_budget;
            continue;
        }

        break;
    }
}

// Compile-time size inference for string literals.
template <size_t N>
static inline void ark_write_lit(const char (&lit)[N]) {
    ark_write_all(lit, N - 1);
}

static inline void ark_write_ch(char c) {
    ark_write_all(&c, 1);
}

// -----------------------------------------------------------------------------
// Helpers: Integer / Hex Formatting
// -----------------------------------------------------------------------------

static size_t ark_fmt_u64(char* buf, uint64_t v) {
    char temp[32];
    size_t pos = 0;

    if (v == 0) {
        buf[0] = '0';
        return 1;
    }

    while (v > 0) {
        temp[pos++] = static_cast<char>('0' + (v % 10));
        v /= 10;
    }

    for (size_t i = 0; i < pos; ++i) {
        buf[i] = temp[pos - 1 - i];
    }

    return pos;
}

// Signal-safe integer formatter.
static size_t ark_fmt_i64(char* buf, int64_t v) {
    if (v == 0) {
        buf[0] = '0';
        return 1;
    }

    const bool neg = (v < 0);
    uint64_t uv = neg ? (static_cast<uint64_t>(-(v + 1)) + 1u) : static_cast<uint64_t>(v);

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

static void ark_write_i64(int64_t v) {
    char num[32];
    ark_write_all(num, ark_fmt_i64(num, v));
}

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

static void ark_write_hex_u64(uint64_t v, int width) {
    char buf[16];
    const size_t n = ark_fmt_hex(buf, v, width);
    ark_write_all(buf, n);
}

static void ark_write_ptr(const void* p) {
    ark_write_lit("0x");
    ark_write_hex_u64(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)),
        static_cast<int>(sizeof(uintptr_t) * 2)
    );
}

// -----------------------------------------------------------------------------
// Helpers: Stack Trace
// -----------------------------------------------------------------------------

static void ark_write_stacktrace() {
#if defined(_WIN32)
    ark_write_lit("Stack Trace:\n");

    void* frames[64] = {};
    const USHORT count = CaptureStackBackTrace(0, 64, frames, nullptr);
    if (count == 0) {
        ark_write_lit("  <unavailable>\n");
        return;
    }

    HANDLE process = GetCurrentProcess();
    (void)SymInitialize(process, nullptr, TRUE);

    unsigned char symStorage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(symStorage);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = MAX_SYM_NAME;

    for (USHORT i = 0; i < count; ++i) {
        DWORD64 displacement = 0;

        ark_write_lit("  #");
        ark_write_i64(static_cast<int64_t>(i));
        ark_write_lit(" ");

        const DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
        if (SymFromAddr(process, addr, &displacement, sym)) {
            ark_write_all(sym->Name, ark_strlen(sym->Name));
            ark_write_lit(" + 0x");
            ark_write_hex_u64(static_cast<uint64_t>(displacement), 8);
            ark_write_lit(" @ ");
            ark_write_ptr(frames[i]);
            ark_write_lit("\n");
        } else {
            ark_write_ptr(frames[i]);
            ark_write_lit("\n");
        }
    }
#else
    ark_write_lit("Stack Trace:\n");

    void* array[64];
    const int size = backtrace(array, 64);
    if (size <= 0) {
        ark_write_lit("  <unavailable>\n");
        return;
    }

    backtrace_symbols_fd(array, size, ark_stderr_fd());
#endif
}

// -----------------------------------------------------------------------------
// Helpers: Banner Output
// -----------------------------------------------------------------------------

static void ark_write_panic_header() {
#if defined(_WIN32)
    ark_write_lit("\n[ArkRuntime] PANIC: ");
#else
    ark_write_lit("\n\033[1;31m🚨 PANIC: ");
#endif
}

static void ark_write_panic_header_bounds() {
#if defined(_WIN32)
    ark_write_lit("\n[ArkRuntime] PANIC: Index Out of Bounds\n");
#else
    ark_write_lit("\n\033[1;31m🚨 PANIC: Index Out of Bounds\033[0m\n");
#endif
}

static void ark_write_color_reset_if_needed() {
#if !defined(_WIN32)
    ark_write_lit("\033[0m");
#endif
}

static void ark_terminate_now(int code) {
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#else
    _exit(code);
#endif
}

// -----------------------------------------------------------------------------
// Helpers: Internal Runtime Utilities
// -----------------------------------------------------------------------------

static inline bool ark_memeq_u8(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        uint64_t wa = 0;
        uint64_t wb = 0;
        std::memcpy(&wa, a + i, 8);
        std::memcpy(&wb, b + i, 8);

        if (wa != wb) {
            return false;
        }
    }

    for (; i < n; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Panic Implementations (Exported C ABI)
// -----------------------------------------------------------------------------

extern "C" {

void arkPanicAt(const char* msg, const char* file, int32_t line, int32_t col) {
    msg = ark_nn(msg, "panic");
    file = ark_nn(file, "<unknown>");

    ark_write_panic_header();
    ark_write_all(msg, ark_strlen(msg));
    ark_write_color_reset_if_needed();
    ark_write_lit("\n  at ");
    ark_write_all(file, ark_strlen(file));
    ark_write_lit(":");
    ark_write_i64(static_cast<int64_t>(line));
    ark_write_lit(":");
    ark_write_i64(static_cast<int64_t>(col));
    ark_write_lit("\n\n");

    ark_write_stacktrace();
    ark_terminate_now(134);
}

void arkPanicBounds(const char* file, int32_t line, int32_t col, int64_t idx, int64_t len) {
    file = ark_nn(file, "<unknown>");

    ark_write_panic_header_bounds();
    ark_write_lit("  index=");
    ark_write_i64(idx);
    ark_write_lit(", length=");
    ark_write_i64(len);
    ark_write_lit("\n  at ");
    ark_write_all(file, ark_strlen(file));
    ark_write_lit(":");
    ark_write_i64(static_cast<int64_t>(line));
    ark_write_lit(":");
    ark_write_i64(static_cast<int64_t>(col));
    ark_write_lit("\n");

    ark_terminate_now(134);
}

// Generic Panic
void panic(const char* msg) {
    arkPanicAt(msg, "<runtime>", 0, 0);
}

// -----------------------------------------------------------------------------
// Segfault / Abort Handler
// -----------------------------------------------------------------------------

void __ark_signal_handler(int sig) {
#if defined(_WIN32)
    ark_write_lit("\n[ArkRuntime] RUNTIME PANIC: Signal ");
#else
    ark_write_lit("\n\033[1;31m🚨 RUNTIME PANIC: Signal ");
#endif
    ark_write_i64(static_cast<int64_t>(sig));
#if defined(_WIN32)
    ark_write_lit("\n");
#else
    ark_write_lit("\033[0m\n");
#endif

    ark_write_stacktrace();
    ark_terminate_now(139);
}

} // extern "C"

// -----------------------------------------------------------------------------
// Auto-register the handler on load
// -----------------------------------------------------------------------------

struct SignalInstaller {
    SignalInstaller() {
#ifdef SIGSEGV
        std::signal(SIGSEGV, __ark_signal_handler);
#endif
#ifdef SIGABRT
        std::signal(SIGABRT, __ark_signal_handler);
#endif
#ifdef SIGFPE
        std::signal(SIGFPE, __ark_signal_handler);
#endif
#ifdef SIGILL
        std::signal(SIGILL, __ark_signal_handler);
#endif
    }
};

static SignalInstaller g_signal_installer;