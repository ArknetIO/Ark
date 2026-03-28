// tools/compiler/Runtime/net/socket_win.cpp
//
// Windows socket runtime for Ark.
//
// Goals:
// - Preserve the POSIX-facing ABI exposed through ark_protocol.h.
// - Preserve the logical behavior of the POSIX runtime as closely as Windows allows.
// - Keep ArkStatus / ArkIoError contracts stable.
// - Keep socket handles CLOEXEC / non-inheritable whenever possible.
// - Avoid leaking raw SOCKET values through the ABI on 64-bit Windows.
//
// Notes:
// - The ABI uses int32_t logical socket ids, not raw SOCKET values.
// - timeout_ms <= 0 is treated as "infinite", matching the current implementation.
// - getaddrinfo failures are surfaced as negative status codes to preserve the
//   current distinction from errno-style failures.
// - recv() returning 0 is treated as EOF / peer closed and returns success with
//   an empty ArkBytes result.

#include <ark_protocol.h>

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <mutex>
#include <unordered_map>

#if defined(_MSC_VER)
#pragma comment(lib, "Ws2_32.lib")
#endif

// -----------------------------------------------------------------------------
// errno compatibility shims
//
// MSVC does not expose the full POSIX errno surface. The POSIX counterpart may
// use these symbolic values, so we provide conservative aliases that preserve
// behavior classes even when exact errno identity is unavailable on Windows.
// -----------------------------------------------------------------------------

#ifndef ENOTSUP
#define ENOTSUP EINVAL
#endif

#ifndef EWOULDBLOCK
#define EWOULDBLOCK EAGAIN
#endif

#ifndef EOPNOTSUPP
#define EOPNOTSUPP ENOTSUP
#endif

#ifndef ESOCKTNOSUPPORT
#define ESOCKTNOSUPPORT EOPNOTSUPP
#endif

#ifndef ETOOMANYREFS
#define ETOOMANYREFS EAGAIN
#endif

#ifndef EHOSTDOWN
#define EHOSTDOWN EHOSTUNREACH
#endif

#ifndef EUSERS
#define EUSERS EAGAIN
#endif

#ifndef EDQUOT
#define EDQUOT ENOSPC
#endif

#ifndef ESTALE
#define ESTALE EIO
#endif

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

static const int64_t MAX_SAFE_TIMEOUT_MS        = INT64_MAX / 1000000LL;
static const int64_t ARK_CONNECT_BACKOFF_START_NS = 1000LL;
static const int64_t ARK_CONNECT_BACKOFF_MAX_NS   = 1000000LL;
static const int64_t ARK_NS_PER_MS             = 1000000LL;
static const int64_t ARK_NS_PER_SEC            = 1000000000LL;

// -----------------------------------------------------------------------------
// String / allocation helpers
// -----------------------------------------------------------------------------

static inline const char* ark_strerror_cstr(int code, char* buf, size_t buflen) {
    if (!buf || buflen == 0) return "unknown error";

#if defined(_MSC_VER)
    strerror_s(buf, buflen, code);
    buf[buflen - 1] = '\0';
    return buf;
#else
    if (strerror_r(code, buf, buflen) == 0) return buf;
    return "unknown error";
#endif
}

static inline size_t ark_trim_message_len(const char* s, size_t n) {
    while (n != 0) {
        const char ch = s[n - 1];
        if (ch != '\r' && ch != '\n' && ch != ' ' && ch != '\t') break;
        --n;
    }
    return n;
}

extern "C" ArkStr __ark_str_from_raw_str(const char* s) {
    if (!s) return ArkStr{nullptr, 0};

    const size_t n = std::strlen(s);
    char* p = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    if (!p) return ArkStr{nullptr, 0};

    if (n != 0) {
        std::memcpy(p, s, n);
    }
    p[n] = '\0';
    return ArkStr{p, (int64_t)n};
}

static inline ArkStr ark_str_from_cstr(const char* s) {
    return __ark_str_from_raw_str(s);
}

static inline ArkStr ark_str_from_n(const char* s, size_t n) {
    if (!s) return ArkStr{nullptr, 0};

    char* p = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    if (!p) return ArkStr{nullptr, 0};

    if (n != 0) {
        std::memcpy(p, s, n);
    }
    p[n] = '\0';
    return ArkStr{p, (int64_t)n};
}

// -----------------------------------------------------------------------------
// Error translation
// -----------------------------------------------------------------------------

static inline int ark_winerr_to_errno(DWORD winerr) {
    switch (winerr) {
        case ERROR_SUCCESS:
            return 0;

        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:
        case ERROR_INVALID_DRIVE:
        case ERROR_NO_MORE_FILES:
        case ERROR_BAD_NETPATH:
        case ERROR_BAD_NET_NAME:
        case ERROR_BAD_PATHNAME:
        case ERROR_MOD_NOT_FOUND:
        case ERROR_PROC_NOT_FOUND:
            return ENOENT;

        case ERROR_TOO_MANY_OPEN_FILES:
            return EMFILE;

        case ERROR_ACCESS_DENIED:
        case ERROR_CURRENT_DIRECTORY:
        case ERROR_CANNOT_MAKE:
        case ERROR_NETWORK_ACCESS_DENIED:
        case ERROR_SHARING_VIOLATION:
            return EACCES;

        case ERROR_LOCK_VIOLATION:
        case ERROR_LOCKED:
        case ERROR_BUSY:
        case ERROR_OPEN_FILES:
            return EBUSY;

        case ERROR_INVALID_HANDLE:
            return EBADF;

        case ERROR_NOT_ENOUGH_MEMORY:
        case ERROR_OUTOFMEMORY:
            return ENOMEM;

        case ERROR_FILE_EXISTS:
        case ERROR_ALREADY_EXISTS:
            return EEXIST;

        case ERROR_BUFFER_OVERFLOW:
        case ERROR_FILENAME_EXCED_RANGE:
            return ENAMETOOLONG;

        case ERROR_DIRECTORY:
            return ENOTDIR;

        case ERROR_DIR_NOT_EMPTY:
            return ENOTEMPTY;

        case ERROR_BROKEN_PIPE:
        case ERROR_NO_DATA:
            return EPIPE;

        case ERROR_HANDLE_DISK_FULL:
        case ERROR_DISK_FULL:
            return ENOSPC;

        case ERROR_BAD_UNIT:
        case ERROR_NOT_READY:
        case ERROR_DEV_NOT_EXIST:
            return ENXIO;

        case ERROR_OPERATION_ABORTED:
            return ECANCELED;

        case ERROR_SEM_TIMEOUT:
        case WAIT_TIMEOUT:
            return ETIMEDOUT;

        case ERROR_PRIVILEGE_NOT_HELD:
            return EPERM;

        case ERROR_INVALID_PARAMETER:
        case ERROR_INVALID_ACCESS:
        case ERROR_INVALID_DATA:
            return EINVAL;

        case ERROR_CALL_NOT_IMPLEMENTED:
            return ENOSYS;

        case ERROR_NOT_SUPPORTED:
            return ENOTSUP;

        default:
            return EIO;
    }
}

static inline int ark_wsa_to_errno(int wsa_err) {
    switch (wsa_err) {
        case 0: return 0;

        case WSAEINTR:           return EINTR;
        case WSAEBADF:           return EBADF;
        case WSAEACCES:          return EACCES;
        case WSAEFAULT:          return EFAULT;
        case WSAEINVAL:          return EINVAL;
        case WSAEMFILE:          return EMFILE;
        case WSAEWOULDBLOCK:     return EWOULDBLOCK;
        case WSAEINPROGRESS:     return EINPROGRESS;
        case WSAEALREADY:        return EALREADY;
        case WSAENOTSOCK:        return ENOTSOCK;
        case WSAEDESTADDRREQ:    return EDESTADDRREQ;
        case WSAEMSGSIZE:        return EMSGSIZE;
        case WSAEPROTOTYPE:      return EPROTOTYPE;
        case WSAENOPROTOOPT:     return ENOPROTOOPT;
        case WSAEPROTONOSUPPORT: return EPROTONOSUPPORT;
        case WSAESOCKTNOSUPPORT: return ESOCKTNOSUPPORT;
        case WSAEOPNOTSUPP:      return EOPNOTSUPP;
        case WSAEPFNOSUPPORT:    return EAFNOSUPPORT;
        case WSAEAFNOSUPPORT:    return EAFNOSUPPORT;
        case WSAEADDRINUSE:      return EADDRINUSE;
        case WSAEADDRNOTAVAIL:   return EADDRNOTAVAIL;
        case WSAENETDOWN:        return ENETDOWN;
        case WSAENETUNREACH:     return ENETUNREACH;
        case WSAENETRESET:       return ENETRESET;
        case WSAECONNABORTED:    return ECONNABORTED;
        case WSAECONNRESET:      return ECONNRESET;
        case WSAENOBUFS:         return ENOBUFS;
        case WSAEISCONN:         return EISCONN;
        case WSAENOTCONN:        return ENOTCONN;
        case WSAESHUTDOWN:       return EPIPE;
        case WSAETOOMANYREFS:    return ETOOMANYREFS;
        case WSAETIMEDOUT:       return ETIMEDOUT;
        case WSAECONNREFUSED:    return ECONNREFUSED;
        case WSAELOOP:           return ELOOP;
        case WSAENAMETOOLONG:    return ENAMETOOLONG;
        case WSAEHOSTDOWN:       return EHOSTDOWN;
        case WSAEHOSTUNREACH:    return EHOSTUNREACH;
        case WSAENOTEMPTY:       return ENOTEMPTY;
        case WSAEPROCLIM:        return EAGAIN;
        case WSAEUSERS:          return EUSERS;
        case WSAEDQUOT:          return EDQUOT;
        case WSAESTALE:          return ESTALE;

        case WSASYSNOTREADY:     return EAGAIN;
        case WSAVERNOTSUPPORTED: return ENOSYS;
        case WSANOTINITIALISED:  return EIO;
        case WSAEDISCON:         return ECONNRESET;

        default:
            return EIO;
    }
}

static inline const char* ark_wsaerror_cstr(int wsa_err, char* buf, size_t buflen) {
    if (!buf || buflen == 0) return "unknown socket error";

    DWORD got = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS |
        FORMAT_MESSAGE_MAX_WIDTH_MASK,
        nullptr,
        (DWORD)wsa_err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buf,
        (DWORD)buflen,
        nullptr
    );

    if (got != 0) {
        const size_t n = ark_trim_message_len(buf, (size_t)got);
        buf[n] = '\0';
        return buf;
    }

    const int mapped = ark_wsa_to_errno(wsa_err);
    return ark_strerror_cstr(mapped, buf, buflen);
}

static inline void ark_err_clear(ArkIoError* e) {
    if (!e) return;
    e->code = 0;
    e->_pad = 0;
    e->msg.ptr = nullptr;
    e->msg.len = 0;
}

static inline void ark_err_set_errno(ArkIoError* e, int code) {
    if (!e) return;

    e->code = (int32_t)code;
    e->_pad = 0;
    e->msg.ptr = nullptr;
    e->msg.len = 0;

    char tmp[128];
    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    e->msg = ark_str_from_cstr(s ? s : "unknown error");
}

static inline void ark_err_set_win32(ArkIoError* e, DWORD winerr) {
    if (!e) return;

    const int code = ark_winerr_to_errno(winerr);
    e->code = (int32_t)code;
    e->_pad = 0;
    e->msg.ptr = nullptr;
    e->msg.len = 0;

    char tmp[256];
    DWORD got = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS |
        FORMAT_MESSAGE_MAX_WIDTH_MASK,
        nullptr,
        winerr,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        tmp,
        (DWORD)sizeof(tmp),
        nullptr
    );

    if (got != 0) {
        const size_t n = ark_trim_message_len(tmp, (size_t)got);
        e->msg = ark_str_from_n(tmp, n);
        return;
    }

    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    e->msg = ark_str_from_cstr(s ? s : "unknown error");
}

static inline void ark_err_set_wsa(ArkIoError* e, int wsa_err) {
    if (!e) return;

    const int code = ark_wsa_to_errno(wsa_err);
    e->code = (int32_t)code;
    e->_pad = 0;
    e->msg.ptr = nullptr;
    e->msg.len = 0;

    char tmp[256];
    const char* s = ark_wsaerror_cstr(wsa_err, tmp, sizeof(tmp));
    e->msg = ark_str_from_cstr(s ? s : "unknown socket error");
}

static inline void ark_err_set_code_msg(ArkIoError* e, int32_t code, const char* msg) {
    if (!e) return;
    e->code = code;
    e->_pad = 0;
    e->msg = ark_str_from_cstr(msg ? msg : "");
}

static inline void ark_err_setf(ArkIoError* e, int32_t code, const char* fmt, ...) {
    if (!e) return;

    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    buf[sizeof(buf) - 1] = '\0';
    e->code = code;
    e->_pad = 0;
    e->msg = ark_str_from_cstr(buf);
}

// -----------------------------------------------------------------------------
// Winsock runtime init
// -----------------------------------------------------------------------------

static std::once_flag g_wsa_once;
static int g_wsa_init_status = 0;

static void ark_wsa_init_once_impl() {
    WSADATA wsa{};
    const int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
    g_wsa_init_status = (rc == 0) ? 0 : ark_wsa_to_errno(rc);
}

static int ark_wsa_init(void) {
    std::call_once(g_wsa_once, ark_wsa_init_once_impl);
    return g_wsa_init_status;
}

// -----------------------------------------------------------------------------
// Logical socket registry
//
// ABI-facing code uses int32_t socket ids like POSIX file descriptors.
// On Windows, SOCKET is pointer-sized and cannot be exposed safely as int32_t.
// -----------------------------------------------------------------------------

static std::mutex g_sock_mu;
static std::unordered_map<int32_t, SOCKET> g_sock_table;
static int32_t g_next_sock_id = 1;

static int ark_sock_register(SOCKET sock, int32_t* out_id) {
    if (!out_id) return EINVAL;

    std::lock_guard<std::mutex> lock(g_sock_mu);

    for (int attempts = 0; attempts < INT_MAX; ++attempts) {
        int32_t id = g_next_sock_id++;
        if (g_next_sock_id <= 0) {
            g_next_sock_id = 1;
        }

        if (id <= 0) {
            continue;
        }

        if (g_sock_table.find(id) == g_sock_table.end()) {
            g_sock_table.emplace(id, sock);
            *out_id = id;
            return 0;
        }
    }

    return EMFILE;
}

static int ark_sock_lookup(int32_t id, SOCKET* out_sock) {
    if (!out_sock) return EINVAL;

    std::lock_guard<std::mutex> lock(g_sock_mu);
    auto it = g_sock_table.find(id);
    if (it == g_sock_table.end()) {
        return ENOTSOCK;
    }

    *out_sock = it->second;
    return 0;
}

static int ark_sock_take(int32_t id, SOCKET* out_sock) {
    if (!out_sock) return EINVAL;

    std::lock_guard<std::mutex> lock(g_sock_mu);
    auto it = g_sock_table.find(id);
    if (it == g_sock_table.end()) {
        return ENOTSOCK;
    }

    *out_sock = it->second;
    g_sock_table.erase(it);
    return 0;
}

// -----------------------------------------------------------------------------
// Socket configuration helpers
// -----------------------------------------------------------------------------

static int ensure_cloexec_socket(SOCKET sock) {
    if (sock == INVALID_SOCKET) return EBADF;

    if (!SetHandleInformation((HANDLE)sock, HANDLE_FLAG_INHERIT, 0)) {
        return ark_winerr_to_errno(GetLastError());
    }

    return 0;
}

static int set_nonblocking(SOCKET sock) {
    u_long mode = 1;
    if (ioctlsocket(sock, FIONBIO, &mode) != 0) {
        return ark_wsa_to_errno(WSAGetLastError());
    }
    return 0;
}

static int set_blocking(SOCKET sock) {
    u_long mode = 0;
    if (ioctlsocket(sock, FIONBIO, &mode) != 0) {
        return ark_wsa_to_errno(WSAGetLastError());
    }
    return 0;
}

static int apply_conn_defaults(SOCKET sock) {
    int one = 1;

    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, (int)sizeof(one)) != 0) {
        return ark_wsa_to_errno(WSAGetLastError());
    }

    if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char*)&one, (int)sizeof(one)) != 0) {
        return ark_wsa_to_errno(WSAGetLastError());
    }

    return 0;
}

static SOCKET create_stream_socket(int family, int type, int protocol) {
#if defined(WSA_FLAG_NO_HANDLE_INHERIT)
    {
        SOCKET ws = WSASocketA(family, type, protocol, nullptr, 0, WSA_FLAG_NO_HANDLE_INHERIT);
        if (ws != INVALID_SOCKET) {
            return ws;
        }
    }
#endif

    return socket(family, type, protocol);
}

static inline bool ark_connect_pending_errno(int e) {
    return e == EWOULDBLOCK || e == EINPROGRESS || e == EALREADY;
}

// -----------------------------------------------------------------------------
// Monotonic deadline helpers
// -----------------------------------------------------------------------------

struct ArkDeadline {
    int64_t deadline_ns;
    bool infinite;
};

static int now_mono_ns(int64_t* out_ns) {
    if (!out_ns) return EINVAL;

    LARGE_INTEGER freq{};
    LARGE_INTEGER ctr{};

    if (!QueryPerformanceFrequency(&freq) || !QueryPerformanceCounter(&ctr)) {
        return EIO;
    }

    if (freq.QuadPart <= 0 || ctr.QuadPart < 0) {
        return EIO;
    }

    const int64_t sec = (int64_t)(ctr.QuadPart / freq.QuadPart);
    const int64_t rem = (int64_t)(ctr.QuadPart % freq.QuadPart);

    if (sec > (INT64_MAX / ARK_NS_PER_SEC)) {
        return EOVERFLOW;
    }

    const int64_t sec_ns = sec * ARK_NS_PER_SEC;
    const int64_t rem_ns = (int64_t)((rem * ARK_NS_PER_SEC) / freq.QuadPart);

    if (sec_ns > INT64_MAX - rem_ns) {
        return EOVERFLOW;
    }

    *out_ns = sec_ns + rem_ns;
    return 0;
}

static int calc_deadline(int64_t timeout_ms, ArkDeadline* out_dl) {
    if (!out_dl) return EINVAL;

    if (timeout_ms <= 0) {
        out_dl->infinite = true;
        out_dl->deadline_ns = 0;
        return 0;
    }

    if (timeout_ms > MAX_SAFE_TIMEOUT_MS) {
        return EINVAL;
    }

    int64_t now_ns = 0;
    int err = now_mono_ns(&now_ns);
    if (err != 0) return err;

    const int64_t add_ns = timeout_ms * ARK_NS_PER_MS;
    if (now_ns > INT64_MAX - add_ns) {
        return EOVERFLOW;
    }

    out_dl->infinite = false;
    out_dl->deadline_ns = now_ns + add_ns;
    return 0;
}

static int get_remaining_ns(const ArkDeadline* dl, int64_t* out_ns) {
    if (!dl || !out_ns) return EINVAL;

    if (dl->infinite) {
        *out_ns = -1;
        return 0;
    }

    int64_t now_ns = 0;
    int err = now_mono_ns(&now_ns);
    if (err != 0) return err;

    *out_ns = (now_ns >= dl->deadline_ns) ? 0 : (dl->deadline_ns - now_ns);
    return 0;
}

// -----------------------------------------------------------------------------
// Connect wait helpers
// -----------------------------------------------------------------------------

static int backoff_sleep(int64_t* backoff_ns, const ArkDeadline* dl) {
    if (!backoff_ns || !dl) return EINVAL;

    int64_t current = *backoff_ns;
    if (current <= 0) {
        current = ARK_CONNECT_BACKOFF_START_NS;
    }

    if (!dl->infinite) {
        int64_t rem_ns = 0;
        int err = get_remaining_ns(dl, &rem_ns);
        if (err != 0) return err;
        if (rem_ns == 0) return ETIMEDOUT;
        if (current > rem_ns) current = rem_ns;
    }

    DWORD sleep_ms = (DWORD)(current / ARK_NS_PER_MS);
    if (current > 0 && sleep_ms == 0) {
        sleep_ms = 1;
    }

    Sleep(sleep_ms);

    *backoff_ns *= 2;
    if (*backoff_ns > ARK_CONNECT_BACKOFF_MAX_NS) {
        *backoff_ns = ARK_CONNECT_BACKOFF_MAX_NS;
    }

    return 0;
}

static int select_connect_event(SOCKET sock, const ArkDeadline* dl, int64_t* backoff_ns) {
    if (sock == INVALID_SOCKET || !dl || !backoff_ns) {
        return EINVAL;
    }

    for (;;) {
        fd_set wfds;
        fd_set efds;

        FD_ZERO(&wfds);
        FD_ZERO(&efds);
        FD_SET(sock, &wfds);
        FD_SET(sock, &efds);

        timeval tv{};
        timeval* tv_ptr = nullptr;

        if (!dl->infinite) {
            int64_t rem_ns = 0;
            int err = get_remaining_ns(dl, &rem_ns);
            if (err != 0) return err;
            if (rem_ns == 0) return ETIMEDOUT;

            long sec  = (long)(rem_ns / ARK_NS_PER_SEC);
            long usec = (long)((rem_ns % ARK_NS_PER_SEC) / 1000LL);

            if (sec < 0) sec = 0;
            if (usec < 0) usec = 0;

            tv.tv_sec = sec;
            tv.tv_usec = usec;
            tv_ptr = &tv;
        }

        const int rc = select(0, nullptr, &wfds, &efds, tv_ptr);
        if (rc == SOCKET_ERROR) {
            const int e = ark_wsa_to_errno(WSAGetLastError());
            if (e == EINTR) continue;
            return e;
        }

        if (rc == 0) {
            int bo_err = backoff_sleep(backoff_ns, dl);
            if (bo_err != 0) return bo_err;
            continue;
        }

        if (!FD_ISSET(sock, &wfds) && !FD_ISSET(sock, &efds)) {
            int bo_err = backoff_sleep(backoff_ns, dl);
            if (bo_err != 0) return bo_err;
            continue;
        }

        return 0;
    }
}

// -----------------------------------------------------------------------------
// Networking ABI
// -----------------------------------------------------------------------------

extern "C" {

ArkStatus __ark_net_connect(const char* host, int32_t port, int64_t timeout_ms, int32_t* out_sockfd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_sockfd) *out_sockfd = -1;

    const int init_err = ark_wsa_init();
    if (init_err != 0) {
        ark_err_set_errno(out_err, init_err);
        return (ArkStatus)init_err;
    }

    if (!out_sockfd || !host || !*host) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (port <= 0 || port > 65535) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (timeout_ms < 0 || timeout_ms > MAX_SAFE_TIMEOUT_MS) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char port_str[16];
    (void)snprintf(port_str, sizeof(port_str), "%d", (int)port);
    port_str[sizeof(port_str) - 1] = '\0';

    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_ADDRCONFIG | AI_NUMERICSERV;

    addrinfo* res = nullptr;
    const int gai_rc = getaddrinfo(host, port_str, &hints, &res);
    if (gai_rc != 0) {
        ark_err_set_code_msg(out_err, (int32_t)(-gai_rc), gai_strerrorA(gai_rc));
        return (ArkStatus)(-gai_rc);
    }

    ArkDeadline deadline{};
    const int dl_err = calc_deadline(timeout_ms, &deadline);
    if (dl_err != 0) {
        freeaddrinfo(res);
        ark_err_set_errno(out_err, dl_err);
        return (ArkStatus)dl_err;
    }

    SOCKET sock = INVALID_SOCKET;
    int32_t last_code = (int32_t)ECONNREFUSED;

    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        if (!deadline.infinite) {
            int64_t rem_ns = 0;
            const int rem_err = get_remaining_ns(&deadline, &rem_ns);
            if (rem_err != 0) {
                last_code = (int32_t)rem_err;
                break;
            }
            if (rem_ns == 0) {
                last_code = (int32_t)ETIMEDOUT;
                break;
            }
        }

        sock = create_stream_socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (sock == INVALID_SOCKET) {
            last_code = (int32_t)ark_wsa_to_errno(WSAGetLastError());
            continue;
        }

        {
            const int ce = ensure_cloexec_socket(sock);
            if (ce != 0) {
                last_code = (int32_t)ce;
                (void)closesocket(sock);
                sock = INVALID_SOCKET;
                continue;
            }
        }

        {
            const int opt_err = apply_conn_defaults(sock);
            if (opt_err != 0) {
                last_code = (int32_t)opt_err;
                (void)closesocket(sock);
                sock = INVALID_SOCKET;
                continue;
            }
        }

        {
            const int nb_err = set_nonblocking(sock);
            if (nb_err != 0) {
                last_code = (int32_t)nb_err;
                (void)closesocket(sock);
                sock = INVALID_SOCKET;
                continue;
            }
        }

        const int c_rc = connect(sock, p->ai_addr, (int)p->ai_addrlen);
        if (c_rc == SOCKET_ERROR) {
            const int e = ark_wsa_to_errno(WSAGetLastError());

            if (!ark_connect_pending_errno(e)) {
                last_code = (int32_t)e;
                (void)closesocket(sock);
                sock = INVALID_SOCKET;
                continue;
            }

            int64_t backoff_ns = ARK_CONNECT_BACKOFF_START_NS;
            bool connected = false;

            for (;;) {
                const int poll_err = select_connect_event(sock, &deadline, &backoff_ns);
                if (poll_err != 0) {
                    last_code = (int32_t)poll_err;
                    break;
                }

                int so_error = 0;
                int so_error_len = (int)sizeof(so_error);
                if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&so_error, &so_error_len) != 0) {
                    last_code = (int32_t)ark_wsa_to_errno(WSAGetLastError());
                    break;
                }

                if (so_error == 0) {
                    connected = true;
                    break;
                }

                const int mapped = ark_wsa_to_errno(so_error);
                if (ark_connect_pending_errno(mapped)) {
                    continue;
                }

                last_code = (int32_t)mapped;
                break;
            }

            if (!connected) {
                (void)closesocket(sock);
                sock = INVALID_SOCKET;

                if (last_code == (int32_t)ETIMEDOUT) {
                    break;
                }
                continue;
            }
        }

        {
            const int blk_err = set_blocking(sock);
            if (blk_err != 0) {
                last_code = (int32_t)blk_err;
                (void)closesocket(sock);
                sock = INVALID_SOCKET;
                continue;
            }
        }

        int32_t logical_id = -1;
        const int reg_err = ark_sock_register(sock, &logical_id);
        if (reg_err != 0) {
            last_code = (int32_t)reg_err;
            (void)closesocket(sock);
            sock = INVALID_SOCKET;
            break;
        }

        *out_sockfd = logical_id;
        freeaddrinfo(res);
        return 0;
    }

    freeaddrinfo(res);

    if (sock != INVALID_SOCKET) {
        (void)closesocket(sock);
    }

    ark_err_setf(out_err, last_code, "connect failed to %s:%d", host, (int)port);
    return (ArkStatus)last_code;
}

ArkStatus __ark_net_listen(const char* host, int32_t port, int32_t* out_sockfd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_sockfd) *out_sockfd = -1;

    const int init_err = ark_wsa_init();
    if (init_err != 0) {
        ark_err_set_errno(out_err, init_err);
        return (ArkStatus)init_err;
    }

    if (!out_sockfd) {
        ark_err_set_code_msg(out_err, EINVAL, "out_sockfd is null");
        return (ArkStatus)EINVAL;
    }

    if (port <= 0 || port > 65535) {
        ark_err_setf(out_err, EINVAL, "invalid port: %d", (int)port);
        return (ArkStatus)EINVAL;
    }

    char port_str[16];
    (void)snprintf(port_str, sizeof(port_str), "%d", (int)port);
    port_str[sizeof(port_str) - 1] = '\0';

    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_PASSIVE | AI_ADDRCONFIG | AI_NUMERICSERV;

    addrinfo* res = nullptr;
    const int gai_rc = getaddrinfo(host, port_str, &hints, &res);
    if (gai_rc != 0) {
        ark_err_set_code_msg(out_err, (int32_t)(-gai_rc), gai_strerrorA(gai_rc));
        return (ArkStatus)(-gai_rc);
    }

    SOCKET listen_sock = INVALID_SOCKET;
    int32_t last_code = (int32_t)EADDRINUSE;

    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        listen_sock = create_stream_socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (listen_sock == INVALID_SOCKET) {
            last_code = (int32_t)ark_wsa_to_errno(WSAGetLastError());
            continue;
        }

        {
            const int ce = ensure_cloexec_socket(listen_sock);
            if (ce != 0) {
                last_code = (int32_t)ce;
                (void)closesocket(listen_sock);
                listen_sock = INVALID_SOCKET;
                continue;
            }
        }

        // POSIX counterpart likely uses SO_REUSEADDR. Windows semantics differ,
        // but this is still the closest practical match for restart-friendly bind.
        {
            int one = 1;
            (void)setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, (int)sizeof(one));
        }

        if (bind(listen_sock, p->ai_addr, (int)p->ai_addrlen) == SOCKET_ERROR) {
            last_code = (int32_t)ark_wsa_to_errno(WSAGetLastError());
            (void)closesocket(listen_sock);
            listen_sock = INVALID_SOCKET;
            continue;
        }

        if (listen(listen_sock, 128) == SOCKET_ERROR) {
            last_code = (int32_t)ark_wsa_to_errno(WSAGetLastError());
            (void)closesocket(listen_sock);
            listen_sock = INVALID_SOCKET;
            continue;
        }

        int32_t logical_id = -1;
        const int reg_err = ark_sock_register(listen_sock, &logical_id);
        if (reg_err != 0) {
            last_code = (int32_t)reg_err;
            (void)closesocket(listen_sock);
            listen_sock = INVALID_SOCKET;
            break;
        }

        *out_sockfd = logical_id;
        freeaddrinfo(res);
        return 0;
    }

    freeaddrinfo(res);

    if (listen_sock != INVALID_SOCKET) {
        (void)closesocket(listen_sock);
    }

    ark_err_setf(out_err, last_code, "listen bind failed for %s:%d", host ? host : "*", (int)port);
    return (ArkStatus)last_code;
}

ArkStatus __ark_net_accept(int32_t sockfd, int32_t* out_clientfd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_clientfd) *out_clientfd = -1;

    const int init_err = ark_wsa_init();
    if (init_err != 0) {
        ark_err_set_errno(out_err, init_err);
        return (ArkStatus)init_err;
    }

    if (!out_clientfd) {
        ark_err_set_code_msg(out_err, EINVAL, "out_clientfd is null");
        return (ArkStatus)EINVAL;
    }

    SOCKET listen_sock = INVALID_SOCKET;
    const int lk_err = ark_sock_lookup(sockfd, &listen_sock);
    if (lk_err != 0) {
        ark_err_set_errno(out_err, lk_err);
        return (ArkStatus)lk_err;
    }

    sockaddr_storage peer_addr{};
    int peer_addr_len = (int)sizeof(peer_addr);

    SOCKET client_sock = INVALID_SOCKET;
    for (;;) {
        peer_addr_len = (int)sizeof(peer_addr);
        client_sock = accept(listen_sock, (sockaddr*)&peer_addr, &peer_addr_len);
        if (client_sock == INVALID_SOCKET) {
            const int e = ark_wsa_to_errno(WSAGetLastError());
            if (e == EINTR) {
                continue;
            }
            ark_err_setf(out_err, (int32_t)e, "accept failed (fd=%d)", (int)sockfd);
            return (ArkStatus)e;
        }
        break;
    }

    {
        const int ce = ensure_cloexec_socket(client_sock);
        if (ce != 0) {
            (void)closesocket(client_sock);
            ark_err_setf(out_err, (int32_t)ce, "accept cloexec failed (fd=%d)", (int)sockfd);
            return (ArkStatus)ce;
        }
    }

    {
        const int opt_err = apply_conn_defaults(client_sock);
        if (opt_err != 0) {
            (void)closesocket(client_sock);
            ark_err_setf(out_err, (int32_t)opt_err, "accept socket opts failed (fd=%d)", (int)sockfd);
            return (ArkStatus)opt_err;
        }
    }

    int32_t logical_id = -1;
    const int reg_err = ark_sock_register(client_sock, &logical_id);
    if (reg_err != 0) {
        (void)closesocket(client_sock);
        ark_err_set_errno(out_err, reg_err);
        return (ArkStatus)reg_err;
    }

    *out_clientfd = logical_id;
    return 0;
}

ArkStatus __ark_net_send(int32_t sockfd, const void* data, int64_t len, ArkIoError* out_err) {
    ark_err_clear(out_err);

    const int init_err = ark_wsa_init();
    if (init_err != 0) {
        ark_err_set_errno(out_err, init_err);
        return (ArkStatus)init_err;
    }

    if (len < 0 || (len > 0 && !data)) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    SOCKET sock = INVALID_SOCKET;
    const int lk_err = ark_sock_lookup(sockfd, &sock);
    if (lk_err != 0) {
        ark_err_set_errno(out_err, lk_err);
        return (ArkStatus)lk_err;
    }

    const uint8_t* p = (const uint8_t*)data;
    int64_t rem = len;

    while (rem > 0) {
        const int chunk = (rem > (int64_t)INT_MAX) ? INT_MAX : (int)rem;
        const int w = send(sock, (const char*)p, chunk, 0);

        if (w == SOCKET_ERROR) {
            const int e = ark_wsa_to_errno(WSAGetLastError());
            if (e == EINTR) continue;
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        if (w == 0) {
            ark_err_set_errno(out_err, EIO);
            return (ArkStatus)EIO;
        }

        p += (size_t)w;
        rem -= (int64_t)w;
    }

    return 0;
}

ArkStatus __ark_net_recv(int32_t sockfd, int64_t max_len, ArkBytes* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) {
        out->ptr = nullptr;
        out->len = 0;
        out->cap = 0;
    }

    const int init_err = ark_wsa_init();
    if (init_err != 0) {
        ark_err_set_errno(out_err, init_err);
        return (ArkStatus)init_err;
    }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    SOCKET sock = INVALID_SOCKET;
    const int lk_err = ark_sock_lookup(sockfd, &sock);
    if (lk_err != 0) {
        ark_err_set_errno(out_err, lk_err);
        return (ArkStatus)lk_err;
    }

    if (max_len <= 0) {
        max_len = 4096;
    }

    if (max_len > ARK_NET_MAX_PACKET) {
        ark_err_set_errno(out_err, EFBIG);
        return (ArkStatus)EFBIG;
    }

    if (max_len > (int64_t)INT_MAX) {
        max_len = (int64_t)INT_MAX;
    }

    uint8_t* buf = (uint8_t*)__ark_alloc((uint64_t)max_len, 16);
    if (!buf) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    int r = 0;
    for (;;) {
        r = recv(sock, (char*)buf, (int)max_len, 0);
        if (r == SOCKET_ERROR) {
            const int e = ark_wsa_to_errno(WSAGetLastError());
            if (e == EINTR) continue;

            __ark_free(buf);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        break;
    }

    if (r == 0) {
        __ark_free(buf);
        return 0;
    }

    out->ptr = buf;
    out->len = (int64_t)r;
    out->cap = (int64_t)max_len;
    return 0;
}

ArkStatus __ark_net_close(int32_t sockfd, ArkIoError* out_err) {
    ark_err_clear(out_err);

    const int init_err = ark_wsa_init();
    if (init_err != 0) {
        ark_err_set_errno(out_err, init_err);
        return (ArkStatus)init_err;
    }

    if (sockfd < 0) {
        return 0;
    }

    SOCKET sock = INVALID_SOCKET;
    const int tk_err = ark_sock_take(sockfd, &sock);
    if (tk_err != 0) {
        ark_err_set_errno(out_err, tk_err);
        return (ArkStatus)tk_err;
    }

    if (closesocket(sock) == SOCKET_ERROR) {
        const int wsa_err = WSAGetLastError();
        ark_err_set_wsa(out_err, wsa_err);
        return (ArkStatus)ark_wsa_to_errno(wsa_err);
    }

    return 0;
}

} // extern "C"