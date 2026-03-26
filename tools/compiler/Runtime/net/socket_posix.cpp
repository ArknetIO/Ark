// tools/arkc/Runtime/net/socket_posix.cpp
#include <ark_protocol.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <climits>
#include <ctime>
#include <cstdarg>

#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#if defined(__APPLE__)
    #define ARK_MSG_NOSIGNAL 0
    #define ARK_SOCK_CLOEXEC 0
#else
    #define ARK_MSG_NOSIGNAL MSG_NOSIGNAL
    #define ARK_SOCK_CLOEXEC SOCK_CLOEXEC
#endif



// -----------------------------------------------------------------------------
// Helpers: Memory & String
// -----------------------------------------------------------------------------

// Add this in socket_posix.cpp (near the existing helpers), so all new ArkStatus ABI
// code can call ark_err_set_errno() safely.

static inline const char* ark_strerror_cstr(int code, char* buf, size_t buflen) {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
    return strerror_r(code, buf, buflen);
#else
    if (strerror_r(code, buf, buflen) == 0) return buf;
    return "unknown error";
#endif
}


extern "C" ArkStr __ark_str_from_raw_str(const char* s) {
    // 1. Safety check
    if (!s) return ArkStr{nullptr, 0};

    // 2. Calculate length
    const size_t n = std::strlen(s);

    // 3. Allocate with Alignment!
    // Signature: void* __ark_alloc(uint64_t bytes, uint64_t align)
    // We use alignment '1' for char arrays (or '8' if you prefer word alignment, but 1 is sufficient for strings).
    char* p = (char*)__ark_alloc((uint64_t)(n + 1), 1);

    // 4. OOM Check
    if (!p) return ArkStr{nullptr, 0};

    // 5. Copy & Null-Terminate
    if (n > 0) {
        std::memcpy(p, s, n);
    }
    p[n] = '\0';

    // 6. Return standard ABI struct
    return ArkStr{p, (int64_t)n};
}

// Internal helper: Wraps the ABI function __ark_str_from_raw
// This ensures we use the correct tracked allocator (memory.cpp)
static inline ArkStr ark_str_from_cstr(const char* s) {
    // __ark_str_from_raw handles NULL checks, allocation, and null-termination
    return __ark_str_from_raw_str(s);
}


static inline void ark_err_set_errno(ArkIoError* e, int code) {
    if (!e) return;

    e->code = (int32_t)code;
    e->_pad = 0;

    // reset msg (do not free here; caller controls lifetime / may reuse structs)
    e->msg.ptr = nullptr;
    e->msg.len = 0;

    char tmp[128];
    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    e->msg = ark_str_from_cstr(s ? s : "unknown error");
}


// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
static const int64_t MAX_SAFE_TIMEOUT_MS = INT64_MAX / 1000000LL;


// -----------------------------------------------------------------------------
// Helpers: Strict System Call Wrappers
// -----------------------------------------------------------------------------
static int fcntl_get(int fd, int cmd) {
    int rc;
    do { rc = fcntl(fd, cmd); } while (rc == -1 && errno == EINTR);
    return rc;
}

static int fcntl_set(int fd, int cmd, int arg) {
    int rc;
    do { rc = fcntl(fd, cmd, arg); } while (rc == -1 && errno == EINTR);
    return rc;
}

static int ensure_cloexec(int fd) {
    if (fd < 0) return EBADF;
    int flags = fcntl_get(fd, F_GETFD);
    if (flags < 0) return errno;
    if (!(flags & FD_CLOEXEC)) {
        if (fcntl_set(fd, F_SETFD, flags | FD_CLOEXEC) < 0) return errno;
    }
    return 0;
}

static int set_nonblocking(int fd) {
    int flags = fcntl_get(fd, F_GETFL);
    if (flags == -1) return errno;
    if (fcntl_set(fd, F_SETFL, flags | O_NONBLOCK) == -1) return errno;
    return 0;
}

static int set_blocking(int fd) {
    int flags = fcntl_get(fd, F_GETFL);
    if (flags == -1) return errno;
    if (fcntl_set(fd, F_SETFL, flags & ~O_NONBLOCK) == -1) return errno;
    return 0;
}

static int apply_conn_defaults(int fd) {
    int opt = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) return errno;
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt)) < 0) return errno;
    #if defined(__APPLE__)
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &opt, sizeof(opt));
    #endif
    return 0;
}

// -----------------------------------------------------------------------------
// Helpers: Unified Nanosecond Timing
// -----------------------------------------------------------------------------
struct ArkDeadline {
    int64_t deadline_ns; // Absolute CLOCK_MONOTONIC value
    bool infinite;
};

// Robust Monotonic Clock Reader with Overflow Protection
static int now_mono_ns(int64_t* out_ns) {
    struct timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return errno;

    const int64_t sec = (int64_t)now.tv_sec;
    const int64_t nsec = (int64_t)now.tv_nsec;
    
    // Sanity check inputs
    if (sec < 0 || nsec < 0 || nsec >= 1000000000LL) return EIO;

    // Check for multiplication overflow
    if (sec > (INT64_MAX / 1000000000LL)) return EOVERFLOW;
    
    // Check for addition overflow
    int64_t total = sec * 1000000000LL;
    if (total > INT64_MAX - nsec) return EOVERFLOW;
    
    *out_ns = total + nsec;
    return 0;
}

static int calc_deadline(int64_t timeout_ms, ArkDeadline* out_dl) {
    if (timeout_ms <= 0) { 
        out_dl->infinite = true;
        return 0;
    }
    
    if (timeout_ms > MAX_SAFE_TIMEOUT_MS) return EINVAL;

    int64_t now_ns = 0;
    int err = now_mono_ns(&now_ns);
    if (err != 0) return err;

    const int64_t add_ns = timeout_ms * 1000000LL;
    
    // Check for Deadline Overflow
    if (now_ns > INT64_MAX - add_ns) return EOVERFLOW;

    out_dl->infinite = false;
    out_dl->deadline_ns = now_ns + add_ns;
    return 0;
}

static int get_remaining_ns(const ArkDeadline* dl, int64_t* out_ns) {
    if (dl->infinite) { *out_ns = -1; return 0; }

    int64_t now_ns = 0;
    int err = now_mono_ns(&now_ns);
    if (err != 0) return err;

    if (now_ns >= dl->deadline_ns) {
        *out_ns = 0;
    } else {
        *out_ns = dl->deadline_ns - now_ns;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Helpers: Singularity Poll
// -----------------------------------------------------------------------------

// Backoff Sleep charged against the deadline
static int backoff_sleep(int64_t* backoff_ns, const ArkDeadline* dl) {
    int64_t current_backoff = *backoff_ns;
    const int64_t MAX_BACKOFF = 1000000LL; // 1ms Cap

    // Clamp sleep to remaining time
    if (!dl->infinite) {
        int64_t rem_ns = 0;
        int err = get_remaining_ns(dl, &rem_ns);
        if (err != 0) return err;
        if (rem_ns == 0) return ETIMEDOUT;
        
        if (current_backoff > rem_ns) current_backoff = rem_ns;
    }

    struct timespec req, rem;
    req.tv_sec = current_backoff / 1000000000LL;
    req.tv_nsec = current_backoff % 1000000000LL;

    while (nanosleep(&req, &rem) == -1) {
        if (errno == EINTR) { req = rem; continue; }
        return errno;
    }

    *backoff_ns *= 2;
    if (*backoff_ns > MAX_BACKOFF) *backoff_ns = MAX_BACKOFF;
    return 0;
}

static int poll_connect_event(struct pollfd* pfd, const ArkDeadline* dl, int64_t* backoff_ns) {
    if (dl->infinite) {
        while (true) {
            pfd->revents = 0; 
            int rc = poll(pfd, 1, -1);
            if (rc == -1) { if (errno == EINTR) continue; return errno; }
            if (rc > 0) {
                if (pfd->revents == 0) {
                    int err = backoff_sleep(backoff_ns, dl);
                    if (err != 0) return err;
                    continue; 
                }
                if (pfd->revents & POLLNVAL) return EBADF;
                return 0; 
            }
            return EIO; 
        }
    }

    while (true) {
        int64_t rem_ns = 0;
        int time_err = get_remaining_ns(dl, &rem_ns);
        if (time_err != 0) return time_err;
        if (rem_ns == 0) return ETIMEDOUT;

        // Safe ms clamp for poll()
        int64_t wait_ms_long = rem_ns / 1000000LL;
        int wait_ms = (wait_ms_long > INT_MAX) ? INT_MAX : (int)wait_ms_long;

        pfd->revents = 0; 
        int rc = poll(pfd, 1, wait_ms);
        
        if (rc > 0) {
            if (pfd->revents == 0) {
                int err = backoff_sleep(backoff_ns, dl);
                if (err != 0) return err;
                continue;
            }
            if (pfd->revents & POLLNVAL) return EBADF;
            return 0;
        }
        
        if (rc == 0) {
            // If we clamped due to INT_MAX, loop again
            if (wait_ms == INT_MAX) continue;

            // Sub-ms tail check
            int64_t check_ns = 0;
            int chk_err = get_remaining_ns(dl, &check_ns);
            if (chk_err != 0) return chk_err;
            if (check_ns == 0) return ETIMEDOUT;

            // Non-spinning backoff for tail
            int bo_err = backoff_sleep(backoff_ns, dl);
            if (bo_err != 0) return bo_err;
            continue;
        }
        if (rc == -1 && errno != EINTR) return errno;
    }
}

// -----------------------------------------------------------------------------
// Networking Implementation
// -----------------------------------------------------------------------------

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
    e->code = code;
    e->_pad = 0;
    e->msg = ark_str_from_cstr(buf);
}


static inline void ark_err_clear(ArkIoError* e) {
    if (!e) return;
    e->code = 0;
    e->_pad = 0;
    e->msg.ptr = nullptr;
    e->msg.len = 0;
}


extern "C" {


ArkStatus __ark_net_connect(const char* host, int32_t port, int64_t timeout_ms, int32_t* out_sockfd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_sockfd) *out_sockfd = -1;

    if (!out_sockfd || !host || !*host) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (port <= 0 || port > 65535) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (timeout_ms < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (timeout_ms > MAX_SAFE_TIMEOUT_MS) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char port_str[16];
    (void)snprintf(port_str, sizeof(port_str), "%d", (int)port);

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_ADDRCONFIG | AI_NUMERICSERV;

    struct addrinfo* res = nullptr;
    int gai_rc = getaddrinfo(host, port_str, &hints, &res);
    if (gai_rc != 0) {
        if (gai_rc == EAI_SYSTEM) {
            int e = errno;
            ark_err_setf(out_err, (int32_t)e, "dns system error for %s:%d", host, (int)port);
            return (ArkStatus)e;
        }
        ark_err_set_code_msg(out_err, (int32_t)(-gai_rc), gai_strerror(gai_rc));
        return (ArkStatus)(-gai_rc);
    }

    ArkDeadline deadline{};
    int dl_err = calc_deadline(timeout_ms, &deadline);
    if (dl_err != 0) {
        freeaddrinfo(res);
        ark_err_set_errno(out_err, dl_err);
        return (ArkStatus)dl_err;
    }

    int sockfd = -1;
    int32_t last_code = (int32_t)ECONNREFUSED;

    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next) {
        if (!deadline.infinite) {
            int64_t rem_ns = 0;
            int tm_err = get_remaining_ns(&deadline, &rem_ns);
            if (tm_err != 0) { last_code = (int32_t)tm_err; break; }
            if (rem_ns == 0) { last_code = (int32_t)ETIMEDOUT; break; }
        }

        sockfd = socket(p->ai_family, p->ai_socktype | ARK_SOCK_CLOEXEC, p->ai_protocol);
        if (sockfd < 0) { last_code = (int32_t)errno; continue; }

        int clo_err = ensure_cloexec(sockfd);
        if (clo_err != 0) { last_code = (int32_t)clo_err; (void)close(sockfd); sockfd = -1; continue; }

        int opt_err = apply_conn_defaults(sockfd);
        if (opt_err != 0) { last_code = (int32_t)opt_err; (void)close(sockfd); sockfd = -1; continue; }

        int nb_err = set_nonblocking(sockfd);
        if (nb_err != 0) { last_code = (int32_t)nb_err; (void)close(sockfd); sockfd = -1; continue; }

        int c_rc = connect(sockfd, p->ai_addr, p->ai_addrlen);
        if (c_rc < 0) {
            if (errno != EINPROGRESS) {
                last_code = (int32_t)errno;
                (void)close(sockfd); sockfd = -1;
                continue;
            }

            struct pollfd pfd{};
            pfd.fd = sockfd;
            pfd.events = POLLOUT;

            int64_t backoff_ns = 1000;
            bool connected = false;

            for (;;) {
                int poll_err = poll_connect_event(&pfd, &deadline, &backoff_ns);
                if (poll_err != 0) { last_code = (int32_t)poll_err; break; }

                int so_error = 0;
                socklen_t slen = (socklen_t)sizeof(so_error);
                if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &so_error, &slen) < 0) {
                    last_code = (int32_t)errno;
                    break;
                }

                if (so_error == 0) { connected = true; break; }
                if (so_error == EINPROGRESS) continue;

                last_code = (int32_t)so_error;
                break;
            }

            if (!connected) {
                (void)close(sockfd); sockfd = -1;
                if (last_code == (int32_t)ETIMEDOUT) break;
                continue;
            }
        }

        int blk_err = set_blocking(sockfd);
        if (blk_err != 0) { last_code = (int32_t)blk_err; (void)close(sockfd); sockfd = -1; continue; }

        if (sockfd > INT32_MAX) {
            last_code = (int32_t)EOVERFLOW;
            (void)close(sockfd); sockfd = -1;
            break;
        }

        *out_sockfd = (int32_t)sockfd;
        freeaddrinfo(res);
        return 0;
    }

    freeaddrinfo(res);

    if (sockfd >= 0) (void)close(sockfd);
    ark_err_setf(out_err, last_code, "connect failed to %s:%d", host, (int)port);
    return (ArkStatus)last_code;
}



ArkStatus __ark_net_listen(const char* host, int32_t port, int32_t* out_sockfd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_sockfd) *out_sockfd = -1;

    if (!out_sockfd) {
        ark_err_set_code_msg(out_err, EINVAL, "out_sockfd is null");
        return (ArkStatus)EINVAL;
    }
    if (port <= 0 || port > 65535) {
        ark_err_setf(out_err, EINVAL, "invalid port: %d", (int)port);
        return (ArkStatus)EINVAL;
    }

    char pstr[16];
    (void)snprintf(pstr, sizeof(pstr), "%d", (int)port);

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_ADDRCONFIG | AI_NUMERICSERV;

    struct addrinfo* res = nullptr;
    int gai_rc = getaddrinfo(host, pstr, &hints, &res);
    if (gai_rc != 0) {
        if (gai_rc == EAI_SYSTEM) {
            int e = errno;
            ark_err_setf(out_err, (int32_t)e, "dns system error for listen %s:%d", host ? host : "*", (int)port);
            return (ArkStatus)e;
        }
        ark_err_set_code_msg(out_err, (int32_t)(-gai_rc), gai_strerror(gai_rc));
        return (ArkStatus)(-gai_rc);
    }

    int s = -1;
    int32_t last_code = (int32_t)EADDRINUSE;

    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next) {
        s = socket(p->ai_family, p->ai_socktype | ARK_SOCK_CLOEXEC, p->ai_protocol);
        if (s < 0) { last_code = (int32_t)errno; continue; }

        int ce = ensure_cloexec(s);
        if (ce != 0) { last_code = (int32_t)ce; (void)close(s); s = -1; continue; }

        int one = 1;
        (void)setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, (socklen_t)sizeof(one));

        if (bind(s, p->ai_addr, p->ai_addrlen) < 0) {
            last_code = (int32_t)errno;
            (void)close(s);
            s = -1;
            continue;
        }

        if (listen(s, 128) < 0) {
            last_code = (int32_t)errno;
            (void)close(s);
            s = -1;
            continue;
        }

        if (s > INT32_MAX) {
            last_code = (int32_t)EOVERFLOW;
            (void)close(s);
            s = -1;
            break;
        }

        *out_sockfd = (int32_t)s;
        freeaddrinfo(res);
        return 0;
    }

    freeaddrinfo(res);
    ark_err_setf(out_err, last_code, "listen bind failed for %s:%d", host ? host : "*", (int)port);
    return (ArkStatus)last_code;
}

ArkStatus __ark_net_accept(int32_t sockfd, int32_t* out_clientfd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_clientfd) *out_clientfd = -1;

    if (!out_clientfd) {
        ark_err_set_code_msg(out_err, EINVAL, "out_clientfd is null");
        return (ArkStatus)EINVAL;
    }

    struct sockaddr_storage addr{};
    socklen_t alen = (socklen_t)sizeof(addr);

    int n = -1;

#if defined(__linux__) && defined(SOCK_CLOEXEC)
    for (;;) {
        alen = (socklen_t)sizeof(addr);
        n = accept4(sockfd, (struct sockaddr*)&addr, &alen, SOCK_CLOEXEC);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == ENOSYS || errno == EINVAL)) break;
        if (n < 0) {
            int e = errno;
            ark_err_setf(out_err, (int32_t)e, "accept4 failed (fd=%d)", (int)sockfd);
            return (ArkStatus)e;
        }
        break;
    }
#endif

    if (n < 0) {
        for (;;) {
            alen = (socklen_t)sizeof(addr);
            n = accept(sockfd, (struct sockaddr*)&addr, &alen);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) {
                int e = errno;
                ark_err_setf(out_err, (int32_t)e, "accept failed (fd=%d)", (int)sockfd);
                return (ArkStatus)e;
            }
            break;
        }
    }

    int ce = ensure_cloexec(n);
    if (ce != 0) {
        (void)close(n);
        ark_err_setf(out_err, (int32_t)ce, "accept cloexec failed (fd=%d)", n);
        return (ArkStatus)ce;
    }

    int opt_err = apply_conn_defaults(n);
    if (opt_err != 0) {
        (void)close(n);
        ark_err_setf(out_err, (int32_t)opt_err, "accept socket opts failed (fd=%d)", n);
        return (ArkStatus)opt_err;
    }

    if (n > INT32_MAX) {
        (void)close(n);
        ark_err_set_code_msg(out_err, EOVERFLOW, "accepted fd overflow");
        return (ArkStatus)EOVERFLOW;
    }

    *out_clientfd = (int32_t)n;
    return 0;
}


ArkStatus __ark_net_send(int32_t sockfd, const void* data, int64_t len, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (len < 0 || (len > 0 && !data)) { ark_err_set_errno(out_err, EINVAL); return (ArkStatus)EINVAL; }
    if (len > (int64_t)SSIZE_MAX) { ark_err_set_errno(out_err, EFBIG); return (ArkStatus)EFBIG; }

    const uint8_t* p = (const uint8_t*)data;
    int64_t rem = len;

    while (rem > 0) {
        ssize_t w = send(sockfd, p, (size_t)rem, ARK_MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            int e = errno;
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

extern "C" ArkStatus __ark_net_recv(int32_t sockfd, int64_t max_len, ArkBytes* out, ArkIoError* out_err) {
    // 1. Reset State
    ark_err_clear(out_err);
    if (out) { out->ptr = nullptr; out->len = 0; out->cap = 0; }

    // 2. Validate Output Pointer
    if (!out) { 
        ark_err_set_errno(out_err, EINVAL); 
        return (ArkStatus)EINVAL; 
    }

    // 3. Validate & Clamp Size
    if (max_len <= 0) max_len = 4096;
    if (max_len > ARK_NET_MAX_PACKET) { 
        ark_err_set_errno(out_err, EFBIG); 
        return (ArkStatus)EFBIG; 
    }
    if (max_len > (int64_t)SSIZE_MAX) max_len = (int64_t)SSIZE_MAX;

    // 4. [FIX] Allocation with Alignment
    // Signature: void* __ark_alloc(uint64_t bytes, uint64_t align)
    // We use 16-byte alignment for optimal network/SIMD performance.
    uint8_t* b = (uint8_t*)__ark_alloc((uint64_t)max_len, 16);
    
    if (!b) { 
        ark_err_set_errno(out_err, ENOMEM); 
        return (ArkStatus)ENOMEM; 
    }

    // 5. Receive Loop (Retry on EINTR)
    ssize_t r;
    for (;;) {
        r = ::recv(sockfd, b, (size_t)max_len, 0);
        if (r < 0 && errno == EINTR) continue;
        break;
    }

    // 6. Handle Error
    if (r < 0) {
        int e = errno;
        __ark_free(b); // Free the unused buffer
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    // 7. Handle EOF (Connection Closed)
    if (r == 0) {
        __ark_free(b); // Don't return an allocated but empty buffer
        // Note: Returning 0 (OK) with out->ptr=NULL implies EOF in many APIs.
        return 0;
    }

    // 8. Success
    out->ptr = b;
    out->len = (int64_t)r;
    out->cap = (int64_t)max_len;
    return 0;
}

ArkStatus __ark_net_close(int32_t sockfd, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (sockfd < 0) return 0;

    int r;
    do { r = close(sockfd); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}


} // extern "C"