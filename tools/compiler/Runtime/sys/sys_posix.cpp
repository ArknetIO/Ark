/* Arklang POSIX System Kernel (SYS capability)
 * Capability-backed runtime namespace for argv/env/process/cwd/system info.
 *
 * ABI RULES:
 * - ark_protocol.h is the ONLY source of truth for public types (ArkStr, ArkIoError, ArkStatus, etc.)
 * - SYS entrypoints return ArkStatus and write results via out-params.
 * - Borrowed strings MUST point to process-lifetime memory (argv/env/libc stable pointers).
 * - Owned strings MUST be allocated via __ark_alloc and are freed via __ark_free by Ark runtime/users.
 */

#include <ark_protocol.h>

#if defined(__cplusplus)
  #include <cstdint>
  #include <cstddef>
  #include <cstdbool>
  #include <cerrno>
  #include <cstdlib>
  #include <cstring>
  #include <unistd.h>
  #include <limits.h>
  #include <time.h>
  #include <sys/types.h>
  #include <sys/utsname.h>
  #include <sys/time.h>
  #include <pwd.h>
  #include <fcntl.h>
  #include <signal.h>
  #include <sys/resource.h>
  #include <sys/stat.h>
#else
  #include <stdint.h>
  #include <stddef.h>
  #include <stdbool.h>
  #include <errno.h>
  #include <stdlib.h>
  #include <string.h>
  #include <unistd.h>
  #include <limits.h>
  #include <time.h>
  #include <sys/types.h>
  #include <sys/utsname.h>
  #include <sys/time.h>
  #include <pwd.h>
  #include <fcntl.h>
  #include <signal.h>
  #include <sys/resource.h>
  #include <sys/stat.h>
#endif

#if defined(__APPLE__)
  #include <crt_externs.h>
#endif

#if !defined(HOST_NAME_MAX)
  #define HOST_NAME_MAX 255
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// =============================================================================
// Error Helpers
// =============================================================================

static inline void ark_err_clear(ArkIoError* e) {
    if (!e) return;
    e->code = 0;
    e->_pad = 0;
    e->msg.ptr = NULL;
    e->msg.len = 0;
}

static inline const char* ark_strerror_cstr(int code, char* buf, size_t buflen) {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
    return strerror_r(code, buf, buflen);
#else
    if (strerror_r(code, buf, buflen) == 0) return buf;
    return "unknown error";
#endif
}

static inline void ark_err_set_errno(ArkIoError* e, int code) {
    if (!e) return;

    e->code = (int32_t)code;
    e->_pad = 0;
    e->msg.ptr = NULL;
    e->msg.len = 0;

    char tmp[128];
    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    if (!s) s = "unknown error";

    size_t n = strlen(s);
    char* owned = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    if (!owned) return;

    memcpy(owned, s, n);
    owned[n] = '\0';
    e->msg.ptr = owned;
    e->msg.len = (int64_t)n;
}

static inline ArkStatus ark_out_borrow_cstr(ArkStr* out, const char* s, ArkIoError* out_err) {
    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (!s) {
        out->ptr = NULL;
        out->len = 0;
        return 0;
    }
    out->ptr = (char*)s;
    out->len = (int64_t)strlen(s);
    return 0;
}

static inline ArkStatus ark_out_copy_bytes(ArkStr* out, const void* data, size_t n, ArkIoError* out_err) {
    if (!out || (!data && n)) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    char* p = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    if (!p) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }
    if (n) memcpy(p, data, n);
    p[n] = '\0';
    out->ptr = p;
    out->len = (int64_t)n;
    return 0;
}

static inline ArkStatus ark_out_copy_cstr(ArkStr* out, const char* s, ArkIoError* out_err) {
    if (!out || !s) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    return ark_out_copy_bytes(out, s, strlen(s), out_err);
}

// =============================================================================
// Process bootstrap state (argv/envp)
// =============================================================================

static int g_ark_argc = 0;
static const char* const* g_ark_argv = NULL;
static const char* const* g_ark_envp = NULL;

#if defined(__APPLE__)
static inline char** ark_environ_ptr(void) { return *_NSGetEnviron(); }
#else
extern char** environ;
static inline char** ark_environ_ptr(void) { return environ; }
#endif

static inline const char* const* ark_envp_view(void) {
    if (g_ark_envp) return g_ark_envp;
    return (const char* const*)ark_environ_ptr();
}

void __ark_sys_process_init(int argc, const char* const* argv, const char* const* envp) {
    g_ark_argc = (argc < 0) ? 0 : argc;
    g_ark_argv = argv;
    g_ark_envp = envp;
}

// =============================================================================
// SYS.args / SYS.env.get / SYS.cwd  (Out-param ABI)
// =============================================================================

/* ABI: ArkStatus __ark_sys_args(ArkStr** out_ptr, int64_t* out_len, ArkIoError* out_err)
 * Returns a BORROWED view into argv (process lifetime).
 * The returned ArkStr array is allocated once using __ark_alloc and contains borrowed ptrs.
 */
ArkStatus __ark_sys_args(ArkStr** out_ptr, int64_t* out_len, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_ptr) *out_ptr = NULL;
    if (out_len) *out_len = 0;

    if (!out_ptr || !out_len) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    static ArkStr* args_arr = NULL;
    static int64_t args_len = 0;

    if (!args_arr && g_ark_argc > 0) {
        args_len = (int64_t)g_ark_argc;
        args_arr = (ArkStr*)__ark_alloc((uint64_t)(sizeof(ArkStr) * (size_t)args_len), 8);
        if (!args_arr) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        for (int64_t i = 0; i < args_len; ++i) {
            const char* s = (g_ark_argv && g_ark_argv[i]) ? g_ark_argv[i] : "";
            args_arr[i].ptr = (char*)s;
            args_arr[i].len = (int64_t)strlen(s);
        }
    }

    *out_ptr = args_arr;
    *out_len = args_len;
    return 0;
}

/* ABI: ArkStatus __ark_sys_env_get(const char* key, ArkStr* out_val, uint8_t* out_ok, ArkIoError* out_err)
 * Returns BORROWED value pointer from libc getenv.
 */
ArkStatus __ark_sys_env_get(const char* key, ArkStr* out_val, uint8_t* out_ok, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_val) { out_val->ptr = NULL; out_val->len = 0; }
    if (out_ok)  { *out_ok = 0; }

    if (!key || !out_val || !out_ok) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    const char* v = getenv(key);
    if (!v) return 0;

    out_val->ptr = (char*)v;
    out_val->len = (int64_t)strlen(v);
    *out_ok = 1;
    return 0;
}

/* ABI: ArkStatus __ark_sys_cwd(ArkStr* out, ArkIoError* out_err)
 * Returns OWNED string allocated with __ark_alloc.
 */
ArkStatus __ark_sys_cwd(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    size_t cap = 256;
    for (;;) {
        char* p = (char*)__ark_alloc((uint64_t)cap, 1);
        if (!p) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        if (getcwd(p, cap)) {
            out->ptr = p;
            out->len = (int64_t)strlen(p);
            return 0;
        }

        int e = errno;
        __ark_free(p);

        if (e != ERANGE) {
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        cap *= 2;
    }
}

// =============================================================================
// env (Mutations)
// =============================================================================

ArkStatus __ark_sys_setenv(const char* key, const char* val, uint8_t overwrite, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!key || !*key || !val) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r;
    do { r = setenv(key, val, overwrite ? 1 : 0); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return 0;
}

ArkStatus __ark_sys_unsetenv(const char* key, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!key || !*key) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r;
    do { r = unsetenv(key); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return 0;
}

// =============================================================================
// process / stdio / identity
// =============================================================================

ArkStatus __ark_sys_chdir(const char* path, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r;
    do { r = chdir(path); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return 0;
}

ArkStatus __ark_sys_pid(int64_t* out_pid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_pid) *out_pid = 0;

    if (!out_pid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_pid = (int64_t)getpid();
    return 0;
}

ArkStatus __ark_sys_ppid(int64_t* out_ppid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_ppid) *out_ppid = 0;

    if (!out_ppid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_ppid = (int64_t)getppid();
    return 0;
}

ArkStatus __ark_sys_uid(int64_t* out_uid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_uid) *out_uid = 0;

    if (!out_uid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_uid = (int64_t)getuid();
    return 0;
}

ArkStatus __ark_sys_euid(int64_t* out_uid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_uid) *out_uid = 0;

    if (!out_uid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_uid = (int64_t)geteuid();
    return 0;
}

ArkStatus __ark_sys_gid(int64_t* out_gid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_gid) *out_gid = 0;

    if (!out_gid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_gid = (int64_t)getgid();
    return 0;
}

ArkStatus __ark_sys_egid(int64_t* out_gid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_gid) *out_gid = 0;

    if (!out_gid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_gid = (int64_t)getegid();
    return 0;
}

ArkStatus __ark_sys_stdin_fd(int64_t* out_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_fd) *out_fd = -1;

    if (!out_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_fd = 0;
    return 0;
}

ArkStatus __ark_sys_stdout_fd(int64_t* out_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_fd) *out_fd = -1;

    if (!out_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_fd = 1;
    return 0;
}

ArkStatus __ark_sys_stderr_fd(int64_t* out_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_fd) *out_fd = -1;

    if (!out_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_fd = 2;
    return 0;
}

// =============================================================================
// time / sleep
// =============================================================================

ArkStatus __ark_sys_sleep_ms(int64_t ms, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (ms < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct timespec req;
    req.tv_sec = (time_t)(ms / 1000);
    req.tv_nsec = (long)((ms % 1000) * 1000000LL);

    for (;;) {
        struct timespec rem;
        if (nanosleep(&req, &rem) == 0) return 0;
        if (errno != EINTR) {
            int e = errno;
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        req = rem;
    }
}

ArkStatus __ark_sys_time_ms(int64_t* out_ms, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_ms) *out_ms = 0;

    if (!out_ms) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

#if defined(CLOCK_REALTIME)
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        *out_ms = (int64_t)ts.tv_sec * 1000 + (int64_t)(ts.tv_nsec / 1000000);
        return 0;
    }
#endif

    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        *out_ms = (int64_t)tv.tv_sec * 1000 + (int64_t)(tv.tv_usec / 1000);
        return 0;
    }

    int e = errno;
    ark_err_set_errno(out_err, e);
    return (ArkStatus)e;
}

// =============================================================================
// system info (hostname/platform/arch/uname)
// =============================================================================

ArkStatus __ark_sys_hostname(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char buf[HOST_NAME_MAX + 1];
    memset(buf, 0, sizeof(buf));

    if (gethostname(buf, HOST_NAME_MAX) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    buf[HOST_NAME_MAX] = '\0';
    return ark_out_copy_cstr(out, buf, out_err);
}

ArkStatus __ark_sys_platform(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

#if defined(__linux__)
    return ark_out_borrow_cstr(out, "linux", out_err);
#elif defined(__APPLE__)
    return ark_out_borrow_cstr(out, "macos", out_err);
#elif defined(__FreeBSD__)
    return ark_out_borrow_cstr(out, "freebsd", out_err);
#else
    return ark_out_borrow_cstr(out, "posix", out_err);
#endif
}

ArkStatus __ark_sys_arch(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

#if defined(__x86_64__) || defined(_M_X64)
    return ark_out_borrow_cstr(out, "x86_64", out_err);
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ark_out_borrow_cstr(out, "aarch64", out_err);
#elif defined(__arm__) || defined(_M_ARM)
    return ark_out_borrow_cstr(out, "arm", out_err);
#elif defined(__i386__) || defined(_M_IX86)
    return ark_out_borrow_cstr(out, "x86", out_err);
#elif defined(__riscv) && (__riscv_xlen == 64)
    return ark_out_borrow_cstr(out, "riscv64", out_err);
#else
    struct utsname u;
    if (uname(&u) == 0) return ark_out_copy_cstr(out, u.machine, out_err);
    return ark_out_borrow_cstr(out, "unknown", out_err);
#endif
}

ArkStatus __ark_sys_uname_sysname(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct utsname u;
    if (uname(&u) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return ark_out_copy_cstr(out, u.sysname, out_err);
}

ArkStatus __ark_sys_uname_release(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct utsname u;
    if (uname(&u) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return ark_out_copy_cstr(out, u.release, out_err);
}

ArkStatus __ark_sys_uname_version(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct utsname u;
    if (uname(&u) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return ark_out_copy_cstr(out, u.version, out_err);
}

ArkStatus __ark_sys_uname_machine(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct utsname u;
    if (uname(&u) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return ark_out_copy_cstr(out, u.machine, out_err);
}

// =============================================================================
// executable / user home
// =============================================================================

ArkStatus __ark_sys_exe(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

#if defined(__linux__)
    char buf[PATH_MAX + 1];
    ssize_t n;
    for (;;) {
        n = readlink("/proc/self/exe", buf, PATH_MAX);
        if (n < 0 && errno == EINTR) continue;
        break;
    }
    if (n < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    if (n > PATH_MAX) n = PATH_MAX;
    buf[n] = '\0';
    return ark_out_copy_bytes(out, buf, (size_t)n, out_err);
#elif defined(__APPLE__)
    ark_err_set_errno(out_err, ENOSYS);
    return (ArkStatus)ENOSYS;
#else
    ark_err_set_errno(out_err, ENOSYS);
    return (ArkStatus)ENOSYS;
#endif
}

ArkStatus __ark_sys_home(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    const char* h = getenv("HOME");
    if (h && *h) return ark_out_copy_cstr(out, h, out_err);

    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir) return ark_out_copy_cstr(out, pw->pw_dir, out_err);

    ark_err_set_errno(out_err, ENOENT);
    return (ArkStatus)ENOENT;
}

// =============================================================================
// signals / priority
// =============================================================================

ArkStatus __ark_sys_kill(int64_t pid, int32_t sig, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (pid <= 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r;
    do { r = kill((pid_t)pid, sig); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return 0;
}

ArkStatus __ark_sys_nice(int32_t inc, int32_t* out_new_prio, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_new_prio) *out_new_prio = 0;

    errno = 0;
    int r = nice(inc);
    if (r == -1 && errno != 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (out_new_prio) *out_new_prio = r;
    return 0;
}

// =============================================================================
// non-returning control
// =============================================================================

void __ark_sys_exit(int32_t code) {
    _exit((int)code);
}

void __ark_sys_abort(void) {
    abort();
}

#if defined(__cplusplus)
} // extern "C"
#endif