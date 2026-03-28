/* Arklang Windows System Kernel (SYS capability)
 * Capability-backed runtime namespace for argv/env/process/cwd/system info.
 *
 * ABI RULES:
 * - ark_protocol.h is the ONLY source of truth for public types (ArkStr, ArkIoError, ArkStatus, etc.)
 * - SYS entrypoints return ArkStatus and write results via out-params.
 * - Borrowed strings MUST point to process-lifetime memory (argv/env/CRT stable pointers).
 * - Owned strings MUST be allocated via __ark_alloc and are freed via __ark_free by Ark runtime/users.
 *
 * Windows Equivalence Notes:
 * - argv/env/getcwd/chdir/std-fd semantics mirror the POSIX contract directly.
 * - uid/euid/gid/egid are mapped to access-token RID values.
 * - uname fields are synthesized from Win32 version/system APIs.
 * - kill(pid, 0) checks process existence; nonzero signals terminate the target process.
 * - nice() is approximated with process priority classes.
 */

#include <ark_protocol.h>

#if defined(__cplusplus)
  #include <cstdint>
  #include <cstddef>
  #include <cstdbool>
  #include <cerrno>
  #include <cstdlib>
  #include <cstring>
  #include <climits>
  #include <ctime>
  #include <cstdio>
#else
  #include <stdint.h>
  #include <stddef.h>
  #include <stdbool.h>
  #include <errno.h>
  #include <stdlib.h>
  #include <string.h>
  #include <limits.h>
  #include <time.h>
  #include <stdio.h>
#endif

#include <windows.h>
#include <tlhelp32.h>
#include <io.h>
#include <direct.h>
#include <process.h>
#include <signal.h>

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

static inline size_t ark_trim_message_len(const char* s, size_t n) {
    while (n != 0) {
        const char ch = s[n - 1];
        if (ch != '\r' && ch != '\n' && ch != ' ' && ch != '\t') break;
        --n;
    }
    return n;
}

static inline void ark_err_set_message_n(ArkIoError* e, const char* s, size_t n) {
    if (!e || !s) return;

    char* owned = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    if (!owned) return;

    if (n) memcpy(owned, s, n);
    owned[n] = '\0';
    e->msg.ptr = owned;
    e->msg.len = (int64_t)n;
}

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
        case ERROR_ARENA_TRASHED:
        case ERROR_INVALID_BLOCK:
            return ENOMEM;

        case ERROR_WRITE_PROTECT:
            return EROFS;

        case ERROR_NOT_SAME_DEVICE:
            return EXDEV;

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

        case ERROR_SEEK_ON_DEVICE:
            return ESPIPE;

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
        case ERROR_NEGATIVE_SEEK:
            return EINVAL;

        case ERROR_CRC:
        case ERROR_BAD_LENGTH:
        case ERROR_SECTOR_NOT_FOUND:
        case ERROR_SEEK:
        case ERROR_NOT_DOS_DISK:
        case ERROR_READ_FAULT:
        case ERROR_WRITE_FAULT:
        case ERROR_GEN_FAILURE:
            return EIO;

        case ERROR_RETRY:
            return EAGAIN;

        case ERROR_CALL_NOT_IMPLEMENTED:
            return ENOSYS;

        case ERROR_NOT_SUPPORTED:
            return ENOTSUP;

        default:
            return EIO;
    }
}

static inline const char* ark_strerror_cstr(int code, char* buf, size_t buflen) {
#if defined(_MSC_VER)
    if (!buf || buflen == 0) return "unknown error";
    strerror_s(buf, buflen, code);
    buf[buflen - 1] = '\0';
    return buf;
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

    ark_err_set_message_n(e, s, strlen(s));
}

static inline void ark_err_set_win32(ArkIoError* e, DWORD winerr) {
    if (!e) return;

    const int code = ark_winerr_to_errno(winerr);

    e->code = (int32_t)code;
    e->_pad = 0;
    e->msg.ptr = NULL;
    e->msg.len = 0;

    char msg[512];
    DWORD got = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        winerr,
        0,
        msg,
        (DWORD)sizeof(msg),
        NULL
    );

    if (got != 0) {
        const size_t n = ark_trim_message_len(msg, (size_t)got);
        if (n != 0) {
            ark_err_set_message_n(e, msg, n);
            if (e->msg.ptr) return;
        }
    }

    char tmp[128];
    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    if (!s) s = "unknown error";
    ark_err_set_message_n(e, s, strlen(s));
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

static inline char** ark_environ_ptr(void) {
#if defined(_MSC_VER)
    char*** penv = __p__environ();
    return penv ? *penv : NULL;
#else
    extern char** environ;
    return environ;
#endif
}

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
// Internal Helpers: Windows String / Env / Time / Identity
// =============================================================================

static inline ArkStatus ark_copy_env_var(ArkStr* out, const char* key, ArkIoError* out_err) {
    if (!out || !key || !*key) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    const char* crt = getenv(key);
    if (crt && *crt) {
        return ark_out_copy_cstr(out, crt, out_err);
    }

    DWORD need = GetEnvironmentVariableA(key, NULL, 0);
    if (need == 0) {
        DWORD we = GetLastError();
        if (we == ERROR_ENVVAR_NOT_FOUND) {
            ark_err_set_errno(out_err, ENOENT);
            return (ArkStatus)ENOENT;
        }
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    char* buf = (char*)__ark_alloc((uint64_t)need, 1);
    if (!buf) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    DWORD got = GetEnvironmentVariableA(key, buf, need);
    if (got == 0 || got >= need) {
        DWORD we = GetLastError();
        __ark_free(buf);
        ark_err_set_win32(out_err, we ? we : ERROR_INSUFFICIENT_BUFFER);
        return (ArkStatus)out_err->code;
    }

    out->ptr = buf;
    out->len = (int64_t)got;
    return 0;
}

static inline const char* ark_arch_name_from_system_info(void) {
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);

    switch (si.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: return "x86_64";
        case PROCESSOR_ARCHITECTURE_ARM64: return "aarch64";
        case PROCESSOR_ARCHITECTURE_ARM:   return "arm";
        case PROCESSOR_ARCHITECTURE_INTEL: return "x86";
#if defined(PROCESSOR_ARCHITECTURE_RISCV64)
        case PROCESSOR_ARCHITECTURE_RISCV64: return "riscv64";
#endif
        default: return "unknown";
    }
}

static inline ArkStatus ark_copy_hostname(ArkStr* out, ArkIoError* out_err) {
    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    DWORD need = 0;
    GetComputerNameExA(ComputerNameDnsHostname, NULL, &need);

    if (need == 0) {
        char buf[MAX_COMPUTERNAME_LENGTH + 1];
        DWORD n = MAX_COMPUTERNAME_LENGTH + 1;
        if (!GetComputerNameA(buf, &n)) {
            ark_err_set_win32(out_err, GetLastError());
            return (ArkStatus)out_err->code;
        }
        return ark_out_copy_bytes(out, buf, (size_t)n, out_err);
    }

    char* buf = (char*)__ark_alloc((uint64_t)(need + 1), 1);
    if (!buf) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    DWORD got = need;
    if (!GetComputerNameExA(ComputerNameDnsHostname, buf, &got)) {
        DWORD we = GetLastError();
        __ark_free(buf);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    buf[got] = '\0';
    out->ptr = buf;
    out->len = (int64_t)got;
    return 0;
}

static inline ArkStatus ark_query_os_version(OSVERSIONINFOEXW* out, ArkIoError* out_err) {
    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    ZeroMemory(out, sizeof(*out));
    out->dwOSVersionInfoSize = sizeof(*out);

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll) {
        typedef LONG (WINAPI *RtlGetVersionFn)(OSVERSIONINFOW*);
        RtlGetVersionFn rtlGetVersion = (RtlGetVersionFn)GetProcAddress(ntdll, "RtlGetVersion");
        if (rtlGetVersion) {
            if (rtlGetVersion((OSVERSIONINFOW*)out) == 0) {
                return 0;
            }
        }
    }

#pragma warning(push)
#pragma warning(disable:4996)
    if (GetVersionExW((OSVERSIONINFOW*)out)) {
        return 0;
    }
#pragma warning(pop)

    ark_err_set_win32(out_err, GetLastError());
    return (ArkStatus)out_err->code;
}

static inline ArkStatus ark_copy_os_release(ArkStr* out, ArkIoError* out_err) {
    OSVERSIONINFOEXW osv;
    ArkStatus st = ark_query_os_version(&osv, out_err);
    if (st != 0) return st;

    char buf[64];
    int n = snprintf(
        buf,
        sizeof(buf),
        "%lu.%lu",
        (unsigned long)osv.dwMajorVersion,
        (unsigned long)osv.dwMinorVersion
    );

    if (n < 0 || (size_t)n >= sizeof(buf)) {
        ark_err_set_errno(out_err, EIO);
        return (ArkStatus)EIO;
    }

    return ark_out_copy_cstr(out, buf, out_err);
}

static inline ArkStatus ark_copy_os_version_string(ArkStr* out, ArkIoError* out_err) {
    OSVERSIONINFOEXW osv;
    ArkStatus st = ark_query_os_version(&osv, out_err);
    if (st != 0) return st;

    char buf[96];
    int n = snprintf(
        buf,
        sizeof(buf),
        "Build %lu",
        (unsigned long)osv.dwBuildNumber
    );

    if (n < 0 || (size_t)n >= sizeof(buf)) {
        ark_err_set_errno(out_err, EIO);
        return (ArkStatus)EIO;
    }

    return ark_out_copy_cstr(out, buf, out_err);
}

static inline ArkStatus ark_current_token_rid(TOKEN_INFORMATION_CLASS tic,
                                              int64_t* out_rid,
                                              ArkIoError* out_err) {
    if (!out_rid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_rid = 0;

    HANDLE token = NULL;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    DWORD need = 0;
    GetTokenInformation(token, tic, NULL, 0, &need);
    if (need == 0) {
        DWORD we = GetLastError();
        CloseHandle(token);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    void* buf = malloc((size_t)need);
    if (!buf) {
        CloseHandle(token);
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    if (!GetTokenInformation(token, tic, buf, need, &need)) {
        DWORD we = GetLastError();
        free(buf);
        CloseHandle(token);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    PSID sid = NULL;
    if (tic == TokenUser) {
        sid = ((TOKEN_USER*)buf)->User.Sid;
    } else if (tic == TokenPrimaryGroup) {
        sid = ((TOKEN_PRIMARY_GROUP*)buf)->PrimaryGroup;
    }

    if (!sid || !IsValidSid(sid)) {
        free(buf);
        CloseHandle(token);
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    UCHAR count = *GetSidSubAuthorityCount(sid);
    if (count == 0) {
        free(buf);
        CloseHandle(token);
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    DWORD* rid = GetSidSubAuthority(sid, count - 1);
    if (!rid) {
        free(buf);
        CloseHandle(token);
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_rid = (int64_t)(*rid);

    free(buf);
    CloseHandle(token);
    return 0;
}

static inline ArkStatus ark_current_parent_pid(int64_t* out_ppid, ArkIoError* out_err) {
    if (!out_ppid) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_ppid = 0;

    DWORD self = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    PROCESSENTRY32 pe;
    ZeroMemory(&pe, sizeof(pe));
    pe.dwSize = sizeof(pe);

    if (!Process32First(snap, &pe)) {
        DWORD we = GetLastError();
        CloseHandle(snap);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    do {
        if (pe.th32ProcessID == self) {
            *out_ppid = (int64_t)pe.th32ParentProcessID;
            CloseHandle(snap);
            return 0;
        }
    } while (Process32Next(snap, &pe));

    CloseHandle(snap);
    ark_err_set_errno(out_err, ESRCH);
    return (ArkStatus)ESRCH;
}

static inline int32_t ark_prio_class_to_nice(DWORD cls) {
    switch (cls) {
        case REALTIME_PRIORITY_CLASS:     return -20;
        case HIGH_PRIORITY_CLASS:         return -15;
        case ABOVE_NORMAL_PRIORITY_CLASS: return -5;
        case NORMAL_PRIORITY_CLASS:       return 0;
        case BELOW_NORMAL_PRIORITY_CLASS: return 5;
        case IDLE_PRIORITY_CLASS:         return 19;
        default:                          return 0;
    }
}

static inline DWORD ark_nice_to_prio_class(int32_t nice) {
    if (nice <= -16) return REALTIME_PRIORITY_CLASS;
    if (nice <= -10) return HIGH_PRIORITY_CLASS;
    if (nice < 0)    return ABOVE_NORMAL_PRIORITY_CLASS;
    if (nice < 10)   return NORMAL_PRIORITY_CLASS;
    if (nice < 19)   return BELOW_NORMAL_PRIORITY_CLASS;
    return IDLE_PRIORITY_CLASS;
}

static inline int64_t ark_filetime_to_unix_ms(const FILETIME* ft) {
    ULARGE_INTEGER u;
    u.LowPart = ft->dwLowDateTime;
    u.HighPart = ft->dwHighDateTime;

    const uint64_t windows_ticks = u.QuadPart;
    const uint64_t epoch_delta_100ns = 116444736000000000ULL;

    if (windows_ticks < epoch_delta_100ns) {
        return 0;
    }

    const uint64_t unix_100ns = windows_ticks - epoch_delta_100ns;
    return (int64_t)(unix_100ns / 10000ULL);
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
 * Returns BORROWED value pointer from CRT getenv.
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

        if (_getcwd(p, (int)cap)) {
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

    if (!overwrite) {
        const char* cur = getenv(key);
        if (cur) return 0;
    }

    if (_putenv_s(key, val) != 0) {
        int e = errno ? errno : EINVAL;
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

    if (_putenv_s(key, "") != 0) {
        int e = errno ? errno : EINVAL;
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

    if (_chdir(path) != 0) {
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

    *out_pid = (int64_t)GetCurrentProcessId();
    return 0;
}

ArkStatus __ark_sys_ppid(int64_t* out_ppid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_ppid) *out_ppid = 0;

    return ark_current_parent_pid(out_ppid, out_err);
}

ArkStatus __ark_sys_uid(int64_t* out_uid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_uid) *out_uid = 0;

    return ark_current_token_rid(TokenUser, out_uid, out_err);
}

ArkStatus __ark_sys_euid(int64_t* out_uid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_uid) *out_uid = 0;

    return ark_current_token_rid(TokenUser, out_uid, out_err);
}

ArkStatus __ark_sys_gid(int64_t* out_gid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_gid) *out_gid = 0;

    return ark_current_token_rid(TokenPrimaryGroup, out_gid, out_err);
}

ArkStatus __ark_sys_egid(int64_t* out_gid, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_gid) *out_gid = 0;

    return ark_current_token_rid(TokenPrimaryGroup, out_gid, out_err);
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

    uint64_t remain = (uint64_t)ms;
    while (remain > 0) {
        DWORD chunk = (remain > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : (DWORD)remain;
        Sleep(chunk);
        remain -= (uint64_t)chunk;
    }

    return 0;
}

ArkStatus __ark_sys_time_ms(int64_t* out_ms, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_ms) *out_ms = 0;

    if (!out_ms) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    FILETIME ft;
    HMODULE k32 = GetModuleHandleA("kernel32.dll");

    if (k32) {
        typedef VOID (WINAPI *GetSystemTimePreciseAsFileTimeFn)(LPFILETIME);
        GetSystemTimePreciseAsFileTimeFn precise =
            (GetSystemTimePreciseAsFileTimeFn)GetProcAddress(k32, "GetSystemTimePreciseAsFileTime");
        if (precise) {
            precise(&ft);
            *out_ms = ark_filetime_to_unix_ms(&ft);
            return 0;
        }
    }

    GetSystemTimeAsFileTime(&ft);
    *out_ms = ark_filetime_to_unix_ms(&ft);
    return 0;
}

// =============================================================================
// system info (hostname/platform/arch/uname)
// =============================================================================

ArkStatus __ark_sys_hostname(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    return ark_copy_hostname(out, out_err);
}

ArkStatus __ark_sys_platform(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    return ark_out_borrow_cstr(out, "windows", out_err);
}

ArkStatus __ark_sys_arch(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    return ark_out_borrow_cstr(out, ark_arch_name_from_system_info(), out_err);
}

ArkStatus __ark_sys_uname_sysname(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    return ark_out_copy_cstr(out, "Windows_NT", out_err);
}

ArkStatus __ark_sys_uname_release(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    return ark_copy_os_release(out, out_err);
}

ArkStatus __ark_sys_uname_version(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    return ark_copy_os_version_string(out, out_err);
}

ArkStatus __ark_sys_uname_machine(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    return ark_out_copy_cstr(out, ark_arch_name_from_system_info(), out_err);
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

    DWORD cap = 260;
    for (;;) {
        char* buf = (char*)__ark_alloc((uint64_t)(cap + 1), 1);
        if (!buf) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        SetLastError(ERROR_SUCCESS);
        DWORD n = GetModuleFileNameA(NULL, buf, cap);
        DWORD we = GetLastError();

        if (n == 0) {
            __ark_free(buf);
            ark_err_set_win32(out_err, we ? we : GetLastError());
            return (ArkStatus)out_err->code;
        }

        if (n < cap && we != ERROR_INSUFFICIENT_BUFFER) {
            buf[n] = '\0';
            out->ptr = buf;
            out->len = (int64_t)n;
            return 0;
        }

        __ark_free(buf);

        if (cap > 32768) {
            ark_err_set_errno(out_err, ENAMETOOLONG);
            return (ArkStatus)ENAMETOOLONG;
        }

        cap *= 2;
    }
}

ArkStatus __ark_sys_home(ArkStr* out, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out) { out->ptr = NULL; out->len = 0; }

    if (!out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    ArkStatus st = ark_copy_env_var(out, "USERPROFILE", out_err);
    if (st == 0) return 0;

    ark_err_clear(out_err);

    const char* drive = getenv("HOMEDRIVE");
    const char* path  = getenv("HOMEPATH");

    if (drive && *drive && path && *path) {
        size_t a = strlen(drive);
        size_t b = strlen(path);

        char* buf = (char*)__ark_alloc((uint64_t)(a + b + 1), 1);
        if (!buf) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        memcpy(buf, drive, a);
        memcpy(buf + a, path, b);
        buf[a + b] = '\0';

        out->ptr = buf;
        out->len = (int64_t)(a + b);
        return 0;
    }

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

    if (sig == 0) {
        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
        if (!h) {
            DWORD we = GetLastError();
            if (we == ERROR_INVALID_PARAMETER) {
                ark_err_set_errno(out_err, ESRCH);
                return (ArkStatus)ESRCH;
            }
            ark_err_set_win32(out_err, we);
            return (ArkStatus)out_err->code;
        }
        CloseHandle(h);
        return 0;
    }

    HANDLE h = OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) {
        DWORD we = GetLastError();
        if (we == ERROR_INVALID_PARAMETER) {
            ark_err_set_errno(out_err, ESRCH);
            return (ArkStatus)ESRCH;
        }
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    if (!TerminateProcess(h, (UINT)sig)) {
        DWORD we = GetLastError();
        CloseHandle(h);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    CloseHandle(h);
    return 0;
}

ArkStatus __ark_sys_nice(int32_t inc, int32_t* out_new_prio, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_new_prio) *out_new_prio = 0;

    HANDLE self = GetCurrentProcess();
    DWORD cur = GetPriorityClass(self);
    if (cur == 0) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    int32_t cur_nice = ark_prio_class_to_nice(cur);
    int32_t target_nice = cur_nice + inc;

    if (target_nice < -20) target_nice = -20;
    if (target_nice > 19)  target_nice = 19;

    DWORD target_cls = ark_nice_to_prio_class(target_nice);
    if (!SetPriorityClass(self, target_cls)) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    if (out_new_prio) {
        DWORD now = GetPriorityClass(self);
        if (now == 0) now = target_cls;
        *out_new_prio = ark_prio_class_to_nice(now);
    }

    return 0;
}

// =============================================================================
// non-returning control
// =============================================================================

void __ark_sys_exit(int32_t code) {
    ExitProcess((UINT)code);
}

void __ark_sys_abort(void) {
    RaiseFailFastException(NULL, NULL, 0);
    TerminateProcess(GetCurrentProcess(), 134);
}

#if defined(__cplusplus)
} // extern "C"
#endif