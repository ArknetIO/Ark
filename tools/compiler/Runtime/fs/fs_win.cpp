/* Arklang Windows Filesystem Kernel
 * Refactor: ArkStatus-return ABI + out_err (optional).
 *
 * Goal:
 *   - Match the fs_posix.cpp ABI and behavior as closely as Windows allows.
 *   - Keep a separate translation unit instead of macro-splitting one file.
 *
 * Notes:
 *   - Directory-fsync semantics are approximated with directory handle flushing.
 *   - dirfd-relative APIs are emulated by resolving the directory handle path.
 *   - Generic ioctl is not available on Win32 CRT handles; unsupported requests
 *     return ENOTSUP.
 */

#include <ark_protocol.h>

#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>

#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#ifndef AT_FDCWD
#define AT_FDCWD (-100)
#endif

#ifndef AT_REMOVEDIR
#define AT_REMOVEDIR 0x200
#endif

#ifndef AT_SYMLINK_NOFOLLOW
#define AT_SYMLINK_NOFOLLOW 0x100
#endif

#ifndef ENOTSUP
#define ENOTSUP EINVAL
#endif

extern "C" {

// -----------------------------------------------------------------------------
// Error Helpers (owned msg; out_err optional)
// -----------------------------------------------------------------------------

static inline void ark_err_clear(ArkIoError* e) {
    if (!e) return;
    e->code = 0;
    e->_pad = 0;
    e->msg.ptr = NULL;
    e->msg.len = 0;
}

static inline size_t ark_trim_winmsg_len(const char* s, size_t n) {
    while (n != 0) {
        const char ch = s[n - 1];
        if (ch != '\r' && ch != '\n' && ch != ' ' && ch != '\t') break;
        --n;
    }
    return n;
}

static inline void ark_err_set_owned_msg(ArkIoError* e, const char* s, size_t n) {
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

static inline void ark_err_set_errno(ArkIoError* e, int code) {
    if (!e) return;

    e->code = (int32_t)code;
    e->_pad = 0;
    e->msg.ptr = NULL;
    e->msg.len = 0;

    char tmp[128];
    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    if (!s) s = "unknown error";

    ark_err_set_owned_msg(e, s, strlen(s));
}


static inline void ark_err_set_win32(ArkIoError* e, DWORD winerr) {
    if (!e) return;

    e->code = (int32_t)ark_winerr_to_errno(winerr);
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
        const size_t n = ark_trim_winmsg_len(msg, (size_t)got);
        if (n != 0) {
            ark_err_set_owned_msg(e, msg, n);
            if (e->msg.ptr) return;
        }
    }

    char tmp[128];
    const char* s = ark_strerror_cstr((int)e->code, tmp, sizeof(tmp));
    if (!s) s = "unknown error";
    ark_err_set_owned_msg(e, s, strlen(s));
}

// durability: 0=strict_posix, 1=strict_db, 2=best_effort
static inline bool ark_dir_fsync_ignored(int durability, int err) {
    if (durability >= 2) return true;
    if (durability == 1) {
        return err == EINVAL || err == ENOTSUP || err == EACCES || err == EPERM;
    }
    return false;
}

static inline size_t ark_read_chunk_limit(int64_t remaining) {
    if (remaining <= 0) return 0;
    const int64_t lim = (int64_t)INT_MAX;
    return (size_t)(remaining > lim ? lim : remaining);
}

static inline size_t ark_write_chunk_limit(int64_t remaining) {
    if (remaining <= 0) return 0;
    const int64_t lim = (int64_t)INT_MAX;
    return (size_t)(remaining > lim ? lim : remaining);
}

// -----------------------------------------------------------------------------
// Path Helpers
// -----------------------------------------------------------------------------

static inline bool ark_is_sep(char c) {
    return c == '\\' || c == '/';
}

static inline bool ark_is_abs_path(const char* path) {
    if (!path || !*path) return false;

    if (ark_is_sep(path[0]) && ark_is_sep(path[1])) return true;
    if (path[0] == '\\' && path[1] == '?' && path[2] == '\\') return true;
    if (((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) &&
        path[1] == ':' && ark_is_sep(path[2])) {
        return true;
    }

    return false;
}

static inline char* ark_alloc_cstr_copy_n(const char* s, size_t n) {
    char* out = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    if (!out) return nullptr;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static inline char* ark_alloc_cstr_copy(const char* s) {
    if (!s) return nullptr;
    return ark_alloc_cstr_copy_n(s, strlen(s));
}

static inline void ark_strip_verbatim_prefix(char* path) {
    if (!path) return;

    // \\?\C:\foo -> C:\foo
    if (strncmp(path, "\\\\?\\", 4) == 0) {
        if (strncmp(path + 4, "UNC\\", 4) == 0) {
            // \\?\UNC\server\share -> \\server\share
            size_t tail_len = strlen(path + 8);
            memmove(path + 2, path + 8, tail_len + 1);
            path[0] = '\\';
            path[1] = '\\';
        } else {
            size_t tail_len = strlen(path + 4);
            memmove(path, path + 4, tail_len + 1);
        }
    }
}

static inline ArkStatus ark_getcwd_owned(char** out_path, ArkIoError* out_err) {
    if (!out_path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_path = nullptr;

    int cap = 260;
    for (;;) {
        char* tmp = (char*)__ark_alloc((uint64_t)cap, 1);
        if (!tmp) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        if (_getcwd(tmp, cap) != nullptr) {
            *out_path = tmp;
            return 0;
        }

        int e = errno;
        __ark_free(tmp);

        if (e == ERANGE) {
            if (cap > INT_MAX / 2) {
                ark_err_set_errno(out_err, ENAMETOOLONG);
                return (ArkStatus)ENAMETOOLONG;
            }
            cap *= 2;
            continue;
        }

        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
}

static inline ArkStatus ark_dirfd_path(int64_t dirfd, char** out_path, ArkIoError* out_err) {
    if (!out_path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_path = nullptr;

    if ((int)dirfd == AT_FDCWD) {
        return ark_getcwd_owned(out_path, out_err);
    }

    intptr_t osfh = _get_osfhandle((int)dirfd);
    if (osfh == -1) {
        ark_err_set_errno(out_err, EBADF);
        return (ArkStatus)EBADF;
    }

    HANDLE h = (HANDLE)osfh;

    DWORD need = GetFinalPathNameByHandleA(h, NULL, 0, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (need == 0) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    char* buf = (char*)__ark_alloc((uint64_t)(need + 1), 1);
    if (!buf) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    DWORD got = GetFinalPathNameByHandleA(h, buf, need + 1, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (got == 0 || got > need) {
        DWORD we = GetLastError();
        __ark_free(buf);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    ark_strip_verbatim_prefix(buf);
    *out_path = buf;
    return 0;
}

static inline ArkStatus ark_join_paths_owned(const char* base,
                                             const char* tail,
                                             char** out_path,
                                             ArkIoError* out_err) {
    if (!base || !tail || !out_path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_path = nullptr;

    size_t a = strlen(base);
    size_t b = strlen(tail);
    bool need_sep = (a > 0 && !ark_is_sep(base[a - 1]));

    size_t total = a + (need_sep ? 1u : 0u) + b;
    char* out = (char*)__ark_alloc((uint64_t)(total + 1), 1);
    if (!out) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    memcpy(out, base, a);
    size_t pos = a;

    if (need_sep) {
        out[pos++] = '\\';
    }

    memcpy(out + pos, tail, b);
    pos += b;
    out[pos] = '\0';

    *out_path = out;
    return 0;
}

static inline ArkStatus ark_resolve_path_at(int64_t dirfd,
                                            const char* path,
                                            char** out_path,
                                            ArkIoError* out_err) {
    if (!path || !*path || !out_path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_path = nullptr;

    if (ark_is_abs_path(path)) {
        *out_path = ark_alloc_cstr_copy(path);
        if (!*out_path) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }
        return 0;
    }

    char* base = nullptr;
    ArkStatus st = ark_dirfd_path(dirfd, &base, out_err);
    if (st != 0) {
        return st;
    }

    st = ark_join_paths_owned(base, path, out_path, out_err);
    __ark_free(base);
    return st;
}

static inline ArkStatus ark_parent_dir_owned(const char* path, char** out_dir, ArkIoError* out_err) {
    if (!path || !*path || !out_dir) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    *out_dir = nullptr;

    const char* last_sep = nullptr;
    for (const char* p = path; *p; ++p) {
        if (ark_is_sep(*p)) {
            last_sep = p;
        }
    }

    if (!last_sep) {
        *out_dir = ark_alloc_cstr_copy(".");
        if (!*out_dir) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }
        return 0;
    }

    // Preserve "C:\" roots.
    size_t n = (size_t)(last_sep - path);
    if (n == 2 && path[1] == ':') {
        n = 3;
    }
    if (n == 0) {
        n = 1;
    }

    *out_dir = ark_alloc_cstr_copy_n(path, n);
    if (!*out_dir) {
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    return 0;
}

static inline DWORD ark_open_flags_for_temp_write(void) {
    return FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;
}

static inline ArkStatus ark_flush_directory_best_match(const char* path, int durability, ArkIoError* out_err) {
    if (durability >= 2) {
        return 0;
    }

    char* dir = nullptr;
    ArkStatus st = ark_parent_dir_owned(path, &dir, out_err);
    if (st != 0) {
        if (durability == 0) return st;
        return 0;
    }

    HANDLE h = CreateFileA(
        dir,
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        NULL
    );

    if (h == INVALID_HANDLE_VALUE) {
        DWORD we = GetLastError();
        int e = ark_winerr_to_errno(we);
        __ark_free(dir);

        if (ark_dir_fsync_ignored(durability, e)) {
            return 0;
        }

        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    BOOL ok = FlushFileBuffers(h);
    DWORD we = ok ? 0 : GetLastError();
    CloseHandle(h);
    __ark_free(dir);

    if (!ok) {
        int e = ark_winerr_to_errno(we);
        if (ark_dir_fsync_ignored(durability, e)) {
            return 0;
        }
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

static inline ArkStatus ark_set_cloexec_fd(int fd, ArkIoError* out_err) {
    intptr_t osfh = _get_osfhandle(fd);
    if (osfh == -1) {
        ark_err_set_errno(out_err, EBADF);
        return (ArkStatus)EBADF;
    }

    if (!SetHandleInformation((HANDLE)osfh, HANDLE_FLAG_INHERIT, 0)) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    return 0;
}

static inline ArkStatus ark_fd_handle(int64_t fdv, HANDLE* out_handle, ArkIoError* out_err) {
    if (!out_handle) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    intptr_t osfh = _get_osfhandle((int)fdv);
    if (osfh == -1) {
        ark_err_set_errno(out_err, EBADF);
        return (ArkStatus)EBADF;
    }

    *out_handle = (HANDLE)osfh;
    return 0;
}

static inline void ark_fill_overlapped(int64_t off, OVERLAPPED* ov) {
    ZeroMemory(ov, sizeof(*ov));
    ov->Offset = (DWORD)(off & 0xFFFFFFFFu);
    ov->OffsetHigh = (DWORD)(((uint64_t)off >> 32) & 0xFFFFFFFFu);
}

// -----------------------------------------------------------------------------
// Filesystem: ACID I/O Contract (High-level Atomic Writes)
// -----------------------------------------------------------------------------

static ArkStatus ark_write_atomic_core(const char* path,
                                       const void* data,
                                       int64_t len,
                                       int durability,
                                       ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || (len > 0 && !data)) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char* dir = nullptr;
    ArkStatus st = ark_parent_dir_owned(path, &dir, out_err);
    if (st != 0) {
        return st;
    }

    char temp_path[MAX_PATH];
    UINT tmp_rc = GetTempFileNameA(dir, "ark", 0, temp_path);
    __ark_free(dir);

    if (tmp_rc == 0) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    int fd = _open(temp_path, _O_WRONLY | _O_BINARY | _O_TRUNC | _O_NOINHERIT, _S_IREAD | _S_IWRITE);
    if (fd < 0) {
        int e = errno;
        DeleteFileA(temp_path);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    int64_t written = 0;
    const uint8_t* p = (const uint8_t*)data;

    while (written < len) {
        size_t chunk = ark_write_chunk_limit(len - written);
        int w = _write(fd, p + written, (unsigned int)chunk);

        if (w < 0) {
            if (errno == EINTR) continue;

            int e = errno;
            (void)_close(fd);
            (void)DeleteFileA(temp_path);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        if (w == 0) {
            (void)_close(fd);
            (void)DeleteFileA(temp_path);
            ark_err_set_errno(out_err, EIO);
            return (ArkStatus)EIO;
        }

        written += (int64_t)w;
    }

    if (_commit(fd) < 0) {
        int e = errno;
        (void)_close(fd);
        (void)DeleteFileA(temp_path);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (_close(fd) < 0) {
        int e = errno;
        (void)DeleteFileA(temp_path);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (!MoveFileExA(temp_path, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DWORD we = GetLastError();
        (void)DeleteFileA(temp_path);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    st = ark_flush_directory_best_match(path, durability, out_err);
    if (st != 0) {
        return st;
    }

    return 0;
}

ArkStatus __ark_file_write_atomic_strict_posix(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    return ark_write_atomic_core(path, content, len, 0, out_err);
}

ArkStatus __ark_file_write_atomic_strict_db(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    return ark_write_atomic_core(path, content, len, 1, out_err);
}

ArkStatus __ark_file_write_atomic_best_effort(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    return ark_write_atomic_core(path, content, len, 2, out_err);
}

// -----------------------------------------------------------------------------
// Filesystem: Read Operations (Strict vs Upto)
// -----------------------------------------------------------------------------

// Helper function: Reads entire file into an ArkStr
static ArkStatus ark_read_all_core(const char* path,
                                   int64_t max_bytes,
                                   bool strict,
                                   ArkStr* out,
                                   ArkIoError* out_err) {
    // 1. Reset Output
    ark_err_clear(out_err);
    if (out) {
        out->ptr = nullptr;
        out->len = 0;
    }

    // 2. Validate Inputs
    if (!path || !*path || !out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (max_bytes < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    // 3. Open File
    int fd = _open(path, _O_RDONLY | _O_BINARY | _O_NOINHERIT);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    // 4. Determine Initial Capacity
    int64_t cap = 4096;
    if (max_bytes > 0 && cap > max_bytes) cap = max_bytes;

    struct _stat64 st;
    if (_fstat64(fd, &st) == 0 && st.st_size > 0) {
        int64_t want = (int64_t)st.st_size + 1;
        if (want > cap) cap = want;
        if (max_bytes > 0 && cap > max_bytes) cap = max_bytes;
    }

    if (cap <= 0) cap = 1;

    // 5. Initial Allocation
    char* buf = (char*)__ark_alloc((uint64_t)(cap + 1), 1);
    if (!buf) {
        (void)_close(fd);
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    int64_t total = 0;

    // 6. Read Loop
    for (;;) {
        if (total == cap) {
            if (max_bytes > 0 && total >= max_bytes) {
                if (strict) {
                    __ark_free(buf);
                    (void)_close(fd);
                    ark_err_set_errno(out_err, EFBIG);
                    return (ArkStatus)EFBIG;
                }
                break;
            }

            int64_t new_cap = cap * 2;
            if (new_cap < cap) {
                __ark_free(buf);
                (void)_close(fd);
                ark_err_set_errno(out_err, EOVERFLOW);
                return (ArkStatus)EOVERFLOW;
            }
            if (max_bytes > 0 && new_cap > max_bytes) new_cap = max_bytes;
            if (new_cap <= cap) new_cap = cap + 1;

            char* nb = (char*)__ark_alloc((uint64_t)(new_cap + 1), 1);
            if (!nb) {
                __ark_free(buf);
                (void)_close(fd);
                ark_err_set_errno(out_err, ENOMEM);
                return (ArkStatus)ENOMEM;
            }

            memcpy(nb, buf, (size_t)total);
            __ark_free(buf);
            buf = nb;
            cap = new_cap;
        }

        size_t space = (size_t)(cap - total);
        if (space > (size_t)INT_MAX) space = (size_t)INT_MAX;

        int r = _read(fd, buf + total, (unsigned int)space);
        if (r < 0) {
            int e = errno;
            __ark_free(buf);
            (void)_close(fd);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        if (r == 0) break;

        total += (int64_t)r;
    }

    (void)_close(fd);

    // 7. Finalize
    buf[total] = '\0';
    out->ptr = buf;
    out->len = total;
    return 0;
}

ArkStatus __ark_file_read_all_strict(const char* path, int64_t max_bytes, ArkStr* out, ArkIoError* out_err) {
    return ark_read_all_core(path, max_bytes, true, out, out_err);
}

ArkStatus __ark_file_read_all_upto(const char* path, int64_t max_bytes, ArkStr* out, ArkIoError* out_err) {
    return ark_read_all_core(path, max_bytes, false, out, out_err);
}

// __ark_file_write_atomic: legacy binding -> pick a default durability policy.
// Chosen default: strict_db.
ArkStatus __ark_file_write_atomic(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    return __ark_file_write_atomic_strict_db(path, content, len, out_err);
}

// __ark_file_append: open(O_APPEND) + write-all + fsync(file).
ArkStatus __ark_file_append(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path || (len > 0 && !content) || len < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fd = _open(path, _O_WRONLY | _O_CREAT | _O_APPEND | _O_BINARY | _O_NOINHERIT, _S_IREAD | _S_IWRITE);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    const uint8_t* p = (const uint8_t*)content;
    int64_t off = 0;

    while (off < len) {
        size_t chunk = ark_write_chunk_limit(len - off);
        int w = _write(fd, p + off, (unsigned int)chunk);

        if (w < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            (void)_close(fd);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        if (w == 0) {
            (void)_close(fd);
            ark_err_set_errno(out_err, EIO);
            return (ArkStatus)EIO;
        }

        off += (int64_t)w;
    }

    if (_commit(fd) < 0) {
        int e = errno;
        (void)_close(fd);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (_close(fd) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

// __ark_file_read_all: legacy binding -> choose a default policy.
ArkStatus __ark_file_read_all(const char* path, ArkStr* out, ArkIoError* out_err) {
    return __ark_file_read_all_upto(path, INT64_MAX, out, out_err);
}

// __ark_file_exists: returns existence boolean.
ArkStatus __ark_file_exists(const char* path, bool* out_exists, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_exists) *out_exists = false;

    if (!path || !*path || !out_exists) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct _stat64 st;
    if (_stat64(path, &st) == 0) {
        *out_exists = true;
        return 0;
    }

    int e = errno;
    if (e == ENOENT) {
        *out_exists = false;
        return 0;
    }

    ark_err_set_errno(out_err, e);
    return (ArkStatus)e;
}

// __ark_fd_fstatat
// NOTE: On Windows, out_stat points to caller-provided `struct _stat64` storage.
ArkStatus __ark_fd_fstatat(int64_t dirfd, const char* path, void* out_stat, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path || !out_stat) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if ((flags & ~AT_SYMLINK_NOFOLLOW) != 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (flags & AT_SYMLINK_NOFOLLOW) {
        ark_err_set_errno(out_err, ENOTSUP);
        return (ArkStatus)ENOTSUP;
    }

    char* full = nullptr;
    ArkStatus st = ark_resolve_path_at(dirfd, path, &full, out_err);
    if (st != 0) return st;

    if (_stat64(full, (struct _stat64*)out_stat) != 0) {
        int e = errno;
        __ark_free(full);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    __ark_free(full);
    return 0;
}

// Close variants:
// - best_effort: never fails
// - unsafe: closes, returns status if close fails
// - strict: same on Windows; close is not EINTR-retriable in CRT

ArkStatus __ark_fd_close_best_effort(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);
    (void)_close((int)fdv);
    return 0;
}

ArkStatus __ark_fd_close_unsafe(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (_close((int)fdv) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_close_strict(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (_close((int)fdv) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_pread(int64_t fdv,
                         void* buf,
                         int64_t cap,
                         int64_t off,
                         int64_t* out_n,
                         ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!buf || cap < 0 || off < 0 || !out_n) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    HANDLE h = INVALID_HANDLE_VALUE;
    ArkStatus st = ark_fd_handle(fdv, &h, out_err);
    if (st != 0) return st;

    OVERLAPPED ov;
    ark_fill_overlapped(off, &ov);

    DWORD got = 0;
    DWORD want = (DWORD)ark_read_chunk_limit(cap);

    BOOL ok = ReadFile(h, buf, want, &got, &ov);
    if (!ok) {
        DWORD we = GetLastError();
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    *out_n = (int64_t)got;
    return 0;
}

ArkStatus __ark_fd_pwrite(int64_t fdv,
                          const void* buf,
                          int64_t len,
                          int64_t off,
                          int64_t* out_n,
                          ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!buf || len < 0 || off < 0 || !out_n) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    HANDLE h = INVALID_HANDLE_VALUE;
    ArkStatus st = ark_fd_handle(fdv, &h, out_err);
    if (st != 0) return st;

    OVERLAPPED ov;
    ark_fill_overlapped(off, &ov);

    DWORD done = 0;
    DWORD want = (DWORD)ark_write_chunk_limit(len);

    BOOL ok = WriteFile(h, buf, want, &done, &ov);
    if (!ok) {
        DWORD we = GetLastError();
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    *out_n = (int64_t)done;
    return 0;
}

ArkStatus __ark_fd_pipe(int64_t* out_r, int64_t* out_w, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_r) *out_r = -1;
    if (out_w) *out_w = -1;

    if (!out_r || !out_w) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fds[2];
    if (_pipe(fds, 4096, _O_BINARY) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_r = (int64_t)fds[0];
    *out_w = (int64_t)fds[1];
    return 0;
}

ArkStatus __ark_fd_write_all(int64_t fdv,
                             const void* buf,
                             int64_t len,
                             int64_t* out_n,
                             ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!out_n || (len > 0 && !buf) || len < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (len == 0) return 0;

    const uint8_t* p = (const uint8_t*)buf;
    int64_t written = 0;

    while (written < len) {
        size_t chunk = ark_write_chunk_limit(len - written);
        int w = _write((int)fdv, p + written, (unsigned int)chunk);

        if (w < 0) {
            if (errno == EINTR) continue;

            int e = errno;
            if (e == EAGAIN || e == EWOULDBLOCK) {
                *out_n = written;
                ark_err_set_errno(out_err, e);
                return (ArkStatus)e;
            }

            *out_n = written;
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        if (w == 0) {
            *out_n = written;
            ark_err_set_errno(out_err, EIO);
            return (ArkStatus)EIO;
        }

        written += (int64_t)w;
    }

    *out_n = written;
    return 0;
}

ArkStatus __ark_fd_pwrite_all(int64_t fdv,
                              const void* buf,
                              int64_t len,
                              int64_t off,
                              int64_t* out_n,
                              ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!out_n || (len > 0 && !buf) || len < 0 || off < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (len == 0) return 0;

    HANDLE h = INVALID_HANDLE_VALUE;
    ArkStatus st = ark_fd_handle(fdv, &h, out_err);
    if (st != 0) return st;

    const uint8_t* p = (const uint8_t*)buf;
    int64_t written = 0;
    int64_t pos = off;

    while (written < len) {
        size_t chunk = ark_write_chunk_limit(len - written);

        OVERLAPPED ov;
        ark_fill_overlapped(pos, &ov);

        DWORD done = 0;
        BOOL ok = WriteFile(h, p + written, (DWORD)chunk, &done, &ov);

        if (!ok) {
            DWORD we = GetLastError();
            int e = ark_winerr_to_errno(we);
            *out_n = written;
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        if (done == 0) {
            *out_n = written;
            ark_err_set_errno(out_err, EIO);
            return (ArkStatus)EIO;
        }

        written += (int64_t)done;
        pos += (int64_t)done;
    }

    *out_n = written;
    return 0;
}

ArkStatus __ark_fd_read_exact(int64_t fdv,
                              void* buf,
                              int64_t cap,
                              int64_t* out_n,
                              ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!out_n || !buf || cap < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (cap == 0) return 0;

    uint8_t* p = (uint8_t*)buf;
    int64_t total = 0;

    while (total < cap) {
        size_t chunk = ark_read_chunk_limit(cap - total);
        int r = _read((int)fdv, p + total, (unsigned int)chunk);

        if (r < 0) {
            if (errno == EINTR) continue;

            int e = errno;
            if (e == EAGAIN || e == EWOULDBLOCK) {
                *out_n = total;
                ark_err_set_errno(out_err, e);
                return (ArkStatus)e;
            }

            *out_n = total;
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        if (r == 0) {
            *out_n = total;
            ark_err_set_errno(out_err, ENODATA);
            return (ArkStatus)ENODATA;
        }

        total += (int64_t)r;
    }

    *out_n = total;
    return 0;
}

// -----------------------------------------------------------------------------
// Low-Level FD Kernel: Foundation Ops
// -----------------------------------------------------------------------------

ArkStatus __ark_fd_open(const char* path, int flags, int mode, int64_t* out_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_fd) *out_fd = -1;

    if (!path || !*path || !out_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fd = _open(path, flags | _O_BINARY | _O_NOINHERIT, mode);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_fd = (int64_t)fd;
    return 0;
}

ArkStatus __ark_fd_openat(int64_t dirfd,
                          const char* path,
                          int flags,
                          int mode,
                          int64_t* out_fd,
                          ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_fd) *out_fd = -1;

    if (!path || !*path || !out_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char* full = nullptr;
    ArkStatus st = ark_resolve_path_at(dirfd, path, &full, out_err);
    if (st != 0) return st;

    int fd = _open(full, flags | _O_BINARY | _O_NOINHERIT, mode);
    __ark_free(full);

    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_fd = (int64_t)fd;
    return 0;
}

ArkStatus __ark_fd_write(int64_t fdv, const void* buf, int64_t len, int64_t* out_n, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!buf || len < 0 || !out_n) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int w = _write((int)fdv, buf, (unsigned int)ark_write_chunk_limit(len));
    if (w < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_n = (int64_t)w;
    return 0;
}

ArkStatus __ark_fd_read(int64_t fdv, void* buf, int64_t cap, int64_t* out_n, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!buf || cap < 0 || !out_n) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r = _read((int)fdv, buf, (unsigned int)ark_read_chunk_limit(cap));
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_n = (int64_t)r;
    return 0;
}

ArkStatus __ark_fd_close(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (_close((int)fdv) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_readlinkat(int64_t dirfd, const char* path, ArkStr* out, ArkIoError* out_err) {
    // 1. Reset State
    ark_err_clear(out_err);
    if (out) {
        out->ptr = nullptr;
        out->len = 0;
    }

    // 2. Validate
    if (!path || !*path || !out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char* full = nullptr;
    ArkStatus st = ark_resolve_path_at(dirfd, path, &full, out_err);
    if (st != 0) return st;

    HANDLE h = CreateFileA(
        full,
        0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        NULL
    );

    __ark_free(full);

    if (h == INVALID_HANDLE_VALUE) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    DWORD cap = 256;
    const DWORD MAX_SANITY = 1024 * 1024;

    for (;;) {
        char* buf = (char*)__ark_alloc((uint64_t)(cap + 1), 1);
        if (!buf) {
            CloseHandle(h);
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        DWORD got = GetFinalPathNameByHandleA(h, buf, cap, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if (got == 0) {
            DWORD we = GetLastError();
            __ark_free(buf);
            CloseHandle(h);
            ark_err_set_win32(out_err, we);
            return (ArkStatus)out_err->code;
        }

        if (got < cap) {
            buf[got] = '\0';
            ark_strip_verbatim_prefix(buf);
            out->ptr = buf;
            out->len = (int64_t)strlen(buf);
            CloseHandle(h);
            return 0;
        }

        __ark_free(buf);

        if (cap >= MAX_SANITY) {
            CloseHandle(h);
            ark_err_set_errno(out_err, ENAMETOOLONG);
            return (ArkStatus)ENAMETOOLONG;
        }

        cap *= 2;
    }
}

ArkStatus __ark_fd_mkdirat(int64_t dirfd, const char* path, int mode, ArkIoError* out_err) {
    (void)mode;
    ark_err_clear(out_err);

    if (!path || !*path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char* full = nullptr;
    ArkStatus st = ark_resolve_path_at(dirfd, path, &full, out_err);
    if (st != 0) return st;

    int r = _mkdir(full);
    int e = (r == 0) ? 0 : errno;
    __ark_free(full);

    if (r != 0) {
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_unlinkat(int64_t dirfd, const char* path, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char* full = nullptr;
    ArkStatus st = ark_resolve_path_at(dirfd, path, &full, out_err);
    if (st != 0) return st;

    BOOL ok;
    if (flags & AT_REMOVEDIR) {
        ok = RemoveDirectoryA(full);
    } else {
        ok = DeleteFileA(full);
    }

    if (!ok) {
        DWORD we = GetLastError();
        __ark_free(full);
        ark_err_set_win32(out_err, we);
        return (ArkStatus)out_err->code;
    }

    __ark_free(full);
    return 0;
}

ArkStatus __ark_fd_renameat(int64_t olddirfd,
                            const char* oldpath,
                            int64_t newdirfd,
                            const char* newpath,
                            ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!oldpath || !*oldpath || !newpath || !*newpath) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char* old_full = nullptr;
    char* new_full = nullptr;

    ArkStatus st = ark_resolve_path_at(olddirfd, oldpath, &old_full, out_err);
    if (st != 0) return st;

    st = ark_resolve_path_at(newdirfd, newpath, &new_full, out_err);
    if (st != 0) {
        __ark_free(old_full);
        return st;
    }

    BOOL ok = MoveFileExA(old_full, new_full, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED);
    __ark_free(old_full);
    __ark_free(new_full);

    if (!ok) {
        ark_err_set_win32(out_err, GetLastError());
        return (ArkStatus)out_err->code;
    }

    return 0;
}

ArkStatus __ark_fd_seek(int64_t fdv, int64_t off, int whence, int64_t* out_off, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_off) *out_off = 0;

    if (!out_off) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    __int64 r = _lseeki64((int)fdv, off, whence);
    if (r == -1) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_off = (int64_t)r;
    return 0;
}

ArkStatus __ark_fd_isatty(int64_t fdv, bool* out_is_tty, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_is_tty) *out_is_tty = false;

    if (!out_is_tty) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    errno = 0;
    int r = _isatty((int)fdv);
    *out_is_tty = (r != 0);

    if (r == 0 && errno != 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_pipe_cloexec(int64_t* out_r, int64_t* out_w, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_r) *out_r = -1;
    if (out_w) *out_w = -1;

    if (!out_r || !out_w) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fds[2];
    if (_pipe(fds, 4096, _O_BINARY) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    ArkStatus s = ark_set_cloexec_fd(fds[0], out_err);
    if (s != 0) {
        (void)_close(fds[0]);
        (void)_close(fds[1]);
        return s;
    }

    s = ark_set_cloexec_fd(fds[1], out_err);
    if (s != 0) {
        (void)_close(fds[0]);
        (void)_close(fds[1]);
        return s;
    }

    *out_r = (int64_t)fds[0];
    *out_w = (int64_t)fds[1];
    return 0;
}

ArkStatus __ark_fd_dup_cloexec(int64_t old_fd, int64_t* out_new_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_new_fd) *out_new_fd = -1;

    if (!out_new_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fd = _dup((int)old_fd);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    ArkStatus s = ark_set_cloexec_fd(fd, out_err);
    if (s != 0) {
        (void)_close(fd);
        return s;
    }

    *out_new_fd = (int64_t)fd;
    return 0;
}

// dup3 (emulated on Windows)
// Semantics: if old_fd == new_fd => EINVAL.
ArkStatus __ark_fd_dup3(int64_t old_fd, int64_t new_fd, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if ((int)old_fd == (int)new_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if ((flags & ~(O_CLOEXEC)) != 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (_dup2((int)old_fd, (int)new_fd) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (flags & O_CLOEXEC) {
        ArkStatus s = ark_set_cloexec_fd((int)new_fd, out_err);
        if (s != 0) return s;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// __ark_fd_dup3_best_effort
// Semantics:
//   - Always falls back to dup2 + optional CLOEXEC on Windows.
//   - Unknown flags are ignored best-effort.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_dup3_best_effort(int64_t old_fd, int64_t new_fd, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if ((int)old_fd == (int)new_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (_dup2((int)old_fd, (int)new_fd) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (flags & O_CLOEXEC) {
        ArkStatus s = ark_set_cloexec_fd((int)new_fd, out_err);
        if (s != 0) return s;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// __ark_fd_ioctl
// Generic ioctl is not provided for Win32 CRT fds.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_ioctl(int64_t fdv, uint64_t req, void* arg, ArkIoError* out_err) {
    (void)fdv;
    (void)req;
    (void)arg;
    ark_err_clear(out_err);
    ark_err_set_errno(out_err, ENOTSUP);
    return (ArkStatus)ENOTSUP;
}

// -----------------------------------------------------------------------------
// __ark_fd_set_nonblock
// Best-effort on Windows:
//   - Named pipes: try PIPE_NOWAIT / PIPE_WAIT.
//   - Other CRT fd types: unsupported.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_set_nonblock(int64_t fdv, bool enabled, ArkIoError* out_err) {
    ark_err_clear(out_err);

    HANDLE h = INVALID_HANDLE_VALUE;
    ArkStatus st = ark_fd_handle(fdv, &h, out_err);
    if (st != 0) return st;

    DWORD type = GetFileType(h);
    if (type == FILE_TYPE_PIPE) {
        DWORD mode = enabled ? PIPE_NOWAIT : PIPE_WAIT;
        if (!SetNamedPipeHandleState(h, &mode, NULL, NULL)) {
            ark_err_set_win32(out_err, GetLastError());
            return (ArkStatus)out_err->code;
        }
        return 0;
    }

    ark_err_set_errno(out_err, ENOTSUP);
    return (ArkStatus)ENOTSUP;
}

// -----------------------------------------------------------------------------
// __ark_fd_poll1
// Best-effort on Windows:
//   - Disk and char handles are treated as immediately ready.
//   - Pipe handles are probed via PeekNamedPipe.
//   - timeout_ms is honored for the pipe-empty case via Sleep polling.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_poll1(int64_t fdv, int events, int timeout_ms, int* out_revents, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_revents) *out_revents = 0;

    if (!out_revents) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    HANDLE h = INVALID_HANDLE_VALUE;
    ArkStatus st = ark_fd_handle(fdv, &h, out_err);
    if (st != 0) return st;

    DWORD type = GetFileType(h);

    if (type == FILE_TYPE_DISK || type == FILE_TYPE_CHAR) {
        *out_revents = events;
        return 0;
    }

    if (type == FILE_TYPE_PIPE) {
        DWORD waited = 0;
        DWORD step = 10;

        for (;;) {
            DWORD avail = 0;
            BOOL ok = PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL);
            if (!ok) {
                ark_err_set_win32(out_err, GetLastError());
                return (ArkStatus)out_err->code;
            }

            if (avail > 0) {
                *out_revents = events;
                return 0;
            }

            if (timeout_ms == 0) {
                *out_revents = 0;
                return 0;
            }

            if (timeout_ms > 0 && waited >= (DWORD)timeout_ms) {
                *out_revents = 0;
                return 0;
            }

            DWORD sleep_ms = step;
            if (timeout_ms > 0) {
                DWORD remaining = (DWORD)timeout_ms - waited;
                if (sleep_ms > remaining) sleep_ms = remaining;
            }

            Sleep(sleep_ms);
            waited += sleep_ms;
        }
    }

    ark_err_set_errno(out_err, ENOTSUP);
    return (ArkStatus)ENOTSUP;
}

} // extern "C"