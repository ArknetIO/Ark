/* Arklang POSIX Filesystem Kernel
 * Refactor: ArkStatus-return ABI + out_err (optional).
 */

#include <ark_protocol.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>

#include <limits.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <string.h> // <--- Includes 'strlen', 'memcpy' in global namespace

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

    // Reset msg state.
    // NOTE: This does NOT free the old pointer. The caller is responsible for 
    // lifecycle management if reusing ArkIoError structs.
    e->msg.ptr = nullptr;
    e->msg.len = 0;

    char tmp[128];
    const char* s = ark_strerror_cstr(code, tmp, sizeof(tmp));
    if (!s) s = "unknown error"; // Fallback safety

    // Calculate length
    size_t n = strlen(s);

    // [FIX] Use 2-arg __ark_alloc with 1-byte alignment
    // Signature: void* __ark_alloc(uint64_t bytes, uint64_t align)
    char* owned = (char*)__ark_alloc((uint64_t)(n + 1), 1);
    
    if (!owned) return; // OOM: Leave msg empty, code is still set

    memcpy(owned, s, n);
    owned[n] = '\0'; // Null-terminate for safety

    e->msg.ptr = owned;
    e->msg.len = (int64_t)n;
}

// durability: 0=strict_posix, 1=strict_db, 2=best_effort
static inline bool ark_dir_fsync_ignored(int durability, int err) {
    if (durability >= 2) return true;
    if (durability == 1) {
        return err == EINVAL || err == EOPNOTSUPP || err == ENOTSUP;
    }
    return false;
}

static inline size_t ark_read_chunk_limit(int64_t remaining) {
    if (remaining <= 0) return 0;
    const int64_t lim = (int64_t)SSIZE_MAX;
    return (size_t)(remaining > lim ? lim : remaining);
}


static inline size_t ark_write_chunk_limit(int64_t remaining) {
    if (remaining <= 0) return 0;
    const int64_t lim = (int64_t)SSIZE_MAX;
    return (size_t)(remaining > lim ? lim : remaining);
}


// -----------------------------------------------------------------------------
// Filesystem: ACID I/O Contract (High-level Atomic Writes)
// -----------------------------------------------------------------------------

static ArkStatus ark_write_atomic_core(const char* path, const void* data, int64_t len, int durability, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || (len > 0 && !data)) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    char temp_path[1024];
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp.XXXXXX", path) >= (int)sizeof(temp_path)) {
        ark_err_set_errno(out_err, ENAMETOOLONG);
        return (ArkStatus)ENAMETOOLONG;
    }

    int fd = mkstemp(temp_path);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    int64_t written = 0;
    const uint8_t* p = (const uint8_t*)data;

    while (written < len) {
        ssize_t w = write(fd, p + written, (size_t)(len - written));
        if (w < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            (void)close(fd);
            (void)unlink(temp_path);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        written += (int64_t)w;
    }

    if (fsync(fd) < 0) {
        int e = errno;
        (void)close(fd);
        (void)unlink(temp_path);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    (void)close(fd);

    if (rename(temp_path, path) < 0) {
        int e = errno;
        (void)unlink(temp_path);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (durability < 2) {
        char* dpath = strdup(path);
        if (dpath) {
            int dfd = open(dirname(dpath), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (dfd >= 0) {
                if (fsync(dfd) < 0) {
                    int e = errno;
                    if (!ark_dir_fsync_ignored(durability, e)) {
                        (void)close(dfd);
                        free(dpath);
                        ark_err_set_errno(out_err, e);
                        return (ArkStatus)e;
                    }
                }
                (void)close(dfd);
            }
            free(dpath);
        } else if (durability == 0) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }
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
static ArkStatus ark_read_all_core(const char* path, int64_t max_bytes, bool strict, ArkStr* out, ArkIoError* out_err) {
    // 1. Reset Output
    ark_err_clear(out_err);
    if (out) { out->ptr = nullptr; out->len = 0; }

    // 2. Validate Inputs
    if (!path || !*path || !out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (max_bytes < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    // 3. Open File (Retry on EINTR)
    int fd;
    do { fd = ::open(path, O_RDONLY | O_CLOEXEC); } while (fd < 0 && errno == EINTR);
    
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    // 4. Determine Initial Capacity
    int64_t cap = 4096;
    if (max_bytes > 0 && cap > max_bytes) cap = max_bytes;

    struct stat st;
    if (::fstat(fd, &st) == 0 && st.st_size > 0) {
        int64_t want = (int64_t)st.st_size + 1; // +1 for safety/null-term
        if (want > cap) cap = want;
        if (max_bytes > 0 && cap > max_bytes) cap = max_bytes;
    }

    if (cap <= 0) cap = 1;

    // 5. Initial Allocation [FIXED ABI]
    // Signature: void* __ark_alloc(uint64_t bytes, uint64_t align)
    // We use alignment 1 for char buffers.
    char* buf = (char*)__ark_alloc((uint64_t)(cap + 1), 1);
    
    if (!buf) {
        (void)::close(fd);
        ark_err_set_errno(out_err, ENOMEM);
        return (ArkStatus)ENOMEM;
    }

    int64_t total = 0;

    // 6. Read Loop
    for (;;) {
        // Expand buffer if full
        if (total == cap) {
            // Check limits
            if (max_bytes > 0 && total >= max_bytes) {
                if (strict) {
                    __ark_free(buf);
                    (void)::close(fd);
                    ark_err_set_errno(out_err, EFBIG);
                    return (ArkStatus)EFBIG;
                }
                break; // Limit reached, strict=false, return partial
            }

            // Calculate new size (Growth factor 2x)
            int64_t new_cap = cap * 2;
            if (new_cap < cap) { // Overflow check
                __ark_free(buf);
                (void)::close(fd);
                ark_err_set_errno(out_err, EOVERFLOW);
                return (ArkStatus)EOVERFLOW;
            }
            if (max_bytes > 0 && new_cap > max_bytes) new_cap = max_bytes;
            if (new_cap <= cap) new_cap = cap + 1;

            // Manual Realloc [FIXED ABI]
            char* nb = (char*)__ark_alloc((uint64_t)(new_cap + 1), 1);
            if (!nb) {
                __ark_free(buf);
                (void)::close(fd);
                ark_err_set_errno(out_err, ENOMEM);
                return (ArkStatus)ENOMEM;
            }
            
            memcpy(nb, buf, (size_t)total);
            __ark_free(buf);
            buf = nb;
            cap = new_cap;
        }

        // Read chunk
        size_t space = (size_t)(cap - total);
        ssize_t r;
        do { r = ::read(fd, buf + total, space); } while (r < 0 && errno == EINTR);

        if (r < 0) {
            int e = errno;
            __ark_free(buf);
            (void)::close(fd);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        if (r == 0) break; // EOF
        
        total += (int64_t)r;
    }

    (void)::close(fd);

    // 7. Finalize
    buf[total] = '\0'; // Null-terminate (convenience, excluded from len)
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
// Chosen default: strict_db (atomic + fsync file + directory fsync tolerant of ENOTSUP/EOPNOTSUPP/EINVAL).
ArkStatus __ark_file_write_atomic(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    return __ark_file_write_atomic_strict_db(path, content, len, out_err);
}

// __ark_file_append: open(O_APPEND) + write-all + fsync(file).
// Notes:
//   - Not atomic with respect to interleaving writers; relies on kernel O_APPEND semantics.
//   - On success, data is durable to the file. Directory durability is not relevant for append.
ArkStatus __ark_file_append(const char* path, const void* content, int64_t len, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path || (len > 0 && !content) || len < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fd;
    do { fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644); } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    const uint8_t* p = (const uint8_t*)content;
    int64_t off = 0;

    while (off < len) {
        ssize_t w = write(fd, p + off, (size_t)(len - off));
        if (w < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            (void)close(fd);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }
        off += (int64_t)w;
    }

    if (fsync(fd) < 0) {
        int e = errno;
        (void)close(fd);
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (close(fd) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

// __ark_file_read_all: legacy binding -> choose a default policy.
// Chosen default: upto with "no cap" (max_bytes = INT64_MAX), returns entire file.
ArkStatus __ark_file_read_all(const char* path, ArkStr* out, ArkIoError* out_err) {
    return __ark_file_read_all_upto(path, INT64_MAX, out, out_err);
}

// __ark_file_exists: returns existence boolean.
// Semantics:
//   - out_exists=true if path exists (any type).
//   - out_exists=false if it does not exist (ENOENT/ENOTDIR).
//   - other errors return nonzero status.
ArkStatus __ark_file_exists(const char* path, bool* out_exists, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_exists) *out_exists = false;

    if (!path || !*path || !out_exists) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct stat st;
    int r;
    do { r = stat(path, &st); } while (r < 0 && errno == EINTR);
    if (r == 0) {
        *out_exists = true;
        return 0;
    }

    int e = errno;
    if (e == ENOENT || e == ENOTDIR) {
        *out_exists = false;
        return 0;
    }

    ark_err_set_errno(out_err, e);
    return (ArkStatus)e;
}

// __ark_fd_fstatat
// NOTE: out_stat points to caller-provided `struct stat` storage (ABI agreement).
ArkStatus __ark_fd_fstatat(int64_t dirfd, const char* path, void* out_stat, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path || !out_stat) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r;
    do { r = fstatat((int)dirfd, path, (struct stat*)out_stat, flags); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

// Close variants:
// - best_effort: never fails (suppresses close errors).
// - unsafe: closes, returns status if close fails (no retries; simple).
// - strict: retries on EINTR and returns status if close fails (strongest).
//
// Note: POSIX close(2) EINTR semantics are subtle; this is an ABI-level policy choice.

ArkStatus __ark_fd_close_best_effort(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);
    (void)close((int)fdv);
    return 0;
}

ArkStatus __ark_fd_close_unsafe(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (close((int)fdv) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_close_strict(int64_t fdv, ArkIoError* out_err) {
    ark_err_clear(out_err);

    int r;
    do { r = close((int)fdv); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}


ArkStatus __ark_fd_pread(int64_t fdv, void* buf, int64_t cap, int64_t off, int64_t* out_n, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!buf || cap < 0 || off < 0 || !out_n) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    ssize_t r;
    size_t n = (size_t)cap;
    do { r = pread((int)fdv, buf, n, (off_t)off); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_n = (int64_t)r;
    return 0;
}

ArkStatus __ark_fd_pwrite(int64_t fdv, const void* buf, int64_t len, int64_t off, int64_t* out_n, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!buf || len < 0 || off < 0 || !out_n) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    ssize_t w;
    size_t n = (size_t)len;
    do { w = pwrite((int)fdv, buf, n, (off_t)off); } while (w < 0 && errno == EINTR);
    if (w < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_n = (int64_t)w;
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
    int r;
    do { r = pipe(fds); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_r = (int64_t)fds[0];
    *out_w = (int64_t)fds[1];
    return 0;
}

ArkStatus __ark_fd_write_all(int64_t fdv, const void* buf, int64_t len, int64_t* out_n, ArkIoError* out_err) {
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
        ssize_t w = write((int)fdv, p + written, chunk);

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

ArkStatus __ark_fd_pwrite_all(int64_t fdv, const void* buf, int64_t len, int64_t off, int64_t* out_n, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_n) *out_n = 0;

    if (!out_n || (len > 0 && !buf) || len < 0 || off < 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }
    if (len == 0) return 0;

    const uint8_t* p = (const uint8_t*)buf;
    int64_t written = 0;
    int64_t pos = off;

    while (written < len) {
        size_t chunk = ark_write_chunk_limit(len - written);
        ssize_t w = pwrite((int)fdv, p + written, chunk, (off_t)pos);

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
        pos += (int64_t)w;
    }

    *out_n = written;
    return 0;
}

ArkStatus __ark_fd_read_exact(int64_t fdv, void* buf, int64_t cap, int64_t* out_n, ArkIoError* out_err) {
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
        ssize_t r = read((int)fdv, p + total, chunk);

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

    int fd;
    do { fd = open(path, flags | O_CLOEXEC, (mode_t)mode); } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    *out_fd = (int64_t)fd;
    return 0;
}

ArkStatus __ark_fd_openat(int64_t dirfd, const char* path, int flags, int mode, int64_t* out_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_fd) *out_fd = -1;

    if (!path || !*path || !out_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int fd;
    do { fd = openat((int)dirfd, path, flags | O_CLOEXEC, (mode_t)mode); } while (fd < 0 && errno == EINTR);
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

    ssize_t w;
    size_t n = (size_t)len;
    do { w = write((int)fdv, buf, n); } while (w < 0 && errno == EINTR);
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

    ssize_t r;
    size_t n = (size_t)cap;
    do { r = read((int)fdv, buf, n); } while (r < 0 && errno == EINTR);
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

    if (close((int)fdv) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_readlinkat(int64_t dirfd, const char* path, ArkStr* out, ArkIoError* out_err) {
    // 1. Reset State
    ark_err_clear(out_err);
    if (out) { out->ptr = nullptr; out->len = 0; }

    // 2. Validate
    if (!path || !*path || !out) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int64_t cap = 256;
    const int64_t MAX_SANITY = 1024 * 1024; // 1MB Limit for symlinks to prevent DOS

    for (;;) {
        if (cap <= 0) cap = 256;

        // [FIX] Allocator Signature: (uint64_t bytes, uint64_t align)
        // We use alignment 1 for char buffers.
        char* buf = (char*)__ark_alloc((uint64_t)(cap + 1), 1);
        
        if (!buf) {
            ark_err_set_errno(out_err, ENOMEM);
            return (ArkStatus)ENOMEM;
        }

        ssize_t r;
        do { 
            r = ::readlinkat((int)dirfd, path, buf, (size_t)cap); 
        } while (r < 0 && errno == EINTR);

        if (r < 0) {
            int e = errno;
            __ark_free(buf);
            ark_err_set_errno(out_err, e);
            return (ArkStatus)e;
        }

        // Check for truncation
        // readlinkat returns the count of bytes written. 
        // If r < cap, we definitely captured the full path.
        if (r < cap) {
            buf[r] = '\0'; // Null-terminate
            out->ptr = buf;
            out->len = (int64_t)r;
            return 0;
        }

        // If r == cap, the link might be larger. Retry.
        __ark_free(buf);
        
        if (cap >= MAX_SANITY) {
             ark_err_set_errno(out_err, ENAMETOOLONG);
             return (ArkStatus)ENAMETOOLONG;
        }

        cap *= 2;
    }
}

ArkStatus __ark_fd_mkdirat(int64_t dirfd, const char* path, int mode, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!path || !*path) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (mkdirat((int)dirfd, path, (mode_t)mode) < 0) {
        int e = errno;
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

    if (unlinkat((int)dirfd, path, flags) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

ArkStatus __ark_fd_renameat(int64_t olddirfd, const char* oldpath, int64_t newdirfd, const char* newpath, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if (!oldpath || !*oldpath || !newpath || !*newpath) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    if (renameat((int)olddirfd, oldpath, (int)newdirfd, newpath) < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
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

    off_t r = lseek((int)fdv, (off_t)off, whence);
    if (r == (off_t)-1) {
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
    int r = isatty((int)fdv);
    *out_is_tty = (r != 0);

    if (r == 0 && errno != 0 && errno != ENOTTY) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}


static inline ArkStatus ark_set_cloexec_fd(int fd, ArkIoError* out_err) {
    int flags;
    do { flags = fcntl(fd, F_GETFD); } while (flags < 0 && errno == EINTR);
    if (flags < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    int r;
    do { r = fcntl(fd, F_SETFD, flags | FD_CLOEXEC); } while (r < 0 && errno == EINTR);
    if (r < 0) {
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

#if defined(__linux__)
    int r;
    do { r = pipe2(fds, O_CLOEXEC); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    *out_r = (int64_t)fds[0];
    *out_w = (int64_t)fds[1];
    return 0;
#else
    int r;
    do { r = pipe(fds); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    ArkStatus s = ark_set_cloexec_fd(fds[0], out_err);
    if (s != 0) { (void)close(fds[0]); (void)close(fds[1]); return s; }

    s = ark_set_cloexec_fd(fds[1], out_err);
    if (s != 0) { (void)close(fds[0]); (void)close(fds[1]); return s; }

    *out_r = (int64_t)fds[0];
    *out_w = (int64_t)fds[1];
    return 0;
#endif
}

ArkStatus __ark_fd_dup_cloexec(int64_t old_fd, int64_t* out_new_fd, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_new_fd) *out_new_fd = -1;

    if (!out_new_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

#if defined(__linux__)
    int fd;
    do { fd = fcntl((int)old_fd, F_DUPFD_CLOEXEC, 0); } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    *out_new_fd = (int64_t)fd;
    return 0;
#else
    int fd;
    do { fd = dup((int)old_fd); } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    ArkStatus s = ark_set_cloexec_fd(fd, out_err);
    if (s != 0) { (void)close(fd); return s; }

    *out_new_fd = (int64_t)fd;
    return 0;
#endif
}

// dup3 (native if available; otherwise emulated)
// Semantics: if old_fd == new_fd => EINVAL.
ArkStatus __ark_fd_dup3(int64_t old_fd, int64_t new_fd, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if ((int)old_fd == (int)new_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

#if defined(__linux__)
    int r;
    do { r = dup3((int)old_fd, (int)new_fd, flags); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return 0;
#else
    if ((flags & ~(O_CLOEXEC)) != 0) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    int r;
    do { r = dup2((int)old_fd, (int)new_fd); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    if (flags & O_CLOEXEC) {
        ArkStatus s = ark_set_cloexec_fd((int)new_fd, out_err);
        if (s != 0) return s;
    }

    return 0;
#endif
}


static inline ArkStatus ark_get_fd_flags(int fd, int* out_flags, ArkIoError* out_err) {
    int r;
    do { r = fcntl(fd, F_GETFL); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    *out_flags = r;
    return 0;
}

static inline ArkStatus ark_set_fd_flags(int fd, int flags, ArkIoError* out_err) {
    int r;
    do { r = fcntl(fd, F_SETFL, flags); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// __ark_fd_dup3_best_effort
// Semantics:
//   - Attempts dup3(old,new,flags).
//   - If dup3 unsupported or flags unsupported, degrades to dup2 + optional CLOEXEC.
//   - Returns 0 if it successfully duplicates with best-effort flag handling.
//   - Returns errno if duplication fails.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_dup3_best_effort(int64_t old_fd, int64_t new_fd, int flags, ArkIoError* out_err) {
    ark_err_clear(out_err);

    if ((int)old_fd == (int)new_fd) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

#if defined(__linux__)
    int r;
    do { r = dup3((int)old_fd, (int)new_fd, flags); } while (r < 0 && errno == EINTR);
    if (r == 0) return 0;

    int e = errno;

    // Best-effort: if flags are rejected or syscall is missing, try fallback.
    // EINVAL commonly indicates unsupported flags; ENOSYS indicates missing dup3.
    if (e != EINVAL && e != ENOSYS) {
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }
#else
    // Non-Linux: always fallback.
    int e = EINVAL;
#endif

    // Fallback: dup2 + optional CLOEXEC only. Ignore unknown flags best-effort.
    int r2;
    do { r2 = dup2((int)old_fd, (int)new_fd); } while (r2 < 0 && errno == EINTR);
    if (r2 < 0) {
        int e2 = errno;
        ark_err_set_errno(out_err, e2);
        return (ArkStatus)e2;
    }

    if (flags & O_CLOEXEC) {
        int fd_flags;
        ArkStatus s = 0;

        // Set FD_CLOEXEC using F_GETFD/F_SETFD (not F_SETFL).
        int f;
        do { f = fcntl((int)new_fd, F_GETFD); } while (f < 0 && errno == EINTR);
        if (f < 0) {
            int e3 = errno;
            ark_err_set_errno(out_err, e3);
            return (ArkStatus)e3;
        }
        int sr;
        do { sr = fcntl((int)new_fd, F_SETFD, f | FD_CLOEXEC); } while (sr < 0 && errno == EINTR);
        if (sr < 0) {
            int e4 = errno;
            ark_err_set_errno(out_err, e4);
            return (ArkStatus)e4;
        }

        (void)fd_flags;
        (void)s;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// __ark_fd_ioctl
// Semantics:
//   - Pass-through ioctl(fd, req, arg).
//   - Returns errno on failure.
// Notes:
//   - Caller is responsible for providing correct arg pointer/type per request.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_ioctl(int64_t fdv, uint64_t req, void* arg, ArkIoError* out_err) {
    ark_err_clear(out_err);

    int r;
    do { r = ioctl((int)fdv, (unsigned long)req, arg); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// __ark_fd_set_nonblock
// Semantics:
//   - Toggles O_NONBLOCK via fcntl(F_GETFL/F_SETFL).
//   - Returns errno on failure.
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_set_nonblock(int64_t fdv, bool enabled, ArkIoError* out_err) {
    ark_err_clear(out_err);

    int flags = 0;
    ArkStatus s = ark_get_fd_flags((int)fdv, &flags, out_err);
    if (s != 0) return s;

    int want = enabled ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    if (want == flags) return 0;

    return ark_set_fd_flags((int)fdv, want, out_err);
}

// -----------------------------------------------------------------------------
// __ark_fd_poll1
// Semantics:
//   - Polls a single fd with requested `events` and timeout_ms.
//   - timeout_ms: <0 means infinite (poll semantics).
//   - On success: *out_revents set (may be 0), returns 0.
//   - On failure: returns errno.
// Notes:
//   - events and revents are `poll(2)` bitmasks (POLLIN, POLLOUT, ...).
// -----------------------------------------------------------------------------
ArkStatus __ark_fd_poll1(int64_t fdv, int events, int timeout_ms, int* out_revents, ArkIoError* out_err) {
    ark_err_clear(out_err);
    if (out_revents) *out_revents = 0;

    if (!out_revents) {
        ark_err_set_errno(out_err, EINVAL);
        return (ArkStatus)EINVAL;
    }

    struct pollfd pfd;
    pfd.fd = (int)fdv;
    pfd.events = (short)events;
    pfd.revents = 0;

    int r;
    do { r = poll(&pfd, 1, timeout_ms); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        int e = errno;
        ark_err_set_errno(out_err, e);
        return (ArkStatus)e;
    }

    // r == 0: timeout, r == 1: events, both are success cases.
    *out_revents = (int)pfd.revents;
    return 0;
}


} // extern "C"