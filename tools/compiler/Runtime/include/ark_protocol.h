#pragma once

// -----------------------------------------------------------------------------
// System Includes (Language Agnostic)
// -----------------------------------------------------------------------------
#if defined(__cplusplus)
    #include <cstdint>
    #include <cstddef>
    #include <cstdbool>
    #include <cerrno>
#else
    #include <stdint.h>
    #include <stddef.h>
    #include <stdbool.h>
    #include <errno.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------
// Protocol Constants (Machine Checkable)
// -----------------------------------------------------------------------------
#define ARK_PROTOCOL_VERSION 1

// Fixed byte sizes for the binary layout
#define ARK_META_HEADER_BYTES 32
#define ARK_META_FIELD_BYTES  16

// Byte Offsets (Little Endian)
#define ARK_META_OFF_VERSION   0
#define ARK_META_OFF_ARGCOUNT  4
#define ARK_META_OFF_CHECKSUM  8   // 16 bytes (u128)
#define ARK_META_OFF_FLAGS     24
#define ARK_META_OFF_RESERVED  28

// -----------------------------------------------------------------------------
// Argument Flags
// -----------------------------------------------------------------------------
#define ARK_KIND_SCALAR  0
#define ARK_KIND_BUFFER  1

#define ARK_ACCESS_READ  1
#define ARK_ACCESS_WRITE 2
#define ARK_ACCESS_RW    3

#define ARK_LEN_IMPLICIT_WHOLE 0
#define ARK_LEN_SEPARATE_ARG   1

// -----------------------------------------------------------------------------
// Network / Auth Protocol (Ticket System)
// -----------------------------------------------------------------------------
// Magic: "ARK" + 0x01 (Version)
constexpr uint32_t ARK_MAGIC = 0x41524B01; 
constexpr size_t   SIG_SIZE  = 64; // Ed25519 signature length

// Fixed-layout binary ticket. Packed, Little-Endian.
#pragma pack(push, 1)
struct TicketV1 {
    
    uint8_t  version;          // 1 byte (Always 1)
    uint8_t  ticket_id[16];    // 16 bytes (UUID)
    uint8_t  org_id[16];       // 16 bytes (UUID)
    uint8_t  provider_id[16];  // 16 bytes (UUID)
    // Wire Format: Little Endian
    // We keep them as uint64_t for alignment, but MUST use helpers to write/read.
    uint64_t issued_at_ms;     // 8 bytes (Unix millis)
    uint64_t expires_at_ms;    // 8 bytes (TTL enforcement)
    
    uint64_t max_burn_cents;   // 8 bytes (Budget cap)
    uint8_t  flags;            // 1 byte  (0x01 = Priority)
};
#pragma pack(pop)

// Validation: 1 + (16*3) + (8*3) + 1 = 74 bytes
static_assert(sizeof(TicketV1) == 74, "TicketV1 layout mismatch! Check packing.");

// The Bundle (Wire Format)
// This is what travels HTTP -> Runtime -> TCP -> Provider
struct AuthBundleV1 {
    TicketV1 ticket;
    uint8_t  signature[SIG_SIZE]; 
};

static_assert(sizeof(AuthBundleV1) == 74 + 64, "AuthBundleV1 size mismatch");

void __ark_register_kernel(
    const char* name,
    const void* meta_bytes, uint64_t meta_len,
    const void* artifact,   uint64_t artifact_len
);

void* __ark_alloc(uint64_t bytes, uint64_t align);

bool ark_resolve_ptr(void* ptr, uint64_t* out_id, uint64_t* out_off, uint64_t* out_len);

int64_t __ark_launch(
    void* grid, 
    void* kernel_stub, 
    uint64_t uid_low, uint64_t uid_high,
    void* args, 
    uint64_t args_size, 
    uint64_t grid_dim, 
    void* config
);

// =========================================================
// 1. Core Constants & Enums (Required for Linking)
// =========================================================

// [FIX] Restored Enum required by async.cpp and remote.cpp
typedef enum ArkTaskStatus {
    ARK_TASK_OK          = 0,
    ARK_TASK_RUNNING     = 1,
    ARK_TASK_ERR_GENERIC = -1,
    ARK_TASK_ERR_TIMEOUT = -2,
    ARK_TASK_ERR_CANCEL  = -3,
    ARK_TASK_ERR_REMOTE  = -4,
    ARK_TASK_ERR_INVALID = -5
} ArkTaskStatus;

// [FIX] Constants
#define ARK_NET_MAX_PACKET (64LL * 1024 * 1024)

#ifndef ARK_AT_FDCWD
#define ARK_AT_FDCWD (-100)
#endif

// =========================================================
// 2. Type Definitions (ABI Stable)
// =========================================================

typedef int32_t ArkStatus; 

// [FIX] Typedefs required by async.cpp
typedef int64_t ArkTaskHandle;
typedef int64_t ArkPresetHandle;

// Function Pointers
typedef void (*ArkKernelFn)(void* grid, void* packed_args);
typedef void (*ArkForBodyFn)(int64_t i, void* ctx);

typedef struct ArkStr {
    char* ptr;
    int64_t len;
} ArkStr;

typedef struct ArkBytes {
    uint8_t* ptr;
    int64_t  len;
    int64_t  cap;
} ArkBytes;

typedef struct ark_vec_t {
    void* ptr;
    int64_t len;
    int64_t cap;
} ark_vec_t;

typedef struct ArkIoError {
    int32_t code;
    int32_t _pad;
    ArkStr  msg;
} ArkIoError;

// =========================================================
// 3. Async Runtime API
// =========================================================


int __ark_await(ArkTaskHandle handle);
// [FIX] Added _ex variant used in async.cpp
int  __ark_await_ex(ArkTaskHandle handle, int32_t timeout_ms); 

void __ark_detach(ArkTaskHandle handle);
void __ark_cancel(ArkTaskHandle handle);
int  __ark_task_status(ArkTaskHandle handle);
void __ark_sync_all(void);

// Parallel For
void __ark_parallel_for_1d(int64_t n, ArkForBodyFn body, void* ctx);

// Structured Concurrency
void __ark_scope_enter(void);
int  __ark_scope_exit(void);

// Lifetime
void __ark_shutdown(void);

// =========================================================
// 4. Configuration / Presets (Required by async.cpp)
// =========================================================

// [FIX] Added Preset API declarations
ArkPresetHandle __ark_runtime_preset_current(void);
void __ark_runtime_preset_release(ArkPresetHandle h);
const char* __ark_runtime_preset_target(ArkPresetHandle h);
const char* __ark_runtime_preset_endpoint(ArkPresetHandle h);
const char* __ark_runtime_preset_token(ArkPresetHandle h);
int32_t __ark_runtime_preset_timeout_ms(ArkPresetHandle h);
void __ark_runtime_preset_pop_all(void);
bool __ark_runtime_target_is_remote(const char* target);

// =========================================================
// 5. Remote Backend ABI (Transport Layer)
// =========================================================

struct ArkRemoteStats {
    uint64_t sent;
    uint64_t recv;
    uint64_t io_errs;
    uint64_t bad_packets;
};

// Opaque surface for remote backend to hold/release core tasks
void __ark_task_retain(void* task_ptr);
void __ark_task_release(void* task_ptr);
void __ark_task_complete_consume(void* task_ptr, uint64_t cookie, int32_t status);

// Return:
//  0: Accepted async. Backend OWNS transfer ref.
//  1: Completed inline. Backend OWNS transfer ref (and consumed it).
// <0: Failed. Backend REJECTED ownership.
int  __ark_remote_submit(
    void* task_ptr,
    std::uint64_t uid_lo,   // [UPDATED] Protocol V1
    std::uint64_t uid_hi,   // [UPDATED] Protocol V1
    std::uint64_t cookie,
    const char* target,
    const char* endpoint,
    const char* token
);

void __ark_remote_cancel(void* task_ptr);

// Control-plane
void __ark_remote_disconnect(void);
int  __ark_remote_get_stats(struct ArkRemoteStats* out_stats);
void __ark_remote_reset_stats(void);

// =========================================================
// 6. Core: Panic Handling
// =========================================================

void arkPanicAt(const char* msg, const char* file, int32_t line, int32_t col);
void arkPanicBounds(const char* file, int32_t line, int32_t col, int64_t idx, int64_t len);
void panic(const char* msg);
bool __ark_str_eq(ArkStr a, ArkStr b);

#ifdef __cplusplus
#define ARK_PANIC(msg) arkPanicAt((msg), __FILE__, __LINE__, 0)
#endif

// =========================================================
// 7. Core: Memory Management
// =========================================================

void* ark_alloc(int64_t size);
void* ark_realloc(void* ptr, int64_t new_size);
void  ark_free(void* ptr);

void* __ark_realloc(void* ptr, int64_t new_size);
void  __ark_free(void* ptr);

void  __ark_free_internal(void* p);
void __ark_drop_opaque(void* ptr);

// =========================================================
// 8. Core: Vector Operations
// =========================================================

ArkStatus __ark_vec_validate(const ark_vec_t* v);
ArkStatus __ark_vec_create(ark_vec_t** out);
ArkStatus __ark_vec_clone(const ark_vec_t* src, int64_t elem_size, ark_vec_t** out);

ArkStatus __ark_vec_reserve_at(ark_vec_t* v, int64_t min_cap, int64_t elem_size);
ArkStatus __ark_vec_shrink_to_fit(ark_vec_t* v, int64_t elem_size);
ArkStatus __ark_vec_clear(ark_vec_t* v);

ArkStatus __ark_vec_at(ark_vec_t* v, int64_t idx, int64_t elem_size, void** out_ptr);
ArkStatus __ark_vec_cat(const ark_vec_t* v, int64_t idx, int64_t elem_size, const void** out_ptr);

ArkStatus __ark_vec_push_uninit(ark_vec_t* v, int64_t elem_size, void** out_slot);
ArkStatus __ark_vec_push_copy(ark_vec_t* v, const void* elem, int64_t elem_size);
ArkStatus __ark_vec_pop_copy(ark_vec_t* v, void* out_elem, int64_t elem_size);

ArkStatus __ark_vec_insert_uninit(ark_vec_t* v, int64_t idx, int64_t elem_size, void** out_slot);
ArkStatus __ark_vec_insert_copy(ark_vec_t* v, int64_t idx, const void* elem, int64_t elem_size);

ArkStatus __ark_vec_remove_shift_copy(ark_vec_t* v, int64_t idx, void* out_elem, int64_t elem_size);
ArkStatus __ark_vec_swap_remove_copy(ark_vec_t* v, int64_t idx, void* out_elem, int64_t elem_size);

ArkStatus __ark_vec_extend_from_raw(ark_vec_t* v, const void* data, int64_t count, int64_t elem_size);

// =========================================================
// 9. Core: Printing
// =========================================================

void __ark_print_str(const char* ptr, int64_t len);

void printStr(const char* s);
void printSpace(void);
void printNewline(void);
void printI32(int32_t i);
void printI64(int64_t i);
void printBool(uint8_t b);
void printF32(float f);
void printF64(double d);
void printNone(void);

void ark_vec_print_i32(ark_vec_t* v);
void ark_vec_print_i64(ark_vec_t* v);
void ark_vec_print_bool(ark_vec_t* v);
void ark_vec_print_f32(ark_vec_t* v);
void ark_vec_print_f32_raw(ark_vec_t* v);
void ark_vec_print_f64(ark_vec_t* v);
void ark_vec_print_f64_raw(ark_vec_t* v);
void ark_vec_print_str(ark_vec_t* v);

// =========================================================
// 10. Filesystem
// =========================================================

ArkStatus __ark_file_write_atomic_strict_posix(const char* path, const void* content, int64_t len, ArkIoError* out_err);
ArkStatus __ark_file_write_atomic_strict_db(const char* path, const void* content, int64_t len, ArkIoError* out_err);
ArkStatus __ark_file_write_atomic_best_effort(const char* path, const void* content, int64_t len, ArkIoError* out_err);

ArkStatus __ark_file_write_atomic(const char* path, const void* content, int64_t len, ArkIoError* out_err);
ArkStatus __ark_file_append(const char* path, const void* content, int64_t len, ArkIoError* out_err);

ArkStatus __ark_file_read_all_strict(const char* path, int64_t max_bytes, ArkStr* out, ArkIoError* out_err);
ArkStatus __ark_file_read_all_upto(const char* path, int64_t max_bytes, ArkStr* out, ArkIoError* out_err);
ArkStatus __ark_file_read_all(const char* path, ArkStr* out, ArkIoError* out_err);

ArkStatus __ark_file_exists(const char* path, bool* out_exists, ArkIoError* out_err);

// =========================================================
// 11. FD Kernel
// =========================================================

ArkStatus __ark_fd_open(const char* path, int flags, int mode, int64_t* out_fd, ArkIoError* out_err);
ArkStatus __ark_fd_openat(int64_t dirfd, const char* path, int flags, int mode, int64_t* out_fd, ArkIoError* out_err);

ArkStatus __ark_fd_mkdirat(int64_t dirfd, const char* path, int mode, ArkIoError* out_err);
ArkStatus __ark_fd_unlinkat(int64_t dirfd, const char* path, int flags, ArkIoError* out_err);
ArkStatus __ark_fd_renameat(int64_t olddirfd, const char* oldpath, int64_t newdirfd, const char* newpath, ArkIoError* out_err);

ArkStatus __ark_fd_fstatat(int64_t dirfd, const char* path, void* out_stat, int flags, ArkIoError* out_err);
ArkStatus __ark_fd_readlinkat(int64_t dirfd, const char* path, ArkStr* out, ArkIoError* out_err);

ArkStatus __ark_fd_close_best_effort(int64_t fdv, ArkIoError* out_err);
ArkStatus __ark_fd_close_unsafe(int64_t fdv, ArkIoError* out_err);
ArkStatus __ark_fd_close_strict(int64_t fdv, ArkIoError* out_err);
ArkStatus __ark_fd_close(int64_t fdv, ArkIoError* out_err);

ArkStatus __ark_fd_read(int64_t fdv, void* buf, int64_t cap, int64_t* out_n, ArkIoError* out_err);
ArkStatus __ark_fd_write(int64_t fdv, const void* buf, int64_t len, int64_t* out_n, ArkIoError* out_err);

ArkStatus __ark_fd_pread(int64_t fdv, void* buf, int64_t cap, int64_t off, int64_t* out_n, ArkIoError* out_err);
ArkStatus __ark_fd_pwrite(int64_t fdv, const void* buf, int64_t len, int64_t off, int64_t* out_n, ArkIoError* out_err);

ArkStatus __ark_fd_seek(int64_t fdv, int64_t off, int whence, int64_t* out_off, ArkIoError* out_err);

ArkStatus __ark_fd_pipe(int64_t* out_r, int64_t* out_w, ArkIoError* out_err);
ArkStatus __ark_fd_pipe_cloexec(int64_t* out_r, int64_t* out_w, ArkIoError* out_err);

ArkStatus __ark_fd_dup_cloexec(int64_t old_fd, int64_t* out_new_fd, ArkIoError* out_err);

ArkStatus __ark_fd_dup3(int64_t old_fd, int64_t new_fd, int flags, ArkIoError* out_err);
ArkStatus __ark_fd_dup3_best_effort(int64_t old_fd, int64_t new_fd, int flags, ArkIoError* out_err);

ArkStatus __ark_fd_ioctl(int64_t fdv, uint64_t req, void* arg, ArkIoError* out_err);
ArkStatus __ark_fd_set_nonblock(int64_t fdv, bool enabled, ArkIoError* out_err);
ArkStatus __ark_fd_poll1(int64_t fdv, int events, int timeout_ms, int* out_revents, ArkIoError* out_err);
ArkStatus __ark_fd_isatty(int64_t fdv, bool* out_is_tty, ArkIoError* out_err);

ArkStatus __ark_fd_write_all(int64_t fdv, const void* buf, int64_t len, int64_t* out_n, ArkIoError* out_err);
ArkStatus __ark_fd_pwrite_all(int64_t fdv, const void* buf, int64_t len, int64_t off, int64_t* out_n, ArkIoError* out_err);
ArkStatus __ark_fd_read_exact(int64_t fdv, void* buf, int64_t cap, int64_t* out_n, ArkIoError* out_err);

// =========================================================
// 12. Networking
// =========================================================

ArkStatus __ark_net_connect(const char* host, int32_t port, int64_t timeout_ms, int32_t* out_sockfd, ArkIoError* out_err);
ArkStatus __ark_net_listen(const char* host, int32_t port, int32_t* out_sockfd, ArkIoError* out_err);
ArkStatus __ark_net_accept(int32_t sockfd, int32_t* out_clientfd, ArkIoError* out_err);

ArkStatus __ark_net_send(int32_t sockfd, const void* data, int64_t len, ArkIoError* out_err);
ArkStatus __ark_net_recv(int32_t sockfd, int64_t max_len, ArkBytes* out, ArkIoError* out_err);

ArkStatus __ark_net_close(int32_t sockfd, ArkIoError* out_err);

// =========================================================
// 13. String Operations
// =========================================================

ArkStr __ark_str_alloc(int64_t size);
ArkStr __ark_str_from_raw(const char* s);

ArkStr  __ark_str_concat(ArkStr s1, ArkStr s2);
int64_t __ark_str_len(ArkStr s);
bool    __ark_str_eq(ArkStr s1, ArkStr s2);

bool __ark_str_contains(ArkStr haystack, ArkStr needle);
bool __ark_str_startswith(ArkStr str, ArkStr prefix);
bool __ark_str_endswith(ArkStr str, ArkStr suffix);

ArkStr __ark_str_trim(ArkStr str);
ArkStr __ark_str_to_lower(ArkStr str);
ArkStr __ark_str_to_upper(ArkStr str);
ArkStr __ark_str_reverse(ArkStr str);
ArkStr __ark_str_repeat(ArkStr str, int64_t count);

ArkStr __ark_str_slice(ArkStr str, int64_t start, int64_t end);
ArkStr __ark_str_replace(ArkStr str, ArkStr target, ArkStr replacement);

int64_t __ark_str_index_of(ArkStr str, ArkStr target);
int64_t __ark_str_last_index_of(ArkStr str, ArkStr target);

ArkStatus __ark_str_regex_match(ArkStr str, ArkStr pattern, bool* out_match);
ArkStatus __ark_str_regex_replace(ArkStr str, ArkStr pattern, ArkStr replacement, ArkStr* out_res);

void __ark_drop_vec_opaque(ark_vec_t* vec);
void __ark_drop_vec_i32(ark_vec_t* vec);
void __ark_drop_vec_i64(ark_vec_t* vec);
void __ark_drop_vec_f32(ark_vec_t* vec);
void __ark_drop_vec_f64(ark_vec_t* vec);
void __ark_drop_vec_str(ark_vec_t* vec);

#ifdef __cplusplus
} // extern "C"
#endif