// tools/compiler/Runtime/gpu/gpu_backend.h
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
  #define ARK_GPU_EXPORT extern "C" __declspec(dllexport)
#else
  #define ARK_GPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ============================================================================
// Ark GPU Backend Plugin ABI
// - One backend plugin (.so/.dll/.dylib) exports a single symbol:
//     ark_gpu_backend_query_v1()
// - The returned vtable is immutable and valid for the lifetime of the process.
// - All structs are versioned + sized for forward/backward extension.
// ============================================================================

static constexpr std::uint32_t ARK_GPU_BACKEND_ABI_V1 = 0x0001'0001u;

enum ark_gpu_status : std::int32_t {
    ARK_GPU_OK                = 0,
    ARK_GPU_ERR_UNKNOWN       = -1,
    ARK_GPU_ERR_NOT_READY     = -2,
    ARK_GPU_ERR_UNSUPPORTED   = -3,
    ARK_GPU_ERR_INVALID_ARG   = -4,
    ARK_GPU_ERR_NO_DEVICE     = -5,
    ARK_GPU_ERR_OOM           = -6,
    ARK_GPU_ERR_DRIVER        = -7,
    ARK_GPU_ERR_COMPILATION   = -8,
    ARK_GPU_ERR_NOT_FOUND     = -9,
    ARK_GPU_ERR_INTERNAL      = -10
};

enum ark_gpu_backend_kind : std::uint8_t {
    ARK_GPU_BACKEND_NONE  = 0,
    ARK_GPU_BACKEND_CUDA  = 1,
    ARK_GPU_BACKEND_HIP   = 2,
    ARK_GPU_BACKEND_METAL = 3
};

enum ark_gpu_log_level : std::uint8_t {
    ARK_GPU_LOG_TRACE = 0,
    ARK_GPU_LOG_DEBUG = 1,
    ARK_GPU_LOG_INFO  = 2,
    ARK_GPU_LOG_WARN  = 3,
    ARK_GPU_LOG_ERROR = 4
};

using ark_gpu_device_ptr   = void*; // opaque device pointer token owned by backend
using ark_gpu_stream       = void*; // opaque stream handle owned by backend
using ark_gpu_event        = void*; // opaque event handle owned by backend
using ark_gpu_module       = void*; // opaque module/library handle owned by backend
using ark_gpu_kernel       = void*; // opaque kernel/function handle owned by backend

using ark_gpu_log_fn = void (*)(ark_gpu_log_level level, const char* msg, void* user);

// Backend-wide feature bits (stable, opt-in)
enum ark_gpu_feature_bits : std::uint64_t {
    ARK_GPU_FEAT_STREAMS             = 1ull << 0,
    ARK_GPU_FEAT_EVENTS              = 1ull << 1,
    ARK_GPU_FEAT_ASYNC_MEMCPY        = 1ull << 2,
    ARK_GPU_FEAT_ASYNC_MEMSET        = 1ull << 3,
    ARK_GPU_FEAT_D2D_MEMCPY          = 1ull << 4,
    ARK_GPU_FEAT_PINNED_HOST_ALLOC   = 1ull << 5,
    ARK_GPU_FEAT_MANAGED_ALLOC       = 1ull << 6,
    ARK_GPU_FEAT_SHARED_HOST_VISIBLE = 1ull << 7,  // e.g. Metal shared
    ARK_GPU_FEAT_MODULES             = 1ull << 8,
    ARK_GPU_FEAT_RUNTIME_COMPILE     = 1ull << 9,  // cuda/hip rtc, metal msl compile
    ARK_GPU_FEAT_KERNEL_HANDLES      = 1ull << 10, // lookup -> handle -> launch
    ARK_GPU_FEAT_LAUNCH_EX           = 1ull << 11, // shared/tg mem, flags, etc
    ARK_GPU_FEAT_DEVICE_QUERY        = 1ull << 12, // device properties
    ARK_GPU_FEAT_MULTI_DEVICE        = 1ull << 13, // set/get device, device count
    ARK_GPU_FEAT_TIMELINE_SEMAPHORE  = 1ull << 14  // future-facing sync primitive
};

static constexpr std::uint64_t ARK_GPU_FEATURE_KNOWN_MASK =
    (ARK_GPU_FEAT_STREAMS |
     ARK_GPU_FEAT_EVENTS |
     ARK_GPU_FEAT_ASYNC_MEMCPY |
     ARK_GPU_FEAT_ASYNC_MEMSET |
     ARK_GPU_FEAT_D2D_MEMCPY |
     ARK_GPU_FEAT_PINNED_HOST_ALLOC |
     ARK_GPU_FEAT_MANAGED_ALLOC |
     ARK_GPU_FEAT_SHARED_HOST_VISIBLE |
     ARK_GPU_FEAT_MODULES |
     ARK_GPU_FEAT_RUNTIME_COMPILE |
     ARK_GPU_FEAT_KERNEL_HANDLES |
     ARK_GPU_FEAT_LAUNCH_EX |
     ARK_GPU_FEAT_DEVICE_QUERY |
     ARK_GPU_FEAT_MULTI_DEVICE |
     ARK_GPU_FEAT_TIMELINE_SEMAPHORE);

enum ark_gpu_mem_kind : std::uint8_t {
    ARK_GPU_MEM_DEVICE        = 0,
    ARK_GPU_MEM_HOST_PINNED   = 1,
    ARK_GPU_MEM_HOST_MAPPED   = 2,
    ARK_GPU_MEM_MANAGED       = 3,
    ARK_GPU_MEM_SHARED        = 4
};

enum ark_gpu_memcpy_kind : std::uint8_t {
    ARK_GPU_COPY_H2D = 0,
    ARK_GPU_COPY_D2H = 1,
    ARK_GPU_COPY_D2D = 2
};

enum ark_gpu_stream_flags : std::uint32_t {
    ARK_GPU_STREAM_DEFAULT      = 0u,
    ARK_GPU_STREAM_NON_BLOCKING = 1u << 0
};

enum ark_gpu_event_flags : std::uint32_t {
    ARK_GPU_EVENT_DEFAULT        = 0u,
    ARK_GPU_EVENT_DISABLE_TIMING = 1u << 0
};

enum ark_gpu_module_kind : std::uint8_t {
    ARK_GPU_MODULE_UNKNOWN  = 0,

    ARK_GPU_MODULE_CUDA_PTX   = 1,
    ARK_GPU_MODULE_CUDA_CUBIN = 2,

    ARK_GPU_MODULE_HIP_HSACO  = 3,

    ARK_GPU_MODULE_METAL_LIB  = 4,

    ARK_GPU_MODULE_SRC_CUDA   = 5,
    ARK_GPU_MODULE_SRC_HIP    = 6,
    ARK_GPU_MODULE_SRC_MSL    = 7
};

struct ark_gpu_init_params {
    std::uint32_t size;          // must be sizeof(ark_gpu_init_params)
    std::uint32_t flags;         // reserved
    ark_gpu_log_fn logger;       // optional
    void*          logger_user;  // optional
};

struct ark_gpu_device_info {
    std::uint32_t size;              // sizeof(ark_gpu_device_info)
    std::uint32_t device_id;

    ark_gpu_backend_kind backend;
    char name[128];

    std::uint32_t major;
    std::uint32_t minor;

    std::uint64_t global_mem_bytes;
    std::uint32_t warp_size;
    std::uint32_t sm_count;
    std::uint32_t max_threads_per_sm;
    std::uint32_t max_threads_per_block;

    std::uint32_t max_grid_dim_x;
    std::uint32_t max_grid_dim_y;
    std::uint32_t max_grid_dim_z;

    std::uint32_t max_block_dim_x;
    std::uint32_t max_block_dim_y;
    std::uint32_t max_block_dim_z;

    std::uint32_t supports_async_copy; // 0/1
    std::uint32_t reserved_u32[15];
};

struct ark_gpu_stream_desc {
    std::uint32_t size;        // sizeof(ark_gpu_stream_desc)
    std::uint32_t flags;       // ark_gpu_stream_flags
    std::int32_t  priority;    // backend-defined range, 0 = default
    std::uint32_t reserved;
};

struct ark_gpu_event_desc {
    std::uint32_t size;        // sizeof(ark_gpu_event_desc)
    std::uint32_t flags;       // ark_gpu_event_flags
};

struct ark_gpu_alloc_desc {
    std::uint32_t size;        // sizeof(ark_gpu_alloc_desc)
    std::uint32_t flags;       // reserved
    ark_gpu_mem_kind kind;
    std::uint32_t alignment;   // 0 = backend default
};

struct ark_gpu_launch_desc {
    std::uint32_t size;              // sizeof(ark_gpu_launch_desc)
    std::uint32_t flags;             // reserved
    std::uint32_t shared_mem_bytes;  // cuda/hip dynamic shared
    std::uint32_t tg_mem_bytes;      // metal threadgroup memory size
    std::uint32_t tg_mem_index;      // metal threadgroup memory index
    std::uint32_t reserved_u32[11];
};

struct ark_gpu_module_desc {
    std::uint32_t size;            // sizeof(ark_gpu_module_desc)
    ark_gpu_module_kind kind;
    std::uint32_t flags;           // reserved
    const char*   module_key;      // stable key for caching
    const void*   bytes;           // binary blob or source bytes
    std::size_t   byte_len;
};

struct ark_gpu_rtc_desc {
    std::uint32_t size;            // sizeof(ark_gpu_rtc_desc)
    ark_gpu_module_kind kind;      // ARK_GPU_MODULE_SRC_*
    const char*   module_key;
    const char*   source_utf8;     // null-terminated
    const char*   program_name;    // optional
    const char**  options;         // optional
    std::int32_t  option_count;
    std::uint32_t flags;           // reserved
};

struct ark_gpu_backend_v1 {
    std::uint32_t abi;             // ARK_GPU_BACKEND_ABI_V1
    std::uint32_t size;            // sizeof(ark_gpu_backend_v1)

    ark_gpu_backend_kind kind;
    const char*          name;     // "cuda" | "hip" | "metal"
    std::uint64_t        features; // ark_gpu_feature_bits

    // -------------------- Lifecycle --------------------
    ark_gpu_status (*init)(std::int32_t device_id, const ark_gpu_init_params* params);
    void           (*shutdown)();

    void (*enter_thread)();
    void (*leave_thread)();

    // -------------------- Device Query --------------------
    ark_gpu_status (*device_count)(std::uint32_t* out_count);
    ark_gpu_status (*set_device)(std::int32_t device_id);
    ark_gpu_status (*get_device)(std::int32_t* out_device_id);
    ark_gpu_status (*get_device_info)(std::int32_t device_id, ark_gpu_device_info* out_info);

    // -------------------- Memory --------------------
    ark_gpu_device_ptr (*alloc)(std::int64_t bytes);
    void               (*free)(ark_gpu_device_ptr p);

    ark_gpu_status (*alloc_ex)(const ark_gpu_alloc_desc* desc, std::int64_t bytes, ark_gpu_device_ptr* out_p);
    ark_gpu_status (*host_alloc)(const ark_gpu_alloc_desc* desc, std::int64_t bytes, void** out_host);
    void           (*host_free)(void* host);

    void (*memcpy_to_device)(ark_gpu_device_ptr dst, const void* src, std::int64_t bytes);
    void (*memcpy_to_host)(void* dst, ark_gpu_device_ptr src, std::int64_t bytes);

    ark_gpu_status (*memcpy)(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind);
    ark_gpu_status (*memcpy_async)(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind, ark_gpu_stream stream);
    ark_gpu_status (*memset_async)(ark_gpu_device_ptr dst, std::int32_t value, std::int64_t bytes, ark_gpu_stream stream);

    // -------------------- Streams --------------------
    ark_gpu_status (*stream_create)(const ark_gpu_stream_desc* desc, ark_gpu_stream* out_stream);
    void           (*stream_destroy)(ark_gpu_stream stream);
    ark_gpu_status (*stream_synchronize)(ark_gpu_stream stream);

    // -------------------- Events --------------------
    ark_gpu_status (*event_create)(const ark_gpu_event_desc* desc, ark_gpu_event* out_evt);
    void           (*event_destroy)(ark_gpu_event evt);
    ark_gpu_status (*event_record)(ark_gpu_event evt, ark_gpu_stream stream);
    ark_gpu_status (*event_synchronize)(ark_gpu_event evt);
    ark_gpu_status (*event_elapsed_ms)(ark_gpu_event start_evt, ark_gpu_event end_evt, float* out_ms);
    ark_gpu_status (*stream_wait_event)(ark_gpu_stream stream, ark_gpu_event evt);

    // -------------------- Modules / RTC --------------------
    ark_gpu_status (*module_load)(const ark_gpu_module_desc* desc, ark_gpu_module* out_mod);
    void           (*module_unload)(ark_gpu_module mod);
    ark_gpu_status (*module_set_default)(ark_gpu_module mod);

    ark_gpu_status (*rtc_compile)(const ark_gpu_rtc_desc* desc, ark_gpu_module* out_mod);
    const char*    (*rtc_last_log)(const char* module_key);

    ark_gpu_status (*kernel_lookup)(ark_gpu_module mod, const char* kernel_name, ark_gpu_kernel* out_kernel);
    ark_gpu_status (*kernel_lookup_default)(const char* kernel_name, ark_gpu_kernel* out_kernel);

    // -------------------- Launch --------------------
    void (*launch)(const char* kernel_name,
                   void** args, std::int32_t arg_count,
                   std::int32_t gx, std::int32_t gy, std::int32_t gz,
                   std::int32_t bx, std::int32_t by, std::int32_t bz,
                   ark_gpu_stream stream);

    ark_gpu_status (*launch_handle)(ark_gpu_kernel kernel,
                                   void** args, std::int32_t arg_count,
                                   std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                   std::int32_t bx, std::int32_t by, std::int32_t bz,
                                   ark_gpu_stream stream);

    ark_gpu_status (*launch_ex)(const char* kernel_name,
                               void** args, std::int32_t arg_count,
                               std::int32_t gx, std::int32_t gy, std::int32_t gz,
                               std::int32_t bx, std::int32_t by, std::int32_t bz,
                               const ark_gpu_launch_desc* desc,
                               ark_gpu_stream stream);

    // -------------------- Global synchronize --------------------
    void (*synchronize)(ark_gpu_stream stream);

    // -------------------- Error / Debug --------------------
    const char* (*last_error_string)();

    void* reserved[32];
};

using ark_gpu_backend_query_v1_fn = const ark_gpu_backend_v1* (*)();

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1();

// ============================================================================
// Ark Runtime-facing strict surface (kept C++ friendly)
// ============================================================================

namespace ark {
namespace gpu {

using DevicePtr    = ark_gpu_device_ptr;
using StreamHandle = ark_gpu_stream;

void init(std::int32_t device_id);

DevicePtr alloc(std::int64_t bytes);
DevicePtr alloc_managed(std::int64_t bytes); // [NEW] Added managed allocator
void      free(DevicePtr p);

void memcpy_to_device(DevicePtr dst, const void* src, std::int64_t bytes);
void memcpy_to_host(void* dst, DevicePtr src, std::int64_t bytes);

void launch(const char* kernel_name,
            void** args,
            std::int32_t arg_count,
            std::int32_t gx, std::int32_t gy, std::int32_t gz,
            std::int32_t bx, std::int32_t by, std::int32_t bz,
            StreamHandle stream);

void synchronize(StreamHandle stream);

} // namespace gpu
} // namespace ark

// ============================================================================
// C++-only ABI helpers (NOT part of the exported C ABI)
// ============================================================================

namespace ark::gpu::abi {

static inline bool validate_abi_verbose(const ark_gpu_backend_v1* v, const char** out_reason) {
    static const char* OK = "OK";
    if (out_reason) *out_reason = OK;

    if (!v) { if (out_reason) *out_reason = "VTable is NULL"; return false; }
    if (v->abi != ARK_GPU_BACKEND_ABI_V1) { if (out_reason) *out_reason = "ABI Version Mismatch"; return false; }

    constexpr std::size_t kMinSize =
        offsetof(ark_gpu_backend_v1, synchronize) + sizeof(((ark_gpu_backend_v1*)nullptr)->synchronize);

    if (v->size < kMinSize) { if (out_reason) *out_reason = "VTable size too small for V1"; return false; }

    if (!v->name || !v->name[0]) { if (out_reason) *out_reason = "Missing name"; return false; }

    if (!v->init) { if (out_reason) *out_reason = "Missing init"; return false; }
    if (!v->shutdown) { if (out_reason) *out_reason = "Missing shutdown"; return false; }
    if (!v->alloc) { if (out_reason) *out_reason = "Missing alloc"; return false; }
    if (!v->free) { if (out_reason) *out_reason = "Missing free"; return false; }
    if (!v->memcpy_to_device) { if (out_reason) *out_reason = "Missing memcpy_to_device"; return false; }
    if (!v->memcpy_to_host) { if (out_reason) *out_reason = "Missing memcpy_to_host"; return false; }
    if (!v->launch) { if (out_reason) *out_reason = "Missing launch"; return false; }
    if (!v->synchronize) { if (out_reason) *out_reason = "Missing synchronize"; return false; }

    if (v->kind != ARK_GPU_BACKEND_CUDA &&
        v->kind != ARK_GPU_BACKEND_HIP &&
        v->kind != ARK_GPU_BACKEND_METAL) {
        if (out_reason) *out_reason = "Invalid backend kind";
        return false;
    }

    if ((v->features & ~ARK_GPU_FEATURE_KNOWN_MASK) != 0) {
        if (out_reason) *out_reason = "Unknown feature bits set";
        return false;
    }

    if ((v->features & ARK_GPU_FEAT_MULTI_DEVICE) != 0) {
        if (!v->device_count || !v->set_device || !v->get_device) {
            if (out_reason) *out_reason = "Claimed FEAT_MULTI_DEVICE but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_DEVICE_QUERY) != 0) {
        if (!v->get_device_info) {
            if (out_reason) *out_reason = "Claimed FEAT_DEVICE_QUERY but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_STREAMS) != 0) {
        if (!v->stream_create || !v->stream_destroy) {
            if (out_reason) *out_reason = "Claimed FEAT_STREAMS but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_EVENTS) != 0) {
        if (!v->event_create || !v->event_destroy || !v->event_record || !v->event_synchronize) {
            if (out_reason) *out_reason = "Claimed FEAT_EVENTS but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_ASYNC_MEMCPY) != 0) {
        if (!v->memcpy_async) {
            if (out_reason) *out_reason = "Claimed FEAT_ASYNC_MEMCPY but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_ASYNC_MEMSET) != 0) {
        if (!v->memset_async) {
            if (out_reason) *out_reason = "Claimed FEAT_ASYNC_MEMSET but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_MODULES) != 0) {
        if (!v->module_load || !v->module_unload) {
            if (out_reason) *out_reason = "Claimed FEAT_MODULES but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_RUNTIME_COMPILE) != 0) {
        if (!v->rtc_compile) {
            if (out_reason) *out_reason = "Claimed FEAT_RUNTIME_COMPILE but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_KERNEL_HANDLES) != 0) {
        const bool has_lookup = (v->kernel_lookup != nullptr) || (v->kernel_lookup_default != nullptr);
        if (!has_lookup || !v->launch_handle) {
            if (out_reason) *out_reason = "Claimed FEAT_KERNEL_HANDLES but missing implementation";
            return false;
        }
    }

    if ((v->features & ARK_GPU_FEAT_LAUNCH_EX) != 0) {
        if (!v->launch_ex) {
            if (out_reason) *out_reason = "Claimed FEAT_LAUNCH_EX but missing implementation";
            return false;
        }
    }

    return true;
}

static inline bool validate_abi(const ark_gpu_backend_v1* v) {
    return validate_abi_verbose(v, nullptr);
}

} // namespace ark::gpu::abi
