// tools/compiler/Runtime/gpu/hip_backend.cpp
#include "gpu_backend.h"

#ifdef ARK_BACKEND_HIP

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// ------------------------------
// Error handling
// ------------------------------

static inline const char* ark_hip_err_name(hipError_t e) { return hipGetErrorName(e); }
static inline const char* ark_hip_err_str(hipError_t e)  { return hipGetErrorString(e); }

static inline void ark_hip_abort(const char* what, const char* file, int line) {
    std::fprintf(stderr, "[ARK HIP] %s at %s:%d\n", what, file, line);
    std::abort();
}

#define HIP_CHECK(call)                                                                \
    do {                                                                               \
        hipError_t _e = (call);                                                        \
        if (_e != hipSuccess) {                                                        \
            std::fprintf(stderr, "[ARK HIP] %s failed: %s (%s) at %s:%d\n",             \
                         #call, ark_hip_err_name(_e), ark_hip_err_str(_e),             \
                         __FILE__, __LINE__);                                          \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

static inline const char* ark_hiprtc_err_str(hiprtcResult r) {
    switch (r) {
        case HIPRTC_SUCCESS: return "HIPRTC_SUCCESS";
        case HIPRTC_ERROR_OUT_OF_MEMORY: return "HIPRTC_ERROR_OUT_OF_MEMORY";
        case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE: return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
        case HIPRTC_ERROR_INVALID_INPUT: return "HIPRTC_ERROR_INVALID_INPUT";
        case HIPRTC_ERROR_INVALID_PROGRAM: return "HIPRTC_ERROR_INVALID_PROGRAM";
        case HIPRTC_ERROR_INVALID_OPTION: return "HIPRTC_ERROR_INVALID_OPTION";
        case HIPRTC_ERROR_COMPILATION: return "HIPRTC_ERROR_COMPILATION";
        case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE: return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
        case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
        case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
        case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID: return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
        case HIPRTC_ERROR_INTERNAL_ERROR: return "HIPRTC_ERROR_INTERNAL_ERROR";
        default: return "HIPRTC_ERROR_UNKNOWN";
    }
}

#define HIPRTC_CHECK(call)                                                             \
    do {                                                                               \
        hiprtcResult _r = (call);                                                      \
        if (_r != HIPRTC_SUCCESS) {                                                    \
            std::fprintf(stderr, "[ARK HIP] %s failed: %s at %s:%d\n",                  \
                         #call, ark_hiprtc_err_str(_r), __FILE__, __LINE__);           \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

// ------------------------------
// Helpers
// ------------------------------

static inline bool ark_sv_starts_with(std::string_view s, std::string_view pfx) {
    return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

static inline std::pair<std::string_view, std::string_view> ark_sv_split_once(std::string_view s, std::string_view delim) {
    const std::size_t pos = s.find(delim);
    if (pos == std::string_view::npos) return {s, std::string_view{}};
    return {s.substr(0, pos), s.substr(pos + delim.size())};
}

static inline bool ark_parse_hex_u64(std::string_view s, std::uint64_t& out) {
    if (s.empty()) return false;
    if (ark_sv_starts_with(s, "0x") || ark_sv_starts_with(s, "0X")) s = s.substr(2);
    if (s.empty()) return false;

    std::uint64_t v = 0;
    for (char c : s) {
        unsigned digit = 0;
        if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') digit = 10u + static_cast<unsigned>(c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10u + static_cast<unsigned>(c - 'A');
        else return false;

        const std::uint64_t nv = (v << 4) | digit;
        if (nv < v) return false;
        v = nv;
    }
    out = v;
    return true;
}

static inline hipStream_t ark_to_hip_stream(ark_gpu_stream s) {
    return reinterpret_cast<hipStream_t>(s);
}
static inline ark_gpu_stream ark_from_hip_stream(hipStream_t s) {
    return reinterpret_cast<ark_gpu_stream>(s);
}

static inline hipEvent_t ark_to_hip_event(ark_gpu_event e) {
    return reinterpret_cast<hipEvent_t>(e);
}
static inline ark_gpu_event ark_from_hip_event(hipEvent_t e) {
    return reinterpret_cast<ark_gpu_event>(e);
}

// ------------------------------
// Backend state
// ------------------------------

namespace {

struct RtcLog {
    std::string log;
};

struct ModuleEntry {
    hipModule_t mod = nullptr;
    std::vector<std::uint8_t> image;
    std::string key;
    std::atomic<std::uint32_t> refcnt{1};
    RtcLog rtc;
};

struct KernelKey {
    std::string mod;
    std::string name;
    bool operator==(const KernelKey& o) const noexcept { return mod == o.mod && name == o.name; }
};

struct KernelKeyHash {
    std::size_t operator()(const KernelKey& k) const noexcept {
        const std::hash<std::string> h;
        std::size_t a = h(k.mod);
        std::size_t b = h(k.name);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }
};

struct HipState {
    std::mutex mu;
    bool initialized = false;
    int32_t device_id = -1;

    std::string gcn_arch;
    bool supports_managed = false;
    bool supports_pinned  = false;

    ark_gpu_log_fn logger = nullptr;
    void* logger_user = nullptr;

    std::mutex mu_modules;
    std::unordered_map<std::string, ModuleEntry> modules;
    std::string default_module_key;
    std::unordered_map<KernelKey, hipFunction_t, KernelKeyHash> kernels;

    const char* last_error = nullptr;
};

static HipState& S() {
    static HipState s;
    return s;
}

static inline void log_msg(ark_gpu_log_level lvl, const char* msg) {
    HipState& s = S();
    if (s.logger) s.logger(lvl, msg, s.logger_user);
    else std::fprintf(stderr, "[ARK HIP] %s\n", msg);
}

static inline void set_last_error(const char* msg) {
    S().last_error = msg;
}

static inline void ensure_init_locked(HipState& s) {
    if (!s.initialized) ark_hip_abort("backend not initialized", __FILE__, __LINE__);
    HIP_CHECK(hipSetDevice(s.device_id));
}

static hipFunction_t resolve_function_locked(HipState& s, const std::string& module_key, const std::string& kernel) {
    KernelKey kk{module_key, kernel};
    auto kit = s.kernels.find(kk);
    if (kit != s.kernels.end()) return kit->second;

    hipFunction_t fn = nullptr;

    if (!module_key.empty()) {
        auto it = s.modules.find(module_key);
        if (it == s.modules.end()) return nullptr;
        if (hipModuleGetFunction(&fn, it->second.mod, kernel.c_str()) != hipSuccess) return nullptr;
        s.kernels.emplace(std::move(kk), fn);
        return fn;
    }

    if (!s.default_module_key.empty()) {
        auto it = s.modules.find(s.default_module_key);
        if (it != s.modules.end()) {
            if (hipModuleGetFunction(&fn, it->second.mod, kernel.c_str()) == hipSuccess) {
                s.kernels.emplace(std::move(kk), fn);
                return fn;
            }
        }
    }

    for (auto& kv : s.modules) {
        if (hipModuleGetFunction(&fn, kv.second.mod, kernel.c_str()) == hipSuccess) {
            s.kernels.emplace(std::move(kk), fn);
            return fn;
        }
    }

    return nullptr;
}

static void invalidate_kernel_cache_for_module_locked(HipState& s, const std::string& module_key) {
    for (auto it = s.kernels.begin(); it != s.kernels.end();) {
        if (it->first.mod == module_key || it->first.mod.empty()) it = s.kernels.erase(it);
        else ++it;
    }
}

static void shutdown_locked(HipState& s) {
    if (!s.initialized) {
        s.device_id = -1;
        s.gcn_arch.clear();
        s.last_error = nullptr;
        return;
    }

    {
        std::lock_guard<std::mutex> lk(s.mu_modules);
        s.kernels.clear();
        for (auto& kv : s.modules) {
            if (kv.second.mod) HIP_CHECK(hipModuleUnload(kv.second.mod));
            kv.second.mod = nullptr;
        }
        s.modules.clear();
        s.default_module_key.clear();
    }

    s.device_id = -1;
    s.initialized = false;
    s.gcn_arch.clear();
    s.supports_managed = false;
    s.supports_pinned = false;
    s.last_error = nullptr;
}

// ------------------------------
// VTable implementations
// ------------------------------

static ark_gpu_status hip_init_v1(std::int32_t device_id, const ark_gpu_init_params* params) {
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);

    if (params) {
        if (params->size != sizeof(ark_gpu_init_params)) {
            set_last_error("init: bad params size");
            return ARK_GPU_ERR_INVALID_ARG;
        }
        s.logger = params->logger;
        s.logger_user = params->logger_user;
    } else {
        s.logger = nullptr;
        s.logger_user = nullptr;
    }

    if (s.initialized) {
        if (device_id == s.device_id) {
            HIP_CHECK(hipSetDevice(s.device_id));
            return ARK_GPU_OK;
        }
        shutdown_locked(s);
    }

    int cnt = 0;
    hipError_t ec = hipGetDeviceCount(&cnt);
    if (ec != hipSuccess || cnt <= 0) {
        set_last_error("no HIP devices available");
        return ARK_GPU_ERR_NO_DEVICE;
    }

    if (device_id < 0) device_id = 0;
    if (device_id >= cnt) device_id = 0;

    hipError_t sd = hipSetDevice(device_id);
    if (sd != hipSuccess) {
        set_last_error("hipSetDevice failed");
        return ARK_GPU_ERR_DRIVER;
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    s.gcn_arch = prop.gcnArchName ? prop.gcnArchName : "";

    // Feature probes (best-effort; safe defaults if attrs unsupported)
    int managed = 0;
#if defined(hipDeviceAttributeManagedMemory)
    hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, device_id);
#endif
    s.supports_managed = (managed != 0);
    s.supports_pinned  = true;

    s.device_id = device_id;
    s.initialized = true;

    log_msg(ARK_GPU_LOG_INFO, "backend initialized");
    return ARK_GPU_OK;
}

static void hip_shutdown_v1() {
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    shutdown_locked(s);
}

static void hip_enter_thread_v1() {
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.initialized) return;
    HIP_CHECK(hipSetDevice(s.device_id));
}

static void hip_leave_thread_v1() {
    // no-op
}

static ark_gpu_status hip_device_count_v1(std::uint32_t* out_count) {
    if (!out_count) return ARK_GPU_ERR_INVALID_ARG;
    int cnt = 0;
    hipError_t e = hipGetDeviceCount(&cnt);
    if (e != hipSuccess) return ARK_GPU_ERR_DRIVER;
    *out_count = (cnt < 0) ? 0u : static_cast<std::uint32_t>(cnt);
    return ARK_GPU_OK;
}

static ark_gpu_status hip_set_device_v1(std::int32_t device_id) {
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    int cnt = 0;
    if (hipGetDeviceCount(&cnt) != hipSuccess || cnt <= 0) return ARK_GPU_ERR_NO_DEVICE;
    if (device_id < 0 || device_id >= cnt) return ARK_GPU_ERR_INVALID_ARG;
    hipError_t e = hipSetDevice(device_id);
    if (e != hipSuccess) return ARK_GPU_ERR_DRIVER;
    s.device_id = device_id;
    return ARK_GPU_OK;
}

static ark_gpu_status hip_get_device_v1(std::int32_t* out_device_id) {
    if (!out_device_id) return ARK_GPU_ERR_INVALID_ARG;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.initialized) return ARK_GPU_ERR_NOT_READY;
    *out_device_id = s.device_id;
    return ARK_GPU_OK;
}

static ark_gpu_status hip_get_device_info_v1(std::int32_t device_id, ark_gpu_device_info* out_info) {
    if (!out_info) return ARK_GPU_ERR_INVALID_ARG;
    if (out_info->size != sizeof(ark_gpu_device_info)) return ARK_GPU_ERR_INVALID_ARG;

    int cnt = 0;
    if (hipGetDeviceCount(&cnt) != hipSuccess || cnt <= 0) return ARK_GPU_ERR_NO_DEVICE;
    if (device_id < 0 || device_id >= cnt) return ARK_GPU_ERR_INVALID_ARG;

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_id) != hipSuccess) return ARK_GPU_ERR_DRIVER;

    std::memset(out_info, 0, sizeof(*out_info));
    out_info->size = sizeof(ark_gpu_device_info);
    out_info->device_id = static_cast<std::uint32_t>(device_id);
    out_info->backend = ARK_GPU_BACKEND_HIP;

    if (prop.name) {
        std::snprintf(out_info->name, sizeof(out_info->name), "%s", prop.name);
    }

    out_info->major = static_cast<std::uint32_t>(prop.major);
    out_info->minor = static_cast<std::uint32_t>(prop.minor);

    out_info->global_mem_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
    out_info->warp_size = static_cast<std::uint32_t>(prop.warpSize);
    out_info->sm_count  = static_cast<std::uint32_t>(prop.multiProcessorCount);
    out_info->max_threads_per_sm    = static_cast<std::uint32_t>(prop.maxThreadsPerMultiProcessor);
    out_info->max_threads_per_block = static_cast<std::uint32_t>(prop.maxThreadsPerBlock);

    out_info->max_grid_dim_x = static_cast<std::uint32_t>(prop.maxGridSize[0]);
    out_info->max_grid_dim_y = static_cast<std::uint32_t>(prop.maxGridSize[1]);
    out_info->max_grid_dim_z = static_cast<std::uint32_t>(prop.maxGridSize[2]);

    out_info->max_block_dim_x = static_cast<std::uint32_t>(prop.maxThreadsDim[0]);
    out_info->max_block_dim_y = static_cast<std::uint32_t>(prop.maxThreadsDim[1]);
    out_info->max_block_dim_z = static_cast<std::uint32_t>(prop.maxThreadsDim[2]);

    out_info->supports_async_copy = 1u;
    return ARK_GPU_OK;
}

static ark_gpu_device_ptr hip_alloc_v1(std::int64_t bytes) {
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    if (bytes <= 0) return nullptr;

    void* p = nullptr;
    hipError_t e = hipMalloc(&p, static_cast<std::size_t>(bytes));
    if (e != hipSuccess) {
        set_last_error("hipMalloc failed");
        return nullptr;
    }
    return p;
}

static void hip_free_v1(ark_gpu_device_ptr p) {
    if (!p) return;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    HIP_CHECK(hipFree(p));
}

static void hip_memcpy_to_device_v1(ark_gpu_device_ptr dst, const void* src, std::int64_t bytes) {
    if (!dst || !src || bytes <= 0) return;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    HIP_CHECK(hipMemcpy(dst, src, static_cast<std::size_t>(bytes), hipMemcpyHostToDevice));
}

static void hip_memcpy_to_host_v1(void* dst, ark_gpu_device_ptr src, std::int64_t bytes) {
    if (!dst || !src || bytes <= 0) return;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    HIP_CHECK(hipMemcpy(dst, src, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost));
}

static ark_gpu_status hip_memcpy_v1(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind) {
    if (!dst || !src || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    hipMemcpyKind hk{};
    switch (kind) {
        case ARK_GPU_COPY_H2D: hk = hipMemcpyHostToDevice; break;
        case ARK_GPU_COPY_D2H: hk = hipMemcpyDeviceToHost; break;
        case ARK_GPU_COPY_D2D: hk = hipMemcpyDeviceToDevice; break;
        default: return ARK_GPU_ERR_INVALID_ARG;
    }

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipError_t e = hipMemcpy(dst, src, static_cast<std::size_t>(bytes), hk);
    return (e == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_memcpy_async_v1(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind, ark_gpu_stream stream) {
    if (!dst || !src || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    hipMemcpyKind hk{};
    switch (kind) {
        case ARK_GPU_COPY_H2D: hk = hipMemcpyHostToDevice; break;
        case ARK_GPU_COPY_D2H: hk = hipMemcpyDeviceToHost; break;
        case ARK_GPU_COPY_D2D: hk = hipMemcpyDeviceToDevice; break;
        default: return ARK_GPU_ERR_INVALID_ARG;
    }

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipError_t e = hipMemcpyAsync(dst, src, static_cast<std::size_t>(bytes), hk, ark_to_hip_stream(stream));
    return (e == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_memset_async_v1(ark_gpu_device_ptr dst, std::int32_t value, std::int64_t bytes, ark_gpu_stream stream) {
    if (!dst || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    hipError_t e = hipMemsetAsync(dst, value, static_cast<std::size_t>(bytes), ark_to_hip_stream(stream));
    return (e == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_stream_create_v1(const ark_gpu_stream_desc* desc, ark_gpu_stream* out_stream) {
    if (!out_stream) return ARK_GPU_ERR_INVALID_ARG;

    bool non_blocking = true;
    int priority = 0;
    if (desc) {
        if (desc->size != sizeof(ark_gpu_stream_desc)) return ARK_GPU_ERR_INVALID_ARG;
        non_blocking = (desc->flags & ARK_GPU_STREAM_NON_BLOCKING) != 0;
        priority = desc->priority;
    }

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipStream_t hs = nullptr;
    unsigned flags = non_blocking ? hipStreamNonBlocking : hipStreamDefault;

    if (priority == 0) {
        hipError_t e = hipStreamCreateWithFlags(&hs, flags);
        if (e != hipSuccess) return ARK_GPU_ERR_DRIVER;
    } else {
        int low = 0, high = 0;
        HIP_CHECK(hipDeviceGetStreamPriorityRange(&low, &high));
        const int prio = (priority > 0) ? high : low;
        hipError_t e = hipStreamCreateWithPriority(&hs, flags, prio);
        if (e != hipSuccess) return ARK_GPU_ERR_DRIVER;
    }

    *out_stream = ark_from_hip_stream(hs);
    return ARK_GPU_OK;
}

static void hip_stream_destroy_v1(ark_gpu_stream stream) {
    hipStream_t s = ark_to_hip_stream(stream);
    if (!s) return;
    HipState& st = S();
    std::lock_guard<std::mutex> lock(st.mu);
    ensure_init_locked(st);
    HIP_CHECK(hipStreamDestroy(s));
}

static ark_gpu_status hip_stream_synchronize_v1(ark_gpu_stream stream) {
    HipState& st = S();
    std::lock_guard<std::mutex> lock(st.mu);
    ensure_init_locked(st);

    hipStream_t s = ark_to_hip_stream(stream);
    hipError_t e = s ? hipStreamSynchronize(s) : hipDeviceSynchronize();
    return (e == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_event_create_v1(const ark_gpu_event_desc* desc, ark_gpu_event* out_evt) {
    if (!out_evt) return ARK_GPU_ERR_INVALID_ARG;
    bool timing = true;
    if (desc) {
        if (desc->size != sizeof(ark_gpu_event_desc)) return ARK_GPU_ERR_INVALID_ARG;
        timing = (desc->flags & ARK_GPU_EVENT_DISABLE_TIMING) == 0;
    }

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipEvent_t e = nullptr;
    unsigned flags = timing ? hipEventDefault : hipEventDisableTiming;
    hipError_t he = hipEventCreateWithFlags(&e, flags);
    if (he != hipSuccess) return ARK_GPU_ERR_DRIVER;
    *out_evt = ark_from_hip_event(e);
    return ARK_GPU_OK;
}

static void hip_event_destroy_v1(ark_gpu_event evt) {
    hipEvent_t e = ark_to_hip_event(evt);
    if (!e) return;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    HIP_CHECK(hipEventDestroy(e));
}

static ark_gpu_status hip_event_record_v1(ark_gpu_event evt, ark_gpu_stream stream) {
    hipEvent_t e = ark_to_hip_event(evt);
    if (!e) return ARK_GPU_ERR_INVALID_ARG;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    hipError_t he = hipEventRecord(e, ark_to_hip_stream(stream));
    return (he == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_event_synchronize_v1(ark_gpu_event evt) {
    hipEvent_t e = ark_to_hip_event(evt);
    if (!e) return ARK_GPU_ERR_INVALID_ARG;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    hipError_t he = hipEventSynchronize(e);
    return (he == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_event_elapsed_ms_v1(ark_gpu_event start_evt, ark_gpu_event end_evt, float* out_ms) {
    if (!out_ms) return ARK_GPU_ERR_INVALID_ARG;
    hipEvent_t a = ark_to_hip_event(start_evt);
    hipEvent_t b = ark_to_hip_event(end_evt);
    if (!a || !b) return ARK_GPU_ERR_INVALID_ARG;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    float ms = 0.0f;
    hipError_t he = hipEventElapsedTime(&ms, a, b);
    if (he != hipSuccess) return ARK_GPU_ERR_DRIVER;
    *out_ms = ms;
    return ARK_GPU_OK;
}

static ark_gpu_status hip_stream_wait_event_v1(ark_gpu_stream stream, ark_gpu_event evt) {
    hipStream_t s = ark_to_hip_stream(stream);
    hipEvent_t e  = ark_to_hip_event(evt);
    if (!e) return ARK_GPU_ERR_INVALID_ARG;

    HipState& st = S();
    std::lock_guard<std::mutex> lock(st.mu);
    ensure_init_locked(st);

    hipError_t he = hipStreamWaitEvent(s, e, 0u);
    return (he == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

// ---- Modules / RTC ----

static ark_gpu_status hip_module_load_v1(const ark_gpu_module_desc* desc, ark_gpu_module* out_mod) {
    if (!desc || !out_mod) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->size != sizeof(ark_gpu_module_desc)) return ARK_GPU_ERR_INVALID_ARG;
    if (!desc->module_key || !desc->module_key[0]) return ARK_GPU_ERR_INVALID_ARG;
    if (!desc->bytes || desc->byte_len == 0) return ARK_GPU_ERR_INVALID_ARG;

    if (desc->kind != ARK_GPU_MODULE_HIP_HSACO) return ARK_GPU_ERR_UNSUPPORTED;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    const std::string key(desc->module_key);

    std::lock_guard<std::mutex> lk(s.mu_modules);
    auto it = s.modules.find(key);
    if (it != s.modules.end()) {
        it->second.refcnt.fetch_add(1, std::memory_order_relaxed);
        *out_mod = reinterpret_cast<ark_gpu_module>(it->second.mod);
        return ARK_GPU_OK;
    }

    ModuleEntry ent;
    ent.key = key;
    ent.image.assign(static_cast<const std::uint8_t*>(desc->bytes),
                     static_cast<const std::uint8_t*>(desc->bytes) + desc->byte_len);

    hipModule_t mod = nullptr;
    hipError_t he = hipModuleLoadData(&mod, ent.image.data());
    if (he != hipSuccess) return ARK_GPU_ERR_DRIVER;

    ent.mod = mod;

    s.modules.emplace(ent.key, std::move(ent));
    if (s.default_module_key.empty()) s.default_module_key = key;

    invalidate_kernel_cache_for_module_locked(s, key);

    *out_mod = reinterpret_cast<ark_gpu_module>(mod);
    return ARK_GPU_OK;
}

static void hip_module_unload_v1(ark_gpu_module mod) {
    if (!mod) return;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    std::lock_guard<std::mutex> lk(s.mu_modules);

    // Find by handle (rare): do linear scan, but only on unload.
    std::string key;
    for (auto& kv : s.modules) {
        if (reinterpret_cast<ark_gpu_module>(kv.second.mod) == mod) { key = kv.first; break; }
    }
    if (key.empty()) return;

    auto it = s.modules.find(key);
    if (it == s.modules.end()) return;

    const std::uint32_t prev = it->second.refcnt.fetch_sub(1, std::memory_order_acq_rel);
    if (prev > 1) return;

    invalidate_kernel_cache_for_module_locked(s, key);

    hipModule_t hm = it->second.mod;
    s.modules.erase(it);

    if (hm) HIP_CHECK(hipModuleUnload(hm));

    if (s.default_module_key == key) {
        s.default_module_key.clear();
        if (!s.modules.empty()) s.default_module_key = s.modules.begin()->first;
    }
}

static ark_gpu_status hip_module_set_default_v1(ark_gpu_module mod) {
    if (!mod) return ARK_GPU_ERR_INVALID_ARG;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    std::lock_guard<std::mutex> lk(s.mu_modules);
    for (auto& kv : s.modules) {
        if (kv.second.mod == reinterpret_cast<hipModule_t>(mod)) {
            s.default_module_key = kv.first;
            return ARK_GPU_OK;
        }
    }
    return ARK_GPU_ERR_NOT_FOUND;
}

static ark_gpu_status hip_rtc_compile_v1(const ark_gpu_rtc_desc* desc, ark_gpu_module* out_mod) {
    if (!desc || !out_mod) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->size != sizeof(ark_gpu_rtc_desc)) return ARK_GPU_ERR_INVALID_ARG;
    if (!desc->module_key || !desc->module_key[0]) return ARK_GPU_ERR_INVALID_ARG;
    if (!desc->source_utf8 || !desc->source_utf8[0]) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->kind != ARK_GPU_MODULE_SRC_HIP) return ARK_GPU_ERR_UNSUPPORTED;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    const std::string key(desc->module_key);

    std::lock_guard<std::mutex> lk(s.mu_modules);
    auto it = s.modules.find(key);
    if (it != s.modules.end()) {
        it->second.refcnt.fetch_add(1, std::memory_order_relaxed);
        *out_mod = reinterpret_cast<ark_gpu_module>(it->second.mod);
        return ARK_GPU_OK;
    }

    hiprtcProgram prog{};
    const char* pname = desc->program_name ? desc->program_name : "ark_hiprtc";
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, desc->source_utf8, pname, 0, nullptr, nullptr));

    std::vector<const char*> opts;
    opts.reserve(static_cast<std::size_t>(desc->option_count) + 4);

    std::string arch_opt;
    if (!s.gcn_arch.empty()) {
        arch_opt = std::string("--gpu-architecture=") + s.gcn_arch;
        opts.push_back(arch_opt.c_str());
    }
    for (int i = 0; i < desc->option_count; ++i) opts.push_back(desc->options[i]);

    hiprtcResult cr = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.empty() ? nullptr : opts.data());

    std::size_t log_size = 0;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size ? log_size : 1);
    if (log_size) HIPRTC_CHECK(hiprtcGetProgramLog(prog, log.data()));

    if (cr != HIPRTC_SUCCESS) {
        HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
        set_last_error("hiprtcCompileProgram failed");
        // Still store log for retrieval:
        ModuleEntry ent;
        ent.key = key;
        ent.rtc.log = std::move(log);
        s.modules.emplace(ent.key, std::move(ent));
        s.modules[ent.key].refcnt.store(1, std::memory_order_relaxed);
        *out_mod = nullptr;
        return ARK_GPU_ERR_COMPILATION;
    }

    std::size_t code_size = 0;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
    std::vector<std::uint8_t> code(code_size);
    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    hipModule_t mod = nullptr;
    hipError_t he = hipModuleLoadData(&mod, code.data());
    if (he != hipSuccess) {
        set_last_error("hipModuleLoadData failed after rtc");
        return ARK_GPU_ERR_DRIVER;
    }

    ModuleEntry ent;
    ent.key = key;
    ent.image = std::move(code);
    ent.mod = mod;
    ent.rtc.log = std::move(log);

    s.modules.emplace(ent.key, std::move(ent));
    if (s.default_module_key.empty()) s.default_module_key = key;

    invalidate_kernel_cache_for_module_locked(s, key);

    *out_mod = reinterpret_cast<ark_gpu_module>(mod);
    return ARK_GPU_OK;
}

static const char* hip_rtc_last_log_v1(const char* module_key) {
    if (!module_key || !module_key[0]) return "";
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.initialized) return "";
    std::lock_guard<std::mutex> lk(s.mu_modules);
    auto it = s.modules.find(std::string(module_key));
    if (it == s.modules.end()) return "";
    return it->second.rtc.log.c_str();
}

static ark_gpu_status hip_kernel_lookup_v1(ark_gpu_module mod, const char* kernel_name, ark_gpu_kernel* out_kernel) {
    if (!kernel_name || !kernel_name[0] || !out_kernel) return ARK_GPU_ERR_INVALID_ARG;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipFunction_t fn = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.mu_modules);

        if (mod) {
            hipModule_t hm = reinterpret_cast<hipModule_t>(mod);
            if (hipModuleGetFunction(&fn, hm, kernel_name) != hipSuccess) return ARK_GPU_ERR_NOT_FOUND;
        } else {
            fn = resolve_function_locked(s, std::string{}, std::string(kernel_name));
            if (!fn) return ARK_GPU_ERR_NOT_FOUND;
        }
    }

    *out_kernel = reinterpret_cast<ark_gpu_kernel>(fn);
    return ARK_GPU_OK;
}

static ark_gpu_status hip_kernel_lookup_default_v1(const char* kernel_name, ark_gpu_kernel* out_kernel) {
    return hip_kernel_lookup_v1(nullptr, kernel_name, out_kernel);
}

// ---- Launch ----

static void hip_launch_v1(const char* kernel_name,
                          void** args, std::int32_t /*arg_count*/,
                          std::int32_t gx, std::int32_t gy, std::int32_t gz,
                          std::int32_t bx, std::int32_t by, std::int32_t bz,
                          ark_gpu_stream stream) {
    if (!kernel_name || !kernel_name[0]) ark_hip_abort("launch: empty kernel_name", __FILE__, __LINE__);
    if (gx <= 0 || gy <= 0 || gz <= 0 || bx <= 0 || by <= 0 || bz <= 0) ark_hip_abort("launch: invalid dims", __FILE__, __LINE__);

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    const std::string_view full(kernel_name);
    hipFunction_t fn = nullptr;

    if (ark_sv_starts_with(full, "hip:function:")) {
        std::uint64_t v = 0;
        if (!ark_parse_hex_u64(full.substr(std::strlen("hip:function:")), v)) {
            ark_hip_abort("launch: bad function token", __FILE__, __LINE__);
        }
        fn = reinterpret_cast<hipFunction_t>(static_cast<std::uintptr_t>(v));
    } else {
        std::string_view mod_sv;
        std::string_view k_sv;
        {
            auto [a, b] = ark_sv_split_once(full, "::");
            if (!b.empty()) { mod_sv = a; k_sv = b; }
            else { k_sv = a; }
        }

        const std::string mod = !mod_sv.empty() ? std::string(mod_sv) : std::string{};
        const std::string kn  = std::string(k_sv);

        std::lock_guard<std::mutex> lk(s.mu_modules);
        fn = resolve_function_locked(s, mod, kn);
    }

    if (!fn) {
        set_last_error("kernel not found");
        ark_hip_abort("launch: kernel not found", __FILE__, __LINE__);
    }

    HIP_CHECK(hipModuleLaunchKernel(fn,
                                   static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz),
                                   static_cast<unsigned>(bx), static_cast<unsigned>(by), static_cast<unsigned>(bz),
                                   0u,
                                   ark_to_hip_stream(stream),
                                   args,
                                   nullptr));
}

static ark_gpu_status hip_launch_handle_v1(ark_gpu_kernel kernel,
                                          void** args, std::int32_t /*arg_count*/,
                                          std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                          std::int32_t bx, std::int32_t by, std::int32_t bz,
                                          ark_gpu_stream stream) {
    if (!kernel) return ARK_GPU_ERR_INVALID_ARG;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipFunction_t fn = reinterpret_cast<hipFunction_t>(kernel);
    hipError_t e = hipModuleLaunchKernel(fn,
                                        static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz),
                                        static_cast<unsigned>(bx), static_cast<unsigned>(by), static_cast<unsigned>(bz),
                                        0u,
                                        ark_to_hip_stream(stream),
                                        args,
                                        nullptr);

    return (e == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status hip_launch_ex_v1(const char* kernel_name,
                                      void** args, std::int32_t /*arg_count*/,
                                      std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                      std::int32_t bx, std::int32_t by, std::int32_t bz,
                                      const ark_gpu_launch_desc* desc,
                                      ark_gpu_stream stream) {
    if (!kernel_name || !kernel_name[0]) return ARK_GPU_ERR_INVALID_ARG;
    std::uint32_t smem = 0;
    if (desc) {
        if (desc->size != sizeof(ark_gpu_launch_desc)) return ARK_GPU_ERR_INVALID_ARG;
        smem = desc->shared_mem_bytes;
    }

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipFunction_t fn = nullptr;

    const std::string_view full(kernel_name);
    if (ark_sv_starts_with(full, "hip:function:")) {
        std::uint64_t v = 0;
        if (!ark_parse_hex_u64(full.substr(std::strlen("hip:function:")), v)) return ARK_GPU_ERR_INVALID_ARG;
        fn = reinterpret_cast<hipFunction_t>(static_cast<std::uintptr_t>(v));
    } else {
        std::string_view mod_sv;
        std::string_view k_sv;
        {
            auto [a, b] = ark_sv_split_once(full, "::");
            if (!b.empty()) { mod_sv = a; k_sv = b; } else { k_sv = a; }
        }
        const std::string mod = !mod_sv.empty() ? std::string(mod_sv) : std::string{};
        const std::string kn  = std::string(k_sv);
        std::lock_guard<std::mutex> lk(s.mu_modules);
        fn = resolve_function_locked(s, mod, kn);
    }

    if (!fn) return ARK_GPU_ERR_NOT_FOUND;

    hipError_t e = hipModuleLaunchKernel(fn,
                                        static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz),
                                        static_cast<unsigned>(bx), static_cast<unsigned>(by), static_cast<unsigned>(bz),
                                        smem,
                                        ark_to_hip_stream(stream),
                                        args,
                                        nullptr);

    return (e == hipSuccess) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static void hip_synchronize_v1(ark_gpu_stream stream) {
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    hipStream_t hs = ark_to_hip_stream(stream);
    if (!hs) HIP_CHECK(hipDeviceSynchronize());
    else HIP_CHECK(hipStreamSynchronize(hs));
}

static const char* hip_last_error_string_v1() {
    const char* e = S().last_error;
    return e ? e : "";
}

static std::uint64_t hip_feature_bits_v1() {
    // Conservative, but true for our implementation:
    // - Streams: yes
    // - Events: yes
    // - Async memcpy/memset: yes
    // - D2D memcpy: yes
    // - Pinned host alloc: yes (hipHostMalloc)
    // - Managed alloc: implemented via alloc_ex if enabled (we expose alloc_ex)
    // - Modules: yes
    // - Runtime compile: yes (hiprtc)
    // - Kernel handles: yes
    // - Launch_ex: yes
    // - Device query: yes
    // - Multi device: yes
    return (ARK_GPU_FEAT_STREAMS |
            ARK_GPU_FEAT_EVENTS |
            ARK_GPU_FEAT_ASYNC_MEMCPY |
            ARK_GPU_FEAT_ASYNC_MEMSET |
            ARK_GPU_FEAT_D2D_MEMCPY |
            ARK_GPU_FEAT_PINNED_HOST_ALLOC |
            ARK_GPU_FEAT_MANAGED_ALLOC |
            ARK_GPU_FEAT_MODULES |
            ARK_GPU_FEAT_RUNTIME_COMPILE |
            ARK_GPU_FEAT_KERNEL_HANDLES |
            ARK_GPU_FEAT_LAUNCH_EX |
            ARK_GPU_FEAT_DEVICE_QUERY |
            ARK_GPU_FEAT_MULTI_DEVICE);
}

static ark_gpu_status hip_alloc_ex_v1(const ark_gpu_alloc_desc* desc, std::int64_t bytes, ark_gpu_device_ptr* out_p) {
    if (!out_p) return ARK_GPU_ERR_INVALID_ARG;
    *out_p = nullptr;

    if (!desc) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->size != sizeof(ark_gpu_alloc_desc)) return ARK_GPU_ERR_INVALID_ARG;
    if (bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    void* p = nullptr;

    switch (desc->kind) {
        case ARK_GPU_MEM_DEVICE: {
            hipError_t e = hipMalloc(&p, static_cast<std::size_t>(bytes));
            if (e != hipSuccess) return ARK_GPU_ERR_OOM;
            break;
        }
        case ARK_GPU_MEM_MANAGED: {
            if (!s.supports_managed) return ARK_GPU_ERR_UNSUPPORTED;
            hipError_t e = hipMallocManaged(&p, static_cast<std::size_t>(bytes), hipMemAttachGlobal);
            if (e != hipSuccess) return ARK_GPU_ERR_OOM;
            break;
        }
        default:
            return ARK_GPU_ERR_UNSUPPORTED;
    }

    *out_p = p;
    return ARK_GPU_OK;
}

static ark_gpu_status hip_host_alloc_v1(const ark_gpu_alloc_desc* desc, std::int64_t bytes, void** out_host) {
    if (!out_host) return ARK_GPU_ERR_INVALID_ARG;
    *out_host = nullptr;

    if (!desc) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->size != sizeof(ark_gpu_alloc_desc)) return ARK_GPU_ERR_INVALID_ARG;
    if (bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    if (desc->kind != ARK_GPU_MEM_HOST_PINNED) return ARK_GPU_ERR_UNSUPPORTED;

    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);

    unsigned flags = hipHostMallocDefault;
    void* p = nullptr;
    hipError_t e = hipHostMalloc(&p, static_cast<std::size_t>(bytes), flags);
    if (e != hipSuccess) return ARK_GPU_ERR_OOM;

    *out_host = p;
    return ARK_GPU_OK;
}

static void hip_host_free_v1(void* host) {
    if (!host) return;
    HipState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_init_locked(s);
    HIP_CHECK(hipHostFree(host));
}

} // namespace

// ------------------------------
// Plugin vtable export
// ------------------------------

extern "C" {

// 1. Define Features as a Compile-Time Constant
static constexpr std::uint64_t kHipFeatures = 
    ARK_GPU_FEAT_STREAMS |
    ARK_GPU_FEAT_EVENTS |
    ARK_GPU_FEAT_ASYNC_MEMCPY |
    ARK_GPU_FEAT_ASYNC_MEMSET |
    ARK_GPU_FEAT_D2D_MEMCPY |
    ARK_GPU_FEAT_PINNED_HOST_ALLOC |
    ARK_GPU_FEAT_MANAGED_ALLOC |
    ARK_GPU_FEAT_MODULES |
    ARK_GPU_FEAT_RUNTIME_COMPILE |
    ARK_GPU_FEAT_KERNEL_HANDLES |
    ARK_GPU_FEAT_LAUNCH_EX |
    ARK_GPU_FEAT_DEVICE_QUERY |
    ARK_GPU_FEAT_MULTI_DEVICE;

// 2. Aggregate Initialization (Safe & Stable)
static const ark_gpu_backend_v1 g_vt = {
    ARK_GPU_BACKEND_ABI_V1,           // abi
    sizeof(ark_gpu_backend_v1),       // size
    ARK_GPU_BACKEND_HIP,              // kind
    "hip",                            // name
    kHipFeatures,                     // features

    // Lifecycle
    hip_init_v1,
    hip_shutdown_v1,
    hip_enter_thread_v1,
    hip_leave_thread_v1,

    // Device Query
    hip_device_count_v1,
    hip_set_device_v1,
    hip_get_device_v1,
    hip_get_device_info_v1,

    // Memory
    hip_alloc_v1,
    hip_free_v1,
    hip_alloc_ex_v1,
    hip_host_alloc_v1,
    hip_host_free_v1,

    hip_memcpy_to_device_v1,
    hip_memcpy_to_host_v1,
    hip_memcpy_v1,
    hip_memcpy_async_v1,
    hip_memset_async_v1,

    // Streams
    hip_stream_create_v1,
    hip_stream_destroy_v1,
    hip_stream_synchronize_v1,

    // Events
    hip_event_create_v1,
    hip_event_destroy_v1,
    hip_event_record_v1,
    hip_event_synchronize_v1,
    hip_event_elapsed_ms_v1,
    hip_stream_wait_event_v1,

    // Modules
    hip_module_load_v1,
    hip_module_unload_v1,
    hip_module_set_default_v1,

    // RTC
    hip_rtc_compile_v1,
    hip_rtc_last_log_v1,

    // Kernel Lookup
    hip_kernel_lookup_v1,
    hip_kernel_lookup_default_v1,

    // Launch
    hip_launch_v1,
    hip_launch_handle_v1,
    hip_launch_ex_v1,

    // Global Sync
    hip_synchronize_v1,

    // Debug
    hip_last_error_string_v1,

    // Reserved Padding (Zero-Initialized)
    { nullptr }
};

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1() {
    return &g_vt;
}

} // extern "C"

#else // ARK_BACKEND_HIP

extern "C" {

// Disabled Stub: Returns a valid struct layout but with 0 features/null pointers.
// The HAL validator will correctly reject this because required pointers (init, alloc) are NULL.
static const ark_gpu_backend_v1 g_vt_disabled = {
    ARK_GPU_BACKEND_ABI_V1,
    sizeof(ark_gpu_backend_v1),
    ARK_GPU_BACKEND_HIP,
    "hip(disabled)",
    0, // No features
    
    // All function pointers default to nullptr
    nullptr
};

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1() {
    return &g_vt_disabled;
}

} // extern "C"

#endif // ARK_BACKEND_HIP
