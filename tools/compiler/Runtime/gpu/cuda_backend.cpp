// tools/compiler/Runtime/gpu/cuda_backend.cpp
#include "gpu_backend.h"
#ifndef ARK_BACKEND_CUDA
#error "CUDA plugin target compiled without ARK_BACKEND_CUDA"
#endif

#ifdef ARK_BACKEND_CUDA

#include <cuda.h>

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

static inline const char* ark_cuda_err_name(CUresult r) {
    const char* n = nullptr;
    if (cuGetErrorName(r, &n) != CUDA_SUCCESS || n == nullptr) return "CUDA_ERROR_UNKNOWN";
    return n;
}

static inline const char* ark_cuda_err_str(CUresult r) {
    const char* s = nullptr;
    if (cuGetErrorString(r, &s) != CUDA_SUCCESS || s == nullptr) return "unknown CUDA driver error";
    return s;
}

static inline void ark_cuda_abort(const char* what, const char* file, int line) {
    std::fprintf(stderr, "[ARK CUDA] %s at %s:%d\n", what, file, line);
    std::abort();
}

#define CU_CHECK(call)                                                                         \
    do {                                                                                       \
        CUresult _r = (call);                                                                  \
        if (_r != CUDA_SUCCESS) {                                                              \
            std::fprintf(stderr, "[ARK CUDA] %s failed: %s (%s) at %s:%d\n",                   \
                         #call, ark_cuda_err_name(_r), ark_cuda_err_str(_r),                   \
                         __FILE__, __LINE__);                                                  \
            std::abort();                                                                      \
        }                                                                                      \
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

static inline CUdeviceptr ark_to_cu_device_ptr(ark_gpu_device_ptr p) {
    return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(p));
}
static inline ark_gpu_device_ptr ark_from_cu_device_ptr(CUdeviceptr p) {
    return reinterpret_cast<ark_gpu_device_ptr>(static_cast<std::uintptr_t>(p));
}

static inline CUstream ark_to_cu_stream(ark_gpu_stream s) { return reinterpret_cast<CUstream>(s); }
static inline ark_gpu_stream ark_from_cu_stream(CUstream s) { return reinterpret_cast<ark_gpu_stream>(s); }

static inline CUevent ark_to_cu_event(ark_gpu_event e) { return reinterpret_cast<CUevent>(e); }
static inline ark_gpu_event ark_from_cu_event(CUevent e) { return reinterpret_cast<ark_gpu_event>(e); }

// ------------------------------
// State
// ------------------------------

namespace {

struct JitLog {
    std::string info;
    std::string error;
};

struct ModuleEntry {
    CUmodule module = nullptr;
    std::vector<std::uint8_t> image;
    std::string key;
    std::atomic<std::uint32_t> refcnt{1};
    JitLog jit;

    ModuleEntry() = default;

    ModuleEntry(const ModuleEntry&) = delete;
    ModuleEntry& operator=(const ModuleEntry&) = delete;

    ModuleEntry(ModuleEntry&& o) noexcept
        : module(o.module),
          image(std::move(o.image)),
          key(std::move(o.key)),
          refcnt(o.refcnt.load(std::memory_order_relaxed)),
          jit(std::move(o.jit)) {
        o.module = nullptr;
        o.refcnt.store(0u, std::memory_order_relaxed);
    }

    ModuleEntry& operator=(ModuleEntry&& o) noexcept {
        if (this == &o) return *this;

        module = o.module;
        image = std::move(o.image);
        key = std::move(o.key);
        refcnt.store(o.refcnt.load(std::memory_order_relaxed), std::memory_order_relaxed);
        jit = std::move(o.jit);

        o.module = nullptr;
        o.refcnt.store(0u, std::memory_order_relaxed);
        return *this;
    }
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

struct CudaState {
    std::mutex mu;
    bool initialized = false;

    CUdevice  dev = 0;
    CUcontext ctx = nullptr;
    int32_t   device_id = -1;

    int  cc_major = 0;
    int  cc_minor = 0;
    bool has_async_copy = false;

    int prio_low  = 0;
    int prio_high = 0;

    bool supports_managed = false;
    bool supports_pinned  = false;

    ark_gpu_log_fn logger = nullptr;
    void* logger_user = nullptr;

    std::mutex mu_modules;
    std::unordered_map<std::string, ModuleEntry> modules;

    std::string default_module_key;
    CUmodule    default_module = nullptr;

    std::unordered_map<KernelKey, CUfunction, KernelKeyHash> kernels;

    // [FIX] Use std::string to store error so it survives scope.
    std::string last_error;
};


static CudaState& S() {
    static CudaState s;
    return s;
}

static inline void log_msg(ark_gpu_log_level lvl, const char* msg) {
    CudaState& s = S();
    if (s.logger) s.logger(lvl, msg, s.logger_user);
    else std::fprintf(stderr, "[ARK CUDA] %s\n", msg);
}

// [FIX] Update to take string_view/string and copy it safely.
static inline void set_last_error(std::string_view msg) {
    S().last_error = std::string(msg);
}

static inline void set_current_ctx_locked(CudaState& s) {
    if (s.ctx == nullptr) ark_cuda_abort("no CUDA context", __FILE__, __LINE__);
    CU_CHECK(cuCtxSetCurrent(s.ctx));
}

static inline void ensure_current_locked(CudaState& s) {
    if (!s.initialized) ark_cuda_abort("backend not initialized", __FILE__, __LINE__);
    set_current_ctx_locked(s);
}

static void invalidate_kernel_cache_for_module_locked(CudaState& s, const std::string& module_key) {
    for (auto it = s.kernels.begin(); it != s.kernels.end();) {
        if (it->first.mod == module_key || it->first.mod.empty()) it = s.kernels.erase(it);
        else ++it;
    }
}

static CUfunction resolve_function_locked(CudaState& s, const std::string& module_key, const std::string& kernel) {
    KernelKey kk{module_key, kernel};
    auto kit = s.kernels.find(kk);
    if (kit != s.kernels.end()) return kit->second;

    CUfunction fn = nullptr;

    if (!module_key.empty()) {
        auto it = s.modules.find(module_key);
        if (it == s.modules.end()) return nullptr;
        if (cuModuleGetFunction(&fn, it->second.module, kernel.c_str()) != CUDA_SUCCESS) return nullptr;
        s.kernels.emplace(std::move(kk), fn);
        return fn;
    }

    if (!s.default_module_key.empty()) {
        auto it = s.modules.find(s.default_module_key);
        if (it != s.modules.end()) {
            if (cuModuleGetFunction(&fn, it->second.module, kernel.c_str()) == CUDA_SUCCESS) {
                s.kernels.emplace(std::move(kk), fn);
                return fn;
            }
        }
    }

    for (auto& kv : s.modules) {
        if (cuModuleGetFunction(&fn, kv.second.module, kernel.c_str()) == CUDA_SUCCESS) {
            s.kernels.emplace(std::move(kk), fn);
            return fn;
        }
    }

    return nullptr;
}

static void shutdown_locked(CudaState& s) {
    if (!s.initialized) {
        s.last_error.clear();
        return;
    }

    {
        std::lock_guard<std::mutex> lk(s.mu_modules);
        s.kernels.clear();
        for (auto& kv : s.modules) {
            if (kv.second.module) CU_CHECK(cuModuleUnload(kv.second.module));
            kv.second.module = nullptr;
        }
        s.modules.clear();
        s.default_module_key.clear();
    }

    if (s.ctx) {
        CU_CHECK(cuCtxSetCurrent(s.ctx));
        CU_CHECK(cuDevicePrimaryCtxRelease(s.dev));
    }

    s.ctx = nullptr;
    s.dev = 0;
    s.device_id = -1;
    s.initialized = false;
    s.cc_major = 0;
    s.cc_minor = 0;
    s.has_async_copy = false;
    s.prio_low = 0;
    s.prio_high = 0;
    s.supports_managed = false;
    s.supports_pinned = false;
    s.last_error.clear();
}

// ------------------------------
// VTable implementations
// ------------------------------

static ark_gpu_status cuda_init_v1(std::int32_t device_id, const ark_gpu_init_params* params) {
    CudaState& s = S();
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
            ensure_current_locked(s);
            return ARK_GPU_OK;
        }
        shutdown_locked(s);
    }

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        set_last_error("cuInit failed");
        return ARK_GPU_ERR_DRIVER;
    }

    int cnt = 0;
    r = cuDeviceGetCount(&cnt);
    if (r != CUDA_SUCCESS || cnt <= 0) {
        set_last_error("no CUDA devices available");
        return ARK_GPU_ERR_NO_DEVICE;
    }

    if (device_id < 0) device_id = 0;
    if (device_id >= cnt) device_id = 0;

    r = cuDeviceGet(&s.dev, device_id);
    if (r != CUDA_SUCCESS) {
        set_last_error("cuDeviceGet failed");
        return ARK_GPU_ERR_INVALID_ARG;
    }
    s.device_id = device_id;

    r = cuDevicePrimaryCtxRetain(&s.ctx, s.dev);
    if (r != CUDA_SUCCESS || s.ctx == nullptr) {
        set_last_error("cuDevicePrimaryCtxRetain failed");
        return ARK_GPU_ERR_DRIVER;
    }

    set_current_ctx_locked(s);

    CU_CHECK(cuDeviceGetAttribute(&s.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, s.dev));
    CU_CHECK(cuDeviceGetAttribute(&s.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, s.dev));

    int async_engine_cnt = 0;
    CU_CHECK(cuDeviceGetAttribute(&async_engine_cnt, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, s.dev));
    s.has_async_copy = (async_engine_cnt > 0);

    CU_CHECK(cuCtxGetStreamPriorityRange(&s.prio_low, &s.prio_high));

    int managed = 0;
    if (cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, s.dev) == CUDA_SUCCESS) {
        s.supports_managed = (managed != 0);
    } else {
        s.supports_managed = false;
    }

    s.supports_pinned = true;
    s.initialized = true;

    log_msg(ARK_GPU_LOG_INFO, "backend initialized");
    return ARK_GPU_OK;
}

static void cuda_shutdown_v1() {
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    shutdown_locked(s);
}

static void cuda_enter_thread_v1() {
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.initialized) return;
    ensure_current_locked(s);
}

static void cuda_leave_thread_v1() {}

// ---- Device Query ----

static ark_gpu_status cuda_device_count_v1(std::uint32_t* out_count) {
    if (!out_count) return ARK_GPU_ERR_INVALID_ARG;
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_DRIVER;
    int cnt = 0;
    r = cuDeviceGetCount(&cnt);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_DRIVER;
    *out_count = (cnt < 0) ? 0u : static_cast<std::uint32_t>(cnt);
    return ARK_GPU_OK;
}

static ark_gpu_status cuda_set_device_v1(std::int32_t device_id) {
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.initialized) return ARK_GPU_ERR_NOT_READY;

    int cnt = 0;
    CU_CHECK(cuDeviceGetCount(&cnt));
    if (device_id < 0 || device_id >= cnt) return ARK_GPU_ERR_INVALID_ARG;

    if (device_id == s.device_id) return ARK_GPU_OK;

    shutdown_locked(s);
    return cuda_init_v1(device_id, nullptr);
}

static ark_gpu_status cuda_get_device_v1(std::int32_t* out_device_id) {
    if (!out_device_id) return ARK_GPU_ERR_INVALID_ARG;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.initialized) return ARK_GPU_ERR_NOT_READY;
    *out_device_id = s.device_id;
    return ARK_GPU_OK;
}

static ark_gpu_status cuda_get_device_info_v1(std::int32_t device_id, ark_gpu_device_info* out_info) {
    if (!out_info) return ARK_GPU_ERR_INVALID_ARG;
    if (out_info->size != sizeof(ark_gpu_device_info)) return ARK_GPU_ERR_INVALID_ARG;

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_DRIVER;

    int cnt = 0;
    r = cuDeviceGetCount(&cnt);
    if (r != CUDA_SUCCESS || cnt <= 0) return ARK_GPU_ERR_NO_DEVICE;
    if (device_id < 0 || device_id >= cnt) return ARK_GPU_ERR_INVALID_ARG;

    CUdevice dev = 0;
    r = cuDeviceGet(&dev, device_id);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_INVALID_ARG;

    std::memset(out_info, 0, sizeof(*out_info));
    out_info->size = sizeof(ark_gpu_device_info);
    out_info->device_id = static_cast<std::uint32_t>(device_id);
    out_info->backend = ARK_GPU_BACKEND_CUDA;

    char name[128] = {0};
    if (cuDeviceGetName(name, static_cast<int>(sizeof(name)), dev) == CUDA_SUCCESS) {
        std::snprintf(out_info->name, sizeof(out_info->name), "%s", name);
    }

    int maj = 0, min = 0;
    CU_CHECK(cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    out_info->major = static_cast<std::uint32_t>(maj);
    out_info->minor = static_cast<std::uint32_t>(min);

    std::size_t total_mem = 0;
    if (cuDeviceTotalMem(&total_mem, dev) == CUDA_SUCCESS) out_info->global_mem_bytes = static_cast<std::uint64_t>(total_mem);

    int warp = 0;
    if (cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev) == CUDA_SUCCESS) out_info->warp_size = static_cast<std::uint32_t>(warp);

    int sm = 0;
    if (cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) == CUDA_SUCCESS) out_info->sm_count = static_cast<std::uint32_t>(sm);

    int maxThreadsPerSM = 0;
    if (cuDeviceGetAttribute(&maxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev) == CUDA_SUCCESS)
        out_info->max_threads_per_sm = static_cast<std::uint32_t>(maxThreadsPerSM);

    int maxThreadsPerBlock = 0;
    if (cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev) == CUDA_SUCCESS)
        out_info->max_threads_per_block = static_cast<std::uint32_t>(maxThreadsPerBlock);

    int gdx = 0, gdy = 0, gdz = 0;
    if (cuDeviceGetAttribute(&gdx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev) == CUDA_SUCCESS) out_info->max_grid_dim_x = static_cast<std::uint32_t>(gdx);
    if (cuDeviceGetAttribute(&gdy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev) == CUDA_SUCCESS) out_info->max_grid_dim_y = static_cast<std::uint32_t>(gdy);
    if (cuDeviceGetAttribute(&gdz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev) == CUDA_SUCCESS) out_info->max_grid_dim_z = static_cast<std::uint32_t>(gdz);

    int bdx = 0, bdy = 0, bdz = 0;
    if (cuDeviceGetAttribute(&bdx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev) == CUDA_SUCCESS) out_info->max_block_dim_x = static_cast<std::uint32_t>(bdx);
    if (cuDeviceGetAttribute(&bdy, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev) == CUDA_SUCCESS) out_info->max_block_dim_y = static_cast<std::uint32_t>(bdy);
    if (cuDeviceGetAttribute(&bdz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev) == CUDA_SUCCESS) out_info->max_block_dim_z = static_cast<std::uint32_t>(bdz);

    int async_engine_cnt = 0;
    if (cuDeviceGetAttribute(&async_engine_cnt, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev) == CUDA_SUCCESS)
        out_info->supports_async_copy = (async_engine_cnt > 0) ? 1u : 0u;

    return ARK_GPU_OK;
}

// ---- Memory ----

static ark_gpu_device_ptr cuda_alloc_v1(std::int64_t bytes) {
    if (bytes <= 0) return nullptr;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUdeviceptr p = 0;
    CUresult r = cuMemAlloc(&p, static_cast<std::size_t>(bytes));
    if (r != CUDA_SUCCESS) {
        set_last_error("cuMemAlloc failed");
        return nullptr;
    }
    return ark_from_cu_device_ptr(p);
}

static void cuda_free_v1(ark_gpu_device_ptr p) {
    if (!p) return;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);
    CU_CHECK(cuMemFree(ark_to_cu_device_ptr(p)));
}

static ark_gpu_status cuda_alloc_ex_v1(const ark_gpu_alloc_desc* desc, std::int64_t bytes, ark_gpu_device_ptr* out_p) {
    if (!out_p) return ARK_GPU_ERR_INVALID_ARG;
    *out_p = nullptr;
    if (!desc || desc->size != sizeof(ark_gpu_alloc_desc) || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUdeviceptr p = 0;

    switch (desc->kind) {
        case ARK_GPU_MEM_DEVICE: {
            CUresult r = cuMemAlloc(&p, static_cast<std::size_t>(bytes));
            if (r != CUDA_SUCCESS) return ARK_GPU_ERR_OOM;
            break;
        }
        case ARK_GPU_MEM_MANAGED: {
            if (!s.supports_managed) return ARK_GPU_ERR_UNSUPPORTED;
            unsigned flags = CU_MEM_ATTACH_GLOBAL;
            CUresult r = cuMemAllocManaged(&p, static_cast<std::size_t>(bytes), flags);
            if (r != CUDA_SUCCESS) return ARK_GPU_ERR_OOM;
            break;
        }
        default:
            return ARK_GPU_ERR_UNSUPPORTED;
    }

    *out_p = ark_from_cu_device_ptr(p);
    return ARK_GPU_OK;
}

static ark_gpu_status cuda_host_alloc_v1(const ark_gpu_alloc_desc* desc, std::int64_t bytes, void** out_host) {
    if (!out_host) return ARK_GPU_ERR_INVALID_ARG;
    *out_host = nullptr;
    if (!desc || desc->size != sizeof(ark_gpu_alloc_desc) || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->kind != ARK_GPU_MEM_HOST_PINNED) return ARK_GPU_ERR_UNSUPPORTED;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    unsigned flags = CU_MEMHOSTALLOC_DEVICEMAP;
    void* p = nullptr;
    CUresult r = cuMemHostAlloc(&p, static_cast<std::size_t>(bytes), flags);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_OOM;

    *out_host = p;
    return ARK_GPU_OK;
}

static void cuda_host_free_v1(void* host) {
    if (!host) return;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);
    CU_CHECK(cuMemFreeHost(host));
}

static void cuda_memcpy_to_device_v1(ark_gpu_device_ptr dst, const void* src, std::int64_t bytes) {
    if (!dst || !src || bytes <= 0) return;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);
    CU_CHECK(cuMemcpyHtoD(ark_to_cu_device_ptr(dst), src, static_cast<std::size_t>(bytes)));
}

static void cuda_memcpy_to_host_v1(void* dst, ark_gpu_device_ptr src, std::int64_t bytes) {
    if (!dst || !src || bytes <= 0) return;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);
    CU_CHECK(cuMemcpyDtoH(dst, ark_to_cu_device_ptr(src), static_cast<std::size_t>(bytes)));
}

static ark_gpu_status cuda_memcpy_v1(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind) {
    if (!dst || !src || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUresult r = CUDA_SUCCESS;
    switch (kind) {
        case ARK_GPU_COPY_H2D:
            r = cuMemcpyHtoD(ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(dst)), src, static_cast<std::size_t>(bytes));
            break;
        case ARK_GPU_COPY_D2H:
            r = cuMemcpyDtoH(dst, ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(const_cast<void*>(src))), static_cast<std::size_t>(bytes));
            break;
        case ARK_GPU_COPY_D2D:
            r = cuMemcpyDtoD(ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(dst)),
                             ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(const_cast<void*>(src))),
                             static_cast<std::size_t>(bytes));
            break;
        default:
            return ARK_GPU_ERR_INVALID_ARG;
    }

    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status cuda_memcpy_async_v1(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind, ark_gpu_stream stream) {
    if (!dst || !src || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUstream st = ark_to_cu_stream(stream);
    CUresult r = CUDA_SUCCESS;

    switch (kind) {
        case ARK_GPU_COPY_H2D:
            r = cuMemcpyHtoDAsync(ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(dst)), src, static_cast<std::size_t>(bytes), st);
            break;
        case ARK_GPU_COPY_D2H:
            r = cuMemcpyDtoHAsync(dst, ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(const_cast<void*>(src))), static_cast<std::size_t>(bytes), st);
            break;
        case ARK_GPU_COPY_D2D:
            r = cuMemcpyDtoDAsync(ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(dst)),
                                  ark_to_cu_device_ptr(reinterpret_cast<ark_gpu_device_ptr>(const_cast<void*>(src))),
                                  static_cast<std::size_t>(bytes), st);
            break;
        default:
            return ARK_GPU_ERR_INVALID_ARG;
    }

    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status cuda_memset_async_v1(ark_gpu_device_ptr dst, std::int32_t value, std::int64_t bytes, ark_gpu_stream stream) {
    if (!dst || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUdeviceptr d = ark_to_cu_device_ptr(dst);
    CUstream st = ark_to_cu_stream(stream);

    if ((bytes % 4) == 0) {
        const unsigned v = static_cast<unsigned>(value) & 0xFFu;
        const unsigned vv = (v << 24) | (v << 16) | (v << 8) | v;
        CUresult r = cuMemsetD32Async(d, vv, static_cast<std::size_t>(bytes / 4), st);
        return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
    }

    CUresult r = cuMemsetD8Async(d, static_cast<unsigned char>(value), static_cast<std::size_t>(bytes), st);
    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

// ---- Streams ----

static ark_gpu_status cuda_stream_create_v1(const ark_gpu_stream_desc* desc, ark_gpu_stream* out_stream) {
    if (!out_stream) return ARK_GPU_ERR_INVALID_ARG;

    bool non_blocking = true;
    int priority = 0;
    if (desc) {
        if (desc->size != sizeof(ark_gpu_stream_desc)) return ARK_GPU_ERR_INVALID_ARG;
        non_blocking = (desc->flags & ARK_GPU_STREAM_NON_BLOCKING) != 0;
        priority = desc->priority;
    }

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    unsigned flags = non_blocking ? CU_STREAM_NON_BLOCKING : CU_STREAM_DEFAULT;
    int prio = 0;
    if (priority != 0) prio = (priority > 0) ? s.prio_high : s.prio_low;

    CUstream st = nullptr;
    CUresult r = cuStreamCreateWithPriority(&st, flags, prio);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_DRIVER;

    *out_stream = ark_from_cu_stream(st);
    return ARK_GPU_OK;
}

static void cuda_stream_destroy_v1(ark_gpu_stream stream) {
    CUstream st = ark_to_cu_stream(stream);
    if (!st) return;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);
    CU_CHECK(cuStreamDestroy(st));
}

static ark_gpu_status cuda_stream_synchronize_v1(ark_gpu_stream stream) {
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUstream st = ark_to_cu_stream(stream);
    CUresult r = st ? cuStreamSynchronize(st) : cuCtxSynchronize();
    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

// ---- Events ----

static ark_gpu_status cuda_event_create_v1(const ark_gpu_event_desc* desc, ark_gpu_event* out_evt) {
    if (!out_evt) return ARK_GPU_ERR_INVALID_ARG;

    bool timing = true;
    if (desc) {
        if (desc->size != sizeof(ark_gpu_event_desc)) return ARK_GPU_ERR_INVALID_ARG;
        timing = (desc->flags & ARK_GPU_EVENT_DISABLE_TIMING) == 0;
    }

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    unsigned flags = timing ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING;
    CUevent e = nullptr;
    CUresult r = cuEventCreate(&e, flags);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_DRIVER;

    *out_evt = ark_from_cu_event(e);
    return ARK_GPU_OK;
}

static void cuda_event_destroy_v1(ark_gpu_event evt) {
    CUevent e = ark_to_cu_event(evt);
    if (!e) return;
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);
    CU_CHECK(cuEventDestroy(e));
}

static ark_gpu_status cuda_event_record_v1(ark_gpu_event evt, ark_gpu_stream stream) {
    CUevent e = ark_to_cu_event(evt);
    if (!e) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUresult r = cuEventRecord(e, ark_to_cu_stream(stream));
    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status cuda_event_synchronize_v1(ark_gpu_event evt) {
    CUevent e = ark_to_cu_event(evt);
    if (!e) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUresult r = cuEventSynchronize(e);
    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

static ark_gpu_status cuda_event_elapsed_ms_v1(ark_gpu_event start_evt, ark_gpu_event end_evt, float* out_ms) {
    if (!out_ms) return ARK_GPU_ERR_INVALID_ARG;
    CUevent a = ark_to_cu_event(start_evt);
    CUevent b = ark_to_cu_event(end_evt);
    if (!a || !b) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    float ms = 0.0f;
    CUresult r = cuEventElapsedTime(&ms, a, b);
    if (r != CUDA_SUCCESS) return ARK_GPU_ERR_DRIVER;
    *out_ms = ms;
    return ARK_GPU_OK;
}

static ark_gpu_status cuda_stream_wait_event_v1(ark_gpu_stream stream, ark_gpu_event evt) {
    CUevent e = ark_to_cu_event(evt);
    if (!e) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUresult r = cuStreamWaitEvent(ark_to_cu_stream(stream), e, 0u);
    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

// ---- Modules / JIT / RTC ----

static ark_gpu_status cuda_module_load_v1(const ark_gpu_module_desc* desc, ark_gpu_module* out_mod) {
    if (!desc || !out_mod) return ARK_GPU_ERR_INVALID_ARG;
    if (desc->size != sizeof(ark_gpu_module_desc)) return ARK_GPU_ERR_INVALID_ARG;
    if (!desc->module_key || !desc->module_key[0]) return ARK_GPU_ERR_INVALID_ARG;
    if (!desc->bytes || desc->byte_len == 0) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    const std::string key(desc->module_key);

    {
        std::lock_guard<std::mutex> lk(s.mu_modules);

        auto it = s.modules.find(key);
        if (it != s.modules.end()) {
            it->second.refcnt.fetch_add(1, std::memory_order_relaxed);
            *out_mod = reinterpret_cast<ark_gpu_module>(it->second.module);

            if (s.default_module == nullptr) {
                s.default_module_key = key;
                s.default_module = it->second.module;
            }
            return ARK_GPU_OK;
        }
    }

    CUmodule mod = nullptr;
    {
        const CUresult r = cuModuleLoadData(&mod, desc->bytes);
        if (r != CUDA_SUCCESS || mod == nullptr) {
            set_last_error("module_load: cuModuleLoadData failed");
            return ARK_GPU_ERR_COMPILATION;
        }
    }

    {
        std::lock_guard<std::mutex> lk(s.mu_modules);

        ModuleEntry e{};
        e.module = mod;
        e.refcnt.store(1, std::memory_order_relaxed);

        s.modules.emplace(key, std::move(e));
        *out_mod = reinterpret_cast<ark_gpu_module>(mod);

        if (s.default_module == nullptr) {
            s.default_module_key = key;
            s.default_module = mod;
        }
    }

    return ARK_GPU_OK;
}


static void cuda_module_unload_v1(ark_gpu_module mod) {
    if (!mod) return;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    const CUmodule target = reinterpret_cast<CUmodule>(mod);

    std::lock_guard<std::mutex> lk(s.mu_modules);

    auto it = s.modules.end();
    for (auto i = s.modules.begin(); i != s.modules.end(); ++i) {
        if (i->second.module == target) { it = i; break; }
    }
    if (it == s.modules.end()) return;

    const std::string key = it->first;

    const std::uint32_t prev = it->second.refcnt.fetch_sub(1, std::memory_order_acq_rel);
    if (prev > 1) return;

    invalidate_kernel_cache_for_module_locked(s, key);

    const bool was_default = (!s.default_module_key.empty() && s.default_module_key == key);

    CUmodule m = it->second.module;
    s.modules.erase(it);

    if (m) CU_CHECK(cuModuleUnload(m));

    if (was_default) {
        s.default_module_key.clear();
        s.default_module = nullptr;

        if (!s.modules.empty()) {
            auto pick = s.modules.begin(); 
            s.default_module_key = pick->first;
            s.default_module     = pick->second.module;
        }
    }
}

static ark_gpu_status cuda_module_set_default_v1(ark_gpu_module mod) {
    if (!mod) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    const CUmodule target = reinterpret_cast<CUmodule>(mod);

    std::lock_guard<std::mutex> lk(s.mu_modules);
    for (auto& kv : s.modules) {
        if (kv.second.module == target) {
            s.default_module_key = kv.first;
            s.default_module     = target;
            return ARK_GPU_OK;
        }
    }

    return ARK_GPU_ERR_NOT_FOUND;
}

static ark_gpu_status cuda_rtc_compile_v1(const ark_gpu_rtc_desc*, ark_gpu_module*) {
    set_last_error("RTC compile unsupported (no NVRTC linked)");
    return ARK_GPU_ERR_UNSUPPORTED;
}

static const char* cuda_rtc_last_log_v1(const char*) {
    return "";
}

static ark_gpu_status cuda_kernel_lookup_v1(ark_gpu_module mod, const char* kernel_name, ark_gpu_kernel* out_kernel) {
    if (!kernel_name || !kernel_name[0] || !out_kernel) return ARK_GPU_ERR_INVALID_ARG;

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUfunction fn = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.mu_modules);

        if (mod) {
            CUmodule m = reinterpret_cast<CUmodule>(mod);
            if (cuModuleGetFunction(&fn, m, kernel_name) != CUDA_SUCCESS) return ARK_GPU_ERR_NOT_FOUND;
        } else {
            fn = resolve_function_locked(s, std::string{}, std::string(kernel_name));
            if (!fn) return ARK_GPU_ERR_NOT_FOUND;
        }
    }

    *out_kernel = reinterpret_cast<ark_gpu_kernel>(fn);
    return ARK_GPU_OK;
}

static ark_gpu_status cuda_kernel_lookup_default_v1(const char* kernel_name, ark_gpu_kernel* out_kernel) {
    return cuda_kernel_lookup_v1(nullptr, kernel_name, out_kernel);
}

// ---- Launch ----

static inline void report_kernel_not_found(const char* kernel_name) {
    if (!kernel_name) kernel_name = "(null)";
    std::string msg;
    msg.reserve(256);
    msg += "launch: kernel not found: ";
    msg += kernel_name;
    set_last_error(msg);
    log_msg(ARK_GPU_LOG_ERROR, msg.c_str());
}

static void cuda_launch_v1(const char* kernel_name,
                          void** args, std::int32_t arg_count,
                          std::int32_t gx, std::int32_t gy, std::int32_t gz,
                          std::int32_t bx, std::int32_t by, std::int32_t bz,
                          ark_gpu_stream stream) {
    if (!kernel_name || !kernel_name[0]) {
        set_last_error("launch: kernel_name is null/empty");
        log_msg(ARK_GPU_LOG_ERROR, "launch: kernel_name is null/empty");
        return;
    }

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    CUmodule mod = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.mu_modules);

        if (s.default_module == nullptr) {
            if (!s.default_module_key.empty()) {
                auto it = s.modules.find(s.default_module_key);
                if (it != s.modules.end()) {
                    s.default_module = it->second.module;
                }
            }

            if (s.default_module == nullptr) {
                if (!s.modules.empty()) {
                    auto pick = s.modules.begin(); 
                    s.default_module_key = pick->first;
                    s.default_module     = pick->second.module;
                }
            }
        }

        mod = s.default_module;
    }

    if (mod == nullptr) {
        set_last_error("launch: no default module set");
        log_msg(ARK_GPU_LOG_ERROR, "launch: no default module set");
        return;
    }

    CUfunction fn = nullptr;
    const CUresult r = cuModuleGetFunction(&fn, mod, kernel_name);
    if (r != CUDA_SUCCESS || fn == nullptr) {
        report_kernel_not_found(kernel_name);
        return;
    }

    if (gx <= 0 || gy <= 0 || gz <= 0 || bx <= 0 || by <= 0 || bz <= 0) {
        set_last_error("launch: invalid grid/block dims");
        log_msg(ARK_GPU_LOG_ERROR, "launch: invalid grid/block dims");
        return;
    }

    const CUresult lr = cuLaunchKernel(fn,
                                       gx, gy, gz,
                                       bx, by, bz,
                                       0,
                                       cu_stream,
                                       args,
                                       nullptr);
    if (lr != CUDA_SUCCESS) {
        set_last_error("cuLaunchKernel failed");
        log_msg(ARK_GPU_LOG_ERROR, "cuLaunchKernel failed");
        return;
    }
}

static ark_gpu_status cuda_launch_handle_v1(ark_gpu_kernel kernel,
                                           void** args, std::int32_t arg_count,
                                           std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                           std::int32_t bx, std::int32_t by, std::int32_t bz,
                                           ark_gpu_stream stream) {
    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);

    if (!s.initialized) { set_last_error("launch_handle: backend not initialized"); return ARK_GPU_ERR_NOT_READY; }
    set_current_ctx_locked(s);

    if (!kernel) { set_last_error("launch_handle: null kernel handle"); return ARK_GPU_ERR_INVALID_ARG; }

    CUfunction fn = reinterpret_cast<CUfunction>(kernel);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    if (gx <= 0 || gy <= 0 || gz <= 0 || bx <= 0 || by <= 0 || bz <= 0) {
        set_last_error("launch_handle: invalid grid/block dims");
        return ARK_GPU_ERR_INVALID_ARG;
    }

    const CUresult lr = cuLaunchKernel(fn,
                                       gx, gy, gz,
                                       bx, by, bz,
                                       0,
                                       cu_stream,
                                       args,
                                       nullptr);
    if (lr != CUDA_SUCCESS) {
        set_last_error("cuLaunchKernel failed");
        return ARK_GPU_ERR_DRIVER;
    }

    (void)arg_count;
    return ARK_GPU_OK;
}

static ark_gpu_status cuda_launch_ex_v1(const char* kernel_name,
                                        void** args, std::int32_t /*arg_count*/,
                                        std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                        std::int32_t bx, std::int32_t by, std::int32_t bz,
                                        const ark_gpu_launch_desc* desc,
                                        ark_gpu_stream stream) {
    if (!kernel_name || !kernel_name[0]) return ARK_GPU_ERR_INVALID_ARG;

    std::uint32_t shared = 0;
    if (desc) {
        if (desc->size != sizeof(ark_gpu_launch_desc)) return ARK_GPU_ERR_INVALID_ARG;
        shared = desc->shared_mem_bytes;
    }

    CudaState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    ensure_current_locked(s);

    CUfunction fn = nullptr;

    const std::string_view full(kernel_name);
    if (ark_sv_starts_with(full, "cu:function:")) {
        std::uint64_t v = 0;
        if (!ark_parse_hex_u64(full.substr(std::strlen("cu:function:")), v)) return ARK_GPU_ERR_INVALID_ARG;
        fn = reinterpret_cast<CUfunction>(static_cast<std::uintptr_t>(v));
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

    CUresult r = cuLaunchKernel(fn,
                               gx, gy, gz,
                               bx, by, bz,
                               shared,
                               ark_to_cu_stream(stream),
                               args,
                               nullptr);

    return (r == CUDA_SUCCESS) ? ARK_GPU_OK : ARK_GPU_ERR_DRIVER;
}

// ---- Global sync ----

static void cuda_synchronize_v1(ark_gpu_stream stream) {
    CudaState& s = S();

    CUstream st = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        ensure_current_locked(s);
        st = ark_to_cu_stream(stream);
    }

    const CUresult r = st ? cuStreamSynchronize(st) : cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        std::string msg = "synchronize failed: ";
        msg += ark_cuda_err_name(r);
        msg += " (";
        msg += ark_cuda_err_str(r);
        msg += ")";
        set_last_error(msg);
        log_msg(ARK_GPU_LOG_ERROR, msg.c_str());
    }
}


static const char* cuda_last_error_string_v1() {
    // [FIX] Return c_str() of the persistent string object
    return S().last_error.c_str();
}

static std::uint64_t cuda_feature_bits_v1() {
    return (ARK_GPU_FEAT_STREAMS |
            ARK_GPU_FEAT_EVENTS |
            ARK_GPU_FEAT_ASYNC_MEMCPY |
            ARK_GPU_FEAT_ASYNC_MEMSET |
            ARK_GPU_FEAT_D2D_MEMCPY |
            ARK_GPU_FEAT_PINNED_HOST_ALLOC |
            ARK_GPU_FEAT_MANAGED_ALLOC |
            ARK_GPU_FEAT_MODULES |
            ARK_GPU_FEAT_KERNEL_HANDLES |
            ARK_GPU_FEAT_LAUNCH_EX |
            ARK_GPU_FEAT_DEVICE_QUERY |
            ARK_GPU_FEAT_MULTI_DEVICE);
}

} // namespace

// ------------------------------
// Plugin vtable export
// ------------------------------
extern "C" {

static constexpr std::uint64_t kCudaFeatures =
    ARK_GPU_FEAT_STREAMS |
    ARK_GPU_FEAT_EVENTS |
    ARK_GPU_FEAT_ASYNC_MEMCPY |
    ARK_GPU_FEAT_ASYNC_MEMSET |
    ARK_GPU_FEAT_D2D_MEMCPY |
    ARK_GPU_FEAT_PINNED_HOST_ALLOC |
    ARK_GPU_FEAT_MANAGED_ALLOC |
    ARK_GPU_FEAT_MODULES |
    ARK_GPU_FEAT_KERNEL_HANDLES |
    ARK_GPU_FEAT_LAUNCH_EX |
    ARK_GPU_FEAT_DEVICE_QUERY |
    ARK_GPU_FEAT_MULTI_DEVICE;

static const ark_gpu_backend_v1 g_vt = {
    .abi = ARK_GPU_BACKEND_ABI_V1,
    .size = sizeof(ark_gpu_backend_v1),

    .kind = ARK_GPU_BACKEND_CUDA,
    .name = "cuda",
    .features = kCudaFeatures,

    .init = cuda_init_v1,
    .shutdown = cuda_shutdown_v1,
    .enter_thread = cuda_enter_thread_v1,
    .leave_thread = cuda_leave_thread_v1,

    .device_count = cuda_device_count_v1,
    .set_device = cuda_set_device_v1,
    .get_device = cuda_get_device_v1,
    .get_device_info = cuda_get_device_info_v1,

    .alloc = cuda_alloc_v1,
    .free = cuda_free_v1,

    .alloc_ex = cuda_alloc_ex_v1,
    .host_alloc = cuda_host_alloc_v1,
    .host_free = cuda_host_free_v1,

    .memcpy_to_device = cuda_memcpy_to_device_v1,
    .memcpy_to_host = cuda_memcpy_to_host_v1,

    .memcpy = cuda_memcpy_v1,
    .memcpy_async = cuda_memcpy_async_v1,
    .memset_async = cuda_memset_async_v1,

    .stream_create = cuda_stream_create_v1,
    .stream_destroy = cuda_stream_destroy_v1,
    .stream_synchronize = cuda_stream_synchronize_v1,

    .event_create = cuda_event_create_v1,
    .event_destroy = cuda_event_destroy_v1,
    .event_record = cuda_event_record_v1,
    .event_synchronize = cuda_event_synchronize_v1,
    .event_elapsed_ms = cuda_event_elapsed_ms_v1,
    .stream_wait_event = cuda_stream_wait_event_v1,

    .module_load = cuda_module_load_v1,
    .module_unload = cuda_module_unload_v1,
    .module_set_default = cuda_module_set_default_v1,

    .rtc_compile = cuda_rtc_compile_v1,
    .rtc_last_log = cuda_rtc_last_log_v1,

    .kernel_lookup = cuda_kernel_lookup_v1,
    .kernel_lookup_default = cuda_kernel_lookup_default_v1,

    .launch = cuda_launch_v1,
    .launch_handle = cuda_launch_handle_v1,
    .launch_ex = cuda_launch_ex_v1,

    .synchronize = cuda_synchronize_v1,

    .last_error_string = cuda_last_error_string_v1,

    .reserved = { nullptr },
};

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1() {
    return &g_vt;
}

} // extern "C"

#else // ARK_BACKEND_CUDA

extern "C" {

// Disabled Stub
static const ark_gpu_backend_v1 g_vt_disabled = {
    ARK_GPU_BACKEND_ABI_V1,
    sizeof(ark_gpu_backend_v1),
    ARK_GPU_BACKEND_CUDA,
    "cuda(disabled)",
    0, // No features
    
    // All function pointers default to nullptr
    nullptr
};

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1() {
    return &g_vt_disabled;
}

} // extern "C"

#endif // ARK_BACKEND_CUDA