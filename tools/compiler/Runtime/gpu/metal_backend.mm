```objc
// tools/compiler/Runtime/gpu/metal_backend.mm
#include "gpu_backend.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(ARK_BACKEND_METAL)

  #include <TargetConditionals.h>
  #import <Foundation/Foundation.h>
  #import <Metal/Metal.h>

// ------------------------------
// ObjC retention (ARC + MRC safe)
// ------------------------------

#if __has_feature(objc_arc)
  #define ARK_OBJC_RELEASE(x) do { (void)(x); } while (0)
#else
  #define ARK_OBJC_RELEASE(x) do { if ((x) != nil) [(id)(x) release]; } while (0)
#endif

static inline void* ark_objc_retain(id obj) {
    return obj ? (void*)CFRetain((__bridge CFTypeRef)obj) : nullptr;
}

static inline void* ark_objc_retain_from_new(id obj) {
    if (!obj) return nullptr;
    void* p = (void*)CFRetain((__bridge CFTypeRef)obj);
    ARK_OBJC_RELEASE(obj);
    return p;
}

static inline void ark_objc_release(void* p) {
    if (p) CFRelease((CFTypeRef)p);
}

template <typename T>
static inline T ark_objc_get(void* p) {
    return (T)(__bridge id)p;
}

// ------------------------------
// Utilities
// ------------------------------

static inline std::pair<std::string_view, std::string_view> split_once(std::string_view s, std::string_view delim) {
    const std::size_t pos = s.find(delim);
    if (pos == std::string_view::npos) return {s, std::string_view{}};
    return {s.substr(0, pos), s.substr(pos + delim.size())};
}

static inline void set_cstr(char* out, std::size_t out_cap, const char* s) {
    if (!out || out_cap == 0) return;
    if (!s) { out[0] = '\0'; return; }
    std::snprintf(out, out_cap, "%s", s);
}

// ------------------------------
// Backend state + objects
// ------------------------------

namespace {

static thread_local std::string g_tls_last_error;

static inline void set_last_error(const char* s) {
    g_tls_last_error = (s ? s : "");
}

static inline void set_last_error(const std::string& s) {
    g_tls_last_error = s;
}

static inline const char* last_error_cstr() {
    return g_tls_last_error.empty() ? nullptr : g_tls_last_error.c_str();
}

static constexpr std::uint64_t ARK_METAL_BUF_MAGIC = 0x41524B4D4554414Cull;

struct MetalBuffer {
    std::uint64_t magic = ARK_METAL_BUF_MAGIC;
    void* mtl_buffer = nullptr;   // retained id<MTLBuffer>
    std::size_t offset = 0;
    std::size_t size = 0;
    bool is_private = true;

    id<MTLBuffer> buffer() const { return ark_objc_get<id<MTLBuffer>>(mtl_buffer); }
};

struct MetalStream {
    std::mutex mu;
    void* queue = nullptr;        // retained id<MTLCommandQueue>
    void* last_cb = nullptr;      // retained id<MTLCommandBuffer>
};

struct MetalEvent {
    void* cb = nullptr;           // retained id<MTLCommandBuffer>
};

struct LibraryEntry {
    void* lib = nullptr;          // retained id<MTLLibrary>
    std::string key;
    std::uint32_t refcnt = 1;
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

static inline MetalBuffer* as_buf(ark_gpu_device_ptr p) { return reinterpret_cast<MetalBuffer*>(p); }

static inline bool is_buf(ark_gpu_device_ptr p) {
    if (!p) return false;
    MetalBuffer* b = reinterpret_cast<MetalBuffer*>(p);
    return b->magic == ARK_METAL_BUF_MAGIC && b->mtl_buffer != nullptr;
}

class MetalBackend final {
public:
    ark_gpu_status init(std::int32_t device_id, const ark_gpu_init_params* params) {
        std::lock_guard<std::mutex> lock(mu_state_);

        logger_ = nullptr;
        logger_user_ = nullptr;
        if (params && params->size >= sizeof(ark_gpu_init_params)) {
            logger_ = params->logger;
            logger_user_ = params->logger_user;
        }

        shutdown_locked();

        @autoreleasepool {
            id<MTLDevice> dev = select_device(device_id);
            if (!dev) {
                set_last_error("no Metal device available");
                return ARK_GPU_ERR_NO_DEVICE;
            }

            device_ = ark_objc_retain(dev);
            device_id_ = device_id;

            id<MTLCommandQueue> q = [dev newCommandQueue];
            if (!q) {
                set_last_error("failed to create MTLCommandQueue");
                shutdown_locked();
                return ARK_GPU_ERR_DRIVER;
            }

            default_stream_.queue = ark_objc_retain_from_new(q);
            default_stream_.last_cb = nullptr;

            initialized_ = true;
            return ARK_GPU_OK;
        }
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mu_state_);
        shutdown_locked();
    }

    ark_gpu_backend_kind kind() const { return ARK_GPU_BACKEND_METAL; }
    const char* name() const { return "metal"; }

    std::uint32_t device_id() const { return device_id_; }

    ark_gpu_status device_count(std::uint32_t* out_count) {
        if (!out_count) return ARK_GPU_ERR_INVALID_ARG;
        @autoreleasepool {
#if TARGET_OS_OSX
            NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
            if (!devs) { *out_count = 0; return ARK_GPU_OK; }
            *out_count = (std::uint32_t)[devs count];
            ARK_OBJC_RELEASE(devs);
            return ARK_GPU_OK;
#else
            *out_count = 1;
            return ARK_GPU_OK;
#endif
        }
    }

    ark_gpu_status get_device_info(std::int32_t id, ark_gpu_device_info* out_info) {
        if (!out_info || out_info->size < sizeof(ark_gpu_device_info)) return ARK_GPU_ERR_INVALID_ARG;

        @autoreleasepool {
            id<MTLDevice> dev = select_device(id);
            if (!dev) return ARK_GPU_ERR_NO_DEVICE;

            std::memset(out_info, 0, sizeof(ark_gpu_device_info));
            out_info->size = (std::uint32_t)sizeof(ark_gpu_device_info);
            out_info->device_id = (std::uint32_t)id;
            out_info->backend = ARK_GPU_BACKEND_METAL;

            set_cstr(out_info->name, sizeof(out_info->name), [[dev name] UTF8String]);

            out_info->major = 0;
            out_info->minor = 0;

            out_info->global_mem_bytes = 0;
            out_info->warp_size = 32;

            out_info->sm_count = 0;
            out_info->max_threads_per_sm = 0;
            out_info->max_threads_per_block = 0;

            out_info->max_grid_dim_x = 0;
            out_info->max_grid_dim_y = 0;
            out_info->max_grid_dim_z = 0;

            out_info->max_block_dim_x = 0;
            out_info->max_block_dim_y = 0;
            out_info->max_block_dim_z = 0;

            out_info->supports_async_copy = 1;
            return ARK_GPU_OK;
        }
    }

    ark_gpu_status set_device(std::int32_t id) {
        return init(id, nullptr);
    }

    ark_gpu_status get_device(std::int32_t* out_id) const {
        if (!out_id) return ARK_GPU_ERR_INVALID_ARG;
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        *out_id = device_id_;
        return ARK_GPU_OK;
    }

    ark_gpu_device_ptr alloc(std::int64_t bytes) {
        ark_gpu_device_ptr p = nullptr;
        ark_gpu_alloc_desc d{};
        d.size = (std::uint32_t)sizeof(ark_gpu_alloc_desc);
        d.flags = 0;
        d.kind = ARK_GPU_MEM_DEVICE;
        d.alignment = 0;
        (void)alloc_ex(&d, bytes, &p);
        return p;
    }

    void free(ark_gpu_device_ptr p) {
        if (!p) return;
        if (!initialized_) return;

        MetalBuffer* b = as_buf(p);
        if (b->magic != ARK_METAL_BUF_MAGIC) {
            set_last_error("free: invalid buffer handle");
            std::abort();
        }

        ark_objc_release(b->mtl_buffer);
        b->mtl_buffer = nullptr;
        b->magic = 0;
        delete b;
    }

    ark_gpu_status alloc_ex(const ark_gpu_alloc_desc* desc, std::int64_t bytes, ark_gpu_device_ptr* out_p) {
        if (!out_p) return ARK_GPU_ERR_INVALID_ARG;
        *out_p = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!desc || desc->size < sizeof(ark_gpu_alloc_desc)) return ARK_GPU_ERR_INVALID_ARG;
        if (bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

        const bool want_shared =
            (desc->kind == ARK_GPU_MEM_SHARED) ||
            (desc->kind == ARK_GPU_MEM_HOST_MAPPED) ||
            (desc->kind == ARK_GPU_MEM_HOST_PINNED);

        @autoreleasepool {
            id<MTLDevice> dev = device();
            if (!dev) return ARK_GPU_ERR_NOT_READY;

            MTLResourceOptions opts = MTLResourceCPUCacheModeDefaultCache |
                                     (want_shared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate);

            id<MTLBuffer> buf = [dev newBufferWithLength:(NSUInteger)bytes options:opts];
            if (!buf) {
                set_last_error(want_shared ? "MTLBuffer allocation failed (shared)" : "MTLBuffer allocation failed (private)");
                return ARK_GPU_ERR_OOM;
            }

            MetalBuffer* h = new MetalBuffer();
            h->mtl_buffer = ark_objc_retain_from_new(buf);
            h->offset = 0;
            h->size = (std::size_t)bytes;
            h->is_private = !want_shared;

            *out_p = reinterpret_cast<ark_gpu_device_ptr>(h);
            return ARK_GPU_OK;
        }
    }

    void memcpy_to_device(ark_gpu_device_ptr dst, const void* src, std::int64_t bytes) {
        memcpy_h2d_impl(dst, src, bytes, nullptr, false);
    }

    void memcpy_to_host(void* dst, ark_gpu_device_ptr src, std::int64_t bytes) {
        memcpy_d2h_impl(dst, src, bytes, nullptr, false);
    }

    ark_gpu_status memcpy_async(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind, ark_gpu_stream stream) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!dst || !src || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

        switch (kind) {
            case ARK_GPU_COPY_H2D:
                memcpy_h2d_impl(reinterpret_cast<ark_gpu_device_ptr>(dst), src, bytes, stream, true);
                return ARK_GPU_OK;
            case ARK_GPU_COPY_D2H:
                memcpy_d2h_impl(dst, reinterpret_cast<ark_gpu_device_ptr>(const_cast<void*>(src)), bytes, stream, true);
                return ARK_GPU_OK;
            case ARK_GPU_COPY_D2D:
                memcpy_d2d_impl(reinterpret_cast<ark_gpu_device_ptr>(dst),
                                reinterpret_cast<ark_gpu_device_ptr>(const_cast<void*>(src)),
                                bytes, stream, true);
                return ARK_GPU_OK;
            default:
                return ARK_GPU_ERR_INVALID_ARG;
        }
    }

    ark_gpu_status memset_async(ark_gpu_device_ptr dst, std::int32_t value, std::int64_t bytes, ark_gpu_stream stream) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!dst || bytes <= 0) return ARK_GPU_ERR_INVALID_ARG;

        MetalBuffer* d = as_buf(dst);
        if (d->magic != ARK_METAL_BUF_MAGIC) { set_last_error("memset_async: invalid dst handle"); std::abort(); }
        if ((std::size_t)bytes > d->size) return ARK_GPU_ERR_INVALID_ARG;

        @autoreleasepool {
            if (!d->is_private) {
                std::memset((std::uint8_t*)[d->buffer() contents] + d->offset, value, (std::size_t)bytes);
                return ARK_GPU_OK;
            }

            MetalStream* st = resolve_stream(stream);
            id<MTLCommandBuffer> cb = make_cb(st);
            if (!cb) return ARK_GPU_ERR_DRIVER;

            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            if (!blit) return ARK_GPU_ERR_DRIVER;

            NSRange r;
            r.location = (NSUInteger)d->offset;
            r.length = (NSUInteger)bytes;
            [blit fillBuffer:d->buffer() range:r value:(uint8_t)value];

            [blit endEncoding];
            [cb commit];
            commit_track(st, cb);
            return ARK_GPU_OK;
        }
    }

    ark_gpu_status stream_create(const ark_gpu_stream_desc* desc, ark_gpu_stream* out_stream) {
        if (!out_stream) return ARK_GPU_ERR_INVALID_ARG;
        *out_stream = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!desc || desc->size < sizeof(ark_gpu_stream_desc)) return ARK_GPU_ERR_INVALID_ARG;

        @autoreleasepool {
            id<MTLCommandQueue> q = [device() newCommandQueue];
            if (!q) return ARK_GPU_ERR_DRIVER;

            MetalStream* s = new MetalStream();
            s->queue = ark_objc_retain_from_new(q);
            s->last_cb = nullptr;

            *out_stream = reinterpret_cast<ark_gpu_stream>(s);
            return ARK_GPU_OK;
        }
    }

    void stream_destroy(ark_gpu_stream stream) {
        if (!stream) return;
        MetalStream* s = reinterpret_cast<MetalStream*>(stream);
        
        // Release contents first
        {
            std::lock_guard<std::mutex> lock(s->mu);
            ark_objc_release(s->last_cb);
            ark_objc_release(s->queue);
        }
        // Delete container after lock is released
        delete s;
    }

    ark_gpu_status stream_synchronize(ark_gpu_stream stream) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;

        MetalStream* s = resolve_stream(stream);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = last_cb(s);
            if (cb) {
                [cb waitUntilCompleted];
                return ARK_GPU_OK;
            }

            id<MTLCommandBuffer> fence = [queue(s) commandBuffer];
            if (!fence) return ARK_GPU_ERR_DRIVER;

            [fence commit];
            [fence waitUntilCompleted];
            commit_track(s, fence);
            return ARK_GPU_OK;
        }
    }

    ark_gpu_status event_create(const ark_gpu_event_desc* desc, ark_gpu_event* out_evt) {
        if (!out_evt) return ARK_GPU_ERR_INVALID_ARG;
        *out_evt = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!desc || desc->size < sizeof(ark_gpu_event_desc)) return ARK_GPU_ERR_INVALID_ARG;

        MetalEvent* e = new MetalEvent();
        e->cb = nullptr;
        *out_evt = reinterpret_cast<ark_gpu_event>(e);
        return ARK_GPU_OK;
    }

    void event_destroy(ark_gpu_event evt) {
        if (!initialized_) return;
        if (!evt) return;

        MetalEvent* e = reinterpret_cast<MetalEvent*>(evt);
        ark_objc_release(e->cb);
        e->cb = nullptr;
        delete e;
    }

    ark_gpu_status event_record(ark_gpu_event evt, ark_gpu_stream stream) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!evt) return ARK_GPU_ERR_INVALID_ARG;

        MetalEvent* e = reinterpret_cast<MetalEvent*>(evt);
        MetalStream* s = resolve_stream(stream);

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue(s) commandBuffer];
            if (!cb) return ARK_GPU_ERR_DRIVER;

            [cb commit];

            ark_objc_release(e->cb);
            e->cb = ark_objc_retain(cb);

            commit_track(s, cb);
            return ARK_GPU_OK;
        }
    }

    ark_gpu_status event_synchronize(ark_gpu_event evt) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!evt) return ARK_GPU_ERR_INVALID_ARG;

        MetalEvent* e = reinterpret_cast<MetalEvent*>(evt);
        if (!e->cb) return ARK_GPU_ERR_NOT_READY;

        @autoreleasepool {
            [ark_objc_get<id<MTLCommandBuffer>>(e->cb) waitUntilCompleted];
            return ARK_GPU_OK;
        }
    }

    ark_gpu_status event_elapsed_ms(ark_gpu_event start_evt, ark_gpu_event end_evt, float* out_ms) {
        if (!out_ms) return ARK_GPU_ERR_INVALID_ARG;
        *out_ms = 0.0f;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!start_evt || !end_evt) return ARK_GPU_ERR_INVALID_ARG;

        MetalEvent* a = reinterpret_cast<MetalEvent*>(start_evt);
        MetalEvent* b = reinterpret_cast<MetalEvent*>(end_evt);
        if (!a->cb || !b->cb) return ARK_GPU_ERR_NOT_READY;

        @autoreleasepool {
            id<MTLCommandBuffer> ca = ark_objc_get<id<MTLCommandBuffer>>(a->cb);
            id<MTLCommandBuffer> cb = ark_objc_get<id<MTLCommandBuffer>>(b->cb);

            [ca waitUntilCompleted];
            [cb waitUntilCompleted];

            if ([ca respondsToSelector:@selector(GPUStartTime)] &&
                [cb respondsToSelector:@selector(GPUEndTime)]) {
                const double t0 = ca.GPUStartTime;
                const double t1 = cb.GPUEndTime;
                if (t1 >= t0) {
                    *out_ms = (float)((t1 - t0) * 1000.0);
                    return ARK_GPU_OK;
                }
            }

            *out_ms = 0.0f;
            return ARK_GPU_OK;
        }
    }

    ark_gpu_status module_load(const ark_gpu_module_desc* desc, ark_gpu_module* out_mod) {
        if (!out_mod) return ARK_GPU_ERR_INVALID_ARG;
        *out_mod = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!desc || desc->size < sizeof(ark_gpu_module_desc)) return ARK_GPU_ERR_INVALID_ARG;
        if (!desc->module_key || desc->module_key[0] == '\0') return ARK_GPU_ERR_INVALID_ARG;

        const std::string key(desc->module_key);

        if (desc->kind != ARK_GPU_MODULE_METAL_LIB) return ARK_GPU_ERR_UNSUPPORTED;
        if (!desc->bytes || desc->byte_len == 0) return ARK_GPU_ERR_INVALID_ARG;

        std::lock_guard<std::mutex> lock(mu_libs_);
        auto it = libs_.find(key);
        if (it != libs_.end()) {
            it->second.refcnt += 1;
            *out_mod = reinterpret_cast<ark_gpu_module>(store_key_handle(key));
            return ARK_GPU_OK;
        }

        @autoreleasepool {
            NSData* data = [NSData dataWithBytes:desc->bytes length:(NSUInteger)desc->byte_len];
            NSError* err = nil;
            id<MTLLibrary> lib = [device() newLibraryWithData:data error:&err];
            if (!lib) {
                set_last_error(err ? [[err localizedDescription] UTF8String] : "newLibraryWithData failed");
                return ARK_GPU_ERR_COMPILATION;
            }

            LibraryEntry ent;
            ent.key = key;
            ent.refcnt = 1;
            ent.lib = ark_objc_retain_from_new(lib);

            libs_.emplace(key, std::move(ent));
            if (default_lib_key_.empty()) default_lib_key_ = key;

            invalidate_pipelines_locked();
            *out_mod = reinterpret_cast<ark_gpu_module>(store_key_handle(key));
            return ARK_GPU_OK;
        }
    }

    void module_unload(ark_gpu_module mod) {
        if (!initialized_) return;
        if (!mod) return;

        const std::string key = consume_key_handle(mod);

        std::lock_guard<std::mutex> lock(mu_libs_);
        auto it = libs_.find(key);
        if (it == libs_.end()) return;

        if (--it->second.refcnt > 0) return;

        invalidate_pipelines_locked();

        ark_objc_release(it->second.lib);
        it->second.lib = nullptr;
        libs_.erase(it);

        if (default_lib_key_ == key) {
            default_lib_key_.clear();
            if (!libs_.empty()) default_lib_key_ = libs_.begin()->first;
        }
    }

    ark_gpu_status module_set_default(ark_gpu_module mod) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!mod) return ARK_GPU_ERR_INVALID_ARG;

        const std::string key = peek_key_handle(mod);

        std::lock_guard<std::mutex> lock(mu_libs_);
        if (libs_.find(key) == libs_.end()) return ARK_GPU_ERR_NOT_FOUND;
        default_lib_key_ = key;
        return ARK_GPU_OK;
    }

    ark_gpu_status rtc_compile(const ark_gpu_rtc_desc* desc, ark_gpu_module* out_mod) {
        if (!out_mod) return ARK_GPU_ERR_INVALID_ARG;
        *out_mod = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!desc || desc->size < sizeof(ark_gpu_rtc_desc)) return ARK_GPU_ERR_INVALID_ARG;
        if (!desc->module_key || desc->module_key[0] == '\0') return ARK_GPU_ERR_INVALID_ARG;
        if (!desc->source_utf8 || desc->source_utf8[0] == '\0') return ARK_GPU_ERR_INVALID_ARG;
        if (desc->kind != ARK_GPU_MODULE_SRC_MSL) return ARK_GPU_ERR_UNSUPPORTED;

        const std::string key(desc->module_key);

        std::lock_guard<std::mutex> lock(mu_libs_);
        auto it = libs_.find(key);
        if (it != libs_.end()) {
            it->second.refcnt += 1;
            *out_mod = reinterpret_cast<ark_gpu_module>(store_key_handle(key));
            return ARK_GPU_OK;
        }

        @autoreleasepool {
            NSString* src = [NSString stringWithUTF8String:desc->source_utf8];
            if (!src) return ARK_GPU_ERR_INVALID_ARG;

            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;

            NSError* err = nil;
            id<MTLLibrary> lib = [device() newLibraryWithSource:src options:opts error:&err];
            ARK_OBJC_RELEASE(opts);

            if (!lib) {
                set_last_error(err ? [[err localizedDescription] UTF8String] : "newLibraryWithSource failed");
                return ARK_GPU_ERR_COMPILATION;
            }

            LibraryEntry ent;
            ent.key = key;
            ent.refcnt = 1;
            ent.lib = ark_objc_retain_from_new(lib);

            libs_.emplace(key, std::move(ent));
            if (default_lib_key_.empty()) default_lib_key_ = key;

            invalidate_pipelines_locked();
            *out_mod = reinterpret_cast<ark_gpu_module>(store_key_handle(key));
            return ARK_GPU_OK;
        }
    }

    const char* rtc_last_log(const char*) {
        return last_error_cstr();
    }

    ark_gpu_status kernel_lookup(ark_gpu_module mod, const char* kernel_name, ark_gpu_kernel* out_kernel) {
        if (!out_kernel) return ARK_GPU_ERR_INVALID_ARG;
        *out_kernel = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!kernel_name || kernel_name[0] == '\0') return ARK_GPU_ERR_INVALID_ARG;
        if (!mod) return ARK_GPU_ERR_INVALID_ARG;

        const std::string key = peek_key_handle(mod);
        void* pso = resolve_pipeline(key, kernel_name);
        if (!pso) return ARK_GPU_ERR_NOT_FOUND;

        *out_kernel = reinterpret_cast<ark_gpu_kernel>(pso);
        return ARK_GPU_OK;
    }

    ark_gpu_status kernel_lookup_default(const char* kernel_name, ark_gpu_kernel* out_kernel) {
        if (!out_kernel) return ARK_GPU_ERR_INVALID_ARG;
        *out_kernel = nullptr;

        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!kernel_name || kernel_name[0] == '\0') return ARK_GPU_ERR_INVALID_ARG;

        std::string key;
        {
            std::lock_guard<std::mutex> lock(mu_libs_);
            key = default_lib_key_.empty() && !libs_.empty() ? libs_.begin()->first : default_lib_key_;
        }
        if (key.empty()) return ARK_GPU_ERR_NOT_FOUND;

        void* pso = resolve_pipeline(key, kernel_name);
        if (!pso) return ARK_GPU_ERR_NOT_FOUND;

        *out_kernel = reinterpret_cast<ark_gpu_kernel>(pso);
        return ARK_GPU_OK;
    }

    void launch(const char* kernel_name,
                void** args, std::int32_t arg_count,
                std::int32_t gx, std::int32_t gy, std::int32_t gz,
                std::int32_t bx, std::int32_t by, std::int32_t bz,
                ark_gpu_stream stream) {
        launch_ex(kernel_name, args, arg_count, gx, gy, gz, bx, by, bz, nullptr, stream);
    }

    ark_gpu_status launch_handle(ark_gpu_kernel kernel,
                                 void** args, std::int32_t arg_count,
                                 std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                 std::int32_t bx, std::int32_t by, std::int32_t bz,
                                 ark_gpu_stream stream) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;
        if (!kernel) return ARK_GPU_ERR_INVALID_ARG;

        id<MTLComputePipelineState> pso = ark_objc_get<id<MTLComputePipelineState>>(kernel);
        if (!pso) return ARK_GPU_ERR_INVALID_ARG;

        return launch_with_pso(pso, args, arg_count, gx, gy, gz, bx, by, bz, 0, 0, stream);
    }

    ark_gpu_status launch_ex(const char* kernel_name,
                             void** args, std::int32_t arg_count,
                             std::int32_t gx, std::int32_t gy, std::int32_t gz,
                             std::int32_t bx, std::int32_t by, std::int32_t bz,
                             const ark_gpu_launch_desc* desc,
                             ark_gpu_stream stream) {
        if (!initialized_) return ARK_GPU_ERR_NOT_READY;

        if (!kernel_name || kernel_name[0] == '\0') { set_last_error("launch: empty kernel_name"); std::abort(); }
        if (!args && arg_count != 0) { set_last_error("launch: args null with nonzero arg_count"); std::abort(); }
        if (gx <= 0 || gy <= 0 || gz <= 0 || bx <= 0 || by <= 0 || bz <= 0) { set_last_error("launch: invalid grid/block"); std::abort(); }

        const std::uint32_t tg_mem_bytes = desc ? desc->tg_mem_bytes : 0;
        const std::uint32_t tg_mem_index = desc ? desc->tg_mem_index : 0;

        std::string mod_key;
        std::string fn_name;
        {
            const std::string_view full(kernel_name);
            auto [a, b] = split_once(full, "::");
            if (!b.empty()) {
                mod_key.assign(a.data(), a.size());
                fn_name.assign(b.data(), b.size());
            } else {
                fn_name.assign(a.data(), a.size());
            }
        }

        std::string lib_key;
        {
            std::lock_guard<std::mutex> lock(mu_libs_);
            if (!mod_key.empty()) lib_key = mod_key;
            else lib_key = default_lib_key_.empty() && !libs_.empty() ? libs_.begin()->first : default_lib_key_;
        }

        if (lib_key.empty()) return ARK_GPU_ERR_NOT_FOUND;

        void* stored = resolve_pipeline(lib_key, fn_name.c_str());
        if (!stored) return ARK_GPU_ERR_NOT_FOUND;

        id<MTLComputePipelineState> pso = ark_objc_get<id<MTLComputePipelineState>>(stored);
        return launch_with_pso(pso, args, arg_count, gx, gy, gz, bx, by, bz, tg_mem_bytes, tg_mem_index, stream);
    }

    void synchronize(ark_gpu_stream stream) {
        (void)stream_synchronize(stream);
    }

    const char* last_error_string() const {
        return last_error_cstr();
    }

    void enter_thread() {}
    void leave_thread() {}

private:
    id<MTLDevice> device() const { return ark_objc_get<id<MTLDevice>>(device_); }

    static id<MTLDevice> select_device(std::int32_t device_id) {
        @autoreleasepool {
#if TARGET_OS_OSX
            NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
            if (!devs || [devs count] == 0) {
                if (devs) ARK_OBJC_RELEASE(devs);
                return nil;
            }
            const NSInteger idx = (device_id >= 0) ? (NSInteger)device_id : 0;
            id<MTLDevice> out = (idx >= 0 && idx < (NSInteger)[devs count]) ? devs[(NSUInteger)idx] : devs[0];
            ARK_OBJC_RELEASE(devs);
            return out;
#else
            (void)device_id;
            return MTLCreateSystemDefaultDevice();
#endif
        }
    }

    id<MTLCommandQueue> queue(MetalStream* s) const { return ark_objc_get<id<MTLCommandQueue>>(s->queue); }

    id<MTLCommandBuffer> last_cb(MetalStream* s) const {
        return s->last_cb ? ark_objc_get<id<MTLCommandBuffer>>(s->last_cb) : nil;
    }

    void commit_track(MetalStream* s, id<MTLCommandBuffer> cb) {
        std::lock_guard<std::mutex> lock(s->mu);
        ark_objc_release(s->last_cb);
        s->last_cb = ark_objc_retain(cb);
    }

    MetalStream* resolve_stream(ark_gpu_stream h) {
        if (!h) return &default_stream_;
        return reinterpret_cast<MetalStream*>(h);
    }

    id<MTLCommandBuffer> make_cb(MetalStream* s) {
        id<MTLCommandQueue> q = queue(s);
        if (!q) return nil;
        return [q commandBuffer];
    }

    id<MTLBuffer> make_staging_shared(std::size_t bytes) {
        id<MTLBuffer> b = [device() newBufferWithLength:(NSUInteger)bytes
                                               options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
        return b;
    }

    void memcpy_h2d_impl(ark_gpu_device_ptr dst, const void* src, std::int64_t bytes, ark_gpu_stream stream, bool async) {
        if (!initialized_) return;
        if (!dst || !src || bytes <= 0) return;

        MetalBuffer* d = as_buf(dst);
        if (d->magic != ARK_METAL_BUF_MAGIC) { set_last_error("memcpy_h2d: invalid dst handle"); std::abort(); }
        if ((std::size_t)bytes > d->size) { set_last_error("memcpy_h2d: size exceeds dst"); std::abort(); }

        @autoreleasepool {
            if (!d->is_private) {
                std::memcpy((std::uint8_t*)[d->buffer() contents] + d->offset, src, (std::size_t)bytes);
                return;
            }

            id<MTLBuffer> staging = make_staging_shared((std::size_t)bytes);
            if (!staging) { set_last_error("memcpy_h2d: staging allocation failed"); return; }

            std::memcpy([staging contents], src, (std::size_t)bytes);

            MetalStream* st = resolve_stream(stream);
            id<MTLCommandBuffer> cb = make_cb(st);
            if (!cb) { set_last_error("memcpy_h2d: commandBuffer creation failed"); ARK_OBJC_RELEASE(staging); return; }

            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            if (!blit) { set_last_error("memcpy_h2d: blit encoder creation failed"); ARK_OBJC_RELEASE(staging); return; }

            [blit copyFromBuffer:staging sourceOffset:0
                        toBuffer:d->buffer() destinationOffset:(NSUInteger)d->offset
                            size:(NSUInteger)bytes];

            [blit endEncoding];
            [cb commit];
            commit_track(st, cb);

            if (!async) [cb waitUntilCompleted];

            ARK_OBJC_RELEASE(staging);
        }
    }

    void memcpy_d2h_impl(void* dst, ark_gpu_device_ptr src, std::int64_t bytes, ark_gpu_stream stream, bool async) {
        if (!initialized_) return;
        if (!dst || !src || bytes <= 0) return;

        MetalBuffer* sdev = as_buf(src);
        if (sdev->magic != ARK_METAL_BUF_MAGIC) { set_last_error("memcpy_d2h: invalid src handle"); std::abort(); }
        if ((std::size_t)bytes > sdev->size) { set_last_error("memcpy_d2h: size exceeds src"); std::abort(); }

        @autoreleasepool {
            if (!sdev->is_private) {
                std::memcpy(dst, (std::uint8_t*)[sdev->buffer() contents] + sdev->offset, (std::size_t)bytes);
                return;
            }

            id<MTLBuffer> staging = make_staging_shared((std::size_t)bytes);
            if (!staging) { set_last_error("memcpy_d2h: staging allocation failed"); return; }

            MetalStream* st = resolve_stream(stream);
            id<MTLCommandBuffer> cb = make_cb(st);
            if (!cb) { set_last_error("memcpy_d2h: commandBuffer creation failed"); ARK_OBJC_RELEASE(staging); return; }

            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            if (!blit) { set_last_error("memcpy_d2h: blit encoder creation failed"); ARK_OBJC_RELEASE(staging); return; }

            [blit copyFromBuffer:sdev->buffer() sourceOffset:(NSUInteger)sdev->offset
                        toBuffer:staging destinationOffset:0
                            size:(NSUInteger)bytes];

            [blit endEncoding];

            if (async) {
                struct Task {
                    void* dst;
                    void* staging_retained;
                    std::size_t n;
                };

                Task* t = new Task();
                t->dst = dst;
                t->staging_retained = ark_objc_retain_from_new(staging);
                t->n = (std::size_t)bytes;

                [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                    @autoreleasepool {
                        id<MTLBuffer> sb = ark_objc_get<id<MTLBuffer>>(t->staging_retained);
                        std::memcpy(t->dst, [sb contents], t->n);
                        ark_objc_release(t->staging_retained);
                        delete t;
                    }
                }];

                [cb commit];
                commit_track(st, cb);
                return;
            }

            [cb commit];
            commit_track(st, cb);
            [cb waitUntilCompleted];

            std::memcpy(dst, [staging contents], (std::size_t)bytes);
            ARK_OBJC_RELEASE(staging);
        }
    }

    void memcpy_d2d_impl(ark_gpu_device_ptr dst, ark_gpu_device_ptr src, std::int64_t bytes, ark_gpu_stream stream, bool async) {
        if (!initialized_) return;
        if (!dst || !src || bytes <= 0) return;

        MetalBuffer* d = as_buf(dst);
        MetalBuffer* sdev = as_buf(src);

        if (d->magic != ARK_METAL_BUF_MAGIC || sdev->magic != ARK_METAL_BUF_MAGIC) { set_last_error("memcpy_d2d: invalid handle"); std::abort(); }
        if ((std::size_t)bytes > d->size || (std::size_t)bytes > sdev->size) { set_last_error("memcpy_d2d: size exceeds"); std::abort(); }

        @autoreleasepool {
            if (!d->is_private && !sdev->is_private) {
                std::memcpy((std::uint8_t*)[d->buffer() contents] + d->offset,
                            (std::uint8_t*)[sdev->buffer() contents] + sdev->offset,
                            (std::size_t)bytes);
                return;
            }

            MetalStream* st = resolve_stream(stream);
            id<MTLCommandBuffer> cb = make_cb(st);
            if (!cb) { set_last_error("memcpy_d2d: commandBuffer creation failed"); return; }

            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            if (!blit) { set_last_error("memcpy_d2d: blit encoder creation failed"); return; }

            [blit copyFromBuffer:sdev->buffer() sourceOffset:(NSUInteger)sdev->offset
                        toBuffer:d->buffer() destinationOffset:(NSUInteger)d->offset
                            size:(NSUInteger)bytes];

            [blit endEncoding];
            [cb commit];
            commit_track(st, cb);

            if (!async) [cb waitUntilCompleted];
        }
    }

    void* resolve_pipeline(const std::string& lib_key, const char* kernel_name_utf8) {
        std::string key_copy;
        {
            std::lock_guard<std::mutex> lock(mu_libs_);
            if (libs_.find(lib_key) == libs_.end()) return nullptr;
            key_copy = lib_key;
        }

        KernelKey kk{key_copy, std::string(kernel_name_utf8 ? kernel_name_utf8 : "")};

        {
            std::lock_guard<std::mutex> lock(mu_pipelines_);
            auto it = pipelines_.find(kk);
            if (it != pipelines_.end()) return it->second;
        }

        id<MTLLibrary> lib = nil;
        {
            std::lock_guard<std::mutex> lock(mu_libs_);
            auto it = libs_.find(key_copy);
            if (it == libs_.end()) return nullptr;
            lib = ark_objc_get<id<MTLLibrary>>(it->second.lib);
        }
        if (!lib) return nullptr;

        @autoreleasepool {
            NSString* fn_name = [NSString stringWithUTF8String:kernel_name_utf8];
            if (!fn_name) return nullptr;

            id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
            if (!fn) return nullptr;

            NSError* err = nil;
            id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&err];
            ARK_OBJC_RELEASE(fn);

            if (!pso) {
                set_last_error(err ? [[err localizedDescription] UTF8String] : "newComputePipelineStateWithFunction failed");
                return nullptr;
            }

            void* stored = ark_objc_retain_from_new(pso);

            {
                std::lock_guard<std::mutex> lock(mu_pipelines_);
                pipelines_.emplace(std::move(kk), stored);
            }

            return stored;
        }
    }

    ark_gpu_status launch_with_pso(id<MTLComputePipelineState> pso,
                                  void** args, std::int32_t arg_count,
                                  std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                  std::int32_t bx, std::int32_t by, std::int32_t bz,
                                  std::uint32_t tg_mem_bytes,
                                  std::uint32_t tg_mem_index,
                                  ark_gpu_stream stream) {
        if (!pso) return ARK_GPU_ERR_NOT_FOUND;

        MetalStream* st = resolve_stream(stream);

        @autoreleasepool {
            id<MTLCommandBuffer> cb = make_cb(st);
            if (!cb) return ARK_GPU_ERR_DRIVER;

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) return ARK_GPU_ERR_DRIVER;

            [enc setComputePipelineState:pso];

            // Note: This default assumes the compiler ABI pads all scalar args to 8 bytes.
            constexpr std::size_t kScalarBytes = 8;

            for (std::int32_t i = 0; i < arg_count; ++i) {
                void* ap = args[i];
                if (!ap) { set_last_error("launch: null arg slot"); std::abort(); }

                // 1. Read the potential handle from the argument slot.
                //    We assume the compiler passes pointers/handles as 64-bit values.
                ark_gpu_device_ptr maybe_handle = nullptr;
                std::memcpy(&maybe_handle, ap, sizeof(ark_gpu_device_ptr));

                // 2. SAFETY CHECK: Is this even a plausible pointer?
                //    If 'maybe_handle' is a small scalar (e.g. 1, 42) or unaligned,
                //    dereferencing it in is_buf() to check magic will SEGFAULT.
                bool is_buffer = false;
                const uintptr_t ptr_val = (uintptr_t)maybe_handle;
                
                // Heuristic: Must be non-null, user-space (e.g., > 64KB), and aligned.
                const bool ptr_plausible = (ptr_val > 0x10000) && (ptr_val % alignof(std::max_align_t) == 0);

                if (ptr_plausible && is_buf(maybe_handle)) {
                    is_buffer = true;
                }

                if (is_buffer) {
                    MetalBuffer* b = as_buf(maybe_handle);
                    [enc setBuffer:b->buffer() offset:(NSUInteger)b->offset atIndex:(NSUInteger)i];
                } else {
                    // Treat as scalar/raw bytes.
                    // copy the data pointed to by ap into the command buffer.
                    [enc setBytes:ap length:kScalarBytes atIndex:(NSUInteger)i];
                }
            }

            if (tg_mem_bytes != 0) {
                [enc setThreadgroupMemoryLength:(NSUInteger)tg_mem_bytes atIndex:(NSUInteger)tg_mem_index];
            }

            const MTLSize tg = MTLSizeMake((NSUInteger)bx, (NSUInteger)by, (NSUInteger)bz);
            const MTLSize grid = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);

            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            [cb commit];
            commit_track(st, cb);
            return ARK_GPU_OK;
        }
    }

    void invalidate_pipelines_locked() {
        std::lock_guard<std::mutex> lock(mu_pipelines_);
        for (auto& kv : pipelines_) ark_objc_release(kv.second);
        pipelines_.clear();
    }

    void shutdown_locked() {
        if (!initialized_) {
            ark_objc_release(device_);
            device_ = nullptr;
            device_id_ = -1;
            return;
        }

        (void)stream_synchronize(nullptr);

        {
            std::lock_guard<std::mutex> lock(mu_libs_);
            invalidate_pipelines_locked();
            for (auto& kv : libs_) ark_objc_release(kv.second.lib);
            libs_.clear();
            default_lib_key_.clear();
        }

        {
            std::lock_guard<std::mutex> lock(default_stream_.mu);
            ark_objc_release(default_stream_.last_cb);
            ark_objc_release(default_stream_.queue);
            default_stream_.last_cb = nullptr;
            default_stream_.queue = nullptr;
        }

        ark_objc_release(device_);
        device_ = nullptr;
        device_id_ = -1;
        initialized_ = false;
    }

    const std::string& peek_key_handle(ark_gpu_module mod) const {
        return *reinterpret_cast<const std::string*>(mod);
    }

    std::string consume_key_handle(ark_gpu_module mod) const {
        std::string* p = reinterpret_cast<std::string*>(mod);
        std::string out = std::move(*p);
        delete p;
        return out;
    }

    ark_gpu_module store_key_handle(const std::string& key) const {
        return reinterpret_cast<ark_gpu_module>(new std::string(key));
    }

private:
    std::mutex mu_state_;
    bool initialized_ = false;

    void* device_ = nullptr;             // retained id<MTLDevice>
    std::int32_t device_id_ = -1;

    MetalStream default_stream_{};

    std::mutex mu_libs_;
    std::unordered_map<std::string, LibraryEntry> libs_;
    std::string default_lib_key_;

    std::mutex mu_pipelines_;
    std::unordered_map<KernelKey, void*, KernelKeyHash> pipelines_;

    ark_gpu_log_fn logger_ = nullptr;
    void* logger_user_ = nullptr;
};

static MetalBackend& B() {
    static MetalBackend b;
    return b;
}

// ------------------------------
// VTable wrappers
// ------------------------------

static ark_gpu_status v_init(std::int32_t device_id, const ark_gpu_init_params* params) { return B().init(device_id, params); }
static void v_shutdown() { B().shutdown(); }

static void v_enter_thread() { B().enter_thread(); }
static void v_leave_thread() { B().leave_thread(); }

static ark_gpu_status v_device_count(std::uint32_t* out_count) { return B().device_count(out_count); }
static ark_gpu_status v_set_device(std::int32_t device_id) { return B().set_device(device_id); }
static ark_gpu_status v_get_device(std::int32_t* out_device_id) { return B().get_device(out_device_id); }
static ark_gpu_status v_get_device_info(std::int32_t device_id, ark_gpu_device_info* out_info) { return B().get_device_info(device_id, out_info); }

static ark_gpu_device_ptr v_alloc(std::int64_t bytes) { return B().alloc(bytes); }
static void v_free(ark_gpu_device_ptr p) { B().free(p); }

static ark_gpu_status v_alloc_ex(const ark_gpu_alloc_desc* desc, std::int64_t bytes, ark_gpu_device_ptr* out_p) { return B().alloc_ex(desc, bytes, out_p); }

static void v_memcpy_to_device(ark_gpu_device_ptr dst, const void* src, std::int64_t bytes) { B().memcpy_to_device(dst, src, bytes); }
static void v_memcpy_to_host(void* dst, ark_gpu_device_ptr src, std::int64_t bytes) { B().memcpy_to_host(dst, src, bytes); }

static ark_gpu_status v_memcpy_async(void* dst, const void* src, std::int64_t bytes, ark_gpu_memcpy_kind kind, ark_gpu_stream stream) {
    return B().memcpy_async(dst, src, bytes, kind, stream);
}

static ark_gpu_status v_memset_async(ark_gpu_device_ptr dst, std::int32_t value, std::int64_t bytes, ark_gpu_stream stream) {
    return B().memset_async(dst, value, bytes, stream);
}

static ark_gpu_status v_stream_create(const ark_gpu_stream_desc* desc, ark_gpu_stream* out_stream) { return B().stream_create(desc, out_stream); }
static void v_stream_destroy(ark_gpu_stream stream) { B().stream_destroy(stream); }
static ark_gpu_status v_stream_synchronize(ark_gpu_stream stream) { return B().stream_synchronize(stream); }

static ark_gpu_status v_event_create(const ark_gpu_event_desc* desc, ark_gpu_event* out_evt) { return B().event_create(desc, out_evt); }
static void v_event_destroy(ark_gpu_event evt) { B().event_destroy(evt); }
static ark_gpu_status v_event_record(ark_gpu_event evt, ark_gpu_stream stream) { return B().event_record(evt, stream); }
static ark_gpu_status v_event_synchronize(ark_gpu_event evt) { return B().event_synchronize(evt); }
static ark_gpu_status v_event_elapsed_ms(ark_gpu_event a, ark_gpu_event b, float* out_ms) { return B().event_elapsed_ms(a, b, out_ms); }

static ark_gpu_status v_module_load(const ark_gpu_module_desc* desc, ark_gpu_module* out_mod) { return B().module_load(desc, out_mod); }
static void v_module_unload(ark_gpu_module mod) { B().module_unload(mod); }
static ark_gpu_status v_module_set_default(ark_gpu_module mod) { return B().module_set_default(mod); }

static ark_gpu_status v_rtc_compile(const ark_gpu_rtc_desc* desc, ark_gpu_module* out_mod) { return B().rtc_compile(desc, out_mod); }
static const char* v_rtc_last_log(const char* key) { return B().rtc_last_log(key); }

static ark_gpu_status v_kernel_lookup(ark_gpu_module mod, const char* name, ark_gpu_kernel* out_k) { return B().kernel_lookup(mod, name, out_k); }
static ark_gpu_status v_kernel_lookup_default(const char* name, ark_gpu_kernel* out_k) { return B().kernel_lookup_default(name, out_k); }

static void v_launch(const char* kernel_name,
                     void** args, std::int32_t arg_count,
                     std::int32_t gx, std::int32_t gy, std::int32_t gz,
                     std::int32_t bx, std::int32_t by, std::int32_t bz,
                     ark_gpu_stream stream) {
    B().launch(kernel_name, args, arg_count, gx, gy, gz, bx, by, bz, stream);
}

static ark_gpu_status v_launch_handle(ark_gpu_kernel kernel,
                                     void** args, std::int32_t arg_count,
                                     std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                     std::int32_t bx, std::int32_t by, std::int32_t bz,
                                     ark_gpu_stream stream) {
    return B().launch_handle(kernel, args, arg_count, gx, gy, gz, bx, by, bz, stream);
}

static ark_gpu_status v_launch_ex(const char* kernel_name,
                                 void** args, std::int32_t arg_count,
                                 std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                 std::int32_t bx, std::int32_t by, std::int32_t bz,
                                 const ark_gpu_launch_desc* desc,
                                 ark_gpu_stream stream) {
    return B().launch_ex(kernel_name, args, arg_count, gx, gy, gz, bx, by, bz, desc, stream);
}

static void v_synchronize(ark_gpu_stream stream) { B().synchronize(stream); }

static const char* v_last_error_string() { return B().last_error_string(); }

static constexpr std::uint64_t kFeatures =
    ARK_GPU_FEAT_STREAMS |
    ARK_GPU_FEAT_EVENTS |
    ARK_GPU_FEAT_ASYNC_MEMCPY |
    ARK_GPU_FEAT_ASYNC_MEMSET |
    ARK_GPU_FEAT_D2D_MEMCPY |
    ARK_GPU_FEAT_SHARED_HOST_VISIBLE |
    ARK_GPU_FEAT_MODULES |
    ARK_GPU_FEAT_RUNTIME_COMPILE |
    ARK_GPU_FEAT_KERNEL_HANDLES |
    ARK_GPU_FEAT_LAUNCH_EX |
    ARK_GPU_FEAT_DEVICE_QUERY |
    ARK_GPU_FEAT_MULTI_DEVICE;

static const ark_gpu_backend_v1 kVTable = {
    ARK_GPU_BACKEND_ABI_V1,
    (std::uint32_t)sizeof(ark_gpu_backend_v1),

    ARK_GPU_BACKEND_METAL,
    "metal",
    kFeatures,

    v_init,
    v_shutdown,

    v_enter_thread,
    v_leave_thread,

    v_device_count,
    v_set_device,
    v_get_device,
    v_get_device_info,

    v_alloc,
    v_free,

    v_alloc_ex,
    nullptr,
    nullptr,

    v_memcpy_to_device,
    v_memcpy_to_host,

    nullptr,
    v_memcpy_async,
    v_memset_async,

    v_stream_create,
    v_stream_destroy,
    v_stream_synchronize,

    v_event_create,
    v_event_destroy,
    v_event_record,
    v_event_synchronize,
    v_event_elapsed_ms,
    nullptr,

    v_module_load,
    v_module_unload,
    v_module_set_default,

    v_rtc_compile,
    v_rtc_last_log,

    v_kernel_lookup,
    v_kernel_lookup_default,

    v_launch,
    v_launch_handle,
    v_launch_ex,

    v_synchronize,

    v_last_error_string,

    { nullptr }
};

} // namespace

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1() {
    return &kVTable;
}

#else  // ARK_BACKEND_METAL

namespace {

static thread_local std::string g_tls_last_error;
static inline const char* v_last_error_string() { return g_tls_last_error.empty() ? nullptr : g_tls_last_error.c_str(); }

static ark_gpu_status v_init(std::int32_t, const ark_gpu_init_params*) {
    g_tls_last_error = "Metal backend not compiled (ARK_BACKEND_METAL disabled)";
    return ARK_GPU_ERR_UNSUPPORTED;
}

static void v_shutdown() {}

static ark_gpu_device_ptr v_alloc(std::int64_t) { return nullptr; }
static void v_free(ark_gpu_device_ptr) {}
static void v_memcpy_to_device(ark_gpu_device_ptr, const void*, std::int64_t) {}
static void v_memcpy_to_host(void*, ark_gpu_device_ptr, std::int64_t) {}
static void v_launch(const char*, void**, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, ark_gpu_stream) {}
static void v_synchronize(ark_gpu_stream) {}

static const ark_gpu_backend_v1 kVTable = {
    ARK_GPU_BACKEND_ABI_V1,
    (std::uint32_t)sizeof(ark_gpu_backend_v1),

    ARK_GPU_BACKEND_METAL,
    "metal",
    0,

    v_init,
    v_shutdown,

    nullptr,
    nullptr,

    nullptr,
    nullptr,
    nullptr,
    nullptr,

    v_alloc,
    v_free,

    nullptr,
    nullptr,
    nullptr,
    nullptr,

    v_memcpy_to_device,
    v_memcpy_to_host,

    nullptr,
    nullptr,
    nullptr,

    nullptr,
    nullptr,
    nullptr,

    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,

    nullptr,
    nullptr,
    nullptr,

    nullptr,
    nullptr,

    nullptr,
    nullptr,

    v_launch,
    nullptr,
    nullptr,

    v_synchronize,

    v_last_error_string,

    { nullptr }
};

} // namespace

ARK_GPU_EXPORT const ark_gpu_backend_v1* ark_gpu_backend_query_v1() {
    return &kVTable;
}

#endif // ARK_BACKEND_METAL
```

Suggestion: add a tiny optional `ark_gpu_kernel_arg_layout` side-channel (per kernel) so Metal can pass exact scalar sizes instead of the fixed 8-byte fallback.
