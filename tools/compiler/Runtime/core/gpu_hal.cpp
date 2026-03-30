// Runtime/core/gpu_hal.cpp
#include "../gpu/gpu_backend.h"
#include "../gpu/gpu_fatbin.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <dlfcn.h>
  #include <limits.h>
  #include <unistd.h>
  #if defined(__APPLE__)
    #include <mach-o/dyld.h>
  #endif
#endif

namespace ark::gpu {
namespace {

// ============================================================================
// Backend Kind
// ============================================================================
enum class BackendKind : std::uint8_t {
    None  = 0,
    Cuda  = 1,
    Hip   = 2,
    Metal = 3
};

static inline BackendKind to_kind(ark_gpu_backend_kind k) {
    switch (k) {
        case ARK_GPU_BACKEND_CUDA:  return BackendKind::Cuda;
        case ARK_GPU_BACKEND_HIP:   return BackendKind::Hip;
        case ARK_GPU_BACKEND_METAL: return BackendKind::Metal;
        default:                    return BackendKind::None;
    }
}

static inline const char* kind_name(BackendKind k) {
    switch (k) {
        case BackendKind::Cuda:  return "cuda";
        case BackendKind::Hip:   return "hip";
        case BackendKind::Metal: return "metal";
        default:                 return "none";
    }
}

// ============================================================================
// Logging
// ============================================================================
enum class LogLevel : std::uint8_t { Debug=0, Info=1, Warn=2, Error=3 };

static inline bool env_true(const char* s) {
    if (!s || !s[0]) return false;
    if (std::strcmp(s, "1") == 0) return true;
    if (std::strcmp(s, "true") == 0) return true;
    if (std::strcmp(s, "TRUE") == 0) return true;
    if (std::strcmp(s, "yes") == 0) return true;
    if (std::strcmp(s, "YES") == 0) return true;
    if (std::strcmp(s, "on") == 0) return true;
    if (std::strcmp(s, "ON") == 0) return true;
    return false;
}

[[noreturn]] static inline void die(const char* msg) {
    std::fprintf(stderr, "[ARK GPU_HAL] FATAL: %s\n", msg ? msg : "(null)");
    std::abort();
}

// [NEW] Graceful exit for environment/configuration errors
[[noreturn]] static inline void user_error_exit(const char* msg) {
    std::fprintf(stderr, "[ARK GPU_HAL] CONFIG ERROR:\n%s\n", msg ? msg : "(null)");
    std::exit(1); // Exits cleanly without triggering Signal 6 / ABORT
}

struct Logger final {
    bool verbose = false;

    void log(LogLevel lvl, const char* op, const char* msg) const {
        if (lvl == LogLevel::Debug && !verbose) return;

        const char* tag = "[INF]";
        FILE* out = stdout;
        switch (lvl) {
            case LogLevel::Debug: tag = "[DBG]"; out = stderr; break;
            case LogLevel::Info:  tag = "[INF]"; out = stdout; break;
            case LogLevel::Warn:  tag = "[WRN]"; out = stderr; break;
            case LogLevel::Error: tag = "[ERR]"; out = stderr; break;
        }

        std::fprintf(out, "[ARK GPU_HAL]%s %s: %s\n",
                     tag,
                     op ? op : "(op)",
                     msg ? msg : "(null)");
    }

    void log(LogLevel lvl, const char* op, const std::string& msg) const {
        log(lvl, op, msg.c_str());
    }

    struct Once final {
        std::mutex mu;
        std::unordered_map<std::string, bool> seen;

        bool hit(const std::string& key) {
            std::lock_guard<std::mutex> lk(mu);
            auto it = seen.find(key);
            if (it != seen.end()) return true;
            seen.emplace(key, true);
            return false;
        }
    };
};

static inline bool backend_required() { return env_true(std::getenv("ARK_GPU_REQUIRED")); }
static inline bool soft_stub_enabled() { return env_true(std::getenv("ARK_GPU_SOFT_STUB")); }

// ============================================================================
// Small string helpers
// ============================================================================
static inline std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1);
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t' || s.back()  == '\n' || s.back()  == '\r')) s.remove_suffix(1);
    return s;
}

static inline bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const char ca = a[i];
        const char cb = b[i];
        const char la = (ca >= 'A' && ca <= 'Z') ? static_cast<char>(ca - 'A' + 'a') : ca;
        const char lb = (cb >= 'A' && cb <= 'Z') ? static_cast<char>(cb - 'A' + 'a') : cb;
        if (la != lb) return false;
    }
    return true;
}

static inline BackendKind parse_backend(std::string_view s) {
    s = trim(s);
    if (s.empty()) return BackendKind::None;
    if (ieq(s, "cuda")) return BackendKind::Cuda;
    if (ieq(s, "hip")) return BackendKind::Hip;
    if (ieq(s, "metal")) return BackendKind::Metal;
    return BackendKind::None;
}

static inline std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
#if defined(_WIN32)
    const char sep = '\\';
    const bool a_sep = (a.back() == '\\' || a.back() == '/');
    const bool b_sep = (b.front() == '\\' || b.front() == '/');
#else
    const char sep = '/';
    const bool a_sep = (a.back() == '/');
    const bool b_sep = (b.front() == '/');
#endif
    if (a_sep && b_sep) return a + b.substr(1);
    if (!a_sep && !b_sep) return a + sep + b;
    return a + b;
}

static std::string get_exe_path() {
#if defined(_WIN32)
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return std::string();
    return std::string(buf, buf + n);
#elif defined(__APPLE__)
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    if (sz == 0) return std::string();
    std::string out;
    out.resize(sz + 1);
    if (_NSGetExecutablePath(out.data(), &sz) != 0) return std::string();
    out.resize(std::strlen(out.c_str()));
    return out;
#else
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    buf[n] = 0;
    return std::string(buf);
#endif
}

static std::string get_home_dir() {
#if defined(_WIN32)
    if (const char* up = std::getenv("USERPROFILE")) return std::string(up);
    return "";
#else
    if (const char* h = std::getenv("HOME")) return std::string(h);
    return "";
#endif
}

static std::string dirname_of(const std::string& p) {
    if (p.empty()) return std::string(".");
#if defined(_WIN32)
    std::size_t pos = p.find_last_of("\\/");
#else
    std::size_t pos = p.find_last_of('/');
#endif
    if (pos == std::string::npos) return std::string(".");
    if (pos == 0) return std::string("/");
    return p.substr(0, pos);
}

static std::vector<std::string> split_path_list(const char* s) {
    std::vector<std::string> out;
    if (!s || !s[0]) return out;

#if defined(_WIN32)
    const char sep = ';';
#else
    const char sep = ':';
#endif

    std::string cur;
    for (const char* p = s; *p; ++p) {
        if (*p == sep) {
            if (!cur.empty()) out.emplace_back(cur);
            cur.clear();
            continue;
        }
        cur.push_back(*p);
    }
    if (!cur.empty()) out.emplace_back(cur);
    return out;
}
// Add near other env helpers
static inline bool fatbin_required() { return env_true(std::getenv("ARK_GPU_REQUIRE_FATBIN")); }

// ============================================================================
// Dynamic library
// ============================================================================
struct DynLib {
#if defined(_WIN32)
    HMODULE h = nullptr;
#else
    void* h = nullptr;
#endif

    void close() {
        if (!h) return;
#if defined(_WIN32)
        FreeLibrary(h);
#else
        dlclose(h);
#endif
        h = nullptr;
    }

    bool try_open(const char* path) {
#if defined(_WIN32)
        h = LoadLibraryA(path);
        return h != nullptr;
#else
        (void)dlerror();
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        return h != nullptr;
#endif
    }

    void* sym(const char* name) const {
        if (!h) return nullptr;
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(h, name));
#else
        (void)dlerror();
        return dlsym(h, name);
#endif
    }

    static std::string last_load_error_string() {
#if defined(_WIN32)
        const DWORD e = GetLastError();
        return std::string("win32 error=") + std::to_string(static_cast<unsigned long>(e));
#else
        const char* e = dlerror();
        return (e && e[0]) ? std::string(e) : std::string("unknown dlerror");
#endif
    }
};

// ============================================================================
// Candidate plugin selection
// ============================================================================
static BackendKind choose_preferred_kind() {
    const char* env = std::getenv("ARK_GPU_BACKEND");
    if (env && env[0]) {
        const BackendKind k = parse_backend(env);
        if (k != BackendKind::None) return k;
        std::fprintf(stderr, "[ARK GPU_HAL] invalid ARK_GPU_BACKEND='%s' (expected cuda|hip|metal)\n", env);
        std::abort();
    }
#if defined(__APPLE__)
    return BackendKind::Metal;
#else
    return BackendKind::Cuda;
#endif
}

static const char* default_lib_for(BackendKind k) {
#if defined(_WIN32)
    switch (k) {
        case BackendKind::Cuda:  return "libark_cuda_backend.dll";
        case BackendKind::Hip:   return "libark_hip_backend.dll";
        case BackendKind::Metal: return "libark_metal_backend.dll";
        default: return "";
    }
#elif defined(__APPLE__)
    switch (k) {
        case BackendKind::Cuda:  return "libark_cuda_backend.dylib";
        case BackendKind::Hip:   return "libark_hip_backend.dylib";
        case BackendKind::Metal: return "libark_metal_backend.dylib";
        default: return "";
    }
#else
    switch (k) {
        case BackendKind::Cuda:  return "libark_cuda_backend.so";
        case BackendKind::Hip:   return "libark_hip_backend.so";
        case BackendKind::Metal: return "libark_metal_backend.so";
        default: return "";
    }
#endif
}

static void push_with_prefixes(std::vector<std::string>& out, const std::string& base_dir, const std::string& libname) {
    out.emplace_back(path_join(base_dir, libname));
    out.emplace_back(path_join(path_join(base_dir, "lib"), libname));
    out.emplace_back(path_join(path_join(base_dir, "lib64"), libname));
    out.emplace_back(path_join(path_join(base_dir, "bin"), path_join("lib", libname)));
}

static std::vector<std::string> build_candidate_paths(BackendKind preferred, const Logger& log) {
    std::vector<std::string> out;
    out.reserve(64);

    // 1. Direct Override (Highest Priority)
    if (const char* p = std::getenv("ARK_GPU_BACKEND_LIB"); p && p[0]) {
        out.emplace_back(p);
        return out;
    }

    const std::string exe_path = get_exe_path();
    const std::string exe_dir  = dirname_of(exe_path);

    const char* plugin_dir_env = std::getenv("ARK_GPU_PLUGIN_DIR");
    const auto plugin_dirs = split_path_list(plugin_dir_env);

    // --- NEW: Best-Effort Global & User Paths ---
    std::vector<std::string> best_effort_dirs;
    const std::string home = get_home_dir();
    if (!home.empty()) {
        best_effort_dirs.push_back(path_join(home, ".arknet/libs"));
        best_effort_dirs.push_back(path_join(home, ".arknet/lib"));
        best_effort_dirs.push_back(path_join(home, ".arknet/plugins"));
    }
#if !defined(_WIN32)
    // Global system installations
    best_effort_dirs.push_back("/usr/local/lib/arknet");
    best_effort_dirs.push_back("/opt/arknet/lib");
#endif
    // --------------------------------------------

    const BackendKind order[3] = {
#if defined(__APPLE__)
        BackendKind::Metal, BackendKind::Cuda, BackendKind::Hip
#else
        BackendKind::Cuda, BackendKind::Hip, BackendKind::Metal
#endif
    };

    auto add_kind = [&](BackendKind k) {
        const char* lib = default_lib_for(k);
        if (!lib || !lib[0]) return;
        const std::string libname(lib);

        // A. Look relative to the executable
        push_with_prefixes(out, exe_dir, libname);
        
        // B. Look in explicitly provided ENV paths
        for (const auto& d : plugin_dirs) push_with_prefixes(out, d, libname);

        // C. Look in Best-Effort paths (~/.arknet/libs, etc)
        for (const auto& d : best_effort_dirs) {
            out.emplace_back(path_join(d, libname));
        }

        // D. Look in current working directory and system library paths (LD_LIBRARY_PATH)
        out.emplace_back(std::string("./") + libname);
        out.emplace_back(libname);
    };

    add_kind(preferred);
    for (BackendKind k : order) {
        if (k == preferred) continue;
        add_kind(k);
    }

    if (log.verbose) {
        log.log(LogLevel::Debug, "probe", std::string("exe_dir=") + exe_dir);
        log.log(LogLevel::Debug, "probe", std::string("candidates=") + std::to_string(out.size()));
    }

    return out;
}

// ============================================================================
// Vtable validation + error reporting
// ============================================================================
template <typename T>
static inline bool validate_backend_vtable(const T* vt, const char** reason) {
    if (reason) *reason = nullptr;
    if constexpr (requires { ark::gpu::abi::validate_abi_verbose(vt, reason); }) {
        return ark::gpu::abi::validate_abi_verbose(vt, reason);
    } else {
        return validate_abi_verbose(vt, reason);
    }
}

static inline const char* backend_last_error(const ark_gpu_backend_v1* vt) {
    if (!vt || !vt->last_error_string) return nullptr;
    return vt->last_error_string();
}
// ============================================================================
// Fatbin query (optional; must never produce unresolved external symbols)
// ============================================================================
using FatbinQueryFn = const ark_gpu_fatbin_v1* (*)();

namespace {
static inline std::string ptr_hex(const void* p) {
    const std::uintptr_t v = reinterpret_cast<std::uintptr_t>(p);
    char buf[2 + sizeof(std::uintptr_t) * 2 + 1];
    static constexpr char kHex[] = "0123456789abcdef";

    buf[0] = '0';
    buf[1] = 'x';

    for (std::size_t i = 0; i < sizeof(std::uintptr_t) * 2; ++i) {
        const std::size_t shift = (sizeof(std::uintptr_t) * 8 - 4) - i * 4;
        buf[2 + i] = kHex[(v >> shift) & 0xF];
    }

    buf[2 + sizeof(std::uintptr_t) * 2] = '\0';
    return std::string(buf);
}

#if defined(_WIN32)
static std::string win32_last_error_string() {
    const DWORD e = ::GetLastError();
    return std::string("win32_last_error=") + std::to_string(static_cast<unsigned long long>(e));
}
#else
static std::string dl_last_error_string() {
    const char* e = ::dlerror();
    if (e && e[0]) return std::string(e);
    return "dlerror=(none)";
}

static std::string dl_addr_string(const void* p) {
    if (!p) return "dladdr=null";

    Dl_info info{};
    if (::dladdr(const_cast<void*>(p), &info) == 0) return "dladdr=failed";

    std::string out;
    out.reserve(256);
    out += "dladdr.sname=";
    out += (info.dli_sname ? info.dli_sname : "(null)");
    out += " dli_fname=";
    out += (info.dli_fname ? info.dli_fname : "(null)");
    return out;
}

static std::string proc_self_maps_hint() {
    FILE* f = std::fopen("/proc/self/maps", "rb");
    if (!f) return "proc_maps=unavailable";
    std::fclose(f);
    return "proc_maps=present";
}
#endif

static void fatbin_debug_not_found(const char* phase, const char* detail) {
    if (!phase) phase = "(phase)";
    if (!detail) detail = "(detail)";

    std::string msg;
    msg.reserve(512);
    msg += "ark_gpu_fatbin_query_v1 not found; phase=";
    msg += phase;
    msg += " detail=";
    msg += detail;
#if !defined(_WIN32)
    msg += " ";
    msg += proc_self_maps_hint();
#endif
}

static void fatbin_debug_found(const void* p) {
    std::string msg;
    msg.reserve(256);
    msg += "ark_gpu_fatbin_query_v1 resolved; addr=";
    msg += ptr_hex(p);
#if !defined(_WIN32)
    msg += " ";
    msg += dl_addr_string(p);
#endif
}
} // namespace


#if !defined(_WIN32)
extern "C" {
  #if defined(__APPLE__)
    __attribute__((weak_import)) const ark_gpu_fatbin_v1* ark_gpu_fatbin_query_v1();
  #else
    __attribute__((weak)) const ark_gpu_fatbin_v1* ark_gpu_fatbin_query_v1();
  #endif
}
#endif

static FatbinQueryFn resolve_fatbin_query() {
    static FatbinQueryFn cached = nullptr;
    static bool tried = false;
    if (tried) return cached;
    tried = true;

#if !defined(_WIN32)
    // Preferred: direct weak symbol reference (does not require export tables).
    if (ark_gpu_fatbin_query_v1) {
        cached = &ark_gpu_fatbin_query_v1;
        fatbin_debug_found(reinterpret_cast<void*>(cached));
        return cached;
    }
#endif

#if defined(_WIN32)
    HMODULE self = ::GetModuleHandleW(nullptr);
    if (!self) {
        fatbin_debug_not_found("GetModuleHandleW(nullptr)", win32_last_error_string().c_str());
        return nullptr;
    }

    FARPROC p = ::GetProcAddress(self, "ark_gpu_fatbin_query_v1");
    if (!p) {
        fatbin_debug_not_found("GetProcAddress(self)", win32_last_error_string().c_str());
        return nullptr;
    }

    cached = reinterpret_cast<FatbinQueryFn>(p);
    fatbin_debug_found(reinterpret_cast<void*>(p));
    return cached;
#else
    (void)::dlerror();
    void* p = ::dlsym(RTLD_DEFAULT, "ark_gpu_fatbin_query_v1");
    const char* e0 = ::dlerror();

    if (!p) {
        std::string d;
        d.reserve(512);
        d += "dlsym(RTLD_DEFAULT) failed; ";
        d += (e0 && e0[0]) ? e0 : "dlerror=(none)";
        d += " (common causes: dynsym stripped; built without -rdynamic; --gc-sections removed registry TU)";
        fatbin_debug_not_found("dlsym(RTLD_DEFAULT)", d.c_str());

        static void* self = nullptr;
        if (!self) {
            (void)::dlerror();
            self = ::dlopen(nullptr, RTLD_NOW);
            const char* e1 = ::dlerror();
            if (!self) {
                std::string d2;
                d2.reserve(256);
                d2 += "dlopen(nullptr) failed; ";
                d2 += (e1 && e1[0]) ? e1 : "dlerror=(none)";
                fatbin_debug_not_found("dlopen(nullptr)", d2.c_str());
                return nullptr;
            }
        }

        (void)::dlerror();
        p = ::dlsym(self, "ark_gpu_fatbin_query_v1");
        const char* e2 = ::dlerror();
        if (!p) {
            std::string d3;
            d3.reserve(512);
            d3 += "dlsym(self-handle) failed; ";
            d3 += (e2 && e2[0]) ? e2 : "dlerror=(none)";
            d3 += " (common causes: registry TU not linked; symbol hidden/visibility; sealed capsule dropped exports)";
            fatbin_debug_not_found("dlsym(self)", d3.c_str());
            return nullptr;
        }
    }

    cached = reinterpret_cast<FatbinQueryFn>(p);
    fatbin_debug_found(p);
    return cached;
#endif
}

static const ark_gpu_fatbin_v1* query_fatbin_optional() {
    FatbinQueryFn fn = resolve_fatbin_query();
    if (!fn) return nullptr;

    const ark_gpu_fatbin_v1* fb = fn();
    if (!fb) {
        // std::fprintf(stderr, "[ARK GPU_HAL][DBG] fatbin: ark_gpu_fatbin_query_v1 returned null fatbin pointer\n");
        return nullptr;
    }

    if (fb->magic != ARK_GPU_FATBIN_V1_MAGIC) {
        // std::fprintf(stderr, "[ARK GPU_HAL][DBG] fatbin: bad magic (got=%llu expected=%llu)\n",
        //              (unsigned long long)fb->magic,
        //              (unsigned long long)ARK_GPU_FATBIN_V1_MAGIC);
        return nullptr;
    }
    if (fb->abi != ARK_GPU_FATBIN_ABI_V1) {
        // std::fprintf(stderr, "[ARK GPU_HAL][DBG] fatbin: bad abi (got=%u expected=%u)\n",
        //              (unsigned)fb->abi,
        //              (unsigned)ARK_GPU_FATBIN_ABI_V1);
        return nullptr;
    }
    if (fb->entry_count == 0 || fb->entries == nullptr) {
        // std::fprintf(stderr, "[ARK GPU_HAL][DBG] fatbin: empty entries (count=%u entries_ptr=%p)\n",
        //              (unsigned)fb->entry_count, (const void*)fb->entries);
        return nullptr;
    }

    return fb;
}


// ============================================================================
// Stub backend
// ============================================================================
static ark_gpu_status stub_init(std::int32_t, const ark_gpu_init_params*) { return ARK_GPU_OK; }
static void* stub_alloc(std::int64_t) { return nullptr; }
static void stub_free(void*) {}
static void stub_memcpy_to_device(void*, const void*, std::int64_t) {}
static void stub_memcpy_to_host(void*, void*, std::int64_t) {}
static void stub_launch(const char*, void**, std::int32_t,
                        std::int32_t, std::int32_t, std::int32_t,
                        std::int32_t, std::int32_t, std::int32_t,
                        ark_gpu_stream) {}
static void stub_sync(ark_gpu_stream) {}
static const char* stub_last_error_string() { return "no GPU backend loaded"; }

static void fill_stub_vtable(ark_gpu_backend_v1& vt) {
    vt.abi = ARK_GPU_BACKEND_ABI_V1;
    vt.kind = ARK_GPU_BACKEND_NONE;
    vt.features = 0ull;
    vt.name = "stub";
    vt.last_error_string = stub_last_error_string;

    vt.init = stub_init;
    vt.alloc = stub_alloc;
    vt.free = stub_free;
    vt.memcpy_to_device = stub_memcpy_to_device;
    vt.memcpy_to_host = stub_memcpy_to_host;
    vt.launch = stub_launch;
    vt.synchronize = stub_sync;

    vt.set_device = nullptr;
    vt.get_device = nullptr;
    vt.device_count = nullptr;
    vt.enter_thread = nullptr;
    vt.leave_thread = nullptr;

    vt.stream_create = nullptr;
    vt.stream_destroy = nullptr;

    vt.event_create = nullptr;
    vt.event_destroy = nullptr;
    vt.event_record = nullptr;
    vt.event_synchronize = nullptr;
    vt.event_elapsed_ms = nullptr;

    vt.memcpy_async = nullptr;
    vt.memset_async = nullptr;

    vt.module_load = nullptr;
    vt.module_unload = nullptr;
    vt.module_set_default = nullptr;
    vt.kernel_lookup = nullptr;
}

// ============================================================================
// HAL State
// ============================================================================
struct HalState {
    std::mutex mu;
    bool fatbin_attempted = false;
    bool fatbin_loaded = false;

    Logger log;
    Logger::Once once;

    bool probed = false;
    bool available = false;
    bool initialized = false;

    BackendKind kind = BackendKind::None;
    std::int32_t active_device = 0;

    DynLib lib;
    const ark_gpu_backend_v1* vt = nullptr;

    std::unordered_map<std::string, std::string> probe_cache;
    std::string last_failure;

    bool banner_printed = false;

    ark_gpu_backend_v1 stub_vt{};
};

static HalState& S() {
    static HalState s;
    return s;
}

static inline void enter_thread_if_needed(const ark_gpu_backend_v1* vt) {
    if (vt && vt->enter_thread) vt->enter_thread();
}

static inline void leave_thread_if_needed(const ark_gpu_backend_v1* vt) {
    if (vt && vt->leave_thread) vt->leave_thread();
}

// ============================================================================
// Fatbin load
// ============================================================================
static inline bool fatbin_kind_matches_backend(std::uint32_t mk, BackendKind k) {
    switch (k) {
        case BackendKind::Cuda:
            return mk == ARK_GPU_MODULE_CUDA_CUBIN || mk == ARK_GPU_MODULE_CUDA_PTX;
        case BackendKind::Hip:
            return mk == ARK_GPU_MODULE_HIP_HSACO;
        case BackendKind::Metal:
            return mk == ARK_GPU_MODULE_METAL_LIB;
        default:
            return false;
    }
}

static bool load_fatbin_into_backend_locked(HalState& s) {
    if (!s.vt) return false;

    if ((s.vt->features & ARK_GPU_FEAT_MODULES) == 0) return true;
    if (!s.vt->module_load || !s.vt->module_set_default) return true;

    if (s.fatbin_loaded) return true;
    if (s.fatbin_attempted) return !fatbin_required();

    s.fatbin_attempted = true;

    const ark_gpu_fatbin_v1* fb = query_fatbin_optional();
    if (!fb) {
        if (s.log.verbose) s.log.log(LogLevel::Debug, "fatbin", "no fatbin in capsule");
        if (fatbin_required()) {
            s.log.log(LogLevel::Error, "fatbin", "fatbin required but missing (ARK_GPU_REQUIRE_FATBIN=1)");
            return false;
        }
        return true;
    }

    ark_gpu_module first = nullptr;
    std::uint32_t matched = 0;
    std::uint32_t loaded  = 0;

    for (std::uint32_t i = 0; i < fb->entry_count; ++i) {
        const ark_gpu_fatbin_entry_v1& e = fb->entries[i];

        if (!fatbin_kind_matches_backend(e.module_kind, s.kind)) continue;
        ++matched;

        if (!e.module_key || !e.module_key[0]) continue;
        if (!e.blob_ptr || e.blob_size == 0) continue;

        ark_gpu_module_desc d{};
        d.size = static_cast<std::uint32_t>(sizeof(ark_gpu_module_desc));
        d.kind = static_cast<ark_gpu_module_kind>(e.module_kind);
        d.flags = 0;
        d.module_key = e.module_key;
        d.bytes = e.blob_ptr;
        d.byte_len = static_cast<std::size_t>(e.blob_size);

        ark_gpu_module mod = nullptr;

        enter_thread_if_needed(s.vt);
        const ark_gpu_status rc = s.vt->module_load(&d, &mod);
        leave_thread_if_needed(s.vt);

        if (rc != ARK_GPU_OK || !mod) {
            if (s.log.verbose) {
                s.log.log(LogLevel::Warn, "fatbin",
                          std::string("module_load failed key=") + e.module_key +
                          " status=" + std::to_string((int)rc));
                const char* be = backend_last_error(s.vt);
                if (be && be[0]) s.log.log(LogLevel::Warn, "backend", be);
            }
            continue;
        }

        ++loaded;
        if (!first) first = mod;

        if (s.log.verbose) {
            s.log.log(LogLevel::Debug, "fatbin",
                      std::string("loaded key=") + e.module_key +
                      " bytes=" + std::to_string((unsigned long long)e.blob_size));
        }
    }

    if (!first) {
        if (matched == 0) {
            if (s.log.verbose) s.log.log(LogLevel::Debug, "fatbin", "no matching entries for backend");
            return true;
        }
        s.log.log(LogLevel::Error, "fatbin", "fatbin present but no modules loaded");
        return false;
    }

    enter_thread_if_needed(s.vt);
    const ark_gpu_status sdef = s.vt->module_set_default(first);
    leave_thread_if_needed(s.vt);

    if (sdef != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "fatbin",
                  std::string("module_set_default failed status=") + std::to_string((int)sdef));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        return false;
    }

    s.fatbin_loaded = true;

    if (s.log.verbose) {
        s.log.log(LogLevel::Debug, "fatbin",
                  std::string("default module set (loaded=") +
                  std::to_string(loaded) + " matched=" + std::to_string(matched) + ")");
    }
    return true;
}

static inline bool ensure_fatbin_ready_locked(HalState& s) {
    if (!s.vt) return false;
    if ((s.vt->features & ARK_GPU_FEAT_MODULES) == 0) return true;
    return load_fatbin_into_backend_locked(s);
}





// ============================================================================
// Banner
// ============================================================================
static void maybe_banner_locked(HalState& s) {
    if (s.banner_printed) return;
    s.banner_printed = true;

    const char* nm = (s.vt && s.vt->name) ? s.vt->name : "none";
    const std::uint64_t feat = s.vt ? s.vt->features : 0ull;

    std::string msg;
    msg.reserve(256);
    msg += "backend="; msg += nm;
    msg += " kind="; msg += kind_name(s.kind);
    msg += " initialized="; msg += (s.initialized ? "1" : "0");
    msg += " features=0x"; msg += std::to_string(static_cast<unsigned long long>(feat));
    msg += " device="; msg += std::to_string(s.active_device);
    msg += " required="; msg += (backend_required() ? "1" : "0");
    msg += " stub="; msg += (soft_stub_enabled() ? "1" : "0");
    msg += " verbose="; msg += (s.log.verbose ? "1" : "0");

    s.log.log(LogLevel::Info, "banner", msg);
}

// ============================================================================
// Plugin loading
// ============================================================================
static const ark_gpu_backend_v1* try_load_plugin(HalState& s, DynLib& L, const char* path, std::string& out_reason) {
    out_reason.clear();

    if (!path || !path[0]) {
        out_reason = "empty path";
        return nullptr;
    }

    if (!L.try_open(path)) {
        out_reason = std::string("dlopen failed: ") + DynLib::last_load_error_string();
        return nullptr;
    }

    void* sym = L.sym("ark_gpu_backend_query_v1");
    if (!sym) {
        out_reason = "missing ark_gpu_backend_query_v1";
        L.close();
        return nullptr;
    }

    auto query = reinterpret_cast<ark_gpu_backend_query_v1_fn>(sym);
    const ark_gpu_backend_v1* vt = query ? query() : nullptr;

    const char* why = nullptr;
    if (!validate_backend_vtable(vt, &why)) {
        out_reason = why ? why : "invalid vtable";
        L.close();
        return nullptr;
    }

    if (to_kind(vt->kind) == BackendKind::None) {
        out_reason = "invalid backend kind";
        L.close();
        return nullptr;
    }

    if (s.log.verbose) {
        std::string msg;
        msg.reserve(256);
        msg += "plugin="; msg += path;
        msg += " name="; msg += (vt->name ? vt->name : "(null)");
        msg += " kind="; msg += std::to_string(static_cast<unsigned>(vt->kind));
        msg += " features=0x"; msg += std::to_string(static_cast<unsigned long long>(vt->features));
        s.log.log(LogLevel::Debug, "probe", msg);
    }

    return vt;
}

static bool ensure_plugin_loaded_locked(HalState& s) {
    if (s.probed) return s.available;

    s.log.verbose = env_true(std::getenv("ARK_GPU_VERBOSE"));

    const BackendKind preferred = choose_preferred_kind();
    const auto paths = build_candidate_paths(preferred, s.log);

    s.probed = true;
    s.available = false;
    s.initialized = false;
    s.kind = BackendKind::None;
    s.vt = nullptr;
    s.last_failure.clear();

    if (paths.empty()) {
        s.last_failure = "no plugin candidates";
        return false;
    }

    for (const auto& p : paths) {
        if (auto it = s.probe_cache.find(p); it != s.probe_cache.end()) {
            if (s.log.verbose) {
                s.log.log(LogLevel::Debug, "probe", std::string("skip ") + p + " reason=" + it->second);
            }
            continue;
        }

        DynLib L;
        std::string reason;
        const ark_gpu_backend_v1* vt = try_load_plugin(s, L, p.c_str(), reason);
        if (!vt) {
            s.probe_cache.emplace(p, reason.empty() ? "probe failed" : reason);
            if (s.log.verbose) s.log.log(LogLevel::Debug, "dlopen", p + " :: " + (reason.empty() ? "fail" : reason));
            continue;
        }

        s.lib = L;
        s.vt = vt;
        s.kind = to_kind(vt->kind);
        s.available = true;

        s.log.log(LogLevel::Info, "load",
                  std::string("loaded plugin=") + p + " kind=" + kind_name(s.kind));
        return true;
    }

    s.last_failure = "no valid plugin loaded";
    return false;
}

static void hal_backend_log_trampoline(ark_gpu_log_level lvl, const char* msg, void* user) {
    auto* L = static_cast<Logger*>(user);
    if (!L) return;

    LogLevel out = LogLevel::Info;
    switch (lvl) {
        case ARK_GPU_LOG_TRACE: out = LogLevel::Debug; break;
        case ARK_GPU_LOG_DEBUG: out = LogLevel::Debug; break;
        case ARK_GPU_LOG_INFO:  out = LogLevel::Info;  break;
        case ARK_GPU_LOG_WARN:  out = LogLevel::Warn;  break;
        case ARK_GPU_LOG_ERROR: out = LogLevel::Error; break;
        default: out = LogLevel::Info; break;
    }
    L->log(out, "backend", msg ? msg : "(null)");
}

static inline std::int32_t clamp_device_id_locked(const ark_gpu_backend_v1* vt, std::int32_t device_id) {
    if (!vt) return 0;

    if (device_id < 0) device_id = 0;

    if ((vt->features & ARK_GPU_FEAT_MULTI_DEVICE) != 0 && vt->device_count) {
        std::uint32_t cnt = 0;
        if (vt->device_count(&cnt) == ARK_GPU_OK && cnt > 0) {
            if (device_id >= static_cast<std::int32_t>(cnt)) device_id = 0;
        } else {
            device_id = 0;
        }
    }

    return device_id;
}

static bool ensure_backend_initialized_locked(HalState& s, std::int32_t device_id) {
    if (!ensure_plugin_loaded_locked(s)) return false;
    if (!s.vt) return false;

    device_id = clamp_device_id_locked(s.vt, device_id);

    if (s.initialized) {
        if (device_id != s.active_device) {
            const bool multi =
                ((s.vt->features & ARK_GPU_FEAT_MULTI_DEVICE) != 0) &&
                (s.vt->set_device != nullptr);

            if (multi) {
                enter_thread_if_needed(s.vt);
                const ark_gpu_status st = s.vt->set_device(device_id);
                leave_thread_if_needed(s.vt);

                if (st != ARK_GPU_OK) {
                    s.log.log(LogLevel::Error, "set_device",
                              std::string("device=") + std::to_string(device_id) +
                              " status=" + std::to_string((int)st));
                    const char* be = backend_last_error(s.vt);
                    if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
                    return false;
                }

                std::int32_t cur = device_id;
                if (s.vt->get_device) {
                    enter_thread_if_needed(s.vt);
                    (void)s.vt->get_device(&cur);
                    leave_thread_if_needed(s.vt);
                }
                s.active_device = cur;
                return true;
            }

            if (s.vt->shutdown) {
                enter_thread_if_needed(s.vt);
                s.vt->shutdown();
                leave_thread_if_needed(s.vt);
            }
            s.initialized = false;
            s.active_device = 0;
        } else {
            return true;
        }
    }

    ark_gpu_init_params params{};
    params.size = static_cast<std::uint32_t>(sizeof(ark_gpu_init_params));
    params.flags = 0;
    params.logger = s.log.verbose ? hal_backend_log_trampoline : nullptr;
    params.logger_user = s.log.verbose ? static_cast<void*>(&s.log) : nullptr;

    enter_thread_if_needed(s.vt);
    const ark_gpu_status st = s.vt->init(device_id, &params);
    leave_thread_if_needed(s.vt);

    if (st != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "init",
                  std::string("device=") + std::to_string(device_id) +
                  " status=" + std::to_string((int)st));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);

        s.lib.close();
        s.vt = nullptr;
        s.kind = BackendKind::None;
        s.available = false;
        s.last_failure = "init failed";
        s.initialized = false;
        s.active_device = 0;
        return false;
    }

    if (!load_fatbin_into_backend_locked(s)) {
        s.lib.close();
        s.vt = nullptr;
        s.kind = BackendKind::None;
        s.available = false;
        s.last_failure = "fatbin load failed";
        s.initialized = false;
        s.active_device = 0;
        return false;
    }

    std::int32_t cur = device_id;
    if ((s.vt->features & ARK_GPU_FEAT_MULTI_DEVICE) != 0 && s.vt->get_device) {
        enter_thread_if_needed(s.vt);
        (void)s.vt->get_device(&cur);
        leave_thread_if_needed(s.vt);
    }

    s.initialized = true;
    s.active_device = cur;

    s.log.log(LogLevel::Info, "init",
              std::string("backend initialized kind=") + kind_name(s.kind) +
              " device=" + std::to_string(s.active_device));
    return true;
}

static void install_stub_locked(HalState& s) {
    fill_stub_vtable(s.stub_vt);
    s.vt = &s.stub_vt;
    s.kind = BackendKind::None;
    s.available = false;
    s.initialized = true;
    if (s.last_failure.empty()) s.last_failure = "stub installed";
}

// ============================================================================
// Runtime Verification
// ============================================================================
static inline void log_missing_backend_once_locked(HalState& s, const char* op) {
    std::string key;
    key.reserve(128);
    key += "missing|";
    key += (op ? op : "?");
    key += "|";
    key += (s.last_failure.empty() ? "?" : s.last_failure);

    if (s.once.hit(key)) return;

    std::string msg;
    msg.reserve(256);
    msg += "GPU backend unavailable. last_failure=";
    msg += (s.last_failure.empty() ? "(none)" : s.last_failure);
    msg += " required="; msg += (backend_required() ? "1" : "0");
    msg += " stub="; msg += (soft_stub_enabled() ? "1" : "0");

    s.log.log(LogLevel::Error, op ? op : "op", msg);
}

static inline bool ensure_ready_locked(HalState& s, std::int32_t device_id, const char* op) {
    const bool ok = ensure_backend_initialized_locked(s, device_id);
    if (!ok) {
        log_missing_backend_once_locked(s, op);

        if (soft_stub_enabled()) {
            install_stub_locked(s);
            maybe_banner_locked(s);
            return true;
        }

        // --- NEW: Hard Panic if backend is missing ---
        std::string panicMsg = std::string("Failed to initialize GPU Backend during: ") + (op ? op : "unknown");
        panicMsg += "\n  Cause: " + (s.last_failure.empty() ? "No valid plugin loaded" : s.last_failure);
        panicMsg += "\n  Action: Ensure ARK_GPU_PLUGIN_DIR is set to the directory containing libark_cuda_backend.so";
        user_error_exit(panicMsg.c_str());
        // ---------------------------------------------
        return false;
    }

    if (!s.vt) return false;

    if ((s.vt->features & ARK_GPU_FEAT_MODULES) != 0 &&
        s.vt->module_load &&
        s.vt->module_set_default) {

        if (!load_fatbin_into_backend_locked(s)) {
            s.log.log(LogLevel::Error, op ? op : "op", "fatbin load failed");
            if (fatbin_required()) die("fatbin required but missing (ARK_GPU_REQUIRE_FATBIN=1)");

            if (soft_stub_enabled()) {
                install_stub_locked(s);
                maybe_banner_locked(s);
                return true;
            }
            return false;
        }
    }

    maybe_banner_locked(s);
    return true;
}

} // namespace

// ============================================================================
// Strict surface (delegates to plugin vtable)
// ============================================================================
void init(std::int32_t device_id) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    (void)ensure_ready_locked(s, device_id, "init");
}

DevicePtr alloc(std::int64_t bytes) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "alloc")) return nullptr;

    enter_thread_if_needed(s.vt);
    DevicePtr p = s.vt->alloc(bytes);
    leave_thread_if_needed(s.vt);

    if (!p && s.log.verbose) {
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Warn, "alloc", be);
    }
    return p;
}

// [NEW] Add alloc_managed right after alloc()
DevicePtr alloc_managed(std::int64_t bytes) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "alloc_managed")) return nullptr;

    if (!s.vt || !s.vt->alloc_ex) {
        s.log.log(LogLevel::Error, "alloc_managed", "alloc_ex not supported by backend");
        return nullptr;
    }

    ark_gpu_alloc_desc desc{};
    desc.size = sizeof(ark_gpu_alloc_desc);
    desc.flags = 0;
    desc.kind = ARK_GPU_MEM_MANAGED; // Request Managed/Unified Memory
    desc.alignment = 0;

    ark_gpu_device_ptr ptr = nullptr;
    enter_thread_if_needed(s.vt);
    const ark_gpu_status rc = s.vt->alloc_ex(&desc, bytes, &ptr);
    leave_thread_if_needed(s.vt);

    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "alloc_managed", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        return nullptr;
    }
    return ptr;
}

void free(DevicePtr p) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "free")) return;

    enter_thread_if_needed(s.vt);
    s.vt->free(p);
    leave_thread_if_needed(s.vt);
}

void memcpy_to_device(DevicePtr dst, const void* src, std::int64_t bytes) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "memcpy_to_device")) return;

    enter_thread_if_needed(s.vt);
    s.vt->memcpy_to_device(dst, src, bytes);
    leave_thread_if_needed(s.vt);
}

void memcpy_to_host(void* dst, DevicePtr src, std::int64_t bytes) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "memcpy_to_host")) return;

    enter_thread_if_needed(s.vt);
    s.vt->memcpy_to_host(dst, src, bytes);
    leave_thread_if_needed(s.vt);
}

void launch(const char* kernel_name,
            void** args,
            std::int32_t arg_count,
            std::int32_t gx, std::int32_t gy, std::int32_t gz,
            std::int32_t bx, std::int32_t by, std::int32_t bz,
            StreamHandle stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "launch")) return;

    if (!kernel_name || !kernel_name[0]) {
        s.log.log(LogLevel::Error, "launch", "kernel_name is null/empty");
        if (backend_required()) die("launch: invalid kernel name");
        return;
    }

    enter_thread_if_needed(s.vt);
    s.vt->launch(kernel_name, args, arg_count, gx, gy, gz, bx, by, bz, stream);
    leave_thread_if_needed(s.vt);
}

void synchronize(StreamHandle stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "synchronize")) return;

    enter_thread_if_needed(s.vt);
    s.vt->synchronize(stream);
    leave_thread_if_needed(s.vt);
}

// ============================================================================
// Optional helper surface
// ============================================================================
namespace hal {

BackendKind backend_kind() {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "backend_kind")) return BackendKind::None;
    return s.kind;
}

const char* backend_name() {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "backend_name")) return "none";
    return (s.vt && s.vt->name) ? s.vt->name : "unknown";
}

std::uint64_t backend_features() {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "backend_features")) return 0ull;
    return s.vt ? s.vt->features : 0ull;
}

void* create_stream(int non_blocking, int priority) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "create_stream")) return nullptr;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_STREAMS) == 0 || !s.vt->stream_create) {
        s.log.log(LogLevel::Warn, "create_stream", "streams not supported by backend");
        return nullptr;
    }

    ark_gpu_stream_desc d{};
    d.size = static_cast<std::uint32_t>(sizeof(ark_gpu_stream_desc));
    d.flags = non_blocking ? ARK_GPU_STREAM_NON_BLOCKING : ARK_GPU_STREAM_DEFAULT;
    d.priority = priority;
    d.reserved = 0;

    ark_gpu_stream st = nullptr;
    const ark_gpu_status rc = s.vt->stream_create(&d, &st);
    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "create_stream", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        return nullptr;
    }
    return st;
}

void destroy_stream(void* stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "destroy_stream")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_STREAMS) == 0 || !s.vt->stream_destroy) return;
    s.vt->stream_destroy(stream);
}

void* create_event(int timing) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "create_event")) return nullptr;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_EVENTS) == 0 || !s.vt->event_create) {
        s.log.log(LogLevel::Warn, "create_event", "events not supported by backend");
        return nullptr;
    }

    ark_gpu_event_desc d{};
    d.size = static_cast<std::uint32_t>(sizeof(ark_gpu_event_desc));
    d.flags = timing ? ARK_GPU_EVENT_DEFAULT : ARK_GPU_EVENT_DISABLE_TIMING;

    ark_gpu_event ev = nullptr;
    const ark_gpu_status rc = s.vt->event_create(&d, &ev);
    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "create_event", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        return nullptr;
    }
    return ev;
}

void destroy_event(void* evt) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "destroy_event")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_EVENTS) == 0 || !s.vt->event_destroy) return;
    s.vt->event_destroy(evt);
}

void record_event(void* evt, void* stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "record_event")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_EVENTS) == 0 || !s.vt->event_record) return;

    const ark_gpu_status rc = s.vt->event_record(evt, stream);
    if (rc != ARK_GPU_OK && s.log.verbose) {
        s.log.log(LogLevel::Warn, "record_event", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Warn, "backend", be);
    }
}

void sync_event(void* evt) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "sync_event")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_EVENTS) == 0 || !s.vt->event_synchronize) return;

    const ark_gpu_status rc = s.vt->event_synchronize(evt);
    if (rc != ARK_GPU_OK && s.log.verbose) {
        s.log.log(LogLevel::Warn, "sync_event", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Warn, "backend", be);
    }
}

float elapsed_ms(void* start_evt, void* end_evt) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "elapsed_ms")) return 0.0f;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_EVENTS) == 0 || !s.vt->event_elapsed_ms) return 0.0f;

    float ms = 0.0f;
    const ark_gpu_status rc = s.vt->event_elapsed_ms(start_evt, end_evt, &ms);
    if (rc != ARK_GPU_OK && s.log.verbose) {
        s.log.log(LogLevel::Warn, "elapsed_ms", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Warn, "backend", be);
        return 0.0f;
    }
    return ms;
}

void memcpy_h2d_async(void* dst_dev, const void* src_host, std::size_t bytes, void* stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "memcpy_h2d_async")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_ASYNC_MEMCPY) == 0 || !s.vt->memcpy_async) {
        s.log.log(LogLevel::Error, "memcpy_h2d_async", "async memcpy not supported by backend");
        if (backend_required()) die("async memcpy not supported");
        return;
    }

    const ark_gpu_status rc =
        s.vt->memcpy_async(dst_dev, src_host, static_cast<std::int64_t>(bytes), ARK_GPU_COPY_H2D, stream);

    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "memcpy_h2d_async", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        if (backend_required()) die("async memcpy_h2d failed");
    }
}

void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, void* stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "memcpy_d2h_async")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_ASYNC_MEMCPY) == 0 || !s.vt->memcpy_async) {
        s.log.log(LogLevel::Error, "memcpy_d2h_async", "async memcpy not supported by backend");
        if (backend_required()) die("async memcpy not supported");
        return;
    }

    const ark_gpu_status rc =
        s.vt->memcpy_async(dst_host, src_dev, static_cast<std::int64_t>(bytes), ARK_GPU_COPY_D2H, stream);

    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "memcpy_d2h_async", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        if (backend_required()) die("async memcpy_d2h failed");
    }
}

void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, void* stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "memcpy_d2d_async")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_ASYNC_MEMCPY) == 0 || !s.vt->memcpy_async) {
        s.log.log(LogLevel::Error, "memcpy_d2d_async", "async memcpy not supported by backend");
        if (backend_required()) die("async memcpy not supported");
        return;
    }

    const ark_gpu_status rc =
        s.vt->memcpy_async(dst_dev, src_dev, static_cast<std::int64_t>(bytes), ARK_GPU_COPY_D2D, stream);

    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "memcpy_d2d_async", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        if (backend_required()) die("async memcpy_d2d failed");
    }
}

void memset_async(void* dst_dev, int value, std::size_t bytes, void* stream) {
    HalState& s = S();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_ready_locked(s, s.active_device, "memset_async")) return;

    if (!s.vt || (s.vt->features & ARK_GPU_FEAT_ASYNC_MEMSET) == 0 || !s.vt->memset_async) {
        s.log.log(LogLevel::Error, "memset_async", "async memset not supported by backend");
        if (backend_required()) die("async memset not supported");
        return;
    }

    const ark_gpu_status rc = s.vt->memset_async(dst_dev, value, static_cast<std::int64_t>(bytes), stream);
    if (rc != ARK_GPU_OK) {
        s.log.log(LogLevel::Error, "memset_async", std::string("status=") + std::to_string((int)rc));
        const char* be = backend_last_error(s.vt);
        if (be && be[0]) s.log.log(LogLevel::Error, "backend", be);
        if (backend_required()) die("async memset failed");
    }
}

} // namespace hal

// ============================================================================
// ABI exports (compiler/runtime bridge)
// ============================================================================
extern "C" void __ark_gpu_launch(const char* name, void** args, std::int32_t count,
                                 std::int32_t gx, std::int32_t gy, std::int32_t gz,
                                 std::int32_t bx, std::int32_t by, std::int32_t bz,
                                 void* stream) { // [NEW] Added stream argument
                                 
    // Pass the stream directly through instead of hardcoding nullptr
    ark::gpu::launch(name, args, count, gx, gy, gz, bx, by, bz, stream);
}

extern "C" void __ark_device_sync(void) {
    ark::gpu::synchronize(nullptr);
}

// [NEW] Export the managed allocator for GenMIR to call
extern "C" void* __ark_gpu_alloc_managed(std::int64_t bytes) {
    return ark::gpu::alloc_managed(bytes);
}

} // namespace ark::gpu
