#include <ark_protocol.h>

#include <map>
#include <mutex>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cinttypes>

#if defined(_WIN32)
#include <malloc.h>
#endif

// -----------------------------------------------------------------------------
// Allocation Tracking (Shadow Table)
// -----------------------------------------------------------------------------

struct AllocRecord {
    void* base;
    size_t size;
    uint64_t id;
};

// Global Tracking State
// Key: Base Address (uintptr_t) -> Record
static std::map<uintptr_t, AllocRecord> g_alloc_map;
static std::mutex g_mem_lock;
static uint64_t g_next_alloc_id = 1;

static bool is_alloc_debug() {
    static bool d = (std::getenv("ARK_ALLOC_DEBUG") != nullptr);
    return d;
}

// -----------------------------------------------------------------------------
// Platform Aligned Allocation Helpers
// Windows: _aligned_malloc / _aligned_free
// POSIX:   posix_memalign / free
// -----------------------------------------------------------------------------

static inline bool ark_is_power_of_two_u64(uint64_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

static inline void* ark_platform_aligned_alloc(size_t bytes, size_t align) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static inline void ark_platform_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

extern "C" {

// -----------------------------------------------------------------------------
// Helpers: Safe Integer Conversion
// -----------------------------------------------------------------------------

// Fallible check: Returns false if negative or overflows size_t
static inline bool ark_to_size_checked(int64_t n, size_t* out) {
    if (n < 0) return false;

    const uint64_t u = static_cast<uint64_t>(n);
    if (u > static_cast<uint64_t>(SIZE_MAX)) return false;

    *out = static_cast<size_t>(u);
    return true;
}

// Infallible check: Panics on negative/overflow, normalizes 0 -> 1
static inline size_t ark_to_size_or_panic_nonzero(int64_t n) {
    if (n < 0) {
        panic("negative allocation size");
    }

    // CONTRACT: alloc(0) returns 1 byte to ensure non-null pointer for ownership tracking
    if (n == 0) {
        return static_cast<size_t>(1);
    }

    const uint64_t u = static_cast<uint64_t>(n);
    if (u > static_cast<uint64_t>(SIZE_MAX)) {
        panic("allocation size overflow");
    }

    return static_cast<size_t>(u);
}

// [CRITICAL FIX] Opaque destructor for compiler-generated drops
void __ark_drop_opaque(void* ptr) {
    (void)ptr;
    // No-op for now. Future: RefCounting or Interface release.
}

// =============================================================================
// 1. Contract Allocators (Tracked)
// =============================================================================

// The official ABI function called by Compiler/GenMIR
void* __ark_alloc(uint64_t bytes, uint64_t align) {
    // Safety clamp for alignment
    if (align < sizeof(void*)) {
        align = sizeof(void*);
    }

    // Power of 2 check
    if (!ark_is_power_of_two_u64(align)) {
        return nullptr;
    }

    // Round up size to multiple of alignment
    // This keeps layout predictable for vectorized and structured accesses.
    if (bytes % align != 0) {
        bytes = (bytes + align - 1) & ~(align - 1);
    }

    // Ensure 0-byte allocs get at least something
    if (bytes == 0) {
        bytes = align;
    }

    // Guard conversion to host allocation sizes
    if (bytes > static_cast<uint64_t>(SIZE_MAX) ||
        align > static_cast<uint64_t>(SIZE_MAX)) {
        return nullptr;
    }

    void* ptr = ark_platform_aligned_alloc(
        static_cast<size_t>(bytes),
        static_cast<size_t>(align)
    );

    if (!ptr) {
        return nullptr;
    }

    if (is_alloc_debug()) {
        std::printf("[Ark] Alloc base=%p size=%" PRIu64 " align=%" PRIu64 "\n", ptr, bytes, align);
    }

    {
        std::lock_guard<std::mutex> lock(g_mem_lock);
        g_alloc_map[reinterpret_cast<uintptr_t>(ptr)] = AllocRecord{
            ptr,
            static_cast<size_t>(bytes),
            g_next_alloc_id++
        };
    }

    return ptr;
}

// The official ABI function called by Compiler/GenMIR
void __ark_free(void* ptr) {
    if (!ptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_mem_lock);
        auto it = g_alloc_map.find(reinterpret_cast<uintptr_t>(ptr));

        if (it == g_alloc_map.end()) {
            std::fprintf(stderr, "[Ark] FATAL: Invalid free or double free: %p\n", ptr);
            std::abort();
        }

        if (is_alloc_debug()) {
            std::printf("[Ark] Free base=%p id=%" PRIu64 "\n", ptr, it->second.id);
        }

        g_alloc_map.erase(it);
    }

    ark_platform_aligned_free(ptr);
}

// Internal free alias (required by some codegen paths)
void __ark_free_internal(void* p) {
    __ark_free(p);
}

// Public Resolver for Remote Packer
// Used to translate local pointers to (ID, Offset) for network transmission
bool ark_resolve_ptr(void* ptr, uint64_t* out_id, uint64_t* out_off, uint64_t* out_len) {
    std::lock_guard<std::mutex> lock(g_mem_lock);
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);

    // Find first allocation starting AFTER p
    auto it = g_alloc_map.upper_bound(p);

    // If it's the beginning, p is before everything -> Invalid
    if (it == g_alloc_map.begin()) {
        return false;
    }

    --it;

    const AllocRecord& rec = it->second;
    const uintptr_t base = reinterpret_cast<uintptr_t>(rec.base);

    // Overflow safe range check
    if (rec.size > UINTPTR_MAX - base) {
        return false;
    }

    const uintptr_t end = base + rec.size;

    if (p >= base && p < end) {
        if (out_id)  *out_id = rec.id;
        if (out_off) *out_off = static_cast<uint64_t>(p - base);
        if (out_len) *out_len = static_cast<uint64_t>(end - p);
        return true;
    }

    return false;
}

// =============================================================================
// 2. User Allocators (Infallible / Panic Wrappers)
// =============================================================================

// User-facing "alloc(size)"
void* ark_alloc(int64_t size) {
    // 1. Safe Size Conversion
    const size_t n = ark_to_size_or_panic_nonzero(size);

    // 2. Delegate to Tracked Allocator
    // Using 16-byte alignment as a safe default for scalars/structs
    void* p = __ark_alloc(static_cast<uint64_t>(n), 16);

    if (!p) {
        panic("out of memory");
    }

    return p;
}

// User-facing "realloc(ptr, new_size)"
// Realloc is modeled as alloc + copy + free so tracking remains correct.
void* ark_realloc(void* ptr, int64_t new_size) {
    const size_t n = ark_to_size_or_panic_nonzero(new_size);

    if (!ptr) {
        return ark_alloc(new_size);
    }

    // 1. Resolve Old Size
    uint64_t id = 0;
    uint64_t off = 0;
    uint64_t old_len = 0;

    if (!ark_resolve_ptr(ptr, &id, &off, &old_len)) {
        panic("realloc: invalid pointer");
    }

    // 2. Realloc only supports base pointers
    if (off != 0) {
        panic("realloc: cannot realloc interior pointer");
    }

    // 3. Allocate New
    void* new_ptr = __ark_alloc(static_cast<uint64_t>(n), 16);
    if (!new_ptr) {
        panic("out of memory");
    }

    // 4. Copy Data
    const size_t copy_size = (n < static_cast<size_t>(old_len))
        ? n
        : static_cast<size_t>(old_len);

    std::memcpy(new_ptr, ptr, copy_size);

    // 5. Free Old
    __ark_free(ptr);

    return new_ptr;
}

// User-facing "free(ptr)"
void ark_free(void* ptr) {
    __ark_free(ptr);
}

// =============================================================================
// 3. Vector Resource Management (Backend Hooks)
// =============================================================================

void __ark_drop_vec_opaque(ark_vec_t* vec) {
    if (!vec) {
        return;
    }

    if (vec->ptr) {
        // Use tracked free
        __ark_free(vec->ptr);
        vec->ptr = nullptr;
    }

    vec->len = 0;
    vec->cap = 0;
}

// Typed wrappers (ABI compatibility)
void __ark_drop_vec_i32(ark_vec_t* vec) { __ark_drop_vec_opaque(vec); }
void __ark_drop_vec_i64(ark_vec_t* vec) { __ark_drop_vec_opaque(vec); }
void __ark_drop_vec_f32(ark_vec_t* vec) { __ark_drop_vec_opaque(vec); }
void __ark_drop_vec_f64(ark_vec_t* vec) { __ark_drop_vec_opaque(vec); }
void __ark_drop_vec_str(ark_vec_t* vec) { __ark_drop_vec_opaque(vec); }

} // extern "C"