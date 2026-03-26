#include <ark_protocol.h>
#include <map>
#include <mutex>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstring>   // memcpy
#include <cinttypes> // PRIu64

// -----------------------------------------------------------------------------
// Allocation Tracking (Shadow Table)
// -----------------------------------------------------------------------------

struct AllocRecord {
    void* base;
    size_t size; // Actual allocation size
    uint64_t id;
};

// Global Tracking State
// Key: Base Address (uintptr_t) -> Record
static std::map<uintptr_t, AllocRecord> g_alloc_map;
static std::mutex g_mem_lock;
static uint64_t g_next_alloc_id = 1;

static bool is_alloc_debug() {
    static bool d = (getenv("ARK_ALLOC_DEBUG") != nullptr);
    return d;
}

extern "C" {

// -----------------------------------------------------------------------------
// Helpers: Safe Integer Conversion
// -----------------------------------------------------------------------------

// Fallible check: Returns false if negative or overflows size_t
static inline bool ark_to_size_checked(int64_t n, size_t* out) {
    if (n < 0) return false;
    const uint64_t u = (uint64_t)n;
    if (u > (uint64_t)SIZE_MAX) return false;
    *out = (size_t)u;
    return true;
}

// Infallible check: Panics on negative/overflow, normalizes 0 -> 1
static inline size_t ark_to_size_or_panic_nonzero(int64_t n) {
    if (n < 0) panic("negative allocation size");
    
    // CONTRACT: alloc(0) returns 1 byte to ensure non-null pointer for ownership tracking
    if (n == 0) return (size_t)1;
    
    const uint64_t u = (uint64_t)n;
    if (u > (uint64_t)SIZE_MAX) panic("allocation size overflow");
    
    return (size_t)u;
}

// [CRITICAL FIX] Opaque destructor for compiler-generated drops
void __ark_drop_opaque(void* ptr) {
    // No-op for now. Future: RefCounting or Interface release.
}

// =============================================================================
// 1. Contract Allocators (Tracked)
// =============================================================================

// The official ABI function called by Compiler/GenMIR
void* __ark_alloc(uint64_t bytes, uint64_t align) {
    // Safety clamp for alignment
    if (align < sizeof(void*)) align = sizeof(void*);
    
    // Power of 2 check
    if ((align & (align - 1)) != 0) return nullptr;

    // Round up size to multiple of alignment (Safe for SIMD/Copy)
    if (bytes % align != 0) {
        bytes = (bytes + align - 1) & ~(align - 1);
    }
    
    // Ensure 0-byte allocs get at least something (so they have a unique address)
    if (bytes == 0) bytes = align;

    void* ptr = nullptr;
    // Using posix_memalign for strict alignment requirements
    if (posix_memalign(&ptr, (size_t)align, (size_t)bytes) != 0) return nullptr;

    if (is_alloc_debug()) {
        printf("[Ark] Alloc base=%p size=%" PRIu64 "\n", ptr, bytes);
    }

    {
        std::lock_guard<std::mutex> lock(g_mem_lock);
        g_alloc_map[(uintptr_t)ptr] = AllocRecord{ptr, (size_t)bytes, g_next_alloc_id++};
    }
    
    return ptr;
}

// The official ABI function called by Compiler/GenMIR
void __ark_free(void* ptr) {
    if (!ptr) return;

    {
        std::lock_guard<std::mutex> lock(g_mem_lock);
        auto it = g_alloc_map.find((uintptr_t)ptr);
        
        if (it == g_alloc_map.end()) {
            fprintf(stderr, "[Ark] FATAL: Invalid free or double free: %p\n", ptr);
            abort();
        }

        if (is_alloc_debug()) {
            printf("[Ark] Free base=%p id=%" PRIu64 "\n", ptr, it->second.id);
        }

        g_alloc_map.erase(it);
    }
    
    free(ptr);
}

// Internal free alias (required by some codegen paths)
void __ark_free_internal(void* p) {
    __ark_free(p);
}

// Public Resolver for Remote Packer
// Used to translate local pointers to (ID, Offset) for network transmission
bool ark_resolve_ptr(void* ptr, uint64_t* out_id, uint64_t* out_off, uint64_t* out_len) {
    std::lock_guard<std::mutex> lock(g_mem_lock);
    uintptr_t p = (uintptr_t)ptr;

    // Find first allocation starting AFTER p
    auto it = g_alloc_map.upper_bound(p);
    
    // If it's the beginning, p is before everything -> Invalid
    if (it == g_alloc_map.begin()) return false;
    
    --it; // Step back to the candidate
    
    const AllocRecord& rec = it->second;
    uintptr_t base = (uintptr_t)rec.base;
    
    // Overflow safe range check
    if (rec.size > UINTPTR_MAX - base) return false; 
    uintptr_t end = base + rec.size;

    if (p >= base && p < end) {
        if (out_id)  *out_id  = rec.id;
        if (out_off) *out_off = (p - base);
        if (out_len) *out_len = (end - p);
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
    void* p = __ark_alloc((uint64_t)n, 16); 
    
    if (!p) panic("out of memory");
    return p;
}

// User-facing "realloc(ptr, new_size)"
// Note: Realloc is complex with shadow tables (it changes base). 
// For MVP, we do alloc + copy + free to ensure correct ID tracking.
void* ark_realloc(void* ptr, int64_t new_size) {
    const size_t n = ark_to_size_or_panic_nonzero(new_size);
    
    if (!ptr) return ark_alloc(new_size);

    // 1. Resolve Old Size (Required for Copy)
    uint64_t id, off, old_len;
    if (!ark_resolve_ptr(ptr, &id, &off, &old_len)) {
        panic("realloc: invalid pointer");
    }
    
    // 2. Allocate New
    void* new_ptr = __ark_alloc((uint64_t)n, 16);
    if (!new_ptr) panic("out of memory");

    // 3. Copy Data
    // We copy the minimum of old and new sizes
    // Note: old_len is valid bytes remaining from ptr. Since ptr is base (checked below), it's total size.
    if (off != 0) panic("realloc: cannot realloc interior pointer");
    
    size_t copy_size = (n < old_len) ? n : old_len;
    std::memcpy(new_ptr, ptr, copy_size);

    // 4. Free Old
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
    if (!vec) return;
    
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