// runtime/gpu/gpu_fatbin.h
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
  #define ARK_GPU_EXPORT extern "C" __declspec(dllexport)
#else
  #define ARK_GPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ============================================================================
// Ark GPU Fatbin ABI (V1)
// - A sealed capsule exports exactly one symbol:
//     ark_gpu_fatbin_query_v1()
// - Returned pointer is immutable and valid for process lifetime.
// - All structs are versioned + sized for extension.
// ============================================================================

static constexpr std::uint32_t ARK_GPU_FATBIN_V1_MAGIC = 0x4B524146u; // 'AFRK'
static constexpr std::uint32_t ARK_GPU_FATBIN_ABI_V1   = 0x0001'0001u;

struct ark_gpu_fatbin_entry_v1 {
    std::uint32_t size;          // sizeof(ark_gpu_fatbin_entry_v1)
    std::uint32_t module_kind;   // ark_gpu_module_kind (from gpu_backend.h)
    std::uint32_t kernel_count;
    std::uint32_t reserved0;

    const char*   module_key;    // stable cache key, must be non-null/non-empty
    const void*   blob_ptr;      // binary blob or backend-specific bytes
    std::size_t   blob_size;

    const char* const* kernels;  // optional array[kernel_count] of names (may be null)

    void* reserved[8];
};

struct ark_gpu_fatbin_v1 {
    std::uint32_t magic;        // ARK_GPU_FATBIN_V1_MAGIC
    std::uint32_t abi;          // ARK_GPU_FATBIN_ABI_V1
    std::uint32_t size;         // sizeof(ark_gpu_fatbin_v1)
    std::uint32_t entry_count;

    const ark_gpu_fatbin_entry_v1* entries;

    std::uint32_t reserved_u32[8];
    void*         reserved_ptr[8];
};

ARK_GPU_EXPORT const ark_gpu_fatbin_v1* ark_gpu_fatbin_query_v1();

// ============================================================================
// C++-only helpers (NOT part of the exported C ABI)
// ============================================================================

namespace ark::gpu::fatbin::abi {

static inline bool validate_verbose(const ark_gpu_fatbin_v1* fb, const char** out_reason) {
    static const char* OK = "OK";
    if (out_reason) *out_reason = OK;

    if (!fb) { if (out_reason) *out_reason = "fatbin is null"; return false; }
    if (fb->magic != ARK_GPU_FATBIN_V1_MAGIC) { if (out_reason) *out_reason = "bad magic"; return false; }
    if (fb->abi != ARK_GPU_FATBIN_ABI_V1) { if (out_reason) *out_reason = "abi mismatch"; return false; }
    if (fb->size < sizeof(ark_gpu_fatbin_v1)) { if (out_reason) *out_reason = "fatbin size too small"; return false; }

    if (fb->entry_count > 0 && !fb->entries) { if (out_reason) *out_reason = "entries is null"; return false; }

    for (std::uint32_t i = 0; i < fb->entry_count; ++i) {
        const ark_gpu_fatbin_entry_v1& e = fb->entries[i];
        if (e.size < sizeof(ark_gpu_fatbin_entry_v1)) { if (out_reason) *out_reason = "entry size too small"; return false; }
        if (!e.module_key || !e.module_key[0]) { if (out_reason) *out_reason = "entry missing module_key"; return false; }
        if (!e.blob_ptr || e.blob_size == 0) { if (out_reason) *out_reason = "entry missing blob"; return false; }
        if (e.kernel_count > 0 && !e.kernels) { if (out_reason) *out_reason = "entry kernels null"; return false; }
    }

    return true;
}

static inline bool validate(const ark_gpu_fatbin_v1* fb) {
    return validate_verbose(fb, nullptr);
}

static inline const ark_gpu_fatbin_entry_v1* find_first_kind(const ark_gpu_fatbin_v1* fb, std::uint32_t module_kind) {
    if (!fb || !fb->entries) return nullptr;
    for (std::uint32_t i = 0; i < fb->entry_count; ++i) {
        if (fb->entries[i].module_kind == module_kind) return &fb->entries[i];
    }
    return nullptr;
}

} // namespace ark::gpu::fatbin::abi
