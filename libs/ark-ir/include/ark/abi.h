// include/ark/ark_abi.h
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================
// ABI Version
// =========================================================
#define ARK_ABI_VERSION_MAJOR 0
#define ARK_ABI_VERSION_MINOR 1

// =========================================================
// Platform helpers
// =========================================================
#if defined(__GNUC__) || defined(__clang__)
  #define ARK_NORETURN __attribute__((noreturn))
#else
  #define ARK_NORETURN
#endif

#if defined(_MSC_VER)
  #define ARK_INLINE __forceinline
#else
  #define ARK_INLINE static inline __attribute__((always_inline))
#endif

#ifndef ARK_PANIC_CODE
  #define ARK_PANIC_CODE 101
#endif

// =========================================================
// 1) Scalars
// =========================================================
typedef int64_t  ark_index_t;   // canonical index type (signed i64)
typedef uint8_t  ark_bool_t;    // stored as u8 (0/1)

// =========================================================
// 2) str (UTF-8 bytes, not null-terminated)
// =========================================================
typedef struct ArkStr {
    const uint8_t* ptr; // may be NULL iff len==0
    int64_t        len; // bytes
} ArkStr;

typedef ArkStr ark_str_t;

// =========================================================
// 3) vec<T> (owning growable buffer)
// =========================================================
typedef struct ArkVec {
    void*   ptr; // points to T[cap], or NULL iff cap==0
    int64_t len; // elements
    int64_t cap; // elements
} ArkVec;

typedef ArkVec ark_vec_t;

// =========================================================
// 4) slice<T> (non-owning view)
// =========================================================
typedef struct ArkSlice {
    void*   ptr; // points to T[len], may be NULL iff len==0
    int64_t len; // elements
} ArkSlice;

typedef ArkSlice ark_slice_t;

// =========================================================
// 5) Enums / Tags
// =========================================================
typedef int32_t ArkTag;
typedef ArkTag  ark_tag_t;

// =========================================================
// 6) Tagged unions header (payload enums)
// =========================================================
typedef struct ArkUnionHeader {
    int32_t tag;
    int32_t _pad; // keeps payload 8-aligned on 64-bit ABIs
} ArkUnionHeader;

// =========================================================
// 7) Tensor (reserved ABI)
// =========================================================
typedef struct ArkTensor {
    void*    ptr;     // base element storage
    int64_t  rank;    // dims
    int64_t* shape;   // len = rank, elements per dim
    int64_t* stride;  // len = rank, in elements
    uint8_t  device;  // 0=host, 1=gpu (reserved)
    uint8_t  flags;   // reserved
    uint16_t _pad;
} ArkTensor;

typedef ArkTensor ark_tensor_t;

// =========================================================
// Panic / traps (runtime surface)
// =========================================================
typedef void (*ArkPanicHandler)(const char *msg, const char *file, int32_t line, int32_t col);

void arkSetPanicHandler(ArkPanicHandler h);

ARK_NORETURN void arkPanicAt(const char *msg, const char *file, int32_t line, int32_t col);
ARK_NORETURN void arkPanicBounds(const char *file, int32_t line, int32_t col, int64_t idx, int64_t len);
ARK_NORETURN void panic(const char *msg);

// =========================================================
// Host memory
// =========================================================
void* ark_alloc(int64_t size);
void* ark_realloc(void* ptr, int64_t new_size);
void  ark_free(void* ptr);

// =========================================================
// GPU stubs (ABI-stable surface)
// =========================================================
void* ark_gpu_alloc(int64_t size, const char *file, int32_t line, int32_t col);
void* ark_gpu_realloc(void* ptr, int64_t new_size, const char *file, int32_t line, int32_t col);
void  ark_gpu_free(void* ptr);

// =========================================================
// vec runtime helpers
// =========================================================
void ark_vec_reserve_at(ark_vec_t* v,
                        int64_t min_cap,
                        int64_t elem_size,
                        const char *file,
                        int32_t line,
                        int32_t col);

void ark_vec_grow(ark_vec_t* v, int64_t elem_size);

ark_vec_t* ark_vec_create(void);
void       ark_vec_free(ark_vec_t* v);

// =========================================================
// I/O primitives (compiler lowers `print` to these)
// =========================================================
void printStr(const char *s);  // NOTE: runtime currently prints cstr
void printSpace(void);
void printNewline(void);

void printI32(int32_t i);
void printI64(int64_t i);
void printBool(uint8_t b);

void printF32(float f);
void printF32Raw(float f);
void printF64(double d);
void printF64Raw(double d);

// =========================================================
// Debug: bounded vec printers (avoid stdout spam)
// =========================================================
void ark_vec_print_i32(ark_vec_t* v);
void ark_vec_print_i64(ark_vec_t* v);
void ark_vec_print_bool(ark_vec_t* v);

void ark_vec_print_f32(ark_vec_t* v);
void ark_vec_print_f32_raw(ark_vec_t* v);
void ark_vec_print_f64(ark_vec_t* v);
void ark_vec_print_f64_raw(ark_vec_t* v);

void ark_vec_print_str(ark_vec_t* v);

#ifdef __cplusplus
}
#endif
