// tools/arkc/Runtime/core/vector.cpp
#include <ark_protocol.h>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cerrno>

static constexpr int64_t ARK_VEC_GROWTH_FLOOR = 8;

static inline ArkStatus vec_err_invalid(void) { return EINVAL; }
static inline ArkStatus vec_err_range(void)   { return ERANGE; }

#if defined(EOVERFLOW)
static inline ArkStatus vec_err_overflow(void){ return EOVERFLOW; }
#else
static inline ArkStatus vec_err_overflow(void){ return ERANGE; }
#endif

static inline unsigned __int128 vec_max_cap_u128(int64_t elem_size) {
    return (unsigned __int128)INT64_MAX / (unsigned __int128)elem_size;
}

static inline ArkStatus vec_validate_basic(const ark_vec_t* v) {
    if (!v) return vec_err_invalid();
    if (v->len < 0 || v->cap < 0) return vec_err_invalid();
    if (v->len > v->cap) return vec_err_invalid();
    if (v->cap == 0) {
        if (v->ptr != nullptr) return vec_err_invalid();
    } else {
        if (v->ptr == nullptr) return vec_err_invalid();
    }
    return 0;
}

static inline ArkStatus vec_require_elem_size(int64_t elem_size) {
    return (elem_size > 0) ? 0 : vec_err_invalid();
}

static inline ArkStatus vec_bytes_for_cap(int64_t cap, int64_t elem_size, int64_t* out_bytes) {
    if (cap < 0) return vec_err_invalid();
    if (cap == 0) { *out_bytes = 0; return 0; }

    const unsigned __int128 u_cap = (unsigned __int128)cap;
    const unsigned __int128 u_max = vec_max_cap_u128(elem_size);
    if (u_cap > u_max) return vec_err_overflow();

    const unsigned __int128 bytes = u_cap * (unsigned __int128)elem_size;
    if (bytes > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    *out_bytes = (int64_t)bytes;
    return 0;
}

static inline ArkStatus vec_next_cap(int64_t current_cap, int64_t min_cap, int64_t elem_size, int64_t* out_cap) {
    if (min_cap < 0) return vec_err_invalid();
    if (min_cap <= current_cap) { *out_cap = current_cap; return 0; }

    const unsigned __int128 u_min = (unsigned __int128)min_cap;
    const unsigned __int128 u_max = vec_max_cap_u128(elem_size);
    if (u_min > u_max) return vec_err_overflow();

    unsigned __int128 next = (current_cap > 0)
        ? (unsigned __int128)current_cap
        : (unsigned __int128)ARK_VEC_GROWTH_FLOOR;

    if (next < (unsigned __int128)ARK_VEC_GROWTH_FLOOR) next = (unsigned __int128)ARK_VEC_GROWTH_FLOOR;

    while (next < u_min) {
        if (next > u_max / 2) { next = u_max; break; }
        next *= 2;
    }

    if (next < u_min) return vec_err_overflow();
    if (next > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    *out_cap = (int64_t)next;
    return 0;
}

static inline uint8_t* vec_byte_ptr(void* p) { return (uint8_t*)p; }
static inline const uint8_t* vec_cbyte_ptr(const void* p) { return (const uint8_t*)p; }

extern "C" {

ArkStatus __ark_vec_validate(const ark_vec_t* v) {
    return vec_validate_basic(v);
}

ArkStatus __ark_vec_create(ark_vec_t** out) {
    if (!out) return vec_err_invalid();
    ark_vec_t* v = (ark_vec_t*)ark_alloc((int64_t)sizeof(ark_vec_t));
    v->ptr = nullptr;
    v->len = 0;
    v->cap = 0;
    *out = v;
    return 0;
}

ArkStatus __ark_vec_clone(const ark_vec_t* src, int64_t elem_size, ark_vec_t** out) {
    if (!out) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(src)) return st;

    ark_vec_t* v = (ark_vec_t*)ark_alloc((int64_t)sizeof(ark_vec_t));
    v->ptr = nullptr;
    v->len = 0;
    v->cap = 0;

    if (src->len == 0) { *out = v; return 0; }

    int64_t bytes = 0;
    if (ArkStatus st = vec_bytes_for_cap(src->len, elem_size, &bytes)) { ark_free(v); return st; }

    v->ptr = ark_alloc(bytes);
    v->len = src->len;
    v->cap = src->len;
    std::memcpy(v->ptr, src->ptr, (size_t)bytes);

    *out = v;
    return 0;
}

ArkStatus __ark_vec_reserve_at(ark_vec_t* v, int64_t min_cap, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (min_cap < 0) return vec_err_invalid();
    if (v->cap >= min_cap) return 0;

    int64_t new_cap = 0;
    if (ArkStatus st = vec_next_cap(v->cap, min_cap, elem_size, &new_cap)) return st;

    int64_t bytes = 0;
    if (ArkStatus st = vec_bytes_for_cap(new_cap, elem_size, &bytes)) return st;

    void* new_ptr = (bytes == 0) ? nullptr : ark_realloc(v->ptr, bytes);
    v->ptr = new_ptr;
    v->cap = new_cap;
    if (v->cap == 0) v->ptr = nullptr;

    return 0;
}

ArkStatus __ark_vec_shrink_to_fit(ark_vec_t* v, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;

    if (v->len == v->cap) return 0;

    if (v->len == 0) {
        if (v->ptr) ark_free(v->ptr);
        v->ptr = nullptr;
        v->cap = 0;
        return 0;
    }

    int64_t bytes = 0;
    if (ArkStatus st = vec_bytes_for_cap(v->len, elem_size, &bytes)) return st;

    v->ptr = ark_realloc(v->ptr, bytes);
    v->cap = v->len;
    return 0;
}

ArkStatus __ark_vec_clear(ark_vec_t* v) {
    if (ArkStatus st = vec_validate_basic(v)) return st;
    v->len = 0;
    return 0;
}

ArkStatus __ark_vec_at(ark_vec_t* v, int64_t idx, int64_t elem_size, void** out_ptr) {
    if (!out_ptr) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    const unsigned __int128 off = (unsigned __int128)idx * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    *out_ptr = vec_byte_ptr(v->ptr) + (int64_t)off;
    return 0;
}

ArkStatus __ark_vec_cat(const ark_vec_t* v, int64_t idx, int64_t elem_size, const void** out_ptr) {
    if (!out_ptr) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    const unsigned __int128 off = (unsigned __int128)idx * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    *out_ptr = vec_cbyte_ptr(v->ptr) + (int64_t)off;
    return 0;
}

ArkStatus __ark_vec_push_uninit(ark_vec_t* v, int64_t elem_size, void** out_slot) {
    if (!out_slot) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;

    if (v->len == INT64_MAX) return vec_err_overflow();
    const int64_t new_len = v->len + 1;

    if (ArkStatus st = __ark_vec_reserve_at(v, new_len, elem_size)) return st;

    const unsigned __int128 off = (unsigned __int128)v->len * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    void* slot = vec_byte_ptr(v->ptr) + (int64_t)off;
    v->len = new_len;
    *out_slot = slot;
    return 0;
}

ArkStatus __ark_vec_push_copy(ark_vec_t* v, const void* elem, int64_t elem_size) {
    if (!elem) return vec_err_invalid();
    void* slot = nullptr;
    if (ArkStatus st = __ark_vec_push_uninit(v, elem_size, &slot)) return st;
    std::memcpy(slot, elem, (size_t)elem_size);
    return 0;
}

ArkStatus __ark_vec_pop_copy(ark_vec_t* v, void* out_elem, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (v->len == 0) return vec_err_range();

    const int64_t idx = v->len - 1;

    const unsigned __int128 off = (unsigned __int128)idx * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    void* src = vec_byte_ptr(v->ptr) + (int64_t)off;
    if (out_elem) std::memcpy(out_elem, src, (size_t)elem_size);

    v->len = idx;
    return 0;
}

ArkStatus __ark_vec_insert_uninit(ark_vec_t* v, int64_t idx, int64_t elem_size, void** out_slot) {
    if (!out_slot) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx > v->len) return vec_err_range();

    if (v->len == INT64_MAX) return vec_err_overflow();
    const int64_t new_len = v->len + 1;

    if (ArkStatus st = __ark_vec_reserve_at(v, new_len, elem_size)) return st;

    const unsigned __int128 off = (unsigned __int128)idx * (unsigned __int128)elem_size;
    const unsigned __int128 tail_bytes = (unsigned __int128)(v->len - idx) * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();
    if (tail_bytes > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    uint8_t* base = vec_byte_ptr(v->ptr);
    uint8_t* dst = base + (int64_t)off;

    if (tail_bytes != 0) {
        std::memmove(dst + elem_size, dst, (size_t)tail_bytes);
    }

    v->len = new_len;
    *out_slot = dst;
    return 0;
}

ArkStatus __ark_vec_insert_copy(ark_vec_t* v, int64_t idx, const void* elem, int64_t elem_size) {
    if (!elem) return vec_err_invalid();
    void* slot = nullptr;
    if (ArkStatus st = __ark_vec_insert_uninit(v, idx, elem_size, &slot)) return st;
    std::memcpy(slot, elem, (size_t)elem_size);
    return 0;
}

ArkStatus __ark_vec_remove_shift_copy(ark_vec_t* v, int64_t idx, void* out_elem, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    const int64_t last = v->len - 1;

    const unsigned __int128 off = (unsigned __int128)idx * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    uint8_t* base = vec_byte_ptr(v->ptr);
    uint8_t* slot = base + (int64_t)off;

    if (out_elem) std::memcpy(out_elem, slot, (size_t)elem_size);

    if (idx != last) {
        const unsigned __int128 tail_bytes = (unsigned __int128)(last - idx) * (unsigned __int128)elem_size;
        if (tail_bytes > (unsigned __int128)INT64_MAX) return vec_err_overflow();
        std::memmove(slot, slot + elem_size, (size_t)tail_bytes);
    }

    v->len = last;
    return 0;
}

ArkStatus __ark_vec_swap_remove_copy(ark_vec_t* v, int64_t idx, void* out_elem, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    const int64_t last = v->len - 1;

    const unsigned __int128 off_idx = (unsigned __int128)idx * (unsigned __int128)elem_size;
    const unsigned __int128 off_last = (unsigned __int128)last * (unsigned __int128)elem_size;
    if (off_idx > (unsigned __int128)INT64_MAX) return vec_err_overflow();
    if (off_last > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    uint8_t* base = vec_byte_ptr(v->ptr);
    uint8_t* slot = base + (int64_t)off_idx;
    uint8_t* tail = base + (int64_t)off_last;

    if (out_elem) std::memcpy(out_elem, slot, (size_t)elem_size);

    if (idx != last) {
        std::memcpy(slot, tail, (size_t)elem_size);
    }

    v->len = last;
    return 0;
}

ArkStatus __ark_vec_extend_from_raw(ark_vec_t* v, const void* data, int64_t count, int64_t elem_size) {
    if (!data) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (count < 0) return vec_err_invalid();
    if (count == 0) return 0;

    if (v->len > INT64_MAX - count) return vec_err_overflow();
    const int64_t new_len = v->len + count;

    if (ArkStatus st = __ark_vec_reserve_at(v, new_len, elem_size)) return st;

    const unsigned __int128 off = (unsigned __int128)v->len * (unsigned __int128)elem_size;
    const unsigned __int128 bytes = (unsigned __int128)count * (unsigned __int128)elem_size;
    if (off > (unsigned __int128)INT64_MAX) return vec_err_overflow();
    if (bytes > (unsigned __int128)INT64_MAX) return vec_err_overflow();

    uint8_t* dst = vec_byte_ptr(v->ptr) + (int64_t)off;
    std::memcpy(dst, data, (size_t)bytes);
    v->len = new_len;
    return 0;
}

// ---------------------------------------------------------
// Existing ABI entrypoints (panic-style) kept intact
// ---------------------------------------------------------

ark_vec_t* ark_vec_create(void) {
    ark_vec_t* v = (ark_vec_t*)ark_alloc((int64_t)sizeof(ark_vec_t));
    v->ptr = nullptr;
    v->len = 0;
    v->cap = 0;
    return v;
}

void ark_vec_reserve_at(ark_vec_t* v, int64_t min_cap, int64_t elem_size,
                        const char* f, int32_t l, int32_t c) {
    ArkStatus st = __ark_vec_reserve_at(v, min_cap, elem_size);
    if (st == 0) return;
    if (st == ERANGE) arkPanicAt("vec reserve: out of range", f, l, c);
    if (st == EINVAL) arkPanicAt("vec reserve: invalid argument", f, l, c);
    arkPanicAt("vec reserve: overflow", f, l, c);
}

void ark_vec_grow(ark_vec_t* v, int64_t elem_size,
                  const char* f, int32_t l, int32_t c) {
    if (!v) arkPanicAt("vec grow: null pointer", f, l, c);
    if (elem_size <= 0) arkPanicAt("vec grow: invalid element size", f, l, c);

    const int64_t need = (v->len < INT64_MAX) ? (v->len + 1) : v->len;
    ArkStatus st = __ark_vec_reserve_at(v, need, elem_size);
    if (st == 0) return;
    if (st == EINVAL) arkPanicAt("vec grow: invalid argument", f, l, c);
    arkPanicAt("vec grow: overflow", f, l, c);
}

void ark_vec_free(ark_vec_t* v) {
    if (!v) return;
    if (v->ptr) ark_free(v->ptr);
    ark_free(v);
}

} // extern "C"
