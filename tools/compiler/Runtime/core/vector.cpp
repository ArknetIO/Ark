// tools/arkc/Runtime/core/vector.cpp
#include <ark_protocol.h>

#include <cstdint>
#include <climits>
#include <cstring>
#include <cerrno>

static constexpr int64_t ARK_VEC_GROWTH_FLOOR = 8;

static inline ArkStatus vec_err_invalid(void)  { return EINVAL; }
static inline ArkStatus vec_err_range(void)    { return ERANGE; }

#if defined(EOVERFLOW)
static inline ArkStatus vec_err_overflow(void) { return EOVERFLOW; }
#else
static inline ArkStatus vec_err_overflow(void) { return ERANGE; }
#endif

// -----------------------------------------------------------------------------
// Checked integer helpers
// Replaces __int128-based math so the runtime stays MSVC/Windows compatible.
// -----------------------------------------------------------------------------

static inline ArkStatus vec_require_elem_size(int64_t elem_size) {
    return (elem_size > 0) ? 0 : vec_err_invalid();
}

static inline ArkStatus vec_max_cap_for_elem_size(int64_t elem_size, int64_t* out_max_cap) {
    if (!out_max_cap) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;

    *out_max_cap = INT64_MAX / elem_size;
    return 0;
}

static inline ArkStatus vec_mul_i64_checked(int64_t a, int64_t b, int64_t* out) {
    if (!out) return vec_err_invalid();
    if (a < 0 || b < 0) return vec_err_invalid();

    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }

    if (a > INT64_MAX / b) {
        return vec_err_overflow();
    }

    *out = a * b;
    return 0;
}

static inline ArkStatus vec_add_i64_checked(int64_t a, int64_t b, int64_t* out) {
    if (!out) return vec_err_invalid();
    if (a < 0 || b < 0) return vec_err_invalid();

    if (a > INT64_MAX - b) {
        return vec_err_overflow();
    }

    *out = a + b;
    return 0;
}

static inline ArkStatus vec_offset_for_index(int64_t idx, int64_t elem_size, int64_t* out_off) {
    if (idx < 0) return vec_err_invalid();
    return vec_mul_i64_checked(idx, elem_size, out_off);
}

static inline ArkStatus vec_bytes_for_span(int64_t count, int64_t elem_size, int64_t* out_bytes) {
    if (count < 0) return vec_err_invalid();
    return vec_mul_i64_checked(count, elem_size, out_bytes);
}

// -----------------------------------------------------------------------------
// Core validation
// -----------------------------------------------------------------------------

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

static inline ArkStatus vec_bytes_for_cap(int64_t cap, int64_t elem_size, int64_t* out_bytes) {
    if (!out_bytes) return vec_err_invalid();
    if (cap < 0) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;

    if (cap == 0) {
        *out_bytes = 0;
        return 0;
    }

    return vec_mul_i64_checked(cap, elem_size, out_bytes);
}

static inline ArkStatus vec_next_cap(int64_t current_cap,
                                     int64_t min_cap,
                                     int64_t elem_size,
                                     int64_t* out_cap) {
    if (!out_cap) return vec_err_invalid();
    if (min_cap < 0) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;

    if (min_cap <= current_cap) {
        *out_cap = current_cap;
        return 0;
    }

    int64_t max_cap = 0;
    if (ArkStatus st = vec_max_cap_for_elem_size(elem_size, &max_cap)) return st;
    if (min_cap > max_cap) return vec_err_overflow();

    int64_t next = (current_cap > 0) ? current_cap : ARK_VEC_GROWTH_FLOOR;
    if (next < ARK_VEC_GROWTH_FLOOR) {
        next = ARK_VEC_GROWTH_FLOOR;
    }
    if (next > max_cap) {
        next = max_cap;
    }

    while (next < min_cap) {
        if (next > max_cap / 2) {
            next = max_cap;
            break;
        }
        next *= 2;
    }

    if (next < min_cap) {
        return vec_err_overflow();
    }

    *out_cap = next;
    return 0;
}

static inline uint8_t* vec_byte_ptr(void* p) {
    return static_cast<uint8_t*>(p);
}

static inline const uint8_t* vec_cbyte_ptr(const void* p) {
    return static_cast<const uint8_t*>(p);
}

extern "C" {

// -----------------------------------------------------------------------------
// Validity / Construction
// -----------------------------------------------------------------------------

ArkStatus __ark_vec_validate(const ark_vec_t* v) {
    return vec_validate_basic(v);
}

ArkStatus __ark_vec_create(ark_vec_t** out) {
    if (!out) return vec_err_invalid();

    ark_vec_t* v = static_cast<ark_vec_t*>(ark_alloc(static_cast<int64_t>(sizeof(ark_vec_t))));
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

    ark_vec_t* v = static_cast<ark_vec_t*>(ark_alloc(static_cast<int64_t>(sizeof(ark_vec_t))));
    v->ptr = nullptr;
    v->len = 0;
    v->cap = 0;

    if (src->len == 0) {
        *out = v;
        return 0;
    }

    int64_t bytes = 0;
    if (ArkStatus st = vec_bytes_for_cap(src->len, elem_size, &bytes)) {
        ark_free(v);
        return st;
    }

    v->ptr = ark_alloc(bytes);
    v->len = src->len;
    v->cap = src->len;
    std::memcpy(v->ptr, src->ptr, static_cast<size_t>(bytes));

    *out = v;
    return 0;
}

// -----------------------------------------------------------------------------
// Capacity Management
// -----------------------------------------------------------------------------

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

    if (v->cap == 0) {
        v->ptr = nullptr;
    }

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

// -----------------------------------------------------------------------------
// Element Access
// -----------------------------------------------------------------------------

ArkStatus __ark_vec_at(ark_vec_t* v, int64_t idx, int64_t elem_size, void** out_ptr) {
    if (!out_ptr) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    int64_t off = 0;
    if (ArkStatus st = vec_offset_for_index(idx, elem_size, &off)) return st;

    *out_ptr = vec_byte_ptr(v->ptr) + off;
    return 0;
}

ArkStatus __ark_vec_cat(const ark_vec_t* v, int64_t idx, int64_t elem_size, const void** out_ptr) {
    if (!out_ptr) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    int64_t off = 0;
    if (ArkStatus st = vec_offset_for_index(idx, elem_size, &off)) return st;

    *out_ptr = vec_cbyte_ptr(v->ptr) + off;
    return 0;
}

// -----------------------------------------------------------------------------
// Push / Pop
// -----------------------------------------------------------------------------

ArkStatus __ark_vec_push_uninit(ark_vec_t* v, int64_t elem_size, void** out_slot) {
    if (!out_slot) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;

    int64_t new_len = 0;
    if (ArkStatus st = vec_add_i64_checked(v->len, 1, &new_len)) return st;

    if (ArkStatus st = __ark_vec_reserve_at(v, new_len, elem_size)) return st;

    int64_t off = 0;
    if (ArkStatus st = vec_offset_for_index(v->len, elem_size, &off)) return st;

    void* slot = vec_byte_ptr(v->ptr) + off;
    v->len = new_len;
    *out_slot = slot;
    return 0;
}

ArkStatus __ark_vec_push_copy(ark_vec_t* v, const void* elem, int64_t elem_size) {
    if (!elem) return vec_err_invalid();

    void* slot = nullptr;
    if (ArkStatus st = __ark_vec_push_uninit(v, elem_size, &slot)) return st;

    std::memcpy(slot, elem, static_cast<size_t>(elem_size));
    return 0;
}

ArkStatus __ark_vec_pop_copy(ark_vec_t* v, void* out_elem, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (v->len == 0) return vec_err_range();

    const int64_t idx = v->len - 1;

    int64_t off = 0;
    if (ArkStatus st = vec_offset_for_index(idx, elem_size, &off)) return st;

    void* src = vec_byte_ptr(v->ptr) + off;
    if (out_elem) {
        std::memcpy(out_elem, src, static_cast<size_t>(elem_size));
    }

    v->len = idx;
    return 0;
}

// -----------------------------------------------------------------------------
// Insert / Remove
// -----------------------------------------------------------------------------

ArkStatus __ark_vec_insert_uninit(ark_vec_t* v, int64_t idx, int64_t elem_size, void** out_slot) {
    if (!out_slot) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx > v->len) return vec_err_range();

    int64_t new_len = 0;
    if (ArkStatus st = vec_add_i64_checked(v->len, 1, &new_len)) return st;

    if (ArkStatus st = __ark_vec_reserve_at(v, new_len, elem_size)) return st;

    int64_t off = 0;
    if (ArkStatus st = vec_offset_for_index(idx, elem_size, &off)) return st;

    int64_t tail_count = v->len - idx;
    int64_t tail_bytes = 0;
    if (ArkStatus st = vec_bytes_for_span(tail_count, elem_size, &tail_bytes)) return st;

    uint8_t* base = vec_byte_ptr(v->ptr);
    uint8_t* dst = base + off;

    if (tail_bytes != 0) {
        std::memmove(dst + elem_size, dst, static_cast<size_t>(tail_bytes));
    }

    v->len = new_len;
    *out_slot = dst;
    return 0;
}

ArkStatus __ark_vec_insert_copy(ark_vec_t* v, int64_t idx, const void* elem, int64_t elem_size) {
    if (!elem) return vec_err_invalid();

    void* slot = nullptr;
    if (ArkStatus st = __ark_vec_insert_uninit(v, idx, elem_size, &slot)) return st;

    std::memcpy(slot, elem, static_cast<size_t>(elem_size));
    return 0;
}

ArkStatus __ark_vec_remove_shift_copy(ark_vec_t* v, int64_t idx, void* out_elem, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    const int64_t last = v->len - 1;

    int64_t off = 0;
    if (ArkStatus st = vec_offset_for_index(idx, elem_size, &off)) return st;

    uint8_t* base = vec_byte_ptr(v->ptr);
    uint8_t* slot = base + off;

    if (out_elem) {
        std::memcpy(out_elem, slot, static_cast<size_t>(elem_size));
    }

    if (idx != last) {
        int64_t tail_count = last - idx;
        int64_t tail_bytes = 0;
        if (ArkStatus st = vec_bytes_for_span(tail_count, elem_size, &tail_bytes)) return st;

        std::memmove(slot, slot + elem_size, static_cast<size_t>(tail_bytes));
    }

    v->len = last;
    return 0;
}

ArkStatus __ark_vec_swap_remove_copy(ark_vec_t* v, int64_t idx, void* out_elem, int64_t elem_size) {
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (idx < 0 || idx >= v->len) return vec_err_range();

    const int64_t last = v->len - 1;

    int64_t off_idx = 0;
    int64_t off_last = 0;
    if (ArkStatus st = vec_offset_for_index(idx, elem_size, &off_idx)) return st;
    if (ArkStatus st = vec_offset_for_index(last, elem_size, &off_last)) return st;

    uint8_t* base = vec_byte_ptr(v->ptr);
    uint8_t* slot = base + off_idx;
    uint8_t* tail = base + off_last;

    if (out_elem) {
        std::memcpy(out_elem, slot, static_cast<size_t>(elem_size));
    }

    if (idx != last) {
        std::memcpy(slot, tail, static_cast<size_t>(elem_size));
    }

    v->len = last;
    return 0;
}

// -----------------------------------------------------------------------------
// Bulk Append
// -----------------------------------------------------------------------------

ArkStatus __ark_vec_extend_from_raw(ark_vec_t* v, const void* data, int64_t count, int64_t elem_size) {
    if (!data) return vec_err_invalid();
    if (ArkStatus st = vec_require_elem_size(elem_size)) return st;
    if (ArkStatus st = vec_validate_basic(v)) return st;
    if (count < 0) return vec_err_invalid();
    if (count == 0) return 0;

    int64_t new_len = 0;
    if (ArkStatus st = vec_add_i64_checked(v->len, count, &new_len)) return st;

    if (ArkStatus st = __ark_vec_reserve_at(v, new_len, elem_size)) return st;

    int64_t off = 0;
    int64_t bytes = 0;
    if (ArkStatus st = vec_offset_for_index(v->len, elem_size, &off)) return st;
    if (ArkStatus st = vec_bytes_for_span(count, elem_size, &bytes)) return st;

    uint8_t* dst = vec_byte_ptr(v->ptr) + off;
    std::memcpy(dst, data, static_cast<size_t>(bytes));
    v->len = new_len;
    return 0;
}

// ---------------------------------------------------------
// Existing ABI entrypoints (panic-style) kept intact
// ---------------------------------------------------------

ark_vec_t* ark_vec_create(void) {
    ark_vec_t* v = static_cast<ark_vec_t*>(ark_alloc(static_cast<int64_t>(sizeof(ark_vec_t))));
    v->ptr = nullptr;
    v->len = 0;
    v->cap = 0;
    return v;
}

void ark_vec_reserve_at(ark_vec_t* v,
                        int64_t min_cap,
                        int64_t elem_size,
                        const char* f,
                        int32_t l,
                        int32_t c) {
    ArkStatus st = __ark_vec_reserve_at(v, min_cap, elem_size);
    if (st == 0) return;
    if (st == ERANGE) arkPanicAt("vec reserve: out of range", f, l, c);
    if (st == EINVAL) arkPanicAt("vec reserve: invalid argument", f, l, c);
    arkPanicAt("vec reserve: overflow", f, l, c);
}

void ark_vec_grow(ark_vec_t* v,
                  int64_t elem_size,
                  const char* f,
                  int32_t l,
                  int32_t c) {
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

    if (v->ptr) {
        ark_free(v->ptr);
    }

    ark_free(v);
}

} // extern "C"