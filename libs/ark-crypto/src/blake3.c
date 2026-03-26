#include "../include/blake3.h"
#include <string.h>

/*
 * =========================================================
 * Internal Constants & Helpers
 * =========================================================
 */

#define BLAKE3_IV_0 0x6A09E667UL
#define BLAKE3_IV_1 0xBB67AE85UL
#define BLAKE3_IV_2 0x3C6EF372UL
#define BLAKE3_IV_3 0xA54FF53AUL
#define BLAKE3_IV_4 0x510E527FUL
#define BLAKE3_IV_5 0x9B05688CUL
#define BLAKE3_IV_6 0x1F83D9ABUL
#define BLAKE3_IV_7 0x5BE0CD19UL

#define CHUNK_START         (1u << 0)
#define CHUNK_END           (1u << 1)
#define PARENT              (1u << 2)
#define ROOT                (1u << 3)
#define KEYED_HASH          (1u << 4)
#define DERIVE_KEY_CONTEXT  (1u << 5)
#define DERIVE_KEY_MATERIAL (1u << 6)

static const uint32_t BLAKE3_IV[8] = {
  BLAKE3_IV_0, BLAKE3_IV_1, BLAKE3_IV_2, BLAKE3_IV_3,
  BLAKE3_IV_4, BLAKE3_IV_5, BLAKE3_IV_6, BLAKE3_IV_7,
};

static const uint8_t MSG_SCHEDULE[7][16] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
  {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
  {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
  {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 4, 0, 6, 1, 3},
  {9, 14, 11, 5, 8, 12, 15, 1, 13, 4, 0, 10, 7, 3, 6, 2},
  {11, 8, 5, 0, 1, 9, 14, 6, 15, 13, 4, 2, 7, 10, 3, 12},
};

static inline uint32_t load32(const void *src) {
  const uint8_t *p = (const uint8_t *)src;
  return ((uint32_t)p[0] << 0) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

static inline uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

static inline void g(uint32_t *s, size_t a, size_t b, size_t c, size_t d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}

static void round_function(uint32_t state[16], const uint32_t msg[16], size_t round) {
  const uint8_t *schedule = MSG_SCHEDULE[round];
  g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
  g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
  g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
  g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);
  g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
  g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
  g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
  g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

static void words_from_little_endian_bytes(const void *bytes, size_t bytes_len, uint32_t *out_words) {
  const uint8_t *p = (const uint8_t *)bytes;
  for (size_t i = 0; i < bytes_len / 4; i++) {
    out_words[i] = load32(p + 4 * i);
  }
}

static void compress_words(const uint32_t chaining_value[8],
                           const uint32_t block_words[16],
                           uint64_t counter,
                           uint32_t block_len,
                           uint32_t flags,
                           uint32_t out_words[16]) {
  uint32_t state[16] = {
    chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
    chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
    BLAKE3_IV_0,       BLAKE3_IV_1,       BLAKE3_IV_2,       BLAKE3_IV_3,
    (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags,
  };

  round_function(state, block_words, 0);
  round_function(state, block_words, 1);
  round_function(state, block_words, 2);
  round_function(state, block_words, 3);
  round_function(state, block_words, 4);
  round_function(state, block_words, 5);
  round_function(state, block_words, 6);

  for (size_t i = 0; i < 8; i++) {
    state[i] ^= state[i + 8];
    state[i + 8] ^= chaining_value[i];
  }

  memcpy(out_words, state, 16 * sizeof(uint32_t));
}

static void compress_xof(const uint32_t cv[8],
                         const uint8_t block[BLAKE3_BLOCK_LEN],
                         uint8_t block_len,
                         uint64_t counter,
                         uint8_t flags,
                         uint8_t out[BLAKE3_BLOCK_LEN]) {
  uint32_t block_words[16];
  words_from_little_endian_bytes(block, BLAKE3_BLOCK_LEN, block_words);

  uint32_t out_words[16];
  compress_words(cv, block_words, counter, (uint32_t)block_len, (uint32_t)flags, out_words);

  for (size_t i = 0; i < 16; i++) {
    store32(out + 4 * i, out_words[i]);
  }
}

/*
 * =========================================================
 * Output Object (captures final compression inputs)
 * =========================================================
 */

typedef struct {
  uint32_t input_cv[8];
  uint8_t block[BLAKE3_BLOCK_LEN];
  uint8_t block_len;
  uint64_t counter;
  uint8_t flags;
} blake3_output_t;

static inline blake3_output_t make_output(const uint32_t input_cv[8],
                                          const uint8_t block[BLAKE3_BLOCK_LEN],
                                          uint8_t block_len,
                                          uint64_t counter,
                                          uint8_t flags) {
  blake3_output_t o;
  memcpy(o.input_cv, input_cv, 8 * sizeof(uint32_t));
  memcpy(o.block, block, BLAKE3_BLOCK_LEN);
  o.block_len = block_len;
  o.counter = counter;
  o.flags = flags;
  return o;
}

static inline void output_chaining_value(const blake3_output_t *o, uint8_t cv[32]) {
  uint8_t wide[BLAKE3_BLOCK_LEN];
  compress_xof(o->input_cv, o->block, o->block_len, o->counter, o->flags, wide);
  memcpy(cv, wide, 32);
}

static inline void output_root_bytes(const blake3_output_t *o, uint8_t *out, size_t out_len) {
  if (out_len == 0) return;

  uint64_t output_block_counter = 0;
  while (out_len > 0) {
    uint8_t wide[BLAKE3_BLOCK_LEN];
    compress_xof(o->input_cv, o->block, o->block_len, output_block_counter, (uint8_t)(o->flags | ROOT), wide);

    const size_t take = out_len < BLAKE3_BLOCK_LEN ? out_len : BLAKE3_BLOCK_LEN;
    memcpy(out, wide, take);

    out += take;
    out_len -= take;
    output_block_counter += 1;
  }
}

/*
 * =========================================================
 * Chunk Management
 * =========================================================
 */

static void blake3_chunk_state_reset(blake3_chunk_state *self, const uint32_t key[8], uint64_t chunk_counter) {
  memcpy(self->cv, key, 32);
  self->chunk_counter = chunk_counter;
  self->buf_len = 0;
  self->blocks_compressed = 0;
  /* self->flags must be set by caller */
}

static size_t blake3_chunk_state_len(const blake3_chunk_state *self) {
  return (size_t)BLAKE3_BLOCK_LEN * (size_t)self->blocks_compressed + (size_t)self->buf_len;
}

static inline uint8_t blake3_chunk_state_maybe_start_flag(const blake3_chunk_state *self) {
  return self->blocks_compressed == 0 ? (uint8_t)CHUNK_START : 0;
}

static size_t blake3_chunk_state_fill_buf(blake3_chunk_state *self, const uint8_t *input, size_t input_len) {
  const size_t avail = (size_t)BLAKE3_BLOCK_LEN - (size_t)self->buf_len;
  const size_t take = input_len < avail ? input_len : avail;
  memcpy(self->buf + self->buf_len, input, take);
  self->buf_len = (uint8_t)(self->buf_len + (uint8_t)take);
  return take;
}

static void blake3_chunk_state_compress_buf(blake3_chunk_state *self) {
  uint32_t block_words[16];
  words_from_little_endian_bytes(self->buf, BLAKE3_BLOCK_LEN, block_words);

  uint32_t out_words[16];
  compress_words(self->cv, block_words, self->chunk_counter, BLAKE3_BLOCK_LEN,
                 (uint32_t)(self->flags | blake3_chunk_state_maybe_start_flag(self)), out_words);

  memcpy(self->cv, out_words, 8 * sizeof(uint32_t));
  self->blocks_compressed += 1;
  self->buf_len = 0;
}

static void blake3_chunk_state_update(blake3_chunk_state *self, const uint8_t *input, size_t input_len) {
  if (input_len == 0) return;

  /* If there's buffered data, fill it first. */
  if (self->buf_len > 0) {
    const size_t take = blake3_chunk_state_fill_buf(self, input, input_len);
    input += take;
    input_len -= take;

    if (self->buf_len == BLAKE3_BLOCK_LEN) {
      blake3_chunk_state_compress_buf(self);
    }
  }

  /* Compress whole blocks directly from input, avoiding memcpy. */
  while (input_len > BLAKE3_BLOCK_LEN) {
    uint32_t block_words[16];
    words_from_little_endian_bytes(input, BLAKE3_BLOCK_LEN, block_words);

    uint32_t out_words[16];
    compress_words(self->cv, block_words, self->chunk_counter, BLAKE3_BLOCK_LEN,
                   (uint32_t)(self->flags | blake3_chunk_state_maybe_start_flag(self)), out_words);

    memcpy(self->cv, out_words, 8 * sizeof(uint32_t));
    self->blocks_compressed += 1;

    input += BLAKE3_BLOCK_LEN;
    input_len -= BLAKE3_BLOCK_LEN;
  }

  /* Buffer the remainder (0..64 bytes). */
  if (input_len > 0) {
    (void)blake3_chunk_state_fill_buf(self, input, input_len);
  }
}

static blake3_output_t blake3_chunk_state_output(const blake3_chunk_state *self) {
  uint8_t block_flags = (uint8_t)(self->flags | blake3_chunk_state_maybe_start_flag(self) | CHUNK_END);

  uint8_t block[BLAKE3_BLOCK_LEN];
  memcpy(block, self->buf, BLAKE3_BLOCK_LEN);

  return make_output(self->cv, block, self->buf_len, self->chunk_counter, block_flags);
}

/*
 * =========================================================
 * Parent Logic
 * =========================================================
 */

static inline blake3_output_t parent_output_obj(const uint8_t block[BLAKE3_BLOCK_LEN], const uint32_t key[8], uint8_t flags) {
  return make_output(key, block, BLAKE3_BLOCK_LEN, 0, (uint8_t)(flags | PARENT));
}

static inline void parent_cv(const uint8_t block[BLAKE3_BLOCK_LEN], const uint32_t key[8], uint8_t flags, uint8_t out_cv[32]) {
  blake3_output_t o = parent_output_obj(block, key, flags);
  output_chaining_value(&o, out_cv);
}

/*
 * =========================================================
 * Hasher API Implementation
 * =========================================================
 */

static void hasher_init_internal(blake3_hasher *self, const uint32_t key[8], uint8_t flags) {
  memcpy(self->key, key, 8 * sizeof(uint32_t));
  blake3_chunk_state_reset(&self->chunk, self->key, 0);
  self->chunk.flags = flags;
  self->cv_stack_len = 0;
}

void blake3_hasher_init(blake3_hasher *self) {
  hasher_init_internal(self, BLAKE3_IV, 0);
}

void blake3_hasher_init_keyed(blake3_hasher *self, const uint8_t key[BLAKE3_KEY_LEN]) {
  uint32_t key_words[8];
  words_from_little_endian_bytes(key, BLAKE3_KEY_LEN, key_words);
  hasher_init_internal(self, key_words, (uint8_t)KEYED_HASH);
}

void blake3_hasher_init_derive_key(blake3_hasher *self, const char *context) {
  blake3_hasher_init_derive_key_raw(self, context, strlen(context));
}

void blake3_hasher_init_derive_key_raw(blake3_hasher *self, const void *context, size_t context_len) {
  blake3_hasher context_hasher;
  hasher_init_internal(&context_hasher, BLAKE3_IV, (uint8_t)DERIVE_KEY_CONTEXT);
  blake3_hasher_update(&context_hasher, context, context_len);

  uint8_t context_key[BLAKE3_KEY_LEN];
  blake3_hasher_finalize(&context_hasher, context_key, BLAKE3_KEY_LEN);

  uint32_t context_key_words[8];
  words_from_little_endian_bytes(context_key, BLAKE3_KEY_LEN, context_key_words);
  hasher_init_internal(self, context_key_words, (uint8_t)DERIVE_KEY_MATERIAL);
}

void blake3_hasher_reset(blake3_hasher *self) {
  const uint8_t mode_flags = self->chunk.flags;
  blake3_chunk_state_reset(&self->chunk, self->key, 0);
  self->chunk.flags = mode_flags;
  self->cv_stack_len = 0;
}

/* Stack push: merge while total_chunks is even (binary carry). */
static void blake3_hasher_push_stack(blake3_hasher *self, const uint8_t cv[32], uint64_t total_chunks) {
  const uint8_t *cur = cv;

  while ((total_chunks & 1) == 0) {
    uint8_t *left = &self->cv_stack[(self->cv_stack_len - 1) * 32];
    uint8_t block[BLAKE3_BLOCK_LEN];
    memcpy(block, left, 32);
    memcpy(block + 32, cur, 32);
    parent_cv(block, self->key, self->chunk.flags, left);
    self->cv_stack_len -= 1;
    total_chunks >>= 1;
    cur = left;
  }

  memcpy(&self->cv_stack[self->cv_stack_len * 32], cur, 32);
  self->cv_stack_len += 1;
}

static void blake3_hasher_add_chunk_cv(blake3_hasher *self, const uint8_t new_cv[32], uint64_t total_chunks) {
  blake3_hasher_push_stack(self, new_cv, total_chunks);
}

void blake3_hasher_update(blake3_hasher *self, const void *input, size_t input_len) {
  const uint8_t *p = (const uint8_t *)input;

  while (input_len > 0) {
    if (blake3_chunk_state_len(&self->chunk) == BLAKE3_CHUNK_LEN) {
      blake3_output_t chunk_out = blake3_chunk_state_output(&self->chunk);
      uint8_t chunk_cv[32];
      output_chaining_value(&chunk_out, chunk_cv);

      const uint64_t total_chunks = self->chunk.chunk_counter + 1;
      blake3_hasher_add_chunk_cv(self, chunk_cv, total_chunks);

      blake3_chunk_state_reset(&self->chunk, self->key, total_chunks);
      self->chunk.flags = chunk_out.flags & (KEYED_HASH | DERIVE_KEY_CONTEXT | DERIVE_KEY_MATERIAL);
    }

    const size_t want = (size_t)BLAKE3_CHUNK_LEN - blake3_chunk_state_len(&self->chunk);
    const size_t take = input_len < want ? input_len : want;

    blake3_chunk_state_update(&self->chunk, p, take);

    p += take;
    input_len -= take;
  }
}

void blake3_hasher_finalize(const blake3_hasher *self, uint8_t *out, size_t out_len) {
  /* 1) Start with the current (possibly incomplete) chunk output. */
  blake3_output_t o = blake3_chunk_state_output(&self->chunk);

  /* 2) Fold in the CV stack as parent nodes until we reach the root output. */
  uint8_t cv[32];
  output_chaining_value(&o, cv);

  for (size_t i = self->cv_stack_len; i > 0; i--) {
    const uint8_t *left = (const uint8_t *)&self->cv_stack[(i - 1) * 32];
    uint8_t block[BLAKE3_BLOCK_LEN];
    memcpy(block, left, 32);
    memcpy(block + 32, cv, 32);

    o = parent_output_obj(block, self->key, self->chunk.flags);
    output_chaining_value(&o, cv);
  }

  /* 3) Root XOF: repeat root compression with output_block_counter = 0.. */
  output_root_bytes(&o, out, out_len);
}
