#ifndef BLAKE3_H
#define BLAKE3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =========================================================
 * BLAKE3 Constants
 * =========================================================
 */
#define BLAKE3_VERSION_STRING "1.3.1"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54

/* * =========================================================
 * Internal Structs
 * (Exposed to allow stack allocation, do not access directly)
 * =========================================================
 */

/* Internal state for the current chunk being hashed */
typedef struct {
  uint32_t cv[8];
  uint64_t chunk_counter;
  uint8_t buf[BLAKE3_BLOCK_LEN];
  uint8_t buf_len;
  uint8_t blocks_compressed;
  uint8_t flags;
} blake3_chunk_state;

/* The main hasher state */
typedef struct {
  uint32_t key[8];
  blake3_chunk_state chunk;
  uint8_t cv_stack_len;
  /* The stack size is MAX_DEPTH + 1 because we do lazy merging. */
  uint8_t cv_stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
} blake3_hasher;

/*
 * =========================================================
 * Public API
 * =========================================================
 */

/**
 * Initialize the hasher for the default hashing mode.
 * @param self Pointer to the hasher state to initialize.
 */
void blake3_hasher_init(blake3_hasher *self);

/**
 * Initialize the hasher for keyed hashing (MAC mode).
 * @param self Pointer to the hasher state to initialize.
 * @param key Pointer to a 32-byte secret key.
 */
void blake3_hasher_init_keyed(blake3_hasher *self, const uint8_t key[BLAKE3_KEY_LEN]);

/**
 * Initialize the hasher for key derivation.
 * @param self Pointer to the hasher state to initialize.
 * @param context Context string (must be hardcoded/globally unique).
 * @param context_len Length of the context string.
 */
void blake3_hasher_init_derive_key(blake3_hasher *self, const char *context);

/**
 * Initialize the hasher for key derivation using a raw void pointer context.
 * @param self Pointer to the hasher state to initialize.
 * @param context Pointer to context data.
 * @param context_len Length of context data.
 */
void blake3_hasher_init_derive_key_raw(blake3_hasher *self, const void *context, size_t context_len);

/**
 * Add input data to the hash.
 * @param self Pointer to the hasher state.
 * @param input Pointer to the data to hash.
 * @param input_len Length of the data in bytes.
 */
void blake3_hasher_update(blake3_hasher *self, const void *input, size_t input_len);

/**
 * Finalize the hash and write the output.
 * @param self Pointer to the hasher state.
 * @param out Buffer to receive the hash output.
 * @param out_len Number of bytes to write (supports XOF/variable length).
 */
void blake3_hasher_finalize(const blake3_hasher *self, uint8_t *out, size_t out_len);

/**
 * Reset the hasher state to reuse it (avoids reallocation).
 * Preserves the original mode (default, keyed, or derive_key).
 * @param self Pointer to the hasher state.
 */
void blake3_hasher_reset(blake3_hasher *self);

#ifdef __cplusplus
}
#endif

#endif /* BLAKE3_H */