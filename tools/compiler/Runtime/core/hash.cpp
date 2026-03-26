#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <atomic>

#if defined(__linux__)
  #include <sys/random.h>
  #include <errno.h>
  #include <unistd.h>
  #include <time.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
  #include <stdlib.h>   // arc4random_buf
  #include <unistd.h>
  #include <time.h>
#endif

extern "C" {

// 0 = uninit, 1 = init-in-progress, 2 = ready
static std::atomic<uint8_t> g_hash_state{0};
static uint64_t g_hash_k0 = 0x0706050403020100ULL;
static uint64_t g_hash_k1 = 0x0f0e0d0c0b0a0908ULL;

static inline uint64_t rotl64(uint64_t x, int b) {
    return (x << b) | (x >> (64 - b));
}

static inline uint64_t load_le64(const uint8_t* p) {
    uint64_t m;
    memcpy(&m, p, sizeof(m));
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    m = __builtin_bswap64(m);
#endif
    return m;
}

static inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void ark_hash_init_key() {
    uint8_t expected = 0;
    if (!g_hash_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        while (g_hash_state.load(std::memory_order_acquire) != 2) {}
        return;
    }

    uint64_t keys[2] = {0, 0};
    bool ok = false;

#if defined(__linux__)
    size_t got = 0;
    while (got < sizeof(keys)) {
        ssize_t r = getrandom(((uint8_t*)keys) + got, sizeof(keys) - got, 0);
        if (r > 0) { got += (size_t)r; continue; }
        if (r < 0 && errno == EINTR) continue;
        break;
    }
    ok = (got == sizeof(keys));
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    arc4random_buf(keys, sizeof(keys));
    ok = true;
#endif

    if (!ok) {
        uint64_t seed = 0;
#if defined(__unix__) || defined(__APPLE__)
        seed ^= (uint64_t)(uintptr_t)&seed;
        seed ^= (uint64_t)getpid();
        struct timespec ts1{};
        struct timespec ts2{};
    #if defined(CLOCK_REALTIME)
        clock_gettime(CLOCK_REALTIME, &ts1);
        seed ^= (uint64_t)ts1.tv_sec ^ (uint64_t)ts1.tv_nsec;
    #endif
    #if defined(CLOCK_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        seed ^= (uint64_t)ts2.tv_sec ^ (uint64_t)ts2.tv_nsec;
    #endif
#endif
        keys[0] = splitmix64(seed);
        keys[1] = splitmix64(seed);
    }

    g_hash_k0 = keys[0];
    g_hash_k1 = keys[1];
    g_hash_state.store(2, std::memory_order_release);
}

#define SIPROUND() \
    do { \
        v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32); \
        v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2; \
        v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0; \
        v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32); \
    } while (0)

// ABI: uint64_t __ark_hash_bytes(const void* ptr, int64_t len)
uint64_t __ark_hash_bytes(const void* ptr, int64_t len) {
    if (g_hash_state.load(std::memory_order_acquire) != 2) ark_hash_init_key();
    if (!ptr || len <= 0) return 0;

    const uint8_t* in = (const uint8_t*)ptr;
    const uint64_t inlen = (uint64_t)len;

    uint64_t v0 = g_hash_k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = g_hash_k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = g_hash_k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = g_hash_k1 ^ 0x7465646279746573ULL;

    const uint8_t* end = in + (size_t)(inlen & ~7ULL);
    const uint32_t left = (uint32_t)(inlen & 7ULL);
    uint64_t b = inlen << 56;

    for (; in != end; in += 8) {
        uint64_t m = load_le64(in);
        v3 ^= m;
        SIPROUND();      // 1 compression round
        v0 ^= m;
    }

    switch (left) {
        case 7: b |= ((uint64_t)in[6]) << 48;
        case 6: b |= ((uint64_t)in[5]) << 40;
        case 5: b |= ((uint64_t)in[4]) << 32;
        case 4: b |= ((uint64_t)in[3]) << 24;
        case 3: b |= ((uint64_t)in[2]) << 16;
        case 2: b |= ((uint64_t)in[1]) <<  8;
        case 1: b |= ((uint64_t)in[0]); break;
        case 0: break;
    }

    v3 ^= b;
    SIPROUND();
    v0 ^= b;

    v2 ^= 0xff;
    SIPROUND(); SIPROUND(); SIPROUND(); // 3 finalization rounds

    return v0 ^ v1 ^ v2 ^ v3;
}


// =============================================================================
// Stable Hash (FNV-1a 64-bit)
// Deterministic across process restarts. Good for caching/checksums.
// =============================================================================

// ABI: uint64_t __ark_stable_hash_bytes(const void* ptr, int64_t len)
uint64_t __ark_stable_hash_bytes(const void* ptr, int64_t len) {
    if (!ptr || len <= 0) return 0;
    
    const uint8_t *p = (const uint8_t *)ptr;
    uint64_t hash = 14695981039346656037ULL; // FNV-1a 64-bit offset basis
    for (int64_t i = 0; i < len; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;            // FNV 64-bit prime
    }
    
    return hash;
}

} // extern "C"