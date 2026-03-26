#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ArkNet Onboarding Protocol - ABI Definition
// =============================================================================

// Architecture IDs (uint16_t)
#define ARK_ARCH_UNKNOWN       0x0000
#define ARK_ARCH_NVIDIA_A100   0x0001
#define ARK_ARCH_NVIDIA_H100   0x0002
#define ARK_ARCH_NVIDIA_B200   0x0003
#define ARK_ARCH_AMD_MI300     0x0010
#define ARK_ARCH_APPLE_M3      0x0020
// ... plenty of room for 65,535 architectures

// Fixed Buffer Sizes
#define ARK_HOST_BUF_SIZE      64
#define ARK_REGION_BUF_SIZE    16

#pragma pack(push, 1)

/**
 * @brief The Onboarding Wire Format (V1.1)
 * * Total Size: 128 Bytes (2 Cache Lines)
 * * Alignment: Strictly packed, Little-Endian
 */
struct OnboardV1 {
    // --- Header (8 Bytes) ---
    uint8_t  version;             // Must be 1
    uint8_t  _pad0;               // Padding for alignment
    uint16_t gpu_arch_le;         // [CHANGED] Now uint16_t (Little Endian)
    uint16_t gpu_count_le;        // Number of GPUs
    uint16_t port_le;             // Service Port (Standard u16)

    // --- Identification (80 Bytes) ---
    // Zero-padded strings. Must be null-terminated if shorter than buffer.
    char     hostname[ARK_HOST_BUF_SIZE];   // "node.ark.net" or "1.2.3.4"
    char     region[ARK_REGION_BUF_SIZE];   // "us-east-1"

    // --- Economics & Replay (24 Bytes) ---
    uint64_t price_micros_le;     // Price per millisecond
    uint64_t vram_total_mb_le;    // Total VRAM
    uint64_t nonce_le;            // Replay protection

    // --- Extensions (16 Bytes) ---
    uint64_t capabilities_le;     // Feature Bitmask
    uint8_t  reserved[8];         // Padding to reach 128 bytes
};

#pragma pack(pop)

// Compilation Safety Checks
static_assert(sizeof(OnboardV1) == 128, "OnboardV1 ABI Size Mismatch (Must be 128 bytes)");
// 1 + 1 + 2 + 2 + 2 = 8 bytes header
static_assert(offsetof(OnboardV1, hostname) == 8, "OnboardV1 Alignment Error (Hostname)");
static_assert(offsetof(OnboardV1, price_micros_le) == 88, "OnboardV1 Alignment Error (Price)");

#ifdef __cplusplus
} // extern "C"
#endif