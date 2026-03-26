#include <ark/crypto/Sign.h>
#include <sodium.h>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace ark::crypto {

// =========================================================
// Helpers
// =========================================================
static void ensure_sodium() {
    if (sodium_init() == -1) {
        throw std::runtime_error("libsodium initialization failed");
    }
}

// =========================================================
// Key Generation
// =========================================================

SigningKeypair Signature::keygen(SignAlgo algo) {
    ensure_sodium();
    
    if (algo == SignAlgo::Ed25519) {
        SigningKeypair kp;
        kp.pub.resize(crypto_sign_PUBLICKEYBYTES);
        
        // Use a temporary secure buffer for the raw secret key
        // Libsodium's crypto_sign_keypair writes the full 64-byte secret key (seed + pub)
        std::vector<uint8_t> temp_sec(crypto_sign_SECRETKEYBYTES);
        
        crypto_sign_keypair(kp.pub.data(), temp_sec.data());
        
        // Move into the robust FixedSecret container
        // [FIX] Use const_cast because FixedSecret::data() returns const ptr by design
        std::memcpy(const_cast<uint8_t*>(kp.sec.data()), temp_sec.data(), temp_sec.size());
        
        // Securely wipe the temp buffer
        sodium_memzero(temp_sec.data(), temp_sec.size());
        
        return kp;
    }
    
    throw std::invalid_argument("Unsupported signing algorithm");
}

SigningKeypair Signature::keygen_from_seed(std::span<const uint8_t> seed, SignAlgo algo) {
    ensure_sodium();

    if (algo == SignAlgo::Ed25519) {
        if (seed.size() != crypto_sign_SEEDBYTES) {
            throw std::invalid_argument("Invalid seed length for Ed25519");
        }

        SigningKeypair kp;
        kp.pub.resize(crypto_sign_PUBLICKEYBYTES);
        
        std::vector<uint8_t> temp_sec(crypto_sign_SECRETKEYBYTES);
        
        // Derives both Public and Secret key from the seed
        crypto_sign_seed_keypair(kp.pub.data(), temp_sec.data(), seed.data());
        
        // [FIX] Use const_cast to initialize the FixedSecret storage
        std::memcpy(const_cast<uint8_t*>(kp.sec.data()), temp_sec.data(), temp_sec.size());
        
        sodium_memzero(temp_sec.data(), temp_sec.size());
        
        return kp;
    }

    throw std::invalid_argument("Unsupported signing algorithm");
}

// =========================================================
// Key Derivation
// =========================================================

std::vector<uint8_t> Signature::get_public_key(std::span<const uint8_t> private_key, SignAlgo algo) {
    ensure_sodium();

    if (algo == SignAlgo::Ed25519) {
        std::vector<uint8_t> pk(crypto_sign_PUBLICKEYBYTES);

        // Case 1: Full 64-byte Secret Key
        if (private_key.size() == crypto_sign_SECRETKEYBYTES) {
            if (crypto_sign_ed25519_sk_to_pk(pk.data(), private_key.data()) != 0) {
                throw std::runtime_error("Failed to extract public key from secret key");
            }
        } 
        // Case 2: 32-byte Seed
        else if (private_key.size() == crypto_sign_SEEDBYTES) {
            // We regenerate the pair to get the public key
            uint8_t dummy_sk[crypto_sign_SECRETKEYBYTES];
            crypto_sign_seed_keypair(pk.data(), dummy_sk, private_key.data());
            sodium_memzero(dummy_sk, sizeof(dummy_sk));
        } 
        else {
            throw std::invalid_argument("Invalid private key length (expected 32 or 64 bytes)");
        }
        return pk;
    }

    throw std::invalid_argument("Unsupported signing algorithm");
}

std::vector<uint8_t> Signature::get_private_key(std::span<const uint8_t> seed, SignAlgo algo) {
    ensure_sodium();

    if (algo == SignAlgo::Ed25519) {
        if (seed.size() != crypto_sign_SEEDBYTES) {
            throw std::invalid_argument("Invalid seed length");
        }

        std::vector<uint8_t> sk(crypto_sign_SECRETKEYBYTES);
        uint8_t dummy_pk[crypto_sign_PUBLICKEYBYTES];
        
        crypto_sign_seed_keypair(dummy_pk, sk.data(), seed.data());
        
        return sk;
    }

    throw std::invalid_argument("Unsupported signing algorithm");
}

// =========================================================
// Operations
// =========================================================

// Overload 1: Using SigningSecretKey container
std::vector<uint8_t> Signature::sign(
    std::span<const uint8_t> message, 
    const SigningSecretKey& secret_key, 
    SignAlgo algo
) {
    // Forward to the raw span overload
    return sign(message, std::span<const uint8_t>{secret_key.data(), secret_key.size()}, algo);
}

// Overload 2: Using raw span
std::vector<uint8_t> Signature::sign(
    std::span<const uint8_t> message, 
    std::span<const uint8_t> secret_key, 
    SignAlgo algo
) {
    ensure_sodium();

    if (algo == SignAlgo::Ed25519) {
        // Validation
        if (secret_key.size() != crypto_sign_SECRETKEYBYTES && secret_key.size() != crypto_sign_SEEDBYTES) {
             throw std::invalid_argument("Invalid secret key length");
        }

        // Handle Seed vs Full SK
        const uint8_t* sk_ptr = secret_key.data();
        uint8_t expanded_sk[crypto_sign_SECRETKEYBYTES];
        bool cleanup = false;

        if (secret_key.size() == crypto_sign_SEEDBYTES) {
            uint8_t dummy_pk[crypto_sign_PUBLICKEYBYTES];
            crypto_sign_seed_keypair(dummy_pk, expanded_sk, secret_key.data());
            sk_ptr = expanded_sk;
            cleanup = true;
        }

        std::vector<uint8_t> sig(crypto_sign_BYTES);
        
        if (crypto_sign_detached(
                sig.data(), 
                nullptr, 
                message.data(), 
                message.size(), 
                sk_ptr
            ) != 0) {
            if (cleanup) sodium_memzero(expanded_sk, sizeof(expanded_sk));
            throw std::runtime_error("Signing failed");
        }

        if (cleanup) sodium_memzero(expanded_sk, sizeof(expanded_sk));
        return sig;
    }
    
    throw std::invalid_argument("Unsupported signing algorithm");
}

bool Signature::verify(
    std::span<const uint8_t> signature, 
    std::span<const uint8_t> message, 
    std::span<const uint8_t> public_key,
    SignAlgo algo
) {
    ensure_sodium();

    if (algo == SignAlgo::Ed25519) {
        if (signature.size() != crypto_sign_BYTES) return false;
        if (public_key.size() != crypto_sign_PUBLICKEYBYTES) return false;
        
        return crypto_sign_verify_detached(
            signature.data(), 
            message.data(), 
            message.size(), 
            public_key.data()
        ) == 0;
    }

    return false;
}

} // namespace ark::crypto