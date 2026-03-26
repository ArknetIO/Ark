#include <ark/crypto/Box.h>
#include "SecretFactory.h"
#include <sodium.h>

namespace ark::crypto {

// -----------------------------------------------------------------------------
// Asymmetric
// -----------------------------------------------------------------------------
X25519Keypair box_keygen() {
    init();
    X25519Keypair kp;
    kp.pub.resize(crypto_box_PUBLICKEYBYTES);
    
    SecureBytes sec(crypto_box_SECRETKEYBYTES);
    crypto_box_keypair(kp.pub.data(), sec.data());
    
    kp.sec = SecretKeyFactory::wrap<BOX_SECKEY_BYTES>(std::move(sec));
    return kp;
}

std::vector<uint8_t> seal_encrypt(std::span<const uint8_t> msg, std::span<const uint8_t> recipient_pub) {
    if (recipient_pub.size() != crypto_box_PUBLICKEYBYTES) throw std::invalid_argument("Bad pubkey size");
    
    std::vector<uint8_t> out(crypto_box_SEALBYTES + msg.size());
    if (crypto_box_seal(out.data(), msg.data(), msg.size(), recipient_pub.data()) != 0) {
        throw std::runtime_error("SealedBox failed");
    }
    return out;
}

std::optional<SecureBytes> seal_decrypt(std::span<const uint8_t> cipher, const X25519Keypair& keys) {
    if (cipher.size() < crypto_box_SEALBYTES) return std::nullopt;
    
    SecureBytes plain(cipher.size() - crypto_box_SEALBYTES);
    if (crypto_box_seal_open(plain.data(), cipher.data(), cipher.size(), 
                             keys.pub.data(), keys.sec.data()) != 0) {
        return std::nullopt;
    }
    return plain;
}

// -----------------------------------------------------------------------------
// Symmetric
// -----------------------------------------------------------------------------
std::vector<uint8_t> secretbox_encrypt(std::span<const uint8_t> msg, const SecretBoxKey& key) {
    size_t nonce_len = crypto_secretbox_xchacha20poly1305_NONCEBYTES;
    size_t mac_len   = crypto_secretbox_xchacha20poly1305_MACBYTES;
    
    std::vector<uint8_t> out(nonce_len + mac_len + msg.size());
    uint8_t* nonce = out.data();
    uint8_t* cipher = out.data() + nonce_len; // MAC is inside cipher for 'easy' API

    randombytes_buf(nonce, nonce_len);
    
    if (crypto_secretbox_xchacha20poly1305_easy(cipher, msg.data(), msg.size(), nonce, key.data()) != 0) {
        throw std::runtime_error("SecretBox encrypt failed");
    }
    return out;
}

std::optional<SecureBytes> secretbox_decrypt(std::span<const uint8_t> packed, const SecretBoxKey& key) {
    size_t nonce_len = crypto_secretbox_xchacha20poly1305_NONCEBYTES;
    size_t mac_len   = crypto_secretbox_xchacha20poly1305_MACBYTES;

    if (packed.size() < nonce_len + mac_len) return std::nullopt;

    const uint8_t* nonce = packed.data();
    const uint8_t* cipher = packed.data() + nonce_len;
    size_t clen = packed.size() - nonce_len;

    SecureBytes plain(clen - mac_len);
    if (crypto_secretbox_xchacha20poly1305_open_easy(plain.data(), cipher, clen, nonce, key.data()) != 0) {
        return std::nullopt;
    }
    return plain;
}

} // namespace ark::crypto