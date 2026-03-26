// libs/ark-crypto/src/SecretFactory.h
#pragma once
#include <ark/crypto/Core.h>

namespace ark::crypto {

// Internal bridge to access private members of FixedSecret<N>
struct SecretKeyFactory {
    template <size_t N>
    static FixedSecret<N> wrap(SecureBytes&& b) {
        FixedSecret<N> k;
        // Friend access to private member '_b'
        k._b = std::move(b);
        return k;
    }

    template <size_t N>
    static uint8_t* mut_ptr(FixedSecret<N>& k) {
        // Friend access to private member 'data_mut'
        return k.data_mut();
    }
};

} // namespace ark::crypto