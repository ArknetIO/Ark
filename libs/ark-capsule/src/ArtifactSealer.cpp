#include <ark/capsule/ArtifactSealer.h>
#include <ark/wire/DynWriter.h>
#include <ark/abi/capsule_v1.h>
#include <ark/crypto.h>
#include <vector>
#include <cstring> // std::memcpy

namespace ark::compiler::builder {

using namespace ark::abi::capsule_v1;

BuilderStatus ArtifactSealer::Seal(
    const Hash32& plan_hash,
    const std::vector<uint8_t>& compiler_id,
    const std::vector<uint8_t>& reqs,
    const Hash32& symtab_hash,
    const std::vector<uint8_t>& key,
    SealedArtifact& out_artifact
) {
    // 1. Pre-flight Validation
    // -------------------------------------------------------------------------
    
    // Key must be a 64-byte Ed25519 Extended Private Key
    if (!key.empty() && key.size() != 64) {
        return BuilderStatus::CryptoError;
    }

    // TLV payloads in Capsule V1 use u16 lengths. strict cap.
    if (compiler_id.size() > 0xFFFF) return BuilderStatus::InputTooLarge;
    if (reqs.size() > 0xFFFF) return BuilderStatus::InputTooLarge;

    // 2. Exact Size Calculation (No Patching Allowed)
    // -------------------------------------------------------------------------
    
    size_t sz_header = HEADER_SIZE;  // 16 bytes
    
    // Payload TLVs
    size_t sz_planhash = TLV_HDR_SIZE + PLANHASH_LEN;       // 4 + 32
    size_t sz_compid   = TLV_HDR_SIZE + compiler_id.size(); // 4 + N
    size_t sz_reqs     = TLV_HDR_SIZE + reqs.size();        // 4 + M
    
    bool has_symtab = false;
    for (auto b : symtab_hash) if (b != 0) has_symtab = true;
    size_t sz_symtab   = has_symtab ? (TLV_HDR_SIZE + SYMTAB_HASH_LEN) : 0;

    // The "Signed Region" ends here.
    size_t signed_len = sz_header + sz_planhash + sz_compid + sz_reqs + sz_symtab;

    // Signature TLV (Always last)
    size_t sz_sig = SIG_TLV_SIZE; // 4 + 64

    size_t total_len = signed_len + sz_sig;

    // Hard limit check for u32 file size
    if (total_len > UINT32_MAX) return BuilderStatus::OutputError;


    // 3. Construction (Append-Only)
    // -------------------------------------------------------------------------
    ark::wire::DynWriter w;

    // --- Header ---
    w.write_bytes(kMagic_bytes, 4);           // Magic
    w.write_u16(kVersion);                    // Version
    w.write_u16(0);                           // Reserved
    
    uint32_t flags = 0;
    if (has_symtab) flags |= FLAG_HAS_SYMTAB;
    w.write_u32(flags);                       // Flags
    
    w.write_u32(static_cast<uint32_t>(total_len)); // Total Length (Finalized)

    // --- Payload TLVs ---
    
    // 1. PlanHash
    w.write_u16(TAG_PLANHASH);
    w.write_u16(PLANHASH_LEN);
    w.write_bytes(plan_hash.data(), PLANHASH_LEN);

    // 2. CompilerID
    w.write_u16(TAG_COMPILER_ID);
    w.write_u16(static_cast<uint16_t>(compiler_id.size()));
    w.write_bytes(compiler_id.data(), compiler_id.size());

    // 3. RuntimeReqs
    w.write_u16(TAG_RUNTIME_REQS);
    w.write_u16(static_cast<uint16_t>(reqs.size()));
    w.write_bytes(reqs.data(), reqs.size());

    // 4. SymtabHash (Optional)
    if (has_symtab) {
        w.write_u16(TAG_SYMTAB_HASH);
        w.write_u16(SYMTAB_HASH_LEN);
        w.write_bytes(symtab_hash.data(), SYMTAB_HASH_LEN);
    }

    // 4. Signing
    // -------------------------------------------------------------------------
    // Invariant Check: We must be exactly at the end of the signed region.
    if (w.size() != signed_len) {
        return BuilderStatus::OutputError; // Logic error in size calc vs write
    }

    std::vector<uint8_t> signature(SIG_LEN, 0);
    
    if (!key.empty()) {
        // [FIX] Use Robust Crypto API
        ark::crypto::SigningSecretKey secret_key(key);
        
        // Sign [0 .. signed_len)
        // This covers Header (with total_len) and all Payload TLVs.
        std::span<const uint8_t> message(w.output().data(), w.size());
        
        std::vector<uint8_t> sig_bytes = ark::crypto::Signature::sign(
            message,
            secret_key
        );
        
        // Copy result
        if (sig_bytes.size() == SIG_LEN) {
            std::memcpy(signature.data(), sig_bytes.data(), SIG_LEN);
        } else {
            return BuilderStatus::CryptoError;
        }
    }

    // 5. Append Signature TLV
    // -------------------------------------------------------------------------
    w.write_u16(TAG_SIGNATURE);
    w.write_u16(SIG_LEN);
    w.write_bytes(signature.data(), SIG_LEN);

    // Final Invariant Check
    if (w.size() != total_len) {
        return BuilderStatus::OutputError;
    }

    out_artifact.capsule_bytes = w.output();
    return BuilderStatus::Ok;
}

} // namespace ark::compiler::builder