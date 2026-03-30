#include "ark/compiler/CLI/SealKeyProvider.hpp"

#include "ark/compiler/CLI/Config.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ark::cli {
namespace {

std::optional<std::string> getEnv(const char* name) {
    if (!name || !*name) return std::nullopt;
    const char* v = std::getenv(name);
    if (!v || !*v) return std::nullopt;
    return std::string(v);
}

std::string trimAscii(std::string_view s) {
    std::size_t b = 0;
    std::size_t e = s.size();

    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;

    return std::string(s.substr(b, e - b));
}

std::string stripAsciiWhitespace(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
    }
    return out;
}

int hexNibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

std::optional<std::vector<std::uint8_t>> decodeHex32(std::string_view in) {
    std::string s = stripAsciiWhitespace(in);

    if (s.rfind("hex:", 0) == 0 || s.rfind("HEX:", 0) == 0) {
        s.erase(0, 4);
    }

    if (s.size() != 64) return std::nullopt;

    std::vector<std::uint8_t> out;
    out.resize(32);

    for (std::size_t i = 0; i < 32; ++i) {
        const int hi = hexNibble(s[i * 2]);
        const int lo = hexNibble(s[i * 2 + 1]);
        if (hi < 0 || lo < 0) return std::nullopt;
        out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }

    return out;
}

std::optional<std::vector<std::uint8_t>> parseKeyMaterial(std::string_view text) {
    const std::string trimmed = trimAscii(text);
    if (trimmed.empty()) return std::nullopt;

    if (auto hex = decodeHex32(trimmed)) return hex;

    return std::nullopt;
}

std::optional<std::vector<std::uint8_t>> loadKeyFromFile(std::string_view path, std::string& why) {
    auto mb = llvm::MemoryBuffer::getFile(path);
    if (!mb) {
        why = "unable to read file";
        return std::nullopt;
    }

    llvm::StringRef buf = (*mb)->getBuffer();

    if (buf.size() == 32) {
        std::vector<std::uint8_t> out(32);
        std::copy(buf.bytes_begin(), buf.bytes_end(), out.begin());
        return out;
    }

    if (auto parsed = parseKeyMaterial(std::string_view(buf.data(), buf.size()))) {
        return parsed;
    }

    why = "expected 32 raw bytes or 64 hex chars";
    return std::nullopt;
}

bool isLikelyHexKey(std::string_view s) {
    std::string x = stripAsciiWhitespace(s);
    if (x.rfind("hex:", 0) == 0 || x.rfind("HEX:", 0) == 0) x.erase(0, 4);
    if (x.size() != 64) return false;
    return std::all_of(x.begin(), x.end(), [](char c) { return hexNibble(c) >= 0; });
}

std::optional<std::vector<std::uint8_t>> tryVaultKey(arklang::hud::Hud& hud) {
    const auto vaultPwd = getEnv("ARKNET_VAULT_PASSWORD");
    if (!vaultPwd) return std::nullopt;

    auto secret = GlobalConfig::getDecryptedSecret("capsule", "seal_key_hex", *vaultPwd);
    if (!secret || secret->empty()) {
        hud.error("Vault password was provided, but capsule key `capsule.seal_key_hex` was not found or could not be decrypted.");
        return std::nullopt;
    }

    auto parsed = parseKeyMaterial(*secret);
    if (!parsed) {
        hud.error("Vault secret `capsule.seal_key_hex` is invalid. Expected 64 hex chars (32 bytes).");
        return std::nullopt;
    }

    return parsed;
}

} // namespace

std::optional<std::vector<std::uint8_t>> resolveCapsuleSealKey(
    arklang::hud::Hud& hud,
    llvm::StringRef explicitKeyFile) {

    if (!explicitKeyFile.empty()) {
        std::string why;
        if (auto key = loadKeyFromFile(explicitKeyFile.str(), why)) {
            hud.note("seal key source: --seal-key-file");
            return key;
        }
        hud.error("Failed to load capsule seal key from file `" + explicitKeyFile.str() + "` (" + why + ")");
        return std::nullopt;
    }

    if (auto p = getEnv("ARKNET_CAPSULE_KEY_FILE")) {
        std::string why;
        if (auto key = loadKeyFromFile(*p, why)) {
            hud.note("seal key source: ARKNET_CAPSULE_KEY_FILE");
            return key;
        }
        hud.error("Failed to load capsule seal key from ARKNET_CAPSULE_KEY_FILE `" + *p + "` (" + why + ")");
        return std::nullopt;
    }

    if (auto hex = getEnv("ARKNET_CAPSULE_KEY_HEX")) {
        if (auto key = parseKeyMaterial(*hex)) {
            hud.note("seal key source: ARKNET_CAPSULE_KEY_HEX");
            return key;
        }
        hud.error("ARKNET_CAPSULE_KEY_HEX is invalid. Expected 64 hex chars (32 bytes).");
        return std::nullopt;
    }

    if (auto key = tryVaultKey(hud)) {
        hud.note("seal key source: config vault (capsule.seal_key_hex)");
        return key;
    }

    hud.error(
        "No capsule seal key available.\n"
        "Provide one via:\n"
        "  --seal-key-file <path>\n"
        "  ARKNET_CAPSULE_KEY_FILE=<path>\n"
        "  ARKNET_CAPSULE_KEY_HEX=<64-hex>\n"
        "  config vault secret `capsule.seal_key_hex` + ARKNET_VAULT_PASSWORD");
    return std::nullopt;
}

} // namespace ark::cli