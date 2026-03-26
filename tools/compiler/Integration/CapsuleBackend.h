#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "../hud.h" // Relative path to hud.h in tools/compiler

namespace ark::compiler::integration {

class CapsuleBackend {
public:
    // Creates a V1 Self-Executing Capsule:
    // [Patched Stub] + [Payload] + [Bundle] + [Footer]
    static bool CreateSelfExecutingCapsule(
        arklang::hud::Hud& hud,
        const std::string& sourceDir,
        const std::string& stubPath,
        const std::string& payloadPath,
        const std::string& outPath,
        const std::vector<uint8_t>& privKey,
        uint64_t* outTotalSize
    );
};

} // namespace