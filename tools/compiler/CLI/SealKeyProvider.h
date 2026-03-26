#pragma once

#include "../hud.h"

#include <llvm/ADT/StringRef.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ark::cli {

std::optional<std::vector<std::uint8_t>> resolveCapsuleSealKey(
    arklang::hud::Hud& hud,
    llvm::StringRef explicitKeyFile = "");

} // namespace ark::cli