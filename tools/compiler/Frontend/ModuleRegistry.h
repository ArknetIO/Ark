#pragma once

#include "Frontend/AST.h"
#include "llvm/Support/Error.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <set>
#include <string>
#include <memory>

namespace arklang {

class ModuleRegistry {
public:
    ModuleRegistry() = default;

    // The Main Entry Point
    // path: The import string (e.g., "math/vec", "@/std/io")
    // relativeTo: The absolute directory of the file requesting the import
    llvm::Expected<Module*> load(const std::string& path, const std::string& relativeTo);

private:
    // --- State ---
    
    // Cache: Real Absolute Path -> Owned Module AST
    // We use std::map because we need pointer stability for Module*
    std::map<std::string, std::unique_ptr<Module>> loadedModules;

    // Cycle Detection: Set of paths currently being parsed
    std::set<std::string> loadingStack;

    // --- Helpers ---
    
    llvm::Expected<Module*> loadFile(const std::string& realPath);
    llvm::Expected<Module*> loadDirectory(const std::string& realPath);
    llvm::Expected<std::string> resolvePath(const std::string& importPath, const std::string& relativeTo);
};

} // namespace arklang