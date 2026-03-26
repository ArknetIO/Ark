#include "Frontend/ModuleRegistry.h"
#include "Frontend/Lexer.h"
#include "Frontend/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace arklang {

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

// -----------------------------------------------------------------------------
// Path Resolution Logic
// -----------------------------------------------------------------------------
llvm::Expected<std::string> ModuleRegistry::resolvePath(const std::string& importPath, const std::string& relativeTo) {
    llvm::SmallString<256> p;

    // 1. Handle Absolute vs Relative
    if (path::is_absolute(importPath)) {
        p = importPath;
    }
    // Root-Relative Import ("@/std/io") - Optional feature convention
    else if (importPath.rfind("@/", 0) == 0) {
        fs::current_path(p);
        path::append(p, importPath.substr(2));
    } 
    // Relative Import
    else {
        p = llvm::StringRef(relativeTo);
        path::append(p, importPath);
    }

    // 2. Try exact match (file or dir)
    if (fs::exists(p)) {
        llvm::SmallString<256> real;
        if (std::error_code ec = fs::real_path(p, real)) 
            return llvm::createStringError(ec, "Failed to resolve path: " + std::string(p));
        return std::string(real.str());
    }

    // 3. Try appending .ark extension (if not present)
    if (!path::has_extension(p)) {
        llvm::SmallString<256> withExt = p;
        path::replace_extension(withExt, "ark"); 
        
        if (fs::exists(withExt)) {
            llvm::SmallString<256> real;
            if (std::error_code ec = fs::real_path(withExt, real)) 
                return llvm::createStringError(ec, "Failed to resolve extension path: " + std::string(withExt));
            return std::string(real.str());
        }
    }

    return llvm::createStringError(std::make_error_code(std::errc::no_such_file_or_directory), 
                                   "Import not found: " + importPath);
}

// -----------------------------------------------------------------------------
// Main Load Function
// -----------------------------------------------------------------------------
llvm::Expected<Module*> ModuleRegistry::load(const std::string& importPath, const std::string& relativeTo) {
    
    // 1. Resolve to Canonical Path
    auto pathOr = resolvePath(importPath, relativeTo);
    if (!pathOr) return pathOr.takeError();
    std::string realPath = *pathOr;

    // 2. Check Cache (Fast Path)
    if (loadedModules.count(realPath)) {
        return loadedModules[realPath].get();
    }

    // 3. Cycle Detection
    if (loadingStack.count(realPath)) {
        return llvm::createStringError(std::make_error_code(std::errc::connection_refused), 
                                       "Circular dependency detected: " + importPath);
    }
    loadingStack.insert(realPath);

    // 4. Dispatch: File vs Directory
    llvm::Expected<Module*> result = [&]() -> llvm::Expected<Module*> {
        if (fs::is_directory(realPath)) {
            return loadDirectory(realPath);
        } else {
            return loadFile(realPath);
        }
    }();

    // 5. Cleanup Stack
    loadingStack.erase(realPath);
    
    return result;
}

// -----------------------------------------------------------------------------
// Single File Loader
// -----------------------------------------------------------------------------
llvm::Expected<Module*> ModuleRegistry::loadFile(const std::string& realPath) {
    // 1. Read File
    auto bufOr = llvm::MemoryBuffer::getFile(realPath);
    if (!bufOr) return llvm::createStringError(bufOr.getError(), "Failed to read file: " + realPath);

    // 2. Parse
    // [FIX] Pass realPath so errors are meaningful
    Lexer lexer(bufOr.get()->getBuffer().str(), realPath);
    auto toks = lexer.tokenize();
    
    if (!toks.errors.empty()) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument), 
                                       "Lexer error in " + realPath + ": " + toks.errors[0]);
    }

    Parser parser(toks);
    std::unique_ptr<Module> mod = parser.parseModule();
    if (parser.hasErrors()) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument), 
                                       "Parser error in " + realPath + ": " + parser.getErrors()[0]);
    }

    // [CRITICAL NEW STEP] Set the Module ID
    // This allows GenMIR to distinguish "vectors" from "main" for name mangling.
    // e.g. "/path/to/generics.ark" -> id = "generics"
    mod->id = llvm::sys::path::stem(realPath).str();

    // 3. Process Imports (Recursive Step)
    // Move ownership to the registry map
    loadedModules[realPath] = std::move(mod);
    Module* m = loadedModules[realPath].get();
    
    std::string baseDir = std::string(path::parent_path(realPath));

    for (auto& imp : m->imports) {
        // Handle Implicit Alias: import "math/vec.ark"; -> alias "vec"
        if (imp->alias.empty()) {
            imp->alias = llvm::sys::path::stem(imp->path).str();
        }

        auto subModOr = load(imp->path, baseDir);
        if (!subModOr) return subModOr.takeError();
        
        // Bind to this module's scope so we can look up 'vec.add'
        m->submodules[imp->alias] = *subModOr;
    }

    return m;
}

// -----------------------------------------------------------------------------
// Directory Bundle Loader
// -----------------------------------------------------------------------------
llvm::Expected<Module*> ModuleRegistry::loadDirectory(const std::string& realPath) {
    // Create a virtual module to hold the folder contents
    auto dirMod = std::make_unique<Module>();
    
    std::error_code ec;
    for (fs::directory_iterator it(realPath, ec), end; it != end; it.increment(ec)) {
        if (ec) return llvm::createStringError(ec, "Directory iteration failed: " + realPath);
        
        std::string filePath = it->path();
        
        // Only load .ark files
        if (path::extension(filePath) == ".ark") {
            // Recursive load of the file inside the directory
            // Note: Empty string for relativeTo because filePath is absolute here
            auto subModOr = load(filePath, ""); 
            if (!subModOr) return subModOr.takeError();
            
            // Auto-bind: file "vec.ark" becomes submodule "vec" inside this virtual module
            std::string stem = path::stem(filePath).str();
            dirMod->submodules[stem] = *subModOr;
        }
    }

    loadedModules[realPath] = std::move(dirMod);
    return loadedModules[realPath].get();
}

} // namespace arklang