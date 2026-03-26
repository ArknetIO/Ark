#pragma once
#include "HashTypes.h"
#include <string>
#include <vector>
#include <filesystem>

namespace ark::compiler::builder {

// Represents a raw file on disk before it becomes a dependency node
struct SourceFile {
    std::string absolute_path;
    std::vector<uint8_t> rel_path_canonical;
    std::vector<uint8_t> content;
};

class Scanner {
public:
    // -------------------------------------------------------------------------
    // SCAN DIRECTORY
    // -------------------------------------------------------------------------
    // 1. Recursively walks `root_dir`.
    // 2. Filters for valid source files (e.g. .ark).
    // 3. Reads content into memory (strict limit enforced).
    // 4. Canonicalizes paths (relative to root, forward slashes).
    //
    // Returns: List of source files, or throws/returns error on IO failure.
    static std::vector<SourceFile> Scan(const std::string& root_dir);

private:
    static std::vector<uint8_t> CanonicalizePath(const std::filesystem::path& rel);
};

} // namespace ark::compiler::builder