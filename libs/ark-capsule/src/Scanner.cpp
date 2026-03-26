#include <ark/capsule/Scanner.h> // [FIX] Was "Scanner.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ark/crypto.h> // We don't hash here, just read.

namespace ark::compiler::builder {

namespace fs = std::filesystem;

// Max source file size (16MB policy limit for V1)
static constexpr size_t MAX_FILE_SIZE = 16 * 1024 * 1024;

std::vector<uint8_t> Scanner::CanonicalizePath(const fs::path& rel) {
    std::string generic = rel.generic_string(); // Forces forward slashes
    // Remove leading "./" if present
    if (generic.size() > 2 && generic[0] == '.' && generic[1] == '/') {
        generic = generic.substr(2);
    }
    return std::vector<uint8_t>(generic.begin(), generic.end());
}

std::vector<SourceFile> Scanner::Scan(const std::string& root_dir) {
    std::vector<SourceFile> results;
    fs::path root(root_dir);

    if (!fs::exists(root) || !fs::is_directory(root)) {
        return results; // Empty result = nothing to build
    }

    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;

        // Policy: Only process specific extensions if needed, or all files?
        // For V1, let's process everything to be safe.
        
        // 1. Check Size
        if (entry.file_size() > MAX_FILE_SIZE) {
            std::cerr << "Skipping " << entry.path() << " (Too large)\n";
            continue;
        }

        // 2. Read Content
        std::ifstream file(entry.path(), std::ios::binary);
        if (!file) continue;
        
        std::vector<uint8_t> content((size_t)entry.file_size());
        file.read(reinterpret_cast<char*>(content.data()), content.size());
        if (!file) continue;

        // 3. Canonicalize Path
        fs::path rel = fs::relative(entry.path(), root);
        
        SourceFile sf;
        sf.absolute_path = entry.path().string();
        sf.rel_path_canonical = CanonicalizePath(rel);
        sf.content = std::move(content);
        
        results.push_back(std::move(sf));
    }

    return results;
}

} // namespace ark::compiler::builder