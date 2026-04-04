// tools/compiler/src/Pipeline/LinkerGpuFatbin.cpp
#include "ark/compiler/Pipeline/Linker.hpp"
#include "ark/compiler/Pipeline/Compiler.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <cctype>
#include <cstdint>
#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace ark::compiler::pipeline {
namespace {

static std::string sanitizeSymbol(std::string s) {
    for (char& c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        const bool ok = (std::isalnum(uc) != 0) || (c == '_');
        if (!ok) c = '_';
    }

    if (s.empty()) s = "ark_gpu_blob";
    if (std::isdigit(static_cast<unsigned char>(s[0])) != 0) s.insert(s.begin(), '_');
    return s;
}

static std::uint64_t fnv1a64(std::string_view s) {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

static std::string hex64(std::uint64_t v) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = kHex[v & 0xF];
        v >>= 4;
    }
    return out;
}

static bool isWindowsAmdGpuBlob(const LinkerConfig& cfg, std::string_view moduleKey) {
    return cfg.toolchain == ToolchainKind::MSVC && moduleKey.ends_with(":amdgpu");
}

static std::string makeSymbolPrefix(const LinkerConfig& cfg, std::string_view moduleKey) {
    if (isWindowsAmdGpuBlob(cfg, moduleKey)) {
        return "ark_gpu_blob_amdgpu_" + hex64(fnv1a64(moduleKey));
    }
    return sanitizeSymbol("ark_gpu_blob_" + std::string(moduleKey));
}

static std::string cEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);

    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out.push_back(static_cast<char>(c));
    }

    return out;
}

static bool writeText(const std::string& path, const std::string& text, std::string& outErr) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
        outErr = ec.message();
        return false;
    }

    os << text;
    os.close();
    return true;
}

static bool writeBytes(const std::string& path, llvm::StringRef bytes, std::string& outErr) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
    if (ec) {
        outErr = ec.message();
        return false;
    }

    os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    os.close();
    return true;
}

static std::string buildRegistryTU(const LinkerConfig& cfg,
                                   const std::vector<CompiledGpuModule>& mods,
                                   const std::vector<std::string>& symPrefixes) {
    std::ostringstream o;

    o << "#include \"gpu/gpu_fatbin.h\"\n";
    o << "#include <cstddef>\n";
    o << "#include <cstdint>\n\n";

    o << "#if defined(__GNUC__) || defined(__clang__)\n";
    o << "  #define ARK_FATBIN_USED __attribute__((used))\n";
    o << "#else\n";
    o << "  #define ARK_FATBIN_USED\n";
    o << "#endif\n\n";

    for (std::size_t i = 0; i < mods.size(); ++i) {
        const std::string& p = symPrefixes[i];

        if (cfg.toolchain == ToolchainKind::MSVC) {
            o << "extern \"C\" const unsigned char " << p << "_start[];\n";
            o << "extern \"C\" const std::uint64_t " << p << "_size;\n";
        } else if (cfg.toolchain == ToolchainKind::Apple) {
            o << "extern \"C\" const unsigned char _" << p << "_start[];\n";
            o << "extern \"C\" const unsigned char _" << p << "_end[];\n";
        } else {
            o << "extern \"C\" const unsigned char " << p << "_start[];\n";
            o << "extern \"C\" const unsigned char " << p << "_end[];\n";
        }
    }

    if (!mods.empty()) o << "\n";

    for (std::size_t i = 0; i < mods.size(); ++i) {
        o << "static const char* const kKernels_" << i << "[] = {\n";
        for (const auto& k : mods[i].kernels) {
            o << "  \"" << cEscape(k) << "\",\n";
        }
        o << "  nullptr\n";
        o << "};\n\n";
    }

    o << "ARK_GPU_EXPORT ARK_FATBIN_USED const ark_gpu_fatbin_v1* ark_gpu_fatbin_query_v1() {\n";
    o << "  constexpr std::uint32_t kCount = " << static_cast<std::uint32_t>(mods.size()) << "u;\n";
    o << "  static ark_gpu_fatbin_entry_v1 entries[(kCount == 0u) ? 1u : kCount];\n";
    o << "  static ark_gpu_fatbin_v1 fb;\n";
    o << "  static bool inited = false;\n";
    o << "  if (!inited) {\n";
    o << "    fb.magic = ARK_GPU_FATBIN_V1_MAGIC;\n";
    o << "    fb.abi = ARK_GPU_FATBIN_ABI_V1;\n";
    o << "    fb.size = static_cast<std::uint32_t>(sizeof(ark_gpu_fatbin_v1));\n";
    o << "    fb.entry_count = kCount;\n";
    o << "    fb.entries = (kCount == 0u) ? nullptr : entries;\n";
    o << "    for (std::size_t i = 0; i < 8; ++i) fb.reserved_u32[i] = 0u;\n";
    o << "    for (std::size_t i = 0; i < 8; ++i) fb.reserved_ptr[i] = nullptr;\n\n";

    for (std::size_t i = 0; i < mods.size(); ++i) {
        const auto& m = mods[i];
        const std::string& p = symPrefixes[i];

        o << "    {\n";
        o << "      ark_gpu_fatbin_entry_v1& e = entries[" << i << "];\n";
        o << "      e.size = static_cast<std::uint32_t>(sizeof(ark_gpu_fatbin_entry_v1));\n";
        o << "      e.module_kind = " << static_cast<std::uint32_t>(m.moduleKind) << "u;\n";
        o << "      e.kernel_count = " << static_cast<std::uint32_t>(m.kernels.size()) << "u;\n";
        o << "      e.reserved0 = 0u;\n";
        o << "      e.module_key = \"" << cEscape(m.moduleKey) << "\";\n";

        if (cfg.toolchain == ToolchainKind::MSVC) {
            o << "      e.blob_ptr = reinterpret_cast<const void*>(" << p << "_start);\n";
            o << "      e.blob_size = static_cast<std::size_t>(" << p << "_size);\n";
        } else if (cfg.toolchain == ToolchainKind::Apple) {
            o << "      e.blob_ptr = reinterpret_cast<const void*>(_" << p << "_start);\n";
            o << "      e.blob_size = static_cast<std::size_t>(_" << p << "_end - _" << p << "_start);\n";
        } else {
            o << "      e.blob_ptr = reinterpret_cast<const void*>(" << p << "_start);\n";
            o << "      e.blob_size = static_cast<std::size_t>(" << p << "_end - " << p << "_start);\n";
        }

        o << "      e.kernels = (e.kernel_count == 0u) ? nullptr : kKernels_" << i << ";\n";
        o << "      for (std::size_t r = 0; r < 8; ++r) e.reserved[r] = nullptr;\n";
        o << "    }\n\n";
    }

    o << "    inited = true;\n";
    o << "  }\n";
    o << "  return &fb;\n";
    o << "}\n";

    return o.str();
}

} // namespace

std::optional<FatbinArtifacts> Linker::emitGpuFatbinArtifacts(const std::vector<CompiledGpuModule>& mods) {
    FatbinArtifacts art;
    art.tmpDir = makeTempDir();

    std::vector<std::string> symPrefixes;
    symPrefixes.reserve(mods.size());

    for (const auto& mod : mods) {
        const std::string symbol = makeSymbolPrefix(cfg_, mod.moduleKey);
        symPrefixes.push_back(symbol);

        llvm::SmallString<256> binPath(art.tmpDir);
        llvm::sys::path::append(binPath, symbol + ".bin");

        std::string err;
        if (!writeBytes(std::string(binPath.str()), mod.blob, err)) {
            hud_.error("Failed to write GPU module blob: " + err);
            return std::nullopt;
        }

        auto embed = createEmbeddingSource(symbol, std::string(binPath.str()));

        llvm::SmallString<256> embedPath(art.tmpDir);
        llvm::sys::path::append(embedPath, symbol + (embed.kind == EmbeddingKind::Cpp ? ".cpp" : ".s"));

        if (!writeText(std::string(embedPath.str()), embed.text, err)) {
            hud_.error("Failed to write GPU blob embedding source: " + err);
            return std::nullopt;
        }

        art.extraSources.push_back(std::string(embedPath.str()));
    }

    {
        const std::string registrySrc = buildRegistryTU(cfg_, mods, symPrefixes);

        llvm::SmallString<256> regPath(art.tmpDir);
        llvm::sys::path::append(regPath, "ark_gpu_fatbin_registry.cpp");

        std::string err;
        if (!writeText(std::string(regPath.str()), registrySrc, err)) {
            hud_.error("Failed to write GPU fatbin registry TU: " + err);
            return std::nullopt;
        }

        art.extraSources.push_back(std::string(regPath.str()));
    }

    return art;
}

} // namespace ark::compiler::pipeline