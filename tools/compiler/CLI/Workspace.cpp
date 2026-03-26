#include "Workspace.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MD5.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <toml++/toml.hpp>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ark::cli {
namespace {

[[noreturn]] void fatal(arklang::hud::Hud& hud, const std::string& msg) {
    hud.error(msg);
    std::exit(1);
}

std::string toString(llvm::StringRef s) {
    return std::string(s.data(), s.size());
}

llvm::StringRef toLlvmRef(std::string_view s) {
    return llvm::StringRef(s.data(), s.size());
}

void md5Sep(llvm::MD5& hash) {
    hash.update(llvm::StringRef("\0", 1));
}

std::string normalizePath(std::string_view p) {
    llvm::SmallString<512> path(p);
    llvm::sys::path::remove_dots(path, true);
    return toString(path.str());
}

std::string makeAbsolutePath(std::string_view p) {
    llvm::SmallString<512> path(p);

    if (path.empty()) {
        llvm::SmallString<512> cwd;
        if (llvm::sys::fs::current_path(cwd)) return ".";
        llvm::sys::path::remove_dots(cwd, true);
        return toString(cwd.str());
    }

    if (llvm::sys::fs::make_absolute(path)) {
        return normalizePath(p);
    }

    llvm::sys::path::remove_dots(path, true);
    return toString(path.str());
}

std::string parentDirOf(std::string_view p) {
    llvm::SmallString<512> path(p);
    llvm::sys::path::remove_dots(path, true);
    llvm::StringRef parent = llvm::sys::path::parent_path(path);
    if (parent.empty()) return ".";
    return toString(parent);
}

std::string joinPath(std::string_view a, std::string_view b) {
    llvm::SmallString<512> out(a);
    llvm::sys::path::append(out, toLlvmRef(b));
    llvm::sys::path::remove_dots(out, true);
    return toString(out.str());
}

bool pathExists(std::string_view p) {
    return llvm::sys::fs::exists(toLlvmRef(p));
}

bool isRegularFile(std::string_view p) {
    llvm::sys::fs::file_status st;
    if (llvm::sys::fs::status(toLlvmRef(p), st)) return false;
    return llvm::sys::fs::is_regular_file(st);
}

bool isDirectory(std::string_view p) {
    llvm::sys::fs::file_status st;
    if (llvm::sys::fs::status(toLlvmRef(p), st)) return false;
    return llvm::sys::fs::is_directory(st);
}

bool entryIsRegularFile(const llvm::sys::fs::directory_entry& entry) {
    auto stOrErr = entry.status();
    if (!stOrErr) return false;
    return llvm::sys::fs::is_regular_file(*stOrErr);
}

bool entryIsDirectory(const llvm::sys::fs::directory_entry& entry) {
    auto stOrErr = entry.status();
    if (!stOrErr) return false;
    return llvm::sys::fs::is_directory(*stOrErr);
}

std::optional<std::string> findWorkspaceManifestUpwardsFrom(std::string_view startPath) {
    llvm::SmallString<512> dir(startPath);

    if (dir.empty()) {
        if (llvm::sys::fs::current_path(dir)) return std::nullopt;
    }

    if (!isDirectory(toString(dir))) {
        llvm::StringRef parent = llvm::sys::path::parent_path(dir);
        if (parent.empty()) return std::nullopt;
        dir = llvm::SmallString<512>(parent);
    }

    llvm::sys::path::remove_dots(dir, true);

    for (;;) {
        llvm::SmallString<512> manifest = dir;
        llvm::sys::path::append(manifest, "ark.toml");

        if (llvm::sys::fs::exists(manifest)) {
            return toString(manifest.str());
        }

        llvm::StringRef parentRef = llvm::sys::path::parent_path(dir);
        if (parentRef.empty() || parentRef == dir.str()) break;

        llvm::SmallString<512> parent(parentRef);
        if (parent == dir) break;
        dir = parent;
    }

    return std::nullopt;
}

std::optional<std::string> findWorkspaceManifestUpwards() {
    llvm::SmallString<512> cwd;
    if (llvm::sys::fs::current_path(cwd)) return std::nullopt;
    return findWorkspaceManifestUpwardsFrom(toString(cwd.str()));
}

void md5UpdateTagged(llvm::MD5& hash, std::string_view key, std::string_view value) {
    hash.update(toLlvmRef(key));
    md5Sep(hash);
    hash.update(toLlvmRef(value));
    md5Sep(hash);
}

void md5UpdateBool(llvm::MD5& hash, std::string_view key, bool value) {
    md5UpdateTagged(hash, key, value ? "1" : "0");
}

void md5UpdateFileContentsIfPresent(llvm::MD5& hash, std::string_view tag, const std::string& path) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufOrErr) {
        md5UpdateTagged(hash, tag, "<missing>");
        return;
    }

    md5UpdateTagged(hash, std::string(tag) + ":path", path);
    hash.update((*bufOrErr)->getBuffer());
    md5Sep(hash);
}

void md5UpdateDirectoryArkSourcesIfPresent(llvm::MD5& hash, std::string_view tag, const std::string& rootDir) {
    if (!pathExists(rootDir) || !isDirectory(rootDir)) {
        md5UpdateTagged(hash, std::string(tag) + ":dir", "<missing>");
        return;
    }

    md5UpdateTagged(hash, std::string(tag) + ":dir", normalizePath(rootDir));

    std::error_code ec;
    llvm::sys::fs::recursive_directory_iterator it(toLlvmRef(rootDir), ec), end;

    std::vector<std::string> files;
    for (; !ec && it != end; it.increment(ec)) {
        const auto& entry = *it;
        if (!entryIsRegularFile(entry)) continue;

        const llvm::StringRef p = entry.path();
        if (llvm::sys::path::extension(p) != ".ark") continue;

        files.push_back(normalizePath(toString(p)));
    }

    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
        md5UpdateFileContentsIfPresent(hash, std::string(tag) + ":file", f);
    }
}

std::string resolveForWorkspace(const WorkspaceConfig& config, std::string_view p) {
    llvm::StringRef ref = toLlvmRef(p);
    if (llvm::sys::path::is_absolute(ref)) return normalizePath(p);
    return joinPath(config.rootDir, p);
}

void pushUniquePath(std::vector<std::string>& out, const std::string& path) {
    const std::string norm = makeAbsolutePath(path);
    if (norm.empty()) return;

    if (std::find(out.begin(), out.end(), norm) == out.end()) {
        out.push_back(norm);
    }
}

void buildModuleSearchPaths(WorkspaceConfig& config) {
    config.moduleSearchPaths.clear();

    pushUniquePath(config.moduleSearchPaths, config.rootDir);
    pushUniquePath(config.moduleSearchPaths, joinPath(config.rootDir, "src"));

    const std::string arkDepsRoot = joinPath(config.rootDir, ".ark/deps");
    if (pathExists(arkDepsRoot) && isDirectory(arkDepsRoot)) {
        pushUniquePath(config.moduleSearchPaths, arkDepsRoot);

        std::error_code ec;
        llvm::sys::fs::directory_iterator it(toLlvmRef(arkDepsRoot), ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            const auto& entry = *it;
            if (!entryIsDirectory(entry)) continue;

            const std::string depPath = normalizePath(toString(entry.path()));
            pushUniquePath(config.moduleSearchPaths, depPath);

            const std::string depSrc = joinPath(depPath, "src");
            if (pathExists(depSrc) && isDirectory(depSrc)) {
                pushUniquePath(config.moduleSearchPaths, depSrc);
            }
        }
    }

    for (const auto& dep : config.dependencies) {
        if (!dep.localPath) continue;

        const std::string resolvedLocal = config.isAdHoc
            ? makeAbsolutePath(*dep.localPath)
            : resolveForWorkspace(config, *dep.localPath);

        if (!pathExists(resolvedLocal) || !isDirectory(resolvedLocal)) continue;

        pushUniquePath(config.moduleSearchPaths, resolvedLocal);

        const std::string depSrc = joinPath(resolvedLocal, "src");
        if (pathExists(depSrc) && isDirectory(depSrc)) {
            pushUniquePath(config.moduleSearchPaths, depSrc);
        }
    }
}

} // namespace

WorkspaceConfig Workspace::discover(std::optional<std::string> explicitInput, arklang::hud::Hud& hud) {
    WorkspaceConfig config;

    if (explicitInput) {
        const std::string inputPath = makeAbsolutePath(*explicitInput);

        if (!pathExists(inputPath)) {
            fatal(hud, "Input path does not exist: " + inputPath);
        }

        if (isRegularFile(inputPath)) {
            const std::string filename = llvm::sys::path::filename(llvm::StringRef(inputPath)).str();

            if (filename == "ark.toml") {
                explicitInput = inputPath;
            } else {
                config.isAdHoc = true;
                config.manifestPath.clear();
                config.entryFile = inputPath;
                config.entryFileResolved = inputPath;
                config.rootDir = parentDirOf(inputPath);
                config.buildDir = joinPath(config.rootDir, ".ark/build");
                config.projectName = llvm::sys::path::stem(llvm::StringRef(inputPath)).str();
                if (config.projectName.empty()) config.projectName = "app";

                buildModuleSearchPaths(config);
                return config;
            }
        } else if (isDirectory(inputPath)) {
            const auto manifestOpt = findWorkspaceManifestUpwardsFrom(inputPath);
            if (!manifestOpt) {
                fatal(hud, "No ark.toml found in directory or parents of: " + inputPath);
            }
            explicitInput = *manifestOpt;
        } else {
            fatal(hud, "Input path is neither a regular file nor a directory: " + inputPath);
        }
    }

    std::optional<std::string> discoveredManifest;
    if (!explicitInput) {
        discoveredManifest = findWorkspaceManifestUpwards();
        if (!discoveredManifest) {
            fatal(hud, "No input file provided and no ark.toml found in current directory or any parent directory.");
        }
    }

    const std::string manifestPath = explicitInput
        ? makeAbsolutePath(*explicitInput)
        : *discoveredManifest;

    if (!pathExists(manifestPath) || !isRegularFile(manifestPath)) {
        fatal(hud, "Workspace manifest not found: " + manifestPath);
    }

    const std::string workspaceRoot = parentDirOf(manifestPath);

    config.isAdHoc = false;
    config.manifestPath = manifestPath;
    config.rootDir = workspaceRoot;
    config.buildDir = joinPath(workspaceRoot, ".ark/build");

    try {
        toml::table tbl = toml::parse_file(manifestPath);

        if (auto* pkg = tbl.get_as<toml::table>("package")) {
            if (auto* name = pkg->get_as<std::string>("name")) {
                config.projectName = name->get();
            }
        }

        if (auto* build = tbl.get_as<toml::table>("build")) {
            if (auto* entry = build->get_as<std::string>("entry")) {
                config.entryFile = entry->get();
            }
            if (auto* target = build->get_as<std::string>("target")) {
                config.profile.targetTriple = target->get();
            }
            if (auto* outDir = build->get_as<std::string>("build_dir")) {
                config.buildDir = resolveForWorkspace(config, outDir->get());
            }
        }

        if (auto* deps = tbl.get_as<toml::table>("dependencies")) {
            config.dependencies.reserve(deps->size());

            for (auto&& [key, val] : *deps) {
                Dependency d;
                d.name = std::string(key.str());

                if (val.is_string()) {
                    d.version = val.as_string()->get();
                } else if (val.is_table()) {
                    auto& dtbl = *val.as_table();

                    d.version = dtbl["version"].value<std::string>().value_or("");

                    if (auto p = dtbl["path"].value<std::string>()) d.localPath = *p;
                    if (auto g = dtbl["git"].value<std::string>()) d.gitUrl = *g;

                    if (d.localPath && d.gitUrl) {
                        fatal(hud, "Dependency '" + d.name + "' cannot specify both 'path' and 'git'.");
                    }
                } else {
                    fatal(hud, "Dependency '" + d.name + "' must be a string or table.");
                }

                config.dependencies.push_back(std::move(d));
            }
        }
    } catch (const toml::parse_error& err) {
        fatal(hud, "Failed to parse " + manifestPath + ": " + std::string(err.description()));
    }

    if (config.entryFile.empty()) config.entryFile = "src/main.ark";
    if (config.projectName.empty()) config.projectName = "app";

    config.entryFileResolved = resolveForWorkspace(config, config.entryFile);
    if (!pathExists(config.entryFileResolved) || !isRegularFile(config.entryFileResolved)) {
        fatal(hud, "Entry file not found: " + config.entryFileResolved);
    }

    for (const auto& dep : config.dependencies) {
        if (!dep.localPath) continue;

        const std::string resolvedLocal = resolveForWorkspace(config, *dep.localPath);
        if (!pathExists(resolvedLocal) || !isDirectory(resolvedLocal)) {
            fatal(hud, "Local dependency path not found for '" + dep.name + "': " + resolvedLocal);
        }

        const std::string depManifest = joinPath(resolvedLocal, "ark.toml");
        if (!pathExists(depManifest) || !isRegularFile(depManifest)) {
            fatal(hud, "Local dependency '" + dep.name + "' is missing ark.toml: " + depManifest);
        }
    }

    buildModuleSearchPaths(config);
    return config;
}

std::string Workspace::computeCacheKey(const WorkspaceConfig& config, arklang::hud::Hud& hud) {
    (void)hud;

    llvm::MD5 hash;

    md5UpdateBool(hash, "isAdHoc", config.isAdHoc);
    md5UpdateTagged(hash, "rootDir", normalizePath(config.rootDir));
    md5UpdateTagged(hash, "buildDir", normalizePath(config.buildDir));
    md5UpdateTagged(hash, "manifestPath", normalizePath(config.manifestPath));
    md5UpdateTagged(hash, "entryFile", config.entryFile);
    md5UpdateTagged(hash, "entryFileResolved", normalizePath(config.entryFileResolved));
    md5UpdateTagged(hash, "projectName", config.projectName);
    md5UpdateTagged(hash, "targetTriple", config.profile.targetTriple);

    for (const auto& p : config.moduleSearchPaths) {
        md5UpdateTagged(hash, "moduleSearchPath", normalizePath(p));
    }

    const std::string resolvedEntry = config.entryFileResolved.empty()
        ? (config.isAdHoc ? makeAbsolutePath(config.entryFile) : resolveForWorkspace(config, config.entryFile))
        : config.entryFileResolved;

    md5UpdateFileContentsIfPresent(hash, "entry", resolvedEntry);

    if (!config.isAdHoc) {
        const std::string manifest = !config.manifestPath.empty()
            ? config.manifestPath
            : joinPath(config.rootDir, "ark.toml");
        md5UpdateFileContentsIfPresent(hash, "manifest", manifest);

        const std::string lockPath = joinPath(config.rootDir, "ark.lock");
        if (pathExists(lockPath)) {
            md5UpdateFileContentsIfPresent(hash, "lock", lockPath);
        }

        const std::string depsRoot = joinPath(config.rootDir, ".ark/deps");
        if (pathExists(depsRoot) && isDirectory(depsRoot)) {
            md5UpdateDirectoryArkSourcesIfPresent(hash, "deps.cache", depsRoot);
        }
    }

    std::vector<Dependency> deps = config.dependencies;
    std::sort(deps.begin(), deps.end(), [](const Dependency& a, const Dependency& b) {
        if (a.name != b.name) return a.name < b.name;
        if (a.version != b.version) return a.version < b.version;

        const std::string aPath = a.localPath ? *a.localPath : "";
        const std::string bPath = b.localPath ? *b.localPath : "";
        if (aPath != bPath) return aPath < bPath;

        const std::string aGit = a.gitUrl ? *a.gitUrl : "";
        const std::string bGit = b.gitUrl ? *b.gitUrl : "";
        return aGit < bGit;
    });

    for (const auto& dep : deps) {
        md5UpdateTagged(hash, "dep.name", dep.name);
        md5UpdateTagged(hash, "dep.version", dep.version);

        if (dep.localPath) {
            const std::string resolvedLocal = config.isAdHoc
                ? makeAbsolutePath(*dep.localPath)
                : resolveForWorkspace(config, *dep.localPath);

            md5UpdateTagged(hash, "dep.localPath", resolvedLocal);

            const std::string depManifest = joinPath(resolvedLocal, "ark.toml");
            if (pathExists(depManifest)) {
                md5UpdateFileContentsIfPresent(hash, "dep.manifest", depManifest);
            }

            const std::string depSrc = joinPath(resolvedLocal, "src");
            md5UpdateDirectoryArkSourcesIfPresent(hash, "dep.src", depSrc);
        }

        if (dep.gitUrl) {
            md5UpdateTagged(hash, "dep.gitUrl", *dep.gitUrl);
        }
    }

    llvm::MD5::MD5Result result;
    hash.final(result);

    llvm::SmallString<32> out;
    llvm::MD5::stringifyResult(result, out);
    return toString(out.str());
}

} // namespace ark::cli