#include "Subcommands.h"

#include <CLI/CLI.hpp>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

#include <toml++/toml.hpp>

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ark::cli {
namespace {

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------
[[noreturn]] void fail(const std::string& msg) {
    llvm::errs() << "[ERROR] " << msg << "\n";
    std::exit(1);
}

void warn(const std::string& msg) {
    llvm::outs() << "[WARN] " << msg << "\n";
}

void note(const std::string& msg) {
    llvm::outs() << msg << "\n";
}

// -----------------------------------------------------------------------------
// Path helpers
// -----------------------------------------------------------------------------
std::string toString(llvm::StringRef s) {
    return std::string(s.data(), s.size());
}

llvm::StringRef toLlvmRef(std::string_view s) {
    return llvm::StringRef(s.data(), s.size());
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
        if (llvm::sys::fs::current_path(cwd)) {
            return ".";
        }

        llvm::sys::path::remove_dots(cwd, true);
        return toString(cwd.str());
    }

    if (llvm::sys::fs::make_absolute(path)) {
        return normalizePath(p);
    }

    llvm::sys::path::remove_dots(path, true);
    return toString(path.str());
}

std::string joinPath(std::string_view a, std::string_view b) {
    llvm::SmallString<512> out(a);
    llvm::sys::path::append(out, toLlvmRef(b));
    llvm::sys::path::remove_dots(out, true);
    return toString(out.str());
}

std::string parentDirOf(std::string_view p) {
    llvm::SmallString<512> path(p);
    llvm::sys::path::remove_dots(path, true);
    const llvm::StringRef parent = llvm::sys::path::parent_path(path);
    if (parent.empty()) {
        return ".";
    }
    return toString(parent);
}

bool pathExists(std::string_view p) {
    return llvm::sys::fs::exists(toLlvmRef(p));
}

bool isRegularFile(std::string_view p) {
    llvm::sys::fs::file_status st;
    if (llvm::sys::fs::status(toLlvmRef(p), st)) {
        return false;
    }
    return llvm::sys::fs::is_regular_file(st);
}

bool isDirectory(std::string_view p) {
    llvm::sys::fs::file_status st;
    if (llvm::sys::fs::status(toLlvmRef(p), st)) {
        return false;
    }
    return llvm::sys::fs::is_directory(st);
}

void createDirOrFail(std::string_view p) {
    const std::error_code ec = llvm::sys::fs::create_directories(toLlvmRef(p));
    if (ec) {
        fail("Failed to create directory '" + std::string(p) + "': " + ec.message());
    }
}

void writeTextFileOrFail(std::string_view path, llvm::StringRef content) {
    std::error_code ec;
    llvm::raw_fd_ostream os(toLlvmRef(path), ec, llvm::sys::fs::OF_Text);
    if (ec) {
        fail("Failed to write file '" + std::string(path) + "': " + ec.message());
    }

    os << content;
    os.flush();

    if (os.has_error()) {
        os.clear_error();
        fail("Failed to write file '" + std::string(path) + "'");
    }
}

// -----------------------------------------------------------------------------
// Manifest discovery
// -----------------------------------------------------------------------------
std::optional<std::string> findManifestUpwardsFrom(std::string_view startPath) {
    llvm::SmallString<512> dir(startPath);

    if (dir.empty()) {
        if (llvm::sys::fs::current_path(dir)) {
            return std::nullopt;
        }
    }

    if (!isDirectory(toString(dir))) {
        const llvm::StringRef parent = llvm::sys::path::parent_path(dir);
        if (parent.empty()) {
            return std::nullopt;
        }
        dir = llvm::SmallString<512>(parent);
    }

    llvm::sys::path::remove_dots(dir, true);

    for (;;) {
        llvm::SmallString<512> manifest = dir;
        llvm::sys::path::append(manifest, "ark.toml");

        if (llvm::sys::fs::exists(manifest)) {
            return toString(manifest.str());
        }

        const llvm::StringRef parentRef = llvm::sys::path::parent_path(dir);
        if (parentRef.empty() || parentRef == dir.str()) {
            break;
        }

        llvm::SmallString<512> parent(parentRef);
        if (parent == dir) {
            break;
        }

        dir = parent;
    }

    return std::nullopt;
}

std::string requireWorkspaceManifest() {
    llvm::SmallString<512> cwd;
    if (llvm::sys::fs::current_path(cwd)) {
        fail("Failed to determine current directory.");
    }

    const auto manifest = findManifestUpwardsFrom(toString(cwd.str()));
    if (!manifest) {
        fail("ark.toml not found in current directory or any parent directory.");
    }

    return *manifest;
}

// -----------------------------------------------------------------------------
// Process helpers
// -----------------------------------------------------------------------------
int runProgram(
    llvm::StringRef program,
    llvm::ArrayRef<llvm::StringRef> args,
    std::string* errOut = nullptr
) {
    std::string errMsg;
    std::optional<llvm::StringRef> redirects[] = {
        std::nullopt,
        std::nullopt,
        std::nullopt
    };

    const int rc = llvm::sys::ExecuteAndWait(
        program,
        args,
        std::nullopt,
        redirects,
        0,
        0,
        &errMsg
    );

    if (errOut) {
        *errOut = std::move(errMsg);
    }

    return rc;
}

int runGit(const std::vector<std::string>& argsOwned) {
    const auto git = llvm::sys::findProgramByName("git");
    if (!git) {
        fail("'git' executable not found in PATH.");
    }

    llvm::SmallVector<llvm::StringRef, 16> argv;
    argv.push_back(*git);
    for (const auto& s : argsOwned) {
        argv.push_back(s);
    }

    std::string errMsg;
    const int rc = runProgram(*git, argv, &errMsg);
    if (rc != 0 && !errMsg.empty()) {
        llvm::errs() << "[git] " << errMsg << "\n";
    }

    return rc;
}

void tryRunArknetFetch() {
    const auto arknet = llvm::sys::findProgramByName("arknet");
    if (!arknet) {
        warn("'arknet' not found in PATH. Run `arknet fetch` manually.");
        return;
    }

    llvm::SmallVector<llvm::StringRef, 4> argv;
    argv.push_back(*arknet);
    argv.push_back("fetch");

    std::string errMsg;
    const int rc = runProgram(*arknet, argv, &errMsg);
    if (rc != 0) {
        warn("Automatic `arknet fetch` failed.");
        if (!errMsg.empty()) {
            llvm::outs() << "[WARN] " << errMsg << "\n";
        }
    }
}

// -----------------------------------------------------------------------------
// TOML helpers
// -----------------------------------------------------------------------------
toml::table loadTomlOrFail(const std::string& path) {
    auto parsed = toml::parse_file(path);

    if (!parsed) {
        std::ostringstream oss;
        oss << parsed.error();
        fail("Failed to parse " + path + ": " + oss.str());
    }

    return std::move(parsed).table();
}

void writeTextFileAtomicOrFail(const std::string& path, const std::string& content) {
    const std::string tmpPath = path + ".tmp";

    {
        std::error_code ec;
        llvm::raw_fd_ostream os(tmpPath, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            fail("Failed to open temp file '" + tmpPath + "': " + ec.message());
        }

        os << content;
        os.flush();

        if (os.has_error()) {
            os.clear_error();
            llvm::sys::fs::remove(tmpPath);
            fail("Failed to write temp file '" + tmpPath + "'");
        }
    }

    const std::error_code ec = llvm::sys::fs::rename(tmpPath, path);
    if (ec) {
        llvm::sys::fs::remove(tmpPath);
        fail("Failed to replace '" + path + "' with temp file: " + ec.message());
    }
}

void saveTomlOrFail(const std::string& path, const toml::table& tbl) {
    std::ostringstream oss;
    oss << tbl;
    writeTextFileAtomicOrFail(path, oss.str());
}

// -----------------------------------------------------------------------------
// Workspace resolution
// -----------------------------------------------------------------------------
struct ResolvedWorkspace {
    std::string manifestPath;
    std::string rootDir;
    std::string depsDir;
};

ResolvedWorkspace resolveWorkspace() {
    ResolvedWorkspace ws;
    ws.manifestPath = requireWorkspaceManifest();
    ws.rootDir = parentDirOf(ws.manifestPath);
    ws.depsDir = joinPath(ws.rootDir, ".ark/deps");
    return ws;
}

std::string depTargetDir(const ResolvedWorkspace& ws, const std::string& pkgName) {
    return joinPath(ws.depsDir, pkgName);
}

} // namespace

// =========================================================================
// 1. arknet add
// =========================================================================
void setupAddCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("add", "Add a dependency to ark.toml");

    auto pkgName = std::make_shared<std::string>();
    auto gitUrl = std::make_shared<std::string>();
    auto pathDep = std::make_shared<std::string>();
    auto version = std::make_shared<std::string>();
    auto noFetch = std::make_shared<bool>(false);

    sub->add_option("name", *pkgName, "The package name to add")->required();
    sub->add_option("--git", *gitUrl, "Git repository URL");
    sub->add_option("--path", *pathDep, "Local dependency path");
    sub->add_option("--version", *version, "Version tag/branch/constraint");
    sub->add_flag("--no-fetch", *noFetch, "Do not run `arknet fetch` after updating manifest");

    sub->callback([pkgName, gitUrl, pathDep, version, noFetch]() {
        if (pkgName->empty()) {
            fail("Package name cannot be empty.");
        }

        if (!gitUrl->empty() && !pathDep->empty()) {
            fail("Use only one source: --git or --path.");
        }

        const ResolvedWorkspace ws = resolveWorkspace();
        toml::table tbl = loadTomlOrFail(ws.manifestPath);

        if (!tbl.contains("dependencies")) {
            tbl.insert("dependencies", toml::table{});
        }

        auto* deps = tbl.get_as<toml::table>("dependencies");
        if (!deps) {
            fail("`dependencies` exists in ark.toml but is not a table.");
        }

        if (!gitUrl->empty()) {
            toml::table depEntry;
            depEntry.insert("git", *gitUrl);
            if (!version->empty()) {
                depEntry.insert("version", *version);
            }
            deps->insert_or_assign(*pkgName, depEntry);
        } else if (!pathDep->empty()) {
            toml::table depEntry;
            depEntry.insert("path", *pathDep);
            if (!version->empty()) {
                depEntry.insert("version", *version);
            }
            deps->insert_or_assign(*pkgName, depEntry);
        } else {
            deps->insert_or_assign(*pkgName, version->empty() ? "*" : *version);
        }

        saveTomlOrFail(ws.manifestPath, tbl);

        llvm::outs() << "Added `" << *pkgName << "` to " << ws.manifestPath << "\n";

        if (!*noFetch) {
            llvm::outs() << "Fetching dependencies...\n";
            tryRunArknetFetch();
        }
    });
}

// =========================================================================
// 2. arknet fetch
// =========================================================================
void setupFetchCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("fetch", "Download dependencies listed in ark.toml");

    sub->callback([]() {
        const ResolvedWorkspace ws = resolveWorkspace();
        createDirOrFail(ws.depsDir);

        toml::table tbl = loadTomlOrFail(ws.manifestPath);
        auto* deps = tbl.get_as<toml::table>("dependencies");

        if (!deps || deps->empty()) {
            llvm::outs() << "No dependencies to fetch.\n";
            return;
        }

        for (auto&& [key, val] : *deps) {
            const std::string pkg = std::string(key.str());
            const std::string targetDir = depTargetDir(ws, pkg);

            if (val.is_string()) {
                const std::string version = val.value<std::string>().value_or("*");
                warn(
                    "Registry dependency `" + pkg + "` (" + version +
                    ") is not implemented yet. Use `--git` or `--path`."
                );
                continue;
            }

            if (!val.is_table()) {
                warn("Skipping dependency `" + pkg + "`: expected string or table.");
                continue;
            }

            auto& dtbl = *val.as_table();

            if (auto local = dtbl["path"].value<std::string>()) {
                std::string resolvedLocal = *local;

                if (!llvm::sys::path::is_absolute(toLlvmRef(resolvedLocal))) {
                    resolvedLocal = joinPath(ws.rootDir, resolvedLocal);
                }

                resolvedLocal = makeAbsolutePath(resolvedLocal);

                if (!pathExists(resolvedLocal) || !isDirectory(resolvedLocal)) {
                    warn("Local dependency `" + pkg + "` path does not exist: " + resolvedLocal);
                    continue;
                }

                llvm::outs() << "[LINK] " << pkg << " -> " << resolvedLocal << "\n";
                continue;
            }

            if (auto gitUrl = dtbl["git"].value<std::string>()) {
                if (pathExists(targetDir)) {
                    llvm::outs() << "[OK] " << pkg << " already fetched at " << targetDir << "\n";
                    continue;
                }

                llvm::outs() << "[FETCH] Cloning " << pkg << " from " << *gitUrl << "...\n";

                std::vector<std::string> args;
                args.emplace_back("clone");
                args.emplace_back("--depth");
                args.emplace_back("1");

                if (auto ver = dtbl["version"].value<std::string>(); ver && !ver->empty()) {
                    args.emplace_back("--branch");
                    args.emplace_back(*ver);
                }

                args.emplace_back(*gitUrl);
                args.emplace_back(targetDir);

                if (runGit(args) != 0) {
                    warn("Failed to fetch `" + pkg + "`.");
                    llvm::sys::fs::remove_directories(targetDir, /*IgnoreErrors=*/true);
                    continue;
                }

                continue;
            }

            warn("Skipping dependency `" + pkg + "`: table must include `git` or `path`.");
        }

        llvm::outs() << "Fetch complete.\n";
    });
}

} // namespace ark::cli