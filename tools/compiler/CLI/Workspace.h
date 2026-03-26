// tools/compiler/CLI/Workspace.h
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../hud.h"

namespace ark::cli {

struct Dependency {
    std::string name;
    std::string version;
    std::optional<std::string> localPath;
    std::optional<std::string> gitUrl;
};

struct ProfileConfig {
    std::string targetTriple;
    bool isRelease = false;
};

struct WorkspaceConfig {
    bool isAdHoc = false;

    std::string rootDir;
    std::string manifestPath;
    std::string entryFile;
    std::string entryFileResolved;
    std::string buildDir;
    std::string projectName;

    ProfileConfig profile;
    std::vector<Dependency> dependencies;

    // Absolute, normalized search directories for resolving imports.
    std::vector<std::string> moduleSearchPaths;
};

class Workspace {
public:
    static WorkspaceConfig discover(std::optional<std::string> explicitInput, arklang::hud::Hud& hud);
    static std::string computeCacheKey(const WorkspaceConfig& config, arklang::hud::Hud& hud);
};

} // namespace ark::cli