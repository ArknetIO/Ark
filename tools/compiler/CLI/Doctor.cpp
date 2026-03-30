// tools/compiler/CLI/Doctor.cpp
#include "Doctor.h"
#include "Config.h"
#include "Common.h"


#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <toml++/toml.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace ark::cli {
namespace {

// =============================================================================
// Small String Helpers
// =============================================================================
std::string toString(llvm::StringRef s) {
    return std::string(s.data(), s.size());
}

llvm::StringRef toRef(std::string_view s) {
    return llvm::StringRef(s.data(), s.size());
}

std::string trim(std::string s) {
    const auto isWs = [](unsigned char c) { return std::isspace(c) != 0; };

    while (!s.empty() && isWs(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }

    while (!s.empty() && isWs(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }

    return s;
}

bool startsWith(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

bool contains(std::string_view s, std::string_view needle) {
    return s.find(needle) != std::string_view::npos;
}

std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;

    for (char c : text) {
        if (c == '\r') {
            continue;
        }

        if (c == '\n') {
            out.push_back(cur);
            cur.clear();
            continue;
        }

        cur.push_back(c);
    }

    if (!cur.empty()) {
        out.push_back(cur);
    }

    return out;
}

// =============================================================================
// File & Path Helpers
// =============================================================================
bool pathExists(std::string_view p) {
    return llvm::sys::fs::exists(toRef(p));
}

std::uint64_t fileSizeOrZero(std::string_view p) {
    std::uint64_t sz = 0;
    if (llvm::sys::fs::file_size(toRef(p), sz)) {
        return 0;
    }
    return sz;
}

std::string readAllFile(llvm::StringRef p) {
    auto mb = llvm::MemoryBuffer::getFile(p);
    if (!mb) {
        return {};
    }
    return std::string(mb.get()->getBuffer());
}

std::string joinPath(std::string_view a, std::string_view b) {
    llvm::SmallString<512> p(a);
    llvm::sys::path::append(p, toRef(b));
    llvm::sys::path::remove_dots(p, true);
    return toString(p.str());
}

std::string parentDir(std::string_view p) {
    llvm::SmallString<512> s(p);
    return toString(llvm::sys::path::parent_path(s));
}

std::string filenameOf(std::string_view p) {
    llvm::SmallString<512> s(p);
    return toString(llvm::sys::path::filename(s));
}

std::string getMainExePath() {
    void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(getMainExePath));
    return llvm::sys::fs::getMainExecutable(nullptr, mainAddr);
}

std::vector<std::string> listDirEntries(const std::string& dir) {
    std::vector<std::string> out;

    std::error_code ec;
    llvm::sys::fs::directory_iterator it(toRef(dir), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        out.push_back(toString(it->path()));
    }

    std::sort(out.begin(), out.end());
    return out;
}

// =============================================================================
// TOML Helpers
// =============================================================================
std::optional<toml::table> parseTomlIfExists(
    const std::string& path,
    std::vector<std::string>* issues = nullptr
) {
    if (!pathExists(path)) {
        return std::nullopt;
    }

    auto parsed = parseTomlFilePortable(path);
    if (!parsed) {
        if (issues) {
            issues->push_back("Failed to parse " + path + ": " + parsed.error);
        }
        return std::nullopt;
    }

    return parsed.take();
}


// =============================================================================
// Process Execution Helpers
// =============================================================================
struct CommandCapture {
    int exitCode = -1;
    std::string stdoutText;
    std::string stderrText;
    std::string execError;
};

CommandCapture runCommandCapture(llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args) {
    CommandCapture r;

    llvm::SmallString<256> outPath;
    llvm::SmallString<256> errPath;

    if (llvm::sys::fs::createTemporaryFile("ark_doctor_out", "log", outPath)) {
        r.execError = "Failed to create stdout capture file";
        return r;
    }

    if (llvm::sys::fs::createTemporaryFile("ark_doctor_err", "log", errPath)) {
        (void)llvm::sys::fs::remove(outPath);
        r.execError = "Failed to create stderr capture file";
        return r;
    }

    const llvm::StringRef outRef(outPath);
    const llvm::StringRef errRef(errPath);
    std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};

    std::string errMsg;
    r.exitCode = llvm::sys::ExecuteAndWait(
        program,
        args,
        std::nullopt,
        redirects,
        0,
        0,
        &errMsg
    );

    r.execError = errMsg;
    r.stdoutText = readAllFile(outPath);
    r.stderrText = readAllFile(errPath);

    (void)llvm::sys::fs::remove(outPath);
    (void)llvm::sys::fs::remove(errPath);

    return r;
}

std::optional<std::string> findProgram(const char* name) {
    auto p = llvm::sys::findProgramByName(name);
    if (!p) {
        return std::nullopt;
    }
    return *p;
}

// =============================================================================
// Release & Package Inspection
// =============================================================================
void collectReleaseEnv(DoctorReport& rep) {
    std::vector<std::string> candidates;

    llvm::SmallString<512> cwd;
    if (!llvm::sys::fs::current_path(cwd)) {
        candidates.push_back(joinPath(toString(cwd.str()), "ark-ops/release.env"));
    }

    if (!rep.exeDir.empty()) {
        candidates.push_back(joinPath(parentDir(rep.exeDir), "ark-ops/release.env"));
        candidates.push_back(joinPath(rep.exeDir, "ark-ops/release.env"));
    }

    for (const auto& path : candidates) {
        if (!pathExists(path)) {
            continue;
        }

        rep.releaseEnvPath = path;
        const std::string text = readAllFile(path);

        for (auto line : splitLines(text)) {
            line = trim(line);
            if (line.empty() || startsWith(line, "#")) {
                continue;
            }

            if (startsWith(line, "export ")) {
                line = trim(line.substr(7));
            }

            const auto eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }

            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));

            if (val.size() >= 2 &&
                ((val.front() == '"' && val.back() == '"') ||
                 (val.front() == '\'' && val.back() == '\''))) {
                val = val.substr(1, val.size() - 2);
            }

            if (!key.empty()) {
                rep.releaseVersions[key] = val;
            }
        }

        return;
    }

    rep.warnings.push_back("release.env not found (searched cwd and executable-relative locations)");
}

void collectPackageArtifacts(DoctorReport& rep) {
    if (rep.exeDir.empty() || !pathExists(rep.exeDir)) {
        return;
    }

    for (const auto& p : listDirEntries(rep.exeDir)) {
        const std::string name = filenameOf(p);
        if (name.empty()) {
            continue;
        }

        if (name == "arknet" ||
            name == "arkc" ||
            name == "ark-stub" ||
            name == "ark-provider" ||
            name == "arknet-registryd" ||
            startsWith(name, "arknet-") ||
            startsWith(name, "ark-compiler-") ||
            startsWith(name, "ark-provider-") ||
            startsWith(name, "ark-registry-")) {
            rep.packageArtifacts.push_back(name);
        }
    }

    std::sort(rep.packageArtifacts.begin(), rep.packageArtifacts.end());
    rep.packageArtifacts.erase(
        std::unique(rep.packageArtifacts.begin(), rep.packageArtifacts.end()),
        rep.packageArtifacts.end()
    );
}

// =============================================================================
// GPU Plugin Inspection
// =============================================================================
std::string gpuLibExt() {
#if defined(__APPLE__)
    return "dylib";
#elif defined(_WIN32)
    return "dll";
#else
    return "so";
#endif
}

bool fileContainsMarkerViaStrings(const std::string& path, const std::string& marker) {
    const auto stringsBin = findProgram("strings");
    if (!stringsBin) {
        return false;
    }

    llvm::SmallVector<llvm::StringRef, 8> argv;
    argv.push_back(llvm::StringRef(stringsBin->data(), stringsBin->size()));
    argv.push_back(llvm::StringRef(path.data(), path.size()));

    const auto cap = runCommandCapture(argv[0], argv);
    if (cap.exitCode != 0) {
        return false;
    }

    return contains(cap.stdoutText, marker);
}

void collectGpuPluginStatus(DoctorReport& rep) {
    if (rep.exeDir.empty()) {
        return;
    }

    const std::string pluginDir = joinPath(rep.exeDir, "lib");
    const std::string ext = gpuLibExt();

    struct ProbeDef {
        const char* backend;
        const char* base;
        const char* disabledMarker;
    };

    const ProbeDef defs[] = {
        {"cuda",  "libark_cuda_backend",  "cuda(disabled)"},
        {"hip",   "libark_hip_backend",   "hip(disabled)"},
        {"metal", "libark_metal_backend", "metal(disabled)"},
    };

    for (const auto& d : defs) {
        DoctorGpuPluginProbe p;
        p.backend = d.backend;
        p.path = joinPath(pluginDir, std::string(d.base) + "." + ext);
        p.exists = pathExists(p.path);

        if (!p.exists) {
            p.status = "missing";
            rep.gpuPlugins.push_back(std::move(p));
            continue;
        }

        p.disabledMarkerFound = fileContainsMarkerViaStrings(p.path, d.disabledMarker);
        p.status = p.disabledMarkerFound ? "disabled-marker-present" : "ok";
        rep.gpuPlugins.push_back(std::move(p));
    }
}

// =============================================================================
// Host GPU Inspection
// =============================================================================
void collectHostGpuHardware(DoctorReport& rep) {
#if defined(__APPLE__)
    if (auto sp = findProgram("system_profiler")) {
        llvm::SmallVector<llvm::StringRef, 8> argv;
        argv.push_back(llvm::StringRef(sp->data(), sp->size()));
        argv.push_back("SPDisplaysDataType");

        const auto cap = runCommandCapture(argv[0], argv);
        if (cap.exitCode == 0) {
            for (auto line : splitLines(cap.stdoutText)) {
                line = trim(line);
                if (startsWith(line, "Chipset Model:") ||
                    startsWith(line, "Vendor:") ||
                    startsWith(line, "Metal Support:")) {
                    rep.hostGpus.push_back(line);
                }
            }
        }
    }
#elif defined(_WIN32)
    if (auto nvsmi = findProgram("nvidia-smi")) {
        llvm::SmallVector<llvm::StringRef, 8> argv;
        argv.push_back(llvm::StringRef(nvsmi->data(), nvsmi->size()));
        argv.push_back("--query-gpu=name,driver_version");
        argv.push_back("--format=csv,noheader");

        const auto cap = runCommandCapture(argv[0], argv);
        if (cap.exitCode == 0) {
            for (auto line : splitLines(cap.stdoutText)) {
                line = trim(line);
                if (!line.empty()) {
                    rep.hostGpus.push_back(std::string("NVIDIA: ") + line);
                }
            }
        }
    }

    if (auto powershell = findProgram("powershell")) {
        llvm::SmallVector<llvm::StringRef, 16> argv;
        argv.push_back(llvm::StringRef(powershell->data(), powershell->size()));
        argv.push_back("-NoProfile");
        argv.push_back("-Command");
        argv.push_back(
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty Name"
        );

        const auto cap = runCommandCapture(argv[0], argv);
        if (cap.exitCode == 0) {
            for (auto line : splitLines(cap.stdoutText)) {
                line = trim(line);
                if (line.empty()) {
                    continue;
                }

                const std::string tagged = std::string("Windows: ") + line;
                if (std::find(rep.hostGpus.begin(), rep.hostGpus.end(), tagged) == rep.hostGpus.end()) {
                    rep.hostGpus.push_back(tagged);
                }
            }
        }
    }
#else
    if (auto nvsmi = findProgram("nvidia-smi")) {
        llvm::SmallVector<llvm::StringRef, 8> argv;
        argv.push_back(llvm::StringRef(nvsmi->data(), nvsmi->size()));
        argv.push_back("--query-gpu=name,driver_version");
        argv.push_back("--format=csv,noheader");

        const auto cap = runCommandCapture(argv[0], argv);
        if (cap.exitCode == 0) {
            for (auto line : splitLines(cap.stdoutText)) {
                line = trim(line);
                if (!line.empty()) {
                    rep.hostGpus.push_back(std::string("NVIDIA: ") + line);
                }
            }
        }
    }

    if (auto rocmsmi = findProgram("rocm-smi")) {
        llvm::SmallVector<llvm::StringRef, 8> argv;
        argv.push_back(llvm::StringRef(rocmsmi->data(), rocmsmi->size()));
        argv.push_back("--showproductname");
        argv.push_back("--showdriverversion");

        const auto cap = runCommandCapture(argv[0], argv);
        if (cap.exitCode == 0) {
            for (auto line : splitLines(cap.stdoutText)) {
                line = trim(line);
                if (line.empty()) {
                    continue;
                }

                if (contains(line, "GPU") ||
                    contains(line, "Card series") ||
                    contains(line, "Driver version")) {
                    rep.hostGpus.push_back(std::string("ROCm: ") + line);
                }
            }
        }
    }

    if (rep.hostGpus.empty()) {
        if (auto lspci = findProgram("lspci")) {
            llvm::SmallVector<llvm::StringRef, 4> argv;
            argv.push_back(llvm::StringRef(lspci->data(), lspci->size()));

            const auto cap = runCommandCapture(argv[0], argv);
            if (cap.exitCode == 0) {
                for (auto line : splitLines(cap.stdoutText)) {
                    const std::string l = trim(line);
                    if (contains(l, "VGA compatible controller") ||
                        contains(l, "3D controller") ||
                        contains(l, "Display controller")) {
                        rep.hostGpus.push_back(std::string("PCI: ") + l);
                    }
                }
            }
        }
    }
#endif

    if (rep.hostGpus.empty()) {
        rep.warnings.push_back("No GPU hardware info detected via system tools");
    }
}

// =============================================================================
// Arknet Home Inspection
// =============================================================================
void collectArknetHomeState(DoctorReport& rep) {
    rep.arknetDir = GlobalConfig::getArknetDir();

    rep.configFile.path = GlobalConfig::getConfigFilePath();
    rep.configFile.exists = pathExists(rep.configFile.path);
    rep.configFile.sizeBytes = rep.configFile.exists ? fileSizeOrZero(rep.configFile.path) : 0;

    rep.credentialsFile.path = GlobalConfig::getCredentialsFilePath();
    rep.credentialsFile.exists = pathExists(rep.credentialsFile.path);
    rep.credentialsFile.sizeBytes = rep.credentialsFile.exists ? fileSizeOrZero(rep.credentialsFile.path) : 0;

    if (rep.configFile.exists) {
        auto tblOpt = parseTomlIfExists(rep.configFile.path, &rep.issues);
        if (tblOpt) {
            for (auto&& [secKey, secVal] : *tblOpt) {
                if (auto* t = secVal.as_table()) {
                    for (auto&& [paramKey, _] : *t) {
                        rep.configKeys.push_back(
                            std::string(secKey.str()) + "." + std::string(paramKey.str())
                        );
                    }
                } else {
                    rep.configKeys.push_back(std::string(secKey.str()));
                }
            }

            std::sort(rep.configKeys.begin(), rep.configKeys.end());
        }
    }

    if (rep.credentialsFile.exists) {
        auto tblOpt = parseTomlIfExists(rep.credentialsFile.path, &rep.issues);
        if (tblOpt) {
            if (const auto* rec = tblOpt->get_as<toml::table>("auth")) {
                if (auto* nested = rec->get_as<toml::table>("provider_token")) {
                    if (auto* alg = nested->get_as<std::string>("algorithm")) {
                        rep.credentialAlgorithm = alg->get();
                    }
                    if (auto* ct = nested->get_as<std::string>("ciphertext")) {
                        rep.credentialsCiphertextPresent = !ct->get().empty();
                    }
                } else {
                    if (auto* alg = rec->get_as<std::string>("algorithm")) {
                        rep.credentialAlgorithm = alg->get();
                    }
                    if (auto* ct = rec->get_as<std::string>("ciphertext")) {
                        rep.credentialsCiphertextPresent = !ct->get().empty();
                    }
                }
            }
        }
    }

    const std::string libsDir = joinPath(rep.arknetDir, "lib");
    if (pathExists(libsDir)) {
        for (const auto& p : listDirEntries(libsDir)) {
            rep.packageArtifacts.push_back(std::string("~/.arknet/lib/") + filenameOf(p));
        }

        std::sort(rep.packageArtifacts.begin(), rep.packageArtifacts.end());
        rep.packageArtifacts.erase(
            std::unique(rep.packageArtifacts.begin(), rep.packageArtifacts.end()),
            rep.packageArtifacts.end()
        );
    }
}

llvm::json::Array toJsonArray(const std::vector<std::string>& xs) {
    llvm::json::Array arr;
    for (const auto& s : xs) {
        arr.push_back(s);
    }
    return arr;
}

} // namespace

DoctorReport collectDoctorReport(const DoctorOptions& opts) {
    (void)opts;

    DoctorReport rep;
    rep.osInfo = llvm::sys::getProcessTriple();
    rep.llvmInfo = "LLVM/MLIR Engine";

    rep.exePath = getMainExePath();
    rep.exeDir = parentDir(rep.exePath);

    collectReleaseEnv(rep);
    collectPackageArtifacts(rep);
    collectArknetHomeState(rep);
    collectHostGpuHardware(rep);
    collectGpuPluginStatus(rep);

    for (const auto& p : rep.gpuPlugins) {
        if (p.exists && p.disabledMarkerFound) {
            rep.issues.push_back("GPU plugin disabled marker present: " + p.backend);
        }
    }

    if (rep.osInfo.empty()) {
        rep.issues.push_back("Host process triple unavailable");
    }

    if (rep.llvmInfo.empty()) {
        rep.issues.push_back("LLVM engine info unavailable");
    }

    rep.ok = rep.issues.empty();
    return rep;
}

void printDoctorReportPretty(const DoctorReport& rep) {
    llvm::outs() << "Arknet Environment Diagnostics\n";
    llvm::outs() << "==============================\n\n";

    llvm::outs() << "[Core]\n";
    llvm::outs() << "Platform: " << rep.osInfo << "\n";
    llvm::outs() << "Engine:   " << rep.llvmInfo << "\n";
    llvm::outs() << "Exe:      " << rep.exePath << "\n";
    llvm::outs() << "Bin Dir:  " << rep.exeDir << "\n\n";

    llvm::outs() << "[Release]\n";
    if (!rep.releaseEnvPath.empty()) {
        llvm::outs() << "release.env: " << rep.releaseEnvPath << "\n";
        for (const auto& [k, v] : rep.releaseVersions) {
            if (k == "ARK_SYSTEM_VERSION" ||
                k == "VER_COMPILER" ||
                k == "VER_PROVIDER" ||
                k == "VER_REGISTRY") {
                llvm::outs() << "  " << k << " = " << v << "\n";
            }
        }
    } else {
        llvm::outs() << "release.env: (not found)\n";
    }
    llvm::outs() << "\n";

    llvm::outs() << "[Package Artifacts]\n";
    if (rep.packageArtifacts.empty()) {
        llvm::outs() << "  (none detected)\n";
    } else {
        for (const auto& a : rep.packageArtifacts) {
            llvm::outs() << "  - " << a << "\n";
        }
    }
    llvm::outs() << "\n";

    llvm::outs() << "[~/.arknet]\n";
    llvm::outs() << "Dir: " << rep.arknetDir << "\n";
    llvm::outs() << "Config: " << (rep.configFile.exists ? "present" : "missing");
    if (rep.configFile.exists) {
        llvm::outs() << " (" << rep.configFile.sizeBytes << " bytes)";
    }
    llvm::outs() << "\n";

    llvm::outs() << "Credentials: " << (rep.credentialsFile.exists ? "present" : "missing");
    if (rep.credentialsFile.exists) {
        llvm::outs() << " (" << rep.credentialsFile.sizeBytes << " bytes)";
    }
    llvm::outs() << "\n";

    if (!rep.credentialAlgorithm.empty()) {
        llvm::outs() << "Credential Algorithm: " << rep.credentialAlgorithm << "\n";
    }

    if (rep.credentialsFile.exists) {
        llvm::outs() << "Ciphertext Present: " << (rep.credentialsCiphertextPresent ? "yes" : "no") << "\n";
    }

    if (!rep.configKeys.empty()) {
        llvm::outs() << "Config Keys:\n";
        for (const auto& k : rep.configKeys) {
            llvm::outs() << "  - " << k << "\n";
        }
    }
    llvm::outs() << "\n";

    llvm::outs() << "[Host GPU]\n";
    if (rep.hostGpus.empty()) {
        llvm::outs() << "  (no GPU info detected)\n";
    } else {
        for (const auto& g : rep.hostGpus) {
            llvm::outs() << "  - " << g << "\n";
        }
    }
    llvm::outs() << "\n";

    llvm::outs() << "[GPU Backend Plugins]\n";
    if (rep.gpuPlugins.empty()) {
        llvm::outs() << "  (no probes)\n";
    } else {
        for (const auto& p : rep.gpuPlugins) {
            llvm::outs() << "  - " << p.backend << ": " << p.status << "\n";
            llvm::outs() << "      path: " << p.path << "\n";
        }
    }
    llvm::outs() << "\n";

    if (!rep.warnings.empty()) {
        llvm::outs() << "[Warnings]\n";
        for (const auto& w : rep.warnings) {
            llvm::outs() << "  - " << w << "\n";
        }
        llvm::outs() << "\n";
    }

    if (!rep.issues.empty()) {
        llvm::outs() << "[Issues]\n";
        for (const auto& e : rep.issues) {
            llvm::outs() << "  - " << e << "\n";
        }
        llvm::outs() << "\n";
    }

    llvm::outs() << (rep.ok ? "OK: Environment looks healthy.\n"
                            : "ERROR: Issues detected.\n");
}

void printDoctorReportJson(const DoctorReport& rep) {
    llvm::json::Object root;
    root["status"] = rep.ok ? "ok" : "error";
    root["ok"] = rep.ok;

    root["platform"] = rep.osInfo;
    root["engine"] = rep.llvmInfo;
    root["exe_path"] = rep.exePath;
    root["exe_dir"] = rep.exeDir;

    {
        llvm::json::Object rel;
        rel["path"] = rep.releaseEnvPath;

        llvm::json::Object versions;
        for (const auto& [k, v] : rep.releaseVersions) {
            versions[k] = v;
        }

        rel["versions"] = std::move(versions);
        root["release"] = std::move(rel);
    }

    root["package_artifacts"] = toJsonArray(rep.packageArtifacts);

    {
        llvm::json::Object ark;
        ark["dir"] = rep.arknetDir;

        llvm::json::Object cfg;
        cfg["path"] = rep.configFile.path;
        cfg["exists"] = rep.configFile.exists;
        cfg["size_bytes"] = static_cast<std::int64_t>(rep.configFile.sizeBytes);
        cfg["keys"] = toJsonArray(rep.configKeys);
        ark["config"] = std::move(cfg);

        llvm::json::Object cred;
        cred["path"] = rep.credentialsFile.path;
        cred["exists"] = rep.credentialsFile.exists;
        cred["size_bytes"] = static_cast<std::int64_t>(rep.credentialsFile.sizeBytes);
        cred["algorithm"] = rep.credentialAlgorithm;
        cred["ciphertext_present"] = rep.credentialsCiphertextPresent;
        ark["credentials"] = std::move(cred);

        root["arknet_home"] = std::move(ark);
    }

    root["host_gpu"] = toJsonArray(rep.hostGpus);

    {
        llvm::json::Array arr;
        for (const auto& p : rep.gpuPlugins) {
            llvm::json::Object o;
            o["backend"] = p.backend;
            o["path"] = p.path;
            o["exists"] = p.exists;
            o["disabled_marker_found"] = p.disabledMarkerFound;
            o["status"] = p.status;
            arr.push_back(std::move(o));
        }
        root["gpu_plugins"] = std::move(arr);
    }

    root["warnings"] = toJsonArray(rep.warnings);
    root["issues"] = toJsonArray(rep.issues);

    llvm::outs() << llvm::json::Value(std::move(root)) << "\n";
}

} // namespace ark::cli