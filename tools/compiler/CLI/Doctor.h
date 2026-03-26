// tools/compiler/CLI/Doctor.h
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ark::cli {

struct DoctorOptions {
    bool verbose = false;
};

struct DoctorFileProbe {
    std::string path;
    bool exists = false;
    std::uint64_t sizeBytes = 0;
};

struct DoctorGpuPluginProbe {
    std::string backend;
    std::string path;
    bool exists = false;
    bool disabledMarkerFound = false;
    std::string status;
};

struct DoctorReport {
    bool ok = true;

    std::string osInfo;
    std::string llvmInfo;

    std::string exePath;
    std::string exeDir;

    std::string releaseEnvPath;
    std::map<std::string, std::string> releaseVersions;

    std::vector<std::string> packageArtifacts;

    std::string arknetDir;
    DoctorFileProbe configFile;
    DoctorFileProbe credentialsFile;
    std::vector<std::string> configKeys;
    std::string credentialAlgorithm;
    bool credentialsCiphertextPresent = false;

    std::vector<std::string> hostGpus;
    std::vector<DoctorGpuPluginProbe> gpuPlugins;

    std::vector<std::string> warnings;
    std::vector<std::string> issues;
};

DoctorReport collectDoctorReport(const DoctorOptions& opts);
void printDoctorReportPretty(const DoctorReport& rep);
void printDoctorReportJson(const DoctorReport& rep);

} // namespace ark::cli