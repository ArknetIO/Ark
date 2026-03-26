#pragma once
#include <cstdint>
#include <string>
#include <span>

namespace ark::capsule::platform {

// -----------------------------------------------------------------------------
// Core Abstractions
// -----------------------------------------------------------------------------

// Robustly finds the absolute path to the currently running executable.
// - Linux:   /proc/self/exe
// - macOS:   _NSGetExecutablePath + realpath
// - Windows: GetModuleFileNameW
std::string get_self_exe_path();

// Executes the given payload bytes as a new process image.
// On success, this function DOES NOT RETURN.
//
// Strategies:
// - Linux:   memfd_create + seals + fexecve (Memory-only, Zero disk artifacts).
// - macOS:   mkstemp + unlink + fexecve (Transient artifact, deleted before exec).
// - Windows: temp .exe + CreateProcessW + DeleteOnReboot (Transient artifact).
[[noreturn]] void exec_payload(std::span<const uint8_t> payload, int argc, char** argv, char** envp);

} // namespace ark::capsule::platform