#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // Required for memfd_create
#endif
#include "ark/capsule/Platform.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <span>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// -----------------------------------------------------------------------------
// Compatibility Polyfills (Older Kernels/Headers)
// -----------------------------------------------------------------------------
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif

// Some older headers might miss these seals even if the kernel supports them
#ifndef F_ADD_SEALS
#define F_ADD_SEALS 1033
#endif
#ifndef F_SEAL_SEAL
#define F_SEAL_SEAL 0x0001
#endif
#ifndef F_SEAL_SHRINK
#define F_SEAL_SHRINK 0x0002
#endif
#ifndef F_SEAL_GROW
#define F_SEAL_GROW 0x0004
#endif
#ifndef F_SEAL_WRITE
#define F_SEAL_WRITE 0x0008
#endif

namespace ark::capsule::platform {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
[[noreturn]] static void die(const char* msg) {
    std::perror(msg);
    std::exit(1);
}

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

std::string get_self_exe_path() {
    char buf[4096];
    // Read the symbolic link /proc/self/exe to find our own path
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf));
    if (n <= 0) return {};
    return std::string(buf, static_cast<size_t>(n));
}

[[noreturn]] 
void exec_payload(std::span<const uint8_t> payload, int argc, char** argv, char** envp) {
    (void)argc; // Unused, argv is null-terminated

    // 1. Create Memory-Backed File Descriptor (Anonymous RAM)
    // "ark_payload" is just a label for debugging (e.g. /proc/self/fd/N)
    int fd = ::memfd_create("ark_payload", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd == -1) die("memfd_create");

    // 2. Write Payload to Memory
    size_t off = 0;
    while (off < payload.size()) {
        const ssize_t n = ::write(fd, payload.data() + off, payload.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            die("write");
        }
        off += static_cast<size_t>(n);
    }

    // 3. Seal the File
    // This makes the file immutable. Most modern security policies (and logic)
    // require executable memory mappings to be immutable (W^X).
    // F_SEAL_WRITE:  Prevent writing
    // F_SEAL_SHRINK: Prevent truncation
    // F_SEAL_GROW:   Prevent expansion
    // F_SEAL_SEAL:   Prevent changing seals
    if (::fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) == -1) {
        die("fcntl(F_ADD_SEALS)");
    }

    // 4. Reset Cursor
    // fexecve executes from the current position if it were a script, 
    // but for ELF binaries strictly it might not matter, yet good practice.
    if (::lseek(fd, 0, SEEK_SET) == -1) die("lseek");

    // 5. Execute
    // Replaces the current process image with the content of the memfd.
    // We pass the provided 'envp' explicitely.
    ::fexecve(fd, argv, envp);
    
    // 6. Failure
    // fexecve only returns on error.
    die("fexecve");
}

} // namespace ark::capsule::platform

#endif // __linux__