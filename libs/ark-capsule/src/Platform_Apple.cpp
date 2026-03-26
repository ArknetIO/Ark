#if defined(__APPLE__) && !defined(_WIN32)
#include "ark/capsule/Platform.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mach-o/dyld.h>

extern char **environ;

namespace ark::capsule::platform {

static void die(const char* msg) {
    std::perror(msg);
    std::exit(1);
}

std::string get_self_exe_path() {
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    if (sz == 0) return {};

    std::vector<char> tmp(sz + 1, '\0');
    if (_NSGetExecutablePath(tmp.data(), &sz) != 0) return {};

    char resolved[4096];
    if (!::realpath(tmp.data(), resolved)) return {};
    return std::string(resolved);
}

[[noreturn]] void exec_payload(std::span<const uint8_t> payload, int argc, char** argv) {
    (void)argc;

    char tmpl[] = "/tmp/ark_exec_XXXXXX";
    const int fd = ::mkstemp(tmpl);
    if (fd == -1) die("mkstemp");

    if (::fchmod(fd, 0700) == -1) die("fchmod");

    size_t off = 0;
    while (off < payload.size()) {
        const ssize_t n = ::write(fd, payload.data() + off, payload.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            ::unlink(tmpl);
            die("write");
        }
        off += static_cast<size_t>(n);
    }

    ::fsync(fd);
    ::unlink(tmpl);

    if (::lseek(fd, 0, SEEK_SET) == -1) die("lseek");

    ::fexecve(fd, argv, environ);
    die("fexecve");
}

} // namespace
#endif