#ifdef _WIN32
#include "ark/capsule/Platform.h"

#define NOMINMAX
#include <windows.h>
#include <string>
#include <vector>
#include <limits>
#include <iostream>

namespace ark::capsule::platform {

// -----------------------------------------------------------------------------
// String Helpers
// -----------------------------------------------------------------------------
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), nullptr, 0);
    if (n <= 0) return L"";
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), w.data(), n);
    return w;
}

static std::string wide_to_utf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) return "";
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

std::string get_self_exe_path() {
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    if (n == 0 || n >= buf.size()) return "";
    buf.resize(n);
    return wide_to_utf8(buf);
}

// -----------------------------------------------------------------------------
// Argument Quoting (Secure)
// -----------------------------------------------------------------------------
static std::wstring win_quote_arg(const std::wstring& a) {
    if (a.empty()) return L"\"\"";
    
    bool needQuotes = false;
    for (wchar_t c : a) {
        if (c == L' ' || c == L'\t' || c == L'\n' || c == L'\"') { 
            needQuotes = true; 
            break; 
        }
    }
    
    if (!needQuotes) return a;

    std::wstring out;
    out.reserve(a.size() + 2);
    out.push_back(L'"');
    
    size_t backslashes = 0;
    for (wchar_t c : a) {
        if (c == L'\\') {
            backslashes++;
            continue;
        }
        if (c == L'"') {
            out.append(backslashes * 2, L'\\');
            out.append(1, L'\\');
            out.push_back(L'"');
            backslashes = 0;
            continue;
        }
        if (backslashes) out.append(backslashes, L'\\');
        backslashes = 0;
        out.push_back(c);
    }
    
    if (backslashes) out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    
    return out;
}

static std::wstring create_temp_payload_path() {
    wchar_t tmpDir[MAX_PATH + 1];
    DWORD n = GetTempPathW(MAX_PATH, tmpDir);
    if (n == 0 || n > MAX_PATH) std::exit(1);

    wchar_t tmpFile[MAX_PATH + 1];
    if (GetTempFileNameW(tmpDir, L"ark", 0, tmpFile) == 0) std::exit(1);

    std::wstring path(tmpFile);
    auto dot = path.find_last_of(L'.');
    if (dot != std::wstring::npos) path.resize(dot);
    path += L".exe";
    
    // Robust rename with replace
    if (!MoveFileExW(tmpFile, path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileW(tmpFile);
        std::exit(1);
    }
    
    return path;
}

// -----------------------------------------------------------------------------
// Execution Logic
// -----------------------------------------------------------------------------
void exec_payload(std::span<const uint8_t> payload, int argc, char** argv, char** envp) {
    (void)envp; 

    // 1. Policy & Type Safety Cap
    if (payload.size() > 512ULL * 1024 * 1024) std::exit(1);
    if (payload.size() > static_cast<size_t>(std::numeric_limits<DWORD>::max())) std::exit(1);

    std::wstring exePath = create_temp_payload_path();

    // 2. Write Payload
    HANDLE h = CreateFileW(
        exePath.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY,
        nullptr
    );

    if (h == INVALID_HANDLE_VALUE) std::exit(1);

    DWORD written = 0;
    if (!WriteFile(h, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr) || 
        written != static_cast<DWORD>(payload.size())) {
        CloseHandle(h);
        DeleteFileW(exePath.c_str());
        std::exit(1);
    }
    FlushFileBuffers(h);
    CloseHandle(h);

    // 3. Build Command Line
    std::wstring cmd;
    cmd.reserve(1024);
    cmd += L"\"" + exePath + L"\"";

    for (int i = 1; i < argc; ++i) {
        cmd += L" ";
        std::string argUtf8 = argv[i] ? std::string(argv[i]) : "";
        cmd += win_quote_arg(utf8_to_wide(argUtf8));
    }
    
    std::vector<wchar_t> cmdMutable(cmd.begin(), cmd.end());
    cmdMutable.push_back(L'\0');

    // 4. Launch
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    if (!CreateProcessW(
            exePath.c_str(),
            cmdMutable.data(),
            nullptr,
            nullptr,
            FALSE,
            0,
            nullptr,
            nullptr,
            &si,
            &pi
        )) {
        DeleteFileW(exePath.c_str());
        std::exit(1);
    }

    // 5. Cleanup
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    DeleteFileW(exePath.c_str()); 
    if (GetFileAttributesW(exePath.c_str()) != INVALID_FILE_ATTRIBUTES) {
        MoveFileExW(exePath.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
    }
    
    std::exit(0);
}

} // namespace
#endif