#include <ark_protocol.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

// Helper: Wrap a std::string into an ArkStr (allocated via ark_alloc)
static ArkStr copy_to_ark(const std::string& source) {
    int64_t len = (int64_t)source.length();
    // Allocate len + 1 for null terminator safety
    char* buffer = (char*)ark_alloc(len + 1);
    memcpy(buffer, source.c_str(), len);
    buffer[len] = '\0'; // Null-terminate just in case
    return { buffer, len };
}

// Helper: View ArkStr as std::string (Zero-copy if possible, but std::string copies)
static std::string view(ArkStr s) {
    if (!s.ptr || s.len == 0) return "";
    return std::string(s.ptr, s.len);
}

extern "C" {

    // =========================================================
    // Constructors
    // =========================================================

    ArkStr __ark_str_alloc(int64_t size) {
        char* ptr = (char*)ark_alloc(size + 1);
        memset(ptr, 0, size + 1);
        return { ptr, size };
    }

    ArkStr __ark_str_from_raw(const char* s) {
        if (!s) return { nullptr, 0 };
        int64_t len = (int64_t)strlen(s);
        // We do NOT copy here? If this is for literals, we might not need to.
        // But for safety/ownership uniformity, let's copy to heap.
        char* ptr = (char*)ark_alloc(len + 1);
        memcpy(ptr, s, len + 1);
        return { ptr, len };
    }

    // =========================================================
    // Core Ops
    // =========================================================

    ArkStr __ark_str_concat(ArkStr s1, ArkStr s2) {
        // Direct buffer manipulation for speed
        int64_t new_len = s1.len + s2.len;
        char* ptr = (char*)ark_alloc(new_len + 1);
        
        if (s1.ptr && s1.len > 0) memcpy(ptr, s1.ptr, s1.len);
        if (s2.ptr && s2.len > 0) memcpy(ptr + s1.len, s2.ptr, s2.len);
        
        ptr[new_len] = '\0';
        return { ptr, new_len };
    }

    int64_t __ark_str_len(ArkStr s) {
        return s.len;
    }

    bool __ark_str_eq(ArkStr s1, ArkStr s2) {
        if (s1.len != s2.len) return false;
        if (s1.ptr == s2.ptr) return true;
        if (s1.len == 0) return true; // Both empty
        return memcmp(s1.ptr, s2.ptr, s1.len) == 0;
    }

    // =========================================================
    // Predicates
    // =========================================================

    bool __ark_str_contains(ArkStr haystack, ArkStr needle) {
        std::string h = view(haystack);
        std::string n = view(needle);
        return h.find(n) != std::string::npos;
    }

    bool __ark_str_startswith(ArkStr str, ArkStr prefix) {
        if (prefix.len > str.len) return false;
        return memcmp(str.ptr, prefix.ptr, prefix.len) == 0;
    }

    bool __ark_str_endswith(ArkStr str, ArkStr suffix) {
        if (suffix.len > str.len) return false;
        char* start = str.ptr + (str.len - suffix.len);
        return memcmp(start, suffix.ptr, suffix.len) == 0;
    }

    // =========================================================
    // Transformations
    // =========================================================

    ArkStr __ark_str_trim(ArkStr str) {
        if (!str.ptr || str.len == 0) return __ark_str_alloc(0);
        std::string s = view(str);
        
        // Trim Left
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        
        // Trim Right
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());

        return copy_to_ark(s);
    }

    ArkStr __ark_str_to_lower(ArkStr str) {
        std::string s = view(str);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return copy_to_ark(s);
    }

    ArkStr __ark_str_to_upper(ArkStr str) {
        std::string s = view(str);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::toupper(c); });
        return copy_to_ark(s);
    }

    ArkStr __ark_str_reverse(ArkStr str) {
        std::string s = view(str);
        std::reverse(s.begin(), s.end());
        return copy_to_ark(s);
    }

    ArkStr __ark_str_repeat(ArkStr str, int64_t count) {
        if (count <= 0) return __ark_str_alloc(0);
        std::string s = view(str);
        std::ostringstream os;
        for (int i = 0; i < count; i++) os << s;
        return copy_to_ark(os.str());
    }

    // =========================================================
    // Slicing & Searching
    // =========================================================

    ArkStr __ark_str_slice(ArkStr str, int64_t start, int64_t end) {
        int64_t len = str.len;
        if (start < 0) start = len + start;
        if (end < 0) end = len + end;
        if (start < 0) start = 0;
        if (end > len) end = len;
        if (start >= end) return __ark_str_alloc(0);

        int64_t new_len = end - start;
        char* ptr = (char*)ark_alloc(new_len + 1);
        memcpy(ptr, str.ptr + start, new_len);
        ptr[new_len] = '\0';
        return { ptr, new_len };
    }

    ArkStr __ark_str_replace(ArkStr str, ArkStr target, ArkStr replacement) {
        std::string s = view(str);
        std::string t = view(target);
        std::string r = view(replacement);
        if (t.empty()) return copy_to_ark(s);

        size_t pos = 0;
        while ((pos = s.find(t, pos)) != std::string::npos) {
            s.replace(pos, t.length(), r);
            pos += r.length();
        }
        return copy_to_ark(s);
    }

    int64_t __ark_str_index_of(ArkStr str, ArkStr target) {
        std::string s = view(str);
        size_t pos = s.find(view(target));
        return (pos == std::string::npos) ? -1 : (int64_t)pos;
    }

    int64_t __ark_str_last_index_of(ArkStr str, ArkStr target) {
        std::string s = view(str);
        size_t pos = s.rfind(view(target));
        return (pos == std::string::npos) ? -1 : (int64_t)pos;
    }

    // =========================================================
    // Regex
    // =========================================================

    ArkStatus __ark_str_regex_match(ArkStr str, ArkStr pattern, bool* out_match) {
        if (!out_match) return -1;

        const std::string input = view(str);
        const std::string pat = view(pattern);

        std::regex re(pat);
        *out_match = std::regex_match(input, re);
        return 0;
    }

    ArkStatus __ark_str_regex_replace(ArkStr str, ArkStr pattern, ArkStr replacement, ArkStr* out_res) {
        if (!out_res) return -1;

        const std::string input = view(str);
        const std::string pat = view(pattern);
        const std::string repl = view(replacement);

        std::regex re(pat);
        const std::string result = std::regex_replace(input, re, repl);
        *out_res = copy_to_ark(result);
        return 0;
    }

} // extern "C"