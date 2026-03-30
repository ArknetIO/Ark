#include "ark/compiler/CLI/Common.hpp"

#include <exception>
#include <sstream>
#include <utility>

namespace ark::cli {

TomlParseOutcome parseTomlFilePortable(const std::string& path) noexcept {
    TomlParseOutcome out;

#if defined(TOML_EXCEPTIONS) && TOML_EXCEPTIONS
    try {
        out.value = toml::parse_file(path);
        return out;
    } catch (const toml::parse_error& err) {
        std::ostringstream oss;
        oss << err;
        out.error = oss.str();
        return out;
    } catch (const std::exception& err) {
        out.error = err.what();
        return out;
    }
#else
    auto parsed = toml::parse_file(path);

    if (!parsed) {
        std::ostringstream oss;
        oss << parsed.error();
        out.error = oss.str();
        return out;
    }

    out.value = std::move(parsed).table();
    return out;
#endif
}

} // namespace ark::cli