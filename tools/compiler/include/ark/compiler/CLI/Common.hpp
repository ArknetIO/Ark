#pragma once

#include <optional>
#include <string>

#include <toml++/toml.hpp>

namespace ark::cli {

struct TomlParseOutcome {
    std::optional<toml::table> value;
    std::string error;

    explicit operator bool() const noexcept {
        return value.has_value();
    }

    toml::table take() {
        return std::move(*value);
    }
};

[[nodiscard]] TomlParseOutcome parseTomlFilePortable(const std::string& path) noexcept;

} // namespace ark::cli