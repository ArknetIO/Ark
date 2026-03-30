// tools/compiler/CLI/Subcommands.h
#pragma once

// Forward declare to avoid including CLI11.hpp in headers everywhere
namespace CLI { class App; }

namespace ark::cli {

void setupInitCmd(CLI::App& app);
void setupCompileCmd(CLI::App& app);
void setupConfigCmd(CLI::App& app);
void setupRunCmd(CLI::App& app);
void setupRegistryCmd(CLI::App& app);
void setupLspCmd(CLI::App& app);
void setupProviderCmd(CLI::App& app);
void setupAddCmd(CLI::App& app);
void setupFetchCmd(CLI::App& app);
} // namespace ark::cli