#include "ark/compiler/CLI/Subcommands.hpp"

#include <CLI/CLI.hpp>

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <string>

#ifndef ARK_VERSION_STRING
#define ARK_VERSION_STRING "dev"
#endif

namespace {

enum class VersionFormat {
    Text,
    Json
};

struct TopLevelFlags {
    bool showVersion = false;
    bool showVersionJson = false;
};

static void printVersion(VersionFormat format) {
    const std::string arkVersion = ARK_VERSION_STRING;
    const std::string llvmVersion = LLVM_VERSION_STRING;
    const std::string hostTriple = llvm::sys::getProcessTriple();

    if (format == VersionFormat::Json) {
        llvm::json::Object obj;
        obj["name"] = "arknet";
        obj["version"] = arkVersion;
        obj["engine"] = "LLVM/MLIR Backend Engine";
        obj["llvm_version"] = llvmVersion;
        obj["host_triple"] = hostTriple;
        llvm::outs() << llvm::json::Value(std::move(obj)) << '\n';
        return;
    }

    llvm::outs() << "Arknet v" << arkVersion << '\n';
    llvm::outs() << "LLVM/MLIR Backend Engine\n";
    llvm::outs() << "LLVM " << llvmVersion << '\n';
    llvm::outs() << hostTriple << '\n';
}

static void registerVersionCmd(CLI::App& app) {
    CLI::App* cmd = app.add_subcommand("version", "Print version information");

    bool asJson = false;
    cmd->add_flag("--json", asJson, "Print version information as JSON");

    cmd->callback([&asJson]() {
        printVersion(asJson ? VersionFormat::Json : VersionFormat::Text);
    });
}

static void registerTopLevelVersionFlags(CLI::App& app, TopLevelFlags& flags) {
    app.add_flag(
        "-V,--version",
        flags.showVersion,
        "Print version information and exit"
    );

    app.add_flag(
        "--version-json",
        flags.showVersionJson,
        "Print version information as JSON and exit"
    );
}

static void registerSubcommands(CLI::App& app) {
    registerVersionCmd(app);

    ark::cli::setupInitCmd(app);
    ark::cli::setupCompileCmd(app);
    ark::cli::setupRunCmd(app);
    ark::cli::setupConfigCmd(app);
    ark::cli::setupLspCmd(app);
    ark::cli::setupAddCmd(app);
    ark::cli::setupFetchCmd(app);
    ark::cli::setupProviderCmd(app);
}

static void configureApp(CLI::App& app, TopLevelFlags& flags) {
    app.set_help_flag("-h,--help", "Show help");
    app.set_help_all_flag("--help-all", "Show help (including hidden options)");
    app.failure_message(CLI::FailureMessage::help);

    registerTopLevelVersionFlags(app, flags);
    registerSubcommands(app);
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init(argc, argv);

    CLI::App app{"The Arknet Project"};
    TopLevelFlags flags;
    configureApp(app, flags);

    if (argc <= 1) {
        llvm::outs() << app.help();
        return 0;
    }

    CLI11_PARSE(app, argc, argv);

    if (flags.showVersionJson) {
        printVersion(VersionFormat::Json);
        return 0;
    }

    if (flags.showVersion) {
        printVersion(VersionFormat::Text);
        return 0;
    }

    return 0;
}