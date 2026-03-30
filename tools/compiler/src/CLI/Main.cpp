#include "ark/compiler/CLI/Subcommands.hpp"

#include <CLI/CLI.hpp>

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <exception>
#include <string>

#ifndef ARK_VERSION_STRING
#define ARK_VERSION_STRING "dev"
#endif

namespace {

enum class VersionFormat {
    Text,
    Json
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

    CLI::Option* jsonOpt = cmd->add_flag(
        "--json",
        "Print version information as JSON"
    );

    cmd->callback([jsonOpt]() {
        const bool asJson = jsonOpt != nullptr && jsonOpt->count() > 0;
        printVersion(asJson ? VersionFormat::Json : VersionFormat::Text);
    });
}

static void registerTopLevelVersionFlags(CLI::App& app) {
    app.add_flag_callback(
        "-V,--version",
        []() {
            printVersion(VersionFormat::Text);
            throw CLI::Success();
        },
        "Print version information and exit"
    );

    app.add_flag_callback(
        "--version-json",
        []() {
            printVersion(VersionFormat::Json);
            throw CLI::Success();
        },
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

static void configureApp(CLI::App& app) {
    app.set_help_flag("-h,--help", "Show help");
    app.set_help_all_flag("--help-all", "Show help (including hidden options)");
    app.failure_message(CLI::FailureMessage::help);

    registerTopLevelVersionFlags(app);
    registerSubcommands(app);
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init(argc, argv);

    CLI::App app{"The Arknet Project"};
    configureApp(app);

    if (argc <= 1) {
        llvm::outs() << app.help();
        return 0;
    }

    try {
        app.parse(argc, argv);
        return 0;
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    } catch (const std::exception& e) {
        llvm::errs() << "arknet: fatal error: " << e.what() << '\n';
        return 1;
    } catch (...) {
        llvm::errs() << "arknet: fatal error: unknown exception\n";
        return 1;
    }
}