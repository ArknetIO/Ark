#include "ark/compiler/CLI/Subcommands.hpp"
#include <CLI/CLI.hpp>
#include <llvm/Support/raw_ostream.h>

namespace ark::cli {

struct ProviderOptions {
    bool start = false;
    bool stop = false;
    int port = 9000;
};

static ProviderOptions provOpts;

static void handleProvider() {
    if (provOpts.start) {
        llvm::outs() << "Starting Ark local provider sandbox on port " << provOpts.port << "...\n";
        // TODO: Boot up tools/provider/supervisor/Supervisor
    } else if (provOpts.stop) {
        llvm::outs() << "Stopping Ark local provider sandbox...\n";
        // TODO: Send kill signal to local supervisor daemon
    }
}

void setupProviderCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("provider", "Manage the Ark local provider sandbox");
    
    // We can use mutually exclusive flags for start/stop
    auto* startFlag = sub->add_flag("--start", provOpts.start, "Start the local provider daemon");
    auto* stopFlag = sub->add_flag("--stop", provOpts.stop, "Stop the local provider daemon");
    startFlag->excludes(stopFlag);
    stopFlag->excludes(startFlag);

    sub->add_option("-p,--port", provOpts.port, "Port for the provider to listen on (default: 9000)")
       ->needs(startFlag); // Only valid if --start is used

    sub->callback([]() { handleProvider(); });
}

} // namespace ark::cli