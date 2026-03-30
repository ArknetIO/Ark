#include "ark/compiler/CLI/Subcommands.hpp"

#include <CLI/CLI.hpp>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>

namespace ark::cli {

struct InitOptions {
    std::string projectName;
    bool isLibrary = false;
};

static InitOptions initOpts;

static void fail(const llvm::Twine& msg) {
    llvm::SmallString<256> buf;
    msg.toVector(buf);
    llvm::errs() << "[ERROR] " << buf << "\n";
    std::exit(1);
}

static void note(const llvm::Twine& msg) {
    llvm::SmallString<256> buf;
    msg.toVector(buf);
    llvm::outs() << buf << "\n";
}

static bool isValidProjectName(llvm::StringRef name) {
    if (name.empty()) return false;
    if (name == "." || name == "..") return false;

    for (char c : name) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_' || c == '-') continue;
        return false;
    }
    return true;
}

static std::string joinPath(llvm::StringRef a, llvm::StringRef b) {
    llvm::SmallString<256> p(a);
    llvm::sys::path::append(p, b);
    return std::string(p.str());
}

static void createDirOrFail(llvm::StringRef path) {
    std::error_code ec = llvm::sys::fs::create_directories(path);
    if (ec) fail(llvm::Twine("Failed to create directory '") + path + "': " + ec.message());
}

static void writeFileOrFail(llvm::StringRef path, llvm::StringRef content) {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
    if (ec) fail(llvm::Twine("Failed to write file '") + path + "': " + ec.message());
    os << content;
    os.flush();
}

static std::string makeManifest(const std::string& name, bool isLibrary) {
    std::string s;
    s += "[package]\n";
    s += "name = \"" + name + "\"\n";
    s += "version = \"0.1.0\"\n";
    s += "description = \"\"\n";
    s += "type = \"" + std::string(isLibrary ? "lib" : "bin") + "\"\n\n";
    s += "[build]\n";
    s += "entry = \"" + std::string(isLibrary ? "src/lib.ark" : "src/main.ark") + "\"\n\n";
    s += "[dependencies]\n";
    s += "# stdlib = { git = \"https://github.com/arklang/stdlib.git\", version = \"main\" }\n";
    return s;
}

static std::string makeMainArk(const std::string& name) {
    std::string s;
    s += "// " + name + " Executable\n\n";
    s += "fn[host] main() -> i32 !IO {\n";
    s += "    print \"Hello, Ark!\";\n";
    s += "    return 0;\n";
    s += "}\n";
    return s;
}

static std::string makeLibArk(const std::string& name) {
    std::string s;
    s += "// " + name + " Library\n\n";
    s += "pub fn[host] add(a: i32, b: i32) -> i32 {\n";
    s += "    return a + b;\n";
    s += "}\n";
    return s;
}

static std::string makeTestArk(const std::string& name, bool isLibrary) {
    std::string s;
    s += "// " + name + " tests\n\n";
    s += "fn[host] main() -> i32 !IO {\n";
    if (isLibrary) {
        s += "    print \"Library test scaffold\";\n";
    } else {
        s += "    print \"Executable test scaffold\";\n";
    }
    s += "    return 0;\n";
    s += "}\n";
    return s;
}

static std::string makeGitignore() {
    return
        "build/\n"
        ".ark/\n"
        "*.o\n"
        "*.ll\n"
        "*.mlir\n"
        "*.llvm.mlir\n"
        "*.raw\n"
        "*.tmp\n";
}

static void tryInitGit(llvm::StringRef projectDir) {
    auto git = llvm::sys::findProgramByName("git");
    if (!git) {
        llvm::outs() << "[WARN] 'git' not found. Skipping repository initialization.\n";
        return;
    }

    llvm::SmallVector<llvm::StringRef, 8> argv;
    argv.push_back(git->c_str());
    argv.push_back("-C");
    argv.push_back(projectDir);
    argv.push_back("init");
    argv.push_back("-q");

    std::string errMsg;
    std::optional<llvm::StringRef> redirects[] = {std::nullopt, std::nullopt, std::nullopt};

    int rc = llvm::sys::ExecuteAndWait(*git, argv, std::nullopt, redirects, 0, 0, &errMsg);
    if (rc != 0) {
        llvm::outs() << "[WARN] git init failed";
        if (!errMsg.empty()) llvm::outs() << ": " << errMsg;
        llvm::outs() << "\n";
    }
}

static void handleInit() {
    if (!isValidProjectName(initOpts.projectName)) {
        fail(llvm::Twine("Invalid project name '") + initOpts.projectName +
             "'. Use only letters, digits, '_' or '-'.");
    }

    const std::string root = initOpts.projectName;
    note(llvm::Twine("Initializing new Ark project: ") + root);

    if (llvm::sys::fs::exists(root)) {
        fail(llvm::Twine("Directory '") + root + "' already exists");
    }

    createDirOrFail(root);
    createDirOrFail(joinPath(root, "src"));
    createDirOrFail(joinPath(root, "tests"));
    createDirOrFail(joinPath(root, "build"));

    writeFileOrFail(joinPath(root, "ark.toml"), makeManifest(root, initOpts.isLibrary));
    writeFileOrFail(joinPath(root, ".gitignore"), makeGitignore());

    if (initOpts.isLibrary) {
        note("Scaffolding as a library project...");
        writeFileOrFail(joinPath(root, "src/lib.ark"), makeLibArk(root));
        writeFileOrFail(joinPath(root, std::string("tests/") + root + "_test.ark"), makeTestArk(root, true));
    } else {
        note("Scaffolding as an executable project...");
        writeFileOrFail(joinPath(root, "src/main.ark"), makeMainArk(root));
        writeFileOrFail(joinPath(root, "tests/smoke.ark"), makeTestArk(root, false));
    }

    tryInitGit(root);

    llvm::outs() << "\n[SUCCESS] Created " << (initOpts.isLibrary ? "library" : "binary")
                 << " `" << root << "` package.\n";
    llvm::outs() << "Get started:\n";
    llvm::outs() << "  cd " << root << "\n";
    if (!initOpts.isLibrary) llvm::outs() << "  arknet run\n";
    else llvm::outs() << "  arknet compile src/lib.ark\n";
}

void setupInitCmd(CLI::App& app) {
    auto* sub = app.add_subcommand("init", "Initialize a new Ark project");

    sub->add_option("name", initOpts.projectName, "The name of the new project")->required();
    sub->add_flag("--lib", initOpts.isLibrary, "Use a library template instead of an executable");

    sub->callback([]() { handleInit(); });
}

} // namespace ark::cli