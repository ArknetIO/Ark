// tools/arkc/arkc.cpp

// -----------------------------------------------------------------------------
// Top-of-file fixes
// -----------------------------------------------------------------------------

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#ifndef ARK_VERSION_STRING
  #if defined(ARK_VERSION_STR)
    #define ARK_VERSION_STRING ARK_VERSION_STR
  #elif defined(ARK_VERSION)
    #define ARK_VERSION_STRING ARK_VERSION
  #else
    #define ARK_VERSION_STRING "dev"
  #endif
#endif

#include "Frontend/AST.h"
#include "Frontend/Lexer.h"
#include "Frontend/Parser.h"
#include "Frontend/ModuleRegistry.h" // [NEW] Add this

#include "Frontend/GenMIR.h"
#include "Analysis/OwnershipVerifier.h"
#include "Transforms/Passes.h"
#include "ark/Conversion/ArkToLLVM/ArkToLLVM.h"
#include "ark/IR/ArkMirOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <atomic>
#include <csetjmp>
#include <csignal>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "ark_config.h"

// HUD
#include "hud.h"
#include "hud_mlir.h"

using namespace arklang;
namespace cl = llvm::cl;

namespace arklang {
void populateGpuLoweringPatterns(mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);
std::unique_ptr<mlir::Pass> createGpuKernelOutliningPass();
} // namespace arklang

// -----------------------------------------------------------------------------
// Runtime Configuration
// -----------------------------------------------------------------------------

static const std::vector<std::string> kRuntimeSources = {
    "core/async.cpp",
    "core/memory.cpp",
    "core/string.cpp",
    "core/panic.cpp",
    "core/print.cpp",
    "core/vector.cpp",
    "core/gpu_hal.cpp",
    "fs/fs_posix.cpp",
    "net/remote.cpp",
    "net/socket_posix.cpp",
};

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

static cl::opt<std::string> inputFilename(
    cl::Positional, cl::desc("<input .ark file>"), cl::init("-"));

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output file"), cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool> buildBin("build", cl::desc("Build a native executable"));
static cl::opt<bool> runBin("run", cl::desc("Build and run the program"));
static cl::opt<bool> useJit("jit", cl::desc("Run code using in-memory JIT"));

static cl::opt<std::string> runtimePath(
    "runtime", cl::desc("Path to Ark Runtime directory"), cl::init("tools/compiler/Runtime"));

static cl::opt<std::string> llvmBinDir(
    "llvm-bin", cl::desc("Directory containing mlir-opt/clang"), cl::value_desc("path"), cl::init(""));

static cl::opt<std::string> mlirOptOverride("mlir-opt", cl::desc("Override mlir-opt path"), cl::init(""));
static cl::opt<std::string> mlirTranslateOverride("mlir-translate", cl::desc("Override mlir-translate path"), cl::init(""));
static cl::opt<std::string> clangOverride("clang", cl::desc("Override clang path"), cl::init(""));

static cl::opt<bool> keepTmp("keep-tmp", cl::desc("Keep temporary build directory"), cl::init(false));

static cl::opt<bool> infoMode(
    "info",
    cl::desc("Show full compiler timeline (banner, timings, hints, step history)"),
    cl::init(false));

static llvm::cl::opt<bool> compilerDebug(
    "compiler-debug",
    llvm::cl::desc("Enable verbose compiler driver logging"),
    llvm::cl::init(false));

// -----------------------------------------------------------------------------
// Targets
// -----------------------------------------------------------------------------

static void initializeCompilerTargets() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
}

// -----------------------------------------------------------------------------
// Helpers: Expected + Tools
// -----------------------------------------------------------------------------

static std::string defaultLlvmBinDirIfPresent() {
  if (!llvmBinDir.empty()) return llvmBinDir;
  auto home = llvm::sys::Process::GetEnv("HOME");
  if (!home) return "";
  llvm::SmallString<256> p(*home);
  llvm::sys::path::append(p, "llvm-install", "bin");
  if (llvm::sys::fs::exists(p)) return std::string(p.str());
  return "";
}

static llvm::Expected<std::string> findTool(llvm::StringRef toolName,
                                            llvm::StringRef overridePath,
                                            llvm::StringRef llvmBin) {
  if (!overridePath.empty()) {
    if (llvm::sys::fs::exists(overridePath)) return overridePath.str();
    return llvm::createStringError(std::make_error_code(std::errc::no_such_file_or_directory),
                                   "tool not found: %s", overridePath.str().c_str());
  }

  if (!llvmBin.empty()) {
    llvm::SmallString<256> p(llvmBin);
    llvm::sys::path::append(p, toolName);
    if (llvm::sys::fs::exists(p)) return std::string(p.str());
  }

  if (auto found = llvm::sys::findProgramByName(toolName)) return *found;

  return llvm::createStringError(std::make_error_code(std::errc::no_such_file_or_directory),
                                 "tool not found in PATH: %s", toolName.str().c_str());
}

// Consumes Expected<> and reports error via HUD (prevents unchecked Expected<> aborts).
static std::optional<std::string> consumeToolOrReport(arklang::hud::Hud &hud,
                                                      llvm::Expected<std::string> ex,
                                                      llvm::StringRef label) {
  if (!ex) {
    std::string msg = llvm::toString(ex.takeError());
    hud.error((label + (": " + msg)).str());
    return std::nullopt;
  }
  return std::move(*ex);
}

// -----------------------------------------------------------------------------
// Helpers: runCmd that captures child stdout/stderr (spinner-safe HUD printing)
// -----------------------------------------------------------------------------



static std::string joinArgsForDiag(llvm::StringRef exe, const std::vector<std::string> &args) {
  std::string s;
  s.reserve(256);
  s.append(exe.data(), exe.size());
  for (const auto &a : args) {
    s.push_back(' ');
    const bool q = a.find_first_of(" \t") != std::string::npos;
    if (q) s.push_back('"');
    s.append(a);
    if (q) s.push_back('"');
  }
  return s;
}

static bool looksLikeWarning(std::string_view out) {
  return out.find("warning:") != std::string_view::npos ||
         out.find("Warning:") != std::string_view::npos ||
         out.find("WARNING:") != std::string_view::npos;
}

static std::optional<std::string> readFileToString(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) return std::nullopt;
  return std::string(buf.get()->getBuffer());
}



static int runCmd(arklang::hud::Hud* hud,
                  llvm::StringRef exe,
                  const std::vector<std::string>& args,
                  llvm::StringRef label = {}) {
  llvm::SmallVector<llvm::StringRef, 32> argv;
  argv.reserve(args.size() + 1);
  argv.push_back(exe);
  for (const auto& a : args) argv.push_back(a);

  llvm::SmallString<256> outPath;
  llvm::SmallString<256> errPath;

  {
    std::error_code ec = llvm::sys::fs::createTemporaryFile("arkc", "stdout", outPath);
    if (ec) {
      if (hud) hud->error(std::string("createTemporaryFile(stdout) failed: ") + ec.message());
      return 1;
    }
  }
  {
    std::error_code ec = llvm::sys::fs::createTemporaryFile("arkc", "stderr", errPath);
    if (ec) {
      if (hud) hud->error(std::string("createTemporaryFile(stderr) failed: ") + ec.message());
      (void)llvm::sys::fs::remove(outPath);
      return 1;
    }
  }

  const llvm::StringRef stdinPath = "/dev/null";
  const llvm::StringRef outRef(outPath);
  const llvm::StringRef errRef(errPath);

  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects;
  redirects.push_back(stdinPath);
  redirects.push_back(outRef);
  redirects.push_back(errRef);

  std::string execErrMsg;
  const int rc = llvm::sys::ExecuteAndWait(
      exe, argv, std::nullopt, llvm::ArrayRef<std::optional<llvm::StringRef>>(redirects),
      /*secondsToWait=*/0, /*memoryLimit=*/0, &execErrMsg, nullptr, nullptr, nullptr);

  auto readFile = [](llvm::StringRef p) -> std::string {
    auto mb = llvm::MemoryBuffer::getFile(p);
    if (!mb) return {};
    return std::string((*mb)->getBuffer());
  };

  const std::string stdoutText = readFile(outRef);
  const std::string stderrText = readFile(errRef);

  (void)llvm::sys::fs::remove(outPath);
  (void)llvm::sys::fs::remove(errPath);

  if (!hud) {
    if (!stdoutText.empty()) llvm::outs() << stdoutText;
    if (!stderrText.empty()) llvm::errs() << stderrText;
    if (!execErrMsg.empty()) llvm::errs() << execErrMsg << "\n";
    return rc;
  }

  const std::string base =
      label.empty() ? llvm::sys::path::filename(exe).str() : label.str();

  if (!stdoutText.empty()) hud->pushLogBlock(base + " stdout", stdoutText);
  if (!stderrText.empty()) hud->pushLogBlock(base + " stderr", stderrText);
  if (!execErrMsg.empty()) hud->pushLogBlock(base + " exec", execErrMsg);

  return rc;
}


static llvm::Expected<std::string> makeTempDir() {
  llvm::SmallString<256> base;
  llvm::sys::path::system_temp_directory(true, base);
  llvm::sys::path::append(base, "arkc");
  (void)llvm::sys::fs::create_directories(base);

  llvm::SmallString<256> outPath;
  llvm::SmallString<256> prefix = base;
  llvm::sys::path::append(prefix, "run");

  if (std::error_code ec = llvm::sys::fs::createUniqueDirectory(prefix, outPath))
    return llvm::createStringError(ec, "createUniqueDirectory failed");

  return std::string(outPath.str());
}

// -----------------------------------------------------------------------------
// Helper: Discovery
// -----------------------------------------------------------------------------

// Robustly finds the CUDA library path on any system.
// Priority:
// 1. CUDA_PATH/CUDA_HOME env vars
// 2. Relative to 'nvcc' in PATH
// 3. Standard locations (/usr/local/cuda)
static std::string findCudaLibPath() {
    // 1. Env Vars
    if (const char* env = std::getenv("CUDA_PATH")) {
        llvm::SmallString<256> p(env);
        llvm::sys::path::append(p, "lib64");
        if (llvm::sys::fs::exists(p)) return std::string(p.str());
    }
    if (const char* env = std::getenv("CUDA_HOME")) {
        llvm::SmallString<256> p(env);
        llvm::sys::path::append(p, "lib64");
        if (llvm::sys::fs::exists(p)) return std::string(p.str());
    }

    // 2. Derive from nvcc location
    if (auto nvccPath = llvm::sys::findProgramByName("nvcc")) {
        // e.g., /usr/local/cuda-13.0/bin/nvcc -> /usr/local/cuda-13.0/lib64
        llvm::SmallString<256> p = llvm::sys::path::parent_path(*nvccPath); // bin
        llvm::sys::path::append(p, "..", "lib64"); // ../lib64
        if (llvm::sys::fs::exists(p)) return std::string(p.str());
    }

    // 3. Fallback standard locations
    const char* standards[] = {
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu", // Linux package managers often put it here
        "/opt/cuda/lib64"
    };

    for (const char* path : standards) {
        if (llvm::sys::fs::exists(path)) return std::string(path);
    }

    return ""; // Not found
}

// -----------------------------------------------------------------------------
// Helper: Linker selection (C++ runtime)
// -----------------------------------------------------------------------------

static bool runtimeNeedsCxxLink() {
  for (const auto &src : kRuntimeSources) {
    llvm::StringRef s(src);
    if (s.ends_with(".cpp") || s.ends_with(".cc") || s.ends_with(".cxx")) return true;
  }
  return false;
}

static std::optional<std::string> trySiblingClangXX(llvm::StringRef clangPath) {
  llvm::SmallString<256> p(clangPath);
  llvm::StringRef file = llvm::sys::path::filename(p);

  auto repl = [&](llvm::StringRef from, llvm::StringRef to) -> std::optional<std::string> {
    if (!file.contains(from)) return std::nullopt;
    llvm::SmallString<256> q = llvm::sys::path::parent_path(p);
    llvm::SmallString<256> f(file);
    f = llvm::StringRef(f).str();
    std::string fs = f.str().str();
    size_t pos = fs.find(from.str());
    if (pos == std::string::npos) return std::nullopt;
    fs.replace(pos, from.size(), to.str());
    llvm::sys::path::append(q, fs);
    if (llvm::sys::fs::exists(q)) return std::string(q.str());
    return std::nullopt;
  };

  if (auto r = repl("clang", "clang++")) return r;
  if (auto r = repl("clang-", "clang++-")) return r;
  return std::nullopt;
}

static llvm::StringRef exeStem(llvm::StringRef p) { return llvm::sys::path::filename(p); }

// Prefer clang++ for link if runtime is C++.
static std::optional<std::string> chooseLinker(arklang::hud::Hud &hud,
                                               llvm::StringRef llvmBin,
                                               llvm::StringRef clangExe,
                                               llvm::StringRef clangOverridePath) {
  if (!runtimeNeedsCxxLink()) return std::string(clangExe);

  llvm::StringRef stem = exeStem(clangExe);
  if (stem.contains("clang++") || stem.contains("g++") || stem.contains("c++"))
    return std::string(clangExe);

  if (!clangOverridePath.empty()) {
    if (auto sib = trySiblingClangXX(clangOverridePath)) return sib.value();
  }

  auto cxx = findTool("clang++", /*overridePath=*/"", llvmBin);
  if (!cxx) {
    hud.warn("C++ runtime needed but clang++ not found; falling back to clang + libstdc++");
    return std::string(clangExe);
  }
  return std::move(*cxx);
}

// -----------------------------------------------------------------------------
// Helper: CUDA runtime presence
// -----------------------------------------------------------------------------

static bool hasCudaRuntimeIn(llvm::StringRef libDir) {
  if (libDir.empty()) return false;
  llvm::SmallString<256> so(libDir);
  llvm::sys::path::append(so, "libcudart.so");
  if (llvm::sys::fs::exists(so)) return true;

  llvm::SmallString<256> sover(libDir);
  llvm::sys::path::append(sover, "libcudart.so.0");
  if (llvm::sys::fs::exists(sover)) return true;

  llvm::SmallString<256> a(libDir);
  llvm::sys::path::append(a, "libcudart_static.a");
  return llvm::sys::fs::exists(a);
}


// -----------------------------------------------------------------------------
// GPU (device binary) - NVPTX PTX emission
// -----------------------------------------------------------------------------

static llvm::Expected<std::string> translateToDeviceBinary(mlir::Operation *root) {
#if ARK_ENABLE_CUDA
  static std::once_flag llvmInitOnce;
  std::call_once(llvmInitOnce, [] {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
  });

  if (!root)
    return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                   "translateToDeviceBinary: null root");

  mlir::gpu::GPUModuleOp gpuMod = mlir::dyn_cast<mlir::gpu::GPUModuleOp>(root);
  if (!gpuMod) {
    if (auto m = mlir::dyn_cast<mlir::ModuleOp>(root)) {
      m.walk([&](mlir::gpu::GPUModuleOp op) { if (!gpuMod) gpuMod = op; });
    } else if (auto m = root->getParentOfType<mlir::ModuleOp>()) {
      m.walk([&](mlir::gpu::GPUModuleOp op) { if (!gpuMod) gpuMod = op; });
    }
  }
  if (!gpuMod)
    return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                   "translateToDeviceBinary: no gpu.module found");

  mlir::MLIRContext *ctx = gpuMod->getContext();

  // Isolated compilation: clone gpu.module into a temp module so we never mutate main IR.
  mlir::ModuleOp tmp = mlir::ModuleOp::create(gpuMod.getLoc());
  tmp.getBody()->push_back(gpuMod.clone());

  // Required for MLIR->LLVM IR translation for NVVM.
  mlir::registerBuiltinDialectTranslation(*ctx);
  mlir::registerLLVMDialectTranslation(*ctx);
  mlir::registerNVVMDialectTranslation(*ctx);

  mlir::PassManager pm(ctx);
  pm.enableVerifier(true);
  pm.addPass(mlir::createConvertGpuOpsToNVVMOps());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(pm.run(tmp)))
    return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                   "translateToDeviceBinary: GPU->NVVM lowering failed");

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(tmp, llvmCtx);
  if (!llvmModule)
    return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                   "translateToDeviceBinary: MLIR->LLVM IR translation failed");

  llvm::Triple triple("nvptx64-nvidia-cuda");

  std::string err;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target)
    return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                   "translateToDeviceBinary: NVPTX target not found: %s", err.c_str());

  // TODO: make configurable via flags (sm + ptx version).
  const char *cpu = "sm_60";
  const char *features = "+ptx60";

  llvm::TargetOptions opt;
  std::unique_ptr<llvm::TargetMachine> tm(
      target->createTargetMachine(triple, cpu, features, opt,
                                  std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_)));
  if (!tm)
    return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                   "translateToDeviceBinary: failed to create TargetMachine");

  llvmModule->setTargetTriple(triple);
  llvmModule->setDataLayout(tm->createDataLayout());

  llvm::SmallString<0> buf;
  llvm::raw_svector_ostream os(buf);

  llvm::legacy::PassManager passes;
  if (tm->addPassesToEmitFile(passes, os, nullptr, llvm::CodeGenFileType::AssemblyFile))
    return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                   "translateToDeviceBinary: cannot emit PTX for this TargetMachine");

  passes.run(*llvmModule);
  return std::string(buf.data(), buf.size());
#else
  (void)root;
  return std::string();
#endif
}

struct GpuEmbedPass
    : public mlir::PassWrapper<GpuEmbedPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::gpu::GPUModuleOp, 4> gpuMods;
    module.walk([&](mlir::gpu::GPUModuleOp op) { gpuMods.push_back(op); });

    bool failedGpu = false;

    for (mlir::gpu::GPUModuleOp op : gpuMods) {
      llvm::Expected<std::string> ptxOrErr = translateToDeviceBinary(op);
      if (!ptxOrErr) {
        op.emitError("GPU kernel compilation failed: ") << llvm::toString(ptxOrErr.takeError());
        failedGpu = true;
        continue;
      }

      const std::string &ptx = *ptxOrErr;
      if (ptx.empty()) {
        op.emitError("GPU kernel compilation produced empty PTX");
        failedGpu = true;
        continue;
      }

      // Stash for later harvesting (even if you choose to keep gpu.module for debugging).
      mlir::MLIRContext *ctx = op->getContext();
      op->setAttr("ark.gpu.ptx", mlir::StringAttr::get(ctx, ptx));
      op->setAttr("ark.gpu.ptx_kind", mlir::StringAttr::get(ctx, "nvptx"));

      // If you want to keep the gpu.module for debugging, gate this behind a flag/env.
      op.erase();
    }

    if (failedGpu) signalPassFailure();
  }
};

// -----------------------------------------------------------------------------
// Pipeline
// -----------------------------------------------------------------------------

// Helper to collect all reachable modules in topological order (roughly)
// so we can register and compile dependencies before the main module.
static void collectDependencies(arklang::Module *mod, 
                                llvm::DenseSet<arklang::Module*> &visited, 
                                std::vector<arklang::Module*> &order) {
    if (!mod || visited.count(mod)) return;
    visited.insert(mod);

    // Recurse into submodules (dependencies)
    for (auto &entry : mod->submodules) {
        collectDependencies(entry.second, visited, order);
    }

    // Add self to order
    order.push_back(mod);
}

static bool compileArkToMir(mlir::MLIRContext &ctx, 
                            arklang::Module *ast, // The Root Module (Entry Point)
                            llvm::StringRef sourcePath, 
                            mlir::ModuleOp &moduleOut,
                            arklang::hud::Hud &hud) {
  mlir::OpBuilder builder(&ctx);
  moduleOut = mlir::ModuleOp::create(builder.getUnknownLoc());

  // 1. Initialize GenMIR
  // Note: Updated constructor takes 'hud' for error reporting
  arklang::GenMIR genMir(moduleOut, builder, hud);

  // 2. Collect All Modules (Recursive Flattening)
  llvm::DenseSet<arklang::Module*> visited;
  std::vector<arklang::Module*> allModules;
  collectDependencies(ast, visited, allModules);

  // 3. Register All Schemas & Functions Globally
  // This allows 'A' to know about types defined in 'C' if they are passed through.
  for (auto *mod : allModules) {
      // [FIX] Determine if this is the root module
      bool isRoot = (mod == ast);
      
      // Register with isRoot flag so GenMIR knows whether to mangle names
      // Root: "main" -> "main"
      // Import: "main" -> "generics_main"
      genMir.registerModule(*mod, isRoot);
  }

  // 4. Compile Modules (Dependencies First)
  // We iterate our collected list. The 'ast' (root) will be last in the list 
  // due to the recursive traversal order, which is what we want.
  for (auto *mod : allModules) {
      // [FIX] Determine isRoot again for this loop
      bool isRoot = (mod == ast);

      // Clear previous file's imports from the symbol table to prevent name collision,
      // then inject the imports specific to *this* module.
      genMir.clearImports();
      
      for (const auto &imp : mod->imports) {
          // Map alias "v" -> Module Pointer for lookup during compilation
          if (mod->submodules.count(imp->alias)) {
              genMir.registerImport(imp->alias, mod->submodules[imp->alias]);
          }
      }

      // Compile
      // [FIX] Pass isRoot flag here too
      if (mlir::failed(genMir.compileModule(*mod, isRoot))) {
          // hud.error is handled inside compileModule usually
          return false;
      }
  }

  // 5. Optimization & Verification
  mlir::PassManager pm(&ctx);
  // [Verify Ownership & Drops]
  pm.addNestedPass<mlir::func::FuncOp>(arklang::mir::createOwnershipVerifierPass());
  pm.addNestedPass<mlir::func::FuncOp>(arklang::mir::createDropInsertionPass());
  
  // [Standard Optimizations]
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  
  // [Lowering]
  pm.addPass(arklang::createArkToLLVMPass());

  if (mlir::failed(pm.run(moduleOut))) return false;
  if (mlir::failed(mlir::verify(moduleOut))) return false;

  return true;
}

struct ArklangToLLVMPass
    : public mlir::PassWrapper<ArklangToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::LLVMConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter converter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());

    mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
    mlir::populateVectorToLLVMConversionPatterns(converter, patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(converter, patterns);

    arklang::populateGpuLoweringPatterns(converter, patterns);

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

static bool lowerToLLVMDialect(mlir::MLIRContext &ctx, mlir::ModuleOp module) {
  mlir::PassManager pm(&ctx);
  pm.addPass(arklang::createGpuKernelOutliningPass());
  pm.addPass(std::make_unique<GpuEmbedPass>());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(std::make_unique<ArklangToLLVMPass>());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return mlir::succeeded(pm.run(module));
}

// -----------------------------------------------------------------------------
// JIT Runtime
// -----------------------------------------------------------------------------

static thread_local sigjmp_buf g_panic_env;

static void arkcPanicHandler(const char *msg, const char *file, int32_t line, int32_t col) {
  fprintf(stderr, "[ArkRuntime] Intercepted Panic:\n");
  fprintf(stderr, "  Msg:  %s\n", msg);
  fprintf(stderr, "  Loc:  %s:%d:%d\n", file, line, col);
  siglongjmp(g_panic_env, 1);
}

namespace arklang { void registerRuntimeSymbols(); }

static int runJit(mlir::ModuleOp module, llvm::StringRef runtimeDir) {
  (void)runtimeDir;

  arklang::registerRuntimeSymbols();
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Failed to construct JIT ExecutionEngine: " << llvm::toString(maybeEngine.takeError()) << "\n";
    return 1;
  }
  auto &engine = maybeEngine.get();

  auto panicSym = engine->lookup("arkSetPanicHandler");
  if (panicSym) {
    auto setPanicPtr = (void (*)(void (*)(const char*, const char*, int32_t, int32_t)))panicSym.get();
    setPanicPtr(arkcPanicHandler);
  }

  if (sigsetjmp(g_panic_env, 1) == 0) {
    auto mainSym = engine->lookup("main");
    if (!mainSym) {
      llvm::errs() << "JIT entrypoint 'main' not found.\n";
      return 1;
    }
    auto mainPtr = (int (*)())mainSym.get();
    return mainPtr();
  }

  llvm::outs() << "Program panicked safely.\n";
  return 0;
}

// -----------------------------------------------------------------------------
// Output + Entry
// -----------------------------------------------------------------------------

static bool writeModuleToFile(mlir::ModuleOp module, llvm::StringRef path) {
  std::string osError;
  auto outFile = mlir::openOutputFile(path, &osError);
  if (!outFile) return false;
  module.print(outFile->os());
  outFile->keep();
  return true;
}

static bool ensureEntrypointForBuild(mlir::ModuleOp module) {
  auto mainFn = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!mainFn) {
    if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main")) {
      module.emitError("entrypoint 'main' not found");
      return false;
    }
    return true;
  }

  if (auto vis = mainFn.getSymVisibilityAttr()) {
    if (vis.getValue() == "private") {
      mainFn.emitError("entrypoint 'main' must be public");
      return false;
    }
  }

  return true;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "ArkLang Compiler\n");

  initializeCompilerTargets();

  const bool wantBuild = buildBin || runBin;
  const bool wantJit = useJit;

  arklang::hud::Theme theme;
  theme.useColor = true;
  theme.unicode = true;
  theme.animations = !compilerDebug;
  theme.verbose = compilerDebug;

  theme.showBanner  = infoMode;
  theme.showTimings = infoMode;
  theme.showHints   = infoMode;

  theme.keepHistory = infoMode || compilerDebug;
  theme.compact = !infoMode;
  theme.stream = arklang::hud::Stream::Err;

  arklang::hud::Hud hud(theme, wantBuild || wantJit ? 7 : 3);
  hud.setLabel("arkc");
  if (infoMode) hud.banner("arkc", ARK_VERSION_STRING, "host + gpu");

  arklang::ModuleRegistry registry;
  arklang::Module *rootModule = nullptr;

  {
    arklang::hud::Step s(hud, {"Parsing", "Frontend → AST",
                               std::optional<std::string>("Includes recursive module loading")});
    
    // [FIX] Explicitly get string value and pass "" as relative root
    auto modOr = registry.load(inputFilename.getValue(), ""); 
    
    if (!modOr) {
      s.fail();
      hud.error(llvm::toString(modOr.takeError()));
      hud.finish(false);
      return 1;
    }

    rootModule = *modOr;
    s.ok();
  }

  if (wantBuild || wantJit) {
    // 1. Trust flag if user provided it
    if (runtimePath.getNumOccurrences() > 0) {
       if (!llvm::sys::fs::exists(runtimePath)) {
         hud.error("Runtime not found at provided path: " + runtimePath);
         hud.finish(false);
         return 1;
       }
    } else {
       // 2. Try default relative to CWD
       if (llvm::sys::fs::exists(runtimePath)) {
          // Found at default CWD-relative path
       } else {
          // 3. Try relative to executable (../tools/compiler/Runtime)
          // Assumption: arkc is in bin/ or tools/compiler/
          auto mainExe = llvm::sys::fs::getMainExecutable(argv[0], (void*)main);
          llvm::SmallString<256> p = llvm::sys::path::parent_path(mainExe);
          
          // Try popping 'bin' if we are in build/bin
          llvm::sys::path::append(p, "../tools/compiler/Runtime");
          
          if (llvm::sys::fs::exists(p)) {
              runtimePath = std::string(p.str());
          } else {
              // Try source layout: ../../../tools/compiler/Runtime (if running from build/tools/compiler)
              p = llvm::sys::path::parent_path(mainExe);
              llvm::sys::path::append(p, "../../../tools/compiler/Runtime");
               if (llvm::sys::fs::exists(p)) {
                  runtimePath = std::string(p.str());
              } else {
                  hud.error("Runtime not found. Use --runtime=<path>");
                  hud.finish(false);
                  return 1;
              }
          }
       }
    }
  }

  mlir::DialectRegistry dialectRegistry;
  dialectRegistry.insert<mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::gpu::GPUDialect,
                  mlir::vector::VectorDialect,
                  mlir::tensor::TensorDialect,
                  arklang::mir::ArkMirDialect,
                  mlir::NVVM::NVVMDialect,
                  mlir::ROCDL::ROCDLDialect>();

  mlir::MLIRContext ctx(dialectRegistry);
  ctx.loadAllAvailableDialects();

  arklang::hud::MlirDiagBridge diagBridge(hud, ctx);

  mlir::ModuleOp module;

  {
    arklang::hud::Step s(hud, {"Lowering to ArkMIR", "AST → ArkMIR",
                               std::optional<std::string>("If this fails: dump IR and verify passes order")});
    
    // [FIX] Pass .getValue() here as well for consistency/safety
    if (!compileArkToMir(ctx, rootModule, inputFilename.getValue(), module, hud)) {
      s.fail();
      hud.finish(false);
      return 1;
    }
    s.ok();
  }
  
  if (!ensureEntrypointForBuild(module)) {
    hud.finish(false);
    return 1;
  }

  if (!wantBuild && !wantJit) {
    arklang::hud::Step s(hud, {"Emitting MLIR", outputFilename,
                               std::optional<std::string>("Use -o out.mlir to write to file")});
    const bool ok = writeModuleToFile(module, outputFilename);
    ok ? s.ok() : s.fail();
    hud.finish(ok);
    return ok ? 0 : 1;
  }

  auto tmpDirEx = makeTempDir();
  std::string tmpDir;
  if (!tmpDirEx) {
    hud.error(llvm::toString(tmpDirEx.takeError()));
    hud.finish(false);
    return 1;
  }
  tmpDir = std::move(*tmpDirEx);

  if (keepTmp) hud.note(std::string("tmp: ") + tmpDir);

  if (wantJit) {
    arklang::hud::Step s(hud, {"JIT Compiling", "ArkMIR → LLVM Dialect → JIT",
                               std::optional<std::string>("Try --compiler-debug to see failing op")});
    if (!lowerToLLVMDialect(ctx, module)) {
      s.fail();
      hud.error("Lowering to LLVM Dialect failed.");
      hud.finish(false);
      if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
      return 1;
    }
    s.ok();

    const int rc = runJit(module, runtimePath);
    hud.finish(rc == 0);
    if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
    return rc;
  }

  llvm::SmallString<256> llPath(tmpDir), binPath(tmpDir);
  llvm::sys::path::append(llPath, "out.ll");
  llvm::sys::path::append(binPath, "out.bin");

  {
    arklang::hud::Step s(hud, {"Lowering to LLVM IR", "ArkMIR → LLVM Dialect → LLVM IR",
                               std::optional<std::string>("If it crashes: check Expected<> consumption")});
    if (!lowerToLLVMDialect(ctx, module)) {
      s.fail();
      hud.error("Lowering to LLVM Dialect failed.");
      hud.finish(false);
      if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
      return 1;
    }
    s.ok();
  }

  const std::string llvmBin = defaultLlvmBinDirIfPresent();

  auto mlirTranslatePath =
      consumeToolOrReport(hud, findTool("mlir-translate", mlirTranslateOverride, llvmBin), "mlir-translate");
  if (!mlirTranslatePath) {
    hud.finish(false);
    if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
    return 1;
  }

  const std::string llvmMlirPath = tmpDir + "/out.mlir";
  if (!writeModuleToFile(module, llvmMlirPath)) {
    hud.error("failed to write temporary MLIR file");
    hud.finish(false);
    if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
    return 1;
  }

  {
    arklang::hud::Step s(hud, {"Translating", "mlir-translate --mlir-to-llvmir",
                               std::optional<std::string>("Ensure LLVM/MLIR versions match")});
    if (runCmd(&hud, *mlirTranslatePath,
               {"--mlir-to-llvmir", llvmMlirPath, "-o", llPath.str().str()},
               "mlir-translate") != 0) {
      s.fail();
      hud.finish(false);
      if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
      return 1;
    }
    s.ok();
  }

  auto clangPath = consumeToolOrReport(hud, findTool("clang", clangOverride, llvmBin), "clang");
  if (!clangPath) {
    hud.finish(false);
    if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
    return 1;
  }

  int progRc = 0;

  {
    arklang::hud::Step s(hud, {"Linking", "compiler driver + runtime",
                               std::optional<std::string>("Pass --keep-tmp to inspect artifacts")});

    auto linkerPath = chooseLinker(hud, llvmBin, *clangPath, clangOverride);
    if (!linkerPath) {
      s.fail();
      hud.finish(false);
      if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
      return 1;
    }

    std::vector<std::string> args = {llPath.str().str(), "-I" + runtimePath + "/include"};

    for (const auto &src : kRuntimeSources) {
      std::string p = runtimePath + "/" + src;
      if (llvm::sys::fs::exists(p)) args.push_back(p);
    }

    args.push_back("-pthread");

#if ARK_ENABLE_METAL
    args.insert(args.end(), {"-framework", "Metal", "-framework", "Foundation"});
#endif

#if ARK_ENABLE_CUDA
    std::string cudaLibPath = findCudaLibPath();
    if (!cudaLibPath.empty()) {
      args.push_back("-L" + cudaLibPath);
      args.push_back("-Wl,-rpath," + cudaLibPath);
    } else {
      hud.warn("CUDA lib directory not found; skipping CUDA runtime link");
    }

    args.push_back("-lcuda");

    if (!cudaLibPath.empty() && hasCudaRuntimeIn(cudaLibPath)) {
      args.push_back("-lcudart");
    } else {
      hud.warn("libcudart not found; skipping -lcudart (CUDA runtime features disabled at link)");
    }
#endif

#if ARK_ENABLE_HIP
    args.insert(args.end(), {"-L/opt/rocm/lib", "-lamdhip64", "-Wl,-rpath,/opt/rocm/lib"});
#endif

    if (runtimeNeedsCxxLink()) {
      llvm::StringRef stem = exeStem(*linkerPath);
      if (stem.contains("clang") && !stem.contains("clang++")) {
        hud.warn("Linker is clang (C driver); adding libstdc++ fallback");
        args.push_back("-lstdc++");
        args.push_back("-lm");
        args.push_back("-ldl");
      }
    }

    args.push_back("-Wno-override-module");
    args.insert(args.end(), {"-O3", "-o", binPath.str().str()});

    if (runCmd(&hud, *linkerPath, args, "linker") != 0) {
      s.fail();
      hud.finish(false);
      if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
      return 1;
    }

    if (outputFilename != "-") (void)llvm::sys::fs::copy_file(binPath, outputFilename);
    s.ok();
  }


    if (runBin) {
      arklang::hud::Step r(hud, {"Running", binPath.str().str(),
                                 std::optional<std::string>("Program stdout/stderr is captured and shown below on failure")});

      progRc = runCmd(&hud, binPath.str().str(), {}, "program");

      if (progRc == 0) {
        r.ok();
        hud.note("program ran");
      } else {
        r.fail(); // ensures the step line prints before the error line
        hud.error(std::string("program exited with code ") + std::to_string(progRc));
        hud.flush(); // prints captured stdout/stderr/exec blocks right here
        hud.finish(false);
        if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
        return progRc;
      }
    }
    
  hud.finish(progRc == 0);
  if (!keepTmp) (void)llvm::sys::fs::remove_directories(tmpDir);
  return progRc;
}

