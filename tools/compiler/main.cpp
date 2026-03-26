// -----------------------------------------------------------------------------
// Ark Compiler Driver (v8.0)
//    - Adds real stage controls (compile/lower/translate/link/seal/run)
//    - Keeps default behavior intact when no stage flags are used
// -----------------------------------------------------------------------------

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/ToolOutputFile.h>

#include "Frontend/ModuleRegistry.h"
#include "Integration/CapsuleBackend.h"
#include "Pipeline/Compiler.h"
#include "Pipeline/JIT.h"
#include "Pipeline/Linker.h"

#include "ark/IR/ArkMirOps.h"

// --- Dialect Headers ---
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

// --- Translation Headers (Export to LLVM IR) ---
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// --- Conversion Interface Headers ---
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"

#include "ark_config.h"
#include "hud.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef ARK_VERSION_STRING
  #define ARK_VERSION_STRING "dev"
#endif

namespace cl = llvm::cl;

// -----------------------------------------------------------------------------
// CLI Options
// -----------------------------------------------------------------------------
static cl::OptionCategory ArkCategory("Ark Compiler Options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input .ark file>"), cl::Required, cl::cat(ArkCategory));
static cl::opt<std::string> outputFilename("o", cl::desc("Output file"), cl::value_desc("filename"), cl::init(""), cl::cat(ArkCategory));
static cl::opt<bool> bareMode("bare", cl::desc("Disable capsule security"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<bool> useJit("jit", cl::desc("Use LLVM JIT"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<bool> runMode("run", cl::desc("Compile, Link, Seal, Execute"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<std::string> runtimePath("runtime", cl::desc("Path to Runtime"), cl::init("tools/compiler/Runtime"), cl::cat(ArkCategory));
static cl::opt<std::string> llvmBinDir("llvm-bin", cl::desc("Directory containing clang"), cl::init(""), cl::cat(ArkCategory));
static cl::opt<bool> keepTmp("keep-tmp", cl::desc("Keep artifacts"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<bool> debugMode("verbose", cl::desc("Verbose logging"), cl::init(false), cl::cat(ArkCategory));

// ---- New: stage controls ----
static cl::opt<bool> stageCompile("emit-mlir", cl::desc("Stop after MLIR is built (MIR lowering passes complete)"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<bool> stageLower("emit-llvm-dialect", cl::desc("Stop after lowering to LLVM dialect (captures GPU modules)"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<bool> stageTranslate("emit-llvm-ir", cl::desc("Emit LLVM IR (.ll) and stop (no link)"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<bool> stageLinkOnly("link-only", cl::desc("Link an existing .ll from -in-ll into a binary (skips compile/translate)"), cl::init(false), cl::cat(ArkCategory));
static cl::opt<std::string> inLl("in-ll", cl::desc("Input LLVM IR (.ll) for -link-only"), cl::init(""), cl::cat(ArkCategory));
static cl::opt<bool> stageNoSeal("no-seal", cl::desc("Do not create capsule even if default would (useful with -emit-llvm-ir/-emit-llvm-dialect)"), cl::init(false), cl::cat(ArkCategory));

static const std::vector<std::uint8_t> MOCK_KEY = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                                    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
                                                    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
                                                    0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20 };

static std::string fmtBytes(std::uint64_t bytes) {
  static constexpr std::array<const char*, 5> units = {"B", "KiB", "MiB", "GiB", "TiB"};
  long double v = static_cast<long double>(bytes);
  std::size_t u = 0;
  while (u + 1 < units.size() && v >= 1024.0L) { v /= 1024.0L; ++u; }
  std::ostringstream o; o.setf(std::ios::fixed);
  if (u == 0) { o.unsetf(std::ios::floatfield); o << bytes << ' ' << units[u]; return o.str(); }
  const int prec = (v < 10.0L) ? 2 : (v < 100.0L ? 1 : 1);
  o << std::setprecision(prec) << static_cast<double>(v) << ' ' << units[u];
  return o.str();
}

static std::string resolveRuntimePath(const char* argv0) {
  if (runtimePath.getNumOccurrences() > 0) return runtimePath;
  void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(resolveRuntimePath));
  auto mainExe = llvm::sys::fs::getMainExecutable(argv0, mainAddr);
  llvm::SmallString<256> p = llvm::sys::path::parent_path(mainExe);
  llvm::sys::path::append(p, "../tools/compiler/Runtime");
  if (llvm::sys::fs::exists(p)) return std::string(p.str());
  p = llvm::sys::path::parent_path(mainExe);
  llvm::sys::path::append(p, "../../tools/compiler/Runtime");
  if (llvm::sys::fs::exists(p)) return std::string(p.str());
  return "tools/compiler/Runtime";
}

template <typename Fn>
static bool runStep(arklang::hud::Hud& hud, arklang::hud::StepMeta meta, Fn&& fn) {
  arklang::hud::Step s(hud, std::move(meta));
  if (!fn()) { s.fail(); return false; }
  s.ok(); return true;
}

static std::string readAllFile(llvm::StringRef p) {
  auto mb = llvm::MemoryBuffer::getFile(p);
  if (!mb) return {};
  return std::string(mb.get()->getBuffer());
}

static void removeQuietly(llvm::StringRef p) { if (!p.empty()) (void)llvm::sys::fs::remove(p); }
static void removeDirQuietly(llvm::StringRef p) { if (p.empty()) return; (void)llvm::sys::fs::remove_directories(p, /*IgnoreErrors=*/true); }

static void bestEffortAuditFatbinExport(arklang::hud::Hud& hud, llvm::StringRef binaryPath, bool enabled) {
  if (!enabled) return;
  auto nm = llvm::sys::findProgramByName("nm");
  if (!nm) return;
  llvm::SmallString<256> outPath; llvm::SmallString<256> errPath;
  (void)llvm::sys::fs::createTemporaryFile("ark_nm_out", "log", outPath);
  (void)llvm::sys::fs::createTemporaryFile("ark_nm_err", "log", errPath);
  llvm::SmallVector<llvm::StringRef, 8> argv;
  argv.push_back(*nm); argv.push_back("-D"); argv.push_back(binaryPath);
  std::string errMsg;
  const llvm::StringRef outRef(outPath); const llvm::StringRef errRef(errPath);
  std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};
  (void)llvm::sys::ExecuteAndWait(*nm, argv, std::nullopt, redirects, 0, 0, &errMsg);
  const std::string so = readAllFile(outPath);
  removeQuietly(outPath); removeQuietly(errPath);
  if (!errMsg.empty()) { hud.pushLogBlock("nm exec error", errMsg); return; }
  const bool found = (so.find("ark_gpu_fatbin_query_v1") != std::string::npos);
  hud.note(std::string("fatbin registry export: ") + (found ? "present" : "missing"));
}

static std::optional<llvm::ArrayRef<llvm::StringRef>>
buildInheritedEnv(llvm::SmallVectorImpl<std::string>& storage, llvm::SmallVectorImpl<llvm::StringRef>& refs) {
#if defined(_WIN32)
  (void)storage; (void)refs; return std::nullopt;
#else
  extern char** environ;
  storage.clear(); refs.clear();
  if (!environ) return std::nullopt;
  for (char** e = environ; *e != nullptr; ++e) storage.emplace_back(*e);
  refs.reserve(storage.size());
  for (auto& s : storage) refs.push_back(s);
  return llvm::ArrayRef<llvm::StringRef>(refs);
#endif
}

static bool anyStageFlagsSpecified() {
  return stageCompile || stageLower || stageTranslate || stageLinkOnly;
}

enum class StopAfter {
  None,
  MLIR,
  LLVMDialect,
  LLVMIR
};



static std::string defaultStageOutputPath(llvm::StringRef input, StopAfter stopAfter) {
  llvm::StringRef stem = llvm::sys::path::stem(input);
  std::string out;
  out.reserve(stem.size() + 16);

  switch (stopAfter) {
    case StopAfter::MLIR:        out = (stem + ".mlir").str(); break;
    case StopAfter::LLVMDialect: out = (stem + ".llvm.mlir").str(); break;
    case StopAfter::LLVMIR:      out = (stem + ".ll").str(); break;
    case StopAfter::None:        out = "a.out"; break;
  }
  return out;
}

static StopAfter computeStopAfter() {
  if (!anyStageFlagsSpecified()) return StopAfter::None;
  if (stageLinkOnly) return StopAfter::None;
  if (stageTranslate) return StopAfter::LLVMIR;
  if (stageLower) return StopAfter::LLVMDialect;
  if (stageCompile) return StopAfter::MLIR;
  return StopAfter::None;
}

static int computeSteps(bool useJitLocal, bool wantCapsule, bool runLocal, StopAfter stopAfter, bool linkOnly) {
  int steps = 0;
  if (linkOnly) {
    steps += 1; // Linking
    if (wantCapsule && !stageNoSeal) steps += 1;
    if (runLocal) steps += 1;
    return steps;
  }

  steps += 1; // Compiling
  if (stopAfter == StopAfter::MLIR) return steps;

  steps += 1; // Lowering
  if (stopAfter == StopAfter::LLVMDialect) return steps;

  if (useJitLocal) {
    steps += 1; // JIT
    return steps;
  }

  steps += 1; // Translating
  if (stopAfter == StopAfter::LLVMIR) return steps;

  steps += 1; // Linking
  if (wantCapsule && !stageNoSeal) steps += 1;
  if (runLocal) steps += 1;
  return steps;
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  llvm::InitLLVM init(argc, argv);
  cl::HideUnrelatedOptions(ArkCategory);
  cl::ParseCommandLineOptions(argc, argv, "Ark Compiler v8.0\n");

  if (stageLinkOnly && inLl.empty()) {
    llvm::errs() << "error: -link-only requires -in-ll=<path-to.ll>\n";
    return 1;
  }
  if (stageLinkOnly && (stageCompile || stageLower || stageTranslate)) {
    llvm::errs() << "error: -link-only cannot be combined with -emit-mlir/-emit-llvm-dialect/-emit-llvm-ir\n";
    return 1;
  }

  const StopAfter stopAfter = computeStopAfter();

  const bool wantCapsuleDefault = (runMode && !bareMode) ? true : (!bareMode && !useJit);
  const bool wantCapsule = wantCapsuleDefault && !stageNoSeal;
  std::string finalOutput = outputFilename;
  bool isTempRunOutput = false;

  const bool willProduceBinary =
      stageLinkOnly ||
      (!useJit && stopAfter == StopAfter::None) ||
      (!useJit && stopAfter == StopAfter::None && runMode) ||
      (!useJit && stopAfter == StopAfter::None && wantCapsuleDefault);

  if (runMode && finalOutput.empty()) {
    llvm::SmallString<256> tmp;
    if (llvm::sys::fs::createTemporaryFile("ark_run", wantCapsule ? "exe" : "bin", tmp)) return 1;
    finalOutput = std::string(tmp.str());
    isTempRunOutput = true;
  } else if (finalOutput.empty()) {
    finalOutput = wantCapsule ? "a.out" : "a.raw";
  }

  arklang::hud::Theme theme; theme.verbose = debugMode;
  const int totalSteps = computeSteps(useJit, wantCapsuleDefault, runMode, stopAfter, stageLinkOnly);
  arklang::hud::Hud hud(theme, totalSteps);
  hud.setLabel("arkc");
  if (debugMode) hud.banner("arkc", ARK_VERSION_STRING, wantCapsuleDefault ? "Secure Pipeline" : "Raw Pipeline");

  const std::string validRuntime = resolveRuntimePath(argv[0]);
  if (!llvm::sys::fs::exists(validRuntime)) {
    hud.error(std::string("Runtime not found at: ") + validRuntime);
    hud.finish(false);
    return 1;
  }

  arklang::ModuleRegistry registry;
  ark::compiler::pipeline::Compiler compiler(hud, registry);
  ark::compiler::pipeline::LinkerConfig linkCfg;
  linkCfg.runtimePath = validRuntime; linkCfg.llvmBinDir = llvmBinDir; linkCfg.keepTmp = keepTmp;
  ark::compiler::pipeline::Linker linker(hud, linkCfg);

  mlir::DialectRegistry dialectRegistry;

  dialectRegistry.insert<
      arklang::mir::ArkMirDialect,
      mlir::func::FuncDialect,
      mlir::LLVM::LLVMDialect,
      mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::cf::ControlFlowDialect,
      mlir::vector::VectorDialect,
      mlir::index::IndexDialect,
      mlir::gpu::GPUDialect,
      mlir::ub::UBDialect,
      mlir::NVVM::NVVMDialect,
      mlir::ROCDL::ROCDLDialect>();

  mlir::arith::registerConvertArithToLLVMInterface(dialectRegistry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(dialectRegistry);
  mlir::registerConvertFuncToLLVMInterface(dialectRegistry);
  mlir::index::registerConvertIndexToLLVMInterface(dialectRegistry);
  mlir::registerConvertMemRefToLLVMInterface(dialectRegistry);
  mlir::vector::registerConvertVectorToLLVMInterface(dialectRegistry);
  mlir::ub::registerConvertUBToLLVMInterface(dialectRegistry);
  mlir::registerConvertNVVMToLLVMInterface(dialectRegistry);

  mlir::registerBuiltinDialectTranslation(dialectRegistry);
  mlir::registerLLVMDialectTranslation(dialectRegistry);
  mlir::registerNVVMDialectTranslation(dialectRegistry);
  mlir::registerROCDLDialectTranslation(dialectRegistry);
  mlir::registerGPUDialectTranslation(dialectRegistry);

  mlir::MLIRContext ctx(dialectRegistry);
  ctx.loadAllAvailableDialects();

  mlir::ModuleOp module;
  std::vector<ark::compiler::pipeline::CompiledGpuModule> gpuMods;

  // ---------------------------------------------------------------------------
  // Link-only path (skips compile/lower/translate)
  // ---------------------------------------------------------------------------
  if (stageLinkOnly) {
    const std::string tmpDir = linker.makeTempDir();
    if (keepTmp) hud.note(std::string("Build Temp: ") + tmpDir);

    const std::string binaryPath = wantCapsuleDefault ? (tmpDir + "/payload.bin") : finalOutput;

    if (!runStep(hud, {"Linking", "LLVM IR → Native Binary"}, [&] {
          return linker.linkToBinary(inLl, binaryPath, gpuMods);
        })) {
      hud.finish(false);
      if (!keepTmp) removeDirQuietly(tmpDir);
      if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
      return 1;
    }

    bestEffortAuditFatbinExport(hud, binaryPath, debugMode && !gpuMods.empty());

    if (wantCapsuleDefault && !stageNoSeal) {
      arklang::hud::Step s(hud, {"Sealing", "Creating Self-Executing Capsule"});
      void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(main));
      auto mainExe = llvm::sys::fs::getMainExecutable(argv[0], mainAddr);
      llvm::SmallString<256> stubPath = llvm::sys::path::parent_path(mainExe);
      llvm::sys::path::append(stubPath, "ark-stub");
      if (!llvm::sys::fs::exists(stubPath)) {
        hud.error(std::string("Stub not found: ") + std::string(stubPath.str()));
        s.fail(); hud.finish(false);
        if (!keepTmp) removeDirQuietly(tmpDir);
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
      }
      llvm::SmallString<256> srcDir = llvm::sys::path::parent_path(inputFilename);
      if (srcDir.empty()) srcDir = ".";
      bool ok = false; std::uint64_t sealedSize = 0;
      {
        arklang::hud::Hud::ScopedDiagnostics diag(hud);
        ok = ark::compiler::integration::CapsuleBackend::CreateSelfExecutingCapsule(
            hud, std::string(srcDir.str()), std::string(stubPath.str()), binaryPath, finalOutput, MOCK_KEY, &sealedSize);
      }
      if (!ok) {
        s.fail(); hud.finish(false);
        if (!keepTmp) removeDirQuietly(tmpDir);
        if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
        return 1;
      }
      (void)llvm::sys::fs::setPermissions(finalOutput, llvm::sys::fs::owner_exe | llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
      s.ok();
      hud.pushLogBlock("Sealed", finalOutput + " (" + fmtBytes(sealedSize) + ")");
      hud.flush();
      if (!keepTmp) removeQuietly(binaryPath);
    }

    if (runMode) {
      arklang::hud::Step s(hud, {"Executing", finalOutput});
      llvm::SmallString<256> outPath; llvm::SmallString<256> errPath;
      (void)llvm::sys::fs::createTemporaryFile("ark_run_out", "log", outPath);
      (void)llvm::sys::fs::createTemporaryFile("ark_run_err", "log", errPath);
      std::string errMsg; int ret = 0;
      {
        arklang::hud::Hud::ScopedDiagnostics diag(hud);
        llvm::SmallVector<llvm::StringRef, 4> args = {finalOutput};
        const llvm::StringRef outRef(outPath); const llvm::StringRef errRef(errPath);
        std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};
        llvm::SmallVector<std::string, 256> envStorage; llvm::SmallVector<llvm::StringRef, 256> envRefs;
        const std::optional<llvm::ArrayRef<llvm::StringRef>> env = buildInheritedEnv(envStorage, envRefs);
        ret = llvm::sys::ExecuteAndWait(finalOutput, args, env, redirects, 0, 0, &errMsg);
      }
      const std::string progOut = readAllFile(outPath); const std::string progErr = readAllFile(errPath);
      removeQuietly(outPath); removeQuietly(errPath);
      if (ret == 0 && errMsg.empty()) s.ok(); else s.fail();
      if (!progOut.empty()) hud.pushLogBlock("Program stdout", progOut);
      if (!progErr.empty()) hud.pushLogBlock("Program stderr", progErr);
      if (!progOut.empty() || !progErr.empty()) hud.flush();
      if (!errMsg.empty()) hud.error(std::string("Execution failed: ") + errMsg);
      else if (ret != 0) hud.error(std::string("Execution failed: non-zero exit code ") + std::to_string(ret));
      if (!keepTmp) removeQuietly(finalOutput);
      hud.finish(ret == 0);
      return ret == 0 ? 0 : 1;
    }

    if (!keepTmp) removeDirQuietly(tmpDir);
    hud.finish(true);
    return 0;
  }

  // ---------------------------------------------------------------------------
  // Normal pipeline (default behavior preserved when no stage flags are used)
  // ---------------------------------------------------------------------------
  if (!runStep(hud, {"Compiling", "Source → MIR"}, [&] { return compiler.compileToMLIR(inputFilename, ctx, module); })) {
    hud.finish(false);
    if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
    return 1;
  }

  // 3) Emit files on stop stages even when -o is missing
  // Replace the StopAfter::MLIR block with:

  if (stopAfter == StopAfter::MLIR) {
    const std::string outPath = outputFilename.empty()
        ? defaultStageOutputPath(inputFilename, StopAfter::MLIR)
        : outputFilename;

    if (!compiler.writeModuleToFile(module, outPath)) {
      hud.error(std::string("Failed to write MLIR to: ") + outPath);
      hud.finish(false);
      return 1;
    }
    hud.note(std::string("wrote MLIR: ") + outPath);
    hud.finish(true);
    return 0;
  }

  // 2) Modify the lowering step: keep ark.gpu.modules when stopping at LLVM dialect
  // Replace your lowering runStep lambda body with this version:

  if (!runStep(hud, {"Lowering", "MIR → LLVM Dialect"}, [&] {
        const bool ok = compiler.lowerToLLVM(ctx, module);
        if (!ok) return false;

        if (stopAfter != StopAfter::LLVMDialect) {
          gpuMods = compiler.extractCompiledGpuModules(module);
          if (debugMode) {
            hud.note(std::string("GPU modules captured: ") + std::to_string(gpuMods.size()));
            for (const auto& m : gpuMods) {
              hud.note(std::string("  gpu: key=") + m.moduleKey +
                       " kind=" + std::to_string(m.moduleKind) +
                       " bytes=" + std::to_string(m.blob.size()) +
                       " kernels=" + std::to_string(m.kernels.size()));
            }
          }
        } else if (debugMode) {
          hud.note("stop after LLVM dialect: preserving ark.gpu.modules in MLIR");
        }

        return true;
      })) {
    hud.finish(false);
    if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
    return 1;
  }


  // Replace the StopAfter::LLVMDialect block with:

  if (stopAfter == StopAfter::LLVMDialect) {
    const std::string outPath = outputFilename.empty()
        ? defaultStageOutputPath(inputFilename, StopAfter::LLVMDialect)
        : outputFilename;

    if (!compiler.writeModuleToFile(module, outPath)) {
      hud.error(std::string("Failed to write LLVM dialect MLIR to: ") + outPath);
      hud.finish(false);
      return 1;
    }
    hud.note(std::string("wrote LLVM dialect MLIR: ") + outPath);
    hud.finish(true);
    return 0;
  }

  if (useJit) {
    arklang::hud::Step s(hud, {"JIT Execution", "Running in-memory"});
    int rc = 0;
    {
      arklang::hud::Hud::ScopedDiagnostics diag(hud);
      rc = ark::compiler::pipeline::JIT::Run(module, validRuntime);
    }
    if (rc == 0) s.ok(); else s.fail();
    hud.finish(rc == 0);
    return rc;
  }

  const std::string tmpDir = linker.makeTempDir();
  if (keepTmp) hud.note(std::string("Build Temp: ") + tmpDir);

  const std::string binaryPath = wantCapsuleDefault ? (tmpDir + "/payload.bin") : finalOutput;

  std::string tmpLl;
  if (stopAfter == StopAfter::LLVMIR) {
    tmpLl = outputFilename.empty() ? (tmpDir + "/out.ll") : outputFilename;
  } else {
    tmpLl = tmpDir + "/out.ll";
  }

  if (!runStep(hud, {"Translating", "MLIR → LLVM IR (In-Process)"}, [&] {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule) { hud.error("Translation to LLVM IR failed internally."); return false; }
        std::error_code ec;
        llvm::ToolOutputFile out(tmpLl, ec, llvm::sys::fs::OF_None);
        if (ec) { hud.error(std::string("Failed to open output file: ") + tmpLl); return false; }
        llvmModule->print(out.os(), nullptr);
        out.keep();
        return true;
      })) {
    hud.finish(false);
    if (!keepTmp) { removeQuietly(tmpLl); removeDirQuietly(tmpDir); }
    if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
    return 1;
  }

  if (stopAfter == StopAfter::LLVMIR) {
    if (debugMode) hud.note(std::string("wrote LLVM IR: ") + tmpLl);
    hud.finish(true);
    if (!keepTmp && outputFilename.empty()) removeDirQuietly(tmpDir);
    return 0;
  }

  if (!runStep(hud, {"Linking", "LLVM IR → Native Binary"}, [&] {
        return linker.linkToBinary(tmpLl, binaryPath, gpuMods);
      })) {
    hud.finish(false);
    if (!keepTmp) { removeQuietly(tmpLl); removeDirQuietly(tmpDir); }
    if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
    return 1;
  }

  bestEffortAuditFatbinExport(hud, binaryPath, debugMode && !gpuMods.empty());

  if (wantCapsuleDefault && !stageNoSeal) {
    arklang::hud::Step s(hud, {"Sealing", "Creating Self-Executing Capsule"});
    void* mainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(main));
    auto mainExe = llvm::sys::fs::getMainExecutable(argv[0], mainAddr);
    llvm::SmallString<256> stubPath = llvm::sys::path::parent_path(mainExe);
    llvm::sys::path::append(stubPath, "ark-stub");
    if (!llvm::sys::fs::exists(stubPath)) {
      hud.error(std::string("Stub not found: ") + std::string(stubPath.str()));
      s.fail(); hud.finish(false);
      if (!keepTmp) { removeQuietly(tmpLl); removeDirQuietly(tmpDir); }
      if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
      return 1;
    }
    llvm::SmallString<256> srcDir = llvm::sys::path::parent_path(inputFilename);
    if (srcDir.empty()) srcDir = ".";
    bool ok = false; std::uint64_t sealedSize = 0;
    {
      arklang::hud::Hud::ScopedDiagnostics diag(hud);
      ok = ark::compiler::integration::CapsuleBackend::CreateSelfExecutingCapsule(
          hud, std::string(srcDir.str()), std::string(stubPath.str()), binaryPath, finalOutput, MOCK_KEY, &sealedSize);
    }
    if (!ok) {
      s.fail(); hud.finish(false);
      if (!keepTmp) { removeQuietly(tmpLl); removeDirQuietly(tmpDir); }
      if (runMode && isTempRunOutput && !keepTmp) removeQuietly(finalOutput);
      return 1;
    }
    (void)llvm::sys::fs::setPermissions(finalOutput, llvm::sys::fs::owner_exe | llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
    s.ok();
    hud.pushLogBlock("Sealed", finalOutput + " (" + fmtBytes(sealedSize) + ")");
    hud.flush();
    if (!keepTmp) removeQuietly(binaryPath);
  } else if (!wantCapsuleDefault && (binaryPath == finalOutput)) {
    // raw binary already at finalOutput
  } else if (wantCapsuleDefault && stageNoSeal) {
    // capsule suppressed; if default would have produced payload.bin, copy it to -o if user asked
    if (!outputFilename.empty() && finalOutput != binaryPath) {
      (void)llvm::sys::fs::copy_file(binaryPath, finalOutput);
    }
  }

  if (runMode) {
    arklang::hud::Step s(hud, {"Executing", finalOutput});
    llvm::SmallString<256> outPath; llvm::SmallString<256> errPath;
    (void)llvm::sys::fs::createTemporaryFile("ark_run_out", "log", outPath);
    (void)llvm::sys::fs::createTemporaryFile("ark_run_err", "log", errPath);
    std::string errMsg; int ret = 0;
    {
      arklang::hud::Hud::ScopedDiagnostics diag(hud);
      llvm::SmallVector<llvm::StringRef, 4> args = {finalOutput};
      const llvm::StringRef outRef(outPath); const llvm::StringRef errRef(errPath);
      std::optional<llvm::StringRef> redirects[] = {std::nullopt, outRef, errRef};
      llvm::SmallVector<std::string, 256> envStorage; llvm::SmallVector<llvm::StringRef, 256> envRefs;
      std::optional<llvm::ArrayRef<llvm::StringRef>> env = buildInheritedEnv(envStorage, envRefs);
      
      // --- NEW: Inject ARK_GPU_PLUGIN_DIR dynamically ---
      // We look relative to where arkc itself is running, not where the temp capsule is.
      void* myMainAddr = reinterpret_cast<void*>(reinterpret_cast<std::intptr_t>(main));
      auto myExe = llvm::sys::fs::getMainExecutable(argv[0], myMainAddr);
      llvm::SmallString<256> myExeDir = llvm::sys::path::parent_path(myExe);
      llvm::sys::path::append(myExeDir, "lib"); // This resolves to bin/lib where the .so files are
      
      std::string pluginEnv = "ARK_GPU_PLUGIN_DIR=" + std::string(myExeDir.str());
      envStorage.push_back(pluginEnv);
      
      // Re-populate envRefs because envStorage might have reallocated
      envRefs.clear();
      for (const auto& s : envStorage) {
          envRefs.push_back(s);
      }
      env = llvm::ArrayRef<llvm::StringRef>(envRefs);
      // ------------------------------------------------
      
      ret = llvm::sys::ExecuteAndWait(finalOutput, args, env, redirects, 0, 0, &errMsg);
    }
    const std::string progOut = readAllFile(outPath); const std::string progErr = readAllFile(errPath);
    removeQuietly(outPath); removeQuietly(errPath);
    if (ret == 0 && errMsg.empty()) s.ok(); else s.fail();
    if (!progOut.empty()) hud.pushLogBlock("Program stdout", progOut);
    if (!progErr.empty()) hud.pushLogBlock("Program stderr", progErr);
    if (!progOut.empty() || !progErr.empty()) hud.flush();
    if (!errMsg.empty()) hud.error(std::string("Execution failed: ") + errMsg);
    else if (ret != 0) hud.error(std::string("Execution failed: non-zero exit code ") + std::to_string(ret));
    if (!keepTmp) removeQuietly(finalOutput);
    hud.finish(ret == 0);
    return ret == 0 ? 0 : 1;
  }

  if (!keepTmp) { removeQuietly(tmpLl); removeDirQuietly(tmpDir); }
  hud.finish(true);
  return 0;
}
