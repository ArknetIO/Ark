#include "ark/compiler/Pipeline/Compiler.hpp"
#include "ark/compiler/Support/Hud.hpp"

// Frontend & Utils
#include "ark/compiler/Frontend/GenMIR.hpp"
#include "ark/compiler/Frontend/ModuleRegistry.hpp"
#include "ark/compiler/Analysis/OwnershipVerifier.hpp"
#include "ark/compiler/Transforms/Passes.hpp"
#include "ark/IR/ArkMirOps.h"

// MLIR Core
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

// Conversion Headers
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

// Analysis
#include "mlir/Analysis/DataLayoutAnalysis.h"

// Dialect IR
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

// Required for amdgpu::Chipset parsing
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"

// GPU Translations & Passes
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

// LLVM Translation
#include "mlir/Target/LLVMIR/Export.h"

// LLVM Backend
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

// Explicit GPU Initialization
extern "C" {
    void LLVMInitializeNVPTXTarget();
    void LLVMInitializeNVPTXTargetInfo();
    void LLVMInitializeNVPTXTargetMC();
    void LLVMInitializeNVPTXAsmPrinter();

    void LLVMInitializeAMDGPUTarget();
    void LLVMInitializeAMDGPUTargetInfo();
    void LLVMInitializeAMDGPUTargetMC();
    void LLVMInitializeAMDGPUAsmPrinter();
    void LLVMInitializeAMDGPUAsmParser();
}

namespace arklang {
void populateGpuLoweringPatterns(mlir::LLVMTypeConverter& converter, mlir::RewritePatternSet& patterns);
std::unique_ptr<mlir::Pass> createGpuKernelOutliningPass();
} // namespace arklang

namespace ark::compiler::pipeline {

using namespace arklang;

static constexpr std::uint32_t kArkGpuModuleCudaPtx  = 1;
static constexpr std::uint32_t kArkGpuModuleHipHsaco = 3;
static constexpr std::uint32_t kArkGpuModuleSrcMsl   = 7;

static void collectDependencies(arklang::Module* mod,
                                llvm::DenseSet<arklang::Module*>& visited,
                                std::vector<arklang::Module*>& order) {
    if (!mod || visited.count(mod)) {
        return;
    }

    visited.insert(mod);
    for (auto& entry : mod->submodules) {
        collectDependencies(entry.second, visited, order);
    }
    order.push_back(mod);
}

// -----------------------------------------------------------------------------
// Central MLIR context bootstrap
// -----------------------------------------------------------------------------
static mlir::DialectRegistry makeCompilerDialectRegistry() {
    mlir::DialectRegistry registry;

    registry.insert<
        arklang::mir::ArkMirDialect,
        mlir::func::FuncDialect,
        mlir::arith::ArithDialect,
        mlir::cf::ControlFlowDialect,
        mlir::memref::MemRefDialect,
        mlir::scf::SCFDialect,
        mlir::tensor::TensorDialect,
        mlir::vector::VectorDialect,
        mlir::index::IndexDialect,
        mlir::ub::UBDialect,
        mlir::LLVM::LLVMDialect,
        mlir::gpu::GPUDialect,
        mlir::NVVM::NVVMDialect,
        mlir::ROCDL::ROCDLDialect
    >();

    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    mlir::vector::registerConvertVectorToLLVMInterface(registry);
    mlir::index::registerConvertIndexToLLVMInterface(registry);
    mlir::ub::registerConvertUBToLLVMInterface(registry);
    mlir::registerConvertNVVMToLLVMInterface(registry);

    ::mlir::registerBuiltinDialectTranslation(registry);
    ::mlir::registerLLVMDialectTranslation(registry);
    ::mlir::registerGPUDialectTranslation(registry);
    ::mlir::registerNVVMDialectTranslation(registry);
    ::mlir::registerROCDLDialectTranslation(registry);

    return registry;
}

static void prepareCompilerContext(mlir::MLIRContext& ctx) {
    static const mlir::DialectRegistry registry = makeCompilerDialectRegistry();
    ctx.appendDialectRegistry(registry);

    ctx.loadDialect<
        arklang::mir::ArkMirDialect,
        mlir::func::FuncDialect,
        mlir::arith::ArithDialect,
        mlir::cf::ControlFlowDialect,
        mlir::memref::MemRefDialect,
        mlir::scf::SCFDialect,
        mlir::tensor::TensorDialect,
        mlir::vector::VectorDialect,
        mlir::index::IndexDialect,
        mlir::ub::UBDialect,
        mlir::LLVM::LLVMDialect,
        mlir::gpu::GPUDialect,
        mlir::NVVM::NVVMDialect,
        mlir::ROCDL::ROCDLDialect
    >();
}

// -----------------------------------------------------------------------------
// LLVM backend bootstrap
// -----------------------------------------------------------------------------
static void initLlvmTargetsOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (llvm::InitializeNativeTarget()) {
            llvm::report_fatal_error("Failed to initialize native LLVM target");
        }
        if (llvm::InitializeNativeTargetAsmPrinter()) {
            llvm::report_fatal_error("Failed to initialize native LLVM asm printer");
        }
        if (llvm::InitializeNativeTargetAsmParser()) {
            llvm::report_fatal_error("Failed to initialize native LLVM asm parser");
        }

#if ARK_ENABLE_NVPTX_GPU_PIPELINE
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
#endif

#if ARK_ENABLE_AMDGPU_GPU_PIPELINE
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUAsmParser();
#endif
    });
}

// -----------------------------------------------------------------------------
// Helpers for llvm::Module target triple API differences
// -----------------------------------------------------------------------------
namespace {
template <class>
inline constexpr bool dependent_false_v = false;

template <class M>
static auto setLlvmModuleTargetTripleImpl(M& m, const llvm::Triple& t, int)
    -> decltype(m.setTargetTriple(t), void()) {
    m.setTargetTriple(t);
}

template <class M>
static auto setLlvmModuleTargetTripleImpl(M& m, const llvm::Triple& t, long)
    -> decltype(m.setTargetTriple(t.getTriple()), void()) {
    m.setTargetTriple(t.getTriple());
}

template <class M>
static void setLlvmModuleTargetTripleImpl(M&, const llvm::Triple&, ...) {
    static_assert(dependent_false_v<M>, "llvm::Module::setTargetTriple overload not supported");
}

static void setLlvmModuleTargetTriple(llvm::Module& m, const llvm::Triple& triple) {
    setLlvmModuleTargetTripleImpl(m, triple, 0);
}
} // namespace

namespace detail {
template <typename RelocOpt>
static auto createTM(const llvm::Target* target,
                     const llvm::Triple& triple,
                     llvm::StringRef cpu,
                     llvm::StringRef feats,
                     const llvm::TargetOptions& opt,
                     RelocOpt reloc)
    -> decltype(target->createTargetMachine(triple, cpu, feats, opt, reloc),
                std::unique_ptr<llvm::TargetMachine>{}) {
    return std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(triple, cpu, feats, opt, reloc)
    );
}

template <typename RelocOpt>
static auto createTM(const llvm::Target* target,
                     const llvm::Triple& triple,
                     llvm::StringRef cpu,
                     llvm::StringRef feats,
                     const llvm::TargetOptions& opt,
                     RelocOpt reloc)
    -> decltype(target->createTargetMachine(llvm::StringRef(triple.getTriple()), cpu, feats, opt, reloc),
                std::unique_ptr<llvm::TargetMachine>{}) {
    const std::string tt = triple.getTriple();
    return std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(llvm::StringRef(tt), cpu, feats, opt, reloc)
    );
}

template <typename RelocOpt, typename = void>
struct CanCreateTM : std::false_type {};

template <typename RelocOpt>
struct CanCreateTM<RelocOpt,
                   std::void_t<decltype(createTM(std::declval<const llvm::Target*>(),
                                                std::declval<const llvm::Triple&>(),
                                                std::declval<llvm::StringRef>(),
                                                std::declval<llvm::StringRef>(),
                                                std::declval<const llvm::TargetOptions&>(),
                                                std::declval<RelocOpt>()))>> : std::true_type {};
} // namespace detail

static std::unique_ptr<llvm::TargetMachine>
makeTargetMachine(const llvm::Target* target,
                  const llvm::Triple& triple,
                  llvm::StringRef cpu,
                  llvm::StringRef feats) {
    llvm::TargetOptions opt;
    using Reloc = std::optional<llvm::Reloc::Model>;

    if constexpr (detail::CanCreateTM<Reloc>::value) {
        return detail::createTM(target, triple, cpu, feats, opt, Reloc{llvm::Reloc::PIC_});
    }

    return nullptr;
}

// -----------------------------------------------------------------------------
// Manifest Helpers
// -----------------------------------------------------------------------------
static mlir::DenseIntElementsAttr bytesToDenseI8(mlir::MLIRContext* ctx, llvm::StringRef bytes) {
    auto i8 = mlir::IntegerType::get(ctx, 8);
    auto ty = mlir::RankedTensorType::get({static_cast<int64_t>(bytes.size())}, i8);

    llvm::SmallVector<llvm::APInt, 256> elems;
    elems.reserve(bytes.size());
    for (unsigned char c : bytes) {
        elems.emplace_back(8, static_cast<uint64_t>(c), false);
    }

    return mlir::DenseIntElementsAttr::get(ty, elems);
}

static mlir::ArrayAttr stringsToArrayAttr(mlir::MLIRContext* ctx, const std::vector<std::string>& ss) {
    llvm::SmallVector<mlir::Attribute, 16> out;
    out.reserve(ss.size());

    for (const auto& s : ss) {
        out.push_back(mlir::StringAttr::get(ctx, s));
    }

    return mlir::ArrayAttr::get(ctx, out);
}

static std::vector<std::string> extractKernelNames(mlir::gpu::GPUModuleOp op) {
    std::vector<std::string> kernels;
    op.walk([&](mlir::gpu::GPUFuncOp f) {
        if (auto n = f.getNameAttr()) {
            kernels.push_back(n.getValue().str());
        }
    });

    std::sort(kernels.begin(), kernels.end());
    kernels.erase(std::unique(kernels.begin(), kernels.end()), kernels.end());
    return kernels;
}

static void addManifest(mlir::MLIRContext* ctx,
                        llvm::SmallVectorImpl<mlir::Attribute>& outAttrs,
                        const std::string& moduleKey,
                        std::uint32_t moduleKind,
                        const std::string& blob,
                        const std::vector<std::string>& kernels) {
    llvm::SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back({mlir::StringAttr::get(ctx, "module_key"), mlir::StringAttr::get(ctx, moduleKey)});
    fields.push_back({mlir::StringAttr::get(ctx, "module_kind"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), moduleKind)});
    fields.push_back({mlir::StringAttr::get(ctx, "blob"), bytesToDenseI8(ctx, blob)});
    fields.push_back({mlir::StringAttr::get(ctx, "kernels"), stringsToArrayAttr(ctx, kernels)});
    outAttrs.push_back(mlir::DictionaryAttr::get(ctx, fields));
}

// -----------------------------------------------------------------------------
// ABI Guard
// -----------------------------------------------------------------------------
static bool isArkSliceStruct(mlir::Type t) {
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(t);
    if (!st || st.isOpaque()) {
        return false;
    }

    auto body = st.getBody();
    if (body.size() != 2) {
        return false;
    }

    const bool a0Ptr = llvm::isa<mlir::LLVM::LLVMPointerType>(body[0]);
    const auto iTy = llvm::dyn_cast<mlir::IntegerType>(body[1]);
    const bool a1I64 = iTy && iTy.getWidth() == 64;

    return a0Ptr && a1I64;
}

static bool verifyGpuKernelAbi(mlir::ModuleOp module) {
    bool ok = true;

    module.walk([&](mlir::gpu::GPUFuncOp f) {
        if (!f.isKernel()) {
            return;
        }

        for (auto it : llvm::enumerate(f.getArgumentTypes())) {
            if (!isArkSliceStruct(it.value())) {
                continue;
            }

            f.emitError()
                << "invalid GPU kernel ABI: arg#" << it.index()
                << " is !llvm.struct<(ptr,i64)>; Ark kernels must take raw base pointers "
                   "and any sizes must be explicit scalar params";
            ok = false;
        }
    });

    return ok;
}

// -----------------------------------------------------------------------------
// GPU Lowering
// -----------------------------------------------------------------------------
static mlir::amdgpu::Chipset resolveAmdChipset(llvm::StringRef arch) {
    llvm::StringRef candidate = arch.empty() ? llvm::StringRef("gfx90a") : arch;

    if (auto chipsetOr = mlir::amdgpu::Chipset::parse(candidate); mlir::succeeded(chipsetOr)) {
        return *chipsetOr;
    }

    if (candidate != "gfx90a") {
        if (auto fallbackOr = mlir::amdgpu::Chipset::parse("gfx90a"); mlir::succeeded(fallbackOr)) {
            return *fallbackOr;
        }
    }

    return mlir::amdgpu::Chipset();
}

template <typename Backend>
struct ArkGpuLoweringPass
    : public mlir::PassWrapper<ArkGpuLoweringPass<Backend>, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArkGpuLoweringPass<Backend>)

    ArkGpuLoweringPass() = default;
    explicit ArkGpuLoweringPass(std::string backendTargetCpu)
        : backendTargetCpu_(std::move(backendTargetCpu)) {}

    void runOnOperation() override {
        mlir::ModuleOp module = this->getOperation();
        const mlir::DataLayoutAnalysis dataLayoutAnalysis(module);
        mlir::MLIRContext& ctx = this->getContext();

        mlir::LLVMConversionTarget target(ctx);
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addLegalOp<mlir::ModuleOp>();
        target.addLegalOp<mlir::gpu::GPUModuleOp>();
        target.addIllegalDialect<mlir::gpu::GPUDialect>();

        Backend::configureLegality(target);

        mlir::LowerToLLVMOptions options(&ctx);
        options.useBarePtrCallConv = true;

        mlir::LLVMTypeConverter converter(&ctx, options, &dataLayoutAnalysis);
        Backend::configureTypeConverter(converter);

        mlir::RewritePatternSet patterns(&ctx);

        ::mlir::populateGpuToLLVMConversionPatterns(
            converter,
            patterns,
            /*kernelBarePtrCallConv=*/options.useBarePtrCallConv,
            /*kernelIntersperseSizeCallConv=*/false
        );

        ::mlir::populateVectorToLLVMConversionPatterns(converter, patterns);
        ::mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
        ::mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
        ::mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
        ::mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
        ::mlir::index::populateIndexToLLVMConversionPatterns(converter, patterns);
        ::mlir::ub::populateUBToLLVMConversionPatterns(converter, patterns);

        Backend::populate(converter, patterns, backendTargetCpu_);

        if (mlir::failed(mlir::applyPartialConversion(this->getOperation(), target, std::move(patterns)))) {
            this->signalPassFailure();
        }
    }

private:
    std::string backendTargetCpu_;
};

struct PopulateNVVM {
    static void configureLegality(mlir::ConversionTarget& target) {
        ::mlir::configureGpuToNVVMConversionLegality(target);
    }

    static void configureTypeConverter(mlir::LLVMTypeConverter& converter) {
        ::mlir::configureGpuToNVVMTypeConverter(converter);
    }

    static void populate(mlir::LLVMTypeConverter& converter,
                         mlir::RewritePatternSet& patterns,
                         llvm::StringRef) {
        ::mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
    }
};

struct PopulateROCDL {
    static void configureLegality(mlir::ConversionTarget& target) {
        ::mlir::configureGpuToROCDLConversionLegality(target);
    }

    static void configureTypeConverter(mlir::LLVMTypeConverter&) {}

    static void populate(mlir::LLVMTypeConverter& converter,
                         mlir::RewritePatternSet& patterns,
                         llvm::StringRef backendTargetCpu) {
        const mlir::amdgpu::Chipset chipset = resolveAmdChipset(backendTargetCpu);
        ::mlir::populateGpuToROCDLConversionPatterns(
            converter,
            patterns,
            mlir::gpu::amd::Runtime::HIP,
            chipset
        );
    }
};

// -----------------------------------------------------------------------------
// GPU Compilation Helpers
// -----------------------------------------------------------------------------
static llvm::Expected<std::string>
compileToPtx(mlir::gpu::GPUModuleOp gpuMod,
             const std::string& arch,
             const std::string& feats) {
    initLlvmTargetsOnce();

    if (!gpuMod) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument), "null gpu.module");
    }

    mlir::MLIRContext* ctx = gpuMod->getContext();
    prepareCompilerContext(*ctx);

    mlir::ModuleOp tmp = mlir::ModuleOp::create(gpuMod.getLoc());
    tmp->setAttr(
        mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
        mlir::StringAttr::get(
            ctx,
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
        )
    );

    auto clonedGpuMod = llvm::cast<mlir::gpu::GPUModuleOp>(gpuMod.clone());
    tmp.getBody()->push_back(clonedGpuMod);

    mlir::PassManager pm(ctx);
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createSCFToControlFlowPass());
    pm.addPass(std::make_unique<ArkGpuLoweringPass<PopulateNVVM>>(arch));
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(tmp))) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                       "GPU->NVVM lowering pipeline failed");
    }

    mlir::Block* tmpBody = tmp.getBody();
    mlir::Block* gpuBody = clonedGpuMod.getBody();
    while (!gpuBody->empty()) {
        mlir::Operation& op = gpuBody->front();
        op.remove();
        tmpBody->push_back(&op);
    }
    clonedGpuMod.erase();

    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(tmp, llvmCtx);
    if (!llvmModule) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                       "MLIR->LLVM translation failed");
    }

    llvm::Triple triple("nvptx64-nvidia-cuda");
    std::string err;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
    if (!target) {
        return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                       "NVPTX target not available: " + err);
    }

    auto tm = makeTargetMachine(target, triple, arch, feats);
    if (!tm) {
        return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                       "failed to create NVPTX TargetMachine");
    }

    setLlvmModuleTargetTriple(*llvmModule, triple);
    llvmModule->setDataLayout(tm->createDataLayout());

    llvm::SmallString<0> buf;
    llvm::raw_svector_ostream os(buf);
    llvm::legacy::PassManager passes;

    if (tm->addPassesToEmitFile(passes, os, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        return llvm::createStringError(std::make_error_code(std::errc::not_supported), "emit PTX failed");
    }

    passes.run(*llvmModule);
    return std::string(buf.data(), buf.size());
}

static llvm::Expected<std::string>
compileToHsaco(mlir::gpu::GPUModuleOp gpuMod,
               const std::string& arch,
               const std::string& feats) {
    initLlvmTargetsOnce();

    if (!gpuMod) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument), "null gpu.module");
    }

    mlir::MLIRContext* ctx = gpuMod->getContext();
    prepareCompilerContext(*ctx);

    mlir::ModuleOp tmp = mlir::ModuleOp::create(gpuMod.getLoc());
    tmp->setAttr(
        mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
        mlir::StringAttr::get(
            ctx,
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-"
            "i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-"
            "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
        )
    );

    auto clonedGpuMod = llvm::cast<mlir::gpu::GPUModuleOp>(gpuMod.clone());
    tmp.getBody()->push_back(clonedGpuMod);

    mlir::PassManager pm(ctx);
    pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createSCFToControlFlowPass());
    pm.addPass(std::make_unique<ArkGpuLoweringPass<PopulateROCDL>>(arch));
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(tmp))) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                       "GPU->ROCDL lowering pipeline failed");
    }

    mlir::Block* tmpBody = tmp.getBody();
    mlir::Block* gpuBody = clonedGpuMod.getBody();
    while (!gpuBody->empty()) {
        mlir::Operation& op = gpuBody->front();
        op.remove();
        tmpBody->push_back(&op);
    }
    clonedGpuMod.erase();

    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(tmp, llvmCtx);
    if (!llvmModule) {
        return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                       "MLIR->LLVM translation failed");
    }

    llvm::Triple triple("amdgcn-amd-amdhsa");
    std::string err;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
    if (!target) {
        return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                       "AMDGPU target not available: " + err);
    }

    auto tm = makeTargetMachine(target, triple, arch, feats);
    if (!tm) {
        return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                       "failed to create AMDGPU TargetMachine");
    }

    setLlvmModuleTargetTriple(*llvmModule, triple);
    llvmModule->setDataLayout(tm->createDataLayout());

    llvm::SmallString<0> buf;
    llvm::raw_svector_ostream os(buf);
    llvm::legacy::PassManager passes;

    if (tm->addPassesToEmitFile(passes, os, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        return llvm::createStringError(std::make_error_code(std::errc::not_supported), "emit HSACO object failed");
    }

    passes.run(*llvmModule);
    return std::string(buf.data(), buf.size());
}

// -----------------------------------------------------------------------------
// Host-Side Lowering
// -----------------------------------------------------------------------------
struct ArklangToLLVMPass
    : public mlir::PassWrapper<ArklangToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArklangToLLVMPass)

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        const mlir::DataLayoutAnalysis dataLayoutAnalysis(module);

        mlir::LLVMConversionTarget target(getContext());
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addLegalOp<mlir::ModuleOp>();

        mlir::LowerToLLVMOptions options(&getContext());
        options.useBarePtrCallConv = true;

        mlir::LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
        mlir::RewritePatternSet patterns(&getContext());

        mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
        mlir::populateVectorToLLVMConversionPatterns(converter, patterns);
        mlir::index::populateIndexToLLVMConversionPatterns(converter, patterns);
        mlir::ub::populateUBToLLVMConversionPatterns(converter, patterns);

        arklang::populateGpuLoweringPatterns(converter, patterns);

        if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

Compiler::Compiler(arklang::hud::Hud& hud,
                   arklang::ModuleRegistry& registry,
                   const CompilerConfig& config)
    : hud_(hud), registry_(registry), cfg_(config) {}

bool Compiler::compileToMLIR(llvm::StringRef inputFilename,
                             mlir::MLIRContext& ctx,
                             mlir::ModuleOp& outModule) {
    prepareCompilerContext(ctx);

    auto modOr = registry_.load(inputFilename.str(), "");
    if (!modOr) {
        hud_.error(llvm::toString(modOr.takeError()));
        return false;
    }

    mlir::OpBuilder builder(&ctx);
    outModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    outModule->setAttr("ark.build.unit", mlir::StringAttr::get(&ctx, inputFilename));

    arklang::GenMIR genMir(outModule, builder, hud_);

    llvm::DenseSet<arklang::Module*> visited;
    std::vector<arklang::Module*> allModules;
    collectDependencies(*modOr, visited, allModules);

    for (auto* m : allModules) {
        genMir.registerModule(*m, (m == *modOr));
    }

    for (auto* m : allModules) {
        genMir.clearImports();
        for (const auto& imp : m->imports) {
            if (m->submodules.count(imp->alias)) {
                genMir.registerImport(imp->alias, m->submodules[imp->alias]);
            }
        }

        if (mlir::failed(genMir.compileModule(*m, (m == *modOr)))) {
            return false;
        }
    }

    mlir::PassManager pm(&ctx);
    pm.addNestedPass<mlir::func::FuncOp>(arklang::mir::createOwnershipVerifierPass());
    pm.addNestedPass<mlir::func::FuncOp>(arklang::mir::createDropInsertionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    return mlir::succeeded(pm.run(outModule)) && mlir::succeeded(mlir::verify(outModule));
}

bool Compiler::lowerToLLVM(mlir::MLIRContext& ctx, mlir::ModuleOp module) {
    prepareCompilerContext(ctx);

    {
        mlir::PassManager pm(&ctx);
        pm.addPass(arklang::createGpuKernelOutliningPass());
        if (mlir::failed(pm.run(module))) {
            return false;
        }
    }

    {
        mlir::PassManager pm(&ctx);
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuAsyncRegionPass());
        if (mlir::failed(pm.run(module))) {
            return false;
        }
    }

    if (!verifyGpuKernelAbi(module)) {
        return false;
    }

    std::string unit = "unknown";
    if (auto u = module->getAttrOfType<mlir::StringAttr>("ark.build.unit")) {
        unit = u.getValue().str();
    }

    llvm::SmallVector<mlir::gpu::GPUModuleOp, 8> gpuMods;
    module.walk([&](mlir::gpu::GPUModuleOp op) { gpuMods.push_back(op); });

    llvm::SmallVector<mlir::Attribute, 8> moduleAttrs;

    for (mlir::gpu::GPUModuleOp op : gpuMods) {
        const std::string modName = op.getName().str();
        const std::vector<std::string> kernels = extractKernelNames(op);

        bool producedAny = false;

        if (cfg_.enableCudaPtx) {
            auto ptxOr = compileToPtx(op, cfg_.cudaArch, cfg_.cudaFeatures);
            if (ptxOr) {
                addManifest(
                    &ctx,
                    moduleAttrs,
                    unit + "::" + modName + ":nvptx",
                    kArkGpuModuleCudaPtx,
                    *ptxOr,
                    kernels
                );
                producedAny = true;
            } else {
                hud_.error("CUDA PTX compilation failed: " + llvm::toString(ptxOr.takeError()));
            }
        }

        if (cfg_.enableHipHsaco) {
            auto hsacoOr = compileToHsaco(op, cfg_.hipArch, cfg_.hipFeatures);
            if (hsacoOr) {
                addManifest(
                    &ctx,
                    moduleAttrs,
                    unit + "::" + modName + ":amdgpu",
                    kArkGpuModuleHipHsaco,
                    *hsacoOr,
                    kernels
                );
                producedAny = true;
            } else {
                hud_.error("HIP HSACO compilation failed: " + llvm::toString(hsacoOr.takeError()));
            }
        }

        if (cfg_.enableMetal && cfg_.metalUseSourceBlob) {
            if (auto msl = op->getAttrOfType<mlir::StringAttr>("ark.metal.msl")) {
                addManifest(
                    &ctx,
                    moduleAttrs,
                    unit + "::" + modName + ":metal",
                    kArkGpuModuleSrcMsl,
                    msl.getValue().str(),
                    kernels
                );
                producedAny = true;
            }
        }

        if (!producedAny && (cfg_.enableCudaPtx || cfg_.enableHipHsaco || cfg_.enableMetal)) {
            op.emitError("no GPU backend produced an artifact for this module");
            return false;
        }

        op.erase();
    }

    if (!moduleAttrs.empty()) {
        module->setAttr("ark.gpu.modules", mlir::ArrayAttr::get(&ctx, moduleAttrs));
    }

    {
        mlir::PassManager pm(&ctx);
        pm.addPass(ark::compiler::transforms::createRuntimeABIVerifierPass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(std::make_unique<ArklangToLLVMPass>());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (mlir::failed(pm.run(module))) {
            return false;
        }
    }

    return true;
}

std::vector<CompiledGpuModule> Compiler::extractCompiledGpuModules(mlir::ModuleOp module) {
    std::vector<CompiledGpuModule> out;

    auto arr = module->getAttrOfType<mlir::ArrayAttr>("ark.gpu.modules");
    if (!arr) {
        return out;
    }

    for (mlir::Attribute a : arr) {
        auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(a);
        auto keyA  = llvm::dyn_cast_or_null<mlir::StringAttr>(dict.get("module_key"));
        auto kindA = llvm::dyn_cast_or_null<mlir::IntegerAttr>(dict.get("module_kind"));
        auto blobA = llvm::dyn_cast_or_null<mlir::DenseIntElementsAttr>(dict.get("blob"));
        auto kernA = llvm::dyn_cast_or_null<mlir::ArrayAttr>(dict.get("kernels"));
        if (!keyA || !kindA || !blobA || !kernA) {
            continue;
        }

        std::string blob;
        blob.reserve(blobA.getNumElements());
        for (auto v : blobA.getValues<llvm::APInt>()) {
            blob.push_back(static_cast<char>(v.getZExtValue()));
        }

        std::vector<std::string> kernels;
        for (mlir::Attribute ka : kernA) {
            if (auto ks = llvm::dyn_cast<mlir::StringAttr>(ka)) {
                kernels.push_back(ks.getValue().str());
            }
        }

        out.push_back({
            keyA.getValue().str(),
            static_cast<uint32_t>(kindA.getInt()),
            std::move(blob),
            std::move(kernels)
        });
    }

    module->removeAttr("ark.gpu.modules");
    return out;
}

bool Compiler::writeModuleToFile(mlir::ModuleOp module, llvm::StringRef path) {
    std::string osError;
    auto outFile = mlir::openOutputFile(path, &osError);
    if (!outFile) {
        return false;
    }

    module.print(outFile->os());
    outFile->keep();
    return true;
}

} // namespace ark::compiler::pipeline