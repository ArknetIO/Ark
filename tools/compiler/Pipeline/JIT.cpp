#include "Pipeline/JIT.h"

// MLIR Execution Engine
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// LLVM Support
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h" // [FIX] Required for consumeError

// System
#include <csetjmp>
#include <csignal>
#include <cstdio>

namespace arklang { 
    // Forward declaration for the runtime symbol registration hook 
    // (Defined in Runtime/SymbolRegistry.cpp)
    void registerRuntimeSymbols(); 
}

namespace ark::compiler::pipeline {

// -----------------------------------------------------------------------------
// Panic Handling (Signal/Setjmp)
// -----------------------------------------------------------------------------

static thread_local sigjmp_buf g_panic_env;

// This function matches the signature expected by the Ark Runtime's panic hook.
static void arkcPanicHandler(const char *msg, const char *file, int32_t line, int32_t col) {
    fprintf(stderr, "\n[ArkRuntime] Intercepted Panic:\n");
    fprintf(stderr, "  Msg:  %s\n", msg);
    fprintf(stderr, "  Loc:  %s:%d:%d\n", file, line, col);
    siglongjmp(g_panic_env, 1);
}

// -----------------------------------------------------------------------------
// JIT Implementation
// -----------------------------------------------------------------------------

int JIT::Run(mlir::ModuleOp module, llvm::StringRef runtimeDir) {
    (void)runtimeDir; 

    // 1. Initialize Targets (Required for CodeGen)
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // 2. Register Host Runtime Symbols
    // This makes functions like 'ark_print' and 'ark_alloc' visible to the JIT.
    arklang::registerRuntimeSymbols();

    // 3. Register Translations
    // Required to convert MLIR LLVM Dialect -> LLVM IR
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // 4. Configure Engine
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    
    // transformer: Optional optimization pipeline for the LLVM module before JIT
    engineOptions.transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr
    );

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "Failed to construct JIT ExecutionEngine: " 
                     << llvm::toString(maybeEngine.takeError()) << "\n";
        return 1;
    }
    auto &engine = maybeEngine.get();

    // 5. Hook Panic Handler
    // Look for the runtime's hook function and inject our handler.
    // [FIX] We verify the Expected<> result. If missing, we MUST consume error.
    auto panicSym = engine->lookup("arkSetPanicHandler");
    if (panicSym) {
        using SetPanicFn = void (*)(void (*)(const char*, const char*, int32_t, int32_t));
        auto setPanicPtr = reinterpret_cast<SetPanicFn>(panicSym.get());
        setPanicPtr(arkcPanicHandler);
    } else {
        // [CRITICAL] Consume the error if symbol is not found (e.g. script doesn't use panic).
        // If we don't do this, LLVM destructor will crash the compiler.
        llvm::consumeError(panicSym.takeError());
    }

    // 6. Execute Main
    if (sigsetjmp(g_panic_env, 1) == 0) {
        auto mainSym = engine->lookup("main");
        
        if (!mainSym) {
            // [CRITICAL] Consume error here too if main is missing
            llvm::consumeError(mainSym.takeError());
            llvm::errs() << "JIT entrypoint 'main' not found in module.\n";
            return 1;
        }
        
        using MainFn = int (*)();
        auto mainPtr = reinterpret_cast<MainFn>(mainSym.get());
        
        // --- RUNNING USER CODE ---
        return mainPtr();
        // -------------------------
    }

    // 7. Panic Recovery
    llvm::outs() << "Program panicked safely (JIT session ended).\n";
    return 1; // Non-zero exit on panic
}

} // namespace ark::compiler::pipeline