#include "ark/compiler/Frontend/GenMIR.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Utils.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Runtime.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Types.hpp"

#include "ark/IR/ArkMirOps.h"
#include "ark/IR/ArkMirTypes.h" 

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" 
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h" 
#include "mlir/IR/Verifier.h" 
#include "mlir/Dialect/MemRef/IR/MemRef.h" 
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/SCF/IR/SCF.h" // [ADD THIS at the top]

// Add near the top of GenMIR.cpp (or a small local header used by it)
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>


#include <iostream>

using namespace mlir;
using namespace arklang;
using namespace arklang::mir;



// =============================================================================
// Vector/Runtime Helpers
// =============================================================================

// Helper: Declare Runtime Function safely
static mlir::LLVM::LLVMFuncOp getOrDeclareVecPush(mlir::ModuleOp module, 
                                                  mlir::OpBuilder &b, 
                                                  mlir::Location loc,
                                                  mlir::Type vecPtrTy, 
                                                  mlir::Type dataPtrTy, 
                                                  mlir::Type ipTy) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_vec_push")) 
        return fn;

    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(module.getBody());

    auto voidTy = mlir::LLVM::LLVMVoidType::get(b.getContext());
    // Signature: void __ark_vec_push(Vec* vec, void* data, intptr_t size)
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(
        voidTy, {vecPtrTy, dataPtrTy, ipTy}, /*isVarArg=*/false);
        
    return b.create<mlir::LLVM::LLVMFuncOp>(loc, "__ark_vec_push", fnTy);
}

// Helper to Create/Get LLVM Function
static mlir::LLVM::LLVMFuncOp getOrCreateFunc(mlir::ModuleOp module, 
                                              llvm::StringRef name, 
                                              llvm::ArrayRef<mlir::Type> args, 
                                              llvm::ArrayRef<mlir::Type> rets, 
                                              bool isPublic = false) {
    if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) return fn;

    mlir::OpBuilder b(module.getBodyRegion());
    auto retTy = rets.empty() ? mlir::LLVM::LLVMVoidType::get(module.getContext()) : rets[0];
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(retTy, args, false);
    
    auto fn = b.create<mlir::LLVM::LLVMFuncOp>(b.getUnknownLoc(), name, fnTy);
    if (isPublic) fn.setLinkage(mlir::LLVM::Linkage::External);
    else fn.setLinkage(mlir::LLVM::Linkage::Private);
    return fn;
}

static SourceLoc toSourceLoc(mlir::Location loc) { return SourceLoc{}; }


namespace arklang {

GenMIR::GenMIR(mlir::ModuleOp m, mlir::OpBuilder &b, arklang::hud::Hud &hud_ref)
    : hud(hud_ref), module(m), builder(b) {

    convertTypeCb = [this](const arklang::Type &t) -> mlir::Type {
        return this->convertType(t);
    };
    coerceCb = [this](mlir::Location l, mlir::Value v, mlir::Type t) -> mlir::Value {
        return this->coerce(this->builder, l, v, t);
    };
    isCopyTypeCb = [this](const arklang::Type &t) -> bool {
        return this->isCopyType(t);
    };

    // [ADD THIS] Register intrinsics so getExprType can use them
    registerIntrinsics(intrinsicRegistry);

    mir = std::make_unique<MirBuilder>(
        builder,
        module,
        convertTypeCb,
        coerceCb,
        isCopyTypeCb
    );

    builtinLowering = std::make_unique<frontend::BuiltinNsLowering>(module, builder);

    globalFunctionMap.clear();
    funcParamNames.clear();
    schemaRegistry.clear();
    importedModules.clear();
}


Location GenMIR::toLoc(SourceLoc loc) { return FileLineColLoc::get(builder.getStringAttr(loc.file), loc.line, loc.col); }

LogicalResult GenMIR::fail(SourceLoc loc, const llvm::Twine &msg) {
    mlir::emitError(toLoc(loc)) << msg;
    return failure();
}

// =============================================================================
// Modules
// =============================================================================

// [NEW] Helper to resolve the correct symbol name
std::string GenMIR::mangleFunction(const std::string& name, const Module* mod) {
    if (!mod) return name;
    auto it = modulePrefixes.find(mod);
    if (it == modulePrefixes.end()) return name; // Should not happen if registered
    
    const std::string& prefix = it->second;
    if (prefix.empty()) return name; // Root module: "main" -> "main"
    return prefix + "_" + name;      // Imported: "main" -> "generics_main"
}

// =============================================================================
// Module Registration
// =============================================================================
// Responsibilities:
// - assign/stash the module prefix used for symbol mangling
// - register functions into the global function registry
// - register schemas into the global schema registry
// =============================================================================
void GenMIR::registerModule(const Module& astMod, bool isRoot) {
    // 1. Determine and store prefix.
    // Root module gets empty prefix. Imports get their module id.
    std::string prefix = isRoot ? "" : astMod.id;
    modulePrefixes[&astMod] = prefix;

    // 2. Index functions.
    for (const auto& fn : astMod.functions) {
        std::string mangledName = prefix.empty() ? fn->name : prefix + "_" + fn->name;

        if (globalFunctionMap.count(mangledName)) {
            llvm::errs() << "[ArkC] Warning: Overwriting symbol '" << mangledName << "'\n";
        }

        globalFunctionMap[mangledName] = fn.get();

        std::vector<std::string> params;
        for (auto& param : fn->args) {
            params.push_back(param.first);
        }
        funcParamNames[mangledName] = std::move(params);
    }

    // 3. Index schemas.
    registerModuleSchemas(astMod);
}

// =============================================================================
// 1. Compile Module
// =============================================================================
// Responsibilities:
// - establish current AST/module context
// - pre-materialize singleton schemas as globals
// - lower all functions in the module
// =============================================================================
mlir::LogicalResult GenMIR::compileModule(const Module& astMod, bool isRoot) {
    this->astModule = &astMod;
    this->currentContextPrefix = isRoot ? "" : astMod.id;

    // A. Pre-pass: lower singleton schemas into internal LLVM globals.
    for (const auto& decl : astMod.schemas) {
        if (!decl->isSingleton) {
            continue;
        }

        if (failed(materializeSingletonSchema(astMod, *decl))) {
            return failure();
        }
    }

    // B. Compile functions.
    for (const auto& fn : astMod.functions) {
        if (failed(lowerFunction(*fn))) {
            return failure();
        }
    }

    return success();
}

// [NEW] Import Visibility Helpers
void GenMIR::registerImport(const std::string& alias, const Module* mod) {
    importedModules[alias] = mod;
}

void GenMIR::clearImports() {
    importedModules.clear();
}

// [Legacy Wrapper] 
// Used by tests or single-file compilation. ensures registration happens.
LogicalResult GenMIR::lowerModule(const Module &astMod) {
    // 1. Register Symbols (Header Scan)
    // Populates globalFunctionMap, globalSchemaMap, and funcParamNames
    registerModule(astMod);

    // 2. Compile
    if (failed(compileModule(astMod))) return failure();

    // 3. Final Verification
    if (failed(mlir::verify(module))) {
        return fail(SourceLoc{}, "Final module verification failed");
    }
    
    return success();
}


// =============================================================================
// Capability Injection
// Handles implicit resources (FS, NET, IO, SYS) based on function effects.
// =============================================================================
void GenMIR::injectCapabilities(const Function &fn) {
    mlir::Location loc = toLoc(fn.loc);
    
    // We treat capabilities as "Handles" (i64) in the runtime.
    // This allows passing them around if needed, even if they are just tokens.
    arklang::Type capType;
    capType.kind = arklang::Type::I64; 

    // Helper to inject a capability into the current scope
    auto inject = [&](const std::string &name) {
        // Create a dummy handle value (0)
        // In a real OS integration, this might be a process ID or file descriptor table index.
        mlir::Value handle = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
        
        // Register:
        // 1. Allocates a stack slot (place)
        // 2. Stores the handle (0)
        // 3. Registers variable "name" in the scope
        // 4. Sets state to "Alive" (true)
        mir->declareLocal(loc, name, capType, RValue{handle, unitAlive(builder, loc).state});
    };

    // Inject based on the defined effect constants
    if (fn.effects & EFF_FS)  inject("FS");
    if (fn.effects & EFF_NET) inject("NET");
    if (fn.effects & EFF_IO)  inject("IO");
    if (fn.effects & EFF_SYS) inject("SYS");
}

// =============================================================================
// Function Lowering Dispatcher
// =============================================================================
mlir::LogicalResult GenMIR::lowerFunction(const Function &fn) {
    switch (fn.domain) {
        case Domain::GPU: 
            return lowerGpuKernel(fn);
        
        case Domain::CPU: 
            return lowerCpuKernel(fn);

        case Domain::Host: 
        default:          
            return lowerHostFunction(fn);
    }
}

// =============================================================================
// Standard Host Function Generation
// Handles: main(), regular functions, etc.
// =============================================================================
mlir::LogicalResult GenMIR::lowerHostFunction(const Function &fn) {
    // 1. Reset State via MirBuilder
    mir->enterFunction();
    mlir::Location loc = toLoc(fn.loc);

    bool isMain = (fn.name == "main");

    // 2. Convert Signature (Force standard C ABI for main)
    llvm::SmallVector<mlir::Type, 4> argTypes;
    if (isMain) {
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        argTypes.push_back(builder.getI32Type()); // argc
        argTypes.push_back(ptrTy);                // argv
        argTypes.push_back(ptrTy);                // envp
    } else {
        for (const auto &argPair : fn.args) {
            argTypes.push_back(convertType(argPair.second));
        }
    }

    mlir::Type retType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    if (fn.returnType.kind != arklang::Type::Void) {
        retType = convertType(fn.returnType);
    }

    // 3. Mangle Name
    std::string funcName = mangleFunction(fn.name, astModule);

    // 4. Create Function Op
    auto funcType = mlir::LLVM::LLVMFunctionType::get(retType, argTypes, false);
    
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(loc, funcName, funcType);
    
    // 5. Attributes
    if (isMain) {
        funcOp->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    }

    auto domainStr = [&]() -> llvm::StringRef {
        switch (fn.domain) {
            case Domain::GPU: return "gpu";
            case Domain::CPU: return "cpu";
            default:          return "host";
        }
    }();
    funcOp->setAttr("ark.domain", builder.getStringAttr(domainStr));

    // 6. Setup Entry Block & Bind Arguments
    mlir::Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    if (isMain) {
        // Initialize POSIX SYS kernel immediately on process startup!
        auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
        auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto i32Ty = builder.getI32Type();
        
        auto initFn = getOrDeclRuntimeFn(module, builder, loc, "__ark_sys_process_init", voidTy, {i32Ty, ptrTy, ptrTy});
        
        mlir::Value argc = entryBlock->getArgument(0);
        mlir::Value argv = entryBlock->getArgument(1);
        mlir::Value envp = entryBlock->getArgument(2);
        
        builder.create<mlir::LLVM::CallOp>(loc, initFn, mlir::ValueRange{argc, argv, envp});
    } else {
        for (size_t i = 0; i < fn.args.size(); ++i) {
            mlir::Value argVal = entryBlock->getArgument(i);
            const std::string &argName = fn.args[i].first;
            const arklang::Type &argType = fn.args[i].second;

            mir->declareLocal(loc, argName, argType, 
                              RValue{argVal, unitAlive(builder, loc).state});
        }
    }

    // 7. Lower Body
    for (const auto &stmt : fn.body) {
        if (failed(lowerStmt(*stmt))) return failure();
        
        if (!builder.getBlock()->empty() &&
            builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>())
            break;
    }

    // 8. Implicit Return
    // [FIX] Check the CURRENT block where the builder is sitting, NOT the entry block!
    mlir::Block *currentBlock = builder.getBlock();
    if (currentBlock && (currentBlock->empty() || !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())) {
        if (llvm::isa<mlir::LLVM::LLVMVoidType>(retType)) {
             builder.create<mlir::LLVM::ReturnOp>(toLoc(fn.loc), mlir::ValueRange{});
        } else if (isMain) {
             mlir::Value retZ = builder.create<mlir::LLVM::ConstantOp>(
                toLoc(fn.loc), builder.getI32Type(), builder.getI32IntegerAttr(0));
             builder.create<mlir::LLVM::ReturnOp>(toLoc(fn.loc), mlir::ValueRange{retZ});
        } else {
             builder.create<mlir::LLVM::UnreachableOp>(toLoc(fn.loc));
        }
    }

    // 9. Cleanup
    mir->exitFunction();
    return success();
}


// =============================================================================
// CPU Kernel Generation (The Loop Wrapper)
// =============================================================================
mlir::LogicalResult GenMIR::lowerCpuKernel(const Function &fn) {
    mir->enterFunction();
    mlir::Location loc = toLoc(fn.loc);
    std::string name = mangleFunction(fn.name, astModule);

    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto i64Ty = builder.getI64Type();
    auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    // Kernel ABI: (void* grid, void* args, i64 dim)
    auto funcTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, i64Ty}, false);

    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, funcTy);
    
    // [Safe Mode] Manually create Entry Block (Compatible with all LLVM/MLIR versions)
    mlir::Block *entryBlock = new mlir::Block();
    funcOp.getBody().push_back(entryBlock);
    for (mlir::Type t : funcTy.getParams()) {
        entryBlock->addArgument(t, loc);
    }
    
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value gridPtrRaw = entryBlock->getArgument(0);
    mlir::Value argsPtrRaw = entryBlock->getArgument(1);
    mlir::Value dimVal     = entryBlock->getArgument(2);

    // 1. Prepare Argument Types
    llvm::SmallVector<mlir::Type, 4> packedArgTypes;
    for (size_t i = 1; i < fn.args.size(); ++i) {
        packedArgTypes.push_back(convertType(fn.args[i].second));
    }
    
    mlir::Type structTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), packedArgTypes);
    mlir::Value argsPtr = castToExpectedPtr(builder, loc, argsPtrRaw, 
        mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    // 2. Setup Loop
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(0));
    mlir::Value one  = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(1));

    mlir::Block *condBlock = funcOp.addBlock();
    mlir::Block *bodyBlock = funcOp.addBlock();
    mlir::Block *endBlock  = funcOp.addBlock();

    // [FIX] BrOp: (Operands, Destination)
    // Pass 'zero' to initialize the block argument in condBlock
    builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange(zero), condBlock);
    
    // -- Condition: i < dim
    builder.setInsertionPointToStart(condBlock);
    mlir::Value idx = condBlock->addArgument(i64Ty, loc);

    mlir::Value cmp = builder.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, idx, dimVal);
    
    // [FIX] CondBrOp: (Condition, TrueDest, TrueArgs, FalseDest, FalseArgs)
    builder.create<mlir::LLVM::CondBrOp>(
        loc, 
        cmp, 
        bodyBlock, mlir::ValueRange(), // True Path
        endBlock,  mlir::ValueRange()  // False Path
    );

    // -- Body: Execute Logic for A[i]
    builder.setInsertionPointToStart(bodyBlock);

    // Load Grid Element A[i]
    mlir::Type gridElemTy = convertType(fn.args[0].second);
    mlir::Value elemPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, voidPtrTy, gridElemTy, gridPtrRaw, mlir::ValueRange(idx)
    ).getResult();
    mlir::Value elemVal = builder.create<mlir::LLVM::LoadOp>(loc, gridElemTy, elemPtr);

    // Bind Arguments to Scope
    mir->declareLocal(loc, fn.args[0].first, fn.args[0].second, RValue{elemVal, unitAlive(builder, loc).state});

    // Packed Args: Load from struct
    if (!packedArgTypes.empty()) {
        mlir::Value zero32 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
        for (size_t i = 0; i < packedArgTypes.size(); ++i) {
            mlir::Value fieldIdx = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(i));
            mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
                loc, voidPtrTy, structTy, argsPtr, mlir::ValueRange{zero32, fieldIdx}
            ).getResult();
            mlir::Value argVal = builder.create<mlir::LLVM::LoadOp>(loc, packedArgTypes[i], fieldPtr);
            
            mir->declareLocal(loc, fn.args[i+1].first, fn.args[i+1].second, RValue{argVal, unitAlive(builder, loc).state});
        }
    }

    // Lower User Body Stmts
    for (const auto &stmt : fn.body) {
        if (stmt->kind == ExprKind::Return) {
             auto* retStmt = static_cast<const ReturnStmt*>(stmt.get());
             if (retStmt->value) {
                 auto retRes = lowerExpr(*retStmt->value);
                 if (succeeded(retRes)) {
                     builder.create<mlir::LLVM::StoreOp>(loc, retRes->val, elemPtr);
                 }
             }
        } else {
             lowerStmt(*stmt);
        }
    }

    // Loop Increment
    mlir::Value nextIdx = builder.create<mlir::LLVM::AddOp>(loc, idx, one);
    
    // [FIX] BrOp: (Operands, Destination)
    // Pass nextIdx back to the condBlock argument
    builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange(nextIdx), condBlock);

    // -- End
    builder.setInsertionPointToStart(endBlock);

    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange());

    mir->exitFunction();
    return success();
}


// =============================================================================
// Vector Method Lowering
// Handles: vec.push(val), vec.len, vec.cap
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerVectorMethod(const MemberCallNode &expr,
                                                  const VarInfo &var,
                                                  const arklang::Type &elemAstTy) { // [FIX] Takes AST Type
    mlir::Location loc = toLoc(expr.loc);

    // 1. Strict AST Validation
    if (var.astTy.kind != arklang::Type::Vec)
        return fail(expr.loc, "Vector method called on non-Vec type");

    // -------------------------------------------------------------------------
    // Method: push(item)
    // -------------------------------------------------------------------------
    if (expr.methodName == "push") {
        if (expr.args.size() != 1) return fail(expr.loc, "push expects 1 argument");

        // A. Lower Argument to RValue
        auto argRv = lowerExpr(*expr.args[0].value);
        if (failed(argRv)) return failure();

        // B. Calculate Element Size
        mlir::Type ipTy = mir->getIntPtrType();
        mlir::DataLayout dl(module);
        
        // Convert AST Type -> LLVM Type
        mlir::Type llvmElemTy = convertType(elemAstTy); 
        const uint64_t bits = dl.getTypeSizeInBits(llvmElemTy);
        
        if (bits == 0 && !llvm::isa<mlir::LLVM::LLVMVoidType>(llvmElemTy)) 
            return fail(expr.loc, "Cannot compute element size (incomplete type?)");
            
        const uint64_t bytes = (bits + 7) / 8;

        // C. Spill Argument to Stack
        //    Runtime needs a pointer to the data to memcpy it.
        mlir::Value itemVal = coerce(builder, loc, argRv->val, llvmElemTy);
        mlir::Value itemAddr = mir->spillTemp(loc, llvmElemTy, itemVal);

        // D. Prepare Call Arguments
        mlir::Value sizeVal = builder.create<mlir::LLVM::ConstantOp>(
            loc, ipTy, builder.getIntegerAttr(ipTy, bytes));

        auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
        auto pushFn = getOrDeclareVecPush(module, builder, loc, voidPtrTy, voidPtrTy, ipTy);
        auto fnTy = pushFn.getFunctionType();

        mlir::Value vecArg  = castPtrTo(builder, loc, var.place,  fnTy.getParamType(0));
        mlir::Value dataArg = castPtrTo(builder, loc, itemAddr,   fnTy.getParamType(1));

        // E. Call Runtime
        builder.create<mlir::LLVM::CallOp>(
            loc, 
            mlir::TypeRange{}, 
            mlir::SymbolRefAttr::get(pushFn), 
            mlir::ValueRange{vecArg, dataArg, sizeVal}
        );

        return unitAlive(builder, loc);
    }

    return fail(expr.loc, "Unknown Vec method: " + expr.methodName);
}

// =============================================================================
// Argument Preparation & Reordering (Named -> Positional)
// =============================================================================
mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> GenMIR::prepareCallArgs(
    SourceLoc astLoc, // [FIX] Accept SourceLoc for error reporting
    mlir::LLVM::LLVMFuncOp callee, 
    const std::vector<CallArg> &args,
    const std::string &funcName) 
{
    // [FIX] Convert to MLIR Location for Builder ops
    mlir::Location irLoc = toLoc(astLoc);

    // 1. Fetch Function Signature from AST (Source of Truth for Names)
    const Function* astFunc = nullptr;
    if (astModule) {
        for (const auto &fn : astModule->functions) {
            if (fn->name == funcName) {
                astFunc = fn.get();
                break;
            }
        }
    }

    size_t paramCount = callee.getFunctionType().getParams().size();
    llvm::SmallVector<mlir::Value, 8> orderedArgs(paramCount);
    std::vector<bool> paramSet(paramCount, false);

    // 2. Process Arguments
    for (size_t i = 0; i < args.size(); ++i) {
        auto valRes = lowerExpr(*args[i].value);
        if (failed(valRes)) return failure();
        RValue argRv = *valRes;

        size_t targetIdx = -1;

        if (args[i].name.empty()) {
            targetIdx = i;
        } else {
            if (!astFunc) {
                return fail(astLoc, "Named arguments not supported for extern/unknown function '" + funcName + "'");
            }
            
            bool found = false;
            for (size_t p = 0; p < astFunc->args.size(); ++p) {
                if (astFunc->args[p].first == args[i].name) {
                    targetIdx = p;
                    found = true;
                    break;
                }
            }
            if (!found) return fail(astLoc, "Unknown named argument '" + args[i].name + "' for function '" + funcName + "'");
        }

        if (targetIdx >= paramCount) {
            return fail(astLoc, "Argument index out of bounds");
        }
        if (paramSet[targetIdx]) {
            return fail(astLoc, "Duplicate argument for parameter index " + std::to_string(targetIdx));
        }

        // 3. Coerce and Store
        mlir::Type expectedTy = callee.getFunctionType().getParams()[targetIdx];
        mlir::Value finalVal = argRv.val;

        // Auto-Load
        if (llvm::isa<mlir::LLVM::LLVMPointerType>(finalVal.getType()) && !llvm::isa<mlir::LLVM::LLVMPointerType>(expectedTy)) {
            finalVal = builder.create<mlir::LLVM::LoadOp>(irLoc, expectedTy, finalVal);
        }
        
        finalVal = coerce(builder, irLoc, finalVal, expectedTy);

        orderedArgs[targetIdx] = finalVal;
        paramSet[targetIdx] = true;
    }

    // 4. Verification
    for (size_t i = 0; i < paramCount; ++i) {
        if (!paramSet[i]) return fail(astLoc, "Missing argument for parameter " + std::to_string(i));
    }

    return orderedArgs;
}

// =============================================================================
// Member Call Lowering (Robust Dispatch)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerMemberCall(const MemberCallNode &expr) {
    mlir::Location loc = toLoc(expr.loc);

    // 1. Resolve Base Object Name
    std::string baseName;
    if (auto sym = dynamic_cast<const SymbolExpr*>(expr.object.get())) {
        baseName = sym->name;
    } else if (auto mem = dynamic_cast<const MemberExpr*>(expr.object.get())) {
        if (auto baseSym = dynamic_cast<const SymbolExpr*>(mem->object.get())) {
            baseName = baseSym->name + "." + mem->member; // Resolves "SYS" + "env" -> "SYS.env"
        }
    }

    // -------------------------------------------------------------------------
    // Case 0: Builtin Namespace Methods (Intercepted Here!)
    // -------------------------------------------------------------------------
    if (!baseName.empty() && isBuiltinNamespace(baseName)) {
        llvm::SmallVector<mlir::Value, 4> loweredArgs;
        for (const auto &arg : expr.args) {
            auto valOr = lowerExpr(*arg.value);
            if (failed(valOr)) return failure();
            loweredArgs.push_back(valOr->val);
        }
        return lowerBuiltin(loc, baseName, expr.methodName, loweredArgs);
    }

    if (baseName.empty()) {
        return fail(expr.loc, "Complex member calls (chained) not supported yet");
    }

    // -------------------------------------------------------------------------
    // Case 1: Local Variable Methods (Vec, Slice, etc.)
    // -------------------------------------------------------------------------
    if (mir->isDeclared(baseName)) {
        VarInfo *var = mir->lookup(baseName);
        if (var->astTy.kind == arklang::Type::Vec) {
            if (var->astTy.genericArgs.empty()) return fail(expr.loc, "Vector missing generic type info");
            return lowerVectorMethod(expr, *var, var->astTy.genericArgs[0]);
        }
    }

    // -------------------------------------------------------------------------
    // Case 2: Enum Constructors (Variant Creation)
    // -------------------------------------------------------------------------
    const SchemaDecl* schema = resolveSchemaAST(baseName);
    if (schema && schema->kind == SchemaDecl::Enum) {
        int tag = -1;
        std::vector<arklang::Type> payloadTypes;
        for (size_t i = 0; i < schema->variants.size(); ++i) {
            if (schema->variants[i].name == expr.methodName) {
                tag = (int)i;
                payloadTypes = schema->variants[i].tuplePayload;
                break;
            }
        }
        if (tag != -1) return lowerVariantConstructor(expr, schema, tag, payloadTypes);
        return fail(expr.loc, "Unknown variant '" + expr.methodName + "' in enum '" + baseName + "'");
    }

    // -------------------------------------------------------------------------
    // Case 3: Imported Module Functions
    // -------------------------------------------------------------------------
    if (importedModules.count(baseName)) {
        const Module* mod = importedModules[baseName];
        bool found = false;
        for (const auto& fn : mod->functions) {
            if (fn->name == expr.methodName) { found = true; break; }
        }
        if (!found) return fail(expr.loc, "Function '" + expr.methodName + "' not found in module '" + baseName + "'");

        std::string mangledName = mangleFunction(expr.methodName, mod);
        auto calleeFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(mangledName);
        if (!calleeFn) return fail(expr.loc, "Linker error: Symbol '" + mangledName + "' not found");

        auto argsOr = prepareCallArgs(expr.loc, calleeFn, expr.args, expr.methodName);
        if (failed(argsOr)) return failure();

        auto callOp = builder.create<mlir::LLVM::CallOp>(
            loc, calleeFn.getFunctionType().getReturnType(), 
            mlir::SymbolRefAttr::get(builder.getContext(), calleeFn.getName()), *argsOr
        );

        mlir::Value res = (callOp.getNumResults() > 0) ? callOp.getOperation()->getResult(0) : unitAlive(builder, loc).val;
        return RValue{res, unitAlive(builder, loc).state};
    }

    // Fallback: This is the exact error you saw!
    return fail(expr.loc, "Unknown identifier '" + baseName + "' (not a variable, enum, or module)");
}

// =============================================================================
// Helper: Resolve Static Enum Base (Color.Red / Mod.Color.Red)
// =============================================================================
std::string GenMIR::resolveStaticEnumBase(const Expr &obj) {
    // Case 1: Simple Symbol (Color)
    if (auto sym = dynamic_cast<const SymbolExpr*>(&obj)) return sym->name;

    // Case 2: Imported Symbol (Mod.Color)
    if (auto mem = dynamic_cast<const MemberExpr*>(&obj)) {
        if (auto sub = dynamic_cast<const SymbolExpr*>(mem->object.get())) {
            // Now valid: accessing non-static member 'importedModules'
            if (importedModules.count(sub->name)) {
                return sub->name + "." + mem->member;
            }
        }
    }
    return {};
}


// =============================================================================
// L-Value Member Access (Address Calculation)
// Used by assignments: x.field = ...
// Returns: A Pointer (Value) to the field in memory.
// =============================================================================
mlir::FailureOr<mlir::Value> GenMIR::lowerMemberPlace(const MemberExpr &expr) {
    mlir::Location loc = toLoc(expr.loc);

    // 1. Lower Base as Place (Must be addressable memory)
    //    This recurses (e.g., x.y.z -> x.y address -> x address)
    //    For a Singleton 'Config.port', this calls lowerExprAsPlace on 'Config',
    //    which returns the @Config Global Variable address.
    auto basePlaceOr = lowerExprAsPlace(*expr.object);
    if (failed(basePlaceOr)) return failure();
    mlir::Value basePtr = *basePlaceOr;

    // 2. Validate Base Type
    //    We need the AST type to know which Schema to look up.
    arklang::Type baseTy = getExprType(*expr.object);

    if (baseTy.kind != arklang::Type::Schema && baseTy.kind != arklang::Type::Generic) {
        return fail(expr.loc, "Member assignment requires struct/enum place");
    }

    // 3. Resolve Schema Info
    //    This gets the LLVM struct layout and field indices.
    const SchemaInfo *info = getOrInstantiateSchema(baseTy.schemaName, baseTy.genericArgs);
    if (!info) return fail(expr.loc, "Unknown schema type: " + baseTy.schemaName);

    // 4. Resolve Field Index
    auto it = info->fieldIndices.find(expr.member);
    if (it == info->fieldIndices.end())
        return fail(expr.loc, "Field '" + expr.member + "' not found in " + baseTy.schemaName);

    const int64_t fieldIdx = it->second;

    // 5. Emit GEP (Get Element Pointer)
    //    We calculate the offset from the BasePtr to the specific Field.
    
    // Pointer Type for the result (Field*)
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    
    // Indices for GEP: [0, fieldIdx]
    // 0 = Dereference the pointer to the struct
    // fieldIdx = Select the member
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0));

    mlir::Value idx = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(fieldIdx));

    // Create GEP
    // We pass 'info->loweredType' (the LLVM Struct Type) so GEP knows the stride.
    auto gep = builder.create<mlir::LLVM::GEPOp>(
        loc, 
        ptrTy,             // Result type
        info->loweredType, // Element type (Struct)
        basePtr,           // Base pointer
        mlir::ValueRange{zero, idx} // Indices
    );

    return gep.getResult();
}


// =============================================================================
// R-Value Member Access (Unified Read)
// Handles:
// - builtin namespace properties
// - static enum constructors
// - builtin container/tuple properties
// - schema field access / enum reflection
// Returns:
// - RValue (value + state)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerMemberAccess(const MemberExpr& expr) {
    mlir::Location loc = toLoc(expr.loc);

    // -------------------------------------------------------------------------
    // 0. Builtin Namespace Properties (SYS.args, FS.xxx)
    // -------------------------------------------------------------------------
    std::string baseName;
    if (auto* objSym = dynamic_cast<const SymbolExpr*>(expr.object.get())) {
        baseName = objSym->name;
    } else if (auto* objMem = dynamic_cast<const MemberExpr*>(expr.object.get())) {
        if (auto* baseSym = dynamic_cast<const SymbolExpr*>(objMem->object.get())) {
            baseName = baseSym->name + "." + objMem->member;
        }
    }

    if (!baseName.empty() && isBuiltinNamespace(baseName)) {
        return lowerBuiltin(loc, baseName, expr.member, {});
    }

    // -------------------------------------------------------------------------
    // 1. Static Enum Access (e.g. Color.Red)
    // -------------------------------------------------------------------------
    if (auto enumRes = lowerStaticEnumAccess(loc, expr); succeeded(enumRes)) {
        return enumRes;
    }

    // -------------------------------------------------------------------------
    // 2. Lower Base Object
    // -------------------------------------------------------------------------
    auto baseRvOr = lowerExpr(*expr.object);
    if (failed(baseRvOr)) {
        return failure();
    }

    RValue base = *baseRvOr;
    arklang::Type baseTy = getExprType(*expr.object);

    // -------------------------------------------------------------------------
    // 3. Introspection (.type)
    // -------------------------------------------------------------------------
    if (expr.member == "type") {
        std::string name = astTypeToString(baseTy);
        return lowerString(StringExpr(expr.loc, name));
    }

    // -------------------------------------------------------------------------
    // 4. Builtin Properties (.len / .cap / .ok / .value)
    // -------------------------------------------------------------------------
    if (baseTy.kind == arklang::Type::Vec ||
        baseTy.kind == arklang::Type::Slice ||
        baseTy.kind == arklang::Type::Str) {
        int field = -1;

        if (expr.member == "len") {
            field = 1;
        } else if (expr.member == "cap" && baseTy.kind == arklang::Type::Vec) {
            field = 2;
        }

        if (field != -1) {
            if (llvm::isa<mlir::LLVM::LLVMPointerType>(base.val.getType())) {
                mlir::Type structTy = convertType(baseTy);
                base.val = builder.create<mlir::LLVM::LoadOp>(loc, structTy, base.val);
            }

            auto ex = builder.create<mlir::LLVM::ExtractValueOp>(
                loc,
                base.val,
                builder.getDenseI64ArrayAttr({static_cast<int64_t>(field)})
            );
            return RValue{ex.getResult(), base.state};
        }
    }

    if (baseTy.kind == arklang::Type::Tuple) {
        int field = -1;

        if (expr.member == "value" || expr.member == "0") {
            field = 0;
        } else if (expr.member == "ok" || expr.member == "1") {
            field = 1;
        }

        if (field != -1) {
            if (llvm::isa<mlir::LLVM::LLVMPointerType>(base.val.getType())) {
                mlir::Type structTy = convertType(baseTy);
                base.val = builder.create<mlir::LLVM::LoadOp>(loc, structTy, base.val);
            }

            auto ex = builder.create<mlir::LLVM::ExtractValueOp>(
                loc,
                base.val,
                builder.getDenseI64ArrayAttr({static_cast<int64_t>(field)})
            );
            return RValue{ex.getResult(), base.state};
        }
    }

    // -------------------------------------------------------------------------
    // 5. Schema Access (fields / enum reflection)
    // -------------------------------------------------------------------------
    if (auto schemaRes = lowerSchemaMemberAccess(loc, expr, base, baseTy); succeeded(schemaRes)) {
        return schemaRes;
    }

    return fail(expr.loc,
                "Member '" + expr.member + "' not found on type " + astTypeToString(baseTy));
}


// =============================================================================
// Launch Lowering (Async Kernel Dispatch)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerLaunch(const LaunchExpr &ln) {
    mlir::Location loc = toLoc(ln.loc);
    
    mlir::Type ipTy = builder.getI64Type(); 
    mlir::Type i64Ty = builder.getI64Type();
    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::Value alive = constBool(builder, loc, true);

    // -------------------------------------------------------------------------
    // 1. Resolve Kernel Symbol & Signature
    // -------------------------------------------------------------------------
    std::string linkage = ln.kernelName;
    const Function* astKernelFn = nullptr;

    if (auto dot = ln.kernelName.find('.'); dot != std::string::npos) {
        std::string mod = ln.kernelName.substr(0, dot);
        std::string fn = ln.kernelName.substr(dot + 1);
        if (importedModules.count(mod)) {
            linkage = mangleFunction(fn, importedModules[mod]);
        }
    } else if (astModule) {
        linkage = mangleFunction(ln.kernelName, astModule);
        for(const auto& f : astModule->functions) {
            if(f->name == ln.kernelName) { astKernelFn = f.get(); break; }
        }
    }
    
    auto kernelFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(linkage);
    if (!kernelFn) return fail(ln.loc, "Unknown kernel: " + linkage);

    // -------------------------------------------------------------------------
    // 2. Dispatch: Direct GPU Stub vs Packed CPU Task
    // -------------------------------------------------------------------------
    // If the host stub signature expects the exact number of arguments provided,
    // it is a GPU kernel stub. CPU tasks expect a single packed struct pointer.
    bool isDirectCall = (kernelFn.getFunctionType().getNumParams() == ln.args.size());

    if (isDirectCall) {
        // =====================================================================
        // PATH A: GPU KERNEL (Direct Asynchronous Call)
        // =====================================================================
        llvm::SmallVector<mlir::Value, 8> directArgs;
        
        // Pass ALL arguments directly, no skipping!
        for (size_t i = 0; i < ln.args.size(); ++i) {
            auto res = lowerExpr(*ln.args[i].value);
            if (failed(res)) return failure();
            
            mlir::Value val = res->val;
            mlir::Type expectedTy = kernelFn.getFunctionType().getParamType(i);
            
            if (val.getType() != expectedTy) {
                val = coerce(builder, loc, val, expectedTy);
            }
            directArgs.push_back(val);
        }
        
        // Emit direct asynchronous call to the GPU Host Stub
        builder.create<mlir::LLVM::CallOp>(loc, kernelFn, directArgs);
        
        // GPU launches use the driver stream, so return dummy token (0) for await
        mlir::Value tokenVal = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(0));
        
        if (!ln.tokenName.empty()) {
            VarInfo* existing = mir->isDeclared(ln.tokenName) ? mir->lookup(ln.tokenName) : nullptr;
            if (existing) mir->writeVar(loc, *existing, RValue{tokenVal, alive});
            else mir->declareLocal(loc, ln.tokenName, {arklang::Type::I64}, RValue{tokenVal, alive});
        }
        
        return RValue{tokenVal, alive};

    } else {
        // =====================================================================
        // PATH B: CPU TASK (Packed Environment Struct)
        // =====================================================================
        if (!mir->isDeclared(ln.destVar)) return fail(ln.loc, "Undefined launch dest");
        VarInfo* destInfo = mir->lookup(ln.destVar);
        RValue gridRv = mir->readVar(loc, *destInfo);
        mlir::Value gridVal = castToExpectedPtr(builder, loc, gridRv.val, voidPtrTy);

        mlir::Value gridDim = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(32));
        if (auto allocOp = gridRv.val.getDefiningOp<AllocOp>()) {
            auto dynamicSizes = allocOp.getSizes();
            if (!dynamicSizes.empty()) {
                mlir::Value sz = dynamicSizes[0];
                if (sz.getType() != i64Ty) sz = builder.create<mlir::LLVM::ZExtOp>(loc, i64Ty, sz);
                gridDim = sz;
            }
        }

        mlir::Value kernelPtr = builder.create<mlir::LLVM::AddressOfOp>(
            loc, voidPtrTy, mlir::SymbolRefAttr::get(builder.getContext(), linkage)
        ).getResult();

        uint64_t kernelUid = 0xcbf29ce484222325; 
        for (char c : linkage) { kernelUid ^= (uint64_t)c; kernelUid *= 0x100000001b3; }
        
        mlir::Value uidLoVal = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(kernelUid));
        mlir::Value uidHiVal = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(0));

        mlir::Value argsPtr;
        mlir::Value argsSize;

        if (ln.args.size() <= 1) { 
            argsPtr = builder.create<mlir::LLVM::ZeroOp>(loc, voidPtrTy);
            argsSize = builder.create<mlir::LLVM::ConstantOp>(loc, ipTy, builder.getIntegerAttr(ipTy, 0));
        } else {
            llvm::SmallVector<mlir::Type, 8> argTys;
            llvm::SmallVector<mlir::Value, 8> argVals;

            // CPU tasks skip the first arg (grid) and pack the rest
            for (size_t i = 1; i < ln.args.size(); ++i) {
                auto res = lowerExpr(*ln.args[i].value);
                if (failed(res)) return failure();
                mlir::Value val = res->val;
                argVals.push_back(val);
                argTys.push_back(val.getType());
            }

            mlir::Type packedTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), argTys);
            mlir::DataLayout dl(module);
            uint64_t bytes = (dl.getTypeSizeInBits(packedTy) + 7) / 8;

            mlir::Value packedSlot = mir->createSlot(loc, packedTy);
            mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));

            for (unsigned i = 0; i < argVals.size(); ++i) {
                mlir::Value idx = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(i));
                mlir::Value fieldPtr = builder.create<mlir::LLVM::GEPOp>(
                    loc, voidPtrTy, packedTy, packedSlot, mlir::ValueRange{zero, idx}
                ).getResult();
                builder.create<mlir::LLVM::StoreOp>(loc, argVals[i], fieldPtr);
            }

            argsPtr = castToExpectedPtr(builder, loc, packedSlot, voidPtrTy);
            argsSize = builder.create<mlir::LLVM::ConstantOp>(loc, ipTy, builder.getIntegerAttr(ipTy, bytes));
        }

        auto launchFn = getOrDeclareArkLaunch(module, builder, loc);
        mlir::Type expVoidPtr = launchFn.getFunctionType().getParamType(0);
        mlir::Type expSizeTy  = launchFn.getFunctionType().getParamType(5);

        mlir::Value configArg = builder.create<mlir::LLVM::ZeroOp>(loc, voidPtrTy);

        auto call = builder.create<mlir::LLVM::CallOp>(
            loc, mlir::TypeRange{i64Ty}, 
            mlir::SymbolRefAttr::get(builder.getContext(), "__ark_launch"),
            mlir::ValueRange{
                castToExpectedPtr(builder, loc, gridVal, expVoidPtr), 
                castToExpectedPtr(builder, loc, kernelPtr, expVoidPtr), 
                uidLoVal, uidHiVal, 
                castToExpectedPtr(builder, loc, argsPtr, expVoidPtr), 
                (argsSize.getType() != expSizeTy) ? builder.create<mlir::LLVM::ZExtOp>(loc, expSizeTy, argsSize).getResult() : argsSize, 
                gridDim, configArg 
            }
        );

        mlir::Value tokenVal = call.getOperation()->getResult(0); 

        if (!ln.tokenName.empty()) {
            VarInfo* existing = mir->isDeclared(ln.tokenName) ? mir->lookup(ln.tokenName) : nullptr;
            if (existing) mir->writeVar(loc, *existing, RValue{tokenVal, alive});
            else mir->declareLocal(loc, ln.tokenName, {arklang::Type::I64}, RValue{tokenVal, alive});
        }

        return RValue{tokenVal, alive};
    }
}

// =============================================================================
// Tuple Lowering
// Layout: Anonymous Struct { T1, T2... }
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerTuple(const TupleExpr &expr) {
    mlir::Location loc = toLoc(expr.loc);
    llvm::SmallVector<mlir::Value, 4> values;
    llvm::SmallVector<mlir::Type, 4> types;

    // 1. Lower all elements to RValues
    mlir::Value currentState = unitAlive(builder, loc).state; // Start alive

    for (const auto &elem : expr.elements) {
        auto resOr = lowerExpr(*elem);
        if (failed(resOr)) return failure();
        
        // Merge state? For now, we assume sequential evaluation keeps state linear.
        // A better approach accumulates state tokens, but simplified RValue works here.
        values.push_back(resOr->val);
        types.push_back(resOr->val.getType());
        currentState = resOr->state; // Update current state token
    }

    // 2. Create LLVM Struct Type
    auto structTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), types);

    // 3. Construct Struct (undef -> insertvalue)
    mlir::Value result = builder.create<mlir::LLVM::UndefOp>(loc, structTy);

    for (size_t i = 0; i < values.size(); ++i) {
        result = builder.create<mlir::LLVM::InsertValueOp>(
            loc, 
            result, 
            values[i], 
            builder.getDenseI64ArrayAttr({(int64_t)i})
        );
    }

    return RValue{result, currentState};
}

// =============================================================================
// Alloc Lowering (Heap Allocation)
// Logic: alloc<T>(N) @tag -> malloc(N * sizeof(T)) -> cast(addrspace)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerAlloc(const AllocExpr &allocExpr, const std::vector<CallArg> &args) {
    mlir::Location loc = toLoc(allocExpr.loc);
    mlir::Type ipTy = mir->getIntPtrType();
    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    if (args.size() != 1) 
        return fail(allocExpr.loc, "'alloc' expects exactly 1 argument (element count)");

    auto countRv = lowerExpr(*args[0].value);
    if (failed(countRv)) return failure();
    mlir::Value countVal = coerce(builder, loc, countRv->val, ipTy);

    mlir::Type elemType = convertType(allocExpr.type); 
    mlir::DataLayout dl(module);
    
    uint64_t elemBits = dl.getTypeSizeInBits(elemType);
    if (elemBits == 0 && !llvm::isa<mlir::LLVM::LLVMVoidType>(elemType)) {
        elemBits = 8; 
    }
    uint64_t elemBytes = (elemBits + 7) / 8;

    mlir::Value sizeBytes = builder.create<mlir::LLVM::MulOp>(
        loc, countVal, 
        builder.create<mlir::LLVM::ConstantOp>(loc, ipTy, builder.getIntegerAttr(ipTy, elemBytes))
    );

    bool isGpuAlloc = allocExpr.location.find("gpu") != std::string::npos;
    mlir::Value rawPtr;

    if (isGpuAlloc) {
        // --- GPU Managed Allocation Route ---
        if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_gpu_alloc_managed")) {
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidPtrTy, {ipTy}, false);
            builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "__ark_gpu_alloc_managed", fnTy);
        }
        
        auto callOp = builder.create<mlir::LLVM::CallOp>(
            loc, voidPtrTy, mlir::SymbolRefAttr::get(builder.getContext(), "__ark_gpu_alloc_managed"), mlir::ValueRange{sizeBytes});
        rawPtr = callOp.getOperation()->getResult(0);

        auto gpuPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 1);
        mlir::Value gpuPtr = builder.create<mlir::LLVM::AddrSpaceCastOp>(loc, gpuPtrTy, rawPtr);
        return RValue{gpuPtr, unitAlive(builder, loc).state};

    } else {
        // --- Standard Host CPU Allocation Route ---
        if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc")) {
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(module.getBody());
            auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidPtrTy, {ipTy}, false);
            builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "malloc", fnTy);
        }
        
        auto callOp = builder.create<mlir::LLVM::CallOp>(
            loc, voidPtrTy, mlir::SymbolRefAttr::get(builder.getContext(), "malloc"), mlir::ValueRange{sizeBytes});
        rawPtr = callOp.getOperation()->getResult(0);
        
        return RValue{rawPtr, unitAlive(builder, loc).state};
    }
}

// =============================================================================
// Await Lowering (Runtime Sync)
// Logic: await token_name -> __ark_await(handle)  // handle type follows runtime decl
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerAwait(const AwaitExpr &aw) {
    mlir::Location loc = toLoc(aw.loc);

    mlir::Type i64Ty = builder.getI64Type();
    mlir::Type i32Ty = builder.getI32Type();
    mlir::Type i1Ty  = builder.getI1Type();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    if (!mir->isDeclared(aw.tokenName)) {
        return fail(aw.loc, "Unknown await token: " + aw.tokenName);
    }

    VarInfo* var = mir->lookup(aw.tokenName);
    if (!var) return fail(aw.loc, "Internal error: Symbol declared but lookup failed");

    RValue tokRv = mir->readVar(loc, *var);
    mlir::Value tokVal = tokRv.val;

    struct AwaitCallee {
        mlir::FlatSymbolRefAttr sym;
        mlir::Type paramTy;
    };

    auto getOrCreateAwait = [&]() -> mlir::FailureOr<AwaitCallee> {
        if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_await")) {
            auto fnTy = fn.getFunctionType();
            if (fnTy.getNumParams() != 1 || fnTy.getReturnType() != i32Ty) {
                mlir::emitError(loc) << "__ark_await has wrong type; expected i32(<T>), got " << fnTy;
                return mlir::failure();
            }
            return AwaitCallee{
                mlir::FlatSymbolRefAttr::get(builder.getContext(), "__ark_await"),
                fnTy.getParamType(0)
            };
        }

        mlir::Type chosenParamTy = tokVal.getType();
        if (chosenParamTy == ptrTy || chosenParamTy == i64Ty ||
            llvm::isa<mlir::LLVM::LLVMStructType>(chosenParamTy)) {
        } else {
            chosenParamTy = ptrTy;
        }

        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());

        auto fnTy = mlir::LLVM::LLVMFunctionType::get(i32Ty, {chosenParamTy}, /*isVarArg=*/false);
        builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "__ark_await", fnTy);

        return AwaitCallee{
            mlir::FlatSymbolRefAttr::get(builder.getContext(), "__ark_await"),
            chosenParamTy
        };
    };

    auto adaptTo = [&](mlir::Value v, mlir::Type targetTy) -> mlir::FailureOr<mlir::Value> {
        if (v.getType() == targetTy) return v;

        const bool vIsPtr = llvm::isa<mlir::LLVM::LLVMPointerType>(v.getType());
        const bool tIsPtr = llvm::isa<mlir::LLVM::LLVMPointerType>(targetTy);

        if (auto tSt = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(targetTy)) {
            if (vIsPtr) {
                return builder.create<mlir::LLVM::LoadOp>(loc, targetTy, v).getResult();
            }
        }

        if (tIsPtr) {
            if (auto vSt = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(v.getType())) {
                if (!vSt.isOpaque() && vSt.getBody().size() >= 1 && vSt.getBody()[0] == ptrTy) {
                    auto idx0 = builder.getDenseI64ArrayAttr({0});
                    return builder.create<mlir::LLVM::ExtractValueOp>(loc, ptrTy, v, idx0).getResult();
                }
            }
            if (v.getType() == i64Ty) {
                return builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrTy, v).getResult();
            }
        }

        if (targetTy == i64Ty) {
            if (vIsPtr) {
                return builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Ty, v).getResult();
            }
            if (auto vSt = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(v.getType())) {
                if (!vSt.isOpaque() && vSt.getBody().size() >= 1 && vSt.getBody()[0] == ptrTy) {
                    auto idx0 = builder.getDenseI64ArrayAttr({0});
                    mlir::Value p = builder.create<mlir::LLVM::ExtractValueOp>(loc, ptrTy, v, idx0).getResult();
                    return builder.create<mlir::LLVM::PtrToIntOp>(loc, i64Ty, p).getResult();
                }
            }
        }

        mlir::emitError(loc) << "Cannot adapt await token type " << v.getType()
                             << " to __ark_await param type " << targetTy;
        return mlir::failure();
    };

    auto calleeOr = getOrCreateAwait();
    if (mlir::failed(calleeOr)) return mlir::failure();

    auto argOr = adaptTo(tokVal, calleeOr->paramTy);
    if (mlir::failed(argOr)) return mlir::failure();

    auto call = builder.create<mlir::LLVM::CallOp>(
        loc,
        mlir::TypeRange{i32Ty},
        calleeOr->sym,
        mlir::ValueRange{*argOr}
    );

    mlir::Value st = call.getResults().front();
    mlir::Value alive = builder.create<mlir::LLVM::ConstantOp>(loc, i1Ty, builder.getBoolAttr(true));

    return RValue{st, alive};
}


// =============================================================================
// String Literal Lowering
// Returns: String Slice { i8* ptr, intptr len }
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerString(const StringExpr &expr) {
    mlir::Location loc = toLoc(expr.loc);
    mlir::Type ipTy = mir->getIntPtrType();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    // 1. Get Pointer to Global Data (i8*)
    mlir::Value dataPtr = getOrCreateGlobalString(loc, builder, module, expr.value);

    // 2. Create Length Constant
    mlir::Value lenVal = builder.create<mlir::LLVM::ConstantOp>(
        loc, ipTy, builder.getIntegerAttr(ipTy, expr.value.size()));

    // 3. Construct Struct { i8*, intptr }
    //    (Matches the Slice<u8> layout used for Strings)
    auto strStructTy = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {ptrTy, ipTy});

    mlir::Value strSlice = builder.create<mlir::LLVM::UndefOp>(loc, strStructTy);
    
    strSlice = builder.create<mlir::LLVM::InsertValueOp>(
        loc, strSlice, dataPtr, builder.getDenseI64ArrayAttr({0}));
        
    strSlice = builder.create<mlir::LLVM::InsertValueOp>(
        loc, strSlice, lenVal, builder.getDenseI64ArrayAttr({1}));

    return RValue{strSlice, unitAlive(builder, loc).state};
}

// =============================================================================
// Generic Literal Lowering
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerLiteral(const LiteralExpr &lit) {
    mlir::Location loc = toLoc(lit.loc);

    // Handle String Literals (Forwarding)
    // Check if type is Str OR if value is quoted (fallback parsing)
    if (lit.type.kind == arklang::Type::Str || 
       (lit.value.size() >= 2 && lit.value.front() == '"')) {
        
        std::string content = lit.value;
        if (content.front() == '"' && content.back() == '"') {
            content = content.substr(1, content.size() - 2);
        }
        return lowerString(StringExpr{lit.loc, content});
    }

    mlir::Value val;
    mlir::Type type;

    // Use MLIR LLVM Dialect Constants
    try {
        switch (lit.type.kind) {
            case arklang::Type::I32:
                type = builder.getI32Type();
                val = builder.create<mlir::LLVM::ConstantOp>(
                    loc, type, builder.getI32IntegerAttr(std::stoi(lit.value)));
                break;

            case arklang::Type::I64:
                type = builder.getI64Type();
                val = builder.create<mlir::LLVM::ConstantOp>(
                    loc, type, builder.getI64IntegerAttr(std::stoll(lit.value)));
                break;

            case arklang::Type::F32:
                type = builder.getF32Type();
                val = builder.create<mlir::LLVM::ConstantOp>(
                    loc, type, builder.getF32FloatAttr(std::stof(lit.value)));
                break;

            case arklang::Type::F64:
                type = builder.getF64Type();
                val = builder.create<mlir::LLVM::ConstantOp>(
                    loc, type, builder.getF64FloatAttr(std::stod(lit.value)));
                break;

            case arklang::Type::Bool:
                type = builder.getI1Type();
                val = builder.create<mlir::LLVM::ConstantOp>(
                    loc, type, builder.getBoolAttr(lit.value == "true"));
                break;

            default:
                return fail(lit.loc, "Unsupported literal type kind: " + std::to_string(lit.type.kind));
        }
    } catch (const std::exception &e) {
        return fail(lit.loc, "Failed to parse literal '" + lit.value + "': " + e.what());
    }

    return RValue{val, unitAlive(builder, loc).state};
}

// =============================================================================
// Binary Expression Lowering (Strict LLVM Dialect)
// Handles: Primitives, String Concatenation, Enum Equality
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerBinary(const BinaryExpr &bin) {
    mlir::Location loc = toLoc(bin.loc);

    auto lhsRes = lowerExpr(*bin.lhs);
    auto rhsRes = lowerExpr(*bin.rhs);
    if (failed(lhsRes) || failed(rhsRes)) return failure();

    mlir::Value lhs = lhsRes->val;
    mlir::Value rhs = rhsRes->val;
    mlir::Value state = rhsRes->state;

    arklang::Type lTy = getExprType(*bin.lhs);
    arklang::Type rTy = getExprType(*bin.rhs);

    auto isUnsignedAstInt = [](const arklang::Type& t) -> bool {
        switch (t.kind) {
            case arklang::Type::U8:
            case arklang::Type::U16:
            case arklang::Type::U32:
            case arklang::Type::U64:
                return true;
            default:
                return false;
        }
    };

    auto isSignedAstInt = [](const arklang::Type& t) -> bool {
        switch (t.kind) {
            case arklang::Type::I8:
            case arklang::Type::I16:
            case arklang::Type::I32:
            case arklang::Type::I64:
                return true;
            default:
                return false;
        }
    };

    auto isFloatLike = [](mlir::Type t) -> bool {
        return llvm::isa<mlir::FloatType>(t);
    };

    if (lTy.kind == arklang::Type::Str || rTy.kind == arklang::Type::Str) {
        mlir::Type strTy = convertType(arklang::Type{arklang::Type::Str});

        if (bin.op == "+") {
            if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_str_concat")) {
                mlir::OpBuilder::InsertionGuard g(builder);
                builder.setInsertionPointToStart(module.getBody());
                auto fnTy = mlir::LLVM::LLVMFunctionType::get(strTy, {strTy, strTy}, false);
                builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "__ark_str_concat", fnTy);
            }

            auto fn = mlir::SymbolRefAttr::get(builder.getContext(), "__ark_str_concat");
            auto call = builder.create<mlir::LLVM::CallOp>(loc, strTy, fn, mlir::ValueRange{lhs, rhs});
            return RValue{call.getOperation()->getResult(0), state};
        }

        if (bin.op == "==" || bin.op == "!=") {
            if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__ark_str_eq")) {
                mlir::OpBuilder::InsertionGuard g(builder);
                builder.setInsertionPointToStart(module.getBody());
                auto fnTy = mlir::LLVM::LLVMFunctionType::get(builder.getI1Type(), {strTy, strTy}, false);
                builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "__ark_str_eq", fnTy);
            }

            auto fn = mlir::SymbolRefAttr::get(builder.getContext(), "__ark_str_eq");
            auto call = builder.create<mlir::LLVM::CallOp>(loc, builder.getI1Type(), fn, mlir::ValueRange{lhs, rhs});

            mlir::Value res = call.getOperation()->getResult(0);
            if (bin.op == "!=") {
                mlir::Value trueVal = builder.create<mlir::LLVM::ConstantOp>(
                    loc,
                    builder.getI1Type(),
                    builder.getIntegerAttr(builder.getI1Type(), 1)
                );
                res = builder.create<mlir::LLVM::XOrOp>(loc, res, trueVal);
            }

            return RValue{res, state};
        }

        return fail(bin.loc, "Unsupported string operator: " + bin.op);
    }

    bool isEnum = false;
    if ((lTy.kind == arklang::Type::Schema || lTy.kind == arklang::Type::Generic) &&
        (lTy.schemaName == rTy.schemaName)) {
        const SchemaDecl *decl = resolveSchemaAST(lTy.schemaName);
        if (decl && decl->kind == SchemaDecl::Enum) {
            isEnum = true;
        }
    }

    if (isEnum && (bin.op == "==" || bin.op == "!=")) {
        mlir::Value lTag = builder.create<mlir::LLVM::ExtractValueOp>(
            loc, lhs, builder.getDenseI64ArrayAttr({0})
        ).getResult();

        mlir::Value rTag = builder.create<mlir::LLVM::ExtractValueOp>(
            loc, rhs, builder.getDenseI64ArrayAttr({0})
        ).getResult();

        auto pred = (bin.op == "==")
            ? mlir::LLVM::ICmpPredicate::eq
            : mlir::LLVM::ICmpPredicate::ne;

        mlir::Value res = builder.create<mlir::LLVM::ICmpOp>(loc, pred, lTag, rTag);
        return RValue{res, state};
    }

    mlir::Type targetTy;
    bool isSigned = true;

    const bool lhsFloat =
        lTy.kind == arklang::Type::F32 ||
        lTy.kind == arklang::Type::F64 ||
        isFloatLike(lhs.getType());

    const bool rhsFloat =
        rTy.kind == arklang::Type::F32 ||
        rTy.kind == arklang::Type::F64 ||
        isFloatLike(rhs.getType());

    if (lhsFloat || rhsFloat) {
        const bool wantF64 =
            lTy.kind == arklang::Type::F64 ||
            rTy.kind == arklang::Type::F64 ||
            lhs.getType().isF64() ||
            rhs.getType().isF64();

        targetTy = wantF64 ? builder.getF64Type() : builder.getF32Type();
    } else if (lTy.kind == arklang::Type::I64 || rTy.kind == arklang::Type::I64 ||
               lhs.getType().isInteger(64) || rhs.getType().isInteger(64)) {
        targetTy = builder.getI64Type();
    } else {
        targetTy = builder.getI32Type();
    }

    if (isSignedAstInt(lTy) || isSignedAstInt(rTy)) {
        isSigned = true;
    } else if (isUnsignedAstInt(lTy) || isUnsignedAstInt(rTy)) {
        isSigned = false;
    } else {
        isSigned = true;
    }

    lhs = coerce(builder, loc, lhs, targetTy);
    rhs = coerce(builder, loc, rhs, targetTy);

    const bool useFloatOps = isFloatLike(lhs.getType()) || isFloatLike(rhs.getType());

    mlir::Value result;

    if (bin.op == "+") {
        if (useFloatOps) {
            result = builder.create<mlir::LLVM::FAddOp>(loc, lhs, rhs);
        } else {
            result = builder.create<mlir::LLVM::AddOp>(loc, lhs, rhs);
        }
    }
    else if (bin.op == "-") {
        if (useFloatOps) {
            result = builder.create<mlir::LLVM::FSubOp>(loc, lhs, rhs);
        } else {
            result = builder.create<mlir::LLVM::SubOp>(loc, lhs, rhs);
        }
    }
    else if (bin.op == "*") {
        if (useFloatOps) {
            result = builder.create<mlir::LLVM::FMulOp>(loc, lhs, rhs);
        } else {
            result = builder.create<mlir::LLVM::MulOp>(loc, lhs, rhs);
        }
    }
    else if (bin.op == "/") {
        if (useFloatOps) {
            result = builder.create<mlir::LLVM::FDivOp>(loc, lhs, rhs);
        } else if (isSigned) {
            result = builder.create<mlir::LLVM::SDivOp>(loc, lhs, rhs);
        } else {
            result = builder.create<mlir::LLVM::UDivOp>(loc, lhs, rhs);
        }
    }
    else if (bin.op == "%") {
        if (useFloatOps) {
            return fail(bin.loc, "Modulo '%' is only supported for integer types");
        } else if (isSigned) {
            result = builder.create<mlir::LLVM::SRemOp>(loc, lhs, rhs);
        } else {
            result = builder.create<mlir::LLVM::URemOp>(loc, lhs, rhs);
        }
    }
    else if (bin.op == "==" || bin.op == "!=") {
        if (useFloatOps) {
            auto pred = (bin.op == "==")
                ? mlir::LLVM::FCmpPredicate::oeq
                : mlir::LLVM::FCmpPredicate::une;
            result = builder.create<mlir::LLVM::FCmpOp>(loc, pred, lhs, rhs);
        } else {
            auto pred = (bin.op == "==")
                ? mlir::LLVM::ICmpPredicate::eq
                : mlir::LLVM::ICmpPredicate::ne;
            result = builder.create<mlir::LLVM::ICmpOp>(loc, pred, lhs, rhs);
        }
    }
    else if (bin.op == "<" || bin.op == "<=" || bin.op == ">" || bin.op == ">=") {
        if (useFloatOps) {
            mlir::LLVM::FCmpPredicate pred;
            if (bin.op == "<") pred = mlir::LLVM::FCmpPredicate::olt;
            else if (bin.op == "<=") pred = mlir::LLVM::FCmpPredicate::ole;
            else if (bin.op == ">") pred = mlir::LLVM::FCmpPredicate::ogt;
            else pred = mlir::LLVM::FCmpPredicate::oge;

            result = builder.create<mlir::LLVM::FCmpOp>(loc, pred, lhs, rhs);
        } else {
            mlir::LLVM::ICmpPredicate pred;
            if (bin.op == "<") pred = isSigned ? mlir::LLVM::ICmpPredicate::slt : mlir::LLVM::ICmpPredicate::ult;
            else if (bin.op == "<=") pred = isSigned ? mlir::LLVM::ICmpPredicate::sle : mlir::LLVM::ICmpPredicate::ule;
            else if (bin.op == ">") pred = isSigned ? mlir::LLVM::ICmpPredicate::sgt : mlir::LLVM::ICmpPredicate::ugt;
            else pred = isSigned ? mlir::LLVM::ICmpPredicate::sge : mlir::LLVM::ICmpPredicate::uge;

            result = builder.create<mlir::LLVM::ICmpOp>(loc, pred, lhs, rhs);
        }
    }
    else {
        return fail(bin.loc, "Unsupported binary operator: " + bin.op);
    }

    return RValue{result, state};
}


// =============================================================================
// Builtin Namespace Router (ONE FUNCTION TO RULE THEM ALL)
// Handles: FS.readAll(), SYS.args, SYS.env.get(), NET.connect()
// =============================================================================
bool GenMIR::isBuiltinNamespace(llvm::StringRef baseName) {
    // This MUST include "SYS.env" to catch chained namespace calls!
    return baseName == "FS" || baseName == "NET" || baseName == "IO" || 
           baseName == "SYS" || baseName == "SYS.env";
}

mlir::FailureOr<RValue> GenMIR::lowerBuiltin(mlir::Location loc, 
                                             llvm::StringRef baseName, 
                                             llvm::StringRef member, 
                                             llvm::ArrayRef<mlir::Value> args) {
    using Ns = frontend::BuiltinNsLowering::Ns;
    Ns targetNs;
    std::string method = member.str();
    
    // 1. Identify Namespace
    if (baseName == "FS") targetNs = Ns::FS;
    else if (baseName == "NET") targetNs = Ns::NET;
    else if (baseName == "IO") targetNs = Ns::IO;
    else if (baseName == "SYS") targetNs = Ns::SYS;
    else if (baseName == "SYS.env") {
        targetNs = Ns::SYS;
        if (method == "get") method = "env_get"; // Map `SYS.env.get` -> `env_get`
    } else {
        return failure(); 
    }

    // 2. Map Method Aliases
    if (targetNs == Ns::FS && method == "save") method = "writeAtomic";
    if (targetNs == Ns::FS && method == "load") method = "readAll";
    
    // 3. Intercept `SYS.env` property read (prevents crash during chain evaluation)
    if (targetNs == Ns::SYS && method == "env") {
        mlir::Value dummy = builder.create<mlir::LLVM::ZeroOp>(loc, builder.getI64Type());
        return RValue{dummy, unitAlive(builder, loc).state};
    }

    // 4. Delegate to BuiltinNsLowering
    if (!builtinLowering) return fail(SourceLoc{}, "Internal: BuiltinNsLowering not initialized");
    auto resOr = builtinLowering->lowerCall(loc, targetNs, method, args);
    if (failed(resOr)) return failure();
    
    return RValue{*resOr, unitAlive(builder, loc).state};
}

// =============================================================================
// 2. Symbol Lowering (Lookup Local Vars & Singletons)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerSymbol(const SymbolExpr &sym) {
    mlir::Location loc = toLoc(sym.loc);

    // -------------------------------------------------------------------------
    // 0. Catch Flattened Builtin Symbols (e.g., "SYS.args" from parser)
    // -------------------------------------------------------------------------
    size_t dotPos = sym.name.find('.');
    if (dotPos != std::string::npos) {
        std::string base = sym.name.substr(0, dotPos);
        std::string mem = sym.name.substr(dotPos + 1);
        if (isBuiltinNamespace(base)) {
            return lowerBuiltin(loc, base, mem, {});
        }
    }

    // A. Local Variable Lookup
    if (mir->isDeclared(sym.name)) {
        VarInfo *var = mir->lookup(sym.name);
        if (!var || !var->place || !var->valueTy || !var->state)
            llvm::report_fatal_error("GenMIR::lowerSymbol: invalid VarInfo (missing place/type/state)");

        // GPU domain must not implicitly move locals when merely *observing* them.
        if (currentFnDomain == Domain::GPU) {
            bool provenDead = false;
            if (auto cst = llvm::dyn_cast_or_null<mlir::LLVM::ConstantOp>(var->state.getDefiningOp())) {
                if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
                    provenDead = ia.getValue().isZero();
                }
            }

            if (provenDead) {
                (void)fail(sym.loc, "Use-after-move: " + sym.name);
                return mlir::failure();
            }

            if (!llvm::isa<mlir::LLVM::ConstantOp>(var->state.getDefiningOp())) {
                (void)fail(sym.loc,
                           "GPU symbol read requires statically-alive borrow state (constant true). symbol='" +
                               sym.name + "'");
                return mlir::failure();
            }

            return mir->borrowVar(loc, *var);
        }

        return mir->readVar(loc, *var);
    }

    // B. Global Singleton Lookup
    if (auto it = globalSingletons.find(sym.name); it != globalSingletons.end()) {
        mlir::LLVM::GlobalOp global = it->second;

        mlir::Value addr = builder.create<mlir::LLVM::AddressOfOp>(loc, global).getResult();

        // Stack spill barrier to prevent constant-expression folding of address-of.
        mlir::Type ptrTy = addr.getType();
        mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(1));

        mlir::Value slot = builder.create<mlir::LLVM::AllocaOp>(
            loc, ptrTy, ptrTy, one, /*alignment=*/8);

        builder.create<mlir::LLVM::StoreOp>(loc, addr, slot);
        mlir::Value barrierAddr = builder.create<mlir::LLVM::LoadOp>(loc, ptrTy, slot);

        return RValue{barrierAddr, constBool(builder, loc, true)};
    }

    (void)fail(sym.loc, "Undefined symbol during lowering: " + sym.name);
    return mlir::failure();
}


// =============================================================================
// Slice Lowering (vec[start..end])
// Logic: Calculate new {ptr, len} based on offsets.
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerSlice(const IndexExpr &expr, const RangeExpr &range) {
    mlir::Location loc = toLoc(expr.loc);
    mlir::Type ipTy = builder.getI64Type(); // Standardize on i64

    // 1. Resolve Base Variable
    const arklang::SymbolExpr* baseSym = llvm::dyn_cast<arklang::SymbolExpr>(expr.base.get());
    if (!baseSym) {
        return fail(expr.loc, "Slicing base must be a symbol for now (e.g. v[1..3])");
    }

    const std::string& baseName = baseSym->name;
    if (!mir->isDeclared(baseName)) {
        return fail(expr.loc, "Unknown slicing base variable: " + baseName);
    }

    VarInfo* varInfo = mir->lookup(baseName);

    // 2. Validate Type
    const arklang::Type &baseType = varInfo->astTy;
    if (baseType.kind != arklang::Type::Vec && baseType.kind != arklang::Type::Slice) {
        return fail(expr.loc, "Slicing requires Vec or Slice, got: " + astTypeToString(baseType));
    }

    // 3. Extract Pointer and Length (Borrowing)
    //    We load fields directly from the stack slot to avoid 'moving' the vector resource.

    mlir::Value basePtr = varInfo->place;
    mlir::Type baseStructTy = convertType(baseType);

    // Helper to get field address
    auto getFieldAddr = [&](int idx) {
        return builder.create<mlir::LLVM::GEPOp>(
            loc,
            mlir::LLVM::LLVMPointerType::get(builder.getContext()),
            baseStructTy,
            basePtr,
            mlir::ValueRange{
                builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0)),
                builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(idx))
            }
        );
    };

    // Load Data Pointer (Field 0)
    mlir::Value dataPtrAddr = getFieldAddr(0);
    mlir::Value oldDataPtr = builder.create<mlir::LLVM::LoadOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), dataPtrAddr);

    // Load Length (Field 1)
    mlir::Value lenAddr = getFieldAddr(1);
    mlir::Value oldLen = builder.create<mlir::LLVM::LoadOp>(loc, ipTy, lenAddr);

    // 4. Calculate Range Start
    mlir::Value startVal;
    if (range.start) {
        auto s = lowerExpr(*range.start);
        if (failed(s)) return failure();
        startVal = coerce(builder, loc, s->val, ipTy);
    } else {
        startVal = builder.create<mlir::LLVM::ConstantOp>(
            loc, ipTy, builder.getIntegerAttr(ipTy, 0));
    }

    // 5. Calculate Range End
    mlir::Value endVal;
    if (range.end) {
        auto e = lowerExpr(*range.end);
        if (failed(e)) return failure();
        endVal = coerce(builder, loc, e->val, ipTy);
    } else {
        endVal = oldLen; // Default to old length
    }

    // 6. Compute New Slice Fields
    //    New Len = End - Start
    mlir::Value newLen = builder.create<mlir::LLVM::SubOp>(loc, endVal, startVal);

    //    New Ptr = Ptr + Start (GEP)
    mlir::Type elemType;
    if (!baseType.genericArgs.empty()) {
        elemType = convertType(baseType.genericArgs[0]);
    } else {
        elemType = builder.getI8Type();
    }

    mlir::Value newPtr = builder.create<mlir::LLVM::GEPOp>(
        loc,
        oldDataPtr.getType(), // Result type
        elemType,             // Pointee type (for stride)
        oldDataPtr,
        mlir::ValueRange{startVal}
    );

    // 7. Construct New Slice { ptr, len }
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto sliceStructTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), {ptrTy, ipTy});

    mlir::Value newSlice = builder.create<mlir::LLVM::UndefOp>(loc, sliceStructTy);

    newSlice = builder.create<mlir::LLVM::InsertValueOp>(
        loc, newSlice, newPtr, builder.getDenseI64ArrayAttr({0}));

    newSlice = builder.create<mlir::LLVM::InsertValueOp>(
        loc, newSlice, newLen, builder.getDenseI64ArrayAttr({1}));

    return RValue{newSlice, unitAlive(builder, loc).state};
}


// =============================================================================
// Shared GPU/Host Indexing Helper
// Extracts { dataPtr, elemLlvmTy, elemAstTy, indexLlvmTy } for Vec/Slice/Tensor/Alloc.
// Handles normalization of indices and extracting pointers from structs or raw values.
// =============================================================================

mlir::Type GenMIR::getIndexLlvmTyOrDie(mlir::Location loc) {
    if (!mir) {
        mlir::emitError(loc) << "getIndexLlvmTyOrDie: MirBuilder not initialized";
        llvm::report_fatal_error("GenMIR: MirBuilder not initialized");
    }
    mlir::Type ty = mir->getIntPtrType();
    if (!llvm::isa<mlir::IntegerType>(ty)) {
        mlir::emitError(loc) << "getIndexLlvmTyOrDie: intptr type must be integer; got '" << ty << "'";
        llvm::report_fatal_error("GenMIR: invalid intptr type");
    }
    return ty;
}

mlir::Value GenMIR::normalizeIndexTo(mlir::Location loc, mlir::Value idx, mlir::Type targetIntTy) {
    auto srcTy = llvm::dyn_cast<mlir::IntegerType>(idx.getType());
    auto dstTy = llvm::dyn_cast<mlir::IntegerType>(targetIntTy);
    if (!srcTy || !dstTy) {
        mlir::emitError(loc) << "normalizeIndexTo: index and target must be integer";
        llvm::report_fatal_error("GenMIR: normalizeIndexTo requires integer types");
    }

    unsigned sw = srcTy.getWidth();
    unsigned dw = dstTy.getWidth();

    if (sw == dw) return idx;
    if (sw < dw)  return builder.create<mlir::LLVM::ZExtOp>(loc, targetIntTy, idx).getResult();
    return builder.create<mlir::LLVM::TruncOp>(loc, targetIntTy, idx).getResult();
}

mlir::FailureOr<GenMIR::IndexBaseInfo>
GenMIR::extractDataPtr(mlir::Location loc, VarInfo &baseVar, const RValue &baseRv) {
    IndexBaseInfo out;
    out.indexLlvmTy = getIndexLlvmTyOrDie(loc);

    // 1. Identify Type Category
    bool isAlloc  = (baseVar.astTy.kind == arklang::Type::Generic && baseVar.astTy.schemaName == "Alloc");
    bool isTensor = (baseVar.astTy.kind == arklang::Type::Tensor) || 
                    (baseVar.astTy.kind == arklang::Type::Generic && baseVar.astTy.schemaName == "tensor");
    bool isContainer = (baseVar.astTy.kind == arklang::Type::Vec || baseVar.astTy.kind == arklang::Type::Slice);

    if (!isAlloc && !isTensor && !isContainer) {
        return mlir::failure(); // Unknown type
    }

    // 2. Extract Element Type
    if (baseVar.astTy.genericArgs.empty()) {
        mlir::emitError(loc) << "extractDataPtr: Type missing generic parameter (element type)";
        return mlir::failure();
    }
    out.elemAstTy  = baseVar.astTy.genericArgs[0];
    out.elemLlvmTy = convertType(out.elemAstTy);

    mlir::Type valTy = baseRv.val.getType();
    mlir::Type voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    // 3. Extract Pointer (Handle both Raw Pointer and Struct Wrappers)
    
    // Case A: It's already a pointer (e.g., GPU Kernel Argument, Raw Alloc)
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(valTy)) {
        // No bitcast needed for opaque pointers, but we ensure type consistency
        out.dataPtr = baseRv.val; 
        return out;
    }

    // Case B: It's a Struct (e.g., Host Tensor, Vec, Slice)
    // We assume the data pointer is always at Field 0.
    if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(valTy)) {
        llvm::ArrayRef<mlir::Type> body = st.getBody();
        if (body.empty() || !llvm::isa<mlir::LLVM::LLVMPointerType>(body[0])) {
            mlir::emitError(loc) << "extractDataPtr: struct field 0 must be a pointer; got '" << valTy << "'";
            return mlir::failure();
        }

        mlir::Value p = builder.create<mlir::LLVM::ExtractValueOp>(
            loc, body[0], baseRv.val, llvm::ArrayRef<int64_t>{0}).getResult();
        
        out.dataPtr = p;
        return out;
    }

    mlir::emitError(loc) << "extractDataPtr: unsupported LLVM type '" << valTy << "'";
    return mlir::failure();
}

mlir::FailureOr<mlir::Value>
GenMIR::indexGep(mlir::Location loc, const IndexBaseInfo &info, mlir::Value idxVal) {
    if (!info.dataPtr || !info.elemLlvmTy || !info.indexLlvmTy) {
        mlir::emitError(loc) << "indexGep: invalid IndexBaseInfo";
        return mlir::failure();
    }

    mlir::Value off = normalizeIndexTo(loc, idxVal, info.indexLlvmTy);

    mlir::Value gep = builder.create<mlir::LLVM::GEPOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()), // Result: ptr
        info.elemLlvmTy,                                        // Pointee: T
        info.dataPtr,                                           // Base: ptr
        mlir::ValueRange{off}                                   // Offset
    ).getResult();

    return gep;
}
// =============================================================================
// Helper: Calculate Address of Element for Vec/Slice/Tensor/Alloc
// Returns: elem_ptr (T*)
// =============================================================================
mlir::FailureOr<mlir::Value>
GenMIR::getVectorElementAddress(mlir::Location loc,
                               VarInfo &baseVar,
                               mlir::Value indexVal) {
    if (!baseVar.place || !baseVar.valueTy || !baseVar.state) {
        mlir::emitError(loc) << "getVectorElementAddress: invalid VarInfo (missing place/type/state)";
        return mlir::failure();
    }

    // Validate alive without relying on pointer-deduction.
    mlir::Operation *stDef = baseVar.state.getDefiningOp();
    mlir::LLVM::ConstantOp stCst = stDef ? llvm::dyn_cast<mlir::LLVM::ConstantOp>(stDef) : mlir::LLVM::ConstantOp();
    if (stCst) {
        mlir::Attribute av = stCst.getValue();
        mlir::IntegerAttr ia = llvm::dyn_cast<mlir::IntegerAttr>(av);
        if (ia && ia.getValue().isZero()) {
            mlir::emitError(loc) << "getVectorElementAddress: use-after-move on indexed base";
            return mlir::failure();
        }
    } else if (currentFnDomain != Domain::GPU) {
        emitHostAssert(loc, baseVar.state, "Use-after-move: indexed base");
    }

    // Load base aggregate/pointer value (non-consuming).
    mlir::Value baseVal = builder.create<mlir::LLVM::LoadOp>(loc, baseVar.valueTy, baseVar.place).getResult();
    RValue baseRv{baseVal, baseVar.state};

    // Uniform base decoding (Vec/Slice/Tensor/Alloc).
    mlir::FailureOr<GenMIR::IndexBaseInfo> infoOr = extractDataPtr(loc, baseVar, baseRv);
    if (mlir::failed(infoOr)) {
        mlir::emitError(loc) << "getVectorElementAddress: base must be Vec/Slice/Tensor/Alloc";
        return mlir::failure();
    }

    // Normalize index to target pointer width (no i64 guessing).
    mlir::Value off = normalizeIndexTo(loc, indexVal, infoOr->indexLlvmTy);

    // Compute element address.
    mlir::Value elementPtr = builder.create<mlir::LLVM::GEPOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        infoOr->elemLlvmTy,
        infoOr->dataPtr,
        mlir::ValueRange{off}
    ).getResult();

    return elementPtr;
}


// =============================================================================
// Index Lowering (Read Access) - C-Style Bare Pointer Unified
// Usage: x = vec[i] or x = tensor[i]
// Returns: RValue (Value + State)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerIndex(const IndexExpr &idx) {
    mlir::Location loc = toLoc(idx.loc);

    // 1. Slicing Check (e.g. arr[0..10])
    if (auto *range = dynamic_cast<const RangeExpr*>(idx.index.get())) {
         return lowerSlice(idx, *range);
    }

    // 2. Lower Base Expression (supports symbol/member/call/...)
    auto baseRes = lowerExpr(*idx.base);
    if (failed(baseRes)) return failure();
    RValue baseRv = *baseRes;

    // 3. Resolve Base AST Type
    arklang::Type baseAstTy = getExprType(*idx.base);

    // 4. Lower Index Expression
    auto idxRes = lowerExpr(*idx.index);
    if (failed(idxRes)) return failure();

    // 5. Build a transient VarInfo shim so we can reuse unified helpers
    VarInfo baseInfo;
    baseInfo.astTy = baseAstTy;
    baseInfo.place = nullptr;
    baseInfo.valueTy = baseRv.val.getType();
    baseInfo.state = baseRv.state;
    baseInfo.elemTy = nullptr;
    baseInfo.len = nullptr;

    // 6. Decode base into {dataPtr, elemTy, indexTy}
    auto infoOr = extractDataPtr(loc, baseInfo, baseRv);
    if (failed(infoOr)) {
        return fail(idx.loc, "Indexing failed: base is not indexable (expected Vec/Slice/Tensor/Alloc)");
    }

    // 7. Compute element address (normalizes index width internally)
    auto ptrOr = indexGep(loc, *infoOr, idxRes->val);
    if (failed(ptrOr)) {
        return fail(idx.loc, "Indexing failed: Invalid type or address calculation");
    }
    mlir::Value elementAddr = *ptrOr;

    // 8. Determine Element Type for Load
    mlir::Type elemTy = builder.getI8Type(); // Fallback

    // Prefer typechecker-attached resolved type on the IndexExpr when available.
    if (idx.resolvedType.kind != arklang::Type::Void) {
        elemTy = convertType(idx.resolvedType);
    } else if (infoOr->elemLlvmTy) {
        elemTy = infoOr->elemLlvmTy;
    } else if (!baseAstTy.genericArgs.empty()) {
        elemTy = convertType(baseAstTy.genericArgs[0]);
    }

    // 9. Emit LLVM Load
    mlir::Value loaded = builder.create<mlir::LLVM::LoadOp>(loc, elemTy, elementAddr);

    // Return R-Value (isLValue=true, maintaining your original linear state logic)
    return RValue{loaded, constBool(builder, loc, true)};
}


// =============================================================================
// L-Value Lowering (Address Calculation) - C-Style Bare Pointer Unified
// Usage: vec[i] = x, tensor[i] = x, alloc[i] = x
// Returns: LLVM Pointer (mlir::Value) for 'llvm.store'
// =============================================================================
mlir::FailureOr<mlir::Value> GenMIR::lowerExprAsPlace(const Expr &expr) {
    mlir::Location loc = toLoc(expr.loc);

    // 1. Symbol (Direct Variable Address)
    if (auto *sym = dynamic_cast<const SymbolExpr*>(&expr)) {
        if (!mir->isDeclared(sym->name)) {
            return fail(expr.loc, "Undefined symbol: " + sym->name);
        }
        VarInfo *var = mir->lookup(sym->name);

        // MemRef guards removed: Everything is now backed by a standard LLVM stack slot!
        return var->place;
    }

    // 2. Index (A[i])
    if (auto *idx = dynamic_cast<const IndexExpr*>(&expr)) {
        // A. Resolve Base
        const auto *baseSym = dynamic_cast<const SymbolExpr*>(idx->base.get());
        if (!baseSym) {
            return fail(idx->loc, "Indexed assignment base is not a bare symbol in this lowering path yet");
        }

        if (!mir->isDeclared(baseSym->name))
            return fail(idx->loc, "Undefined base variable: " + baseSym->name);

        VarInfo *baseVar = mir->lookup(baseSym->name);

        // B. Lower Index
        auto idxRes = lowerExpr(*idx->index);
        if (mlir::failed(idxRes)) return mlir::failure();

        // Coerce index to i64 (Standard for LLVM GEP)
        mlir::Value idxVal = coerce(builder, loc, idxRes->val, builder.getI64Type());
        if (!idxVal) return mlir::failure();

        // C. Compute Element Address
        // We delegate to 'elementPtr' which safely handles:
        // - Raw Pointers (Alloc<T>, GPU Tensors): Direct GEP
        // - Structs (Vec<T>, Slice<T>): Load Struct -> Extract Data Ptr -> GEP
        auto ptrOr = mir->elementPtr(loc, *baseVar, idxVal);

        if (mlir::failed(ptrOr)) {
            return fail(idx->loc, "Indexing failed: Invalid type or address calculation");
        }

        return *ptrOr;
    }

    return fail(expr.loc, "Expression is not a valid assignment target (L-Value)");
}

// =============================================================================
// Type Accessor (Context-Aware Resolution)
// =============================================================================
arklang::Type GenMIR::getExprType(const Expr &expr) {
    switch (expr.kind) {
        case ExprKind::Literal:
            return static_cast<const LiteralExpr&>(expr).type;
        
        case ExprKind::String:
            return arklang::Type{arklang::Type::Str};

        case ExprKind::ArrayLiteral: {
            const auto &arr = static_cast<const ArrayLiteral&>(expr);
            arklang::Type t;
            t.kind = arklang::Type::Vec;
            
            if (!arr.elements.empty()) {
                t.genericArgs.push_back(getExprType(*arr.elements[0]));
            } else {
                t.genericArgs.push_back({arklang::Type::Void});
            }
            return t;
        }

        case ExprKind::Symbol: {
            auto &sym = static_cast<const SymbolExpr&>(expr);
            
            // Catch flattened parser Builtins (Fallback)
            if (sym.name == "SYS.args") {
                arklang::Type t; t.kind = arklang::Type::Slice;
                t.genericArgs.push_back({arklang::Type::Str});
                return t;
            }
            
            // 1. Check Local Variable Table
            if (mir->isDeclared(sym.name)) {
                return mir->lookup(sym.name)->astTy;
            }
            
            // 2. Check Global Singletons
            if (globalSingletons.count(sym.name)) {
                arklang::Type t;
                t.kind = arklang::Type::Schema;
                t.schemaName = sym.name;
                return t;
            }

            // 3. Check Schema Registry (Static Access, e.g. Enum names)
            if (resolveSchemaAST(sym.name)) {
                arklang::Type t;
                t.kind = arklang::Type::Schema;
                t.schemaName = sym.name;
                return t;
            }

            return arklang::Type{arklang::Type::Void}; 
        }

        case ExprKind::MemberCall: {
            const auto &mcall = static_cast<const MemberCallNode&>(expr);
            
            std::string baseName;
            if (auto sym = dynamic_cast<const SymbolExpr*>(mcall.object.get())) {
                baseName = sym->name;
            } else if (auto mem = dynamic_cast<const MemberExpr*>(mcall.object.get())) {
                if (auto baseSym = dynamic_cast<const SymbolExpr*>(mem->object.get())) {
                    baseName = baseSym->name + "." + mem->member;
                }
            }

            // Map System methods
            if (baseName == "SYS" && mcall.methodName == "cwd") {
                return arklang::Type{arklang::Type::Str};
            }
            if (baseName == "SYS.env" && mcall.methodName == "get") {
                // Map Option<Str> to a Tuple {Str, Bool}
                arklang::Type t; t.kind = arklang::Type::Tuple;
                t.subtypes.push_back({arklang::Type::Str});  // Field 0: value
                t.subtypes.push_back({arklang::Type::Bool}); // Field 1: ok
                return t;
            }
            
            // Generic fallback for local variables (e.g. vec.pop())
            if (mir->isDeclared(baseName)) {
                VarInfo* var = mir->lookup(baseName);
                if (mcall.methodName == "pop" && !var->astTy.genericArgs.empty()) {
                    return var->astTy.genericArgs[0];
                }
                if (mcall.methodName == "push" || mcall.methodName == "clear") {
                    return arklang::Type{arklang::Type::Void};
                }
            }

            return arklang::Type{arklang::Type::Void};
        }

        case ExprKind::Call: {
            const auto &call = static_cast<const CallExpr&>(expr);
            
            // Handle 'AllocExpr' callee wrapper
            if (auto *alloc = dynamic_cast<const AllocExpr*>(call.callee.get())) {
                arklang::Type t;
                t.kind = arklang::Type::Generic;
                t.schemaName = "Alloc";
                t.genericArgs.push_back(alloc->type); 
                return t;
            }

            if (auto sym = dynamic_cast<const SymbolExpr*>(call.callee.get())) {
                // Handle 'allocof'
                if (sym->name == "allocof") {
                    arklang::Type t;
                    t.kind = arklang::Type::Generic;
                    t.schemaName = "Alloc";
                    if (!call.explicitGenericArgs.empty()) {
                        t.genericArgs = call.explicitGenericArgs;
                    } else {
                        t.genericArgs.push_back({arklang::Type::Void});
                    }
                    return t;
                }

                // [CRITICAL FIX] Explicitly tell the AST that these intrinsics return 64-bit integers!
                if (sym->name == "hash" || sym->name == "shash" || sym->name == "stable_hash" || 
                    sym->name == "sizeof" || sym->name == "len") {
                    return arklang::Type{arklang::Type::I64};
                }

                // Intrinsic Registry
                if (const Intrinsic* i = intrinsicRegistry.lookup(sym->name)) {
                    auto resolveCb = [&](const Expr& e) { return getExprType(e); };
                    auto res = i->infer(call, resolveCb);
                    if (res) return *res;
                }
                
                // Function Call Lookup
                if (astModule) {
                    for (const auto& fn : astModule->functions) {
                        if (fn->name == sym->name) return fn->returnType;
                    }
                }
                
                // Global Function Map Lookup (Cross-Module)
                if (globalFunctionMap.count(sym->name)) {
                    return globalFunctionMap[sym->name]->returnType;
                }
            }
            
            return arklang::Type{arklang::Type::I32}; 
        }

        case ExprKind::Index: {
            const auto &idx = static_cast<const IndexExpr&>(expr);

            arklang::Type baseTy = getExprType(*idx.base);

            // Handle Slicing Type (vec[1..2] -> Slice<T>)
            if (dynamic_cast<const RangeExpr*>(idx.index.get())) {
                arklang::Type t;
                t.kind = arklang::Type::Slice;

                if (!baseTy.genericArgs.empty()) {
                    t.genericArgs.push_back(baseTy.genericArgs[0]);
                } else {
                    t.genericArgs.push_back({arklang::Type::Void});
                }
                return t;
            }

            // Universal Container Resolution
            if (!baseTy.genericArgs.empty()) {
                return baseTy.genericArgs[0];
            }
            
            return arklang::Type{arklang::Type::I32};
        }

        case ExprKind::Binary: {
            const auto &bin = static_cast<const BinaryExpr&>(expr);
            arklang::Type lhs = getExprType(*bin.lhs);
            arklang::Type rhs = getExprType(*bin.rhs);

            if (lhs.kind == arklang::Type::F64 || rhs.kind == arklang::Type::F64) return arklang::Type{arklang::Type::F64};
            if (lhs.kind == arklang::Type::F32 || rhs.kind == arklang::Type::F32) return arklang::Type{arklang::Type::F32};
            if (lhs.kind == arklang::Type::Bool) return arklang::Type{arklang::Type::Bool}; 
            
            if (bin.op == "==" || bin.op == "!=" || bin.op == "<" || bin.op == ">") return arklang::Type{arklang::Type::Bool};

            return lhs;
        }

        case ExprKind::Lambda:
            return arklang::Type{arklang::Type::Func};

        case ExprKind::Launch:
            return arklang::Type{arklang::Type::I64};

        case ExprKind::Await:
            return arklang::Type{arklang::Type::Void};

        case ExprKind::Alloc: {
             const auto &alloc = static_cast<const AllocExpr&>(expr);
             arklang::Type t;
             t.kind = arklang::Type::Generic;
             t.schemaName = "Alloc";
             t.genericArgs.push_back(alloc.type);
             return t;
        }

        case ExprKind::SchemaExpr: {
            const auto &sch = static_cast<const SchemaExpr&>(expr);
            arklang::Type t;
            t.kind = arklang::Type::Schema;
            t.schemaName = sch.name;
            t.genericArgs = sch.genericArgs;
            return t;
        }

        case ExprKind::MemberAccess: {
            const auto &mem = static_cast<const MemberExpr&>(expr);

            // 1. Unified Base Name Extraction (Handles Namespace.Subspace like "SYS.env")
            std::string baseName;
            if (auto *objSym = dynamic_cast<const SymbolExpr*>(mem.object.get())) {
                baseName = objSym->name;
            } else if (auto *objMem = dynamic_cast<const MemberExpr*>(mem.object.get())) {
                if (auto *baseSym = dynamic_cast<const SymbolExpr*>(objMem->object.get())) {
                    baseName = baseSym->name + "." + objMem->member;
                }
            }

            // 2. Intercept Builtin Namespace Properties
            if (!baseName.empty() && isBuiltinNamespace(baseName)) {
                if (baseName == "SYS" && mem.member == "args") {
                    arklang::Type t;
                    t.kind = arklang::Type::Slice;
                    t.genericArgs.push_back({arklang::Type::Str});
                    return t;
                }
                if (baseName == "SYS" && mem.member == "cwd") {
                    return arklang::Type{arklang::Type::Str};
                }
                if (baseName == "SYS" && mem.member == "env") {
                    // Virtual namespace token (used for chaining .get)
                    return arklang::Type{arklang::Type::Void}; 
                }
            }

            // 3. Normal Type Inference
            arklang::Type baseTy = getExprType(*mem.object);
            
            // Built-in Container Properties
            if (mem.member == "len" || mem.member == "cap") return arklang::Type{arklang::Type::I64};
            if (mem.member == "name" && (baseTy.kind == arklang::Type::Schema)) return arklang::Type{arklang::Type::Str};

            // Option / Tuple destructuring (.ok and .value)
            if (baseTy.kind == arklang::Type::Tuple) {
                if (mem.member == "value" || mem.member == "0") {
                    if (!baseTy.subtypes.empty()) return baseTy.subtypes[0];
                }
                if (mem.member == "ok" || mem.member == "1") {
                    if (baseTy.subtypes.size() > 1) return baseTy.subtypes[1];
                }
            }

            // Schema Fields & Enum Variants
            if (baseTy.kind == arklang::Type::Schema) {
                const SchemaDecl* decl = resolveSchemaAST(baseTy.schemaName);
                if (decl) {
                    if (decl->kind == SchemaDecl::Enum) {
                        return baseTy; 
                    }
                    for (const auto& f : decl->fields) {
                        if (f.name == mem.member) return f.type;
                    }
                }
            }
            
            // Fallback
            return arklang::Type{arklang::Type::I32}; 
        }
        
        default:
            return arklang::Type{arklang::Type::Void};
    }
}

// =============================================================================
// Block Lowering (As Value)
// Used for Implicit Returns in Lambdas/Blocks
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerBlockAsValue(const BlockExpr &blk, mlir::Type expectedRetTy) {
    mir->pushScope();
    mlir::Location loc = toLoc(blk.loc);
    RValue finalResult = unitAlive(builder, loc); // Default to void

    for (size_t i = 0; i < blk.stmts.size(); ++i) {
        // Stop if we hit a terminator (like an explicit return)
        if (builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) break;

        bool isLast = (i == blk.stmts.size() - 1);
        bool expectsValue = !llvm::isa<mlir::LLVM::LLVMVoidType>(expectedRetTy);
        ExprKind k = blk.stmts[i]->kind;

        // [CRITICAL] Filter: Is this node actually a Value Expression?
        // We CANNOT call lowerExpr on Statements (Let, Print, While, etc.)
        bool isValueExpr = (k != ExprKind::Let && k != ExprKind::Assign && k != ExprKind::Print && 
                            k != ExprKind::While && k != ExprKind::For && k != ExprKind::Return);

        if (isLast && expectsValue && isValueExpr) {
            // Safe to lower as expression and capture result
            auto res = lowerExpr(*blk.stmts[i]);
            if (failed(res)) { mir->popScope(); return failure(); }
            finalResult = *res;
        } else {
            // Lower as statement (side-effects only)
            if (failed(lowerStmt(*blk.stmts[i]))) { mir->popScope(); return failure(); }
        }
    }
    mir->popScope();
    return finalResult;
}

// =============================================================================
// Lambda Lowering (Stateless Function Lifting)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerLambda(const LambdaExpr &expr) {
    mlir::Location loc = toLoc(expr.loc);

    // 1. Resolve Return Type
    mlir::Type retTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    if (expr.returnType.kind != arklang::Type::Void) {
        retTy = convertType(expr.returnType);
    }

    // 2. Setup Function
    static int lambdaCounter = 0;
    std::string name = "__lambda_" + std::to_string(lambdaCounter++);
    llvm::SmallVector<mlir::Type, 4> llvmParamTys;
    for (const auto &p : expr.params) llvmParamTys.push_back(convertType(p.type));

    {
        mlir::OpBuilder::InsertionGuard moduleGuard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto fnTy = mlir::LLVM::LLVMFunctionType::get(retTy, llvmParamTys, false);
        auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, fnTy);
        funcOp.setLinkage(mlir::LLVM::Linkage::Private);

        mlir::Block *entryBlock = funcOp.addEntryBlock(builder);
        builder.setInsertionPointToStart(entryBlock);

        // 3. Setup MirBuilder (Uses our fixed GLOBAL coerce)
        std::unique_ptr<MirBuilder> parentMir = std::move(this->mir);
        this->mir = std::make_unique<MirBuilder>(builder, module,
            [&](const arklang::Type &t) { return convertType(t); },
            [&](mlir::Location l, mlir::Value v, mlir::Type t) { return this->coerce(builder, l, v, t); },
            [&](const arklang::Type &t) { return isCopyType(t); }
        );

        // 4. Register Params
        for (size_t i = 0; i < expr.params.size(); ++i) {
            mlir::Value argVal = entryBlock->getArgument(i);
            this->mir->declareLocal(loc, expr.params[i].name, expr.params[i].type, 
                                   RValue{argVal, unitAlive(builder, loc).state});
        }

        // 5. Lower Body (Delegates to our safe block helper)
        mlir::FailureOr<RValue> bodyRes = (expr.body->kind == ExprKind::Block)
            ? lowerBlockAsValue(static_cast<const BlockExpr&>(*expr.body), retTy)
            : lowerExpr(*expr.body);

        if (failed(bodyRes)) { this->mir = std::move(parentMir); return failure(); }

        // 6. Final Implicit Return
        // Only if the block hasn't returned yet (e.g. fallthrough)
        mlir::Block *currentBlock = builder.getBlock();
        if (currentBlock->empty() || !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            if (llvm::isa<mlir::LLVM::LLVMVoidType>(retTy)) {
                builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
            } else {
                // Coerce the captured implicit result
                mlir::Value finalVal = coerce(builder, loc, bodyRes->val, retTy);
                builder.create<mlir::LLVM::ReturnOp>(loc, finalVal);
            }
        }
        this->mir = std::move(parentMir);
    } 

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto addrOf = builder.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, mlir::SymbolRefAttr::get(builder.getContext(), name));
    return RValue{addrOf.getResult(), unitAlive(builder, loc).state};
}

// =============================================================================
// Function Call Lowering (Direct, Indirect, & Intrinsics)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerCall(const CallExpr &call) {
    mlir::Location loc = toLoc(call.loc);

    // 1. Intercept 'allocof'
    if (auto *allocExpr = dynamic_cast<const AllocExpr*>(call.callee.get())) {
        return lowerAlloc(*allocExpr, call.args);
    }

    // 2. Intrinsic Dispatch
    if (auto *sym = dynamic_cast<const SymbolExpr*>(call.callee.get())) {
        if (arklang::isIntrinsicFn(sym->name)) {
            return lowerIntrinsicCall(call, sym->name);
        }
    }

    mlir::LLVM::LLVMFuncOp directCallee = nullptr;
    mlir::Value indirectPtr = nullptr;
    std::string funcName;
    
    arklang::Type astRetTy = {arklang::Type::Void}; 

    // 3. Resolve Callee
    if (auto mem = dynamic_cast<const MemberExpr*>(call.callee.get())) {
        if (auto objSym = dynamic_cast<const SymbolExpr*>(mem->object.get())) {
            if (importedModules.count(objSym->name)) {
                const Module* mod = importedModules[objSym->name];
                std::string linkage = mangleFunction(mem->member, mod);
                directCallee = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(linkage);
                funcName = mem->member;
                if (!directCallee) return fail(call.loc, "Function '" + linkage + "' not found");
            } else {
                return fail(call.loc, "Method calls via CallExpr not supported");
            }
        }
    }
    else if (auto sym = dynamic_cast<const SymbolExpr*>(call.callee.get())) {
        funcName = sym->name;
        if (mir->isDeclared(sym->name)) {
            // Indirect Call
            VarInfo* var = mir->lookup(sym->name);
            RValue ptrRv = mir->readVar(loc, *var);
            indirectPtr = ptrRv.val;
            
            if (var->astTy.kind == arklang::Type::Func && !var->astTy.genericArgs.empty()) {
                astRetTy = var->astTy.genericArgs.back();
            } else {
                astRetTy = {arklang::Type::I32}; 
            }
        } 
        else {
            // Direct Call
            std::string linkage = sym->name;
            if (astModule) {
                bool isLocal = false;
                for(const auto& fn : astModule->functions) {
                    if (fn->name == sym->name) { isLocal = true; break; }
                }
                if (isLocal) linkage = mangleFunction(sym->name, astModule);
            }
            directCallee = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(linkage);
            if (!directCallee) return fail(call.loc, "Unknown function: " + sym->name);
        }
    }
    else {
        return fail(call.loc, "Unsupported callee kind");
    }

    // -------------------------------------------------------------------------
    // 4. Emit Call
    // -------------------------------------------------------------------------
    
    // Path A: Direct Call
    if (directCallee) {
        auto argsRes = prepareCallArgs(call.loc, directCallee, call.args, funcName);
        if (failed(argsRes)) return failure();

        auto callOp = builder.create<mlir::LLVM::CallOp>(
            loc, 
            directCallee.getFunctionType().getReturnType(),
            mlir::SymbolRefAttr::get(directCallee),
            *argsRes
        );

        mlir::Value res;
        if (callOp.getNumResults() > 0) 
            res = callOp.getOperation()->getResult(0); 
        else 
            res = getUnitUndef(builder, loc);
        
        return RValue{res, constBool(builder, loc, true)};
    }
    
    // Path B: Indirect Call (Corrected Segments & Safety)
    if (indirectPtr) {
        // [Safety 1] Validate Pointer
        if (!indirectPtr) return fail(call.loc, "Indirect call pointer is null");
        
        // [Safety 2] Verify it is an LLVM Pointer (Prevents Segfault in Builder)
        if (!llvm::isa<mlir::LLVM::LLVMPointerType>(indirectPtr.getType())) {
            return fail(call.loc, "Indirect call target is not an LLVM Pointer");
        }

        // 1. Lower Arguments
        llvm::SmallVector<mlir::Value, 8> args;
        for (const auto &arg : call.args) {
            auto valRes = lowerExpr(*arg.value);
            if (failed(valRes)) return failure();
            if (!valRes->val) return fail(call.loc, "Argument lowered to null");
            args.push_back(valRes->val);
        }

        // 2. Determine Return Type
        mlir::Type retTy;
        if (astRetTy.kind == arklang::Type::Void) {
            retTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
        } else {
            retTy = convertType(astRetTy);
        }

        // 3. Construct OperationState
        mlir::OperationState state(loc, mlir::LLVM::CallOp::getOperationName());
        
        // Operands: [Callee, Arg0, Arg1, ...]
        // Note: For LLVM::CallOp, the Indirect Callee is simply the first operand.
        state.addOperands(indirectPtr);
        state.addOperands(args);
        
        // Types
        if (!llvm::isa<mlir::LLVM::LLVMVoidType>(retTy)) {
            state.addTypes(retTy);
        }

        // 4. Attributes (CORRECTED SEGMENT LOGIC)
        // According to documentation, 'callee_operands' includes BOTH the pointer and the args.
        // Therefore, we have 2 segments:
        // Segment 0 (callee_operands) = 1 (Ptr) + N (Args)
        // Segment 1 (op_bundle_operands) = 0
        
        int32_t totalCalleeOps = 1 + static_cast<int32_t>(args.size());
        int32_t segmentData[] = { totalCalleeOps, 0 };

        // Use C-array to ArrayRef implicit conversion (Safe because array outlives the call)
        state.addAttribute("operandSegmentSizes", 
            builder.getDenseI32ArrayAttr(segmentData));
        
        // Empty bundle sizes (explicitly empty)
        state.addAttribute("op_bundle_sizes", 
            builder.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>{}));

        // 5. Create Operation
        mlir::Operation *op = builder.create(state);
        
        if (!op) return fail(call.loc, "Failed to create indirect call operation");

        mlir::Value res;
        if (op->getNumResults() > 0)
            res = op->getResult(0);
        else
            res = getUnitUndef(builder, loc);

        return RValue{res, constBool(builder, loc, true)};
    }

    return failure();
}

// =============================================================================
// Block Lowering
// Handles: Scoping, Statement Sequencing, and Dead Code Prevention
// =============================================================================
mlir::LogicalResult GenMIR::lowerBlock(const BlockExpr &blk) {
    // 1. Enter Lexical Scope
    mir->pushScope();

    // 2. Generate Statements
    for (const auto &stmt : blk.stmts) {
        // Stop generating if the block is already terminated.
        // This handles cases like: return 0; print "dead";
        mlir::Block *curBlock = builder.getBlock();
        if (curBlock && !curBlock->empty() && curBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            break;
        }

        if (failed(lowerStmt(*stmt))) {
            mir->popScope(); // Ensure scope balance on failure
            return failure();
        }
    }

    // 3. Exit Lexical Scope
    //    (Future: RAII drops for scope-bound variables would be inserted here)
    mir->popScope();

    return success();
}

// =============================================================================
// Variable Declaration (Let Binding)
// Handles: let x = val; AND let (a, b) = tuple;
// =============================================================================
mlir::LogicalResult GenMIR::lowerVarDecl(const VarDecl &decl) {
    mlir::Location loc = toLoc(decl.loc);

    // 1. Lower Initializer
    auto initRes = lowerExpr(*decl.init);
    if (failed(initRes)) return failure();
    RValue init = *initRes;

    // 2. Infer Type
    //    We must use getExprType to correctly identify Vec/Slice types
    //    from the AST so that VarInfo is accurate.
    arklang::Type varTy = getExprType(*decl.init);

    // 3. Handle Declaration
    if (decl.names.size() == 1) {
        // Case A: Single Variable (let x = ...)
        mir->declareLocal(loc, decl.names[0], varTy, init);
    } 
    else {
        // Case B: Tuple Destructuring (let (a,b) = ...)
        // Check if we are destructing a tuple structure
        auto structTy = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(init.val.getType());
        
        if (!structTy && varTy.kind != arklang::Type::Tuple) {
             return fail(decl.loc, "Destructuring requires a tuple type");
        }

        // Unpack Fields
        for (size_t i = 0; i < decl.names.size(); ++i) {
            // Extract
            mlir::Value fieldVal = builder.create<mlir::LLVM::ExtractValueOp>(
                loc, init.val, builder.getDenseI64ArrayAttr({(int64_t)i})
            );

            // Determine Field Type
            arklang::Type fieldTy = {arklang::Type::I32}; // Fallback
            if (i < varTy.subtypes.size()) fieldTy = varTy.subtypes[i];
            
            // Declare
            mir->declareLocal(loc, decl.names[i], fieldTy, RValue{fieldVal, init.state});
        }
    }
    return success();
}

// =============================================================================
// Assignment Lowering (C-Style Bare Pointer Unified)
// Handles: x = rhs, A[i] = rhs (Host & Device)
// =============================================================================
mlir::LogicalResult GenMIR::lowerAssign(const AssignStmt &stmt) {
    mlir::Location loc = toLoc(stmt.loc);

    // 1. Lower RHS (The Value)
    auto rhsRes = lowerExpr(*stmt.value);
    if (failed(rhsRes)) return failure();
    RValue rhs = *rhsRes;

    // 2. Lower LHS (The Address)
    // [UNIFIED] Since we dropped MemRefs, lowerExprAsPlace natively handles 
    // standard vars, Alloc<T>, Vec<T>, Slice<T>, AND GPU Kernel Tensors (!llvm.ptr).
    auto placeRes = lowerExprAsPlace(*stmt.target);
    if (failed(placeRes)) return failure();
    mlir::Value place = *placeRes;

    // 3. Resolve Target Type & Coerce
    arklang::Type targetAstTy = getExprType(*stmt.target);
    mlir::Type llvmTargetTy = convertType(targetAstTy);
    
    // Safety: If the place is a raw pointer (Alloc/Tensor), convertType returns T*.
    // But we need to store T.
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(llvmTargetTy)) {
        // This usually happens if getExprType returns the type of the pointer 
        // instead of the pointee. For safety, we trust the RHS type if it matches
        // the pointer's element type, or we rely on 'coerce' handling it.
    }

    mlir::Value finalVal = coerce(builder, loc, rhs.val, llvmTargetTy);
    if (!finalVal) return failure();

    // 4. Emit Store (The Write)
    // [UNIFIED] Standard LLVM store for both CPU and GPU!
    builder.create<mlir::LLVM::StoreOp>(loc, finalVal, place);

    // 5. Update Liveness (for simple variables)
    if (auto sym = dynamic_cast<const SymbolExpr*>(stmt.target.get())) {
        if (mir->isDeclared(sym->name)) {
            VarInfo* var = mir->lookup(sym->name);
            var->state = rhs.state; 
        }
    }

    return success();
}


// =============================================================================
// Array Literal Lowering (Vector Construction)
// Syntax: [1, 2, 3] -> Vec<i32>
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerArrayLiteral(const ArrayLiteral &expr) {
    mlir::Location loc = toLoc(expr.loc);
    mlir::Type ipTy = mir->getIntPtrType();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    size_t count = expr.elements.size();
    if (count == 0) {
        return fail(expr.loc, "Empty array literals require explicit type annotation (e.g. [] : Vec<T>)");
    }

    // 1. Determine Element Type (Source of Truth: AST)
    //    We extract T from the Vec<T> type of the literal, or infer from first element.
    arklang::Type litType = getExprType(expr);
    arklang::Type elemAstTy;

    if (litType.kind == arklang::Type::Vec && !litType.genericArgs.empty()) {
        elemAstTy = litType.genericArgs[0];
    } else {
        // Fallback: Use the type of the first element (assuming homogeneity)
        elemAstTy = getExprType(*expr.elements[0]);
    }

    mlir::Type elemTy = convertType(elemAstTy);
    
    // 2. Calculate Allocation Size
    //    Size = Count * sizeof(T)
    mlir::Value countVal = builder.create<mlir::LLVM::ConstantOp>(
        loc, ipTy, builder.getIntegerAttr(ipTy, count));

    mlir::DataLayout dl(module);
    uint64_t elemBytes = dl.getTypeSize(elemTy);
    mlir::Value sizeBytes = builder.create<mlir::LLVM::ConstantOp>(
        loc, ipTy, builder.getIntegerAttr(ipTy, elemBytes));
    
    mlir::Value totalBytes = builder.create<mlir::LLVM::MulOp>(loc, countVal, sizeBytes);

    // 3. Allocate Memory (Runtime Call)
    //    Ideally, call a runtime helper: __ark_alloc_vec(size)
    //    For now, we inline malloc.
    if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc")) {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto fnTy = mlir::LLVM::LLVMFunctionType::get(ptrTy, {ipTy}, false);
        builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "malloc", fnTy);
    }
    
    auto mallocFn = mlir::SymbolRefAttr::get(builder.getContext(), "malloc");
    auto callOp = builder.create<mlir::LLVM::CallOp>(loc, ptrTy, mallocFn, mlir::ValueRange{totalBytes});
    
    // [FIX] Use getOperation()->getResult(0) for robustness
    mlir::Value rawPtr = callOp.getOperation()->getResult(0);

    // 4. Store Elements
    //    We loop through elements, lower them, and store them at offsets.
    for (size_t i = 0; i < count; ++i) {
        // Lower Element
        auto valRes = lowerExpr(*expr.elements[i]);
        if (failed(valRes)) return failure();
        RValue val = *valRes;

        // Calculate Offset: ptr + i
        mlir::Value idx = builder.create<mlir::LLVM::ConstantOp>(
            loc, ipTy, builder.getIntegerAttr(ipTy, i));
        
        mlir::Value elemPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, ptrTy, elemTy, rawPtr, mlir::ValueRange{idx}
        ).getResult();

        // Store
        // Note: Array literal contents are "moved" into the array.
        // If the element is a resource (Move type), its state is consumed.
        mlir::Value coerced = coerce(builder, loc, val.val, elemTy);
        builder.create<mlir::LLVM::StoreOp>(loc, coerced, elemPtr);
    }

    // 5. Construct Vector Struct { ptr, len, cap }
    auto vecStructTy = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {ptrTy, ipTy, ipTy});

    mlir::Value vec = builder.create<mlir::LLVM::UndefOp>(loc, vecStructTy);
    vec = builder.create<mlir::LLVM::InsertValueOp>(loc, vec, rawPtr,   builder.getDenseI64ArrayAttr({0}));
    vec = builder.create<mlir::LLVM::InsertValueOp>(loc, vec, countVal, builder.getDenseI64ArrayAttr({1}));
    vec = builder.create<mlir::LLVM::InsertValueOp>(loc, vec, countVal, builder.getDenseI64ArrayAttr({2}));

    // Return as RValue (Alive)
    return RValue{vec, unitAlive(builder, loc).state};
}


// =============================================================================
// Return Statement Lowering
// =============================================================================
mlir::LogicalResult GenMIR::lowerReturn(const ReturnStmt &stmt) {
    mlir::Location loc = toLoc(stmt.loc);

    // 1. Get Expected Return Type
    //    We look up the parent function to see what it expects.
    auto parentOp = builder.getBlock()->getParentOp();
    auto llvmFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(parentOp);
    if (!llvmFunc) {
        return fail(stmt.loc, "Return statement found outside of an LLVM function");
    }
    mlir::Type expectedTy = llvmFunc.getFunctionType().getReturnType();

    // 2. Handle Void Return
    if (!stmt.value) {
        if (!llvm::isa<mlir::LLVM::LLVMVoidType>(expectedTy)) {
            return fail(stmt.loc, "Missing return value in non-void function");
        }
        builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
        return success();
    }

    // 3. Lower Return Value (RValue)
    auto valRes = lowerExpr(*stmt.value);
    if (failed(valRes)) return failure();
    RValue retRVal = *valRes;

    // 4. Coerce & Emit
    //    We ensure the value matches the function signature strictly.
    mlir::Value retVal = coerce(builder, loc, retRVal.val, expectedTy);
    builder.create<mlir::LLVM::ReturnOp>(loc, retVal);

    return success();
}


// =============================================================================
// Loop Lowering (For Stmt)
// Handles: CFG Construction, State Threading (Phis), and Scope Management
// =============================================================================
mlir::LogicalResult GenMIR::lowerFor(const ForStmt &stmt) {
    mlir::Location loc = toLoc(stmt.loc);
    mlir::Type i64Ty = builder.getI64Type();
    
    // 1. Lower Bounds (Pre-Header)
    // [FIX] Use stmt.start and stmt.end directly
    auto startRes = lowerExpr(*stmt.start);
    auto endRes   = lowerExpr(*stmt.end);
    
    if (failed(startRes) || failed(endRes)) return failure();

    mlir::Value startVal = coerce(builder, loc, startRes->val, i64Ty);
    mlir::Value endVal   = coerce(builder, loc, endRes->val, i64Ty);

    // 2. Loop Scope & Iterator
    // [FIX] Use pushScope() matching your MirBuilder API
    mir->pushScope(); 
    
    // Declare Iterator 'i' (Initialized to start)
    // [FIX] Use declareLocal() and stmt.iterVar
    VarInfo &iterVar = mir->declareLocal(
        loc,
        stmt.iterVar, 
        {arklang::Type::I64}, 
        RValue{startVal, constBool(builder, loc, true)}
    );

    // 3. Snapshot Active Linear States
    //    We must create Phi nodes for the ownership state of *every* active variable.
    std::vector<VarInfo*> liveVars = mir->getActiveVars();
    llvm::SmallVector<mlir::Value, 16> preHeaderStates;
    for (auto *v : liveVars) preHeaderStates.push_back(v->state);

    // 4. Create Blocks
    mlir::Block *preHeader = builder.getInsertionBlock();
    mlir::Region *region = preHeader->getParent();
    
    mlir::Block *headerBlock = builder.createBlock(region);
    mlir::Block *bodyBlock   = builder.createBlock(region);
    mlir::Block *latchBlock  = builder.createBlock(region);
    mlir::Block *exitBlock   = builder.createBlock(region);

    // 5. Setup Header & Exit Arguments (Phis)
    //    The Header takes [State] arguments (PreHeader vs Backedge).
    //    The Exit Block ALSO needs arguments to receive the final states.
    for (auto *v : liveVars) {
        headerBlock->addArgument(builder.getI1Type(), loc); // Phi inputs
        exitBlock->addArgument(builder.getI1Type(), loc);   // Exit outputs
    }

    // Branch PreHeader -> Header
    builder.setInsertionPointToEnd(preHeader);
    if (preHeader->empty() || !preHeader->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.create<mlir::LLVM::BrOp>(loc, preHeaderStates, headerBlock);
    }

    // 6. Header Logic
    builder.setInsertionPointToStart(headerBlock);
    
    // A. Bind Phis to VarInfo
    //    Now, inside the loop, 'v.state' refers to the Phi result.
    for (size_t i = 0; i < liveVars.size(); ++i) {
        liveVars[i]->state = headerBlock->getArgument(i);
    }

    // B. Check Condition: i < end
    RValue currentI = mir->readVar(loc, iterVar); // Read current 'i'
    mlir::Value cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::slt, currentI.val, endVal);

    // Capture states for the "Exit" edge (Header -> Exit)
    // If we exit immediately, the state is whatever the Phi says.
    llvm::SmallVector<mlir::Value, 16> headerStates;
    for (auto *v : liveVars) headerStates.push_back(v->state);

    builder.create<mlir::LLVM::CondBrOp>(
        loc, cond, 
        bodyBlock, mlir::ValueRange{}, // Vars flow implicitly to body via VarInfo update above
        exitBlock, headerStates        // Vars flow to exit explicitly
    );

    // 7. Body Logic
    builder.setInsertionPointToStart(bodyBlock);
    
    if (stmt.body) {
        if (auto *blk = dynamic_cast<const BlockExpr*>(stmt.body.get())) {
            if (failed(lowerBlock(*blk))) return failure();
        } else {
            lowerExpr(*stmt.body);
        }
    }
    
    // Jump Body -> Latch
    // (Only if body didn't return/break)
    mlir::Block *bodyEnd = builder.getInsertionBlock();
    if (bodyEnd->empty() || !bodyEnd->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, latchBlock);
    }

    // 8. Latch Logic (Increment & Backedge)
    builder.setInsertionPointToStart(latchBlock);
    
    {
        // Increment 'i'
        RValue iVal = mir->readVar(loc, iterVar);
        mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
            loc, i64Ty, builder.getIntegerAttr(i64Ty, 1));
        mlir::Value nextI = builder.create<mlir::LLVM::AddOp>(loc, iVal.val, one);
        
        // Write back 'i' (Updates iterVar.state/place)
        mir->writeVar(loc, iterVar, RValue{nextI, iVal.state});
    }

    // Collect Backedge States
    // These are the states *after* the body/increment execution.
    // Since we updated VarInfo pointers during body gen, 'v->state' holds the newest state.
    llvm::SmallVector<mlir::Value, 16> backedgeStates;
    for (auto *v : liveVars) backedgeStates.push_back(v->state);

    builder.create<mlir::LLVM::BrOp>(loc, backedgeStates, headerBlock);

    // 9. Exit Logic
    builder.setInsertionPointToStart(exitBlock);
    
    // Bind the Exit Arguments back to VarInfo
    // The loop is over; the state of the world is now what came out of the Exit edge.
    for (size_t i = 0; i < liveVars.size(); ++i) {
        liveVars[i]->state = exitBlock->getArgument(i);
    }

    // Pop the Loop Scope (Drops 'i')
    // [FIX] Use popScope() matching your MirBuilder API
    mir->popScope(); 

    return success();
}

// =============================================================================
// If-Expression/Statement Lowering
// Handles: CFG Construction, Linear State Threading (Phis), and Result Merging
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerIf(const IfStmt &stmt) {
    auto loc = toLoc(stmt.loc);

    // 1. Lower Condition
    auto condRes = lowerExpr(*stmt.condition);
    if (failed(condRes)) return failure();
    
    arklang::Type condAstTy = getExprType(*stmt.condition);
    mlir::Value condVal = condRes->val;

    // [FIX 1] Smart Truthiness Evaluation (Tuples, Pointers, and Integers)
    if (condAstTy.kind == arklang::Type::Tuple && !condAstTy.subtypes.empty()) {
        size_t lastIdx = condAstTy.subtypes.size() - 1;
        if (condAstTy.subtypes[lastIdx].kind == arklang::Type::Bool) {
            // Extract the .ok flag from Option structs natively!
            if (llvm::isa<mlir::LLVM::LLVMPointerType>(condVal.getType())) {
                condVal = builder.create<mlir::LLVM::LoadOp>(loc, convertType(condAstTy), condVal);
            }
            condVal = builder.create<mlir::LLVM::ExtractValueOp>(
                loc, condVal, builder.getDenseI64ArrayAttr({(int64_t)lastIdx})).getResult();
        } else {
            condVal = coerce(builder, loc, condVal, builder.getI1Type());
        }
    } else if (llvm::isa<mlir::LLVM::LLVMPointerType>(condVal.getType())) {
        mlir::Value nullPtr = builder.create<mlir::LLVM::ZeroOp>(loc, condVal.getType());
        condVal = builder.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, condVal, nullPtr).getResult();
    } else if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(condVal.getType())) {
        if (intTy.getWidth() != 1) {
            mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(loc, intTy, builder.getIntegerAttr(intTy, 0));
            condVal = builder.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, condVal, zero).getResult();
        }
    } else {
        condVal = coerce(builder, loc, condVal, builder.getI1Type());
    }

    if (!condVal) return failure(); // Safety bailout

    // 2. Snapshot Initial States
    std::vector<VarInfo*> liveVars = mir->getActiveVars();
    llvm::SmallVector<mlir::Value, 16> initialStates;
    for (auto *v : liveVars) initialStates.push_back(v->state);

    // 3. Create Blocks
    mlir::Block *curBlock    = builder.getBlock();
    mlir::Region *region     = curBlock->getParent();
    
    mlir::Block *thenBlock  = builder.createBlock(region, region->end());
    mlir::Block *elseBlock  = builder.createBlock(region, region->end());
    mlir::Block *mergeBlock = builder.createBlock(region, region->end());

    for (size_t i = 0; i < liveVars.size(); ++i) {
        thenBlock->addArgument(builder.getI1Type(), loc);
        elseBlock->addArgument(builder.getI1Type(), loc);
    }

    builder.setInsertionPointToEnd(curBlock);
    builder.create<mlir::LLVM::CondBrOp>(loc, condVal, thenBlock, initialStates, elseBlock, initialStates);

    auto generateArm = [&](mlir::Block *block, const Expr *bodyStmt) -> std::tuple<RValue, bool, mlir::Block*> {
        builder.setInsertionPointToStart(block);
        for (size_t i = 0; i < liveVars.size(); ++i) liveVars[i]->state = block->getArgument(i);

        mir->pushScope();
        RValue result = unitAlive(builder, loc);
        
        if (bodyStmt) {
            if (auto *blk = dynamic_cast<const BlockExpr*>(bodyStmt)) {
                if (failed(lowerBlock(*blk))) return {result, false, nullptr};
            } else {
                auto res = lowerExpr(*bodyStmt);
                if (succeeded(res)) result = *res;
            }
        }
        mir->popScope();

        mlir::Block *endBlock = builder.getInsertionBlock();
        bool isTerminated = !endBlock->empty() && endBlock->back().hasTrait<mlir::OpTrait::IsTerminator>();
        return {result, !isTerminated, endBlock};
    };

    auto [thenVal, thenFalls, thenEndBlock] = generateArm(thenBlock, stmt.thenBranch.get());
    llvm::SmallVector<mlir::Value, 16> thenStates;
    if (thenFalls) for (auto *v : liveVars) thenStates.push_back(v->state);

    auto [elseVal, elseFalls, elseEndBlock] = generateArm(elseBlock, stmt.elseBranch.get());
    llvm::SmallVector<mlir::Value, 16> elseStates;
    if (elseFalls) for (auto *v : liveVars) elseStates.push_back(v->state);

    bool returnsValue = thenFalls && elseFalls && !llvm::isa<mlir::LLVM::LLVMVoidType>(thenVal.val.getType());

    if (thenFalls) {
        builder.setInsertionPointToEnd(thenEndBlock);
        llvm::SmallVector<mlir::Value, 16> args;
        if (returnsValue) args.push_back(thenVal.val);
        args.append(thenStates.begin(), thenStates.end());
        builder.create<mlir::LLVM::BrOp>(loc, args, mergeBlock);
    }

    if (elseFalls) {
        builder.setInsertionPointToEnd(elseEndBlock);
        llvm::SmallVector<mlir::Value, 16> args;
        if (returnsValue) args.push_back(elseVal.val);
        args.append(elseStates.begin(), elseStates.end());
        builder.create<mlir::LLVM::BrOp>(loc, args, mergeBlock);
    }

    builder.setInsertionPointToStart(mergeBlock);
    RValue finalRes = unitAlive(builder, loc);
    if (returnsValue) {
        mlir::Value resPhi = mergeBlock->addArgument(thenVal.val.getType(), loc);
        finalRes = {resPhi, unitAlive(builder, loc).state};
    }
    for (auto *v : liveVars) v->state = mergeBlock->addArgument(builder.getI1Type(), loc);

    if (!thenFalls && !elseFalls) builder.create<mlir::LLVM::UnreachableOp>(loc);

    return finalRes;
}

// =============================================================================
// Print Statement Lowering
// Handles: Primitives, Strings, Pointers, Vectors, and Tuples (Recursive)
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerPrint(const PrintStmt &stmt) {
    mlir::Location loc = toLoc(stmt.loc);
    mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    // Helper: Declare external print functions (1 argument)
    auto getOrDecl = [&](std::string name, mlir::Type argTy) {
        if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) return fn;
        
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {argTy}, false);
        return builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, fnTy);
    };

    // Helper: Recursive Printer
    std::function<void(RValue, arklang::Type)> emitPrint;
    
    emitPrint = [&](RValue rv, arklang::Type astTy) {
        mlir::Value val = rv.val;

        // 1. Strings
        if (astTy.kind == arklang::Type::Str) {
            mlir::Type strTy = convertType(astTy); 
            if (llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
                val = builder.create<mlir::LLVM::LoadOp>(loc, strTy, val);
            }
            auto fn = getOrDecl("printStr", strTy);
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
            return;
        }

        // 2. Primitives (Scalars)
        if (astTy.isScalar() || astTy.isInteger() || astTy.isFloat()) {
            // Load primitive values from memory, unless it's a Ptr or explicitly Ptr AST type
            if (llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType()) && astTy.kind != arklang::Type::Ptr) {
                mlir::Type valTy = convertType(astTy);
                val = builder.create<mlir::LLVM::LoadOp>(loc, valTy, val);
            }
        }

        if (astTy.kind == arklang::Type::I32) {
            auto fn = getOrDecl("printI32", builder.getI32Type());
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
        } 
        else if (astTy.kind == arklang::Type::I64) {
            auto fn = getOrDecl("printI64", builder.getI64Type());
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
        }
        else if (astTy.kind == arklang::Type::F32) {
            auto fn = getOrDecl("printF32", builder.getF32Type());
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
        }
        else if (astTy.kind == arklang::Type::F64) {
            auto fn = getOrDecl("printF64", builder.getF64Type());
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
        }
        else if (astTy.kind == arklang::Type::Bool) {
            mlir::Value b8 = builder.create<mlir::LLVM::ZExtOp>(loc, builder.getI8Type(), val);
            auto fn = getOrDecl("printBool", builder.getI8Type());
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{b8});
        }

        // 3. Pointers (print addr(A)) - AST Explicit
        else if (astTy.kind == arklang::Type::Ptr) {
            auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
            if (val.getType() != voidPtrTy) {
                 val = builder.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, val);
            }
            auto fn = getOrDecl("printRawPtr", voidPtrTy);
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
        }

        // 4. Vectors
        else if (astTy.kind == arklang::Type::Vec) {
            mlir::Value vecPtr = val;
            if (!llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
                vecPtr = mir->spillTemp(loc, val.getType(), val);
            }
            std::string funcName = "ark_vec_print_generic";
            if (!astTy.genericArgs.empty()) {
                auto et = astTy.genericArgs[0].kind;
                if (et == arklang::Type::I32) funcName = "ark_vec_print_i32";
                else if (et == arklang::Type::F32) funcName = "ark_vec_print_f32";
                else if (et == arklang::Type::Str) funcName = "ark_vec_print_str";
                else if (et == arklang::Type::F64) funcName = "ark_vec_print_f64";
            }
            auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
            auto fn = getOrDecl(funcName, ptrTy);
            builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{vecPtr});
        }

        // 5. Tuples / Structs
        else if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(convertType(astTy))) {
             if (llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
                val = builder.create<mlir::LLVM::LoadOp>(loc, st, val);
            }

            auto printLit = [&](llvm::StringRef s) {
                mlir::Value sVal = getOrCreateGlobalString(loc, builder, module, s);
                auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
                auto i64Ty = builder.getI64Type();
                auto strStructTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), {ptrTy, i64Ty});
                mlir::Value str = builder.create<mlir::LLVM::UndefOp>(loc, strStructTy);
                str = builder.create<mlir::LLVM::InsertValueOp>(loc, str, sVal, builder.getDenseI64ArrayAttr({0}));
                mlir::Value len = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getIntegerAttr(i64Ty, s.size()));
                str = builder.create<mlir::LLVM::InsertValueOp>(loc, str, len, builder.getDenseI64ArrayAttr({1}));
                auto fn = getOrDecl("printStr", strStructTy);
                builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{str});
            };

            printLit("(");
            for (size_t i = 0; i < st.getBody().size(); ++i) {
                if (i > 0) printLit(", ");
                mlir::Value field = builder.create<mlir::LLVM::ExtractValueOp>(
                    loc, val, builder.getDenseI64ArrayAttr({(int64_t)i}));
                
                arklang::Type fieldTy = {arklang::Type::I64}; 
                if (i < astTy.subtypes.size()) fieldTy = astTy.subtypes[i];
                else if (i < astTy.genericArgs.size()) fieldTy = astTy.genericArgs[i];
                
                emitPrint(RValue{field, rv.state}, fieldTy);
            }
            printLit(")");
        }
        else {
             // [FIX] Fallback: If AST type is unknown/generic but LLVM type is a pointer, print as pointer.
             if (llvm::isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
                 auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
                 if (val.getType() != voidPtrTy) {
                     val = builder.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, val);
                 }
                 auto fn = getOrDecl("printRawPtr", voidPtrTy);
                 builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{val});
             } else {
                 // True Unknown
                 auto fn = getOrDecl("printUnknown", builder.getI64Type()); 
                 auto zero = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
                 builder.create<mlir::LLVM::CallOp>(loc, fn, mlir::ValueRange{zero});
             }
        }
    };

    // --- Main Loop ---
    for (size_t i = 0; i < stmt.values.size(); ++i) {
        auto valRes = lowerExpr(*stmt.values[i]);
        if (failed(valRes)) return failure();
        
        arklang::Type valType = getExprType(*stmt.values[i]);
        emitPrint(*valRes, valType);

        if (i < stmt.values.size() - 1) {
             auto sFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printSpace");
             if (!sFn) {
                 mlir::OpBuilder::InsertionGuard g(builder);
                 builder.setInsertionPointToStart(module.getBody());
                 auto ft = mlir::LLVM::LLVMFunctionType::get(voidTy, {}, false);
                 sFn = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "printSpace", ft);
             }
             builder.create<mlir::LLVM::CallOp>(loc, sFn, mlir::ValueRange{});
        }
    }

    // Newline
    auto nlFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printNewline");
    if (!nlFn) {
         mlir::OpBuilder::InsertionGuard g(builder);
         builder.setInsertionPointToStart(module.getBody());
         auto ft = mlir::LLVM::LLVMFunctionType::get(voidTy, {}, false);
         nlFn = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "printNewline", ft);
    }
    builder.create<mlir::LLVM::CallOp>(loc, nlFn, mlir::ValueRange{});

    return unitAlive(builder, loc);
}

// =============================================================================
// Match Statement Lowering
// Handles: Tag Switching, Payload Extraction, and Bindings
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerMatch(const MatchStmt &stmt) {
    mlir::Location loc = toLoc(stmt.loc);
    mlir::Type i32Ty = builder.getI32Type();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    // 1. Lower Target Expression
    auto targetRes = lowerExpr(*stmt.target);
    if (failed(targetRes)) return failure();
    RValue targetRV = *targetRes;

    // Get Authoritative AST Type
    arklang::Type targetAstTy = getExprType(*stmt.target);

    // 2. Ensure Addressable Memory
    mlir::Value targetPtr;
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(targetRV.val.getType())) {
        targetPtr = targetRV.val;
    } else {
        targetPtr = mir->spillTemp(loc, targetRV.val.getType(), targetRV.val);
    }

    // 3. Resolve Schema Info
    if (stmt.cases.empty()) return fail(stmt.loc, "Empty match statement");
    
    std::string schemaName;
    for (const auto& c : stmt.cases) {
        if (!c.pattern.isDefault) {
            schemaName = c.pattern.schemaName;
            break;
        }
    }
    if (schemaName.empty()) {
        if (targetAstTy.kind == arklang::Type::Schema) {
            schemaName = targetAstTy.schemaName;
        } else {
            return fail(stmt.loc, "Cannot infer schema for match target");
        }
    }

    const SchemaDecl* decl = resolveSchemaAST(schemaName);
    if (!decl || decl->kind != SchemaDecl::Enum) return fail(stmt.loc, "Match target is not an Enum");

    // 4. Load Tag (Field 0)
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(0));
    mlir::Type enumTy = convertType(targetAstTy); 
    
    mlir::Value tagPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrTy, enumTy, targetPtr, mlir::ValueRange{zero, zero}
    ).getResult();

    mlir::Value tagVal = builder.create<mlir::LLVM::LoadOp>(loc, i32Ty, tagPtr);

    // 5. Prepare Blocks
    mlir::Block *startBlock = builder.getInsertionBlock();
    mlir::Region *region = startBlock->getParent();
    
    mlir::Block *endBlock = builder.createBlock(region);
    mlir::Block *defaultBlock = nullptr;
    
    llvm::SmallVector<int32_t, 4> caseValues;
    llvm::SmallVector<mlir::Block*, 4> caseBlocks;

    // 6. Generate Cases
    for (const auto &c : stmt.cases) {
        mlir::Block *caseBlock = builder.createBlock(region);
        
        builder.setInsertionPointToStart(caseBlock);
        mir->pushScope();

        if (c.pattern.isDefault) {
            defaultBlock = caseBlock;
        } else {
            // A. Identify Variant Tag
            int tag = -1;
            
            // [FIX] Use decltype to deduce the variant pointer type safely
            //       This avoids the 'SchemaDecl::Variant' naming issue.
            using VariantPtr = const std::decay_t<decltype(decl->variants[0])>*;
            VariantPtr variant = nullptr;

            for(size_t i=0; i<decl->variants.size(); ++i) {
                if (decl->variants[i].name == c.pattern.variantName) {
                    tag = (int)i;
                    variant = &decl->variants[i];
                    break;
                }
            }
            if (tag == -1) return fail(stmt.loc, "Unknown variant: " + c.pattern.variantName);
            
            caseValues.push_back(tag);
            caseBlocks.push_back(caseBlock);

            // B. Bind Payload (Destructuring)
            if (!c.pattern.bindings.empty()) {
                mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(1));
                mlir::Value payloadRawPtr = builder.create<mlir::LLVM::GEPOp>(
                    loc, ptrTy, enumTy, targetPtr, mlir::ValueRange{zero, one}
                ).getResult();

                llvm::SmallVector<mlir::Type, 4> fieldTys;
                for(auto &ft : variant->tuplePayload) fieldTys.push_back(convertType(ft));
                
                mlir::Type variantStructTy = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), fieldTys);
                mlir::Value variantPtr = payloadRawPtr;

                for(size_t i=0; i<c.pattern.bindings.size(); ++i) {
                    mlir::Value fieldIdx = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(i));
                    
                    mlir::Value fieldAddr = builder.create<mlir::LLVM::GEPOp>(
                        loc, ptrTy, variantStructTy, variantPtr, mlir::ValueRange{zero, fieldIdx}
                    ).getResult();

                    mlir::Value val = builder.create<mlir::LLVM::LoadOp>(loc, fieldTys[i], fieldAddr);
                    
                    mir->declareLocal(loc, c.pattern.bindings[i], variant->tuplePayload[i], 
                                      RValue{val, unitAlive(builder, loc).state});
                }
            }
        }

        // C. Lower Body
        if (c.body) {
            if (auto blk = dynamic_cast<const BlockExpr*>(c.body.get())) {
                if (failed(lowerBlock(*blk))) return failure();
            } else {
                if (failed(lowerExpr(*c.body))) return failure();
            }
        }

        mir->popScope();

        if (!builder.getBlock()->getTerminator()) {
            builder.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{}, endBlock);
        }
    }

    // 7. Handle Missing Default
    if (!defaultBlock) {
        defaultBlock = builder.createBlock(region);
        builder.setInsertionPointToStart(defaultBlock);
        builder.create<mlir::LLVM::UnreachableOp>(loc);
    }

    // 8. Generate Switch
    builder.setInsertionPointToEnd(startBlock);
    builder.create<mlir::LLVM::SwitchOp>(
        loc, tagVal, defaultBlock, 
        mlir::ValueRange{}, 
        caseValues, caseBlocks, 
        llvm::SmallVector<mlir::ValueRange>(caseBlocks.size(), mlir::ValueRange{})
    );

    builder.setInsertionPointToStart(endBlock);
    return unitAlive(builder, loc);
}

// =============================================================================
// While Loop Lowering
// Handles: Condition Evaluation, CFG, and Linear State Threading (SSA)
// =============================================================================
mlir::LogicalResult GenMIR::lowerWhile(const WhileStmt &stmt) {
    mlir::Location loc = toLoc(stmt.loc);

    // 1. Snapshot Active Variables (Pre-Loop)
    //    We need to capture the ownership state of all live variables so we can
    //    merge them (Phi) at the loop header.
    std::vector<VarInfo*> liveVars = mir->getActiveVars();
    llvm::SmallVector<mlir::Value, 16> initialStates;
    for (auto *v : liveVars) initialStates.push_back(v->state);

    // 2. Create Blocks
    //    PreHeader -> Header (Condition) <-> Body
    //                    |
    //                    v
    //                  Exit
    mlir::Block *preHeader = builder.getInsertionBlock();
    mlir::Region *region   = preHeader->getParent();

    mlir::Block *headerBlock = builder.createBlock(region);
    mlir::Block *bodyBlock   = builder.createBlock(region);
    mlir::Block *exitBlock   = builder.createBlock(region);

    // 3. PreHeader -> Header
    builder.setInsertionPointToEnd(preHeader);
    builder.create<mlir::LLVM::BrOp>(loc, initialStates, headerBlock);

    // =========================================================================
    // Header Block (Phi Nodes + Condition)
    // =========================================================================
    builder.setInsertionPointToStart(headerBlock);

    // A. Bind Phis to VarInfo
    //    The state of the world at the start of an iteration is either:
    //    1. The state coming from PreHeader (First iteration)
    //    2. The state coming from Body (Subsequent iterations)
    for (size_t i = 0; i < liveVars.size(); ++i) {
        // Add Phi argument to Header
        mlir::Value phi = headerBlock->addArgument(builder.getI1Type(), loc);
        // Update the compiler's view of the variable
        liveVars[i]->state = phi;
    }

    // B. Evaluate Condition
    //    Note: The condition sees the merged states from the Phis.
    auto condRes = lowerExpr(*stmt.condition);
    if (failed(condRes)) return failure();
    
    // Coerce to i1
    mlir::Value condVal = coerce(builder, loc, condRes->val, builder.getI1Type());

    // C. Conditional Branch
    //    If True  -> Go to Body (No args needed, vars are already updated in Header)
    //    If False -> Go to Exit (Must pass current states to resolve Phis there)
    
    // Capture states at the moment of exit (after condition evaluation)
    llvm::SmallVector<mlir::Value, 16> headerStates;
    for (auto *v : liveVars) headerStates.push_back(v->state);

    builder.create<mlir::LLVM::CondBrOp>(
        loc, condVal, 
        bodyBlock, mlir::ValueRange{}, 
        exitBlock, headerStates
    );

    // =========================================================================
    // Body Block
    // =========================================================================
    builder.setInsertionPointToStart(bodyBlock);
    
    // A. Enter Scope
    mir->pushScope();

    // B. Lower Body
    if (stmt.body) {
        if (auto *blk = dynamic_cast<const BlockExpr*>(stmt.body.get())) {
            if (failed(lowerBlock(*blk))) return failure();
        } else {
            // Single expression body
            if (failed(lowerStmt(*stmt.body))) return failure();
        }
    }

    // C. Exit Scope
    mir->popScope();

    // D. Backedge (Jump to Header)
    //    Only emit if the body didn't already return/break.
    if (!builder.getBlock()->getTerminator()) {
        // Capture states after body execution
        llvm::SmallVector<mlir::Value, 16> backedgeStates;
        for (auto *v : liveVars) backedgeStates.push_back(v->state);
        
        builder.create<mlir::LLVM::BrOp>(loc, backedgeStates, headerBlock);
    }

    // =========================================================================
    // Exit Block
    // =========================================================================
    builder.setInsertionPointToStart(exitBlock);

    // A. Bind Exit Phis
    //    The loop is done. The state of the world is now defined by what
    //    flowed out of the Header (when condition became false).
    for (size_t i = 0; i < liveVars.size(); ++i) {
        mlir::Value exitState = exitBlock->addArgument(builder.getI1Type(), loc);
        liveVars[i]->state = exitState;
    }

    return success();
}

// =============================================================================
// Parallel Loop Lowering (The 'par' Keyword)
// Handles: Execution Domain Splitting (CPU Loops vs GPU Kernels)
// =============================================================================
mlir::LogicalResult GenMIR::lowerParLoop(const ParLoop &loop) {
    mlir::Location loc = toLoc(loop.loc);

    if (loop.iterVars.empty())
        return fail(loop.loc, "par requires iterator");

    const int dims = static_cast<int>(loop.iterVars.size());
    if (dims != 1 && dims != 2)
        return fail(loop.loc, "par supports 1D or 2D only");

    // Helper: Create i64 Constant
    auto cI64 = [&](int64_t v) -> mlir::Value {
        return builder.create<mlir::arith::ConstantIntOp>(loc, v, 64);
    };

    // Helper: Cast to i64 (Robust)
    auto asI64 = [&](mlir::Value v) -> mlir::Value {
        if (!v) return {};
        if (v.getType().isInteger(64)) return v;
        if (v.getType().isIndex())
            return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), v);

        if (auto it = llvm::dyn_cast<mlir::IntegerType>(v.getType())) {
            const unsigned w = it.getWidth();
            if (w < 64) return builder.create<mlir::arith::ExtUIOp>(loc, builder.getI64Type(), v);
            if (w > 64) return builder.create<mlir::arith::TruncIOp>(loc, builder.getI64Type(), v);
            return v;
        }

        (void)fail(loop.loc, "par bound must be integer/index");
        return {};
    };

    // 1. Resolve Bounds
    struct Bounds { mlir::Value start; mlir::Value limit; };
    Bounds dimX{cI64(0), cI64(1)};
    Bounds dimY{cI64(0), cI64(1)};

    if (loop.domain.kind == ParLoop::DomainKind::Range) {
        auto *r = static_cast<const RangeExpr*>(loop.domain.expr.get());

        mlir::Value start = cI64(0);
        if (r->start) {
            auto s = lowerExpr(*r->start);
            if (mlir::failed(s)) return mlir::failure();
            start = asI64(s->val);
            if (!start) return mlir::failure();
        }

        if (!r->end) return fail(loop.loc, "par range requires end");
        auto e = lowerExpr(*r->end);
        if (mlir::failed(e)) return mlir::failure();
        mlir::Value limit = asI64(e->val);
        if (!limit) return mlir::failure();

        dimX = {start, limit};
    } else if (loop.domain.kind == ParLoop::DomainKind::LenSugar) {
        // [FIX] Support MemRef length inference for GPU Kernels
        auto container = lowerExpr(*loop.domain.expr);
        if (mlir::failed(container)) return mlir::failure();
        
        mlir::Value containerVal = container->val;
        mlir::Type ty = containerVal.getType();
        mlir::Value limit;

        if (auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(ty)) {
            // Pull the dynamic size from the MemRef. Under BarePtr+Intersperse,
            // this lowers to a direct read of the hidden 'size' kernel argument.
            mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value dimIndex = builder.create<mlir::memref::DimOp>(loc, containerVal, c0);
            limit = asI64(dimIndex);
        } else {
            // Fallback for Host-side structs or fixed-size arrays
            auto lenOr = getContainerLen(loc, static_cast<const Expr&>(*loop.domain.expr));
            if (mlir::failed(lenOr)) return mlir::failure();
            limit = asI64(*lenOr);
        }

        if (!limit) return mlir::failure();
        dimX = {cI64(0), limit};
    } else if (loop.domain.kind == ParLoop::DomainKind::DimsCall) {
        if (dims != 2) return fail(loop.loc, "dims(...) is for 2D loops");
        auto *call = static_cast<const CallExpr*>(loop.domain.expr.get());
        auto dimsOr = getDimsFromCall(loc, *call);
        if (mlir::failed(dimsOr)) return mlir::failure();

        mlir::Value w = asI64(dimsOr->w);
        mlir::Value h = asI64(dimsOr->h);
        if (!w || !h) return mlir::failure();

        dimX = {cI64(0), w};
        dimY = {cI64(0), h};
    } else {
        return fail(loop.loc, "Invalid par domain");
    }

    // Helper: Safe Terminator Check (Avoids Assertions)
    auto blockNeedsTerminator = [](mlir::Block *b) -> bool {
        if (!b) return false;
        if (b->empty()) return true;
        return !b->back().hasTrait<mlir::OpTrait::IsTerminator>();
    };

    // =========================================================================
    // Device Path: par inside GPU kernel => Grid-Stride Loop
    // =========================================================================
    if (currentFnDomain == Domain::GPU) {
        if (dims == 2)
            return fail(loop.loc, "2D par inside GPU kernel not yet supported (use 1D linearization)");

        mlir::Type idxTy = builder.getIndexType();

        // Get Hardware IDs
        mlir::Value tId  = builder.create<mlir::gpu::ThreadIdOp>(loc, idxTy, mlir::gpu::Dimension::x);
        mlir::Value bId  = builder.create<mlir::gpu::BlockIdOp>(loc, idxTy, mlir::gpu::Dimension::x);
        mlir::Value bDim = builder.create<mlir::gpu::BlockDimOp>(loc, idxTy, mlir::gpu::Dimension::x);
        mlir::Value gDim = builder.create<mlir::gpu::GridDimOp>(loc, idxTy, mlir::gpu::Dimension::x);

        // Calculate Global ID and Stride
        // tid = blockIdx.x * blockDim.x + threadIdx.x
        mlir::Value blkOffset = builder.create<mlir::arith::MulIOp>(loc, bId, bDim);
        mlir::Value glbOffset = builder.create<mlir::arith::AddIOp>(loc, blkOffset, tId);
        mlir::Value tid = asI64(glbOffset);
        if (!tid) return mlir::failure();

        // stride = gridDim.x * blockDim.x
        mlir::Value totalThreads = builder.create<mlir::arith::MulIOp>(loc, gDim, bDim);
        mlir::Value stride = asI64(totalThreads);
        if (!stride) return mlir::failure();

        // Init IV = start + tid
        mlir::Value initIv = builder.create<mlir::arith::AddIOp>(loc, dimX.start, tid);

        // CFG Structure
        mlir::Block *preHeader = builder.getInsertionBlock();
        mlir::Region *parentRegion = preHeader->getParent();

        mlir::Block *header = builder.createBlock(parentRegion);
        mlir::Block *body   = builder.createBlock(parentRegion);
        mlir::Block *exit   = builder.createBlock(parentRegion);

        // Jump to Header
        builder.setInsertionPointToEnd(preHeader);
        builder.create<mlir::cf::BranchOp>(loc, header, initIv);

        // Header: Check Loop Condition
        builder.setInsertionPointToStart(header);
        mlir::Value iv = header->addArgument(builder.getI64Type(), loc);

        mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, iv, dimX.limit);
        builder.create<mlir::cf::CondBranchOp>(loc, cond, body, exit);

        // Body: Execute User Code
        builder.setInsertionPointToStart(body);
        mir->pushScope();

        RValue rv{iv, constBool(builder, loc, true)};
        mir->declareLocal(loc, loop.iterVars[0], {arklang::Type::U64}, rv);

        if (mlir::failed(lowerBlock(*loop.body))) {
            mir->popScope();
            return mlir::failure();
        }

        mir->popScope();

        // Latch: Increment and Jump Back
        if (blockNeedsTerminator(builder.getInsertionBlock())) {
            mlir::Value nextIv = builder.create<mlir::arith::AddIOp>(loc, iv, stride);
            builder.create<mlir::cf::BranchOp>(loc, header, nextIv);
        }

        builder.setInsertionPointToStart(exit);
        return mlir::success();
    }

    // =========================================================================
    // Host Path: emit gpu.launch and lower body as GPU-domain IR
    // =========================================================================
    assertParBoundsHost(loc, dimX.start, dimX.limit);
    if (dims == 2) assertParBoundsHost(loc, dimY.start, dimY.limit);

    // Calculate Grid/Block Dims
    int64_t bx = 256, by = 1, bz = 1;
    if (!loop.blockDims.empty()) {
        auto getLit = [&](const Expr *e) -> int64_t {
            auto *l = dynamic_cast<const LiteralExpr*>(e);
            if (!l) return -1;
            try { return std::stoll(l->value); } catch (...) { return -1; }
        };

        bx = getLit(loop.blockDims[0].get());
        if (bx <= 0) bx = 256;

        if (loop.blockDims.size() > 1) {
            by = getLit(loop.blockDims[1].get());
            if (by <= 0) by = 1;
        }
    } else if (dims == 2) {
        bx = 16; by = 16;
    }

    // Grid Size Calculation
    auto calcGridDim = [&](mlir::Value limit, mlir::Value start, int64_t blockSize) -> mlir::Value {
        mlir::Value diff = builder.create<mlir::arith::SubIOp>(loc, limit, start);
        mlir::Value zero = cI64(0);
        mlir::Value n    = builder.create<mlir::arith::MaxSIOp>(loc, diff, zero);
        mlir::Value bVal = cI64(blockSize);
        mlir::Value one  = cI64(1);

        mlir::Value num = builder.create<mlir::arith::AddIOp>(
            loc, n, builder.create<mlir::arith::SubIOp>(loc, bVal, one));
        return builder.create<mlir::arith::DivUIOp>(loc, num, bVal);
    };

    mlir::Value gX = calcGridDim(dimX.limit, dimX.start, bx);
    mlir::Value gY = cI64(1);
    if (dims == 2) gY = calcGridDim(dimY.limit, dimY.start, by);

    auto toIndex = [&](mlir::Value v) -> mlir::Value {
        if (v.getType().isIndex()) return v;
        return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), v);
    };
    auto cIndex = [&](int64_t v) -> mlir::Value {
        return builder.create<mlir::arith::ConstantIndexOp>(loc, v);
    };

    auto launch = builder.create<mlir::gpu::LaunchOp>(
        loc,
        toIndex(gX), toIndex(gY), cIndex(1),
        cIndex(bx), cIndex(by), cIndex(bz)
    );

    // Generate Launch Body
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Region &bodyRegion = launch.getBody();
        mlir::Block  *entryBlock = &bodyRegion.front();

        if (!entryBlock->empty() && entryBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
            entryBlock->back().erase();

        builder.setInsertionPointToStart(entryBlock);

        const Domain prevDomain = currentFnDomain;
        currentFnDomain = Domain::GPU;
        auto restoreDomain = [&]() noexcept { currentFnDomain = prevDomain; };

        auto kCast64 = [&](mlir::Value v) -> mlir::Value {
            if (v.getType().isInteger(64)) return v;
            return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), v);
        };

        auto idxTy = builder.getIndexType();
        mlir::Value bIdXVal = builder.create<mlir::gpu::BlockIdOp>(loc, idxTy, mlir::gpu::Dimension::x);
        mlir::Value bDimXVal = builder.create<mlir::gpu::BlockDimOp>(loc, idxTy, mlir::gpu::Dimension::x);
        mlir::Value tIdXVal = builder.create<mlir::gpu::ThreadIdOp>(loc, idxTy, mlir::gpu::Dimension::x);

        mlir::Value bIdX  = kCast64(bIdXVal);
        mlir::Value bDimX = kCast64(bDimXVal);
        mlir::Value tIdX  = kCast64(tIdXVal);

        mlir::Value groupX = builder.create<mlir::arith::MulIOp>(loc, bIdX, bDimX);
        mlir::Value idxX   = builder.create<mlir::arith::AddIOp>(loc, groupX, tIdX);
        mlir::Value absX   = builder.create<mlir::arith::AddIOp>(loc, dimX.start, idxX);

        mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, absX, dimX.limit);

        mlir::Value absY;
        if (dims == 2) {
            mlir::Value bIdYVal = builder.create<mlir::gpu::BlockIdOp>(loc, idxTy, mlir::gpu::Dimension::y);
            mlir::Value bDimYVal = builder.create<mlir::gpu::BlockDimOp>(loc, idxTy, mlir::gpu::Dimension::y);
            mlir::Value tIdYVal = builder.create<mlir::gpu::ThreadIdOp>(loc, idxTy, mlir::gpu::Dimension::y);

            mlir::Value bIdY  = kCast64(bIdYVal);
            mlir::Value bDimY = kCast64(bDimYVal);
            mlir::Value tIdY  = kCast64(tIdYVal);

            mlir::Value groupY = builder.create<mlir::arith::MulIOp>(loc, bIdY, bDimY);
            mlir::Value idxY   = builder.create<mlir::arith::AddIOp>(loc, groupY, tIdY);
            absY               = builder.create<mlir::arith::AddIOp>(loc, dimY.start, idxY);

            mlir::Value condY = builder.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, absY, dimY.limit);

            cond = builder.create<mlir::arith::AndIOp>(loc, cond, condY);
        }

        mlir::Block *bodyBlock = builder.createBlock(&bodyRegion);
        mlir::Block *exitBlock = builder.createBlock(&bodyRegion);

        builder.setInsertionPointToEnd(entryBlock);
        builder.create<mlir::cf::CondBranchOp>(
            loc, cond, bodyBlock, mlir::ValueRange{}, exitBlock, mlir::ValueRange{}
        );

        builder.setInsertionPointToStart(bodyBlock);
        mir->pushScope();

        RValue rvX{absX, constBool(builder, loc, true)};
        mir->declareLocal(loc, loop.iterVars[0], {arklang::Type::U64}, rvX);

        if (dims == 2) {
            RValue rvY{absY, constBool(builder, loc, true)};
            mir->declareLocal(loc, loop.iterVars[1], {arklang::Type::U64}, rvY);
        }

        if (mlir::failed(lowerBlock(*loop.body))) {
            mir->popScope();
            restoreDomain();
            return mlir::failure();
        }

        mir->popScope();

        if (blockNeedsTerminator(builder.getInsertionBlock()))
            builder.create<mlir::cf::BranchOp>(loc, exitBlock);

        builder.setInsertionPointToStart(exitBlock);
        builder.create<mlir::gpu::TerminatorOp>(loc);

        restoreDomain();
    }

    builder.setInsertionPointAfter(launch);

    auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto syncFn = getOrDeclRuntimeFn(module, builder, loc, "__ark_device_sync", voidTy, {});
    builder.create<mlir::LLVM::CallOp>(loc, syncFn, mlir::ValueRange{});

    return mlir::success();
}


// =============================================================================
// Main Statement Dispatcher
// =============================================================================
mlir::LogicalResult GenMIR::lowerStmt(const Expr &stmt) {
    // Helper: If an expression returns a Linear value but is used as a statement,
    // we must explicitly drop it to prevent memory leaks.
    auto dropIgnoredRValue = [&](RValue rv) {
        // [FIX] Use getExprType() instead of stmt.type
        arklang::Type stmtType = getExprType(stmt);

        // 1. If void or Copy type, nothing to do.
        if (stmtType.kind == arklang::Type::Void || isCopyType(stmtType)) {
            return;
        }

        // 2. Prepare for Drop (Requires a Pointer)
        //    lowerExpr usually returns a loaded Value (in a register).
        //    The runtime drop glue expects a pointer (stack slot).
        mlir::Location loc = toLoc(stmt.loc);
        mlir::Value place = rv.val;

        if (!llvm::isa<mlir::LLVM::LLVMPointerType>(place.getType())) {
            // Spill the value to a temporary stack slot
            place = mir->spillTemp(loc, rv.val.getType(), rv.val);
        }

        // 3. Emit Conditional Drop
        mir->dropPlaceIfOwned(loc, stmtType, place, rv.state);
    };

    switch (stmt.kind) {
        // --- Control Flow & Scoping ---
        case ExprKind::Block:   
            return lowerBlock(static_cast<const BlockExpr&>(stmt));

        // --- Variable Management ---
        case ExprKind::Let:     
            return lowerVarDecl(static_cast<const VarDecl&>(stmt));
        case ExprKind::Assign:  
            return lowerAssign(static_cast<const AssignStmt&>(stmt));

        // --- Loops & Returns ---
        case ExprKind::For:     
            return lowerFor(static_cast<const ForStmt&>(stmt));
        case ExprKind::Return:  
            return lowerReturn(static_cast<const ReturnStmt&>(stmt));
        case ExprKind::While:   
            return lowerWhile(static_cast<const WhileStmt&>(stmt));

        // --- Expressions used as Statements ---
        // These might return values that we are discarding.
        // Inside GenMIR::lowerStmt switch(stmt.kind)
        case ExprKind::ParLoop:
            return lowerParLoop(static_cast<const ParLoop&>(stmt));
        case ExprKind::If: {
            auto res = lowerIf(static_cast<const IfStmt&>(stmt));
            if (failed(res)) return failure();
            dropIgnoredRValue(*res);
            return success();
        }
        case ExprKind::Match: {
            auto res = lowerMatch(static_cast<const MatchStmt&>(stmt));
            if (failed(res)) return failure();
            dropIgnoredRValue(*res);
            return success();
        }
        case ExprKind::Print: {
            auto res = lowerPrint(static_cast<const PrintStmt&>(stmt));
            if (failed(res)) return failure();
            // Print returns void/unit, so drop is a no-op, but safe to call.
            return success();
        }

        // --- Side-Effect Expressions ---
        case ExprKind::Call:
        case ExprKind::MemberCall:
        case ExprKind::Launch:
        case ExprKind::Await:
        case ExprKind::Alloc: 
        case ExprKind::SchemaExpr:
        {
            auto res = lowerExpr(stmt);
            if (failed(res)) return failure();
            dropIgnoredRValue(*res);
            return success();
        }

        default: 
            return fail(stmt.loc, "Unsupported statement kind: " + std::to_string((int)stmt.kind));
    }
}


// =============================================================================
// Core Expression Dispatcher
// =============================================================================
mlir::FailureOr<RValue> GenMIR::lowerExpr(const Expr &expr) {
    switch (expr.kind) {
        // --- Literals ---
        case ExprKind::Literal: 
            return lowerLiteral(static_cast<const LiteralExpr&>(expr));
        
        case ExprKind::String:
            return lowerString(static_cast<const StringExpr&>(expr));
            
        case ExprKind::ArrayLiteral:
            return lowerArrayLiteral(static_cast<const ArrayLiteral&>(expr));
        case ExprKind::Lambda:
            return lowerLambda(static_cast<const LambdaExpr&>(expr));

        // --- Variables & Access ---
        case ExprKind::Symbol:
            return lowerSymbol(static_cast<const SymbolExpr&>(expr));
        case ExprKind::Index:
            return lowerIndex(static_cast<const IndexExpr&>(expr));
        case ExprKind::MemberAccess:
            return lowerMemberAccess(static_cast<const MemberExpr&>(expr));

        // [FIX] Handle Method Calls (e.g. obj.push(x))
        // This was missing! The parser generates MemberCallNode, so we must dispatch it.
        case ExprKind::MemberCall:
             return lowerMemberCall(static_cast<const MemberCallNode&>(expr));

        // --- Operations ---
        case ExprKind::Binary:
            return lowerBinary(static_cast<const BinaryExpr&>(expr));
        
        // --- Control Flow ---
        case ExprKind::Block:
            return lowerBlock(static_cast<const BlockExpr&>(expr));
        case ExprKind::If:
            return lowerIf(static_cast<const IfStmt&>(expr));
        case ExprKind::Match:
            return lowerMatch(static_cast<const MatchStmt&>(expr));

        // --- Calls & Async ---
        case ExprKind::Call:
            return lowerCall(static_cast<const CallExpr&>(expr));
        case ExprKind::Launch:
            return lowerLaunch(static_cast<const LaunchExpr&>(expr));
        case ExprKind::Await:
            return lowerAwait(static_cast<const AwaitExpr&>(expr));
        
        // [NOTE] Alloc is wrapped in CallExpr by the parser. 
        // If we see it here raw, it means it wasn't called correctly.
        case ExprKind::Alloc:
            return fail(expr.loc, "Alloc expression must be used as a function call (e.g. alloc<T>(...))");

        // --- Schema ---
        case ExprKind::SchemaExpr:
            return lowerSchemaInit(static_cast<const SchemaExpr&>(expr));

        default:
            // [FIX] Use the toString() helper for better debug messages
            return fail(expr.loc, "Unsupported expression kind: " + expr.toString());
    }
}


} // namespace arklang