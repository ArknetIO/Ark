#include "ark/compiler/Frontend/GenMIR.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.GPU.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Runtime.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Types.hpp"
#include "ark/compiler/Frontend/GenMIR/GenMIR.Utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"


namespace arklang {



// Helper: Ensure we have a container module for GPU code
mlir::gpu::GPUModuleOp GenMIR::getOrCreateGpuModule() {
    constexpr llvm::StringLiteral kSym = "ark.gpu.module";

    if (auto op = module.lookupSymbol<mlir::gpu::GPUModuleOp>(kSym))
        return op;

    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());

    // Create the module
    auto gm = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(), kSym);

    // [FIX] Ensure the GPU module has a body block correctly.
    // GPUModuleOp has a specific internal structure. 
    // If the body is empty, we add a block so we can insert functions into it.
    if (gm.getBodyRegion().empty()) {
        gm.getBodyRegion().emplaceBlock();
    }

#if defined(ARK_ENABLE_CUDA)
    gm->setAttr("ark.gpu.backend", builder.getStringAttr("cuda"));
    // Consider making this dynamic based on the user's hardware
    gm->setAttr("nvvm.target", builder.getStringAttr("sm_70")); 
#elif defined(ARK_ENABLE_HIP)
    gm->setAttr("ark.gpu.backend", builder.getStringAttr("hip"));
#elif defined(ARK_ENABLE_METAL)
    gm->setAttr("ark.gpu.backend", builder.getStringAttr("metal"));
#endif

    return gm;
}




// =============================================================================
// GPU Host Stub Emission (Restored LLVMFuncOp & 1:1 Pointer Passing)
// =============================================================================
mlir::LogicalResult GenMIR::emitGpuHostStub(const Function &fn) {
    mir->enterFunction();
    mlir::Location loc = toLoc(fn.loc);

    llvm::SmallVector<mlir::Type, 4> hostArgTypes;
    for (const auto &argPair : fn.args) hostArgTypes.push_back(convertType(argPair.second));

    auto voidTy = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto funcTy = mlir::LLVM::LLVMFunctionType::get(voidTy, hostArgTypes, false);
    const std::string name = mangleFunction(fn.name, astModule);

    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(module.getBody());
    
    // Must be LLVMFuncOp so `lowerAsyncLaunch` can find it!
    auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, funcTy);

    mlir::Block *entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto i32Ty = builder.getI32Type();
    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(1));
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(0));

    int totalArgSlots = fn.args.size();
    mlir::Value argsArray;

    if (totalArgSlots > 0) {
        // Create an array of void pointers (void* array[N])
        auto arrayTy = mlir::LLVM::LLVMArrayType::get(voidPtrTy, totalArgSlots);
        argsArray = builder.create<mlir::LLVM::AllocaOp>(loc, voidPtrTy, arrayTy, one);

        for (size_t i = 0; i < fn.args.size(); ++i) {
            mlir::Value rawPtr = entryBlock->getArgument(i);

            // 1. Allocate a local stack slot for this specific argument
            mlir::Value pSlot = builder.create<mlir::LLVM::AllocaOp>(loc, voidPtrTy, rawPtr.getType(), one);
            
            // 2. Store the actual device pointer into the stack slot
            builder.create<mlir::LLVM::StoreOp>(loc, rawPtr, pSlot);
            
            // 3. Get the address of argsArray[i]
            mlir::Value pIdx = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(i));
            mlir::Value pGep = builder.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, arrayTy, argsArray, mlir::ValueRange{zero, pIdx});
            
            // 4. Store the address of the stack slot (pSlot) into argsArray[i]
            // This strictly satisfies the void** requirement for cuLaunchKernel
            builder.create<mlir::LLVM::StoreOp>(loc, pSlot, pGep);
        }
    } else {
        // Safe fallback for 0-argument kernels
        argsArray = builder.create<mlir::LLVM::ZeroOp>(loc, voidPtrTy);
    }

    const std::string gpuKernelName = fn.name;
    mlir::Value kernelNameStr = arklang::mir::getOrCreateGlobalString(loc, builder, module, gpuKernelName);
    
    // Grid X = 32 blocks of 32 threads = 1024 total elements
    mlir::Value dim1 = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(1));
    mlir::Value dim32 = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(32));
    mlir::Value finalSlotCount = builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(totalArgSlots));
    
    // [FIX] We must pass a NULL pointer for the ark_gpu_stream! 
    mlir::Value nullStream = builder.create<mlir::LLVM::ZeroOp>(loc, voidPtrTy);

    auto launchFn = arklang::getOrDeclareArkGpuLaunch(module, builder, loc);
    mlir::Value argsPtrOpaque = builder.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, argsArray);

    // [FIX] Pass exactly 10 arguments!
    builder.create<mlir::LLVM::CallOp>(loc, launchFn, mlir::ValueRange{
        kernelNameStr, argsPtrOpaque, finalSlotCount,
        dim32, dim1, dim1, dim32, dim1, dim1, nullStream
    });

    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
    mir->exitFunction();
    return mlir::success();
}





// =============================================================================
// Device Kernel Emission (1:1 ABI Mapping & Global Address Space)
// =============================================================================
mlir::LogicalResult GenMIR::emitGpuDeviceKernel(const Function &fn) {
    mlir::Location loc = toLoc(fn.loc);
    mlir::gpu::GPUModuleOp gm = getOrCreateGpuModule();
    if (gm.lookupSymbol<mlir::gpu::GPUFuncOp>(fn.name)) return mlir::success();

    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(gm.getBody());

    llvm::SmallVector<mlir::Type, 8> argTys;
    auto globalPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 1); // AS 1 Global
    auto localPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 0);  // AS 0 Local/Private

    // Exactly 1 pointer argument per tensor.
    for (const auto &arg : fn.args) {
        if (arklang::mir::isTensorType(arg.second)) {
            argTys.push_back(globalPtrTy);
        } else {
            argTys.push_back(convertType(arg.second));
        }
    }

    auto funcTy = builder.getFunctionType(argTys, {}); 
    auto k = builder.create<mlir::gpu::GPUFuncOp>(loc, fn.name, funcTy);
    k->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());
    k->setAttr("ark.gpu.kernel", builder.getUnitAttr());

    mlir::Block *entry = k.getBody().empty() ? k.addEntryBlock() : &k.getBody().front();
    if (entry->getNumArguments() == 0) {
        for (auto ty : argTys) entry->addArgument(ty, loc);
    }

    builder.setInsertionPointToStart(entry);
    const Domain prevDomain = currentFnDomain;
    currentFnDomain = Domain::GPU;
    mir->enterFunction();
    mir->pushScope();

    int argIdx = 0;
    auto i64Ty = builder.getI64Type();

    for (size_t i = 0; i < fn.args.size(); ++i) {
        const std::string &name = fn.args[i].first;
        arklang::Type astTy = fn.args[i].second;

        if (arklang::mir::isTensorType(astTy)) {
            mlir::Value ptrArg = entry->getArgument(argIdx++); // The raw AS 1 device pointer
            
            // [FIX] Allocate a local stack slot (AS 0) to hold the device pointer.
            // This turns the R-Value block argument into a valid L-Value for MIR!
            mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(1));
            mlir::Value localSlot = builder.create<mlir::LLVM::AllocaOp>(loc, localPtrTy, globalPtrTy, one);
            builder.create<mlir::LLVM::StoreOp>(loc, ptrArg, localSlot);

            // Re-apply length for internal loop bounding (par i in C)
            mlir::Value sizeArg = builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(1024));

            VarInfo info;
            info.place = localSlot; // [FIX] Now point the environment to the local stack slot
            info.len = sizeArg; 
            info.valueTy = globalPtrTy;
            info.astTy = astTy;
            
            info.elemTy = builder.getF32Type(); 
            if (!astTy.genericArgs.empty()) {
                info.elemTy = convertType(astTy.genericArgs[0]);
            }
            
            info.state = arklang::constBool(builder, loc, true);
            mir->defineVar(name, info); 
        } else {
            mlir::Value val = entry->getArgument(argIdx++);
            // declareLocal handles the local allocation automatically for scalars
            mir->declareLocal(loc, name, astTy, RValue{val, arklang::constBool(builder, loc, true)});
        }
    }

    for (const auto &stmtPtr : fn.body) {
        if (!stmtPtr || mlir::failed(lowerStmt(*stmtPtr))) return mlir::failure();
    }

    mlir::Block *curBlk = builder.getInsertionBlock();
    if (curBlk && (curBlk->empty() || !curBlk->back().hasTrait<mlir::OpTrait::IsTerminator>())) {
        builder.create<mlir::gpu::ReturnOp>(loc);
    }

    mir->popScope();
    mir->exitFunction();
    currentFnDomain = prevDomain;
    return mlir::success();
}


// Exhaustive GPU Legality Checker
bool GenMIR::checkGpuLegality(const Expr& e, std::string& outError) {
    auto illegal = [&](llvm::StringRef msg) { outError = msg.str(); return true; };

    // 1. Forbidden Roots
    if (e.kind == ExprKind::Return) return illegal("Return forbidden in GPU kernel");
    if (e.kind == ExprKind::Await)  return illegal("Await forbidden in GPU kernel");
    if (e.kind == ExprKind::Alloc)  return illegal("Host alloc forbidden in GPU kernel");
    if (e.kind == ExprKind::Import) return illegal("Import forbidden in GPU kernel");
    if (e.kind == ExprKind::Launch) return illegal("Dynamic launch forbidden in GPU kernel");
    if (e.kind == ExprKind::Print)  return illegal("Print forbidden in GPU kernel");
    
    // Member calls are effectively indirect calls to host code unless proven otherwise
    if (e.kind == ExprKind::MemberCall) {
        return illegal("Method calls forbidden in GPU kernels (use standalone functions)");
    }

    // 2. Semantic Checks
    if (e.kind == ExprKind::Call) {
        auto& c = static_cast<const CallExpr&>(e);
        // Scan args first
        for (auto& a : c.args) if (checkGpuLegality(*a.value, outError)) return true;

        auto* sym = dynamic_cast<const SymbolExpr*>(c.callee.get());
        if (!sym) return illegal("Indirect call forbidden in GPU kernel");

        if (arklang::isIntrinsicFn(sym->name)) {
            if (!arklang::mir::isGpuSafeIntrinsic(sym->name)) {
                return illegal(("Intrinsic '" + sym->name + "' is not GPU-safe").c_str());
            }
            return false;
        }

        auto it = globalFunctionMap.find(sym->name);
        if (it == globalFunctionMap.end()) {
            return illegal(("Call to unknown symbol '" + sym->name + "'").c_str());
        }
        if (it->second->domain != Domain::GPU) {
            return illegal(("Call to host function '" + sym->name + "'").c_str());
        }
        return false;
    }

    // 3. Structural Traversal
    auto scan = [&](const Expr* sub) { return sub && checkGpuLegality(*sub, outError); };
    auto scanList = [&](const auto& list) {
        for (const auto& item : list) if (checkGpuLegality(*item, outError)) return true;
        return false;
    };

    switch (e.kind) {
        case ExprKind::Block: return scanList(static_cast<const BlockExpr&>(e).stmts);
        case ExprKind::If: {
            auto& s = static_cast<const IfStmt&>(e);
            return scan(s.condition.get()) || scan(s.thenBranch.get()) || scan(s.elseBranch.get());
        }
        case ExprKind::Match: {
            auto& s = static_cast<const MatchStmt&>(e);
            if (scan(s.target.get())) return true;
            for (const auto& c : s.cases) if (scan(c.body.get())) return true;
            return false;
        }
        case ExprKind::While: return scan(static_cast<const WhileStmt&>(e).body.get());
        case ExprKind::For:   return scan(static_cast<const ForStmt&>(e).body.get());
        case ExprKind::ParLoop: return scan(static_cast<const ParLoop&>(e).body.get());
        case ExprKind::Iter:  return scan(static_cast<const IterStmt&>(e).body.get());

        case ExprKind::Binary: {
            auto& bin = static_cast<const BinaryExpr&>(e);
            return scan(bin.lhs.get()) || scan(bin.rhs.get());
        }
        case ExprKind::Assign: {
            auto& a = static_cast<const AssignStmt&>(e);
            return scan(a.target.get()) || scan(a.value.get());
        }
        case ExprKind::Index: return scan(static_cast<const IndexExpr&>(e).index.get());
        case ExprKind::MemberAccess: return scan(static_cast<const MemberExpr&>(e).object.get());
        case ExprKind::Tuple: return scanList(static_cast<const TupleExpr&>(e).elements);
        case ExprKind::ArrayLiteral: return scanList(static_cast<const ArrayLiteral&>(e).elements);
        case ExprKind::Range: {
            auto& r = static_cast<const RangeExpr&>(e);
            return scan(r.start.get()) || scan(r.end.get());
        }
        // Explicitly handle Unary (assuming UnaryExpr matches structure)
        case ExprKind::Unary: {
             // auto& u = static_cast<const UnaryExpr&>(e);
             // return scan(u.operand.get());
             return illegal("Unary traversal pending AST update"); 
        }
        case ExprKind::SchemaExpr: {
            auto& s = static_cast<const SchemaExpr&>(e);
            for(auto& f : s.fields) if(scan(f.value.get())) return true;
            return false;
        }

        // Safe Leaves
        case ExprKind::Lambda:
        case ExprKind::Literal: 
        case ExprKind::String:
        case ExprKind::Symbol:
            return false;
            
        // Conservative Default
        default: 
            return illegal("Unhandled node type in GPU checker");
    }
}



// =============================================================================
// GPU Kernel Lowering (Host Stub Generation)
// =============================================================================
mlir::LogicalResult GenMIR::lowerGpuKernel(const Function &fn) {
    if (mlir::failed(emitGpuDeviceKernel(fn)))
        return mlir::failure();
    if (mlir::failed(emitGpuHostStub(fn)))
        return mlir::failure();
    return mlir::success();
}



} // namespace arklang