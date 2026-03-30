#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h" // For logging

#include "ark/IR/ArkMirOps.h"

using namespace mlir;

namespace arklang {

// Helper: Declare runtime function if missing in the module
static LLVM::LLVMFuncOp getOrInsertFunc(ModuleOp module, PatternRewriter &rewriter, 
                                        StringRef name, Type resultType, ArrayRef<Type> argTypes) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes);
    return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnType);
}

static Value i32const(Location loc, const LLVMTypeConverter *tc, ConversionPatternRewriter &rewriter, int32_t v) {
    Type i32Ty = tc->convertType(rewriter.getI32Type());
    return rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(v));
}

static Value getI32AttrOr(Location loc, const LLVMTypeConverter *tc, ConversionPatternRewriter &rewriter, 
                          Operation *op, StringRef key, int32_t fallback) {
    if (auto a = op->getAttrOfType<IntegerAttr>(key)) {
        return i32const(loc, tc, rewriter, (int32_t)a.getInt());
    }
    return i32const(loc, tc, rewriter, fallback);
}

// =============================================================================
// Pattern: Convert memref.alloc -> __ark_gpu_alloc
// =============================================================================
struct GpuAllocLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
    explicit GpuAllocLowering(LLVMTypeConverter &converter)
        : ConvertOpToLLVMPattern<memref::AllocOp>(converter, /*benefit=*/20) {} // High benefit

    LogicalResult matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        
        // --- DEBUG LOGGING ---
        // Uncomment this to see every alloc being visited
        // llvm::errs() << "[GpuAllocLowering] Visiting AllocOp: " << op.getLoc() << "\n";
        
        int space = op.getType().getMemorySpaceAsInt();
        bool hasAttr = op->hasAttr("ark.device_id");

        // Relaxed Matcher: Space 1 OR ark.device_id attribute
        bool isGpu = (space == 1) || hasAttr;
        
        if (!isGpu) {
            // llvm::errs() << "  -> Skipped (Not GPU)\n";
            return failure();
        }

        llvm::errs() << "[GpuAllocLowering] MATCHED GPU Alloc! Space=" << space << " Attr=" << hasAttr << "\n";

        Location loc = op.getLoc();
        MemRefType memRefType = op.getType();
        auto module = op->getParentOfType<ModuleOp>();
        
        auto i64Ty = typeConverter->convertType(rewriter.getI64Type());
        auto i32Ty = typeConverter->convertType(rewriter.getI32Type());

        // 1. Calculate Total Size
        Value numElements = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(1));
        int dynamicIdx = 0;

        for (int64_t dimSize : memRefType.getShape()) {
            Value dimVal;
            if (ShapedType::isDynamic(dimSize)) {
                if (dynamicIdx >= adaptor.getDynamicSizes().size()) {
                    llvm::errs() << "  -> ERROR: Dynamic size index out of bounds\n";
                    return failure();
                }
                dimVal = adaptor.getDynamicSizes()[dynamicIdx++];
                dimVal = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, dimVal); 
            } else {
                dimVal = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(dimSize));
            }
            numElements = rewriter.create<LLVM::MulOp>(loc, numElements, dimVal);
        }

        int64_t elemSize = memRefType.getElementTypeBitWidth() / 8;
        if (elemSize > 1) {
            Value bytesPerElem = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(elemSize));
            numElements = rewriter.create<LLVM::MulOp>(loc, numElements, bytesPerElem);
        }

        // 2. Device ID
        int32_t devId = 0;
        if (auto attr = op->getAttrOfType<IntegerAttr>("ark.device_id")) devId = attr.getInt();
        Value devIdVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(devId));

        // 3. Call Runtime (returns ptr<0>)
        auto voidPtrTy = LLVM::LLVMPointerType::get(getContext()); 
        auto allocFn = getOrInsertFunc(module, rewriter, "__ark_gpu_alloc", voidPtrTy, {i64Ty, i32Ty});
        
        Value rawPtr = rewriter.create<LLVM::CallOp>(
            loc, TypeRange{voidPtrTy}, allocFn.getSymName(), ValueRange{numElements, devIdVal}
        )->getResult(0);

        // 4. Address Space Cast (ptr<0> -> ptr<Space>)
        // We must cast the raw pointer to match the MemRef's expected memory space.
        Value castPtr = rawPtr;
        if (space != 0) {
             auto targetPtrTy = LLVM::LLVMPointerType::get(getContext(), space);
             castPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, targetPtrTy, rawPtr);
        }

        // 5. Build Descriptor
        auto llvmMemRefType = typeConverter->convertType(memRefType);
        if (!llvmMemRefType) {
            llvm::errs() << "  -> ERROR: Failed to convert MemRefType to LLVM Type\n";
            return failure();
        }

        Value undef = rewriter.create<LLVM::UndefOp>(loc, llvmMemRefType);
        MemRefDescriptor desc(undef);
        
        desc.setAllocatedPtr(rewriter, loc, castPtr);
        desc.setAlignedPtr(rewriter, loc, castPtr);
        desc.setConstantOffset(rewriter, loc, 0);
        
        dynamicIdx = 0;
        for (unsigned i = 0; i < memRefType.getRank(); ++i) {
            int64_t dimSize = memRefType.getShape()[i];
            if (ShapedType::isDynamic(dimSize)) {
                desc.setSize(rewriter, loc, i, adaptor.getDynamicSizes()[dynamicIdx++]);
            } else {
                desc.setConstantSize(rewriter, loc, i, dimSize);
            }
        }
        
        if (memRefType.getRank() == 1) {
             desc.setConstantStride(rewriter, loc, 0, 1);
        }

        rewriter.replaceOp(op, {desc});
        return success();
    }
};

// =============================================================================
// Pattern: Convert ark.launch -> __ark_gpu_launch
// =============================================================================
struct LaunchLowering : public ConvertOpToLLVMPattern<arklang::mir::LaunchOp> {
    using ConvertOpToLLVMPattern<arklang::mir::LaunchOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(arklang::mir::LaunchOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        ModuleOp module = op->getParentOfType<ModuleOp>();
        MLIRContext *mctx = module.getContext();

        const LLVMTypeConverter *tc = this->getTypeConverter();
        const size_t numArgs = op.getOperands().size();

        auto ptrTy = LLVM::LLVMPointerType::get(mctx);
        Type i32Ty = tc->convertType(rewriter.getI32Type());
        Type i64Ty = tc->convertType(rewriter.getI64Type());
        
        Type i8Ty = IntegerType::get(mctx, 8);
        auto i8PtrTy = LLVM::LLVMPointerType::get(mctx);

        Value one = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(1));
        Value argcVal = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)numArgs));

        // Alloc args array: void* [numArgs]
        Value argsArray = rewriter.create<LLVM::AllocaOp>(loc, ptrTy, ptrTy, argcVal, 0);

        for (size_t i = 0; i < numArgs; ++i) {
            Value argVal = adaptor.getOperands()[i];
            Type argType = argVal.getType(); 

            // Spill argument to stack: slot = alloca T
            Value slot = rewriter.create<LLVM::AllocaOp>(loc, ptrTy, argType, one, 0);
            rewriter.create<LLVM::StoreOp>(loc, argVal, slot);

            // Store pointer to slot into argsArray[i]
            Value idx = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)i));
            // GEPOp requires element type (ptrTy)
            Value elemPtr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, ptrTy, argsArray, ValueRange{idx});
            rewriter.create<LLVM::StoreOp>(loc, slot, elemPtr);
        }

        // Kernel Name Global
        StringRef kernelName = op.getCallee().getLeafReference();
        std::string globalName = ("__kname_" + kernelName).str();

        LLVM::GlobalOp g;
        if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
            g = existing;
        } else {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            auto arrTy = LLVM::LLVMArrayType::get(i8Ty, kernelName.size() + 1);
            std::string sData = kernelName.str(); sData.push_back('\0');
            g = rewriter.create<LLVM::GlobalOp>(loc, arrTy, true, LLVM::Linkage::Private, globalName, rewriter.getStringAttr(sData));
            rewriter.setInsertionPoint(op);
        }

        // Get i8* to string
        Value gAddr = rewriter.create<LLVM::AddressOfOp>(loc, g);
        Value zero = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(0));
        Value knameI8Ptr = rewriter.create<LLVM::GEPOp>(loc, i8PtrTy, g.getType(), gAddr, ValueRange{zero, zero});

        // Dimensions
        Value gx = getI32AttrOr(loc, tc, rewriter, op, "ark.gx", 1);
        Value gy = getI32AttrOr(loc, tc, rewriter, op, "ark.gy", 1);
        Value gz = getI32AttrOr(loc, tc, rewriter, op, "ark.gz", 1);
        Value bx = getI32AttrOr(loc, tc, rewriter, op, "ark.bx", 1);
        Value by = getI32AttrOr(loc, tc, rewriter, op, "ark.by", 1);
        Value bz = getI32AttrOr(loc, tc, rewriter, op, "ark.bz", 1);

        Value nullStream = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);

        SmallVector<Type, 10> launchArgTys = {
            i8PtrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, ptrTy
        };

        auto launchFn = getOrInsertFunc(module, rewriter, "__ark_gpu_launch", i64Ty, launchArgTys);

        // [FIX] Use '->getResult(0)'
        Value token = rewriter.create<LLVM::CallOp>(
            loc, TypeRange{i64Ty}, launchFn.getSymName(),
            ValueRange{knameI8Ptr, argsArray, argcVal, gx, gy, gz, bx, by, bz, nullStream}
        )->getResult(0);

        rewriter.replaceOp(op, token);
        return success();
    }
};

// =============================================================================
// Pattern: Convert ark.await -> __ark_gpu_await
// =============================================================================
struct AwaitLowering : public ConvertOpToLLVMPattern<arklang::mir::AwaitOp> {
    using ConvertOpToLLVMPattern<arklang::mir::AwaitOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(arklang::mir::AwaitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto i64Ty = typeConverter->convertType(rewriter.getI64Type());
        auto voidTy = LLVM::LLVMVoidType::get(getContext());

        auto awaitFn = getOrInsertFunc(module, rewriter, "__ark_gpu_await", voidTy, {i64Ty});
        
        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{voidTy}, awaitFn.getSymName(), ValueRange{adaptor.getToken()}
        );
        
        rewriter.eraseOp(op);
        return success();
    }
};

// =============================================================================
// Registration
// =============================================================================

void populateGpuLoweringPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
    patterns.add<GpuAllocLowering, LaunchLowering, AwaitLowering>(converter);
}

} // namespace arklang