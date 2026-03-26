#include "ark/Conversion/ArkToLLVM/ArkToLLVM.h"
#include "ark/IR/ArkMirOps.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace arklang::mir;

namespace {

// =============================================================================
// Runtime ABI & Mangling
// =============================================================================

static bool isTrivialDrop(Type t) {
  return t && (t.isIntOrIndexOrFloat());
}

static std::string mangleType(Type t) {
  if (!t) return "void";

  if (auto st = dyn_cast<StructType>(t)) {
    return ("struct_" + st.getName().str());
  }
  if (auto vt = dyn_cast<VecType>(t)) {
    return ("vec_" + mangleType(vt.getElementType()));
  }
  if (auto sl = dyn_cast<SliceType>(t)) {
    return ("slice_" + mangleType(sl.getElementType()));
  }

  if (t.isInteger(1)) return "i1";
  if (auto it = dyn_cast<IntegerType>(t)) return ("i" + llvm::utostr(it.getWidth()));
  if (t.isF32()) return "f32";
  if (t.isF64()) return "f64";

  return "opaque";
}

static std::string dropSymbolFor(Type pointee) {
  if (!pointee) return "";
  if (isa<SliceType>(pointee)) return "";
  if (isTrivialDrop(pointee)) return "";
  return ("__ark_drop_" + mangleType(pointee));
}

// =============================================================================
// Type Converter
// =============================================================================

class ArkTypeConverter final : public LLVMTypeConverter {
public:
  explicit ArkTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](PlaceType) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });

    addConversion([ctx](StateType) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });

    addConversion([ctx](VecType, SmallVectorImpl<Type> &out) -> LogicalResult {
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      out.assign(1, LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty, i64Ty}));
      return success();
    });

    addConversion([ctx](SliceType, SmallVectorImpl<Type> &out) -> LogicalResult {
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);
      auto i64Ty = IntegerType::get(ctx, 64);
      out.assign(1, LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty}));
      return success();
    });

    addConversion([ctx](StructType st, SmallVectorImpl<Type> &out) -> LogicalResult {
      llvm::SmallString<64> nm;
      nm += "ark.struct.";
      nm += st.getName();
      out.assign(1, LLVM::LLVMStructType::getIdentified(ctx, nm.str()));
      return success();
    });
  }
};

// =============================================================================
// Helpers
// =============================================================================

static Value ensureI64(Value v, Location loc, ConversionPatternRewriter &rewriter) {
  auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

  if (auto it = dyn_cast<IntegerType>(v.getType())) {
    if (it.getWidth() == 64) return v;
    if (it.getWidth() < 64) return rewriter.create<LLVM::ZExtOp>(loc, i64Ty, v);
    return rewriter.create<LLVM::TruncOp>(loc, i64Ty, v);
  }

  if (v.getType().isIndex()) {
    return rewriter.create<LLVM::SExtOp>(loc, i64Ty, v);
  }

  return v;
}

static LLVM::LLVMFuncOp getOrCreateDropDecl(ModuleOp mod, StringRef sym,
                                            ConversionPatternRewriter &rewriter,
                                            Location loc) {
  if (auto f = mod.lookupSymbol<LLVM::LLVMFuncOp>(sym)) return f;

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());

  auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  auto ptrTy  = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto fnTy   = LLVM::LLVMFunctionType::get(voidTy, {ptrTy}, false);

  return rewriter.create<LLVM::LLVMFuncOp>(loc, sym, fnTy);
}

static Value makeDummyState(Location loc, ConversionPatternRewriter &rewriter) {
  return rewriter.create<LLVM::UndefOp>(loc, LLVM::LLVMPointerType::get(rewriter.getContext()));
}

static Block *findEnclosingEntryBlock(Operation *op) {
  Operation *p = op->getParentOp();
  while (p) {
    if (auto f = dyn_cast<func::FuncOp>(p)) {
      if (!f.getBody().empty()) return &f.getBody().front();
    }
    if (auto f = dyn_cast<LLVM::LLVMFuncOp>(p)) {
      if (!f.getBody().empty()) return &f.getBody().front();
    }
    p = p->getParentOp();
  }
  return nullptr;
}

static Value i64c(Location loc, ConversionPatternRewriter &rewriter, int64_t v) {
  auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
  return rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(v));
}

static Value gepStructField(Location loc,
                            ConversionPatternRewriter &rewriter,
                            Value basePtr,
                            LLVM::LLVMStructType structTy,
                            unsigned fieldIndex) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value zero = i64c(loc, rewriter, 0);
  Value fld  = i64c(loc, rewriter, static_cast<int64_t>(fieldIndex));
  return rewriter.create<LLVM::GEPOp>(loc, ptrTy, structTy, basePtr, ValueRange{zero, fld});
}

// =============================================================================
// Op Patterns
// =============================================================================

struct SlotOpLowering final : ConvertOpToLLVMPattern<SlotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SlotOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto placeTy = op.getPlace().getType();
    Type elemTy = typeConverter->convertType(placeTy.getPointee());
    if (!elemTy) return rewriter.notifyMatchFailure(op, "failed to convert pointee type");

    Block *entryBlock = findEnclosingEntryBlock(op.getOperation());
    if (!entryBlock) return rewriter.notifyMatchFailure(op, "no enclosing function entry block");

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    Value one  = i64c(op.getLoc(), rewriter, 1);

    Value alloca = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy, elemTy, one, 0);
    rewriter.replaceOp(op, ValueRange{alloca, makeDummyState(op.getLoc(), rewriter)});
    return success();
  }
};

struct StoreOpLowering final : ConvertOpToLLVMPattern<StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.create<LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), adaptor.getPlace());
    rewriter.replaceOp(op, makeDummyState(op.getLoc(), rewriter));
    return success();
  }
};

struct ReadOpLowering final : ConvertOpToLLVMPattern<ReadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type outTy = typeConverter->convertType(op.getValue().getType());
    if (!outTy) return rewriter.notifyMatchFailure(op, "failed to convert read result type");
    Value v = rewriter.create<LLVM::LoadOp>(op.getLoc(), outTy, adaptor.getPlace());
    rewriter.replaceOp(op, ValueRange{v, makeDummyState(op.getLoc(), rewriter)});
    return success();
  }
};

struct MoveOutOpLowering final : ConvertOpToLLVMPattern<MoveOutOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(MoveOutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type outTy = typeConverter->convertType(op.getValue().getType());
    if (!outTy) return rewriter.notifyMatchFailure(op, "failed to convert moveout result type");
    Value v = rewriter.create<LLVM::LoadOp>(op.getLoc(), outTy, adaptor.getPlace());
    rewriter.replaceOp(op, ValueRange{v, makeDummyState(op.getLoc(), rewriter)});
    return success();
  }
};

struct DropOpLowering final : ConvertOpToLLVMPattern<DropOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(DropOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value place = adaptor.getPlace();
    Type pointee = op.getPlace().getType().getPointee();

    std::string sym = dropSymbolFor(pointee);
    if (!sym.empty()) {
      ModuleOp mod = op->getParentOfType<ModuleOp>();
      auto callee = getOrCreateDropDecl(mod, sym, rewriter, op.getLoc());
      rewriter.create<LLVM::CallOp>(op.getLoc(), callee, ValueRange{place});
    }

    rewriter.replaceOp(op, makeDummyState(op.getLoc(), rewriter));
    return success();
  }
};

struct IndexOpLowering final : ConvertOpToLLVMPattern<IndexOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static std::optional<int64_t> tryGetConstI64(Value v) {
    if (auto c = v.getDefiningOp<LLVM::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) return ia.getValue().getSExtValue();
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(IndexOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value basePtr = adaptor.getBase();

    Type basePointee = typeConverter->convertType(op.getBase().getType().getPointee());
    auto baseStruct = dyn_cast_or_null<LLVM::LLVMStructType>(basePointee);
    if (!baseStruct) return rewriter.notifyMatchFailure(op, "base pointee is not an LLVM struct");

    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    Value zeroV = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(0));

    // Identified LLVM structs correspond to your schema StructType lowering.
    if (baseStruct.isIdentified()) {
      auto c = tryGetConstI64(adaptor.getIndexVal());
      if (!c) return rewriter.notifyMatchFailure(op, "struct field index must be constant");

      const int64_t fieldIndex = *c;
      if (fieldIndex < 0) return rewriter.notifyMatchFailure(op, "negative field index");

      Value fieldPtr = rewriter.create<LLVM::GEPOp>(
          loc, ptrTy, baseStruct, basePtr,
          llvm::ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(fieldIndex)});

      rewriter.replaceOp(op, ValueRange{fieldPtr});
      return success();
    }

    // Literal structs are Vec/Slice ABI ({ptr,len,cap} or {ptr,len}).
    Value idx64 = ensureI64(adaptor.getIndexVal(), loc, rewriter);

    Value gepPtrField = rewriter.create<LLVM::GEPOp>(
        loc, ptrTy, baseStruct, basePtr,
        llvm::ArrayRef<LLVM::GEPArg>{0, 0});

    Value dataPtr = rewriter.create<LLVM::LoadOp>(loc, ptrTy, gepPtrField);

    Type elemTy = typeConverter->convertType(op.getElem().getType().getPointee());
    if (!elemTy) return rewriter.notifyMatchFailure(op, "failed to convert element type");

    Value elemPtr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, elemTy, dataPtr, ValueRange{idx64});
    rewriter.replaceOp(op, ValueRange{elemPtr});
    return success();
  }
};


// AllocOp: keep it in MemRef until finalize-memref-to-llvm converts it.
struct AllocOpLowering final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, op.getType(), op.getSizes());
    return success();
  }
};

struct LaunchOpLowering final : ConvertOpToLLVMPattern<LaunchOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(LaunchOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto i64Ty = IntegerType::get(getContext(), 64);
    Value token = rewriter.create<LLVM::ConstantOp>(op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOp(op, token);
    return success();
  }
};

struct AwaitOpLowering final : ConvertOpToLLVMPattern<AwaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(AwaitOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// The Pass
// =============================================================================

struct ArkToLLVMPass final : PassWrapper<ArkToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArkToLLVMPass)

  StringRef getArgument() const override { return "ark-to-llvm"; }
  StringRef getDescription() const override { return "Lower Ark MIR to LLVM dialect"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    ArkTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);

    patterns.add<SlotOpLowering,
                 StoreOpLowering,
                 ReadOpLowering,
                 MoveOutOpLowering,
                 DropOpLowering,
                 IndexOpLowering,
                 LaunchOpLowering,
                 AwaitOpLowering>(typeConverter);

    patterns.add<AllocOpLowering>(ctx);

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();

    target.addIllegalDialect<arklang::mir::ArkMirDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
    target.addIllegalDialect<mlir::memref::MemRefDialect>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isa<ModuleOp>(op) || isa<LLVM::LLVMDialect>(op->getDialect());
    });

    if (failed(applyFullConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace arklang {
std::unique_ptr<mlir::Pass> createArkToLLVMPass() {
  return std::make_unique<ArkToLLVMPass>();
}
} // namespace arklang
