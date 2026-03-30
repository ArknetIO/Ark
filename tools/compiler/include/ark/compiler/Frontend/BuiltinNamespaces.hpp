#pragma once

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h" // FailureOr
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace arklang::frontend {

class BuiltinNsLowering final {
public:
    // Namespaces exposed to Ark surface syntax.
    enum class Ns : uint8_t { FS, IO, NET, SYS };

    // Table entry kind.
    enum class Kind : uint8_t { Constant, Call };

    struct MapKey {
        Ns ns;
        Kind kind;
        llvm::StringRef name;
    };

    struct ConstSpec {
        mlir::Type ty;
        int64_t i64;
    };

    struct CallSpec {
        llvm::StringRef abi;
        llvm::SmallVector<mlir::Type, 8> argTys;
        llvm::SmallVector<mlir::Type, 8> retTys;
    };

    struct MapEntry {
        MapKey key{};
        ConstSpec cst{};
        CallSpec call{};
    };

    BuiltinNsLowering(mlir::ModuleOp module, mlir::OpBuilder &b);

    mlir::FailureOr<mlir::Value> lowerConstant(mlir::Location loc, Ns ns, llvm::StringRef name);
    mlir::FailureOr<mlir::Value> lowerCall(mlir::Location loc, Ns ns, llvm::StringRef name, llvm::ArrayRef<mlir::Value> args);

    static llvm::ArrayRef<MapEntry> table(mlir::MLIRContext *ctx);

private:
    // =============================================================================
    // ABI Type Helpers (must match ark_protocol.h)
    // =============================================================================

    mlir::LLVM::LLVMStructType arkStrTy();   // { i8*, i64 }
    mlir::LLVM::LLVMStructType arkBytesTy(); // { i8*, i64, i64 }
    mlir::LLVM::LLVMStructType arkIoErrTy(); // { i32, i32, ArkStr }

    // =============================================================================
    // Extern Declaration Helpers
    // =============================================================================

    mlir::func::FuncOp externFn(llvm::StringRef name, mlir::TypeRange args, mlir::TypeRange rets);
    mlir::LLVM::LLVMFuncOp externLLVMFn(llvm::StringRef name, mlir::Type ret, mlir::TypeRange args);

    void ensureMemcpy();
    void ensureMemset();

    // =============================================================================
    // Low-Level Builder Helpers
    // =============================================================================

    mlir::Value alloca1(mlir::Location loc, mlir::Type ty, int64_t align = 8);
    mlir::Value zero(mlir::Location loc, mlir::Type ty);
    void storeZero(mlir::Location loc, mlir::Value ptr, mlir::Type ty);

    bool isArkStrLike(mlir::Type ty) const;
    bool isArkBytesLike(mlir::Type ty) const;

    // =============================================================================
    // Marshalling Helpers
    // =============================================================================

    void unpackArkStr(mlir::Location loc, mlir::Value s, mlir::Value &outPtr, mlir::Value &outLen);
    void unpackArkBytes(mlir::Location loc, mlir::Value s, mlir::Value &outPtr, mlir::Value &outLen);

    mlir::Value materializeCString(mlir::Location loc, mlir::Value arkStr, mlir::Value &outLenI64);
    void freeCString(mlir::Location loc, mlir::Value cstrPtr);

    mlir::Value packCallResult(mlir::Location loc,
                               mlir::Value status,
                               llvm::ArrayRef<mlir::Value> outs,
                               mlir::Value errVal);

    // =============================================================================
    // Implementations: FS
    // =============================================================================

    mlir::FailureOr<mlir::Value> lowerFS_writeAtomic(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerFS_append(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerFS_readAll(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerFS_exists(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);

    // =============================================================================
    // Implementations: IO
    // =============================================================================

    mlir::FailureOr<mlir::Value> lowerIO_open(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerIO_close(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);

    // =============================================================================
    // Implementations: NET
    // =============================================================================

    mlir::FailureOr<mlir::Value> lowerNET_connect(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerNET_send(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerNET_recv(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);

    // =============================================================================
    // Implementations: SYS
    // =============================================================================

    mlir::FailureOr<mlir::Value> lowerSYS_cwd(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerSYS_args(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);
    mlir::FailureOr<mlir::Value> lowerSYS_env_get(mlir::Location loc, llvm::ArrayRef<mlir::Value> args);

private:
    mlir::ModuleOp module;
    mlir::OpBuilder &b;
};

} // namespace arklang::frontend