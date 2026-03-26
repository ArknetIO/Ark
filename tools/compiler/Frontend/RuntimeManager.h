#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

namespace arklang {

class RuntimeManager {
    mlir::ModuleOp module;
    mlir::OpBuilder &builder;

    // --- Type Cache (Lazy Initialized) ---
    // We cache these to prevent rebuilding complex struct types repeatedly.
    mlir::Type arkStatusTy;    // i32 (0=Ok, !=0 Error)
    mlir::Type arkStrTy;       // { ptr, i64 }
    mlir::Type arkIoErrorTy;   // { i32, i32, ArkStr }
    mlir::Type voidPtrTy;      // !llvm.ptr
    mlir::Type arkVecStructTy; // { ptr, i64, i64 }

public:
    RuntimeManager(mlir::ModuleOp m, mlir::OpBuilder &b);

    // =========================================================
    // Type Accessors (ABI Definitions)
    // =========================================================
    mlir::Type getStatusTy();      // i32
    mlir::Type getArkStrTy();      // { ptr, i64 }
    mlir::Type getIoErrorTy();     // { i32, i32, {ptr, i64} }
    mlir::Type getVoidPtrTy();     // !llvm.ptr
    mlir::Type getArkVecStructTy();// { ptr, i64, i64 }

    // =========================================================
    // Core Error Handling API (The "Call with Check" Pattern)
    // =========================================================
    
    // Calls a runtime function that follows the strict Ark Error ABI:
    //   ArkStatus func(..., [OutResult*], ArkIoError* out_err)
    //
    // This automates the safety boilerplate:
    // 1. Allocates stack space for ArkIoError.
    // 2. Allocates stack space for the Result (if outputValueTy is provided).
    // 3. Calls the C function.
    // 4. Checks the returned ArkStatus. If != 0, emits a TRAP/Panic.
    // 5. Loads and returns the success value from the stack (if any).
    //
    // funcName:      Name of C function (e.g., "__ark_file_read_all")
    // outputValueTy: The MLIR type of the *success* value (e.g., ArkStrTy), or null if void.
    // inputs:        The arguments to pass BEFORE the out-params.
    mlir::Value callWithCheck(
        mlir::Location loc, 
        llvm::StringRef funcName, 
        mlir::Type outputValueTy, 
        mlir::ValueRange inputs
    );

    // =========================================================
    // Memory Management API (Raw Pointers)
    // =========================================================

    // Maps to: void* ark_alloc(int64_t size)
    mlir::Value arkAlloc(mlir::Location loc, mlir::Value size);
    
    // Maps to: void* ark_realloc(void* ptr, int64_t new_size)
    mlir::Value arkRealloc(mlir::Location loc, mlir::Value ptr, mlir::Value newSize);
    
    // Maps to: void ark_free(void* ptr)
    void arkFree(mlir::Location loc, mlir::Value ptr);

    // Generic helper for any other function returning void* (pointer)
    mlir::Value callPtr(
        mlir::Location loc, 
        llvm::StringRef funcName, 
        mlir::ValueRange inputs
    );

    // =========================================================
    // Void / Side-Effect API
    // =========================================================

    // Generic helper for functions returning void (e.g. printStr, printI32)
    void callVoid(
        mlir::Location loc, 
        llvm::StringRef funcName, 
        mlir::ValueRange inputs
    );

    // =========================================================
    // Vector Lifecycle API (Stateful Operations)
    // =========================================================

    // Bridge for passing an SSA Vector Value to a C function expecting `ark_vec_t*`.
    // Used for read-only operations like printing where the vector is not modified.
    // 1. Spills SSA value to stack.
    // 2. Passes pointer to stack slot.
    void callVectorVoid(
        mlir::Location loc, 
        llvm::StringRef funcName, 
        mlir::Value vectorStructVal
    );

    // Wrapper for: void ark_vec_grow(ark_vec_t* v, int64_t elem_size, char* f, i32 l, i32 c)
    // Handles the spill-call-reload cycle automatically.
    // Returns the UPDATED vector struct (captured from stack after call).
    mlir::Value arkVecGrow(
        mlir::Location loc, 
        mlir::Value vecStructVal, 
        mlir::Value elemSize,
        llvm::StringRef debugFile, int debugLine, int debugCol
    );

    // Wrapper for: void ark_vec_reserve_at(...)
    // Handles the spill-call-reload cycle automatically.
    // Returns the UPDATED vector struct.
    mlir::Value arkVecReserveAt(
        mlir::Location loc, 
        mlir::Value vecStructVal, 
        mlir::Value minCap, 
        mlir::Value elemSize,
        llvm::StringRef debugFile, int debugLine, int debugCol
    );

    // Wrapper for: void ark_vec_free(ark_vec_t* v)
    // Consumes the vector. Returns nothing.
    void arkVecFree(mlir::Location loc, mlir::Value vecStructVal);

    // Returns a pointer (i8*) to a null-terminated global string constant
    mlir::Value getGlobalString(mlir::Location loc, llvm::StringRef str);

    // [NEW] Enforces C-String Invariant
    // Takes an ArkStr {ptr, len}, allocates (len+1) bytes on the stack, 
    // copies the data, appends \0, and returns the new i8*.
    mlir::Value asCString(mlir::Location loc, mlir::Value arkStrVal);

private:
    // Internal Helper: Declares function if missing, ensures private linkage
    mlir::func::FuncOp getOrInsert(llvm::StringRef name, mlir::Type retTy, llvm::ArrayRef<mlir::Type> args);
    
    // Internal Helper: Generates the check-branch-trap block for error handling
    void emitTrapIfError(mlir::Location loc, mlir::Value statusVal, mlir::Value errStructPtr);

    // Internal Helper: Creates Global String Constant and returns i8* to it
    mlir::Value getGlobalStringPtr(mlir::Location loc, llvm::StringRef str);

    // Internal Helper: Pushes file/line/col args into the provided vector
    void getDebugArgs(mlir::Location loc, llvm::StringRef file, int line, int col, llvm::SmallVectorImpl<mlir::Value>& out);
};

} // namespace arklang