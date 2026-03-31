#pragma once

// =============================================================================
// GenMIR GPU Surface
// =============================================================================
// Class-fragment header.
// Include only from inside class GenMIR in GenMIR.hpp.
// =============================================================================

// Scan an expression subtree for GPU-illegal operations.
bool checkGpuLegality(const Expr& e, std::string& outError);

mlir::gpu::GPUModuleOp getOrCreateGpuModule();

mlir::LogicalResult emitGpuHostStub(const Function& fn);

mlir::LogicalResult emitGpuDeviceKernel(const Function& fn);

mlir::LogicalResult lowerGpuKernel(const Function& fn);