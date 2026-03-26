cmake_minimum_required(VERSION 3.20)
# [FIX] Removed project(ark-compiler) so we inherit "arknet" from the root CMakeLists!

include(GNUInstallDirs)
include(FetchContent)

# =========================================================
# 1. Fetch CLI11 & tomlplusplus
# =========================================================
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG        v2.4.1
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
  tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# =========================================================
# 2. LLVM Components
# =========================================================
set(LLVM_LINK_COMPONENTS
  Support Core OrcJIT ExecutionEngine Target native nativecodegen
  MC AsmParser BitWriter BitReader Linker Analysis TransformUtils
  NVPTX AMDGPU
)
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

# =========================================================
# 3. Runtime (Core Only) + GPU HAL
# =========================================================
set(RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")

set(RUNTIME_CORE_SOURCES
  "${RUNTIME_DIR}/core/async.cpp"
  "${RUNTIME_DIR}/core/memory.cpp"
  "${RUNTIME_DIR}/core/string.cpp"
  "${RUNTIME_DIR}/core/print.cpp"
  "${RUNTIME_DIR}/core/panic.cpp"
  "${RUNTIME_DIR}/core/hash.cpp"
  "${RUNTIME_DIR}/core/vector.cpp"
  "${RUNTIME_DIR}/fs/fs_posix.cpp"
  "${RUNTIME_DIR}/sys/sys_posix.cpp"
  "${RUNTIME_DIR}/net/socket_posix.cpp"
  "${RUNTIME_DIR}/net/remote.cpp"
  "${RUNTIME_DIR}/core/gpu_hal.cpp"
)

set(ARK_PLATFORM_DL_LIBS "")
if(UNIX AND NOT APPLE)
  set(ARK_PLATFORM_DL_LIBS ${CMAKE_DL_LIBS})
endif()

# =========================================================
# 4. Config Header
# =========================================================
configure_file(
  "${RUNTIME_DIR}/include/ark_config.h.in"
  "${CMAKE_BINARY_DIR}/include/ark_config.h"
)

# =========================================================
# 5. GenMIR Reorganization
# =========================================================
# We are migrating GenMIR incrementally into Frontend/GenMIR/.
# Add files here ONLY after they are actually split out.
set(ARK_GENMIR_SOURCES
  Frontend/GenMIR/GenMIR.Utils.cpp
  Frontend/GenMIR.cpp
)

# Future incremental additions:
#   Frontend/GenMIR/GenMIR.Types.cpp
#   Frontend/GenMIR/GenMIR.Schemas.cpp
#   Frontend/GenMIR/GenMIR.Modules.cpp
#   Frontend/GenMIR/GenMIR.Functions.cpp
#   Frontend/GenMIR/GenMIR.GPU.cpp
#   Frontend/GenMIR/GenMIR.Intrinsics.cpp
#   Frontend/GenMIR/GenMIR.Builtins.cpp
#   Frontend/GenMIR/GenMIR.Expr.cpp
#   Frontend/GenMIR/GenMIR.Expr.Call.cpp
#   Frontend/GenMIR/GenMIR.Expr.Container.cpp
#   Frontend/GenMIR/GenMIR.Expr.Control.cpp
#   Frontend/GenMIR/GenMIR.Stmt.cpp

# =========================================================
# 6. Shared Compiler Sources
# =========================================================
# Group the heavy compiler logic so both `arkc` and `arknet` can use it.
set(ARK_COMPILER_SHARED_SOURCES
  hud.cpp

  # Modular Pipeline
  Pipeline/Compiler.cpp
  Pipeline/Linker.cpp
  Pipeline/LinkerGpuFatbin.cpp
  Pipeline/JIT.cpp
  Integration/CapsuleBackend.cpp

  # Frontend
  Frontend/Lexer.cpp
  Frontend/Parser.cpp
  ${ARK_GENMIR_SOURCES}
  Frontend/MirBuilder.cpp
  Frontend/BuiltinNamespaces.cpp
  Frontend/Intrinsics.cpp

  # Middle-End
  Analysis/OwnershipVerifier.cpp
  Analysis/OwnershipUtils.cpp
  Transforms/DropInsertion.cpp

  # GPU Passes
  Transforms/GpuLowering.cpp
  Transforms/GpuKernelOutlining.cpp
  Transforms/RuntimeABIVerifier.cpp

  # Runtime Manager
  Frontend/RuntimeManager.cpp
  Frontend/ModuleRegistry.cpp
  SymbolRegistry.cpp

  # Runtime Source Injection
  ${RUNTIME_CORE_SOURCES}
)

# =========================================================
# 7. Build Targets (arkc & arknet)
# =========================================================
# Original raw driver (Optional, can be removed if arknet completely replaces it)
add_ark_tool(arkc
  main.cpp
  ${ARK_COMPILER_SHARED_SOURCES}
)

# New CLI11 package manager & orchestrator
add_ark_tool(arknet
  arknet.cpp
  CLI/Init.cpp
  CLI/Compile.cpp
  CLI/Registry.cpp
  CLI/Lsp.cpp
  CLI/Provider.cpp
  CLI/Config.cpp
  CLI/Doctor.cpp
  CLI/Workspace.cpp
  ${ARK_COMPILER_SHARED_SOURCES}
)

# =========================================================
# 8. Common Target Configuration
# =========================================================
# Iterate over both targets to apply includes, deps, and libs consistently
foreach(TOOL_TARGET arkc arknet)

  add_dependencies(${TOOL_TARGET}
    ArkMirDialectIncGen
    ArkMirOpsIncGen
    ArkMirTypesIncGen
    PrepareRuntimeHeaders
  )

  target_include_directories(${TOOL_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/ark-ir/include
    ${CMAKE_SOURCE_DIR}/libs/ark-abi/include
    ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include
    ${CMAKE_SOURCE_DIR}/libs/ark-crypto/include
    ${CMAKE_BINARY_DIR}/libs/ark-ir/IR
    ${CMAKE_BINARY_DIR}/include
    ${RUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${TOOL_TARGET} PRIVATE
    # --- Platinum Architecture Libs ---
    ark-capsule
    ark-diagnostics
    ark-abi
    ark-wire
    Ark::Crypto

    # --- Ark Monorepo Libs ---
    ArkMirDialect
    ArkToLLVM

    # --- LLVM Libs ---
    ${llvm_libs}

    # --- MLIR Core & Conversions ---
    MLIRIR MLIRParser MLIRSupport MLIRPass MLIRTransforms MLIRExecutionEngine
    MLIRTargetLLVMIRExport MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
    MLIRCallInterfaces MLIRAnalysis MLIRDataLayoutInterfaces
    MLIRUBToLLVM MLIRVectorToLLVM MLIRComplexToLLVM

    # --- Dialects ---
    MLIRFuncDialect MLIRArithDialect MLIRControlFlowDialect MLIRMemRefDialect
    MLIRSCFDialect MLIRLLVMDialect MLIRVectorDialect MLIRTensorDialect
    MLIRGPUDialect MLIRNVVMDialect MLIRROCDLDialect MLIRNVVMToLLVM

    # --- GPU Libraries ---
    MLIRGPUTransforms MLIRGPUPipelines MLIRGPUToNVVMTransforms MLIRGPUToROCDLTransforms
    MLIRGPUToLLVMIRTranslation MLIRNVVMToLLVMIRTranslation MLIRROCDLToLLVMIRTranslation

    # --- Conversions ---
    MLIRFuncToLLVM MLIRControlFlowToLLVM MLIRArithToLLVM MLIRLLVMCommonConversion
    MLIRReconcileUnrealizedCasts MLIRSCFToControlFlow MLIRMemRefToLLVM MLIRIndexToLLVM

    # --- Host Runtime ---
    ${ARK_PLATFORM_DL_LIBS}
  )

  # Export Symbols for JIT
  if(UNIX AND NOT APPLE)
    target_link_options(${TOOL_TARGET} PRIVATE "-rdynamic")
  elseif(APPLE)
    target_link_options(${TOOL_TARGET} PRIVATE "-Wl,-export_dynamic")
  endif()

  set_target_properties(${TOOL_TARGET} PROPERTIES ENABLE_EXPORTS 1)

endforeach()

# [FIX] Consolidated external package linking specifically for arknet
target_link_libraries(arknet PRIVATE CLI11::CLI11 tomlplusplus::tomlplusplus)

# =========================================================
# 9. Build the Capsule Stub (ark-stub)
# =========================================================
add_executable(ark-stub Runner/Stub.cpp)
target_compile_definitions(ark-stub PRIVATE ARK_STUB_STANDALONE)
target_include_directories(ark-stub PRIVATE
  ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include
  ${CMAKE_SOURCE_DIR}/libs/ark-crypto/include
)
target_link_libraries(ark-stub PRIVATE
  ark-capsule
  Ark::Crypto
)
if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
  target_compile_options(ark-stub PRIVATE -Os)
  if(UNIX AND NOT APPLE)
    target_link_options(ark-stub PRIVATE -Wl,--strip-all)
  endif()
endif()

# =========================================================
# 10. Installation
# =========================================================
install(TARGETS arkc arknet ark-stub DESTINATION bin)

# =========================================================
# 11. Runtime Headers Preparation
# =========================================================
add_custom_target(PrepareRuntimeHeaders ALL
  COMMENT "Copying Protocol & ABI headers to Runtime/include..."
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/libs/ark-abi/include/
    ${RUNTIME_DIR}/include/
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/libs/ark-wire/include/
    ${RUNTIME_DIR}/include/
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/libs/ark-capsule/include/
    ${RUNTIME_DIR}/include/
)

# =========================================================
# 12. GPU Plugins (SHARED)
# =========================================================
add_subdirectory(Runtime/gpu)