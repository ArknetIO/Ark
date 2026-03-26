cmake_minimum_required(VERSION 3.13)

project(ArkRuntime LANGUAGES C CXX)

include(GNUInstallDirs)

# =========================================================
# 0. Defaults
# =========================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================================================
# 1. The Core Runtime (Static Library)
# =========================================================
# Contains HAL logic, no vendor GPU code. Uses dlopen()/LoadLibrary.

set(CORE_SOURCES
    core/async.cpp
    core/memory.cpp
    core/string.cpp
    core/panic.cpp
    core/hash.cpp
    core/print.cpp
    core/vector.cpp
    core/gpu_hal.cpp
    fs/fs_posix.cpp
    sys/sys_posix.cpp
    net/socket_posix.cpp
    net/remote.cpp
)

add_library(ArkRuntime STATIC ${CORE_SOURCES})

target_include_directories(ArkRuntime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

target_compile_features(ArkRuntime PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(ArkRuntime PRIVATE /W4)
else()
    target_compile_options(ArkRuntime PRIVATE -fPIC)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(ArkRuntime PRIVATE ${CMAKE_DL_LIBS})
endif()

# =========================================================
# 2. Build Plugins (Shared Libraries)
# =========================================================

# Plugins are optional; build only if SDKs are found.
# --- A. Apple Metal Plugin ---
if(APPLE)
    find_library(METAL_LIB Metal)
    find_library(FOUNDATION_LIB Foundation)

    if(METAL_LIB AND FOUNDATION_LIB)
        message(STATUS "[Ark] Found Metal: Building libark_metal_backend")

        add_library(ark_metal_backend SHARED gpu/metal_backend.mm)
        target_link_libraries(ark_metal_backend PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})

        target_compile_definitions(ark_metal_backend PRIVATE ARK_BACKEND_METAL)
        target_include_directories(ark_metal_backend PRIVATE include)
        target_compile_features(ark_metal_backend PRIVATE cxx_std_17)

        set_target_properties(ark_metal_backend PROPERTIES PREFIX "lib")
    endif()
endif()

# --- B. NVIDIA CUDA Plugin ---
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "[Ark] Found CUDA ${CUDAToolkit_VERSION}: Building libark_cuda_backend")

    add_library(ark_cuda_backend SHARED gpu/cuda_backend.cpp)
    target_link_libraries(ark_cuda_backend PRIVATE CUDA::cuda_driver)

    target_compile_definitions(ark_cuda_backend PRIVATE ARK_BACKEND_CUDA)
    target_include_directories(ark_cuda_backend PRIVATE include)
    target_compile_features(ark_cuda_backend PRIVATE cxx_std_17)

    set_target_properties(ark_cuda_backend PROPERTIES PREFIX "lib")
else()
    message(STATUS "[Ark] CUDA not found. Skipping CUDA plugin.")
endif()

# --- C. AMD HIP Plugin ---
if(EXISTS "/opt/rocm/hip")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    find_package(hip QUIET)

    if(hip_FOUND)
        message(STATUS "[Ark] Found HIP: Building libark_hip_backend")

        add_library(ark_hip_backend SHARED gpu/hip_backend.cpp)
        target_link_libraries(ark_hip_backend PRIVATE hip::host)

        target_compile_definitions(ark_hip_backend PRIVATE ARK_BACKEND_HIP)
        target_include_directories(ark_hip_backend PRIVATE include)
        target_compile_features(ark_hip_backend PRIVATE cxx_std_17)

        set_target_properties(ark_hip_backend PROPERTIES PREFIX "lib")
    else()
        message(STATUS "[Ark] ROCm directory found but 'hip' package missing. Skipping HIP plugin.")
    endif()
else()
    message(STATUS "[Ark] ROCm/HIP not found. Skipping HIP plugin.")
endif()


# =========================================================
# 3. Legacy Config Generation (Optional)
# =========================================================
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ark_config.h.in"
    "${CMAKE_BINARY_DIR}/include/ark_config.h"
)

# =========================================================
# 4. Optional: Install (kept simple)
# =========================================================
install(TARGETS ArkRuntime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TARGET ark_cuda_backend)
    install(TARGETS ark_cuda_backend
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(TARGET ark_hip_backend)
    install(TARGETS ark_hip_backend
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(TARGET ark_metal_backend)
    install(TARGETS ark_metal_backend
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
