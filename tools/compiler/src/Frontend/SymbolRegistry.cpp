#include <llvm/Support/DynamicLibrary.h>
#include <iostream>

namespace arklang {

// -----------------------------------------------------------------------------
// Zero-Maintenance Registry
// -----------------------------------------------------------------------------
// Because we compiled 'arkc' with -rdynamic (Linux) or similar, all 
// extern "C" functions in our runtime sources are automatically exposed.
// We just need to permit the JIT to search the main process memory.

void registerRuntimeSymbols() {
    // This tells LLVM: "Search the current process for symbols"
    // It works for 'printf', 'malloc', AND your 'ark_*' functions.
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
}

} // namespace arklang