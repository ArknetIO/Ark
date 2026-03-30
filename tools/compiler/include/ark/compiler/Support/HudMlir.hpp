#pragma once

#include "ark/compiler/Support/Hud.hpp"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h" // [FIX] Added for StringAttr
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace arklang::hud {

static inline bool isUnknownLoc(mlir::Location loc) {
  return llvm::isa<mlir::UnknownLoc>(loc);
}

static inline std::string flcToString(mlir::FileLineColLoc flc) {
  std::string out;
  llvm::raw_string_ostream os(out);
  // flc.getFilename() returns StringAttr, getValue() returns StringRef.
  // Now valid because BuiltinAttributes.h is included.
  os << flc.getFilename().getValue() << ":" << flc.getLine() << ":" << flc.getColumn();
  return os.str();
}

static inline mlir::FileLineColLoc peelToFileLineCol(mlir::Location loc) {
  // NOTE: mlir::Location is not nullable, so we only test for UnknownLoc.
  while (!isUnknownLoc(loc)) {
    if (auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) return flc;

    if (auto name = llvm::dyn_cast<mlir::NameLoc>(loc)) {
      loc = name.getChildLoc();
      continue;
    }

    if (auto cs = llvm::dyn_cast<mlir::CallSiteLoc>(loc)) {
      // Prefer callee; if it’s unknown, fall back to caller.
      auto callee = cs.getCallee();
      loc = !isUnknownLoc(callee) ? callee : cs.getCaller();
      continue;
    }

    if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
      auto locs = fused.getLocations();
      if (locs.empty()) break;

      // Pick first non-unknown, else first.
      mlir::Location chosen = locs.front();
      for (mlir::Location l : locs) {
        if (!isUnknownLoc(l)) { chosen = l; break; }
      }
      loc = chosen;
      continue;
    }

    if (auto opaque = llvm::dyn_cast<mlir::OpaqueLoc>(loc)) {
      loc = opaque.getFallbackLocation();
      continue;
    }

    break;
  }
  return mlir::FileLineColLoc(); // null FileLineColLoc (prints as empty by our formatter)
}

static inline std::string formatArkLocPrefix(mlir::Location loc) {
  auto flc = peelToFileLineCol(loc);
  if (!flc) return {};
  return flcToString(flc);
}

static inline void stripLeadingMlirLocPrefix(std::string &s) {
  // If MLIR prints: loc(...): <message>
  if (s.rfind("loc(", 0) != 0) return;
  auto p = s.find("): ");
  if (p == std::string::npos) return;
  s.erase(0, p + 3);
}

class MlirDiagBridge final {
public:
  MlirDiagBridge(Hud &hud, mlir::MLIRContext &ctx)
      : hud_(hud),
        handler_(&ctx, [this](mlir::Diagnostic &d) -> mlir::LogicalResult {
          std::string msg;
          {
            llvm::raw_string_ostream os(msg);
            d.print(os);
          }

          stripLeadingMlirLocPrefix(msg);

          const std::string locPrefix = formatArkLocPrefix(d.getLocation());

          std::string out;
          {
            llvm::raw_string_ostream os(out);
            if (!locPrefix.empty()) os << locPrefix << ": ";
            os << msg;
          }

          hud_.onDiagnosticsBegin();
          hud_.error(out);
          hud_.onDiagnosticsEnd();
          return mlir::success();
        }) {
    ctx.printOpOnDiagnostic(true);
    // ctx.printStackTraceOnDiagnostic(true);
  }

  MlirDiagBridge(const MlirDiagBridge &) = delete;
  MlirDiagBridge &operator=(const MlirDiagBridge &) = delete;

private:
  Hud &hud_;
  mlir::ScopedDiagnosticHandler handler_;
};

} // namespace arklang::hud