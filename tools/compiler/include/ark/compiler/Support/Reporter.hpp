#pragma once
#include <string>
#include <iostream>
#include <ark/diagnostics/Diagnostic.h>

namespace ark::compiler {

class Reporter {
public:
    // Human-readable, colored output
    static void ReportHuman(
        std::ostream& out,
        ark::diagnostics::LoaderStatus status,
        const ark::diagnostics::VerifyDiag& diag,
        bool color
    );

    // Machine-readable JSON output
    static void ReportJson(
        std::ostream& out,
        ark::diagnostics::LoaderStatus status,
        const ark::diagnostics::VerifyDiag& diag
    );

private:
    static std::string hex_u16(uint16_t v);
    static std::string hex_u32(uint32_t v);
};

} // namespace ark::compiler