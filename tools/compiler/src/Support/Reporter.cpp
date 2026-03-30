#include "ark/compiler/Support/Reporter.hpp"
#include <iomanip>
#include <sstream>

namespace ark::compiler {

using namespace ark::diagnostics;

// Helper helpers
static std::string hex_val(uint32_t v, int width) {
    std::ostringstream ss;
    ss << "0x" << std::hex << std::setw(width) << std::setfill('0') << v;
    return ss.str();
}

void Reporter::ReportHuman(std::ostream& out, VerifyStatus status, const VerifyDiag& d, bool color) {
    std::string red = color ? "\033[1;31m" : "";
    std::string yel = color ? "\033[1;33m" : "";
    std::string rst = color ? "\033[0m" : "";

    out << "\n" << red << "[ERROR] Verification Failed: " << to_string(status) << rst << "\n";
    out << "  Rule:    " << to_string(d.rule) << "\n";
    out << "  Details: " << rule_help(d.rule) << "\n";
    
    std::string hint = rule_hint(d.rule);
    if (!hint.empty()) {
        out << "  -> " << yel << hint << rst << "\n";
    }

    out << "\n  Context:\n";
    out << "    Stage:   " << to_string(d.stage) << "\n";
    out << "    Offset:  " << hex_val(d.offset, 8) << "\n";

    if (d.kind == DiagKind::U32) {
        out << "    Expect:  " << hex_val(d.expected_u32, 8) << "\n";
        out << "    Actual:  " << hex_val(d.actual_u32, 8) << "\n";
    } else if (d.kind == DiagKind::U16) {
        out << "    Expect:  " << hex_val(d.expected_u16, 4) << "\n";
        out << "    Actual:  " << hex_val(d.actual_u16, 4) << "\n";
    }

    if (d.stage == VerifyStage::PayloadTLV) {
        out << "    Tag:     " << hex_val(d.tlv_tag, 4) << "\n";
        out << "    Len:     " << hex_val(d.tlv_len, 4) << "\n";
        if (d.rule == DiagRule::PayloadTagOrder) {
            out << "    LastTag: " << hex_val(d.last_tag, 4) << "\n";
        }
    }
    
    if (d.rule == DiagRule::MissingRequired) {
        out << "    Missing: " << hex_val(d.missing_mask, 8) << "\n";
    }
    out << std::endl;
}

void Reporter::ReportJson(std::ostream& out, VerifyStatus status, const VerifyDiag& d) {
    out << "{\n";
    out << "  \"status\": \"" << json_escape(to_string(status)) << "\",\n";
    out << "  \"rule\": \"" << json_escape(to_string(d.rule)) << "\",\n";
    out << "  \"help\": \"" << json_escape(rule_help(d.rule)) << "\",\n";
    out << "  \"hint\": \"" << json_escape(rule_hint(d.rule)) << "\",\n";
    out << "  \"stage\": \"" << json_escape(to_string(d.stage)) << "\",\n";
    out << "  \"offset\": " << d.offset << ",\n";
    out << "  \"offset_hex\": \"" << hex_val(d.offset, 8) << "\",\n";
    
    // Topology
    out << "  \"total_len\": " << d.total_len << ",\n";
    out << "  \"signed_len\": " << d.signed_len << ",\n";

    // Expected / Actual
    if (d.kind == DiagKind::U32) {
        out << "  \"expected\": " << d.expected_u32 << ",\n";
        out << "  \"actual\": " << d.actual_u32 << ",\n";
        out << "  \"expected_hex\": \"" << hex_val(d.expected_u32, 8) << "\",\n";
        out << "  \"actual_hex\": \"" << hex_val(d.actual_u32, 8) << "\",\n";
    } else if (d.kind == DiagKind::U16) {
        out << "  \"expected\": " << d.expected_u16 << ",\n";
        out << "  \"actual\": " << d.actual_u16 << ",\n";
        out << "  \"expected_hex\": \"" << hex_val(d.expected_u16, 4) << "\",\n";
        out << "  \"actual_hex\": \"" << hex_val(d.actual_u16, 4) << "\",\n";
    } else {
        out << "  \"value_kind\": \"none\",\n";
    }

    // TLV Context
    if (d.stage == VerifyStage::PayloadTLV) {
        out << "  \"tlv_tag\": " << d.tlv_tag << ",\n";
        out << "  \"tlv_tag_hex\": \"" << hex_val(d.tlv_tag, 4) << "\",\n";
        out << "  \"tlv_len\": " << d.tlv_len << ",\n";
        out << "  \"last_tag\": " << d.last_tag << ",\n";
        out << "  \"missing_mask\": " << d.missing_mask << "\n";
    } else {
        out << "  \"tlv_context\": null\n";
    }

    out << "}\n";
}

} // namespace ark::compiler