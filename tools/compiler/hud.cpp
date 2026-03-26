// tools/arkc/hud.cpp
#include "hud.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace arklang::hud {

static bool envTrue(const char *k) {
  const char *v = std::getenv(k);
  if (!v) return false;

  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

static std::string padRight(std::string s, size_t w) {
  if (s.size() >= w) return s;
  s.append(w - s.size(), ' ');
  return s;
}

static std::string spaces(size_t n) { return std::string(n, ' '); }

static std::string trimFor(std::string_view s, size_t max) {
  if (s.size() <= max) return std::string(s);
  if (max <= 1) return "…";
  return std::string(s.substr(0, max - 1)) + "…";
}

static std::string_view trimCRLFEdges(std::string_view s) {
  while (!s.empty() && (s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

static bool endsWithLF(std::string_view s) { return !s.empty() && s.back() == '\n'; }

static void splitLinesCRLF(std::string_view s, std::vector<std::string_view> &out) {
  out.clear();
  if (s.empty()) {
    out.push_back(std::string_view{});
    return;
  }

  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find('\n', i);
    if (j == std::string_view::npos) {
      std::string_view ln = s.substr(i);
      if (!ln.empty() && ln.back() == '\r') ln.remove_suffix(1);
      out.push_back(ln);
      break;
    }

    std::string_view ln = s.substr(i, j - i);
    if (!ln.empty() && ln.back() == '\r') ln.remove_suffix(1);
    out.push_back(ln);
    i = j + 1;
  }
}

// ----- Hud -----

Hud::Hud(Theme t, int totalSteps) : theme_(t) {
  st_.total = totalSteps;
  st_.ok = true;
  start_ = std::chrono::steady_clock::now();

  if (envTrue("NO_COLOR")) theme_.useColor = false;
  if (envTrue("ARKC_NO_ANIM")) theme_.animations = false;
  if (envTrue("ARKC_VERBOSE")) theme_.verbose = true;
  if (envTrue("ARKC_COMPACT")) theme_.compact = true;
}

Hud::~Hud() { stopSpinner(); }

bool Hud::supportsColor() const {
  if (!theme_.useColor) return false;
  const char *term = std::getenv("TERM");
  if (!term) return false;
  return std::string(term) != "dumb";
}

bool Hud::supportsUnicode() const {
  if (!theme_.unicode) return false;
  const char *lang = std::getenv("LANG");
  if (!lang) return true;

  std::string s(lang);
  return s.find("UTF-8") != std::string::npos || s.find("utf8") != std::string::npos;
}

std::string Hud::color(std::string_view ansi) const {
  if (!supportsColor()) return "";
  return std::string(ansi);
}

std::string Hud::reset() const {
  if (!supportsColor()) return "";
  return "\033[0m";
}

void Hud::printRaw(Stream s, std::string_view text) {
  if (text.empty()) return;
  if (s == Stream::Err) llvm::errs() << text;
  else llvm::outs() << text;
}

void Hud::printLn(Stream s, std::string_view text) {
  if (text.empty()) return;

  // Do not destroy internal '\n' (blocks use them). Only trim *edge* CRLF so we don't
  // manufacture blank spacer lines.
  const std::string_view stripped = trimCRLFEdges(text);
  if (stripped.empty()) return;

  printRaw(s, stripped);

  // Critical: avoid double-newlines when caller already ended with '\n'.
  if (!endsWithLF(text)) printRaw(s, "\n");
}

void Hud::renderBannerLocked(std::string_view toolName, std::string_view version,
                             std::string_view triple) {
  if (!theme_.showBanner) return;

  const bool uni = supportsUnicode();
  const char *tl = uni ? "╭" : "+";
  const char *tr = uni ? "╮" : "+";
  const char *bl = uni ? "╰" : "+";
  const char *br = uni ? "╯" : "+";
  const char *hz = uni ? "─" : "-";
  const char *vt = uni ? "│" : "|";

  const std::string title = std::string(toolName) + " " + std::string(version);
  const std::string sub = std::string("target: ") + std::string(triple);

  const size_t w = std::max(title.size(), sub.size()) + 6;

  std::ostringstream o;
  o << color("\033[38;5;81m") << tl;
  for (size_t i = 0; i < w; ++i) o << hz;
  o << tr << reset() << "\n";

  o << color("\033[38;5;81m") << vt << reset() << "  " << color("\033[1m")
    << padRight(title, w - 4) << reset() << "  " << color("\033[38;5;81m") << vt
    << reset() << "\n";

  o << color("\033[38;5;81m") << vt << reset() << "  " << color("\033[38;5;245m")
    << padRight(sub, w - 4) << reset() << "  " << color("\033[38;5;81m") << vt
    << reset() << "\n";

  o << color("\033[38;5;81m") << bl;
  for (size_t i = 0; i < w; ++i) o << hz;
  o << br << reset();

  printLn(theme_.stream, o.str());
}

void Hud::banner(std::string_view toolName, std::string_view version,
                 std::string_view targetTripleHint) {
  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);
  clearLineLocked();
  renderBannerLocked(toolName, version, targetTripleHint);
  maybeResumeSpinnerLocked();
}

void Hud::setLabel(std::string_view label) {
  std::lock_guard<std::mutex> lock(mu_);
  st_.label = std::string(label);
}

void Hud::setTotalSteps(int totalSteps) {
  std::lock_guard<std::mutex> lock(mu_);
  st_.total = totalSteps;
}

bool Hud::isVerbose() const { return theme_.verbose; }

std::string Hud::fmtStepPrefixLocked() const {
  std::ostringstream o;
  const bool uni = supportsUnicode();
  const char *dot = uni ? "•" : "*";

  if (!st_.label.empty()) {
    o << color("\033[38;5;245m") << st_.label << reset() << " "
      << color("\033[38;5;245m") << dot << reset() << " ";
  }

  if (st_.total > 0) {
    o << color("\033[38;5;245m") << std::setw(2) << st_.index << "/" << std::setw(2)
      << st_.total << reset() << color("\033[38;5;245m") << " " << dot << " "
      << reset();
  }

  return o.str();
}

std::string Hud::fmtStatusSpinLocked() const {
  const bool uni = supportsUnicode();
  static const char *framesUni[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
  static const char *framesAsc[] = {"|","/","-","\\"};

  const int f = spinnerFrame_.load(std::memory_order_relaxed);
  const std::string g = color("\033[38;5;81m");
  const std::string r = reset();

  if (!theme_.animations) return g + "…" + r;
  return uni ? (g + framesUni[f % 10] + r) : (g + framesAsc[f % 4] + r);
}

std::string Hud::fmtStatusOkLocked() const {
  const bool uni = supportsUnicode();
  const std::string g = color("\033[38;5;82m");
  const std::string r = reset();
  return uni ? (g + "✓" + r) : (g + "OK" + r);
}

std::string Hud::fmtStatusFailLocked() const {
  const bool uni = supportsUnicode();
  const std::string g = color("\033[38;5;196m");
  const std::string r = reset();
  return uni ? (g + "✗" + r) : (g + "ERR" + r);
}

std::string Hud::fmtTimeLocked(std::chrono::nanoseconds ns) const {
  if (!theme_.showTimings) return "";
  using namespace std::chrono;
  const auto ms = duration_cast<milliseconds>(ns).count();
  std::ostringstream o;
  o << color("\033[38;5;245m") << " (" << ms << "ms)" << reset();
  return o.str();
}

void Hud::clearLineLocked() {
  if (theme_.verbose) return;
  printRaw(theme_.stream, "\r\033[K");
}

void Hud::renderLineLocked() {
  if (theme_.verbose) return;

  clearLineLocked();

  std::ostringstream o;
  o << fmtStepPrefixLocked();

  if (st_.inStep) o << fmtStatusSpinLocked() << " ";
  else o << (st_.ok ? fmtStatusOkLocked() : fmtStatusFailLocked()) << " ";

  std::string title = st_.currentTitle;
  if (!st_.currentDetail.empty()) title += " — " + st_.currentDetail;

  const size_t max = theme_.compact ? 72 : 96;
  o << color("\033[1m") << trimFor(title, max) << reset();

  if (!st_.inStep && theme_.showTimings) o << fmtTimeLocked(st_.lastStep);

  printRaw(theme_.stream, o.str());

  const bool emitHistory = theme_.keepHistory || !st_.ok;
  if (!st_.inStep && emitHistory) printRaw(theme_.stream, "\n");

  if (!st_.inStep && theme_.showHints && st_.currentHint && emitHistory) {
    std::ostringstream h;
    h << color("\033[38;5;245m") << "   ↳ " << *st_.currentHint << reset();
    printLn(theme_.stream, h.str());
  }
}

void Hud::startSpinnerLocked() {
  if (!theme_.animations || theme_.verbose) return;
  if (spinnerRun_.load(std::memory_order_relaxed)) return;

  spinnerRun_.store(true, std::memory_order_relaxed);

  spinner_ = std::thread([this] {
    while (spinnerRun_.load(std::memory_order_relaxed)) {
      spinnerFrame_.fetch_add(1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (diagDepth_.load(std::memory_order_relaxed) == 0) renderLineLocked();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(70));
    }
  });
}

void Hud::stopSpinner() {
  std::thread th;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!spinnerRun_.exchange(false, std::memory_order_relaxed)) return;
    th = std::move(spinner_);
  }
  if (th.joinable()) th.join();
}

void Hud::maybeResumeSpinnerLocked() {
  if (theme_.verbose) return;
  if (!st_.inStep) return;
  if (diagDepth_.load(std::memory_order_relaxed) != 0) return;

  if (!spinnerRun_.load(std::memory_order_relaxed)) startSpinnerLocked();
  renderLineLocked();
}

void Hud::emitLine(Stream s, std::string_view text) {
  const std::string_view stripped = trimCRLFEdges(text);
  if (stripped.empty()) return;

  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);
  clearLineLocked();
  printLn(s, stripped);
  maybeResumeSpinnerLocked();
}

void Hud::printErr(std::string_view text) { emitLine(Stream::Err, text); }
void Hud::printOut(std::string_view text) { emitLine(Stream::Out, text); }

void Hud::note(std::string_view msg) {
  std::ostringstream o;
  o << color("\033[38;5;245m") << "• " << reset() << msg;
  emitLine(theme_.stream, o.str());
}

void Hud::debug(std::string_view msg) {
  std::ostringstream o;
  // Use ANSI 105 (Light Purple) + Gear to distinguish from Grey Notes
  o << color("\033[38;5;105m") << "⚙" << reset() << " " << msg;
  emitLine(theme_.stream, o.str());
}

void Hud::warn(std::string_view msg) {
  std::ostringstream o;
  o << color("\033[38;5;214m") << "⚠" << reset() << " " << msg;
  emitLine(theme_.stream, o.str());
}

void Hud::error(std::string_view msg) {
  std::ostringstream o;
  o << color("\033[38;5;196m") << "✗" << reset() << " " << msg;
  emitLine(theme_.stream, o.str());
}

void Hud::stepBegin(const StepMeta &meta) {
  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);

  st_.index += 1;
  st_.inStep = true;
  st_.ok = true;
  st_.currentTitle = meta.title;
  st_.currentDetail = meta.detail;
  st_.currentHint = meta.hint;
  st_.stepStart = std::chrono::steady_clock::now();
  st_.lastStep = std::chrono::nanoseconds(0);

  if (theme_.verbose) {
    std::ostringstream o;
    o << ">> [" << st_.index << "/" << st_.total << "] " << st_.currentTitle;
    if (!st_.currentDetail.empty()) o << " — " << st_.currentDetail;
    printLn(theme_.stream, o.str());
    return;
  }

  renderLineLocked();
  if (diagDepth_.load(std::memory_order_relaxed) == 0) startSpinnerLocked();
}

void Hud::stepDetail(std::string_view detail) {
  std::lock_guard<std::mutex> lock(mu_);
  st_.currentDetail = std::string(detail);

  if (theme_.verbose) {
    std::ostringstream o;
    o << "   " << detail;
    printLn(theme_.stream, o.str());
    return;
  }

  if (st_.inStep && diagDepth_.load(std::memory_order_relaxed) == 0) renderLineLocked();
}

void Hud::stepHint(std::string_view hint) {
  std::lock_guard<std::mutex> lock(mu_);
  st_.currentHint = std::string(hint);

  if (theme_.verbose) {
    std::ostringstream o;
    o << "   hint: " << hint;
    printLn(theme_.stream, o.str());
  }
}

void Hud::stepOk() {
  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);

  st_.inStep = false;
  st_.ok = true;
  st_.lastStep = std::chrono::steady_clock::now() - st_.stepStart;

  if (!theme_.keepHistory && !theme_.verbose) {
    clearLineLocked();
    return;
  }
  renderLineLocked();
}

void Hud::stepFail() {
  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);

  st_.inStep = false;
  st_.ok = false;
  st_.lastStep = std::chrono::steady_clock::now() - st_.stepStart;

  renderLineLocked();
}

void Hud::renderBlockLocked(const LogBlock &b) {
  clearLineLocked();

  const bool uni = supportsUnicode();
  const char *tl = uni ? "┌" : "+";
  const char *tr = uni ? "┐" : "+";
  const char *bl = uni ? "└" : "+";
  const char *br = uni ? "┘" : "+";
  const char *hz = uni ? "─" : "-";
  const char *vt = uni ? "│" : "|";

  std::vector<std::string_view> lines;
  splitLinesCRLF(b.body, lines);

  size_t contentW = 0;
  for (auto ln : lines) contentW = std::max(contentW, ln.size());

  constexpr size_t kMaxW = 120;
  contentW = std::min(contentW, kMaxW);

  const std::string title = b.title;
  const size_t titleW = std::min(title.size(), kMaxW);

  const size_t w = std::max(contentW, titleW) + 2; // left+right padding

  auto top = [&] {
    std::ostringstream o;
    o << color("\033[38;5;81m") << tl;
    for (size_t i = 0; i < w + 2; ++i) o << hz;
    o << tr << reset();
    printLn(theme_.stream, o.str());
  };

  auto row = [&](std::string_view payload, bool bold, bool dim) {
    const std::string clipped = trimFor(payload, w);
    const size_t used = std::min(clipped.size(), w);
    const size_t pad = (w > used) ? (w - used) : 0;

    std::ostringstream o;
    o << color("\033[38;5;81m") << vt << reset() << " ";
    if (bold) o << color("\033[1m");
    if (dim) o << color("\033[38;5;245m");
    o << clipped << reset() << spaces(pad) << " " << color("\033[38;5;81m") << vt
      << reset();
    printLn(theme_.stream, o.str());
  };

  auto bot = [&] {
    std::ostringstream o;
    o << color("\033[38;5;81m") << bl;
    for (size_t i = 0; i < w + 2; ++i) o << hz;
    o << br << reset();
    printLn(theme_.stream, o.str());
  };

  top();
  row(title, /*bold=*/true, /*dim=*/false);
  for (auto ln : lines) row(ln, /*bold=*/false, /*dim=*/true);
  bot();
}

void Hud::flush() {
  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &b : blocks_) renderBlockLocked(b);
  blocks_.clear();
  maybeResumeSpinnerLocked();
}

Summary Hud::finish(bool ok) {
  stopSpinner();

  Summary s;
  std::lock_guard<std::mutex> lock(mu_);

  for (const auto &b : blocks_) renderBlockLocked(b);
  blocks_.clear();

  s.stepsTotal = st_.total;
  s.stepsDone = st_.index;
  s.ok = ok;
  s.wall = std::chrono::steady_clock::now() - start_;
  s.label = st_.label;

  clearLineLocked();

  std::ostringstream o;
  if (ok) o << fmtStatusOkLocked() << " " << color("\033[1m") << "Done" << reset();
  else o << fmtStatusFailLocked() << " " << color("\033[1m") << "Failed" << reset();

  if (theme_.showTimings) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(s.wall).count();
    o << color("\033[38;5;245m") << " in " << ms << "ms" << reset();
  }

  printLn(theme_.stream, o.str());
  return s;
}

void Hud::onDiagnosticsBegin() {
  stopSpinner();
  std::lock_guard<std::mutex> lock(mu_);
  diagDepth_.fetch_add(1, std::memory_order_relaxed);

  // IMPORTANT: do not inject '\n' here (causes blank lines when no tool output).
  // Just ensure the in-flight HUD line isn't left half-written.
  clearLineLocked();
}

void Hud::onDiagnosticsEnd() {
  std::lock_guard<std::mutex> lock(mu_);
  const int v = diagDepth_.fetch_sub(1, std::memory_order_relaxed);
  if (v <= 1) maybeResumeSpinnerLocked();
}

void Hud::pushLogBlock(std::string_view title, std::string_view body) {
  std::lock_guard<std::mutex> lock(mu_);
  blocks_.push_back(LogBlock{std::string(title), std::string(body)});
}

// ----- Step -----

Step::Step(Hud &hud, StepMeta meta) : hud_(&hud) { hud_->stepBegin(meta); }

Step::~Step() {
  if (hud_ && !closed_) hud_->stepFail();
}

void Step::detail(std::string_view d) {
  if (hud_ && !closed_) hud_->stepDetail(d);
}

void Step::hint(std::string_view h) {
  if (hud_ && !closed_) hud_->stepHint(h);
}

void Step::ok() {
  if (!hud_ || closed_) return;
  hud_->stepOk();
  closed_ = true;
}

void Step::fail() {
  if (!hud_ || closed_) return;
  hud_->stepFail();
  closed_ = true;
}

} // namespace arklang::hud
