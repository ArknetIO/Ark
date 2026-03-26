// tools/arkc/hud.h
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace arklang::hud {

enum class Stream : uint8_t { Err, Out };

struct Theme {
  bool useColor = true;
  bool unicode = true;
  bool animations = true;

  bool verbose = false;
  bool compact = false;
  bool showBanner = true;
  bool showTimings = true;
  bool showHints = true;

  bool keepHistory = true;
  Stream stream = Stream::Err;
};

struct StepMeta {
  std::string title;
  std::string detail;
  std::optional<std::string> hint;

  StepMeta() = default;
  StepMeta(std::string_view t, std::string_view d, std::optional<std::string> h = std::nullopt)
      : title(t), detail(d), hint(std::move(h)) {}
};

struct Summary {
  int stepsTotal = 0;
  int stepsDone = 0;
  bool ok = false;
  std::chrono::nanoseconds wall{0};
  std::string label;
};

class Hud;

class Step {
public:
  Step(Hud &hud, StepMeta meta);
  ~Step();

  Step(const Step &) = delete;
  Step &operator=(const Step &) = delete;

  Step(Step &&o) noexcept : hud_(o.hud_), closed_(o.closed_) { o.hud_ = nullptr; }
  Step &operator=(Step &&o) noexcept {
    if (this == &o) return *this;
    hud_ = o.hud_;
    closed_ = o.closed_;
    o.hud_ = nullptr;
    return *this;
  }

  void detail(std::string_view d);
  void hint(std::string_view h);

  void ok();
  void fail();

private:
  Hud *hud_ = nullptr;
  bool closed_ = false;
};

class Hud {
public:
  Hud(Theme t, int totalSteps);
  ~Hud();

  Hud(const Hud &) = delete;
  Hud &operator=(const Hud &) = delete;

  void banner(std::string_view toolName, std::string_view version, std::string_view targetTripleHint);

  void setLabel(std::string_view label);
  void setTotalSteps(int totalSteps);
  bool isVerbose() const;

  // Safe at any time: pauses spinner, clears HUD line, prints, restores.
  void printErr(std::string_view text);
  void printOut(std::string_view text);

  void note(std::string_view msg);
  void debug(std::string_view msg);
  void warn(std::string_view msg);
  void error(std::string_view msg);

  void stepBegin(const StepMeta &meta);
  void stepDetail(std::string_view detail);
  void stepHint(std::string_view hint);
  void stepOk();
  void stepFail();

  // Wrap pass-through output (clang/ld/child stdout/stderr) with these.
  void onDiagnosticsBegin();
  void onDiagnosticsEnd();

  class ScopedDiagnostics {
  public:
    explicit ScopedDiagnostics(Hud &h) : hud_(&h) { hud_->onDiagnosticsBegin(); }
    ~ScopedDiagnostics() { if (hud_) hud_->onDiagnosticsEnd(); }

    ScopedDiagnostics(const ScopedDiagnostics &) = delete;
    ScopedDiagnostics &operator=(const ScopedDiagnostics &) = delete;

    ScopedDiagnostics(ScopedDiagnostics &&o) noexcept : hud_(o.hud_) { o.hud_ = nullptr; }
    ScopedDiagnostics &operator=(ScopedDiagnostics &&o) noexcept {
      if (this == &o) return *this;
      hud_ = o.hud_;
      o.hud_ = nullptr;
      return *this;
    }

  private:
    Hud *hud_ = nullptr;
  };

  void flush();
  Summary finish(bool ok);

  // External code can attach captured output; flush() prints blocks cleanly.
  void pushLogBlock(std::string_view title, std::string_view body);

private:
  struct State {
    int index = 0;
    int total = 0;

    bool inStep = false;
    bool ok = true;

    std::string label;
    std::string currentTitle;
    std::string currentDetail;
    std::optional<std::string> currentHint;

    std::chrono::steady_clock::time_point stepStart{};
    std::chrono::nanoseconds lastStep{0};
  };

  struct LogBlock {
    std::string title;
    std::string body;
  };

  bool supportsColor() const;
  bool supportsUnicode() const;

  std::string color(std::string_view ansi) const;
  std::string reset() const;

  void printRaw(Stream s, std::string_view text);
  void printLn(Stream s, std::string_view text);

  void renderBannerLocked(std::string_view toolName, std::string_view version, std::string_view triple);

  std::string fmtStepPrefixLocked() const;
  std::string fmtStatusSpinLocked() const;
  std::string fmtStatusOkLocked() const;
  std::string fmtStatusFailLocked() const;
  std::string fmtTimeLocked(std::chrono::nanoseconds ns) const;

  void clearLineLocked();
  void renderLineLocked();

  void startSpinnerLocked();
  void stopSpinner();

  // If we were mid-step and not verbose, restore the spinner + line.
  void maybeResumeSpinnerLocked();

  void renderBlockLocked(const LogBlock &b);

  // Core primitive for all out-of-band lines:
  // stops spinner (no deadlock), clears HUD line, prints, then restores.
  void emitLine(Stream s, std::string_view text);

private:
  Theme theme_;
  State st_;

  std::chrono::steady_clock::time_point start_{};

  std::mutex mu_;
  std::atomic<int> diagDepth_{0};

  std::atomic<bool> spinnerRun_{false};
  std::atomic<int> spinnerFrame_{0};
  std::thread spinner_;

  std::vector<LogBlock> blocks_;
};

} // namespace arklang::hud
