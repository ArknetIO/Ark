#include <ark_protocol.h>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>  // [FIX] size_t bounds check
#include <mutex>
#include <random>  // [FIX] portable entropy
#include <thread>
#include <vector>
#include <new>     // std::nothrow

#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


extern "C" {

// =============================================================================
// Runtime Configuration Stub (Presets)
// =============================================================================
// In a full implementation, this would access thread-local storage stacks.
// For the MVP, we return a null handle (0) which means "Default / Local".

ArkPresetHandle __ark_runtime_preset_current(void) {
    return 0; // 0 = Default Context
}

void __ark_runtime_preset_release(ArkPresetHandle) {
    // No-op for MVP (int64_t handle doesn't need freeing)
}

const char* __ark_runtime_preset_target(ArkPresetHandle) {
    // Return empty string -> run locally
    // Return "runtimes.arknet.io" -> run remote
    return ""; 
}

const char* __ark_runtime_preset_endpoint(ArkPresetHandle) {
    return "";
}

const char* __ark_runtime_preset_token(ArkPresetHandle) {
    return "";
}

int32_t __ark_runtime_preset_timeout_ms(ArkPresetHandle) {
    return -1; // Infinite wait
}

void __ark_runtime_preset_pop_all(void) {
    // No-op
}

// Simple heuristic: if target string is not empty, assume remote.
bool __ark_runtime_target_is_remote(const char* target) {
    if (!target || !*target) return false;
    return true; 
}

} // extern "C"

// =============================================================================
// Remote Backend Contract
// =============================================================================
extern "C" {
    // Return: 0=Async Accepted, 1=Inline Complete, <0=Failed
    int  __ark_remote_submit(
        void* task_ptr,
        std::uint64_t uid_lo,   // [UPDATED]
        std::uint64_t uid_hi,   // [UPDATED]
        std::uint64_t cookie,
        const char* target,
        const char* endpoint,
        const char* token
    );
    void __ark_remote_cancel(void* task_ptr);
}
namespace ark::rt {

static inline void fatal(const char* msg) noexcept {
    std::fprintf(stderr, "[ARK RT] Fatal: %s\n", msg);
    std::abort();
}

static inline void warn(const char* msg) noexcept {
    std::fprintf(stderr, "[ARK RT] Warning: %s\n", msg);
}

// =============================================================================
// Security: Unforgeable Cookies (Portable Entropy + SplitMix64)
// =============================================================================
static inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static std::atomic<std::uint64_t> g_cookie_ctr{1};
static std::atomic<std::uint64_t> g_bad_completions{0}; // [METRICS] silent drop counter

static inline std::uint64_t cookie_secret() noexcept {
    // [FIX] One-time secret with best-effort entropy; never throws.
    static const std::uint64_t s = []{
        std::uint64_t r = 0;
        try {
            std::random_device rd;
            const std::uint64_t hi = static_cast<std::uint64_t>(rd());
            const std::uint64_t lo = static_cast<std::uint64_t>(rd());
            r = (hi << 32) ^ lo;
        } catch (...) {
            r = 0;
        }

        const std::uint64_t a = static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(&g_cookie_ctr)
        );
        const std::uint64_t b = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        return splitmix64(r ^ a ^ (b + 0xD1B54A32D192ED03ull));
    }();
    return s;
}

static inline std::uint64_t make_cookie(std::uint32_t idx, std::uint32_t gen) noexcept {
    const std::uint64_t ctr = g_cookie_ctr.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t x =
        (static_cast<std::uint64_t>(idx) << 32) ^
        static_cast<std::uint64_t>(gen) ^
        (ctr * 0x9E3779B97F4A7C15ull) ^
        cookie_secret();
    return splitmix64(x);
}

// =============================================================================
// Telemetry (TLS Ring Buffer, No Locks)
// =============================================================================
enum EventType : std::uint8_t {
    EVT_LAUNCH = 1,
    EVT_COMPLETE = 2,
    EVT_CONSUME = 3,
    EVT_REMOTE = 4,
    EVT_SCOPE_EXIT = 5
};

struct TraceEvent {
    std::uint64_t ts;
    std::uint64_t entity;
    std::uint32_t data;
    std::uint8_t  type;
    std::uint8_t  pad[3];
};

struct TraceBuf final {
    static constexpr std::size_t kN = 4096;
    std::array<TraceEvent, kN> buf{};
    std::uint32_t head = 0;

    static std::uint64_t read_ticks() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        unsigned aux;
        return __rdtscp(&aux);
#else
        return static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
#endif
    }

    void push(EventType t, std::uint64_t ent, std::uint32_t d) noexcept {
        const std::uint32_t i = head++ & (kN - 1);
        buf[i] = { read_ticks(), ent, d, static_cast<std::uint8_t>(t), {0,0,0} };
    }
};

static thread_local TraceBuf tls_trace;

static inline void trace(EventType t, std::uint64_t ent, std::uint32_t d) noexcept {
    tls_trace.push(t, ent, d);
}

// =============================================================================
// Inflight & Backpressure (Rate-Limited Warning)
// =============================================================================
static constexpr std::int64_t MAX_GLOBAL_INFLIGHT = 1000000;

static std::atomic<std::int64_t> g_inflight{0};
static std::mutex g_inflight_mu;
static std::condition_variable g_inflight_cv;

static inline std::uint64_t steady_now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

static std::atomic<std::uint64_t> g_bp_last_warn_ns{0}; // [FIX] monotonic nanos

static void throttle_launch() noexcept {
    int spins = 0;
    while (g_inflight.load(std::memory_order_relaxed) >= MAX_GLOBAL_INFLIGHT) {
        if (spins++ < 100) {
            std::this_thread::yield();
            continue;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // [FIX] warn after ~2s+ and then rate-limit to <= 1/sec while throttled
        if (spins >= 2000) {
            const std::uint64_t now = steady_now_ns();
            const std::uint64_t prev = g_bp_last_warn_ns.load(std::memory_order_relaxed);
            if (now - prev > 1'000'000'000ull) {
                if (g_bp_last_warn_ns.exchange(now, std::memory_order_relaxed) == prev) {
                    warn("Backpressure active > 2s.");
                }
            }
        }
    }
}

static inline void inflight_inc() noexcept {
    g_inflight.fetch_add(1, std::memory_order_acq_rel);
}

static inline void inflight_dec() noexcept {
    const std::int64_t v = g_inflight.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (v == 0) {
        std::lock_guard<std::mutex> lk(g_inflight_mu);
        g_inflight_cv.notify_all();
    }
}

// =============================================================================
// Scope
// =============================================================================
struct Scope final {
    std::atomic<std::uint32_t> ref{1};
    std::atomic<std::int64_t>  active{0};
    std::atomic<std::uint8_t>  bad{0};
    std::mutex mu;
    std::condition_variable cv;

    void retain() noexcept { ref.fetch_add(1, std::memory_order_relaxed); }
    void release() noexcept { if (ref.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this; }

    void track() noexcept {
        active.fetch_add(1, std::memory_order_relaxed);
        retain();
    }

    void untrack() noexcept {
        if (active.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(mu);
            cv.notify_all();
        }
        release();
    }

    void mark_bad() noexcept { bad.store(1, std::memory_order_release); }
};

struct TlsScopes final {
    std::vector<Scope*> st;
    TlsScopes() { st.reserve(8); }
    ~TlsScopes() { for (Scope* s : st) s->release(); }
};
static thread_local TlsScopes tls_scopes;

// =============================================================================
// Task (Updated for Protocol V1)
// =============================================================================
struct Task final {
    std::atomic<std::uint32_t> refcnt{0};
    std::atomic<std::int32_t>  status{ARK_TASK_RUNNING};
    std::atomic<std::uint8_t>  cancel_req{0};

    // [SECURE] One-shot token for remote ownership transfer.
    std::atomic<std::uint8_t>  transfer_owed{0};

    // [SECURE] Cookie used to authenticate remote completions.
    std::uint64_t remote_cookie = 0;

    void* grid = nullptr;
    void* kernel_stub = nullptr; // Generic pointer to the kernel function/stub

    // [FIX] UID Split (Protocol 128-bit checksum)
    std::uint64_t uid_lo = 0;
    std::uint64_t uid_high = 0;

    void* args = nullptr;
    std::int64_t args_size = 0;
    
    // [FIX] Explicit Dimension for Kernel Wrapper
    std::int64_t grid_dim = 0; 

    ArkPresetHandle preset = 0; 
    
    // [FIX] User Config Pass-through (runtime{...} struct)
    void* config = nullptr; 

    std::atomic<Scope*> scope{nullptr};

    std::mutex wait_mu;
    std::condition_variable wait_cv;

    std::uint32_t generation = 0;
    std::uint32_t table_index = 0;

    void retain() noexcept { refcnt.fetch_add(1, std::memory_order_relaxed); }

    void release() noexcept {
        if (refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (preset) __ark_runtime_preset_release(preset);
            if (args) std::free(args);
            // Config is owned by user stack/global, do not free here.
            delete this;
        }
    }

    bool complete_transition(std::int32_t terminal) noexcept {
        std::int32_t expected = ARK_TASK_RUNNING;
        if (!status.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel)) return false;
        { std::lock_guard<std::mutex> lk(wait_mu); wait_cv.notify_all(); }
        return true;
    }
};

// =============================================================================
// Handle Table (Striped Locks + Safe Broadcast)
// =============================================================================
static constexpr std::uint32_t ARK_HANDLE_INVALID = 0xFFFFFFFFu;
static constexpr std::uint32_t ARK_MAX_HANDLES = 1u << 20;
static constexpr std::size_t   ARK_LOCK_STRIPES = 4096;

enum SlotState : std::uint8_t { SLOT_FREE = 0, SLOT_LIVE = 1 };

struct HandleSlot final {
    std::atomic<std::uint32_t> generation{1};
    Task* task = nullptr;
    std::uint32_t next_free = ARK_HANDLE_INVALID;
    SlotState state = SLOT_FREE;
};

struct HandleTable final {
    std::vector<HandleSlot> slots;
    std::uint32_t free_head = 0;
    std::mutex free_mu;
    std::condition_variable free_cv;
    std::array<std::mutex, ARK_LOCK_STRIPES> stripes;

    HandleTable() : slots(ARK_MAX_HANDLES) {
        for (std::uint32_t i = 0; i + 1 < ARK_MAX_HANDLES; ++i) slots[i].next_free = i + 1;
        slots[ARK_MAX_HANDLES - 1].next_free = ARK_HANDLE_INVALID;
    }

    std::mutex& get_lock(std::uint32_t idx) { return stripes[idx & (ARK_LOCK_STRIPES - 1)]; }

    ArkTaskHandle alloc(Task* t) {
        std::uint32_t idx;
        {
            std::unique_lock<std::mutex> lk(free_mu);
            // [BACKPRESSURE] blocks when table is full
            free_cv.wait(lk, [this]{ return free_head != ARK_HANDLE_INVALID; });
            idx = free_head;
            free_head = slots[idx].next_free;
        }

        std::lock_guard<std::mutex> lk(get_lock(idx));
        HandleSlot& s = slots[idx];
        if (s.state != SLOT_FREE) fatal("Alloc live slot");

        const std::uint32_t gen = s.generation.load(std::memory_order_relaxed);
        s.task = t;
        s.state = SLOT_LIVE;

        t->table_index = idx;
        t->generation = gen;
        t->remote_cookie = make_cookie(idx, gen); // [SECURE] unforgeable cookie

        trace(EVT_LAUNCH, idx, gen);
        return (static_cast<std::int64_t>(gen) << 32) | static_cast<std::int64_t>(idx);
    }

    Task* resolve_retain(ArkTaskHandle h) {
        const std::uint32_t idx = static_cast<std::uint32_t>(static_cast<std::uint64_t>(h) & 0xFFFFFFFFull);
        const std::uint32_t gen = static_cast<std::uint32_t>(static_cast<std::uint64_t>(h) >> 32);
        if (idx >= ARK_MAX_HANDLES) return nullptr;

        std::lock_guard<std::mutex> lk(get_lock(idx));
        HandleSlot& s = slots[idx];
        if (s.generation.load(std::memory_order_relaxed) != gen) return nullptr;
        if (s.state != SLOT_LIVE) return nullptr;
        if (!s.task) fatal("Live slot null task");

        s.task->retain();
        return s.task;
    }

    bool consume_handle(ArkTaskHandle h, Task* expected_task) {
        const std::uint32_t idx = static_cast<std::uint32_t>(static_cast<std::uint64_t>(h) & 0xFFFFFFFFull);
        const std::uint32_t gen = static_cast<std::uint32_t>(static_cast<std::uint64_t>(h) >> 32);
        if (idx >= ARK_MAX_HANDLES) return false;

        {
            std::lock_guard<std::mutex> lk(get_lock(idx));
            HandleSlot& s = slots[idx];
            if (s.generation.load(std::memory_order_relaxed) != gen) return false;
            if (s.state != SLOT_LIVE) return false;
            if (s.task != expected_task) fatal("Handle ABA");

            s.task = nullptr;
            s.state = SLOT_FREE;
            s.generation.fetch_add(1, std::memory_order_relaxed);
            s.next_free = ARK_HANDLE_INVALID;
            trace(EVT_CONSUME, idx, gen);
        }

        {
            std::lock_guard<std::mutex> lk(free_mu);
            slots[idx].next_free = free_head;
            free_head = idx;
        }
        free_cv.notify_one();
        return true;
    }

    void broadcast_cancel() {
        // [FIX] Snapshot tasks under locks; call remote code after unlocking.
        std::vector<Task*> todo;
        todo.reserve(1024);

        for (std::uint32_t i = 0; i < ARK_MAX_HANDLES; ++i) {
            std::lock_guard<std::mutex> lk(get_lock(i));
            HandleSlot& s = slots[i];
            if (s.state == SLOT_LIVE && s.task) {
                s.task->retain(); // [FIX] stable snapshot ref
                todo.push_back(s.task);
            }
        }

        for (Task* t : todo) {
            t->cancel_req.store(1, std::memory_order_release);
            { std::lock_guard<std::mutex> lk(t->wait_mu); t->wait_cv.notify_all(); }

            const char* tgt = t->preset ? __ark_runtime_preset_target(t->preset) : "";
            if (__ark_runtime_target_is_remote(tgt)) __ark_remote_cancel(t);

            t->release(); // drop snapshot ref
        }
    }
};

static HandleTable& handles() {
    static HandleTable ht;
    return ht;
}

// =============================================================================
// ThreadPool
// =============================================================================
struct Job { void (*fn)(void*); void* ctx; };

class ThreadPool final {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> q;
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    std::once_flag start_once;

    void worker_loop() {
        while (true) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]{ return stop.load(std::memory_order_relaxed) || !q.empty(); });
                
                if (stop.load(std::memory_order_relaxed) && q.empty()) return;
                
                j = q.front();
                q.pop_front();
            }
            if (j.fn) j.fn(j.ctx);
        }
    }

public:
    // [FIX] Destructor ensures threads are joined on static destruction
    ~ThreadPool() {
        shutdown();
    }

    void ensure_started() {
        std::call_once(start_once, [&]{
            unsigned n = std::thread::hardware_concurrency();
            if (n == 0) n = 4;
            workers.reserve(n);
            for (unsigned i = 0; i < n; ++i) workers.emplace_back([this]{ worker_loop(); });
        });
    }

    bool try_submit(Job j) {
        if (!j.fn) return false;
        ensure_started();
        {
            std::lock_guard<std::mutex> lk(mu);
            if (stop.load(std::memory_order_relaxed)) return false;
            q.push_back(j);
        }
        cv.notify_one();
        return true;
    }

    bool try_run_one() {
        Job j;
        {
            std::unique_lock<std::mutex> lk(mu, std::try_to_lock);
            if (!lk.owns_lock() || q.empty()) return false;
            j = q.front();
            q.pop_front();
        }
        j.fn(j.ctx);
        return true;
    }

    int run_until_status(Task* t, std::int32_t timeout_ms) {
        using Clock = std::chrono::steady_clock;
        const bool inf = (timeout_ms <= 0);
        const auto dl = Clock::now() + std::chrono::milliseconds(timeout_ms);

        while (true) {
            if (t->status.load(std::memory_order_acquire) != ARK_TASK_RUNNING)
                return t->status.load(std::memory_order_relaxed);

            if (!inf && Clock::now() >= dl) return ARK_TASK_ERR_TIMEOUT;

            if (try_run_one()) continue;

            std::unique_lock<std::mutex> lk(t->wait_mu);
            if (t->status.load(std::memory_order_acquire) != ARK_TASK_RUNNING)
                return t->status.load(std::memory_order_relaxed);

            if (inf) {
                t->wait_cv.wait(lk);
            } else if (t->wait_cv.wait_until(lk, dl) == std::cv_status::timeout) {
                return (t->status.load(std::memory_order_acquire) == ARK_TASK_RUNNING)
                    ? ARK_TASK_ERR_TIMEOUT
                    : t->status.load(std::memory_order_relaxed);
            }
        }
    }

    void shutdown() {
        { 
            std::lock_guard<std::mutex> lk(mu); 
            // [FIX] Atomic exchange ensures we only signal stop once
            if (stop.exchange(true, std::memory_order_relaxed)) return; 
        }
        
        cv.notify_all();
        
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
        workers.clear();
        q.clear();
    }
};

static ThreadPool& pool() {
    static ThreadPool p;
    return p;
}

// =============================================================================
// Completion Logic
// =============================================================================
static inline void on_complete_transition(Task* t, std::int32_t st) noexcept {
    trace(EVT_COMPLETE, t->table_index, static_cast<std::uint32_t>(st));
    Scope* s = t->scope.exchange(nullptr, std::memory_order_acq_rel);
    if (s) {
        if (st != ARK_TASK_OK) s->mark_bad();
        s->untrack();
    }
    inflight_dec();
}

static inline void finish_runner(Task* t, std::int32_t st) noexcept {
    if (t->complete_transition(st)) on_complete_transition(t, st);
    t->release(); // drop local runner ref
}

// =============================================================================
// Cooperative Cancellation
// =============================================================================
static thread_local std::atomic<std::uint8_t>* tls_cancel_ptr = nullptr;

struct CancelBind final {
    std::atomic<std::uint8_t>* prev;
    explicit CancelBind(std::atomic<std::uint8_t>* p) noexcept : prev(tls_cancel_ptr) { tls_cancel_ptr = p; }
    ~CancelBind() { tls_cancel_ptr = prev; }
};

extern "C" const std::atomic<std::uint8_t>* __ark_tls_cancel_ptr(void) {
    static std::atomic<std::uint8_t> d{0};
    return tls_cancel_ptr ? tls_cancel_ptr : &d;
}

// =============================================================================
// Execution (Adversarial Safe) [UPDATED]
// =============================================================================
extern "C" void __ark_device_sync(void);

static void task_job(void* p) {
    Task* t = static_cast<Task*>(p);
    if (!t) return;

    // 1. Check Cancellation
    if (t->cancel_req.load(std::memory_order_acquire) != 0) {
        finish_runner(t, ARK_TASK_ERR_CANCEL);
        return;
    }

    // 2. Check Remote Config
    const char* target = t->preset ? __ark_runtime_preset_target(t->preset) : "";

    if (__ark_runtime_target_is_remote(target)) {
        trace(EVT_REMOTE, t->table_index, 0);

        // [SECURE] Create transfer ref + arm token
        t->retain();
        t->transfer_owed.store(1, std::memory_order_release);

        const char* ep  = t->preset ? __ark_runtime_preset_endpoint(t->preset) : "";
        const char* tok = t->preset ? __ark_runtime_preset_token(t->preset) : "";

        const int rc = __ark_remote_submit(
            t,
            t->uid_lo,
            t->uid_high,
            t->remote_cookie,
            target,
            ep,
            tok
        );

        if (rc == 0) {
            t->release(); // Backend accepted ownership
            return;
        }

        if (rc == 1) {
            // Backend completed inline. Verify consumption.
            if (t->transfer_owed.exchange(0, std::memory_order_acq_rel) == 1) {
                t->release(); // leaked ref
                finish_runner(t, ARK_TASK_ERR_REMOTE);
                return;
            }
            t->release(); // normal completion
            return;
        }

        // Failed submission
        t->transfer_owed.store(0, std::memory_order_release);
        t->release();
        finish_runner(t, ARK_TASK_ERR_REMOTE);
        return;
    }

    // 3. Local Execution
    if (!t->kernel_stub) {
        finish_runner(t, ARK_TASK_ERR_GENERIC);
        return;
    }

    // Correct Signature: (void* grid, void* args, int64_t dim)
    using KernelWithDim = void (*)(void*, void*, std::int64_t);
    auto k_impl = reinterpret_cast<KernelWithDim>(t->kernel_stub);

    {
        CancelBind g(&t->cancel_req);
        k_impl(t->grid, t->args, t->grid_dim);
    }

    // If the stub enqueues async GPU work, block here so await == "GPU finished".
    __ark_device_sync();

    // Cancellation can be requested while we were blocked in device sync.
    if (t->cancel_req.load(std::memory_order_acquire) != 0) {
        finish_runner(t, ARK_TASK_ERR_CANCEL);
        return;
    }

    finish_runner(t, ARK_TASK_OK);
}


} // namespace ark::rt

// =============================================================================
// Opaque Surface
// =============================================================================
extern "C" void __ark_task_retain(void* t) { static_cast<ark::rt::Task*>(t)->retain(); }
extern "C" void __ark_task_release(void* t) { static_cast<ark::rt::Task*>(t)->release(); }

// =============================================================================
// Remote Completion Surface (Token-Gated + Cookie-Gated, Silent Drop)
// =============================================================================
extern "C" void __ark_task_complete_consume(void* t_ptr, std::uint64_t cookie, std::int32_t st) {
    auto* t = static_cast<ark::rt::Task*>(t_ptr);
    if (!t) return;

    // [SECURE] Cookie gate (silent)
    if (t->remote_cookie != cookie) {
        ark::rt::g_bad_completions.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // [SECURE] Token gate (silent): claim before touching any other state
    std::uint8_t expected = 1;
    if (!t->transfer_owed.compare_exchange_strong(expected, 0, std::memory_order_acq_rel)) {
        ark::rt::g_bad_completions.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Authorized completion: transition + consume transfer ref
    if (t->complete_transition(st)) ark::rt::on_complete_transition(t, st);
    t->release(); // consumes transfer ref
}

// =============================================================================
// Public ABI Implementation [Protocol V1]
// =============================================================================
extern "C" int64_t __ark_launch(
    void* grid, 
    void* kernel_stub,       // Generic stub pointer
    uint64_t uid_low,        // UID Split Low (Protocol V1)
    uint64_t uid_high,       // UID Split High (Protocol V1)
    void* args,              // Non-const in ABI (though we treat as read-only)
    uint64_t args_size,      // uint64 matches Protocol
    uint64_t grid_dim,       // uint64 matches Protocol
    void* config             // Configuration Struct Ptr (Reserved)
) {
    // 1. Validation
    if (!kernel_stub) {
        ark::rt::fatal("launch: null kernel stub");
        return ARK_TASK_ERR_INVALID;
    }
    
    // Bounds check for size (converting uint64 -> size_t safely)
    if (args_size > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        ark::rt::fatal("launch: args too large");
        return ARK_TASK_ERR_INVALID;
    }
    if (args_size > 0 && !args) {
        ark::rt::fatal("launch: null args");
        return ARK_TASK_ERR_INVALID;
    }

    // 2. Throttle & Pool check
    ark::rt::throttle_launch();
    ark::rt::pool().ensure_started();

    // 3. Task Allocation
    auto* t = new (std::nothrow) ark::rt::Task();
    if (!t) {
        ark::rt::fatal("launch: OOM task");
        return ARK_TASK_ERR_GENERIC;
    }
    t->refcnt.store(2, std::memory_order_relaxed); // Refs: Handle + Runner
    
    // 4. Initialize Standard Fields
    t->grid = grid;
    t->kernel_stub = kernel_stub;
    
    // 5. Initialize Protocol V1 Fields
    t->uid_lo = uid_low;
    t->uid_high = uid_high;
    t->grid_dim = static_cast<int64_t>(grid_dim);
    t->config = config;
    
    t->args_size = static_cast<int64_t>(args_size);
    t->preset = __ark_runtime_preset_current(); // Snapshot current preset configuration

    // 6. Track Scope (Structured Concurrency)
    if (!ark::rt::tls_scopes.st.empty()) {
        auto* s = ark::rt::tls_scopes.st.back();
        s->track();
        t->scope.store(s, std::memory_order_release);
    }

    // 7. Copy Arguments
    if (args_size > 0) {
        t->args = std::malloc(static_cast<std::size_t>(args_size));
        if (!t->args) {
            delete t;
            ark::rt::fatal("launch: OOM args");
            return ARK_TASK_ERR_GENERIC;
        }
        std::memcpy(t->args, args, static_cast<std::size_t>(args_size));
    } else {
        t->args = nullptr;
    }

    // 8. Register Handle
    ark::rt::inflight_inc();
    const ArkTaskHandle h = ark::rt::handles().alloc(t); 

    // 9. Submit to Scheduler
    ark::rt::Job j{ &ark::rt::task_job, t };
    if (!ark::rt::pool().try_submit(j)) {
        // Fallback: Run on calling thread if pool is full/stopped (prevents deadlock)
        // This is a safety valve for extreme backpressure
        ark::rt::task_job(t);
    }
    
    return h;
}


extern "C" void __ark_shutdown(void) {
    ark::rt::handles().broadcast_cancel();

    using Clock = std::chrono::steady_clock;
    const auto dl = Clock::now() + std::chrono::seconds(2);

    std::unique_lock<std::mutex> lk(ark::rt::g_inflight_mu);
    if (!ark::rt::g_inflight_cv.wait_until(lk, dl, []{ return ark::rt::g_inflight.load() == 0; })) {
        ark::rt::warn("Shutdown timeout. Forcing exit.");
        std::_Exit(2); // [HARDENING] guaranteed termination
    }

    ark::rt::pool().shutdown();
    __ark_runtime_preset_pop_all();
}

extern "C" void __ark_scope_enter(void) {
    ark::rt::tls_scopes.st.push_back(new ark::rt::Scope());
}

extern "C" int __ark_scope_exit(void) {
    if (ark::rt::tls_scopes.st.empty()) return 0;
    ark::rt::Scope* s = ark::rt::tls_scopes.st.back();
    ark::rt::tls_scopes.st.pop_back();
    ark::rt::trace(ark::rt::EVT_SCOPE_EXIT, 0, 0);

    while (s->active.load(std::memory_order_acquire) > 0) {
        if (ark::rt::pool().try_run_one()) continue;
        std::unique_lock<std::mutex> lk(s->mu);
        if (s->active.load(std::memory_order_acquire) == 0) break;
        s->cv.wait_for(lk, std::chrono::microseconds(100));
    }

    const int rc = s->bad.load(std::memory_order_acquire) ? -1 : 0;
    s->release();
    return rc;
}

extern "C" int __ark_await(ArkTaskHandle handle) {
    ark::rt::Task* t = ark::rt::handles().resolve_retain(handle);
    if (!t) return ARK_TASK_ERR_INVALID;

    ArkPresetHandle p = __ark_runtime_preset_current();
    const std::int32_t ms = __ark_runtime_preset_timeout_ms(p);
    __ark_runtime_preset_release(p);

    const int st = ark::rt::pool().run_until_status(t, ms);

    const bool consumed = ark::rt::handles().consume_handle(handle, t);
    t->release();
    if (consumed) t->release();

    return consumed ? st : ARK_TASK_ERR_INVALID;
}


extern "C" int __ark_await_ex(ArkTaskHandle handle, std::int32_t timeout_ms) {
    ark::rt::Task* t = ark::rt::handles().resolve_retain(handle);
    if (!t) return ARK_TASK_ERR_INVALID;

    const int st = ark::rt::pool().run_until_status(t, timeout_ms);
    if (st == ARK_TASK_ERR_TIMEOUT || st == ARK_TASK_RUNNING) {
        t->release();
        return ARK_TASK_ERR_TIMEOUT;
    }

    const bool consumed = ark::rt::handles().consume_handle(handle, t);
    t->release();
    if (consumed) t->release();
    return consumed ? st : ARK_TASK_ERR_INVALID;
}

extern "C" void __ark_detach(ArkTaskHandle handle) {
    ark::rt::Task* t = ark::rt::handles().resolve_retain(handle);
    if (!t) return;

    ark::rt::Scope* s = t->scope.exchange(nullptr, std::memory_order_acq_rel);
    if (s) s->untrack();

    const bool consumed = ark::rt::handles().consume_handle(handle, t);
    t->release();
    if (consumed) t->release();
}

extern "C" void __ark_cancel(ArkTaskHandle handle) {
    ark::rt::Task* t = ark::rt::handles().resolve_retain(handle);
    if (!t) return;

    t->cancel_req.store(1, std::memory_order_release);
    { std::lock_guard<std::mutex> lk(t->wait_mu); t->wait_cv.notify_all(); }

    const char* tgt = t->preset ? __ark_runtime_preset_target(t->preset) : "";
    if (__ark_runtime_target_is_remote(tgt)) __ark_remote_cancel(t);

    t->release();
}

extern "C" int __ark_task_status(ArkTaskHandle handle) {
    ark::rt::Task* t = ark::rt::handles().resolve_retain(handle);
    if (!t) return ARK_TASK_ERR_INVALID;
    const int s = t->status.load(std::memory_order_acquire);
    t->release();
    return s;
}

extern "C" void __ark_sync_all(void) {
    std::unique_lock<std::mutex> lk(ark::rt::g_inflight_mu);
    ark::rt::g_inflight_cv.wait(lk, []{ return ark::rt::g_inflight.load() == 0; });
}

extern "C" void __ark_parallel_for_1d(std::int64_t n, ArkForBodyFn body, void* ctx) {
    // [FIX] sanitize body pointer
    if (n <= 0 || !body) return;
    for (std::int64_t i = 0; i < n; ++i) body(i, ctx);
}

extern "C" std::uint64_t __ark_metrics_bad_completions(void) {
    return ark::rt::g_bad_completions.load(std::memory_order_relaxed);
}

extern "C" {

// [FIX] Implements the bridge that remote.cpp calls
void __ark_internal_get_task_info(void* task_ptr, void** out_args, std::int64_t* out_size) {
    // Cast the opaque pointer back to the internal Task struct
    auto* t = static_cast<ark::rt::Task*>(task_ptr);
    
    if (t) {
        *out_args = t->args;
        *out_size = t->args_size;
    } else {
        *out_args = nullptr;
        *out_size = 0;
    }
}

} // extern "C"