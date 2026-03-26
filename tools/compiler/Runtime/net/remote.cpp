#include <ark_protocol.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <condition_variable>

// Platform Sockets
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

// =============================================================================
// Internal ABI Bridge
// =============================================================================
extern "C" void __ark_internal_get_task_info(
    void* task_ptr,
    void** out_args,
    std::int64_t* out_size
);

extern "C" void __ark_task_complete_consume(void* task_ptr, std::uint64_t cookie, std::int32_t status);

namespace ark::net {

// =============================================================================
// Protocol Constants
// =============================================================================
static constexpr std::uint32_t MAGIC = 0x41524B4E;
static constexpr std::uint32_t MAX_PAYLOAD = 16u * 1024u * 1024u;

enum OpCode : std::uint8_t {
    OP_SUBMIT = 1,
    OP_CANCEL = 2,
    OP_COMPLETE = 3,
    OP_HEARTBEAT = 4
};

#pragma pack(push, 1)
struct PacketHeader {
    std::uint32_t magic;
    std::uint32_t payload_len;
    std::uint64_t cookie;
    std::uint8_t  op;
    std::uint8_t  flags;
    std::uint16_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 20, "Protocol header alignment error");

// =============================================================================
// Strict Endianness
// =============================================================================
static inline bool host_is_le() noexcept {
    const int x = 1;
    return *reinterpret_cast<const char*>(&x) == 1;
}

static inline std::uint16_t bswap16(std::uint16_t v) noexcept {
#if defined(_MSC_VER)
    return _byteswap_ushort(v);
#else
    return __builtin_bswap16(v);
#endif
}

static inline std::uint32_t bswap32(std::uint32_t v) noexcept {
#if defined(_MSC_VER)
    return _byteswap_ulong(v);
#else
    return __builtin_bswap32(v);
#endif
}

static inline std::uint64_t bswap64(std::uint64_t v) noexcept {
#if defined(_MSC_VER)
    return _byteswap_uint64(v);
#else
    return __builtin_bswap64(v);
#endif
}

static inline std::uint16_t to_le16(std::uint16_t v) noexcept { return host_is_le() ? v : bswap16(v); }
static inline std::uint32_t to_le32(std::uint32_t v) noexcept { return host_is_le() ? v : bswap32(v); }
static inline std::uint64_t to_le64(std::uint64_t v) noexcept { return host_is_le() ? v : bswap64(v); }

static inline std::uint16_t from_le16(std::uint16_t v) noexcept { return host_is_le() ? v : bswap16(v); }
static inline std::uint32_t from_le32(std::uint32_t v) noexcept { return host_is_le() ? v : bswap32(v); }
static inline std::uint64_t from_le64(std::uint64_t v) noexcept { return host_is_le() ? v : bswap64(v); }

// =============================================================================
// Robust I/O
// =============================================================================
static bool write_full(int fd, const void* data, std::size_t len) noexcept {
    const char* p = static_cast<const char*>(data);
    std::size_t rem = len;
    while (rem > 0) {
#ifdef MSG_NOSIGNAL
        ssize_t n = ::send(fd, p, rem, MSG_NOSIGNAL);
#else
        ssize_t n = ::send(fd, p, rem, 0);
#endif
        if (n <= 0) {
            if (n < 0 && (errno == EINTR)) continue;
            return false;
        }
        p += static_cast<std::size_t>(n);
        rem -= static_cast<std::size_t>(n);
    }
    return true;
}

static bool read_full(int fd, void* data, std::size_t len) noexcept {
    char* p = static_cast<char*>(data);
    std::size_t rem = len;
    while (rem > 0) {
        ssize_t n = ::recv(fd, p, rem, 0);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR)) continue;
            return false;
        }
        p += static_cast<std::size_t>(n);
        rem -= static_cast<std::size_t>(n);
    }
    return true;
}

// RAII to restore socket blocking mode
struct FlagGuard {
    int fd;
    int flags;
    bool active;
    FlagGuard(int f, int fl) : fd(f), flags(fl), active(true) {}
    ~FlagGuard() { if (active && fd >= 0) ::fcntl(fd, F_SETFL, flags); }
};

static bool connect_with_timeout(int fd, const struct sockaddr* addr, socklen_t len, int ms) noexcept {
    int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0) return false;
    
    if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return false;
    FlagGuard guard(fd, flags);

    int rc;
    do { rc = ::connect(fd, addr, len); } while (rc < 0 && errno == EINTR);

    if (rc == 0) return true;
    if (errno != EINPROGRESS) return false;

    struct pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLOUT;
    
    int prc;
    do { prc = ::poll(&pfd, 1, ms); } while (prc < 0 && errno == EINTR);

    if (prc <= 0) return false; // Timeout or Error

    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;
    if (!(pfd.revents & POLLOUT)) return false;

    int err = 0;
    socklen_t elen = sizeof(err);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &elen) < 0 || err != 0) return false;

    return true;
}

// =============================================================================
// TCP Client
// =============================================================================
class TcpClient final {
    struct PendingEntry { void* task_ptr = nullptr; };

    enum State : std::uint8_t {
        ST_DISCONNECTED = 0,
        ST_CONNECTING   = 1,
        ST_CONNECTED    = 2
    };

    std::atomic<int> sock{-1};
    std::atomic<State> state{ST_DISCONNECTED};
    std::atomic<std::uint64_t> conn_epoch{1}; 
    
    std::thread receiver_thread;
    std::mutex conn_mu;            
    std::condition_variable conn_cv; 
    
    std::mutex write_mu;           
    std::mutex map_mu;             

    // [FIX] Track current endpoint for reconnection logic
    std::string current_endpoint; 

    std::unordered_map<std::uint64_t, PendingEntry> pending_by_cookie;
    std::unordered_map<void*, std::uint64_t> cookie_by_task;

    std::atomic<std::uint64_t> stat_sent{0};
    std::atomic<std::uint64_t> stat_recv{0};
    std::atomic<std::uint64_t> stat_io_errs{0};
    std::atomic<std::uint64_t> stat_bad_packets{0};

public:
    TcpClient() {
        pending_by_cookie.reserve(2048);
        cookie_by_task.reserve(2048);
    }

    ~TcpClient() { disconnect(); }

    // [FIX] Protocol V1 Submit
    int submit(
        void* task_ptr,
        std::uint64_t uid_lo,
        std::uint64_t uid_hi,
        std::uint64_t cookie,
        const void* args,
        std::int64_t len,
        const char* target,
        const char* endpoint,
        const char* token
    ) {
        if (!task_ptr || len < 0) return -1;
        const std::uint64_t ulen = static_cast<std::uint64_t>(len);
        
        if (ulen > (MAX_PAYLOAD - 32u)) return -1; 
        if (ulen > 0 && !args) return -1;

        // 1. Connection Logic
        {
            std::lock_guard<std::mutex> lk(conn_mu);
            
            // Check if we need to reconnect (socket closed OR endpoint changed)
            bool needs_connect = (sock.load(std::memory_order_relaxed) == -1);
            
            if (endpoint && current_endpoint != endpoint) {
                needs_connect = true;
                // If we are switching endpoints, force close existing socket first
                int existing = sock.load(std::memory_order_relaxed);
                if (existing != -1) {
                    __ark_net_close(existing, nullptr);
                    sock.store(-1, std::memory_order_relaxed);
                }
                current_endpoint = endpoint;
            }

            if (needs_connect) {
                const char* host = (endpoint && *endpoint) ? endpoint : "127.0.0.1";
                int32_t port = 8080; // Hardcoded for MVP
                
                int32_t new_fd = -1;
                ArkIoError err;
                if (__ark_net_connect(host, port, 2000, &new_fd, &err) != 0) {
                    return -1; 
                }
                
                // Update atomic socket state
                sock.store(new_fd, std::memory_order_release);
                state.store(ST_CONNECTED, std::memory_order_release);
                
                // Start receiver if not running
                if (!receiver_thread.joinable()) {
                    receiver_thread = std::thread(&TcpClient::recv_loop, this);
                }
            }
        }

        // 2. Track Pending Task
        {
            std::lock_guard<std::mutex> lk(map_mu);
            if (pending_by_cookie.find(cookie) != pending_by_cookie.end()) return -1;
            if (cookie_by_task.find(task_ptr) != cookie_by_task.end()) return -1;
            pending_by_cookie.emplace(cookie, PendingEntry{task_ptr});
            cookie_by_task.emplace(task_ptr, cookie);
        }

        // 3. Prepare Header (Protocol V1)
        // [FIX] Define struct type first, then instantiate to avoid closure errors
        struct __attribute__((packed)) HeaderV1 {
            uint32_t magic;
            uint32_t payload_len;
            uint64_t cookie;
            uint8_t  op;
            uint8_t  flags;
            uint16_t reserved;
            uint64_t uid_lo;
            uint64_t uid_hi;
            uint64_t args_len;
        };

        HeaderV1 hdr;
        hdr.magic       = to_le32(MAGIC);
        hdr.payload_len = to_le32(static_cast<uint32_t>(24u + ulen));
        hdr.cookie      = to_le64(cookie);
        hdr.op          = OP_SUBMIT;
        hdr.flags       = 0;
        hdr.reserved    = 0;
        hdr.uid_lo      = to_le64(uid_lo);
        hdr.uid_hi      = to_le64(uid_hi);
        hdr.args_len    = to_le64(ulen);

        // 4. Send Data
        bool ok = true;
        {
            std::lock_guard<std::mutex> wk(write_mu);
            int fd = sock.load(std::memory_order_relaxed);
            
            if (fd != -1) {
                ArkIoError err;
                if (__ark_net_send(fd, &hdr, sizeof(hdr), &err) != 0) ok = false;
                if (ok && ulen > 0) {
                    if (__ark_net_send(fd, args, ulen, &err) != 0) ok = false;
                }
            } else {
                ok = false;
            }
        }

        if (ok) {
            stat_sent.fetch_add(1, std::memory_order_relaxed);
            return 0; // Accepted
        }

        // 5. Cleanup on Failure
        {
            std::lock_guard<std::mutex> lk(map_mu);
            pending_by_cookie.erase(cookie);
            cookie_by_task.erase(task_ptr);
        }
        
        __ark_task_complete_consume(task_ptr, cookie, ARK_TASK_ERR_REMOTE);
        stat_io_errs.fetch_add(1, std::memory_order_relaxed);
        disconnect(); 
        return 1; 
    }

    void cancel_task(void* task_ptr) {
        if (!task_ptr) return;
        std::uint64_t cookie = 0;
        {
            std::lock_guard<std::mutex> lk(map_mu);
            auto it = cookie_by_task.find(task_ptr);
            if (it == cookie_by_task.end()) return;
            cookie = it->second;
        }

        PacketHeader hdr{};
        hdr.magic = to_le32(MAGIC);
        hdr.payload_len = to_le32(0);
        hdr.cookie = to_le64(cookie);
        hdr.op = OP_CANCEL;
        hdr.flags = 0;
        hdr.reserved = to_le16(0);

        std::lock_guard<std::mutex> wk(write_mu);
        int fd = sock.load(std::memory_order_relaxed);
        if (fd != -1) {
            ArkIoError err;
            (void)__ark_net_send(fd, &hdr, sizeof(hdr), &err);
        }
    }

private:
    bool ensure_connected() {
        // [DEPRECATED helper, logic moved into submit() for atomic endpoint switching]
        // Keeping this method stub or implementation is optional depending on other callers.
        // For this class, submit() handles its own connection logic now.
        return sock.load() != -1;
    }

    void disconnect() {
        teardown_internal();
        
        std::thread t;
        {
            std::lock_guard<std::mutex> lk(conn_mu);
            if (receiver_thread.joinable() && 
                receiver_thread.get_id() != std::this_thread::get_id()) {
                t = std::move(receiver_thread);
            }
        }
        if (t.joinable()) t.join();
    }

    void teardown_internal() {
        int old_fd = -1;
        {
            std::lock_guard<std::mutex> lk(conn_mu);
            conn_epoch.fetch_add(1, std::memory_order_acq_rel);
            state.store(ST_DISCONNECTED, std::memory_order_release);
        }

        {
            std::lock_guard<std::mutex> wk(write_mu);
            old_fd = sock.exchange(-1, std::memory_order_acq_rel);
            if (old_fd != -1) {
                ::shutdown(old_fd, SHUT_RDWR);
                ::close(old_fd);
            }
        }

        conn_cv.notify_all();
        fail_all_pending();
    }

    void fail_all_pending() {
        std::vector<std::pair<std::uint64_t, void*>> items;
        {
            std::lock_guard<std::mutex> lk(map_mu);
            items.reserve(pending_by_cookie.size());
            for (auto& kv : pending_by_cookie) items.emplace_back(kv.first, kv.second.task_ptr);
            pending_by_cookie.clear();
            cookie_by_task.clear();
        }
        for (auto& it : items) {
            __ark_task_complete_consume(it.second, it.first, ARK_TASK_ERR_REMOTE);
        }
    }

    void recv_loop() {
        PacketHeader hdr{};
        std::size_t hdr_got = 0;

        while (state.load(std::memory_order_acquire) == ST_CONNECTED) {
            int fd = sock.load(std::memory_order_relaxed);
            if (fd == -1) break;

            const ssize_t n = ::recv(
                fd,
                reinterpret_cast<char*>(&hdr) + hdr_got,
                sizeof(PacketHeader) - hdr_got,
                0
            );

            if (n == 0) break; 

            if (n < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // In blocking mode with timeout, we might get EAGAIN
                    continue; 
                }
                break; 
            }

            hdr_got += static_cast<std::size_t>(n);
            if (hdr_got < sizeof(PacketHeader)) continue;

            hdr_got = 0; 

            const std::uint32_t magic = from_le32(hdr.magic);
            const std::uint32_t payload_len = from_le32(hdr.payload_len);
            const std::uint64_t cookie = from_le64(hdr.cookie);

            if (magic != MAGIC || payload_len > MAX_PAYLOAD || hdr.flags != 0 || hdr.reserved != 0) {
                stat_bad_packets.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            // Payload Handling
            std::unique_ptr<char[]> body;
            if (payload_len > 0) {
                body.reset(new (std::nothrow) char[payload_len]);
                if (!body) { stat_io_errs.fetch_add(1, std::memory_order_relaxed); break; }
                
                // We need a blocking read loop for the body
                size_t body_got = 0;
                bool body_ok = true;
                while (body_got < payload_len) {
                    ssize_t bn = ::recv(fd, body.get() + body_got, payload_len - body_got, 0);
                    if (bn <= 0) { body_ok = false; break; }
                    body_got += bn;
                }
                if (!body_ok) break;
            }

            switch (hdr.op) {
                case OP_COMPLETE:
                    if (payload_len >= 4 && body) {
                        std::uint32_t st_u32 = 0;
                        std::memcpy(&st_u32, body.get(), 4);
                        const std::int32_t st = static_cast<std::int32_t>(from_le32(st_u32));

                        void* task_ptr = nullptr;
                        {
                            std::lock_guard<std::mutex> lk(map_mu);
                            auto it = pending_by_cookie.find(cookie);
                            if (it != pending_by_cookie.end()) {
                                task_ptr = it->second.task_ptr;
                                pending_by_cookie.erase(it);
                                cookie_by_task.erase(task_ptr);
                            }
                        }
                        if (task_ptr) {
                            __ark_task_complete_consume(task_ptr, cookie, st);
                            stat_recv.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    break;
                case OP_HEARTBEAT:
                    break;
                default:
                    stat_bad_packets.fetch_add(1, std::memory_order_relaxed);
                    goto disconnect_label; 
            }
        }
        
    disconnect_label:
        teardown_internal();
    }
};

static TcpClient& get_client() {
    static TcpClient c;
    return c;
}

} // namespace ark::net

// =============================================================================
// ABI Exports
// =============================================================================
extern "C" int __ark_remote_submit(
    void* task_ptr,
    std::uint64_t uid_lo,   // [FIX] Protocol V1 Split UID Low
    std::uint64_t uid_hi,   // [FIX] Protocol V1 Split UID High
    std::uint64_t cookie,
    const char* target,     // [FIX] Name arguments to pass them
    const char* endpoint, 
    const char* token
) {
    void* args_blob = nullptr;
    std::int64_t args_size = 0;
    
    // Retrieve raw arguments from the opaque task pointer
    __ark_internal_get_task_info(task_ptr, &args_blob, &args_size);
    
    // Forward to Network Client
    // We assume ark::net::Client::submit is also updated to handle V1 Protocol fields
    return ark::net::get_client().submit(
        task_ptr, 
        uid_lo, 
        uid_hi, 
        cookie, 
        args_blob, 
        args_size,
        target, 
        endpoint, 
        token
    );
}

extern "C" void __ark_remote_cancel(void* task_ptr) {
    ark::net::get_client().cancel_task(task_ptr);
}