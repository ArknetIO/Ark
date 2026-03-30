#include <ark_protocol.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// =============================================================================
// Internal ABI Bridge
// =============================================================================
extern "C" void __ark_internal_get_task_info(
    void* task_ptr,
    void** out_args,
    std::int64_t* out_size
);

extern "C" void __ark_task_complete_consume(
    void* task_ptr,
    std::uint64_t cookie,
    std::int32_t status
);

namespace ark::net {

// =============================================================================
// Protocol Constants
// =============================================================================
static constexpr std::uint32_t MAGIC = 0x41524B4E;
static constexpr std::uint32_t MAX_PAYLOAD = 16u * 1024u * 1024u;
static constexpr std::size_t   RECV_CHUNK  = 64u * 1024u;

enum OpCode : std::uint8_t {
    OP_SUBMIT    = 1,
    OP_CANCEL    = 2,
    OP_COMPLETE  = 3,
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

struct SubmitMeta {
    std::uint64_t uid_lo;
    std::uint64_t uid_hi;
    std::uint64_t args_len;
};

struct CompleteMeta {
    std::uint32_t status;
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 20, "Protocol header alignment error");
static_assert(sizeof(SubmitMeta) == 24, "Protocol submit meta alignment error");
static_assert(sizeof(CompleteMeta) == 4, "Protocol complete meta alignment error");

// =============================================================================
// Strict Endianness
// =============================================================================
static inline bool host_is_le() noexcept {
    const std::uint32_t x = 1;
    return *reinterpret_cast<const std::uint8_t*>(&x) == 1;
}

static inline std::uint16_t bswap16(std::uint16_t v) noexcept {
    return static_cast<std::uint16_t>((v >> 8) | (v << 8));
}

static inline std::uint32_t bswap32(std::uint32_t v) noexcept {
    return ((v & 0x000000FFu) << 24) |
           ((v & 0x0000FF00u) << 8)  |
           ((v & 0x00FF0000u) >> 8)  |
           ((v & 0xFF000000u) >> 24);
}

static inline std::uint64_t bswap64(std::uint64_t v) noexcept {
    return ((v & 0x00000000000000FFull) << 56) |
           ((v & 0x000000000000FF00ull) << 40) |
           ((v & 0x0000000000FF0000ull) << 24) |
           ((v & 0x00000000FF000000ull) << 8)  |
           ((v & 0x000000FF00000000ull) >> 8)  |
           ((v & 0x0000FF0000000000ull) >> 24) |
           ((v & 0x00FF000000000000ull) >> 40) |
           ((v & 0xFF00000000000000ull) >> 56);
}

static inline std::uint16_t to_le16(std::uint16_t v) noexcept { return host_is_le() ? v : bswap16(v); }
static inline std::uint32_t to_le32(std::uint32_t v) noexcept { return host_is_le() ? v : bswap32(v); }
static inline std::uint64_t to_le64(std::uint64_t v) noexcept { return host_is_le() ? v : bswap64(v); }

static inline std::uint16_t from_le16(std::uint16_t v) noexcept { return host_is_le() ? v : bswap16(v); }
static inline std::uint32_t from_le32(std::uint32_t v) noexcept { return host_is_le() ? v : bswap32(v); }
static inline std::uint64_t from_le64(std::uint64_t v) noexcept { return host_is_le() ? v : bswap64(v); }

// =============================================================================
// Helpers: Small Utilities
// =============================================================================
static inline const char* safe_endpoint(const char* endpoint) noexcept {
    return (endpoint && *endpoint) ? endpoint : "127.0.0.1";
}

static inline void free_ark_bytes(ArkBytes* b) noexcept {
    if (!b) return;
    if (b->ptr) {
        __ark_free(b->ptr);
    }
    b->ptr = nullptr;
    b->len = 0;
    b->cap = 0;
}

// =============================================================================
// Buffered Stream Reader
// Reads an arbitrary TCP byte stream through the Ark net ABI and reconstructs
// exact protocol frames safely across platform-specific recv chunking behavior.
// =============================================================================
class StreamReader final {
    std::vector<std::uint8_t> buf;
    std::size_t rd = 0;

    void compact_if_needed() {
        if (rd == 0) return;

        if (rd >= buf.size()) {
            buf.clear();
            rd = 0;
            return;
        }

        if (rd >= 4096 && rd * 2 >= buf.size()) {
            const std::size_t remain = buf.size() - rd;
            std::memmove(buf.data(), buf.data() + rd, remain);
            buf.resize(remain);
            rd = 0;
        }
    }

    std::size_t available() const noexcept {
        return buf.size() - rd;
    }

    bool append_from_socket(int32_t sockfd, std::size_t want) {
        while (available() < want) {
            ArkBytes chunk{};
            ArkIoError err{};
            const std::int64_t req = static_cast<std::int64_t>(want - available() > RECV_CHUNK ? RECV_CHUNK : (want - available()));

            const ArkStatus st = __ark_net_recv(sockfd, req, &chunk, &err);
            if (st != 0) {
                free_ark_bytes(&chunk);
                return false;
            }

            if (!chunk.ptr || chunk.len == 0) {
                free_ark_bytes(&chunk);
                return false;
            }

            const std::size_t old_size = buf.size();
            const std::size_t add = static_cast<std::size_t>(chunk.len);

            try {
                buf.resize(old_size + add);
            } catch (...) {
                free_ark_bytes(&chunk);
                return false;
            }

            std::memcpy(buf.data() + old_size, chunk.ptr, add);
            free_ark_bytes(&chunk);
        }

        return true;
    }

public:
    StreamReader() {
        buf.reserve(RECV_CHUNK);
    }

    bool read_exact(int32_t sockfd, void* out, std::size_t len) {
        compact_if_needed();

        if (len == 0) {
            return true;
        }

        if (!append_from_socket(sockfd, len)) {
            return false;
        }

        std::memcpy(out, buf.data() + rd, len);
        rd += len;
        compact_if_needed();
        return true;
    }

    bool read_payload(int32_t sockfd, std::uint32_t len, std::vector<std::uint8_t>& out) {
        out.clear();

        if (len == 0) {
            return true;
        }

        compact_if_needed();

        if (!append_from_socket(sockfd, static_cast<std::size_t>(len))) {
            return false;
        }

        try {
            out.resize(len);
        } catch (...) {
            return false;
        }

        std::memcpy(out.data(), buf.data() + rd, static_cast<std::size_t>(len));
        rd += static_cast<std::size_t>(len);
        compact_if_needed();
        return true;
    }
};

// =============================================================================
// TCP Client
// =============================================================================
class TcpClient final {
    struct PendingEntry {
        void* task_ptr = nullptr;
    };

    enum State : std::uint8_t {
        ST_DISCONNECTED = 0,
        ST_CONNECTED    = 1
    };

    std::atomic<int32_t> sock{-1};
    std::atomic<State> state{ST_DISCONNECTED};
    std::atomic<bool> stopping{false};

    std::thread receiver_thread;

    std::mutex conn_mu;
    std::mutex write_mu;
    std::mutex map_mu;

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

    ~TcpClient() {
        disconnect();
    }

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
        (void)target;
        (void)token;

        if (!task_ptr || len < 0) return -1;

        const std::uint64_t ulen = static_cast<std::uint64_t>(len);
        if (ulen > (MAX_PAYLOAD - sizeof(SubmitMeta))) return -1;
        if (ulen > 0 && !args) return -1;

        const char* host = safe_endpoint(endpoint);

        if (!ensure_connected(host)) {
            __ark_task_complete_consume(task_ptr, cookie, ARK_TASK_ERR_REMOTE);
            stat_io_errs.fetch_add(1, std::memory_order_relaxed);
            return 1;
        }

        // Track pending before sending so a very fast completion cannot race us.
        {
            std::lock_guard<std::mutex> lk(map_mu);

            if (pending_by_cookie.find(cookie) != pending_by_cookie.end()) return -1;
            if (cookie_by_task.find(task_ptr) != cookie_by_task.end()) return -1;

            pending_by_cookie.emplace(cookie, PendingEntry{task_ptr});
            cookie_by_task.emplace(task_ptr, cookie);
        }

        PacketHeader hdr{};
        hdr.magic       = to_le32(MAGIC);
        hdr.payload_len = to_le32(static_cast<std::uint32_t>(sizeof(SubmitMeta) + ulen));
        hdr.cookie      = to_le64(cookie);
        hdr.op          = OP_SUBMIT;
        hdr.flags       = 0;
        hdr.reserved    = to_le16(0);

        SubmitMeta meta{};
        meta.uid_lo   = to_le64(uid_lo);
        meta.uid_hi   = to_le64(uid_hi);
        meta.args_len = to_le64(ulen);

        bool ok = true;

        {
            std::lock_guard<std::mutex> wk(write_mu);

            const int32_t fd = sock.load(std::memory_order_acquire);
            if (fd < 0) {
                ok = false;
            } else {
                ArkIoError err{};
                if (__ark_net_send(fd, &hdr, (std::int64_t)sizeof(hdr), &err) != 0) ok = false;
                if (ok && __ark_net_send(fd, &meta, (std::int64_t)sizeof(meta), &err) != 0) ok = false;
                if (ok && ulen > 0) {
                    if (__ark_net_send(fd, args, (std::int64_t)ulen, &err) != 0) ok = false;
                }
            }
        }

        if (ok) {
            stat_sent.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

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
        hdr.magic       = to_le32(MAGIC);
        hdr.payload_len = to_le32(0);
        hdr.cookie      = to_le64(cookie);
        hdr.op          = OP_CANCEL;
        hdr.flags       = 0;
        hdr.reserved    = to_le16(0);

        std::lock_guard<std::mutex> wk(write_mu);
        const int32_t fd = sock.load(std::memory_order_acquire);
        if (fd >= 0) {
            ArkIoError err{};
            (void)__ark_net_send(fd, &hdr, (std::int64_t)sizeof(hdr), &err);
        }
    }

private:
    bool ensure_connected(const char* endpoint) {
        const std::string want_endpoint = endpoint ? endpoint : "127.0.0.1";

        {
            std::lock_guard<std::mutex> lk(conn_mu);
            if (sock.load(std::memory_order_acquire) >= 0 &&
                state.load(std::memory_order_acquire) == ST_CONNECTED &&
                current_endpoint == want_endpoint) {
                return true;
            }
        }

        disconnect();

        int32_t new_fd = -1;
        ArkIoError err{};
        if (__ark_net_connect(want_endpoint.c_str(), 8080, 2000, &new_fd, &err) != 0) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lk(conn_mu);
            current_endpoint = want_endpoint;
            stopping.store(false, std::memory_order_release);
            sock.store(new_fd, std::memory_order_release);
            state.store(ST_CONNECTED, std::memory_order_release);
            receiver_thread = std::thread(&TcpClient::recv_loop, this);
        }

        return true;
    }

    void disconnect() {
        teardown_internal(false);
    }

    void teardown_internal(bool from_receiver) {
        const int32_t old_fd = sock.exchange(-1, std::memory_order_acq_rel);
        state.store(ST_DISCONNECTED, std::memory_order_release);
        stopping.store(true, std::memory_order_release);

        if (old_fd >= 0) {
            ArkIoError err{};
            (void)__ark_net_close(old_fd, &err);
        }

        fail_all_pending();

        if (!from_receiver) {
            std::thread t;
            {
                std::lock_guard<std::mutex> lk(conn_mu);
                if (receiver_thread.joinable() &&
                    receiver_thread.get_id() != std::this_thread::get_id()) {
                    t = std::move(receiver_thread);
                }
            }

            if (t.joinable()) {
                t.join();
            }
        }
    }

    void fail_all_pending() {
        std::vector<std::pair<std::uint64_t, void*>> items;

        {
            std::lock_guard<std::mutex> lk(map_mu);
            items.reserve(pending_by_cookie.size());

            for (auto& kv : pending_by_cookie) {
                items.emplace_back(kv.first, kv.second.task_ptr);
            }

            pending_by_cookie.clear();
            cookie_by_task.clear();
        }

        for (auto& it : items) {
            __ark_task_complete_consume(it.second, it.first, ARK_TASK_ERR_REMOTE);
        }
    }

    void recv_loop() {
        StreamReader reader;
        std::vector<std::uint8_t> payload;

        for (;;) {
            if (stopping.load(std::memory_order_acquire)) {
                break;
            }

            const int32_t fd = sock.load(std::memory_order_acquire);
            if (fd < 0) {
                break;
            }

            PacketHeader hdr{};
            if (!reader.read_exact(fd, &hdr, sizeof(hdr))) {
                break;
            }

            const std::uint32_t magic       = from_le32(hdr.magic);
            const std::uint32_t payload_len = from_le32(hdr.payload_len);
            const std::uint64_t cookie      = from_le64(hdr.cookie);
            const std::uint16_t reserved    = from_le16(hdr.reserved);

            if (magic != MAGIC || payload_len > MAX_PAYLOAD || hdr.flags != 0 || reserved != 0) {
                stat_bad_packets.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            if (!reader.read_payload(fd, payload_len, payload)) {
                break;
            }

            switch (hdr.op) {
                case OP_COMPLETE: {
                    if (payload_len < sizeof(CompleteMeta)) {
                        stat_bad_packets.fetch_add(1, std::memory_order_relaxed);
                        goto disconnect_label;
                    }

                    CompleteMeta cm{};
                    std::memcpy(&cm, payload.data(), sizeof(cm));
                    const std::int32_t st = static_cast<std::int32_t>(from_le32(cm.status));

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
                    break;
                }

                case OP_HEARTBEAT:
                    if (payload_len != 0) {
                        stat_bad_packets.fetch_add(1, std::memory_order_relaxed);
                        goto disconnect_label;
                    }
                    break;

                default:
                    stat_bad_packets.fetch_add(1, std::memory_order_relaxed);
                    goto disconnect_label;
            }
        }

    disconnect_label:
        teardown_internal(true);

        std::lock_guard<std::mutex> lk(conn_mu);
        if (receiver_thread.joinable() &&
            receiver_thread.get_id() == std::this_thread::get_id()) {
            receiver_thread.detach();
        }
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
    std::uint64_t uid_lo,
    std::uint64_t uid_hi,
    std::uint64_t cookie,
    const char* target,
    const char* endpoint,
    const char* token
) {
    void* args_blob = nullptr;
    std::int64_t args_size = 0;

    // Retrieve raw arguments from the opaque task pointer.
    __ark_internal_get_task_info(task_ptr, &args_blob, &args_size);

    // Forward to the portable remote client.
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