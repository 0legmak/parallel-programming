#include <atomic>
#include <optional>
#include <vector>

class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t size) : buffer(size), state() {
    }
    bool enqueue(int val) {
        State curr_state = state.load();
        const bool is_full = curr_state.head == curr_state.tail && !curr_state.is_empty;
        if (is_full) {
            return false;
        }
        buffer[curr_state.tail] = val;
        State new_state;
        do {
            new_state = curr_state;
            new_state.tail = (new_state.tail + 1) % buffer.size();
            new_state.is_empty = false;
        } while (!state.compare_exchange_weak(curr_state, new_state));
        return true;
    }
    std::optional<int> dequeue() {
        State curr_state = state.load();
        if (curr_state.is_empty) {
            return std::nullopt;
        }
        const int res = buffer[curr_state.head];
        State new_state;
        do {
            new_state = curr_state;
            new_state.head = (new_state.head + 1) % buffer.size();
            new_state.is_empty = new_state.head == new_state.tail;
        } while (!state.compare_exchange_weak(curr_state, new_state));
        return res;
    }
private:
    struct State {
        uint64_t head : 31;
        uint64_t tail : 31;
        uint64_t is_empty : 1;
        State() : head(0), tail(0), is_empty(true) {}
    };
    static_assert(sizeof(State) == sizeof(uint64_t), "State size is not 8 bytes");
    static_assert(std::atomic<State>::is_always_lock_free, "State is not lock-free");
    std::vector<int> buffer;
    std::atomic<State> state;
};
