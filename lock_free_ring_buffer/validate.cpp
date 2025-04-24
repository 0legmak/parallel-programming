#include "lock_free_ring_buffer.h"

#include <barrier>
#include <future>
#include <iostream>

int main() {
    constexpr size_t kSize = 111;
    constexpr size_t kElemCnt = 1234;
    constexpr size_t kTestCnt = 1000;
    LockFreeRingBuffer ring_buffer(kSize);
    for (int t = 0; t < kTestCnt; ++t) {
        std::vector<std::future<void>> futures;
        std::barrier sync(2);
        futures.push_back(std::async(std::launch::async, [&]() {
            sync.arrive_and_wait();
            for (int i = 0; i < kElemCnt; ++i) {
                while (!ring_buffer.enqueue(i)) {
                }
            }
        }));
        std::vector<int> dequeued(kElemCnt);
        futures.push_back(std::async(std::launch::async, [&]() {
            sync.arrive_and_wait();
            for (int i = 0; i < kElemCnt; ++i) {
                std::optional<int> val;
                do {
                    val = ring_buffer.dequeue();
                } while (!val.has_value());
                dequeued[i] = *val;
            }
        }));
        for (auto& future : futures) {
            future.get();
        }
        for (int i = 0; i < kElemCnt; ++i) {
            if (dequeued[i] != i) {
                std::cerr << "Error: dequeued[" << i << "] = " << dequeued[i] << ", expected " << i << ", round " << t << "\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
