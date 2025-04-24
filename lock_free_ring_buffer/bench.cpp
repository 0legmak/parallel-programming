#include "lock_free_ring_buffer.h"

#include <barrier>
#include <future>

#include <benchmark/benchmark.h>

constexpr size_t kSize = 111;
constexpr size_t kElemCnt = 1234;

void bench1(benchmark::State& state) {
    LockFreeRingBuffer ring_buffer(kSize);
    for (auto _ : state) {
        std::vector<std::future<void>> futures;
        std::barrier sync(2);
        auto f1 = std::async(std::launch::async, [&]() {
            sync.arrive_and_wait();
            for (int i = 0; i < kElemCnt; ++i) {
                while (!ring_buffer.enqueue(i)) {
                }
            }
        });
        std::vector<int> dequeued(kElemCnt);
        auto f2 = std::async(std::launch::async, [&]() {
            sync.arrive_and_wait();
            for (int i = 0; i < kElemCnt; ++i) {
                std::optional<int> val;
                do {
                    val = ring_buffer.dequeue();
                } while (!val.has_value());
                dequeued[i] = *val;
            }
        });
        f1.wait();
        f2.wait();
    }
}

BENCHMARK(bench1)->Unit(benchmark::kMillisecond)->UseRealTime()->MeasureProcessCPUTime();

BENCHMARK_MAIN();
