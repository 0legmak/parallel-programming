#include <thread>
#include <atomic>
#include <iostream>
#include <barrier>

void prevent_compiler_reordering() {
    std::atomic_signal_fence(std::memory_order_acq_rel);
    // std::atomic_thread_fence(std::memory_order_seq_cst);
}

int main() {
    std::barrier sync{3};
    std::atomic<bool> w1, w2, r1, r2;
    std::thread th1{[&]() {
        while (true) {
            sync.arrive_and_wait();

            // w1 = true; r2 = w2;
            w1.store(true, std::memory_order_relaxed);
            prevent_compiler_reordering();
            r2.store(w2.load(std::memory_order_relaxed), std::memory_order_relaxed);

            sync.arrive_and_wait();
        }
    }};
    std::thread th2{[&]() {
        while (true) {
            sync.arrive_and_wait();

            // w2 = true; r1 = w1;
            w2.store(true, std::memory_order_relaxed);
            prevent_compiler_reordering();
            r1.store(w1.load(std::memory_order_relaxed), std::memory_order_relaxed);

            sync.arrive_and_wait();
        }
    }};
    for (long long iter = 1, detected = 0; ; ++iter) {
        w1.store(false, std::memory_order_relaxed);
        w2.store(false, std::memory_order_relaxed);
        sync.arrive_and_wait();
        sync.arrive_and_wait();
        if (!r1.load(std::memory_order_relaxed) && !r2.load(std::memory_order_relaxed)) {
            ++detected;
            std::cout << detected << " / " << iter << std::endl;
        }
    }
    return 0;
}
