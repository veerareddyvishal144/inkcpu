// ThreadPool: persistent spin-wait workers, one per physical core. Owner: agent B1. See docs/KERNELS_GEMV.md.
//
//   ink::ThreadPool pool(16);                       // caller is thread 0, 15 workers pinned to CPUs 2,4,...,30
//   pool.run([&](int tid, int n) { ...phase 1...; pool.barrier(); ...phase 2... });
//
// Hot path (dispatch, join, barrier) uses only cache-line-padded atomics and _mm_pause: no futex, no condvar.
// Every thread of a run() must call barrier() the same number of times. run() must not be called concurrently
// or re-entrantly (one dispatcher thread per pool).
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace ink {

struct WorkerHandle;

class ThreadPool {
public:
    using Fn = void (*)(void* ctx, int tid, int nthreads);

    // nthreads includes the calling thread (tid 0). cpus[i] is the logical CPU for thread i (shorter lists wrap).
    // Empty list = CPUs 0,2,4,... (first hyperthread of each physical core), then 1,3,5,... if nthreads > ncpu/2.
    // pin=false skips all affinity calls (the caller thread is pinned too when pin=true).
    // idle_spin_us: workers spin this long after finishing a job, then poll with 20 us sleeps until the next job
    // (the next run() after a long idle gap then costs up to ~100 us once).
    explicit ThreadPool(int nthreads, std::vector<int> cpus = {}, bool pin = true, int idle_spin_us = 2000);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return n_; }

    // Executes fn(tid, nthreads) on every thread (caller = tid 0) and returns when all have finished.
    template <class F>
    void run(F&& f) {
        using FD = std::remove_reference_t<F>;
        run_raw([](void* c, int t, int n) { (*static_cast<FD*>(c))(t, n); }, const_cast<void*>(static_cast<const void*>(&f)));
    }
    void run_raw(Fn fn, void* ctx);

    // Dynamic row scheduling: f(tid, begin, end) is called for consecutive chunks [b, b+chunk) of 0..n-1 (last one shorter);
    // every thread repeatedly claims the next unclaimed chunk (one relaxed fetch_add on a padded counter, ~50 ns).
    // Balances itself when some cores are slower (interrupts on core 0, other tenants). Kernel results do not depend
    // on the split, so this is safe for every gemv_* kernel. Use chunk = multiple of 4 rows, >= 16.
    template <class F>
    void run_chunks(int64_t n, int64_t chunk, F&& f) {
        using FD = std::remove_reference_t<F>;
        struct Ctx {
            std::atomic<uint32_t>* ctr;
            int64_t n, chunk;
            FD* f;
        } c{&slots_[2 + 3 * (size_t)n_].v, n, chunk < 1 ? 1 : chunk, const_cast<FD*>(&f)};
        c.ctr->store(0, std::memory_order_relaxed);
        run_raw(
            [](void* cc, int tid, int) {
                Ctx* x = static_cast<Ctx*>(cc);
                for (;;) {
                    const int64_t b = (int64_t)x->ctr->fetch_add(1, std::memory_order_relaxed) * x->chunk;
                    if (b >= x->n) break;
                    (*x->f)(tid, b, b + x->chunk < x->n ? b + x->chunk : x->n);
                }
            },
            &c);
    }

    // Full barrier among all threads of the current run(); call only from inside fn, by every thread.
    void barrier();

    // Thread id inside run(), -1 outside.
    static int current_tid();

private:
    struct alignas(64) Slot {
        std::atomic<uint32_t> v;
        uint32_t aux;
        char pad[56];
    };
    struct alignas(64) JobDesc {
        Fn fn;
        void* ctx;
    };
    void worker_main(int tid);
    uint32_t wait_epoch(uint32_t seen);

    int n_;
    bool pin_;
    uint64_t idle_cycles_;
    std::vector<int> cpus_;
    std::unique_ptr<Slot[]> slots_;   // [0]=epoch, [1]=barrier release, [2+i]=done[i], [2+n+i]=arrive[i], [2+2n+i]=barrier counter[i], [2+3n]=chunk counter
    std::unique_ptr<JobDesc> job_;
    std::vector<std::unique_ptr<WorkerHandle>> workers_;
    uint32_t epoch_local_ = 0;
};

}  // namespace ink
