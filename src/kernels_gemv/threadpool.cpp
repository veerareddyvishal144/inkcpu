// ThreadPool implementation (see include/ink/threadpool.h). Owner: agent B1.
//
// Protocol (all flags are 64-byte padded; monotonic 32-bit counters, compared for equality, so there is no ABA/sense bug):
//   dispatch : caller writes job, then epoch.store(e, release). Workers spin on epoch (acquire), run fn, done[i].store(e).
//   join     : caller runs fn as tid 0, then polls done[1..n-1] == e.
//   barrier  : thread i != 0 stores arrive[i] = c and spins on release == c; thread 0 polls arrive[1..n-1] == c and
//              stores release = c. c is a per-thread call counter (all threads call barrier() equally often).
#include "ink/threadpool.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace ink {

struct WorkerHandle {
    std::thread th;
};

namespace {
thread_local ThreadPool* tls_pool = nullptr;
thread_local int tls_tid = -1;

void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

uint64_t calibrate_tsc_per_us() {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const uint64_t c0 = __rdtsc();
    while (clk::now() - t0 < std::chrono::microseconds(1000)) {
    }
    const uint64_t c1 = __rdtsc();
    const auto t1 = clk::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return (uint64_t)((double)(c1 - c0) / us);
}
}  // namespace

ThreadPool::ThreadPool(int nthreads, std::vector<int> cpus, bool pin, int idle_spin_us)
    : n_(nthreads < 1 ? 1 : nthreads), pin_(pin), cpus_(std::move(cpus)) {
    const int ncpu = std::max(1, (int)std::thread::hardware_concurrency());
    if (cpus_.empty()) {
        const int nphys = std::max(1, ncpu / 2);
        for (int i = 0; i < n_; ++i) cpus_.push_back(i < nphys ? 2 * i : (2 * (i - nphys) + 1) % ncpu);
    }
    static const uint64_t tsc_per_us = calibrate_tsc_per_us();
    idle_cycles_ = (uint64_t)std::max(0, idle_spin_us) * tsc_per_us;
    slots_.reset(new Slot[3 + 3 * (size_t)n_]);
    for (size_t i = 0; i < 3 + 3 * (size_t)n_; ++i) {
        slots_[i].v.store(0, std::memory_order_relaxed);
        slots_[i].aux = 0;
    }
    job_.reset(new JobDesc{nullptr, nullptr});
    if (pin_) pin_to_cpu(cpus_[0]);
    for (int i = 1; i < n_; ++i) {
        auto w = std::make_unique<WorkerHandle>();
        w->th = std::thread([this, i] { worker_main(i); });
        workers_.push_back(std::move(w));
    }
}

ThreadPool::~ThreadPool() {
    if (n_ > 1) {
        job_->fn = nullptr;
        job_->ctx = nullptr;
        slots_[0].v.store(++epoch_local_, std::memory_order_release);
        for (auto& w : workers_) w->th.join();
    }
}

int ThreadPool::current_tid() { return tls_tid; }

uint32_t ThreadPool::wait_epoch(uint32_t seen) {
    const std::atomic<uint32_t>& ep = slots_[0].v;
    const uint64_t t0 = __rdtsc();
    unsigned cnt = 0;
    for (;;) {
        const uint32_t e = ep.load(std::memory_order_acquire);
        if (e != seen) return e;
        _mm_pause();
        if ((++cnt & 127u) == 0 && __rdtsc() - t0 > idle_cycles_) break;
    }
    for (;;) {  // idle: poll with short sleeps (not in the hot path)
        const uint32_t e = ep.load(std::memory_order_acquire);
        if (e != seen) return e;
        struct timespec ts = {0, 20000};
        nanosleep(&ts, nullptr);
    }
}

void ThreadPool::worker_main(int tid) {
    if (pin_) pin_to_cpu(cpus_[(size_t)tid % cpus_.size()]);
    tls_pool = this;
    tls_tid = tid;
    uint32_t seen = 0;
    for (;;) {
        seen = wait_epoch(seen);
        const Fn fn = job_->fn;
        void* ctx = job_->ctx;
        if (!fn) break;
        fn(ctx, tid, n_);
        slots_[2 + tid].v.store(seen, std::memory_order_release);
    }
}

void ThreadPool::run_raw(Fn fn, void* ctx) {
    ThreadPool* const prev_pool = tls_pool;
    const int prev_tid = tls_tid;
    tls_pool = this;
    tls_tid = 0;
    if (n_ == 1) {
        fn(ctx, 0, 1);
    } else {
        job_->fn = fn;
        job_->ctx = ctx;
        const uint32_t e = ++epoch_local_;
        slots_[0].v.store(e, std::memory_order_release);
        fn(ctx, 0, n_);
        for (int i = 1; i < n_; ++i) {
            const std::atomic<uint32_t>& d = slots_[2 + i].v;
            while (d.load(std::memory_order_acquire) != e) _mm_pause();
        }
    }
    tls_pool = prev_pool;
    tls_tid = prev_tid;
}

void ThreadPool::barrier() {
    if (n_ == 1) return;
    const int tid = tls_tid;
    if (tls_pool != this || tid < 0) {
        std::fprintf(stderr, "ink::ThreadPool::barrier() called outside run()\n");
        std::abort();
    }
    const uint32_t c = ++slots_[2 + 2 * (size_t)n_ + tid].aux;
    if (tid != 0) {
        slots_[2 + n_ + tid].v.store(c, std::memory_order_release);
        const std::atomic<uint32_t>& rel = slots_[1].v;
        while (rel.load(std::memory_order_acquire) != c) _mm_pause();
    } else {
        for (int i = 1; i < n_; ++i) {
            const std::atomic<uint32_t>& a = slots_[2 + n_ + i].v;
            while (a.load(std::memory_order_acquire) != c) _mm_pause();
        }
        slots_[1].v.store(c, std::memory_order_release);
    }
}

}  // namespace ink
