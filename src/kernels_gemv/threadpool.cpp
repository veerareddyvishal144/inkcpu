// ThreadPool implementation (see include/ink/threadpool.h). Owner: agent B1; idle parking: agent T (wave 1).
//
// Protocol (all flags are 64-byte padded; monotonic 32-bit counters, compared for equality, so there is no ABA/sense bug):
//   dispatch : caller writes job, then epoch.store(e) (seq_cst: one xchg). If sleepers != 0 it also futex-wakes the epoch word.
//   join     : caller runs fn as tid 0, then polls done[1..n-1] == e.
//   barrier  : thread i != 0 stores arrive[i] = c and spins on release == c; thread 0 polls arrive[1..n-1] == c and
//              stores release = c. c is a per-thread call counter (all threads call barrier() equally often).
//   idle     : a worker spins idle_spin_us on the epoch, then parks: sleepers++ (seq_cst), re-check epoch, FUTEX_WAIT on the
//              epoch word (250 ms safety timeout), sleepers--. The seq_cst pair (epoch store / sleepers load on the dispatcher,
//              sleepers increment / epoch load on the worker) makes a lost wake-up impossible: one side always sees the other.
//              Env INK_POOL_PARK=0 restores the old behaviour (20 us nanosleep polling) for A/B measurements.
#include "ink/threadpool.h"

#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <climits>
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

long futex_op(std::atomic<uint32_t>* word, int op, uint32_t val, const struct timespec* ts) {
    return syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), op, val, ts, nullptr, 0);
}

bool park_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("INK_POOL_PARK");
        return !(e && e[0] == '0');
    }();
    return on;
}
}  // namespace

ThreadPool::ThreadPool(int nthreads, std::vector<int> cpus, bool pin, int idle_spin_us)
    : n_(nthreads < 1 ? 1 : nthreads), pin_(pin), park_(park_enabled()), cpus_(std::move(cpus)) {
    const int ncpu = std::max(1, (int)std::thread::hardware_concurrency());
    if (cpus_.empty()) {
        const int nphys = std::max(1, ncpu / 2);
        for (int i = 0; i < n_; ++i) cpus_.push_back(i < nphys ? 2 * i : (2 * (i - nphys) + 1) % ncpu);
    }
    static const uint64_t tsc_per_us = calibrate_tsc_per_us();
    idle_cycles_ = (uint64_t)std::max(0, idle_spin_us) * tsc_per_us;
    slots_.reset(new Slot[4 + 3 * (size_t)n_]);
    for (size_t i = 0; i < 4 + 3 * (size_t)n_; ++i) {
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
        publish_epoch(++epoch_local_);
        for (auto& w : workers_) w->th.join();
    }
}

int ThreadPool::current_tid() { return tls_tid; }

void ThreadPool::publish_epoch(uint32_t e) {
    slots_[0].v.store(e, std::memory_order_seq_cst);                                  // xchg: full fence before the sleepers load
    if (slots_[3 + 3 * (size_t)n_].v.load(std::memory_order_seq_cst) != 0) {         // only when a worker parked (never in a token)
        futex_op(&slots_[0].v, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr);
        wakeups_++;
    }
}

uint32_t ThreadPool::wait_epoch(uint32_t seen) {
    std::atomic<uint32_t>& ep = slots_[0].v;
    const uint64_t t0 = __rdtsc();
    unsigned cnt = 0;
    for (;;) {
        const uint32_t e = ep.load(std::memory_order_acquire);
        if (e != seen) return e;
        _mm_pause();
        if ((++cnt & 127u) == 0 && __rdtsc() - t0 > idle_cycles_) break;
    }
    if (!park_) {
        for (;;) {  // legacy idle mode: poll with short sleeps (INK_POOL_PARK=0)
            const uint32_t e = ep.load(std::memory_order_acquire);
            if (e != seen) return e;
            struct timespec ts = {0, 20000};
            nanosleep(&ts, nullptr);
        }
    }
    std::atomic<uint32_t>& sleepers = slots_[3 + 3 * (size_t)n_].v;
    for (;;) {  // park: the kernel re-checks epoch == seen atomically with queueing us, the dispatcher checks sleepers after its store
        sleepers.fetch_add(1, std::memory_order_seq_cst);
        uint32_t e = ep.load(std::memory_order_seq_cst);
        if (e == seen) {
            struct timespec ts = {0, 250 * 1000000L};  // safety net only: a lost wake-up would cost 250 ms, never a hang
            futex_op(&ep, FUTEX_WAIT_PRIVATE, seen, &ts);
            e = ep.load(std::memory_order_acquire);
        }
        sleepers.fetch_sub(1, std::memory_order_seq_cst);
        if (e != seen) return e;
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
        publish_epoch(e);
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

int ThreadPool::parked() const { return (int)slots_[3 + 3 * (size_t)n_].v.load(std::memory_order_acquire); }

}  // namespace ink
