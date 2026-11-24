// ThreadPool latency: run() dispatch+join round trip with an empty body, and one barrier() inside a run. Agent B1.
// Idle modes (agent T, wave 1): --idle S  = CPU time of the whole process while the pool sits idle for S seconds (after one job);
//                               --wake N  = latency of the first run() after an idle gap long enough for the workers to park (N samples).
// usage: bench_pool [--cpus 2,4,...] [--spin US] [--idle S] [--wake N] [nthreads ...]   (default thread counts 4 8 12 16)
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink/threadpool.h"
#include "testutil.h"

static double now_ns() {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double cpu_seconds() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}
static void sleep_ms(double ms) {
    struct timespec ts = {(time_t)(ms / 1000), (long)((ms - (long)(ms / 1000) * 1000) * 1e6)};
    nanosleep(&ts, nullptr);
}

int main(int argc, char** argv) {
    std::vector<int> ns, cpus;
    int spin_us = 2000, idle_s = 0, wake_n = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpus" && i + 1 < argc) {
            for (const char* p = argv[++i]; *p;) {
                cpus.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        } else if (a == "--spin" && i + 1 < argc) spin_us = std::atoi(argv[++i]);
        else if (a == "--idle" && i + 1 < argc) idle_s = std::atoi(argv[++i]);
        else if (a == "--wake" && i + 1 < argc) wake_n = std::atoi(argv[++i]);
        else ns.push_back(std::atoi(a.c_str()));
    }
    if (ns.empty()) ns = {4, 8, 12, 16};
    const char* park = std::getenv("INK_POOL_PARK");
    std::printf("loadavg at start: %s; idle mode %s; idle spin %d us\n", tu::loadavg().c_str(), (park && park[0] == '0') ? "nanosleep polling (INK_POOL_PARK=0)" : "futex park",
                spin_us);
    if (idle_s > 0) {   // idle CPU of the process: one job, then nothing for idle_s seconds
        for (int n : ns) {
            ink::ThreadPool pool(n, cpus, true, spin_us);
            pool.run([](int, int) {});
            sleep_ms(50);   // let the spin phase end before the window
            const double c0 = cpu_seconds(), t0 = now_ns();
            sleep_ms(idle_s * 1000.0);
            const double cpu = cpu_seconds() - c0, wall = (now_ns() - t0) * 1e-9;
            std::printf("idle %2d threads: %.2f%% of one core over %.1f s (parked workers now: %d)   load %s\n", n, 100.0 * cpu / wall, wall, pool.parked(),
                        tu::loadavg().c_str());
        }
        return 0;
    }
    if (wake_n > 0) {   // first run() after the workers parked: dispatch + wake + join latency
        for (int n : ns) {
            ink::ThreadPool pool(n, cpus, true, spin_us);
            for (int i = 0; i < 100; ++i) pool.run([](int, int) {});
            std::vector<double> lat;
            for (int i = 0; i < wake_n; ++i) {
                sleep_ms(spin_us / 1000.0 + 20.0);   // long enough for every worker to leave the spin phase (and the caller's own wake-up)
                const double t0 = now_ns();
                pool.run([](int, int) {});
                lat.push_back((now_ns() - t0) / 1000.0);
            }
            std::sort(lat.begin(), lat.end());
            std::printf("wake %2d threads: first run() after idle: min %.1f / median %.1f / p90 %.1f / max %.1f us (%d samples, %llu futex wakes)   load %s\n", n,
                        lat.front(), lat[lat.size() / 2], lat[lat.size() * 9 / 10], lat.back(), wake_n, (unsigned long long)pool.wakeups(), tu::loadavg().c_str());
        }
        return 0;
    }
    std::printf("%8s %22s %22s\n", "threads", "run()+join [ns] min/med", "barrier() [ns] min/med");
    for (int n : ns) {
        ink::ThreadPool pool(n, cpus, true, spin_us);
        for (int i = 0; i < 5000; ++i) pool.run([](int, int) {});
        std::vector<double> disp, bar;
        const int rounds = 11, N = 20000, M = 20000;
        for (int rd = 0; rd < rounds; ++rd) {
            double t0 = now_ns();
            for (int i = 0; i < N; ++i) pool.run([](int, int) {});
            const double d = (now_ns() - t0) / N;
            disp.push_back(d);
            t0 = now_ns();
            pool.run([&](int, int) {
                for (int i = 0; i < M; ++i) pool.barrier();
            });
            const double b = ((now_ns() - t0) - d) / M;
            bar.push_back(b);
        }
        std::sort(disp.begin(), disp.end());
        std::sort(bar.begin(), bar.end());
        std::printf("%8d %12.0f / %-8.0f %12.0f / %-8.0f   load %s  (futex wakes during the run: %llu)\n", n, disp.front(), disp[disp.size() / 2], bar.front(),
                    bar[bar.size() / 2], tu::loadavg().c_str(), (unsigned long long)pool.wakeups());
    }
    return 0;
}
