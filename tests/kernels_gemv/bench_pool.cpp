// ThreadPool latency: run() dispatch+join round trip with an empty body, and one barrier() inside a run. Agent B1.
// usage: bench_pool [nthreads ...]   (default 4 8 12 16)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ink/threadpool.h"
#include "testutil.h"

static double now_ns() {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::vector<int> ns;
    for (int i = 1; i < argc; ++i) ns.push_back(std::atoi(argv[i]));
    if (ns.empty()) ns = {4, 8, 12, 16};
    std::printf("loadavg at start: %s\n", tu::loadavg().c_str());
    std::printf("%8s %22s %22s\n", "threads", "run()+join [ns] min/med", "barrier() [ns] min/med");
    for (int n : ns) {
        ink::ThreadPool pool(n);
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
        std::printf("%8d %12.0f / %-8.0f %12.0f / %-8.0f   load %s\n", n, disp.front(), disp[disp.size() / 2], bar.front(),
                    bar[bar.size() / 2], tu::loadavg().c_str());
    }
    return 0;
}
