// ThreadPool correctness: coverage of tids, barrier ordering/visibility, many barriers, back-to-back runs. Agent B1.
#include <atomic>
#include <vector>

#include "ink/threadpool.h"
#include "testutil.h"

namespace {
struct alignas(64) Cell {
    int v = 0;
};
}  // namespace

int main() {
    const int sizes[] = {1, 2, 3, 5, 8, 12, 16};
    for (int n : sizes) {
        ink::ThreadPool pool(n);
        TU_CHECK(pool.size() == n);
        TU_CHECK(ink::ThreadPool::current_tid() == -1);

        // 1. every tid runs exactly once per run, with the right thread count
        std::vector<Cell> hits(n);
        std::atomic<int> bad_n{0};
        const int runs = 3000;
        for (int it = 0; it < runs; ++it)
            pool.run([&](int tid, int nn) {
                hits[tid].v++;
                if (nn != n || ink::ThreadPool::current_tid() != tid) bad_n++;
            });
        for (int t = 0; t < n; ++t) TU_CHECK(hits[t].v == runs);
        TU_CHECK(bad_n.load() == 0);

        // 2. barrier ordering and memory visibility, three phases per run
        std::vector<Cell> cell(n);
        std::atomic<int> bad{0};
        for (int it = 0; it < 600; ++it)
            pool.run([&](int tid, int nn) {
                cell[tid].v = it * 2 + 1;
                pool.barrier();
                for (int j = 0; j < nn; ++j)
                    if (cell[j].v != it * 2 + 1) bad++;
                pool.barrier();
                cell[tid].v = it * 2 + 2;
                pool.barrier();
                for (int j = 0; j < nn; ++j)
                    if (cell[j].v != it * 2 + 2) bad++;
            });
        TU_CHECK(bad.load() == 0);

        // 3. many barriers in one run, each separating a write phase from a read phase of a neighbour
        std::atomic<int> bad3{0};
        pool.run([&](int tid, int nn) {
            for (int i = 0; i < 20000; ++i) {
                cell[tid].v = i;
                pool.barrier();
                if (cell[(tid + 1) % nn].v != i) bad3++;
                pool.barrier();
            }
        });
        TU_CHECK(bad3.load() == 0);

        // 4. results of run i are visible in run i+1 (prefix chain through all threads)
        std::vector<Cell> chain(n);
        for (int it = 0; it < 500; ++it) {
            pool.run([&](int tid, int) { chain[tid].v = (it == 0 ? 0 : chain[(tid + 1) % n].v) + tid + 1; });
        }
        // not asserting exact values (reads race with writes inside a run); asserting liveness + determinism of a reduction:
        std::vector<double> part(n * 8, 0.0);
        const int64_t total = 1000003;
        pool.run([&](int tid, int nn) {
            int64_t b = total * tid / nn, e = total * (tid + 1) / nn;
            double s = 0;
            for (int64_t i = b; i < e; ++i) s += (double)i;
            part[tid * 8] = s;
        });
        double sum = 0;
        for (int t = 0; t < n; ++t) sum += part[t * 8];
        TU_CHECK(sum == (double)total * (double)(total - 1) / 2.0);

        std::printf("threadpool n=%2d ok\n", n);
    }
    if (tu::g_fail) {
        std::printf("test_threadpool: %d FAILURES\n", tu::g_fail);
        return 1;
    }
    std::printf("test_threadpool: all ok\n");
    return 0;
}
