// p2pbw: GPU-to-GPU write bandwidth with SM stores (16 B per thread, as the MegaMoE kernel pushes rows), cold peers.
// Scenarios: one pair; one source to several destinations at once; every GPU to a set of peers at once.
// Each destination gets kBytes; time = CUDA events on every source, the slowest source; GB/s = bytes sent / time.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @ %s\n", cudaGetErrorString(e), #x); exit(1); } } while (0)

constexpr size_t kBytes = size_t(64) << 20;
constexpr int kMaxDst = 8;
struct Dsts { uint4 *p[kMaxDst]; int n; };

__global__ void push(Dsts d, size_t n16) {  // blocks split evenly over the destinations
    const int k = blockIdx.x % d.n;
    const size_t nb = gridDim.x / d.n, b = blockIdx.x / d.n;
    uint4 v = make_uint4(threadIdx.x, b, k, 1);
    for (size_t i = b * blockDim.x + threadIdx.x; i < n16; i += nb * blockDim.x) d.p[k][i] = v;
}

int G;
std::vector<uint4 *> buf;  // per GPU: kMaxDst * kBytes receive area (slot = source)

// srcs[i] writes to dsts[i]; returns GB/s per source (bytes it sent / slowest source time)
double run(const std::vector<int> &srcs, const std::vector<std::vector<int>> &dsts, int reps = 5) {
    std::vector<cudaEvent_t> a(srcs.size()), b(srcs.size());
    std::vector<float> best(srcs.size(), 1e30f);
    for (size_t i = 0; i < srcs.size(); ++i) {
        CK(cudaSetDevice(srcs[i]));
        CK(cudaEventCreate(&a[i]));
        CK(cudaEventCreate(&b[i]));
    }
    std::vector<std::vector<float>> ts(srcs.size());
    for (int r = 0; r < reps + 1; ++r) {
        for (int g = 0; g < G; ++g) { CK(cudaSetDevice(g)); CK(cudaDeviceSynchronize()); }
        for (size_t i = 0; i < srcs.size(); ++i) {
            CK(cudaSetDevice(srcs[i]));
            Dsts d{};
            d.n = int(dsts[i].size());
            for (int k = 0; k < d.n; ++k) d.p[k] = buf[dsts[i][k]] + size_t(srcs[i]) * (kBytes / 16);
            int sms;
            CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, srcs[i]));
            CK(cudaEventRecord(a[i]));
            push<<<sms * 4 / d.n * d.n, 256>>>(d, kBytes / 16);
            CK(cudaEventRecord(b[i]));
        }
        for (size_t i = 0; i < srcs.size(); ++i) {
            CK(cudaSetDevice(srcs[i]));
            CK(cudaEventSynchronize(b[i]));
            float ms;
            CK(cudaEventElapsedTime(&ms, a[i], b[i]));
            if (r) ts[i].push_back(ms);
        }
    }
    double gbs = 1e30;
    for (size_t i = 0; i < srcs.size(); ++i) {
        std::sort(ts[i].begin(), ts[i].end());
        const double med = ts[i][ts[i].size() / 2];
        gbs = std::min(gbs, double(kBytes) * dsts[i].size() / (med * 1e-3) / 1e9);
    }
    return gbs;
}

int main() {
    CK(cudaGetDeviceCount(&G));
    buf.resize(G);
    for (int g = 0; g < G; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&buf[g], size_t(kMaxDst) * kBytes));
        for (int h = 0; h < G; ++h) {
            if (h == g) continue;
            int ok;
            CK(cudaDeviceCanAccessPeer(&ok, g, h));
            if (!ok) { printf("GPU %d cannot access GPU %d\n", g, h); return 1; }
            CK(cudaDeviceEnablePeerAccess(h, 0));
        }
    }
    auto one = [&](const char *name, int s, std::vector<int> d) {
        printf("%-44s %7.1f GB/s\n", name, run({s}, {d}));
    };
    one("0 -> 1 (pair)", 0, {1});
    one("0 -> 2 (same socket, other switch)", 0, {2});
    one("0 -> 4 (other socket)", 0, {4});
    one("0 -> 1, 2, 3 at once (same socket)", 0, {1, 2, 3});
    one("0 -> 2, 4 at once (two root ports)", 0, {2, 4});
    one("0 -> 2, 3 at once (one other switch)", 0, {2, 3});
    one("0 -> all 7 at once", 0, {1, 2, 3, 4, 5, 6, 7});
    std::vector<int> all;
    std::vector<std::vector<int>> d7, d3;
    for (int g = 0; g < G; ++g) {
        all.push_back(g);
        std::vector<int> o, s;
        for (int h = 0; h < G; ++h)
            if (h != g) { o.push_back(h); if (h / 4 == g / 4) s.push_back(h); }
        d7.push_back(o);
        d3.push_back(s);
    }
    printf("%-44s %7.1f GB/s per GPU\n", "all 8 -> their 7 peers at once", run(all, d7));
    printf("%-44s %7.1f GB/s per GPU\n", "all 8 -> their 3 same-socket peers at once", run(all, d3));
    return 0;
}
