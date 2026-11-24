/**
 * H20 two-node EP16 NIC session (host only, NCCL 2.28 Device API; after the SM120 MegaMoE's
 * sm120_fp4_mega_moe_gin.cu): a buffer as one NCCL symmetric window on the caller's communicator (torch
 * ProcessGroupNCCL of the 2-rank rail pair, borrowed) and an NCCL device communicator with the GIN resources the
 * fused H20 MegaMoE kernel uses (DG_H20_EP16_GIN, deep_gemm/mega/ep16.py).
 */
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int kGinSignals = 1 + 256;  // sm90_fp4_mega_moe_h20_fused.cuh: input rows + one per combined row (<= 256)

void check_nccl(ncclResult_t rc, const char *what) {
    TORCH_CHECK(rc == ncclSuccess, what, " failed: ", ncclGetErrorString(rc));
}
void check_cuda(cudaError_t rc, const char *what) {
    TORCH_CHECK(rc == cudaSuccess, what, " failed: ", cudaGetErrorString(rc));
}

struct Session {
    ncclComm_t comm = nullptr;
    void *base = nullptr;
    ncclWindow_t win = nullptr;
    ncclDevComm host_dev_comm{};
    ncclDevComm *dev_comm = nullptr;  // device copy, read by the kernel
    bool dev_comm_created = false;

    void release() {
        if (dev_comm) cudaFree(dev_comm);
        if (dev_comm_created) ncclDevCommDestroy(comm, &host_dev_comm);
        if (win) ncclCommWindowDeregister(comm, win);
        if (base) ncclMemFree(base);
        dev_comm = nullptr, dev_comm_created = false, win = nullptr, base = nullptr;
    }
};

}  // namespace

// collective over the communicator; returns a handle for the calls below
int64_t create(int64_t comm_ptr, int64_t rank, int64_t world, int64_t num_bytes, int64_t gin_contexts) {
    TORCH_CHECK(comm_ptr != 0, "null NCCL communicator");
    auto *s = new Session();
    try {
        s->comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        int r = -1, n = -1;
        check_nccl(ncclCommUserRank(s->comm, &r), "ncclCommUserRank");
        check_nccl(ncclCommCount(s->comm, &n), "ncclCommCount");
        TORCH_CHECK(r == rank && n == world, "NCCL communicator is rank ", r, " of ", n);
        check_nccl(ncclMemAlloc(&s->base, static_cast<size_t>(num_bytes)), "ncclMemAlloc");
        check_nccl(ncclCommWindowRegister(s->comm, s->base, static_cast<size_t>(num_bytes), &s->win,
                                          NCCL_WIN_COLL_SYMMETRIC),
                   "ncclCommWindowRegister");
        check_cuda(cudaMemset(s->base, 0, static_cast<size_t>(num_bytes)), "cudaMemset");
        ncclDevCommRequirements_t req;
        std::memset(&req, 0, sizeof(req));
        req.ginForceEnable = true;
        req.ginContextCount = static_cast<int>(gin_contexts);
        req.ginSignalCount = kGinSignals;
        check_nccl(ncclDevCommCreate(s->comm, &req, &s->host_dev_comm), "ncclDevCommCreate");
        s->dev_comm_created = true;
        check_cuda(cudaMalloc(&s->dev_comm, sizeof(ncclDevComm)), "cudaMalloc");
        check_cuda(cudaMemcpy(s->dev_comm, &s->host_dev_comm, sizeof(ncclDevComm), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } catch (...) {
        s->release();
        delete s;
        throw;
    }
    return reinterpret_cast<int64_t>(s);
}

// [window base, ncclWindow_t, device ncclDevComm*, GIN contexts, GIN type of context 0 (0: none, 2 proxy, 3 GDAKI)]
std::vector<int64_t> info(int64_t handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    const auto &c = s->host_dev_comm;
    return {reinterpret_cast<int64_t>(s->base), reinterpret_cast<int64_t>(s->win),
            reinterpret_cast<int64_t>(s->dev_comm), c.ginContextCount, c.ginContextCount ? c.ginTypes[0] : 0};
}

// collective (window deregistration); no kernel may still use the buffer
void destroy(int64_t handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    cudaDeviceSynchronize();
    s->release();
    delete s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &create);
    m.def("info", &info);
    m.def("destroy", &destroy);
}
