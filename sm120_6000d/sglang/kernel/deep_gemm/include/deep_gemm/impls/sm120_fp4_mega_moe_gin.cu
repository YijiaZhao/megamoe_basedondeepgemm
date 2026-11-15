/**
 * SM120 FP4 MegaMoE NIC transport session (host only): the buffer as one NCCL symmetric window on the caller's
 * communicator (torch ProcessGroupNCCL, borrowed), the load/store peer pointer of every rank NCCL maps (0 where it
 * cannot, e.g. another host) and an NCCL device communicator with the GIN resources the kernel uses. The kernel
 * extension built with SM120_FP4_GIN=1 reaches the unmapped ranks with GIN puts through the NIC.
 */
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kGinSignals = 80;  // sm120_fp4_mega_moe.cu: 3 * kMaxWorld kernel signals + 2 * kMaxWorld probe signals

void check_nccl(ncclResult_t rc, const char *what) {
    TORCH_CHECK(rc == ncclSuccess, what, " failed: ", ncclGetErrorString(rc));
}
void check_cuda(cudaError_t rc, const char *what) {
    TORCH_CHECK(rc == cudaSuccess, what, " failed: ", cudaGetErrorString(rc));
}

struct Session {
    ncclComm_t comm = nullptr;
    void *base = nullptr;
    size_t bytes = 0;
    ncclWindow_t win = nullptr;
    ncclDevComm host_dev_comm{};
    ncclDevComm *dev_comm = nullptr;  // device copy, read by the kernel
    bool dev_comm_created = false;
    int gin_type = 0;
    std::vector<int64_t> peer_ptrs;  // load/store mapping per rank, 0 when NCCL cannot map the pair

    void release() {
        if (dev_comm) cudaFree(dev_comm);
        if (dev_comm_created) ncclDevCommDestroy(comm, &host_dev_comm);
        if (win) ncclCommWindowDeregister(comm, win);
        if (base) ncclMemFree(base);
        dev_comm = nullptr, dev_comm_created = false, win = nullptr, base = nullptr;
    }
};

ncclComm_t comm_of(int64_t comm_ptr, int64_t rank, int64_t world, ncclCommProperties_t &props) {
    TORCH_CHECK(comm_ptr != 0, "null NCCL communicator");
    auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
    props = NCCL_COMM_PROPERTIES_INITIALIZER;
    check_nccl(ncclCommQueryProperties(comm, &props), "ncclCommQueryProperties");
    TORCH_CHECK(props.rank == rank && props.nRanks == world, "NCCL communicator is rank ", props.rank, " of ",
                props.nRanks);
    TORCH_CHECK(props.deviceApiSupport, "the NCCL communicator does not support the Device API");
    return comm;
}

}  // namespace

// the communicator's GIN type (NCCL_GIN_TYPE_*; 0: no GIN), which selects the kernel build
int64_t gin_type(int64_t comm_ptr, int64_t rank, int64_t world) {
    ncclCommProperties_t props;
    comm_of(comm_ptr, rank, world, props);
    return static_cast<int64_t>(props.ginType);
}

// collective over the communicator; returns a handle for the calls below
int64_t create(int64_t comm_ptr, int64_t rank, int64_t world, int64_t num_bytes, int64_t gin_contexts) {
    auto *s = new Session();
    try {
        ncclCommProperties_t props;
        s->comm = comm_of(comm_ptr, rank, world, props);
        s->bytes = static_cast<size_t>(num_bytes);
        s->gin_type = static_cast<int>(props.ginType);
        check_nccl(ncclMemAlloc(&s->base, s->bytes), "ncclMemAlloc");
        check_nccl(ncclCommWindowRegister(s->comm, s->base, s->bytes, &s->win, NCCL_WIN_COLL_SYMMETRIC),
                   "ncclCommWindowRegister");
        check_cuda(cudaMemset(s->base, 0, s->bytes), "cudaMemset");
        s->peer_ptrs.assign(static_cast<size_t>(world), 0);
        for (int p = 0; p < world; ++p) {
            void *ptr = nullptr;
            if (ncclGetPeerDevicePointer(s->win, 0, p, &ptr) == ncclSuccess) s->peer_ptrs[p] = reinterpret_cast<int64_t>(ptr);
        }
        s->peer_ptrs[rank] = reinterpret_cast<int64_t>(s->base);  // local accesses go through the allocation itself
        if (s->gin_type != NCCL_GIN_TYPE_NONE) {
            ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            req.ginContextCount = static_cast<int>(gin_contexts);
            req.ginSignalCount = kGinSignals;
            req.worldGinBarrierCount = 0;
            req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
            req.ginStrongSignalsRequired = true;
            req.ginVaSignalsRequired = false;
            check_nccl(ncclDevCommCreate(s->comm, &req, &s->host_dev_comm), "ncclDevCommCreate");
            s->dev_comm_created = true;
            check_cuda(cudaMalloc(&s->dev_comm, sizeof(ncclDevComm)), "cudaMalloc");
            check_cuda(cudaMemcpy(s->dev_comm, &s->host_dev_comm, sizeof(ncclDevComm), cudaMemcpyHostToDevice),
                       "cudaMemcpy");
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } catch (...) {
        s->release();
        delete s;
        throw;
    }
    return reinterpret_cast<int64_t>(s);
}

// [window base, ncclWindow_t, device ncclDevComm* (0 without GIN), GIN contexts]
std::vector<int64_t> info(int64_t handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    return {reinterpret_cast<int64_t>(s->base), reinterpret_cast<int64_t>(s->win),
            reinterpret_cast<int64_t>(s->dev_comm), s->dev_comm_created ? s->host_dev_comm.ginContextCount : 0};
}

// load/store pointer of every rank's window, 0 where NCCL does not map the pair
std::vector<int64_t> peers(int64_t handle) { return reinterpret_cast<Session *>(handle)->peer_ptrs; }

// collective (window deregistration); no kernel may still use the buffer
void destroy(int64_t handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    cudaDeviceSynchronize();
    s->release();
    delete s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gin_type", &gin_type);
    m.def("create", &create);
    m.def("info", &info);
    m.def("peers", &peers);
    m.def("destroy", &destroy);
}
