#pragma once

#include <cstdint>

namespace deep_gemm {

// NIC side of the EP16 launch (all zero without DG_H20_EP16_GIN). The window (one NCCL symmetric
// window on the 2-rank rail communicator, same layout on both ranks):
//   [in_send: max_own rows][in_recv: max_own rows][y_send: max_own x H bf16][y_recv: same][epoch u32][send counter u32]
// An input row is [x: H fp8][x_sf: H/128 fp32][topk_idx: K int64 global experts][topk_weights: K fp32].
// GIN signals: 0 = the peer's input rows (one put), 1 + t = the peer's combined row t; signal s lives on
// GIN context s % num_contexts.
struct H20Ep16Gin {
    const void* comm;   // ncclDevComm const * (device memory)
    void* win;          // ncclWindow_t
    uint8_t* base;      // this rank's window
    uint32_t node;      // this rank's node (the rail communicator rank); the peer is 1 - node
    uint32_t num_own;   // T: own rows this launch (== the peer's)
    uint32_t max_own;   // rows of each window region
    uint32_t num_contexts;  // GIN contexts of the device communicator (>= 1)
};

}  // namespace deep_gemm
