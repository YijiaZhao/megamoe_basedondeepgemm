/**
 * SM120 decode MegaMoE (EP over PCIe P2P), one persistent kernel per rank:
 *   dispatch (MXFP8 quant + P2P push) -> FC1 + SwiGLU (swap-AB W4A8 mma.sync) -> FC2 -> P2P push back -> top-k combine.
 * Design follows megamoe_basedondeepgemm (H20/B200): no cross-rank barriers, counters/flags embedded in the data flow
 * (epoch-tagged flags written by the producer into the consumer's memory, polled locally), swap-AB with the token slab
 * on MMA N. Compute core: w4a8_decode_gemm.cu (weights global -> registers, mma.sync kind::mxf8f6f4 m16n8k32).
 *
 * Weight layout (offline): W13 per expert [2I/16 blocks][H/64][32 lanes] uint4, block b = gate rows 8b..8b+7 then up
 * rows 8b..8b+7 (SwiGLU pairs land in one thread: acc rows g and g+8). W2 per expert [H/16][I/64][32] uint4.
 * Scales [blocks][K/128][16 rows] u32 (byte s = ue8m0 of K32 step s).
 * Every spin wait times out after kSpinNs and reports through the error word instead of hanging the GPU.
 */
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK_CUDA(x)                                                                      \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        TORCH_CHECK(e_ == cudaSuccess, "CUDA error: ", cudaGetErrorString(e_), " @ ", #x); \
    } while (0)

#ifndef NB_PER_WARP
#define NB_PER_WARP 1
#endif
constexpr int kWarps = 8;
constexpr int kNbPerWarp = NB_PER_WARP;
constexpr int kRowsPerItem = kWarps * kNbPerWarp * 16;  // 256 weight rows per work item
constexpr int kTile = 8;                                // tokens per item (MMA N)
#ifndef PF_BYTES
#define PF_BYTES 8192
#endif
constexpr int kPfBytes = PF_BYTES;                      // per 16-row weight block, prefetched to L2 during dispatch
#ifndef PF0_MB
#define PF0_MB 64
#endif
constexpr size_t kPf0Bytes = size_t(PF0_MB) << 20;      // t=0 L2 prefetch budget over the locally known experts
constexpr int kUnroll = 4;
constexpr int kMaxWorld = 8;
constexpr int kMaxEl = 256;  // TP mode (world 1): every expert is local
constexpr uint64_t kSpinNs = 30000000000ull;  // generous: startup warmups can skew ranks by seconds

// error codes
constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4;

// counters (local, reset by the last CTA to exit)
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4, kCtrEpoch = 5, kCtrL1 = 8;
constexpr int kCtrInts = kCtrL1 + kMaxEl;

struct Peers {
    uint8_t *pool_x[kMaxWorld];     // [EL][W][TCAP][H] e4m3            (dest-owned)
    uint32_t *pool_xs[kMaxWorld];   // [EL][W][TCAP][H/128] ue8m0 x4
    int *pool_meta[kMaxWorld];      // [EL][W][TCAP] src token*8 + k
    int *recv_cnt[kMaxWorld];       // [W][EL]
    int *recv_flag[kMaxWorld];      // [W] epoch
    __nv_bfloat16 *comb[kMaxWorld]; // [TCAP][TOPK][H]                    (src-owned)
    int *comb_flag[kMaxWorld];      // [W] epoch
    int *route[kMaxWorld];          // [W][1 + TCAP*TOPK]: T, then the source's top-k ids (dest-owned)
    int *route_flag[kMaxWorld];     // [W] epoch
};

struct Args {
    const __nv_bfloat16 *x;   // [T][H]
    const int *topk_ids;      // [T][TOPK] global expert ids
    const float *topk_w;      // [T][TOPK]
    __nv_bfloat16 *out;       // [T][H]
    const uint4 *w13;         // [EL] packed
    const uint32_t *ws13;
    const uint4 *w2;
    const uint32_t *ws2;
    uint8_t *inter;           // [EL][W*TCAP][I] e4m3 (local)
    uint32_t *inter_s;        // [EL][W*TCAP][I/128]
    int *ctr;                 // kCtrInts
    unsigned long long *stamps;  // optional [grid][8] globaltimer stamps (debug)
    Peers peers;
    int T, H, I, EL, W, TOPK, TCAP, rank;
    float swiglu_limit;
};

__device__ __forceinline__ uint64_t gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ int ld_acquire_sys(const int *p) {
    int v;
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ int ld_acquire_gpu(const int *p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_sys(int *p, int v) {
    asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// thread 0 spins until *p >= v (sys or gpu scope); returns false on timeout
__device__ __forceinline__ bool wait_ge(const int *p, int v, bool sys) {
    const uint64_t t0 = gtime();
    while ((sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) < v)
        if (gtime() - t0 > kSpinNs) return false;
    return true;
}

__device__ __forceinline__ void mma_w4a8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                         uint16_t byte_a, uint32_t sfb, uint16_t byte_b) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3."
        "f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(byte_a), "h"(uint16_t{0}),
          "r"(sfb), "h"(byte_b), "h"(uint16_t{0}));
}
__device__ __forceinline__ void expand(uint32_t w, uint32_t &lo, uint32_t &hi) {
    lo = (w & 0x0F0F0F0Fu) << 2;
    hi = (w >> 2) & 0x3C3C3C3Cu;
}

// warp GEMM over one 256-row item: acc[j] = rows (nb0+j)*16 + {g, g+8}, cols = tokens 2c, 2c+1.
// xrow: this lane's token row (e4m3, K bytes), xsrow: its ue8m0 x4 words (K/128).
// [kc_begin, kc_end) restricts K to a range of K64 chunks (split-K); both multiples of kUnroll (and of 2 for scales).
__device__ __forceinline__ void item_gemm(float (&acc)[kNbPerWarp][4], const uint4 *W, const uint32_t *WS, int nblocks,
                                          int nb0, int K, const uint8_t *xrow, const uint32_t *xsrow, int lane,
                                          int kc_begin = 0, int kc_end = -1) {
    const int g = lane / 4, c = lane % 4;
    const int kch = K / 64, kq = K / 128;
    const uint4 *wp[kNbPerWarp];
    const uint32_t *sp[kNbPerWarp];
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j) {
        wp[j] = W + size_t(nb0 + j) * kch * 32 + lane;
        sp[j] = WS + size_t(nb0 + j) * kq * 16 + g + (lane & 1) * 8;
    }
    const uint32_t *xp = reinterpret_cast<const uint32_t *>(xrow) + c;
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    if (kc_end < 0) kc_end = kch;
    for (int k0 = kc_begin; k0 < kc_end; k0 += kUnroll) {
        uint4 wv[kNbPerWarp][kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) wv[j][u] = __ldcs(wp[j] + (k0 + u) * 32);
        uint32_t sfa[kNbPerWarp][(kUnroll + 1) / 2];
#pragma unroll
        for (int q = 0; q < (kUnroll + 1) / 2; ++q)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) sfa[j][q] = __ldg(sp[j] + (k0 / 2 + q) * 16);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int kc = k0 + u;
            const uint32_t sfb = xsrow[kc / 2];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int step = kc * 2 + s;
                const uint32_t b0 = xp[step * 8];
                const uint32_t b1 = xp[step * 8 + 4];
                const uint16_t sel = uint16_t(step & 3);
#pragma unroll
                for (int j = 0; j < kNbPerWarp; ++j) {
                    const uint32_t w0 = s ? wv[j][u].z : wv[j][u].x;
                    const uint32_t w1 = s ? wv[j][u].w : wv[j][u].y;
                    uint32_t a[4];
                    expand(w0, a[0], a[1]);
                    expand(w1, a[2], a[3]);
                    mma_w4a8(acc[j], a, b0, b1, sfa[j][u / 2], sel, sfb, sel);
                }
            }
        }
    }
}

// quantize 32 consecutive fp32 values (in registers) to e4m3 with one ue8m0 scale; returns packed bytes + scale byte
__device__ __forceinline__ uint8_t quant32(const float (&v)[32], uint32_t (&q)[8]) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    int e = amax > 0.f ? int(ceilf(log2f(amax / 448.f))) : -127;
    e = max(-127, min(127, e));
    const float inv = exp2f(float(-e));
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const __nv_fp8_storage_t f = __nv_cvt_float_to_fp8(v[w * 4 + b] * inv, __NV_SATFINITE, __NV_E4M3);
            word |= uint32_t(f) << (8 * b);
        }
        q[w] = word;
    }
    return uint8_t(e + 127);
}

struct Map {  // per-expert token bookkeeping, built identically by every CTA after the dispatch flags
    int cnt[kMaxWorld][kMaxEl];
    int ne[kMaxEl];
    int tiles_prefix[kMaxEl + 1];
};

__device__ __forceinline__ void token_src(const Map &m, int el, int j, int W, int &w, int &i) {
    w = 0;
    i = j;
    while (w < W - 1 && i >= m.cnt[w][el]) {
        i -= m.cnt[w][el];
        ++w;
    }
}

__global__ void __launch_bounds__(kWarps * 32) sm120_mega_moe_kernel(Args a) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
    __shared__ Map m;
    __shared__ int s_item, s_ok, s_epoch;
    // L1: [tok][kRowsPerItem/2 inter] / L2: [tok][kRowsPerItem hidden] / dispatch: one MXFP8 token row (H + H/32 bytes)
    __shared__ __align__(16) float s_tile[kTile][kRowsPerItem < 256 ? 256 : kRowsPerItem];

    unsigned long long st_[16] = {gtime()};  // +8 route-push 9 route-ok 10 map 11 pf 12 flag-sent  // 0 start 1 pushed 2 flags 3 work end 4 comb flags 5 end 6 L1-wait ns 7 items
    if (threadIdx.x == 0) s_epoch = *reinterpret_cast<volatile int *>(a.ctr + kCtrEpoch) + 1;
    __syncthreads();
    const int epoch = s_epoch;
    int *ctr = a.ctr;

    // ---------------- routing exchange first (tiny): every rank learns the full expert load before the rows land
    const int RS = 1 + TCAP * TOPK;
    if (blockIdx.x == (a.T < int(gridDim.x) ? a.T : int(gridDim.x) - 1)) {
        for (int u = threadIdx.x; u < W * (1 + a.T * TOPK); u += blockDim.x) {
            const int dst = u / (1 + a.T * TOPK), v = u % (1 + a.T * TOPK);
            a.peers.route[dst][a.rank * RS + v] = v == 0 ? a.T : a.topk_ids[v - 1];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0)
            for (int dst = 0; dst < W; ++dst) st_release_sys(a.peers.route_flag[dst] + a.rank, epoch);
        st_[8] = gtime();
    }

    // ---------------- t=0: this rank's own tokens already tell which local experts are active; pull the head of
    // their FC1 weights into L2 while the rows and the other ranks' routing are in flight
    if (kPf0Bytes > 0 && EL <= 64) {
        __shared__ unsigned long long s_known;
        if (threadIdx.x == 0) s_known = 0;
        __syncthreads();
        for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
            const int e = a.topk_ids[u];
            if (e >= 0 && e / EL == a.rank) atomicOr(&s_known, 1ull << (e % EL));
        }
        __syncthreads();
        const unsigned long long known = s_known;
        const int nk = __popcll(known);
        if (nk > 0) {
            const size_t w13_expert_bytes = size_t(2 * I) * H / 2;
            const size_t per = min(w13_expert_bytes, kPf0Bytes / nk) & ~size_t(127);
            const size_t lines = per / 128 * nk;
            for (size_t l = size_t(blockIdx.x) * blockDim.x + threadIdx.x; l < lines; l += size_t(gridDim.x) * blockDim.x) {
                const int ki = int(l / (per / 128));
                unsigned long long rest = known;
                for (int q = 0; q < ki; ++q) rest &= rest - 1;
                const int el = __ffsll(rest) - 1;
                const char *p = reinterpret_cast<const char *>(a.w13) + el * w13_expert_bytes + (l % (per / 128)) * 128;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
            }
        }
    }

    // ---------------- dispatch: one CTA per token: quantize once to MXFP8 in smem, push TOPK copies with every thread
    const int nitems = a.T * TOPK;
    {
        uint8_t *s_q = reinterpret_cast<uint8_t *>(&s_tile[0][0]);  // [H] e4m3, then [H/32] ue8m0 (H + H/32 <= smem)
        uint8_t *s_sc = s_q + H;
        __shared__ int s_rowk[8], s_dstk[8];
        for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
            if (threadIdx.x < H / 32) {  // one thread per K32 group
                const uint4 *s4 = reinterpret_cast<const uint4 *>(a.x + size_t(t) * H + threadIdx.x * 32);
                float v[32];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint4 u = s4[q];
                    const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const float2 f = __bfloat1622float2(b2[r]);
                        v[q * 8 + r * 2] = f.x;
                        v[q * 8 + r * 2 + 1] = f.y;
                    }
                }
                uint32_t q8[8];
                s_sc[threadIdx.x] = quant32(v, q8);
                uint4 *d4 = reinterpret_cast<uint4 *>(s_q + threadIdx.x * 32);
                d4[0] = make_uint4(q8[0], q8[1], q8[2], q8[3]);
                d4[1] = make_uint4(q8[4], q8[5], q8[6], q8[7]);
            }
            if (warp < TOPK) {  // warp k: destination row of copy k (slot = tokens t' < t routed to the same expert)
                const int e = a.topk_ids[t * TOPK + warp];  // < 0: padding token (vLLM), nothing to send
                int slot = 0;
                for (int u = lane; u < t * TOPK; u += 32) slot += (a.topk_ids[u] == e);
#pragma unroll
                for (int o = 16; o; o >>= 1) slot += __shfl_xor_sync(0xffffffffu, slot, o);
                if (lane == 0) {
                    if (e < 0) {
                        s_rowk[warp] = -1;
                    } else {
                        const int dst = e / EL, el = e % EL;
                        const int row = (el * W + a.rank) * TCAP + slot;
                        s_rowk[warp] = row;
                        s_dstk[warp] = dst;
                        a.peers.pool_meta[dst][row] = t * 8 + warp;
                    }
                }
            }
            __syncthreads();
            const int per = H / 16 + H / 32 / 16;  // uint4 per copy: data + scales
            for (int u = threadIdx.x; u < TOPK * per; u += blockDim.x) {
                const int k = u / per, q = u % per;
                if (s_rowk[k] < 0) continue;
                const size_t row = size_t(s_rowk[k]);
                const int dst = s_dstk[k];
                if (q < H / 16)
                    reinterpret_cast<uint4 *>(a.peers.pool_x[dst] + row * H)[q] = reinterpret_cast<const uint4 *>(s_q)[q];
                else
                    reinterpret_cast<uint4 *>(a.peers.pool_xs[dst] + row * (H / 128))[q - H / 16] =
                        reinterpret_cast<const uint4 *>(s_sc)[q - H / 16];
            }
            __syncthreads();
        }
    }
    __threadfence_system();
    __syncthreads();
    st_[1] = gtime();
    if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
    __syncthreads();
    if (s_item && threadIdx.x == 0) {  // last CTA out of the dispatch: every row of this rank is visible -> flag
        __threadfence_system();
        for (int dst = 0; dst < W; ++dst) st_release_sys(a.peers.recv_flag[dst] + a.rank, epoch);
        st_[12] = gtime();
    }
    __syncthreads();

    // ---------------- per-expert map from the exchanged routing (same slot order as the dispatch)
    if (threadIdx.x == 0) {
        s_ok = 1;
        for (int w = 0; w < W; ++w)
            if (!wait_ge(a.peers.route_flag[a.rank] + w, epoch, true)) s_ok = 0;
        if (!s_ok) atomicExch(ctr + kCtrErr, kErrRouteWait);
    }
    __syncthreads();
    st_[9] = gtime();
    if (!s_ok) goto exit_kernel;
    {  // every source's routing in one parallel pass (all loads in flight together), then a warp scan for the prefix
        __shared__ int s_T[kMaxWorld];
        const int *rb = a.peers.route[a.rank];
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = 0;
        if (threadIdx.x < W) s_T[threadIdx.x] = __ldcv(rb + threadIdx.x * RS);
        __syncthreads();
        constexpr int kPer = 8;  // entries per thread per pass
        for (int base = 0; base < W * RS; base += kPer * int(blockDim.x)) {
            int e[kPer];
#pragma unroll
            for (int q = 0; q < kPer; ++q) {
                const int u = base + q * blockDim.x + threadIdx.x;
                const int w = u / RS, v = u % RS;
                e[q] = (u < W * RS && v > 0 && v - 1 < s_T[w] * TOPK) ? __ldcv(rb + u) : -1;  // -1 also = padding id
            }
#pragma unroll
            for (int q = 0; q < kPer; ++q) {
                const int u = base + q * blockDim.x + threadIdx.x;
                if (e[q] >= 0 && e[q] / EL == a.rank) atomicAdd(&m.cnt[u / RS][e[q] % EL], 1);  // e < 0: padding
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        int tiles_sum = 0;
        for (int base = 0; base < EL; base += 32) {
            const int el = base + lane;
            int n = 0;
            if (el < EL)
                for (int w = 0; w < W; ++w) n += m.cnt[w][el];
            int incl = (n + kTile - 1) / kTile;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            if (el < EL) {
                m.ne[el] = n;
                m.tiles_prefix[el + 1] = tiles_sum + incl;
            }
            tiles_sum += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) m.tiles_prefix[0] = 0;
    }
    __syncthreads();
    st_[10] = gtime();
    {  // this CTA's first FC1 item is static (item = blockIdx.x): start pulling its weights into L2 now
        const int nblk1 = 2 * I / kRowsPerItem;
        const int item = blockIdx.x;
        if (item < m.tiles_prefix[EL] * nblk1) {
            const int tile_g = item / nblk1, rb = item % nblk1;
            int el = 0;
            while (m.tiles_prefix[el + 1] <= tile_g) ++el;
            const size_t w13_expert = size_t(2 * I / 16) * (H / 64) * 32;
            const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
            const int blk_bytes = (H / 64) * 32 * 16;
#pragma unroll
            for (int jb = 0; jb < kNbPerWarp; ++jb) {
                const char *base = reinterpret_cast<const char *>(a.w13 + el * w13_expert + size_t(nb0 + jb) * (H / 64) * 32);
                for (int off = lane * 128; off < min(kPfBytes, blk_bytes); off += 32 * 128)
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(base + off));
            }
        }
    }
    st_[11] = gtime();

    // ---------------- wait for every source rank, build the per-expert map
    if (threadIdx.x == 0) {
        s_ok = 1;
        for (int w = 0; w < W; ++w)
            if (!wait_ge(a.peers.recv_flag[a.rank] + w, epoch, true)) s_ok = 0;
        if (!s_ok) atomicExch(ctr + kCtrErr, kErrDispatchWait);
    }
    __syncthreads();
    st_[2] = gtime();
    if (!s_ok) goto exit_kernel;
    {
        const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
        const int ntiles = m.tiles_prefix[EL];
        const int n1 = ntiles * nblk1, n2 = ntiles * nblk2;
        const uint8_t *pool_x = a.peers.pool_x[a.rank];
        const uint32_t *pool_xs = a.peers.pool_xs[a.rank];
        const int *pool_meta = a.peers.pool_meta[a.rank];
        const int g = lane / 4, c = lane % 4;
        const size_t w13_expert = size_t(2 * I / 16) * (H / 64) * 32, ws13_expert = size_t(2 * I / 16) * (H / 128) * 16;
        const size_t w2_expert = size_t(H / 16) * (I / 64) * 32, ws2_expert = size_t(H / 16) * (I / 128) * 16;
        const int TOKE = W * TCAP;

        bool first = true;
        while (true) {
            if (threadIdx.x == 0) s_item = first ? int(blockIdx.x) : int(gridDim.x) + atomicAdd(ctr + kCtrQueue, 1);
            first = false;
            __syncthreads();
            const int item = s_item;
            if (item >= n1 + n2) break;
            st_[7] += 1;
            const bool l1 = item < n1;
            const int li = l1 ? item : item - n1;
            const int nblk = l1 ? nblk1 : nblk2;
            const int tile_g = li / nblk, rb = li % nblk;
            int el = 0;
            while (m.tiles_prefix[el + 1] <= tile_g) ++el;
            const int tile = tile_g - m.tiles_prefix[el];
            const int ntok = min(kTile, m.ne[el] - tile * kTile);
            const int j = tile * kTile + (g < ntok ? g : 0);  // this lane's token (B column g)

            if (!l1) {  // FC2 needs every FC1 item of this expert
                const unsigned long long tw0 = gtime();
                if (threadIdx.x == 0) {
                    const int need = ((m.ne[el] + kTile - 1) / kTile) * nblk1;
                    if (!wait_ge(ctr + kCtrL1 + el, need, false)) {
                        atomicExch(ctr + kCtrErr, kErrL1Wait);
                        s_ok = 0;
                    }
                }
                __syncthreads();
                st_[6] += gtime() - tw0;
                if (!s_ok) goto exit_kernel;
            }

            float acc[kNbPerWarp][4];
            const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
            if (l1) {
                int w, i;
                token_src(m, el, j, W, w, i);
                const size_t row = (size_t(el) * W + w) * TCAP + i;
                item_gemm(acc, a.w13 + el * w13_expert, a.ws13 + el * ws13_expert, 2 * I / 16, nb0, H,
                          pool_x + row * H, pool_xs + row * (H / 128), lane);
                // SwiGLU: acc rows g (gate) / g+8 (up) of block nb -> inter col nb*8+g; tokens 2c, 2c+1
                constexpr int kItemInter = kRowsPerItem / 2;  // inter cols per item (multiple of 32)
                constexpr int kGroups = kItemInter / 32;
#pragma unroll
                for (int jb = 0; jb < kNbPerWarp; ++jb) {
                    const int col = (warp * kNbPerWarp + jb) * 8 + g;  // 0..kItemInter-1 within the item
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float gate = fminf(acc[jb][h], a.swiglu_limit);
                        const float up = fminf(fmaxf(acc[jb][2 + h], -a.swiglu_limit), a.swiglu_limit);
                        s_tile[2 * c + h][col] = gate / (1.f + __expf(-gate)) * up;
                    }
                }
                __syncthreads();
                if (threadIdx.x < kTile * kGroups) {  // one thread per (token, K32 group)
                    const int tk = threadIdx.x / kGroups, gi = threadIdx.x % kGroups;
                    float v[32];
#pragma unroll
                    for (int u = 0; u < 32; ++u) v[u] = s_tile[tk][gi * 32 + u];
                    uint32_t q8[8];
                    const uint8_t sc = quant32(v, q8);
                    if (tk < ntok) {
                        const size_t r = size_t(el) * TOKE + tile * kTile + tk;
                        const int col0 = rb * kItemInter + gi * 32;
                        uint4 *d4 = reinterpret_cast<uint4 *>(a.inter + r * I + col0);
                        d4[0] = make_uint4(q8[0], q8[1], q8[2], q8[3]);
                        d4[1] = make_uint4(q8[4], q8[5], q8[6], q8[7]);
                        reinterpret_cast<uint8_t *>(a.inter_s + r * (I / 128))[col0 / 32] = sc;
                    }
                }
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) atomicAdd(ctr + kCtrL1 + el, 1);
            } else {
                const size_t r = size_t(el) * TOKE + j;
                item_gemm(acc, a.w2 + el * w2_expert, a.ws2 + el * ws2_expert, H / 16, nb0, I, a.inter + r * I,
                          a.inter_s + r * (I / 128), lane);
#pragma unroll
                for (int jb = 0; jb < kNbPerWarp; ++jb) {
                    const int col = (warp * kNbPerWarp + jb) * 16 + g;  // 0..kRowsPerItem-1 within the item
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        s_tile[2 * c + h][col] = acc[jb][h];
                        s_tile[2 * c + h][col + 8] = acc[jb][2 + h];
                    }
                }
                __syncthreads();
                // push: token tk of this tile -> its source rank's combine slot, kRowsPerItem bf16 in 16 B stores
                constexpr int kVec = kRowsPerItem / 8;
                for (int u = threadIdx.x; u < ntok * kVec; u += blockDim.x) {
                    const int tk = u / kVec, q = u % kVec;
                    int w, i;
                    token_src(m, el, tile * kTile + tk, W, w, i);
                    const int meta = pool_meta[(size_t(el) * W + w) * TCAP + i];
                    const int st = meta / 8, sk = meta % 8;
                    uint32_t pk[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const __nv_bfloat162 b2 =
                            __floats2bfloat162_rn(s_tile[tk][q * 8 + 2 * p], s_tile[tk][q * 8 + 2 * p + 1]);
                        pk[p] = *reinterpret_cast<const uint32_t *>(&b2);
                    }
                    __nv_bfloat16 *dst = a.peers.comb[w] + (size_t(st) * TOPK + sk) * H + rb * kRowsPerItem + q * 8;
                    *reinterpret_cast<uint4 *>(dst) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
                }
                __threadfence_system();
                __syncthreads();
                if (threadIdx.x == 0) {
                    const int done = atomicAdd(ctr + kCtrL2Done, 1);
                    if (done == n2 - 1) {  // this rank's last FC2 item: tell every source rank
                        __threadfence_system();
                        for (int w = 0; w < W; ++w) st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
                    }
                }
            }
            __syncthreads();
        }
        st_[3] = gtime();
        if (n2 == 0 && blockIdx.x == 0 && threadIdx.x == 0) {  // no local work: still release every source
            __threadfence_system();
            for (int w = 0; w < W; ++w) st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
        }
    }

    // ---------------- combine: wait for every expert rank, out[t] = sum_k w[t][k] * partial[t][k]
    if (threadIdx.x == 0) {
        s_ok = 1;
        for (int w = 0; w < W; ++w)
            if (!wait_ge(a.peers.comb_flag[a.rank] + w, epoch, true)) s_ok = 0;
        if (!s_ok) atomicExch(ctr + kCtrErr, kErrCombineWait);
    }
    __syncthreads();
    st_[4] = gtime();
    if (s_ok) {
        const __nv_bfloat16 *comb = a.peers.comb[a.rank];
        const int nvec = a.T * H / 8;
        for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += gridDim.x * blockDim.x) {
            const int t = v / (H / 8), h8 = (v % (H / 8)) * 8;
            float s[8] = {};
            for (int k = 0; k < TOPK; ++k) {
                if (a.topk_ids[t * TOPK + k] < 0) continue;  // padding: never dispatched
                const float wk = a.topk_w[t * TOPK + k];
                const uint4 u = *reinterpret_cast<const uint4 *>(comb + (size_t(t) * TOPK + k) * H + h8);
                const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float2 f = __bfloat1622float2(b2[r]);
                    s[2 * r] += wk * f.x;
                    s[2 * r + 1] += wk * f.y;
                }
            }
            uint32_t pk[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __nv_bfloat162 b2 = __floats2bfloat162_rn(s[2 * r], s[2 * r + 1]);
                pk[r] = *reinterpret_cast<const uint32_t *>(&b2);
            }
            *reinterpret_cast<uint4 *>(a.out + size_t(t) * H + h8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
    }

exit_kernel:
    __syncthreads();
    st_[5] = gtime();
    if (a.stamps && threadIdx.x == 0)
        for (int u = 0; u < 16; ++u) a.stamps[blockIdx.x * 16 + u] = st_[u];
    if (threadIdx.x == 0) {
        __threadfence();
        const int done = atomicAdd(ctr + kCtrExit, 1);
        if (done == int(gridDim.x) - 1) {  // last CTA out: advance the epoch, reset the per-call counters
            ctr[kCtrEpoch] = epoch;
            ctr[kCtrDispatch] = 0;
            ctr[kCtrQueue] = 0;
            ctr[kCtrL2Done] = 0;
            for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
            __threadfence();
            ctr[kCtrExit] = 0;
        }
    }
}

// ================================================================================================ TP mode
// One GPU, every expert local with an I/TP slice (vLLM MoE-TP, the all-reduce stays outside). No dispatch: each FC1
// item quantizes the K slice of its (<= 8) tokens into smem; FC1 is split-K when few experts are active (fp32 red into
// acc1, the last split does SwiGLU + MXFP8); FC2 applies the top-k weight and reds into acc2 [T][H]; after the last FC2
// item every CTA converts its share of acc2 to bf16 and zeroes it.
constexpr int kTpMaxSlots = 64 * 8;    // T * TOPK
constexpr int kTpMaxE = 256;
constexpr int kTpCtrQ = 0, kTpCtrFc2 = 1, kTpCtrExit = 2, kTpCtrErr = 3, kTpCtrFc1Done = 8;
constexpr int kTpCtrRb = kTpCtrFc1Done + kTpMaxE;
constexpr int kTpMaxRb = kTpMaxSlots * 32;  // tiles x FC1 row blocks
constexpr int kTpCtrInts = kTpCtrRb + kTpMaxRb;

struct TpArgs {
    const __nv_bfloat16 *x;
    const int *topk_ids;
    const float *topk_w;
    __nv_bfloat16 *out;
    const uint4 *w13;
    const uint32_t *ws13;
    const uint4 *w2;
    const uint32_t *ws2;
    float *acc1;          // [T*TOPK][2I] fp32 (zero between calls)
    uint8_t *inter;       // [T*TOPK][I] e4m3
    uint32_t *inter_s;    // [T*TOPK][I/128]
    float *acc2;          // [T][H] fp32 (zero between calls)
    int *ctr;
    unsigned long long *stamps;
    int T, H, I, E, TOPK, nsm;
    float swiglu_limit;
};

__global__ void __launch_bounds__(kWarps * 32) sm120_moe_tp_kernel(TpArgs a) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int H = a.H, I = a.I, E = a.E, TOPK = a.TOPK;
    const int g = lane / 4, c = lane % 4;
    __shared__ int s_cnt[kTpMaxE], s_tokp[kTpMaxE + 1], s_tilep[kTpMaxE + 1];
    __shared__ short s_slot[kTpMaxSlots];
    __shared__ int s_item, s_ok, s_last;
    // FC1: kTile token rows of the K slice (e4m3 + scale words); reused as the fp32 epilogue tile
    __shared__ __align__(16) uint8_t s_buf[kTile * (4096 + 128)];
    unsigned long long st_[16] = {gtime()};
    int *ctr = a.ctr;

    // ---- expert map from the local routing (identical in every CTA)
    const int nslot = a.T * TOPK;
    for (int u = threadIdx.x; u < E; u += blockDim.x) s_cnt[u] = 0;
    __syncthreads();
    for (int u = threadIdx.x; u < nslot; u += blockDim.x) {
        const int e = a.topk_ids[u];
        if (e >= 0) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();
    if (warp == 0) {
        int tok_sum = 0, tile_sum = 0;
        for (int base = 0; base < E; base += 32) {
            const int e = base + lane;
            const int n = e < E ? s_cnt[e] : 0;
            int it = n, ti = (n + kTile - 1) / kTile;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v1 = __shfl_up_sync(0xffffffffu, it, o), v2 = __shfl_up_sync(0xffffffffu, ti, o);
                if (lane >= o) it += v1, ti += v2;
            }
            if (e < E) s_tokp[e + 1] = tok_sum + it, s_tilep[e + 1] = tile_sum + ti;
            tok_sum += __shfl_sync(0xffffffffu, it, 31);
            tile_sum += __shfl_sync(0xffffffffu, ti, 31);
        }
        if (lane == 0) s_tokp[0] = 0, s_tilep[0] = 0;
    }
    __syncthreads();
    for (int u = threadIdx.x; u < nslot; u += blockDim.x) {  // slot position: expert prefix + rank among same-expert slots
        const int e = a.topk_ids[u];
        if (e < 0) continue;
        int r = 0;
        for (int v = 0; v < u; ++v) r += (a.topk_ids[v] == e);
        s_slot[s_tokp[e] + r] = short(u);
    }
    __syncthreads();
    st_[10] = gtime();

    const int ntiles = s_tilep[E];
    const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
    int ks = 1;  // split-K for FC1 so that enough items are in flight
    while (ks < 16 && ntiles * nblk1 * ks < 2 * a.nsm && (H / 64) % (ks * 2 * kUnroll) == 0) ks *= 2;
    const int n1 = ntiles * nblk1 * ks, n2 = ntiles * nblk2;
    const int kch1 = H / 64, kc_per = kch1 / ks;
    const size_t w13_expert = size_t(2 * I / 16) * (H / 64) * 32, ws13_expert = size_t(2 * I / 16) * (H / 128) * 16;
    const size_t w2_expert = size_t(H / 16) * (I / 64) * 32, ws2_expert = size_t(H / 16) * (I / 128) * 16;
    float *s_tile = reinterpret_cast<float *>(s_buf);  // [kTile][kRowsPerItem] after the GEMM

#ifndef TP_PF_FC2
#define TP_PF_FC2 0  // measured slower (TP8 T1 33 -> 45 us): the per-thread expert loop + extra traffic
#endif
    if (TP_PF_FC2) {  // the active experts are known now: pull their FC2 weights into L2 while FC1 runs
        const size_t lines = size_t(H) * I / 2 / 128;
        const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x, nth = size_t(gridDim.x) * blockDim.x;
        for (int e = 0; e < E; ++e) {
            if (s_cnt[e] == 0) continue;
            const char *base = reinterpret_cast<const char *>(a.w2 + e * w2_expert);
            for (size_t l = tid; l < lines; l += nth) asm volatile("prefetch.global.L2 [%0];" ::"l"(base + l * 128));
        }
    }

    bool first = true;
    while (true) {
        if (threadIdx.x == 0) s_item = first ? int(blockIdx.x) : int(gridDim.x) + atomicAdd(ctr + kTpCtrQ, 1);
        first = false;
        __syncthreads();
        const int item = s_item;
        if (item >= n1 + n2) break;
        st_[7] += 1;
        const bool l1 = item < n1;
        int li = l1 ? item : item - n1;
        const int kseg = l1 ? li % ks : 0;
        if (l1) li /= ks;
        const int nblk = l1 ? nblk1 : nblk2;
        const int tile_g = li / nblk, rb = li % nblk;
        int e = 0;
        while (s_tilep[e + 1] <= tile_g) ++e;
        const int tile = tile_g - s_tilep[e];
        const int ntok = min(kTile, s_cnt[e] - tile * kTile);
        const int p0 = s_tokp[e] + tile * kTile;  // first slot position of this tile
        float acc[kNbPerWarp][4];
        const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
        if (l1) {
            const int kc0 = kseg * kc_per, kc1 = kc0 + kc_per;
            const int sl = kc_per * 64;                    // slice bytes
            uint8_t *sx = s_buf;                           // [kTile][sl]
            uint32_t *sxs = reinterpret_cast<uint32_t *>(s_buf + kTile * sl);  // [kTile][sl/128]
            const int ngrp = sl / 32;
            for (int q = threadIdx.x; q < kTile * ngrp; q += blockDim.x) {  // quantize this K slice of each token
                const int tk = q / ngrp, gi = q % ngrp;
                float v[32];
                if (tk < ntok) {
                    const int t = s_slot[p0 + tk] / TOPK;
                    const uint4 *s4 = reinterpret_cast<const uint4 *>(a.x + size_t(t) * H + kc0 * 64 + gi * 32);
#pragma unroll
                    for (int qq = 0; qq < 4; ++qq) {
                        const uint4 u = s4[qq];
                        const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const float2 f = __bfloat1622float2(b2[r]);
                            v[qq * 8 + r * 2] = f.x;
                            v[qq * 8 + r * 2 + 1] = f.y;
                        }
                    }
                } else {
#pragma unroll
                    for (int u = 0; u < 32; ++u) v[u] = 0.f;
                }
                uint32_t q8[8];
                const uint8_t sc = quant32(v, q8);
                uint4 *d4 = reinterpret_cast<uint4 *>(sx + tk * sl + gi * 32);
                d4[0] = make_uint4(q8[0], q8[1], q8[2], q8[3]);
                d4[1] = make_uint4(q8[4], q8[5], q8[6], q8[7]);
                reinterpret_cast<uint8_t *>(sxs + tk * (sl / 128))[gi] = sc;
            }
            __syncthreads();
            const int tkl = g < ntok ? g : 0;
            // absolute K indexing inside item_gemm: shift the smem row back by the slice start
            item_gemm(acc, a.w13 + e * w13_expert, a.ws13 + e * ws13_expert, 2 * I / 16, nb0, H,
                      sx + tkl * sl - kc0 * 64, sxs + tkl * (sl / 128) - kc0 / 2, lane, kc0, kc1);
            __syncthreads();  // smem slice no longer needed
            if (ks == 1) {
#pragma unroll
                for (int jb = 0; jb < kNbPerWarp; ++jb) {
                    const int col = (warp * kNbPerWarp + jb) * 8 + g;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float gate = fminf(acc[jb][h], a.swiglu_limit);
                        const float up = fminf(fmaxf(acc[jb][2 + h], -a.swiglu_limit), a.swiglu_limit);
                        s_tile[(2 * c + h) * kRowsPerItem + col] = gate / (1.f + __expf(-gate)) * up;
                    }
                }
                if (threadIdx.x == 0) s_last = 1;
            } else {  // red the partial, the last split finishes the row block
#pragma unroll
                for (int jb = 0; jb < kNbPerWarp; ++jb) {
                    const int row = (nb0 + jb) * 16 + g;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int tk = 2 * c + h;
                        if (tk < ntok) {
                            float *r = a.acc1 + size_t(p0 + tk) * (2 * I) + row;
                            atomicAdd(r, acc[jb][h]);
                            atomicAdd(r + 8, acc[jb][2 + h]);
                        }
                    }
                }
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) s_last = (atomicAdd(ctr + kTpCtrRb + tile_g * nblk1 + rb, 1) == ks - 1);
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    for (int q = threadIdx.x; q < kTile * (kRowsPerItem / 2); q += blockDim.x) {
                        const int tk = q / (kRowsPerItem / 2), col = q % (kRowsPerItem / 2);
                        float val = 0.f;
                        if (tk < ntok) {
                            const int blk = col / 8, gg = col % 8;
                            float *r = a.acc1 + size_t(p0 + tk) * (2 * I) + rb * kRowsPerItem + blk * 16 + gg;
                            const float gate = fminf(__ldcg(r), a.swiglu_limit);
                            const float up = fminf(fmaxf(__ldcg(r + 8), -a.swiglu_limit), a.swiglu_limit);
                            r[0] = 0.f;
                            r[8] = 0.f;
                            val = gate / (1.f + __expf(-gate)) * up;
                        }
                        s_tile[tk * kRowsPerItem + col] = val;
                    }
                    if (threadIdx.x == 0) ctr[kTpCtrRb + tile_g * nblk1 + rb] = 0;
                }
            }
            __syncthreads();
            if (s_last) {
                constexpr int kItemInter = kRowsPerItem / 2, kGroups = kItemInter / 32;
                if (threadIdx.x < kTile * kGroups) {
                    const int tk = threadIdx.x / kGroups, gi = threadIdx.x % kGroups;
                    float v[32];
#pragma unroll
                    for (int u = 0; u < 32; ++u) v[u] = s_tile[tk * kRowsPerItem + gi * 32 + u];
                    uint32_t q8[8];
                    const uint8_t sc = quant32(v, q8);
                    if (tk < ntok) {
                        const size_t r = size_t(p0 + tk);
                        const int col0 = rb * kItemInter + gi * 32;
                        uint4 *d4 = reinterpret_cast<uint4 *>(a.inter + r * I + col0);
                        d4[0] = make_uint4(q8[0], q8[1], q8[2], q8[3]);
                        d4[1] = make_uint4(q8[4], q8[5], q8[6], q8[7]);
                        reinterpret_cast<uint8_t *>(a.inter_s + r * (I / 128))[col0 / 32] = sc;
                    }
                }
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) atomicAdd(ctr + kTpCtrFc1Done + e, 1);
            }
        } else {
            if (threadIdx.x == 0) {
                const int need = (s_tilep[e + 1] - s_tilep[e]) * nblk1;
                s_ok = wait_ge(ctr + kTpCtrFc1Done + e, need, false);
                if (!s_ok) atomicExch(ctr + kTpCtrErr, kErrL1Wait);
            }
            __syncthreads();
            if (!s_ok) break;
            const int j = p0 + (g < ntok ? g : 0);
            item_gemm(acc, a.w2 + e * w2_expert, a.ws2 + e * ws2_expert, H / 16, nb0, I, a.inter + size_t(j) * I,
                      a.inter_s + size_t(j) * (I / 128), lane);
#pragma unroll
            for (int jb = 0; jb < kNbPerWarp; ++jb) {
                const int row = (nb0 + jb) * 16 + g;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int tk = 2 * c + h;
                    if (tk < ntok) {
                        const int slot = s_slot[p0 + tk];
                        const float wk = a.topk_w[slot];
                        float *o = a.acc2 + size_t(slot / TOPK) * H + row;
                        atomicAdd(o, wk * acc[jb][h]);
                        atomicAdd(o + 8, wk * acc[jb][2 + h]);
                    }
                }
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) atomicAdd(ctr + kTpCtrFc2, 1);
        }
        __syncthreads();
    }
    st_[3] = gtime();
    // ---- every FC2 item done -> bf16 output, zero acc2
    if (threadIdx.x == 0) {
        s_ok = wait_ge(ctr + kTpCtrFc2, n2, false);
        if (!s_ok) atomicExch(ctr + kTpCtrErr, kErrCombineWait);
    }
    __syncthreads();
    st_[4] = gtime();
    if (s_ok) {
        const int nvec = a.T * H / 8;
        for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += gridDim.x * blockDim.x) {
            float4 *src = reinterpret_cast<float4 *>(a.acc2 + size_t(v) * 8);
            const float4 f0 = __ldcg(src), f1 = __ldcg(src + 1);
            src[0] = make_float4(0.f, 0.f, 0.f, 0.f);
            src[1] = make_float4(0.f, 0.f, 0.f, 0.f);
            const __nv_bfloat162 b0 = __floats2bfloat162_rn(f0.x, f0.y), b1 = __floats2bfloat162_rn(f0.z, f0.w);
            const __nv_bfloat162 b2 = __floats2bfloat162_rn(f1.x, f1.y), b3 = __floats2bfloat162_rn(f1.z, f1.w);
            *reinterpret_cast<uint4 *>(a.out + size_t(v) * 8) =
                make_uint4(*reinterpret_cast<const uint32_t *>(&b0), *reinterpret_cast<const uint32_t *>(&b1),
                           *reinterpret_cast<const uint32_t *>(&b2), *reinterpret_cast<const uint32_t *>(&b3));
        }
    }
    __syncthreads();
    st_[5] = gtime();
    if (a.stamps && threadIdx.x == 0)
        for (int u = 0; u < 16; ++u) a.stamps[blockIdx.x * 16 + u] = st_[u];
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(ctr + kTpCtrExit, 1) == int(gridDim.x) - 1) {
            ctr[kTpCtrQ] = 0;
            ctr[kTpCtrFc2] = 0;
            for (int u = 0; u < E; ++u) ctr[kTpCtrFc1Done + u] = 0;
            __threadfence();
            ctr[kTpCtrExit] = 0;
        }
    }
}

// ------------------------------------------------------------------------------------------------ host
struct State {
    int rank = 0, world = 0, EL = 0, TCAP = 0, H = 0, I = 0, TOPK = 0;
    void *local_alloc = nullptr;  // one cudaMalloc holding every peer-visible buffer
    size_t off_pool_x, off_pool_xs, off_meta, off_rcnt, off_rflag, off_comb, off_cflag, off_route, off_rtflag, total;
    void *peer_base[kMaxWorld] = {};
    uint8_t *inter = nullptr;
    uint32_t *inter_s = nullptr;
    int *ctr = nullptr;
    int grid = 0;
};
static State g;

static size_t align_up(size_t v) { return (v + 255) & ~size_t(255); }

// returns this rank's cudaIpcMemHandle bytes
torch::Tensor init(int rank, int world, int EL, int TCAP, int H, int I, int TOPK) {
    TORCH_CHECK(world <= kMaxWorld && EL <= kMaxEl && H % kRowsPerItem == 0 && (2 * I) % kRowsPerItem == 0 &&
                H + H / 32 <= kTile * 256 * 4 && H / 32 <= kWarps * 32 && TOPK <= 8);
    g.rank = rank, g.world = world, g.EL = EL, g.TCAP = TCAP, g.H = H, g.I = I, g.TOPK = TOPK;
    size_t o = 0;
    g.off_pool_x = o; o = align_up(o + size_t(EL) * world * TCAP * H);
    g.off_pool_xs = o; o = align_up(o + size_t(EL) * world * TCAP * (H / 128) * 4);
    g.off_meta = o; o = align_up(o + size_t(EL) * world * TCAP * 4);
    g.off_rcnt = o; o = align_up(o + size_t(world) * EL * 4);
    g.off_rflag = o; o = align_up(o + size_t(world) * 4);
    g.off_comb = o; o = align_up(o + size_t(TCAP) * TOPK * H * 2);
    g.off_cflag = o; o = align_up(o + size_t(world) * 4);
    g.off_route = o; o = align_up(o + size_t(world) * (1 + TCAP * TOPK) * 4);
    g.off_rtflag = o; o = align_up(o + size_t(world) * 4);
    g.total = o;
    CHECK_CUDA(cudaMalloc(&g.local_alloc, g.total));
    CHECK_CUDA(cudaMemset(g.local_alloc, 0, g.total));
    CHECK_CUDA(cudaMalloc(&g.inter, size_t(EL) * world * TCAP * I));
    CHECK_CUDA(cudaMalloc(&g.inter_s, size_t(EL) * world * TCAP * (I / 128) * 4));
    CHECK_CUDA(cudaMalloc(&g.ctr, kCtrInts * 4));
    CHECK_CUDA(cudaMemset(g.ctr, 0, kCtrInts * 4));
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    int occ = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sm120_mega_moe_kernel, kWarps * 32, 0));
    g.grid = occ * prop.multiProcessorCount;
    cudaIpcMemHandle_t h;
    CHECK_CUDA(cudaIpcGetMemHandle(&h, g.local_alloc));
    auto t = torch::empty({int64_t(sizeof(h))}, torch::dtype(torch::kUInt8));
    std::memcpy(t.data_ptr(), &h, sizeof(h));
    return t;
}

void open_peers(std::vector<torch::Tensor> handles) {
    for (int w = 0; w < g.world; ++w) {
        if (w == g.rank) {
            g.peer_base[w] = g.local_alloc;
            continue;
        }
        cudaIpcMemHandle_t h;
        std::memcpy(&h, handles[w].data_ptr(), sizeof(h));
        CHECK_CUDA(cudaIpcOpenMemHandle(&g.peer_base[w], h, cudaIpcMemLazyEnablePeerAccess));
    }
}

int grid_size() { return g.grid; }

std::vector<int64_t> counters() {
    std::vector<int> h(kCtrInts);
    CHECK_CUDA(cudaMemcpy(h.data(), g.ctr, kCtrInts * 4, cudaMemcpyDeviceToHost));
    return std::vector<int64_t>(h.begin(), h.end());
}

void run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out, torch::Tensor w13,
         torch::Tensor ws13, torch::Tensor w2, torch::Tensor ws2, double swiglu_limit,
         c10::optional<torch::Tensor> stamps) {
    Args a{};
    a.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
    a.topk_ids = topk_ids.data_ptr<int>();
    a.topk_w = topk_w.data_ptr<float>();
    a.out = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    a.w13 = reinterpret_cast<const uint4 *>(w13.data_ptr());
    a.ws13 = reinterpret_cast<const uint32_t *>(ws13.data_ptr());
    a.w2 = reinterpret_cast<const uint4 *>(w2.data_ptr());
    a.ws2 = reinterpret_cast<const uint32_t *>(ws2.data_ptr());
    a.inter = g.inter;
    a.inter_s = g.inter_s;
    a.ctr = g.ctr;
    a.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long *>(stamps->data_ptr()) : nullptr;
    for (int w = 0; w < g.world; ++w) {
        uint8_t *b = static_cast<uint8_t *>(g.peer_base[w]);
        a.peers.pool_x[w] = b + g.off_pool_x;
        a.peers.pool_xs[w] = reinterpret_cast<uint32_t *>(b + g.off_pool_xs);
        a.peers.pool_meta[w] = reinterpret_cast<int *>(b + g.off_meta);
        a.peers.recv_cnt[w] = reinterpret_cast<int *>(b + g.off_rcnt);
        a.peers.recv_flag[w] = reinterpret_cast<int *>(b + g.off_rflag);
        a.peers.comb[w] = reinterpret_cast<__nv_bfloat16 *>(b + g.off_comb);
        a.peers.comb_flag[w] = reinterpret_cast<int *>(b + g.off_cflag);
        a.peers.route[w] = reinterpret_cast<int *>(b + g.off_route);
        a.peers.route_flag[w] = reinterpret_cast<int *>(b + g.off_rtflag);
    }
    a.T = int(x.size(0));
    TORCH_CHECK(a.T <= g.TCAP, "T exceeds TCAP");
    a.H = g.H, a.I = g.I, a.EL = g.EL, a.W = g.world, a.TOPK = g.TOPK, a.TCAP = g.TCAP, a.rank = g.rank;
    a.swiglu_limit = float(swiglu_limit);
    auto stream = c10::cuda::getCurrentCUDAStream();
    sm120_mega_moe_kernel<<<g.grid, kWarps * 32, 0, stream>>>(a);
    CHECK_CUDA(cudaGetLastError());
}


struct TpState {
    int E = 0, H = 0, I = 0, TOPK = 0, TCAP = 0, grid = 0, nsm = 0;
    float *acc1 = nullptr, *acc2 = nullptr;
    uint8_t *inter = nullptr;
    uint32_t *inter_s = nullptr;
    int *ctr = nullptr;
};
static TpState gt;

void init_tp(int E, int TCAP, int H, int I, int TOPK) {
    TORCH_CHECK(E <= kTpMaxE && TCAP * TOPK <= kTpMaxSlots && H % kRowsPerItem == 0 && (2 * I) % kRowsPerItem == 0 &&
                H <= 4096 && I % 128 == 0);
    gt.E = E, gt.H = H, gt.I = I, gt.TOPK = TOPK, gt.TCAP = TCAP;
    const size_t slots = size_t(TCAP) * TOPK;
    CHECK_CUDA(cudaMalloc(&gt.acc1, slots * 2 * I * 4));
    CHECK_CUDA(cudaMemset(gt.acc1, 0, slots * 2 * I * 4));
    CHECK_CUDA(cudaMalloc(&gt.inter, slots * I));
    CHECK_CUDA(cudaMalloc(&gt.inter_s, slots * (I / 128) * 4));
    CHECK_CUDA(cudaMalloc(&gt.acc2, size_t(TCAP) * H * 4));
    CHECK_CUDA(cudaMemset(gt.acc2, 0, size_t(TCAP) * H * 4));
    CHECK_CUDA(cudaMalloc(&gt.ctr, kTpCtrInts * 4));
    CHECK_CUDA(cudaMemset(gt.ctr, 0, kTpCtrInts * 4));
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    int occ = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sm120_moe_tp_kernel, kWarps * 32, 0));
    gt.nsm = prop.multiProcessorCount;
    gt.grid = occ * prop.multiProcessorCount;
}

int tp_grid_size() { return gt.grid; }
int64_t tp_error() {
    int v = 0;
    CHECK_CUDA(cudaMemcpy(&v, gt.ctr + kTpCtrErr, 4, cudaMemcpyDeviceToHost));
    return v;
}

void run_tp(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out, torch::Tensor w13,
            torch::Tensor ws13, torch::Tensor w2, torch::Tensor ws2, double swiglu_limit,
            c10::optional<torch::Tensor> stamps) {
    TpArgs a{};
    a.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
    a.topk_ids = topk_ids.data_ptr<int>();
    a.topk_w = topk_w.data_ptr<float>();
    a.out = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    a.w13 = reinterpret_cast<const uint4 *>(w13.data_ptr());
    a.ws13 = reinterpret_cast<const uint32_t *>(ws13.data_ptr());
    a.w2 = reinterpret_cast<const uint4 *>(w2.data_ptr());
    a.ws2 = reinterpret_cast<const uint32_t *>(ws2.data_ptr());
    a.acc1 = gt.acc1, a.inter = gt.inter, a.inter_s = gt.inter_s, a.acc2 = gt.acc2, a.ctr = gt.ctr;
    a.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long *>(stamps->data_ptr()) : nullptr;
    a.T = int(x.size(0));
    TORCH_CHECK(a.T <= gt.TCAP, "T exceeds TCAP");
    a.H = gt.H, a.I = gt.I, a.E = gt.E, a.TOPK = gt.TOPK, a.nsm = gt.nsm;
    a.swiglu_limit = float(swiglu_limit);
    sm120_moe_tp_kernel<<<gt.grid, kWarps * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
    CHECK_CUDA(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_tp", &init_tp);
    m.def("run_tp", &run_tp);
    m.def("tp_grid_size", &tp_grid_size);
    m.def("tp_error", &tp_error);
    m.def("init", &init);
    m.def("open_peers", &open_peers);
    m.def("grid_size", &grid_size);
    m.def("counters", &counters);
    m.def("run", &run);
}
