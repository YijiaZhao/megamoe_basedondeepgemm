/**
 * SM120 decode MegaMoE, W4A4 (FP4 weights x FP4 activations) and W4A8 (MXFP4 weights x MXFP8 activations), EP over
 * PCIe P2P, one persistent kernel per rank.
 * Dataflow: dispatch (FP4 quant + P2P push) -> FC1 +
 * SwiGLU (swap-AB mma.sync) -> FC2 -> P2P push back -> top-k combine, epoch-tagged flags, no cross-rank barriers.
 * One format per build (SM120_FP4_NV):
 *   0 MXFP4: E2M1, UE8M0 scale per 32 values; mma.sync kind::mxf4nvf4 scale_vec::2X m16n8k64.
 *   1 NVFP4: E2M1, UE4M3 scale per 16 values and fp32 global scales (per expert for the weights, per tensor for the
 *            FC1 / FC2 inputs, all dequant factors); mma.sync kind::mxf4nvf4 scale_vec::4X m16n8k64.
 * One MMA covers K64 and the expert inputs are half the bytes of W4A8.
 * SM120_FP4_A8 = 1 builds W4A8 instead: activations E4M3 with a UE8M0 scale per 32, two mma.sync kind::mxf8f6f4
 * m16n8k32 per K64, weights in the W4A8 kernel's layout (`sm120.py` transform; lane word pairs {x, y} / {z, w} = K32
 * steps, a byte = row g's nibble | row g+8's << 4).
 *
 * Weight layout (offline): per expert [rows/16 blocks][K/64][32 lanes] uint4; lane (g, c) holds {row g, k c*8..c*8+7},
 * {row g+8, same k}, {row g, k 32+c*8..}, {row g+8, same k}, 8 E2M1 per word, low nibble first (the MMA A fragment).
 * W13 block b = gate rows 8b..8b+7 then up rows 8b..8b+7 (SwiGLU pairs land in one thread: acc rows g and g+8).
 * Scales [blocks][K/kSfWordK][16 rows] u32, byte s = scale of the s-th K group of the word (4 groups per word).
 * SM120_FP4_RM = 1 (W4A8 only): the weights stay in the checkpoint's row-major layout ([rows][K/2] bytes, W13 gate
 * rows then up rows, the layout FlashInfer's SM120 CUTLASS MoE uses), so a layer can share them with that backend;
 * the item GEMM loads 16 contiguous bytes of a row per lane and K128 and transposes them across the quad
 * (item_gemm_rm). The scales keep the layout above.
 * Every spin wait times out after kSpinNs and reports through the error word instead of hanging the GPU.
 *
 * NIC transport (SM120_FP4_GIN=1, NCCL Device API): the buffer is one NCCL symmetric window (the same layout on every
 * rank) and the ranks in Gin::mask (e.g. on another host) are reached with GIN puts through the NIC instead of P2P
 * stores. A payload is staged in this rank's window first, then put to the peer's receive region at the offset a P2P
 * store would use, a token row (values, scales, metadata) in one put. The partials of a finished FC2 tile go back as
 * one put per GIN source, its rows packed in (local expert, slot) order. Each put adds its rows (1 for the routing
 * counts) to a signal of the receiver, one per source and kind; the receiver knows how many rows to expect and waits
 * for the cumulative signal instead of a flag.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

#ifndef SM120_FP4_GIN
#define SM120_FP4_GIN 0
#endif
// GIN code is device-pass only: NCCL's device headers do not compile in nvcc's host pass
#if SM120_FP4_GIN && defined(__CUDA_ARCH__)
#define SM120_FP4_GIN_DEVICE 1
#include <nccl_device.h>
#else
#define SM120_FP4_GIN_DEVICE 0
#endif

#define CHECK_CUDA(x)                                                                      \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        TORCH_CHECK(e_ == cudaSuccess, "CUDA error: ", cudaGetErrorString(e_), " @ ", #x); \
    } while (0)

#ifndef SM120_FP4_NV
#define SM120_FP4_NV 0
#endif
#ifndef SM120_FP4_A8
#define SM120_FP4_A8 0
#endif
#ifndef SM120_FP4_RM
#define SM120_FP4_RM 0
#endif
constexpr bool kNv = SM120_FP4_NV;
constexpr bool kA8 = SM120_FP4_A8;
constexpr bool kRm = SM120_FP4_RM;
static_assert(!(kNv && kA8));
static_assert(!kRm || kA8, "row-major weights: W4A8 only");
constexpr int kXB = kA8 ? 1 : 2;       // activation values per byte
constexpr int kSfVec = kNv ? 16 : 32;  // values per block scale
constexpr int kSfWordK = 4 * kSfVec;   // K covered by one u32 of scales
constexpr int kQW = kSfVec / (4 * kXB);  // u32 per quantized scale group

constexpr int kWarps = 8;
constexpr int kNbPerWarp = 1;
constexpr int kRowsPerItem = kWarps * kNbPerWarp * 16;  // 128 weight rows per work item
constexpr int kTile = 8;                                // tokens per MMA (N); an item takes NT of them
constexpr int kMaxNt = 2;
constexpr size_t kPf0Bytes = size_t(64) << 20;          // t=0 L2 prefetch budget over the locally known experts
constexpr int kUnroll = 4;
constexpr int kMaxWorld = 16;
constexpr int kMaxEl = 256;
constexpr uint64_t kSpinNs = 30000000000ull;

constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4;
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4, kCtrEpoch = 5;
constexpr int kCtrFlag = 8, kCtrL1 = 16;  // kCtrFlag: local copies (epoch) of the route / rows / combine flags
// GIN: FC2 items done per tile (reset per call), cumulative signal totals received per kind and source (u64, kept)
constexpr int kMaxTiles = 2304;  // sum over local experts of ceil(rows / 8): EL + W * TCAP * TOPK / 8, TCAP <= 64
constexpr int kCtrTile = kCtrL1 + kMaxEl, kCtrCum = kCtrTile + kMaxTiles;
constexpr int kCtrInts = kCtrCum + 2 * 3 * kMaxWorld;
static_assert(kCtrCum % 2 == 0, "the cumulative GIN totals must be 8-byte aligned");
constexpr int kSigRoute = 0, kSigRows = kMaxWorld, kSigComb = 2 * kMaxWorld;  // GIN signal ids, + source rank

// a pool row: H/kXB E2M1 / E4M3 values, H/kSfVec scale bytes, then 16 bytes holding the int source token*8 + k
__host__ __device__ __forceinline__ int pool_rx(int H) { return H / kXB + H / kSfVec + 16; }

struct Peers {
    uint8_t *pool[kMaxWorld];       // [EL][W][TCAP] pool rows                    (dest-owned)
    int *recv_flag[kMaxWorld];      // [W] epoch
    __nv_bfloat16 *comb[kMaxWorld]; // [TCAP][TOPK][H]                    (src-owned)
    int *comb_flag[kMaxWorld];      // [W] epoch
    int *route[kMaxWorld];          // [W][EL]: tokens each source rank sends to each expert (dest-owned)
    int *route_flag[kMaxWorld];     // [W] epoch
};

// NIC transport: window offsets of the receive regions (the P2P layout) and of this rank's staging regions, and the
// staging pointers. mask 0 (always without SM120_FP4_GIN): every peer over P2P.
struct Gin {
    const void *comm;  // ncclDevComm const * (device memory)
    void *win;         // ncclWindow_t of the whole buffer
    uint32_t mask;     // bit w: rank w is reached with GIN puts
    int nctx;          // GIN contexts: CTA b posts on context b % nctx; a signal is the sum over them
    size_t off_pool, off_route, off_rcomb, st_route, st_row, st_comb;
    int *s_route;           // [W][EL] this rank's token counts for each GIN peer's experts
    uint8_t *s_row;         // [TCAP][TOPK] pool rows of the copies to GIN peers
    __nv_bfloat16 *s_comb;  // [W][TCAP*TOPK][H] partials for each GIN source, packed (local expert, slot)
    __nv_bfloat16 *r_comb;  // [W][TCAP*TOPK][H] partials from each GIN expert rank, packed the same way
    int *cidx;              // [TCAP][TOPK] packed index at the GIN expert rank of copy (t, k)
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
    const float *gscale;      // NVFP4: [0] FC1 input, [1] FC2 input, [2, 2+EL) W13, [2+EL, 2+2EL) W2; MXFP4: unused
    uint8_t *inter;           // [EL][W*TCAP][I/kXB] E2M1 / E4M3 (local)
    uint32_t *inter_s;        // [EL][W*TCAP][I/kSfWordK]
    int *ctr;                 // kCtrInts
    unsigned long long *stamps;  // optional [grid][16] globaltimer stamps (debug)
    Peers peers;
    Gin gin;
    int T, H, I, EL, W, TOPK, TCAP, rank;
    float swiglu_limit;
    int w13_up_first;  // SM120_FP4_RM: W13 holds the up rows first (FlashInfer CUTLASS order)
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
// flag store; every caller has fenced its data with __threadfence_system first (a release store costs ~1 us more)
__device__ __forceinline__ void st_release_sys(int *p, int v) { *reinterpret_cast<volatile int *>(p) = v; }
__device__ __forceinline__ bool wait_ge(const int *p, int v, bool sys) {
    const uint64_t t0 = gtime();
    while ((sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) < v)
        if (gtime() - t0 > kSpinNs) return false;
    return true;
}

__device__ __forceinline__ unsigned long long *gin_cum(int *ctr) {  // [3 kinds][kMaxWorld sources]
    return reinterpret_cast<unsigned long long *>(ctr + kCtrCum);
}
#if SM120_FP4_GIN_DEVICE
__device__ __forceinline__ bool is_gin(const Args &a, int w) { return (a.gin.mask >> w) & 1u; }
// a CTA posts every put on one context; the CTAs spread over the contexts' queue pairs
__device__ __forceinline__ void gin_put(const Args &a, int peer, size_t dst_off, size_t src_off, size_t bytes, int sig,
                                        int add) {
    const ncclDevComm &c = *static_cast<const ncclDevComm *>(a.gin.comm);
    const ncclWindow_t win = static_cast<ncclWindow_t>(a.gin.win);
    ncclGin(c, int(blockIdx.x) % a.gin.nctx)
        .put(ncclTeamWorld(c), peer, win, dst_off, win, src_off, bytes,
             ncclGin_WeakSignalAdd{ncclGinSignal_t(sig), uint64_t(add)});
}
// thread spins until GIN signal sig, summed over the contexts, reaches target (64-bit rolling compare)
__device__ __forceinline__ bool gin_wait(const Args &a, int sig, unsigned long long target) {
    const ncclDevComm &c = *static_cast<const ncclDevComm *>(a.gin.comm);
    const uint64_t t0 = gtime();
    while (true) {
        unsigned long long v = 0;
        for (int x = 0; x < a.gin.nctx; ++x) v += ncclGin(c, x).readSignal(static_cast<ncclGinSignal_t>(sig));
        if (static_cast<long long>(v - target) >= 0) return true;
        if (gtime() - t0 > kSpinNs) return false;
    }
}
#else
__device__ __forceinline__ constexpr bool is_gin(const Args &, int) { return false; }
#endif

// Wait for every peer's flag (written over PCIe), or for a GIN peer's signal to reach target(w). One CTA polls them,
// a thread per peer, and republishes on a local word that the other CTAs wait on: with every CTA polling, the loads
// delay the flags' arrival by microseconds.
template <class Target>
__device__ __forceinline__ bool wait_peers(const Args &a, const int *flags, int W, int epoch, int *local, bool poller,
                                           int sig, Target target) {
    if (!poller) return threadIdx.x != 0 || wait_ge(local, epoch, false);
    if (threadIdx.x >= 32) return true;
    const int w = threadIdx.x;
    bool ok = true;
    if (w < W) {
#if SM120_FP4_GIN_DEVICE
        if (is_gin(a, w))
            ok = gin_wait(a, sig + w, target(w));
        else
#endif
            ok = wait_ge(flags + w, epoch, true);
    }
    if (__all_sync(0xffffffffu, ok) && threadIdx.x == 0) atomicExch(local, epoch);
    return ok;
}

// D[16 x 8] += A[16 x 64] B[64 x 8], E2M1 x E2M1 with block scales (A row g's from lane c even, g+8's from c odd,
// B column g's from every lane of the quad; byte s = K group s)
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint4 &a, uint32_t b0, uint32_t b1, uint32_t sfa,
                                        uint32_t sfb) {
#if SM120_FP4_NV
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
#else
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
#endif
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(sfa), "h"(uint16_t{0}), "h"(uint16_t{0}),
          "r"(sfb), "h"(uint16_t{0}), "h"(uint16_t{0}));
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
__device__ __forceinline__ void expand(uint32_t w, uint32_t &lo, uint32_t &hi) {  // packed E2M1 -> one per byte
    lo = (w & 0x0F0F0F0Fu) << 2;
    hi = (w >> 2) & 0x3C3C3C3Cu;
}

// warp GEMM over one 128-row item and NT n8 token sub-tiles: acc[j][s] = rows (nb0+j)*16 + {g, g+8}, cols = tokens
// s*8 + 2c, 2c+1. Each weight fragment is loaded once (streaming load) and feeds NT MMAs.
// xrow[s]: this lane's token row of sub-tile s (K/kXB bytes, as words), xsrow[s]: its scale words (K/kSfWordK).
template <int NT>
__device__ __forceinline__ void item_gemm(float (&acc)[kNbPerWarp][NT][4], const uint4 *W, const uint32_t *WS, int nb0,
                                          int K, const uint32_t *const (&xrow)[NT], const uint32_t *const (&xsrow)[NT],
                                          int lane) {
    const int g = lane / 4, c = lane % 4;
    const int kch = K / 64, kw = K / kSfWordK;
    const uint4 *wp[kNbPerWarp];
    const uint32_t *sp[kNbPerWarp];
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j) {
        wp[j] = W + size_t(nb0 + j) * kch * 32 + lane;
        sp[j] = WS + size_t(nb0 + j) * kw * 16 + g + (lane & 1) * 8;
    }
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
        for (int s = 0; s < NT; ++s) acc[j][s][0] = acc[j][s][1] = acc[j][s][2] = acc[j][s][3] = 0.f;
    constexpr int kSw = kNv ? kUnroll : kUnroll / 2;  // scale words per unrolled step
    // cold L2: the loop waits on DRAM. With one MMA per weight fragment (NT 1) the next step's weights and scales are
    // loaded explicitly while this step's MMAs run; with NT MMAs per fragment two unrolled steps cover the latency,
    // and a second buffer only costs registers
    constexpr bool kDb = NT == 1;
    uint4 wv[kNbPerWarp][kUnroll], wn[kNbPerWarp][kUnroll];
    uint32_t sfa[kNbPerWarp][kSw], sfn[kNbPerWarp][kSw];
    auto load = [&](int k0, uint4 (&w)[kNbPerWarp][kUnroll], uint32_t (&sf)[kNbPerWarp][kSw]) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) w[j][u] = __ldcs(wp[j] + (k0 + u) * 32);
#pragma unroll
        for (int q = 0; q < kSw; ++q)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) sf[j][q] = __ldg(sp[j] + (k0 * kSw / kUnroll + q) * 16);
    };
    if constexpr (kDb) load(0, wv, sfa);
#pragma unroll 2
    for (int k0 = 0; k0 < kch; k0 += kUnroll) {
        if constexpr (kDb) {
            if (k0 + kUnroll < kch) load(k0 + kUnroll, wn, sfn);
        } else {
            load(k0, wv, sfa);
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int kc = k0 + u;
            if constexpr (kA8) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // the two K32 steps of the fragment
                    const int step = kc * 2 + h;
                    const uint16_t sel = uint16_t(step & 3);
                    uint32_t b0[NT], b1[NT], sfb[NT];
#pragma unroll
                    for (int s = 0; s < NT; ++s) {
                        b0[s] = xrow[s][step * 8 + c];
                        b1[s] = xrow[s][step * 8 + 4 + c];
                        sfb[s] = xsrow[s][kc / 2];
                    }
#pragma unroll
                    for (int j = 0; j < kNbPerWarp; ++j) {
                        uint32_t wa[4];
                        expand(h ? wv[j][u].z : wv[j][u].x, wa[0], wa[1]);
                        expand(h ? wv[j][u].w : wv[j][u].y, wa[2], wa[3]);
#pragma unroll
                        for (int s = 0; s < NT; ++s)
                            mma_w4a8(acc[j][s], wa, b0[s], b1[s], sfa[j][u / 2], sel, sfb[s], sel);
                    }
                }
                continue;
            }
            uint32_t b0[NT], b1[NT], sfb[NT];
#pragma unroll
            for (int s = 0; s < NT; ++s) {
                b0[s] = xrow[s][kc * 8 + c];
                b1[s] = xrow[s][kc * 8 + 4 + c];
                sfb[s] = kNv ? xsrow[s][kc] : xsrow[s][kc / 2] >> ((kc & 1) * 16);
            }
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) {
                const uint32_t sa = kNv ? sfa[j][u] : sfa[j][u / 2] >> ((u & 1) * 16);
#pragma unroll
                for (int s = 0; s < NT; ++s) mma_fp4(acc[j][s], wv[j][u], b0[s], b1[s], sa, sfb[s]);
            }
        }
        if constexpr (kDb) {
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) wv[j][u] = wn[j][u];
#pragma unroll
                for (int q = 0; q < kSw; ++q) sfa[j][q] = sfn[j][q];
            }
        }
    }
}

#if SM120_FP4_RM
// 4x4 word transpose across the lanes of a quad: v[q] of lane c becomes v[c] of lane q
__device__ __forceinline__ void quad_transpose(uint32_t (&v)[4], int c) {
#pragma unroll
    for (int m = 2; m >= 1; m >>= 1) {
        const bool hi = (c & m) != 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q & m) continue;
            const uint32_t r = __shfl_xor_sync(0xffffffffu, hi ? v[q] : v[q | m], m);
            if (hi)
                v[q] = r;
            else
                v[q | m] = r;
        }
    }
}

// item_gemm on row-major weights (W4A8, SM120_FP4_RM): W [rows][K/2] bytes, low nibble first. Row g of block nb is
// row nb * rpb + g, row g + 8 is that + roff (W13: rpb 8, roff I for gate rows then up rows, or W at the gate rows
// and roff -I for up rows first; W2: rpb 16, roff 8).
// Within each 32-wide scale block the K order is permuted for both operands (MMA k = 16p + 4c + i reads element
// 8c + 2i + p): lane c loads 16 contiguous bytes (block c of a K128) of rows g and g + 8, the quad transposes, and
// lane c holds bytes 4c..4c+3 of every block, whose even / odd nibbles are its A registers of k 4c.. / 16+4c..
// (`expand`, as for the packed layout); its activation bytes 8c..8c+7 split the same way (byte_perm).
template <int NT>
__device__ __forceinline__ void item_gemm_rm(float (&acc)[kNbPerWarp][NT][4], const uint8_t *W, const uint32_t *WS,
                                             int nb0, int rpb, int roff, int K, const uint32_t *const (&xrow)[NT],
                                             const uint32_t *const (&xsrow)[NT], int lane) {
    const int g = lane / 4, c = lane % 4;
    const int kch = K / 64, kw = K / kSfWordK;
    const uint4 *wp[kNbPerWarp][2];
    const uint32_t *sp[kNbPerWarp];
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r)
            wp[j][r] = reinterpret_cast<const uint4 *>(W + ptrdiff_t((nb0 + j) * rpb + g + r * roff) * (K / 2)) + c;
        sp[j] = WS + size_t(nb0 + j) * kw * 16 + g + (lane & 1) * 8;
    }
#pragma unroll
    for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
        for (int s = 0; s < NT; ++s) acc[j][s][0] = acc[j][s][1] = acc[j][s][2] = acc[j][s][3] = 0.f;
    constexpr int kSw = kUnroll / 2;  // scale words per unrolled step
    constexpr int kQ = kUnroll / 2;   // K128 chunks per unrolled step
    // explicit second buffer for every NT: the quad shuffles are warp-synchronous, so the compiler does not hoist the
    // next step's loads above them (with NT 2 and an unrolled loop instead, 16-token items were ~13% slower)
    constexpr bool kDb = true;
    uint4 wq[kNbPerWarp][kQ][2], wn[kNbPerWarp][kQ][2];
    uint32_t sfa[kNbPerWarp][kSw], sfn[kNbPerWarp][kSw];
    auto load = [&](int k0, uint4 (&q)[kNbPerWarp][kQ][2], uint32_t (&sf)[kNbPerWarp][kSw]) {
#pragma unroll
        for (int p = 0; p < kQ; ++p)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
                for (int r = 0; r < 2; ++r) q[j][p][r] = __ldcs(wp[j][r] + (k0 / 2 + p) * 4);
#pragma unroll
        for (int s = 0; s < kSw; ++s)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) sf[j][s] = __ldg(sp[j] + (k0 / 2 + s) * 16);
    };
    if constexpr (kDb) load(0, wq, sfa);
#pragma unroll 2
    for (int k0 = 0; k0 < kch; k0 += kUnroll) {
        if constexpr (kDb) {
            if (k0 + kUnroll < kch) load(k0 + kUnroll, wn, sfn);
        } else {
            load(k0, wq, sfa);
        }
        uint32_t wv[kNbPerWarp][kUnroll][2][2];  // [block][K64 chunk][K32 step][row g / g + 8]
#pragma unroll
        for (int p = 0; p < kQ; ++p)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    uint32_t v[4] = {wq[j][p][r].x, wq[j][p][r].y, wq[j][p][r].z, wq[j][p][r].w};
                    quad_transpose(v, c);  // v[b]: bytes 4c..4c+3 of 32-block b of this K128
#pragma unroll
                    for (int b = 0; b < 4; ++b) wv[j][p * 2 + b / 2][b % 2][r] = v[b];
                }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int kc = k0 + u;
            uint32_t sfb[NT];
#pragma unroll
            for (int s = 0; s < NT; ++s) sfb[s] = xsrow[s][kc / 2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int step = kc * 2 + h;
                const uint16_t sel = uint16_t(step & 3);
                uint32_t b0[NT], b1[NT];  // activation elements 8c + {0, 2, 4, 6} / 8c + {1, 3, 5, 7}
#pragma unroll
                for (int s = 0; s < NT; ++s) {
                    const uint2 x = reinterpret_cast<const uint2 *>(xrow[s])[step * 4 + c];
                    b0[s] = __byte_perm(x.x, x.y, 0x6420u);
                    b1[s] = __byte_perm(x.x, x.y, 0x7531u);
                }
#pragma unroll
                for (int j = 0; j < kNbPerWarp; ++j) {
                    uint32_t wa[4];  // even nibbles (elements 8c + 2i) of rows g / g + 8, then the odd ones
                    expand(wv[j][u][h][0], wa[0], wa[2]);
                    expand(wv[j][u][h][1], wa[1], wa[3]);
#pragma unroll
                    for (int s = 0; s < NT; ++s) mma_w4a8(acc[j][s], wa, b0[s], b1[s], sfa[j][u / 2], sel, sfb[s], sel);
                }
            }
        }
        if constexpr (kDb) {
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
                for (int p = 0; p < kQ; ++p) wq[j][p][0] = wn[j][p][0], wq[j][p][1] = wn[j][p][1];
#pragma unroll
                for (int s = 0; s < kSw; ++s) sfa[j][s] = sfn[j][s];
            }
        }
    }
}
#endif

// 8 fp32 -> 8 E2M1 (round to nearest even, saturating), v[0] in the low nibble
__device__ __forceinline__ uint32_t cvt_e2m1x8(const float *v) {
    uint32_t r;
    asm volatile(
        "{\n.reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n}"
        : "=r"(r)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return r;
}

// quantize kSfVec values with one block scale; returns the scale byte.
// MXFP4: 2^ceil(log2(amax/6)). NVFP4: E4M3(amax/6 / g), `ginv` = 1/g (g: the tensor's global dequant scale).
// W4A8: E4M3 values, scale 2^ceil(log2(amax/448)).
__device__ __forceinline__ uint8_t quant_fp4(float (&v)[kSfVec], uint32_t (&q)[kQW], float ginv) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < kSfVec; ++i) amax = fmaxf(amax, fabsf(v[i]));
    float inv;
    uint8_t sc;
    if constexpr (kA8) {
        const int bits = __float_as_int(__fdiv_rn(amax, 448.f));
        int e = amax > 0.f ? ((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0) : -127;
        e = max(-126, min(127, e));
        inv = __int_as_float((127 - e) << 23);
#pragma unroll
        for (int w = 0; w < kQW; ++w) {
            uint32_t word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b)
                word |= uint32_t(__nv_cvt_float_to_fp8(v[w * 4 + b] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * b);
            q[w] = word;
        }
        return uint8_t(e + 127);
    }
    if constexpr (kNv) {
        const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(__fdiv_rn(amax, 6.f) * ginv, __NV_SATFINITE, __NV_E4M3);
        const float s = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
        inv = s > 0.f ? __fdiv_rn(ginv, s) : 0.f;  // IEEE divisions: fast-math ones move rounding boundaries
        sc = s8;
    } else {  // exact ceil(log2(amax / 6)) from the bits (fast-math log2f is off by one at powers of two)
        const int bits = __float_as_int(__fdiv_rn(amax, 6.f));
        int e = amax > 0.f ? ((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0) : -127;
        e = max(-126, min(127, e));
        inv = __int_as_float((127 - e) << 23);
        sc = uint8_t(e + 127);
    }
#pragma unroll
    for (int i = 0; i < kSfVec; ++i) v[i] *= inv;
#pragma unroll
    for (int w = 0; w < kSfVec / 8; ++w) q[w] = cvt_e2m1x8(v + 8 * w);
    return sc;
}

__device__ __forceinline__ void store_q(uint8_t *dst, const uint32_t (&q)[kQW]) {  // kSfVec / kXB bytes
    if constexpr (kNv) {
        *reinterpret_cast<uint2 *>(dst) = make_uint2(q[0], q[1]);
    } else {
#pragma unroll
        for (int w = 0; w < kQW / 4; ++w)
            reinterpret_cast<uint4 *>(dst)[w] = make_uint4(q[4 * w], q[4 * w + 1], q[4 * w + 2], q[4 * w + 3]);
    }
}

struct Map {  // per-expert token bookkeeping, built identically by every CTA after the routing exchange
    int cnt[kMaxWorld][kMaxEl];
    int ne[kMaxEl];
    int tiles_prefix[kMaxEl + 1];
    int lag;  // queue: FC2 of tile t follows FC1 of tile t + lag
};

__device__ __forceinline__ void token_src(const Map &m, int el, int j, int W, int &w, int &i) {
    w = 0;
    i = j;
    while (w < W - 1 && i >= m.cnt[w][el]) {
        i -= m.cnt[w][el];
        ++w;
    }
}

// Queue order over the tiles (a tile = one expert's kT tokens; nblk1 FC1 items, nblk2 FC2 items): FC1 of the first
// `lag` tiles, then FC1 of tile t + lag followed by FC2 of tile t, then the last FC2s. The partials leave over PCIe
// during the whole weight stream instead of after the last FC1 (the link, not DRAM, bounds a late FC2 burst), and FC1
// of a tile has finished when its FC2 is claimed.
__device__ __forceinline__ void item_decode(int item, int ntiles, int nblk1, int nblk2, int lag, bool &l1, int &tile_g,
                                            int &rb) {
    const int head = lag * nblk1, per = nblk1 + nblk2, mid = (ntiles - lag) * per;
    if (item < head) {
        l1 = true, tile_g = item / nblk1, rb = item % nblk1;
    } else if (item < head + mid) {
        const int t = (item - head) / per, r = (item - head) % per;
        l1 = r < nblk1;
        tile_g = l1 ? t + lag : t;
        rb = l1 ? r : r - nblk1;
    } else {
        const int j = item - head - mid;
        l1 = false, tile_g = ntiles - lag + j / nblk2, rb = j % nblk2;
    }
}

// A CTA's first item is static. The first CTAs dispatch this rank's tokens (and exchange the routing) and reach the
// queue last, so the head of the queue goes to the last CTAs.
__device__ __forceinline__ int first_item(unsigned block, unsigned grid) { return int(grid - 1 - block); }

template <int NT>  // tokens per item: NT x 8
// two CTAs per SM (the grid is sized from the occupancy): the register budget is 128, which the unrolled weight loop
// uses to keep loads in flight
__global__ void __launch_bounds__(kWarps * 32, 2) sm120_fp4_mega_moe_kernel(Args a) {
    constexpr int kT = kTile * NT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
    __shared__ Map m;
    __shared__ int s_item, s_ok, s_epoch, s_last;
    // L1: [tok][64 inter] / L2: [tok][128 hidden] fp32 / dispatch: one FP4 token row (H/2 + H/kSfVec bytes)
    __shared__ __align__(16) float s_tile[kT][kRowsPerItem];

    unsigned long long st_[16] = {gtime()};  // 0 start 1 pushed 2 flags 3 work end 4 comb flags 5 end 6 L1-wait ns 7 items 8 route-push 9 route-ok 10 map 11 pf 12 flag-sent 13 last FC1 14 first FC2 15 comb release
    if (threadIdx.x == 0) {
        s_epoch = *reinterpret_cast<volatile int *>(a.ctr + kCtrEpoch) + 1;
        s_ok = 1;  // cleared by a timed-out wait
    }
    __syncthreads();
    const int epoch = s_epoch;
    int *ctr = a.ctr;
    const float ginv1 = kNv ? __frcp_rn(a.gscale[0]) : 1.f, ginv2 = kNv ? __frcp_rn(a.gscale[1]) : 1.f;

    // ---------------- routing exchange first (tiny): every rank learns its experts' load before the rows land. A
    // rank sends each peer the number of its tokens per expert of that peer (the rows of one source are in token order)
    const int ndisp = min(a.T + 1, int(gridDim.x));  // CTAs busy with the exchange and the dispatch
    if (blockIdx.x == ndisp - 1) {
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = 0;
        __syncthreads();
        for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
            const int e = a.topk_ids[u];
            if (e >= 0) atomicAdd(&m.cnt[e / EL][e % EL], 1);
        }
        __syncthreads();
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x) {
            const int d = u / EL;
            *(is_gin(a, d) ? a.gin.s_route + u : a.peers.route[d] + a.rank * EL + u % EL) = m.cnt[d][u % EL];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x < W) {  // a thread per peer
            const int d = threadIdx.x;
#if SM120_FP4_GIN_DEVICE
            if (is_gin(a, d))
                gin_put(a, d, a.gin.off_route + size_t(a.rank) * EL * 4, a.gin.st_route + size_t(d) * EL * 4, EL * 4,
                        kSigRoute + a.rank, 1);
            else
#endif
                st_release_sys(a.peers.route_flag[d] + a.rank, epoch);
        }
        st_[8] = gtime();
    }

    // ---------------- t=0: pull the locally known experts' FC1 weights into L2 (budget kPf0Bytes), on the CTAs that
    // do not dispatch: the rows and their flags must not wait for it
    const int nfree = int(gridDim.x) - ndisp;
    if (kPf0Bytes > 0 && EL <= 64 && int(blockIdx.x) >= ndisp) {
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
        if (nk > 0) {  // whole experts in ascending order: the order the FC1 items consume them
            const size_t per = size_t(2 * I) * H / 2;
            const size_t lines = min(kPf0Bytes, per * nk) / 128;
            const size_t first = size_t(blockIdx.x - ndisp) * blockDim.x + threadIdx.x;
            for (size_t l = first; l < lines; l += size_t(nfree) * blockDim.x) {
                const int ki = int(l / (per / 128));
                unsigned long long rest = known;
                for (int q = 0; q < ki; ++q) rest &= rest - 1;
                const int el = __ffsll(rest) - 1;
                const char *p = reinterpret_cast<const char *>(a.w13) + el * per + (l % (per / 128)) * 128;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
            }
        }
    }

    // ---------------- dispatch: one CTA per token: quantize once in smem, push TOPK copies with every thread. The
    // token goes in chunks of kDispGroups scale groups (values, then scale bytes), so any H fits s_tile
    {
        constexpr int kDispGroups = (kTile * kRowsPerItem * 4 / (kSfVec / kXB + 1)) / 16 * 16;
        uint8_t *s_q = reinterpret_cast<uint8_t *>(&s_tile[0][0]);  // [chunk values], then [chunk scale bytes]
        uint8_t *s_sc = s_q + kDispGroups * kSfVec / kXB;
        __shared__ int s_rowk[8], s_dstk[8];
        for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
            if (warp < TOPK) {  // warp k: destination row of copy k (slot = tokens t' < t routed to the same expert)
                const int e = a.topk_ids[t * TOPK + warp];  // < 0: padding token, nothing to send
                int slot = 0;
                for (int u = lane; u < t * TOPK; u += 32) slot += (a.topk_ids[u] == e);
#pragma unroll
                for (int o = 16; o; o >>= 1) slot += __shfl_xor_sync(0xffffffffu, slot, o);
#if SM120_FP4_GIN_DEVICE
                if (e >= 0 && is_gin(a, e / EL)) {  // packed index of this copy's partial: routes to lower experts there
                    int pre = 0;
                    for (int u = lane; u < a.T * TOPK; u += 32) {
                        const int e2 = a.topk_ids[u];
                        pre += e2 >= 0 && e2 / EL == e / EL && e2 % EL < e % EL;
                    }
#pragma unroll
                    for (int o = 16; o; o >>= 1) pre += __shfl_xor_sync(0xffffffffu, pre, o);
                    if (lane == 0) a.gin.cidx[t * TOPK + warp] = pre + slot;
                }
#endif
                if (lane == 0) {
                    if (e < 0) {
                        s_rowk[warp] = -1;
                    } else {
                        const int dst = e / EL, el = e % EL;
                        const int row = (el * W + a.rank) * TCAP + slot;
                        s_rowk[warp] = row;
                        s_dstk[warp] = dst;
                        uint8_t *rec = is_gin(a, dst) ? a.gin.s_row + size_t(t * TOPK + warp) * pool_rx(H)
                                                      : a.peers.pool[dst] + size_t(row) * pool_rx(H);
                        *reinterpret_cast<int *>(rec + H / kXB + H / kSfVec) = t * 8 + warp;
                    }
                }
            }
            for (int g0 = 0; g0 < H / kSfVec; g0 += kDispGroups) {
                const int ng = min(kDispGroups, H / kSfVec - g0);
                for (int gi = threadIdx.x; gi < ng; gi += blockDim.x) {  // one thread per scale group
                    const uint4 *s4 = reinterpret_cast<const uint4 *>(a.x + size_t(t) * H + (g0 + gi) * kSfVec);
                    float v[kSfVec];
#pragma unroll
                    for (int q = 0; q < kSfVec / 8; ++q) {
                        const uint4 u = s4[q];
                        const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            const float2 f = __bfloat1622float2(b2[r]);
                            v[q * 8 + r * 2] = f.x;
                            v[q * 8 + r * 2 + 1] = f.y;
                        }
                    }
                    uint32_t q4[kQW];
                    s_sc[gi] = quant_fp4(v, q4, ginv1);
                    store_q(s_q + gi * (kSfVec / kXB), q4);
                }
                __syncthreads();
                const int nq = ng * kSfVec / kXB / 16, per = nq + ng / 16;  // uint4 per copy: data + scales
                const int oq = g0 * kSfVec / kXB / 16, os = g0 / 16;      // the chunk's uint4 offsets in the row
                for (int u = threadIdx.x; u < TOPK * per; u += blockDim.x) {
                    const int k = u / per, q = u % per;
                    if (s_rowk[k] < 0) continue;
                    // a GIN copy goes to its staging row, put in one piece below
                    uint8_t *dx = is_gin(a, s_dstk[k]) ? a.gin.s_row + size_t(t * TOPK + k) * pool_rx(H)
                                                       : a.peers.pool[s_dstk[k]] + size_t(s_rowk[k]) * pool_rx(H);
                    uint8_t *ds = dx + H / kXB;
                    if (q < nq)
                        reinterpret_cast<uint4 *>(dx)[oq + q] = reinterpret_cast<const uint4 *>(s_q)[q];
                    else
                        reinterpret_cast<uint4 *>(ds)[os + q - nq] = reinterpret_cast<const uint4 *>(s_sc)[q - nq];
                }
                __syncthreads();
            }
#if SM120_FP4_GIN_DEVICE
            if (a.gin.mask) {  // each GIN copy: its whole pool row from the staging row, one put
                __threadfence_system();
                __syncthreads();
                if (lane == 0 && warp < TOPK && s_rowk[warp] >= 0 && is_gin(a, s_dstk[warp]))
                    gin_put(a, s_dstk[warp], a.gin.off_pool + size_t(s_rowk[warp]) * pool_rx(H),
                            a.gin.st_row + size_t(t * TOPK + warp) * pool_rx(H), pool_rx(H), kSigRows + a.rank, 1);
                __syncthreads();
            }
#endif
        }
    }
    __threadfence_system();
    __syncthreads();
    st_[1] = gtime();
    if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
    __syncthreads();
    if (s_item && threadIdx.x < W && !is_gin(a, threadIdx.x)) {  // last CTA out of the dispatch: rows fenced -> flag
        st_release_sys(a.peers.recv_flag[threadIdx.x] + a.rank, epoch);
        st_[12] = gtime();
    }
    __syncthreads();

    // ---------------- per-expert map from the exchanged token counts (same slot order as the dispatch)
    const bool poller = blockIdx.x == gridDim.x - 1;
    if (!wait_peers(a, a.peers.route_flag[a.rank], W, epoch, ctr + kCtrFlag, poller, kSigRoute,
                    [&](int w) { return gin_cum(ctr)[w] + 1; })) {
        atomicExch(ctr + kCtrErr, kErrRouteWait);
        s_ok = 0;
    }
    __syncthreads();
    st_[9] = gtime();
    if (!s_ok) goto exit_kernel;
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = __ldcv(a.peers.route[a.rank] + u);
    __syncthreads();
    if (warp == 0) {
        int tiles_sum = 0, tiles_max = 0;
        for (int base = 0; base < EL; base += 32) {
            const int el = base + lane;
            int n = 0;
            if (el < EL)
                for (int w = 0; w < W; ++w) n += m.cnt[w][el];
            int incl = (n + kT - 1) / kT;
            tiles_max = max(tiles_max, incl);
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
#pragma unroll
        for (int o = 16; o; o >>= 1) tiles_max = max(tiles_max, __shfl_xor_sync(0xffffffffu, tiles_max, o));
        if (lane == 0) {  // lag: one wave of CTAs (their static first items are FC1) + an expert's own later tiles
            m.tiles_prefix[0] = 0;
            m.lag = min(tiles_sum, (int(gridDim.x) + 2 * I / kRowsPerItem - 1) / (2 * I / kRowsPerItem) + tiles_max);
        }
    }
    __syncthreads();
    st_[10] = gtime();
    {  // this CTA's first item is static: pull this warp's share of its weights and scales into L2 now. The item is
        // FC1 while the queue head lasts; with few tokens the later CTAs start on FC2 items, whose weights then arrive
        // while FC1 runs. Skipped up to 8 tiles, where it only delays the rows wait
        const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
        const int ntiles = m.tiles_prefix[EL];
        const int item = first_item(blockIdx.x, gridDim.x);
        if (ntiles > 8 && item < ntiles * (nblk1 + nblk2)) {
            bool l1;
            int tile_g, rb;
            item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
            int el = 0;
            while (m.tiles_prefix[el + 1] <= tile_g) ++el;
            const int K = l1 ? H : I;
            const size_t nb = size_t(el) * (l1 ? 2 * I / 16 : H / 16) + (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
            const char *sb = reinterpret_cast<const char *>((l1 ? a.ws13 : a.ws2) + nb * (K / kSfWordK) * 16);
#if SM120_FP4_RM
            // row-major W13: the block's gate rows and up rows are two runs of 8 rows, I rows apart
            const size_t e_rows = l1 ? size_t(2 * I) : size_t(H), e0 = nb / (e_rows / 16) * e_rows;
            const size_t blk = nb % (e_rows / 16);
            for (int half = 0; half < 2; ++half) {
                const size_t row0 = l1 ? e0 + blk * 8 + half * I : e0 + blk * 16 + half * 8;
                const char *wb = reinterpret_cast<const char *>(l1 ? a.w13 : a.w2) + row0 * (K / 2);
                for (int off = lane * 128; off < kNbPerWarp * 8 * (K / 2); off += 32 * 128)
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(wb + off));
            }
#else
            const char *wb = reinterpret_cast<const char *>((l1 ? a.w13 : a.w2) + nb * (K / 64) * 32);
            for (int off = lane * 128; off < kNbPerWarp * (K / 64) * 512; off += 32 * 128)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(wb + off));
#endif
            for (int off = lane * 128; off < kNbPerWarp * (K / kSfWordK) * 64; off += 32 * 128)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(sb + off));
        }
    }
    st_[11] = gtime();

    // ---------------- wait for every source rank's rows
    if (!wait_peers(a, a.peers.recv_flag[a.rank], W, epoch, ctr + kCtrFlag + 1, poller, kSigRows, [&](int w) {
            int n = 0;  // rows from w
            for (int el = 0; el < EL; ++el) n += m.cnt[w][el];
            return gin_cum(ctr)[kMaxWorld + w] + (unsigned long long)n;
        })) {
        atomicExch(ctr + kCtrErr, kErrDispatchWait);
        s_ok = 0;
    }
    __syncthreads();
    st_[2] = gtime();
    if (!s_ok) goto exit_kernel;
    {
        const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
        const int ntiles = m.tiles_prefix[EL];
        const int n1 = ntiles * nblk1, n2 = ntiles * nblk2;
        const uint8_t *pool = a.peers.pool[a.rank];
        const int RX = pool_rx(H);
        auto meta_of = [&](size_t row) { return *reinterpret_cast<const int *>(pool + row * RX + H / kXB + H / kSfVec); };
        const int g = lane / 4, c = lane % 4;
        const size_t w13_expert = size_t(2 * I / 16) * (H / 64) * 32, ws13_expert = size_t(2 * I / 16) * (H / kSfWordK) * 16;
        const size_t w2_expert = size_t(H / 16) * (I / 64) * 32, ws2_expert = size_t(H / 16) * (I / kSfWordK) * 16;
        const int TOKE = W * TCAP;

        bool first = true;
        while (true) {
            if (threadIdx.x == 0)
                s_item = first ? first_item(blockIdx.x, gridDim.x) : int(gridDim.x) + atomicAdd(ctr + kCtrQueue, 1);
            first = false;
            __syncthreads();
            const int item = s_item;
            if (item >= n1 + n2) break;
            st_[7] += 1;
            bool l1;
            int tile_g, rb;
            item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
            int el = 0;
            while (m.tiles_prefix[el + 1] <= tile_g) ++el;
            const int tile = tile_g - m.tiles_prefix[el];
            const int ntok = min(kT, m.ne[el] - tile * kT);
            int js[NT];  // this lane's token of sub-tile s (B column g; past the end: the tile's first token)
#pragma unroll
            for (int s = 0; s < NT; ++s) js[s] = tile * kT + (s * kTile + g < ntok ? s * kTile + g : 0);

            if (!l1) {  // FC2 needs every FC1 item of this expert
                const unsigned long long tw0 = gtime();
                if (threadIdx.x == 0) {
                    const int need = ((m.ne[el] + kT - 1) / kT) * nblk1;
                    if (!wait_ge(ctr + kCtrL1 + el, need, false)) {
                        atomicExch(ctr + kCtrErr, kErrL1Wait);
                        s_ok = 0;
                    }
                }
                __syncthreads();
                st_[6] += gtime() - tw0;
                if (!st_[14]) st_[14] = gtime();
                if (!s_ok) goto exit_kernel;
            }

            float acc[kNbPerWarp][NT][4];
            const uint32_t *xr[NT], *xs[NT];
            const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
            if (l1) {
#pragma unroll
                for (int s = 0; s < NT; ++s) {
                    int w, i;
                    token_src(m, el, js[s], W, w, i);
                    const size_t row = (size_t(el) * W + w) * TCAP + i;
                    xr[s] = reinterpret_cast<const uint32_t *>(pool + row * RX);
                    xs[s] = reinterpret_cast<const uint32_t *>(pool + row * RX + H / kXB);
                }
#if SM120_FP4_RM
                // gate rows: block row g, up rows: g + 8 (I rows apart, before or after the gate rows)
                item_gemm_rm<NT>(acc, reinterpret_cast<const uint8_t *>(a.w13 + el * w13_expert) +
                                          (a.w13_up_first ? size_t(I) * (H / 2) : 0),
                                 a.ws13 + el * ws13_expert, nb0, 8, a.w13_up_first ? -I : I, H, xr, xs, lane);
#else
                item_gemm<NT>(acc, a.w13 + el * w13_expert, a.ws13 + el * ws13_expert, nb0, H, xr, xs, lane);
#endif
                const float alpha = kNv ? a.gscale[0] * a.gscale[2 + el] : 1.f;
                // SwiGLU: acc rows g (gate) / g+8 (up) of block nb -> inter col nb*8+g; tokens 2c, 2c+1
                constexpr int kItemInter = kRowsPerItem / 2;
                constexpr int kGroups = kItemInter / kSfVec;
#pragma unroll
                for (int jb = 0; jb < kNbPerWarp; ++jb) {
                    const int col = (warp * kNbPerWarp + jb) * 8 + g;
#pragma unroll
                    for (int s = 0; s < NT; ++s)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const float gate = fminf(acc[jb][s][h] * alpha, a.swiglu_limit);
                            const float up = fminf(fmaxf(acc[jb][s][2 + h] * alpha, -a.swiglu_limit), a.swiglu_limit);
                            s_tile[s * kTile + 2 * c + h][col] = gate / (1.f + __expf(-gate)) * up;
                        }
                }
                __syncthreads();
                if (threadIdx.x < kT * kGroups) {  // one thread per (token, scale group)
                    const int tk = threadIdx.x / kGroups, gi = threadIdx.x % kGroups;
                    float v[kSfVec];
#pragma unroll
                    for (int u = 0; u < kSfVec; ++u) v[u] = s_tile[tk][gi * kSfVec + u];
                    uint32_t q4[kQW];
                    const uint8_t sc = quant_fp4(v, q4, ginv2);
                    if (tk < ntok) {
                        const size_t r = size_t(el) * TOKE + tile * kT + tk;
                        const int col0 = rb * kItemInter + gi * kSfVec;
                        store_q(a.inter + r * (I / kXB) + col0 / kXB, q4);
                        reinterpret_cast<uint8_t *>(a.inter_s + r * (I / kSfWordK))[col0 / kSfVec] = sc;
                    }
                }
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) atomicAdd(ctr + kCtrL1 + el, 1);
                st_[13] = gtime();
            } else {
#pragma unroll
                for (int s = 0; s < NT; ++s) {
                    const size_t r = size_t(el) * TOKE + js[s];
                    xr[s] = reinterpret_cast<const uint32_t *>(a.inter + r * (I / kXB));
                    xs[s] = a.inter_s + r * (I / kSfWordK);
                }
#if SM120_FP4_RM
                item_gemm_rm<NT>(acc, reinterpret_cast<const uint8_t *>(a.w2 + el * w2_expert), a.ws2 + el * ws2_expert,
                                 nb0, 16, 8, I, xr, xs, lane);
#else
                item_gemm<NT>(acc, a.w2 + el * w2_expert, a.ws2 + el * ws2_expert, nb0, I, xr, xs, lane);
#endif
                const float alpha = kNv ? a.gscale[1] * a.gscale[2 + EL + el] : 1.f;
#pragma unroll
                for (int jb = 0; jb < kNbPerWarp; ++jb) {
                    const int col = (warp * kNbPerWarp + jb) * 16 + g;
#pragma unroll
                    for (int s = 0; s < NT; ++s)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            s_tile[s * kTile + 2 * c + h][col] = acc[jb][s][h] * alpha;
                            s_tile[s * kTile + 2 * c + h][col + 8] = acc[jb][s][2 + h] * alpha;
                        }
                }
                __syncthreads();
                // push: token tk of this tile -> its source rank's combine slot, kRowsPerItem bf16 in 16 B stores; a
                // GIN source's rows go to its staging region at their packed index, put once the tile is done
                __shared__ int s_tw[kT], s_tp[kT];
                if (threadIdx.x < ntok) {
                    int w, i;
                    token_src(m, el, tile * kT + threadIdx.x, W, w, i);
                    int p = i;  // packed index at the source: its rows to lower experts of this rank first
                    if (is_gin(a, w))
                        for (int e2 = 0; e2 < el; ++e2) p += m.cnt[w][e2];
                    s_tw[threadIdx.x] = w, s_tp[threadIdx.x] = p;
                }
                __syncthreads();
                constexpr int kVec = kRowsPerItem / 8;
                for (int u = threadIdx.x; u < ntok * kVec; u += blockDim.x) {
                    const int tk = u / kVec, q = u % kVec;
                    int w, i;
                    token_src(m, el, tile * kT + tk, W, w, i);
                    const int meta = meta_of((size_t(el) * W + w) * TCAP + i);
                    const int st = meta / 8, sk = meta % 8;
                    uint32_t pk[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const __nv_bfloat162 b2 =
                            __floats2bfloat162_rn(s_tile[tk][q * 8 + 2 * p], s_tile[tk][q * 8 + 2 * p + 1]);
                        pk[p] = *reinterpret_cast<const uint32_t *>(&b2);
                    }
                    __nv_bfloat16 *dst = (is_gin(a, w) ? a.gin.s_comb + (size_t(w) * TCAP * TOPK + s_tp[tk]) * H
                                                       : a.peers.comb[w] + (size_t(st) * TOPK + sk) * H) +
                                         rb * kRowsPerItem + q * 8;
                    *reinterpret_cast<uint4 *>(dst) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
                }
                __threadfence_system();
                __syncthreads();
#if SM120_FP4_GIN_DEVICE
                if (a.gin.mask) {  // the tile's last FC2 item: each GIN source's run of rows (contiguous), one put
                    if (threadIdx.x == 0) s_last = atomicAdd(ctr + kCtrTile + tile_g, 1) == nblk2 - 1;
                    __syncthreads();
                    if (s_last && threadIdx.x < ntok && is_gin(a, s_tw[threadIdx.x]) &&
                        (threadIdx.x == 0 || s_tw[threadIdx.x - 1] != s_tw[threadIdx.x])) {
                        const int w = s_tw[threadIdx.x];
                        int n = 1;
                        while (threadIdx.x + n < ntok && s_tw[threadIdx.x + n] == w) ++n;
                        const size_t row0 = size_t(w) * TCAP * TOPK + s_tp[threadIdx.x];  // [W][TCAP*TOPK] rows
                        gin_put(a, w, a.gin.off_rcomb + (size_t(a.rank) * TCAP * TOPK + s_tp[threadIdx.x]) * H * 2,
                                a.gin.st_comb + row0 * H * 2, size_t(n) * H * 2, kSigComb + a.rank, n);
                    }
                }
#endif
                if (threadIdx.x == 0) s_item = atomicAdd(ctr + kCtrL2Done, 1) == n2 - 1;
                __syncthreads();
                if (s_item && threadIdx.x < W && !is_gin(a, threadIdx.x)) {  // last FC2 item: tell every source rank
                    st_release_sys(a.peers.comb_flag[threadIdx.x] + a.rank, epoch);
                    st_[15] = gtime();
                }
            }
            __syncthreads();
        }
        st_[3] = gtime();
        if (n2 == 0 && blockIdx.x == 0 && threadIdx.x < W && !is_gin(a, threadIdx.x))  // no work: still release
            st_release_sys(a.peers.comb_flag[threadIdx.x] + a.rank, epoch);
    }

    // ---------------- combine: wait for every expert rank, out[t] = sum_k w[t][k] * partial[t][k]
    if (!wait_peers(a, a.peers.comb_flag[a.rank], W, epoch, ctr + kCtrFlag + 2, poller, kSigComb, [&](int w) {
            int n = 0;  // this rank's copies at w: their partial rows
            for (int u = 0; u < a.T * TOPK; ++u) n += a.topk_ids[u] >= 0 && a.topk_ids[u] / EL == w;
            return gin_cum(ctr)[2 * kMaxWorld + w] + (unsigned long long)n;
        })) {
        atomicExch(ctr + kCtrErr, kErrCombineWait);
        s_ok = 0;
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
                const int d = a.topk_ids[t * TOPK + k] / EL;  // a GIN expert rank's partial: at its packed index
                const __nv_bfloat16 *src = is_gin(a, d) ? a.gin.r_comb + (size_t(d) * TCAP * TOPK + a.gin.cidx[t * TOPK + k]) * H
                                                        : comb + (size_t(t) * TOPK + k) * H;
                const uint4 u = *reinterpret_cast<const uint4 *>(src + h8);
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
#if SM120_FP4_GIN_DEVICE
            for (int w = 0; w < W; ++w)  // the GIN signal totals of this call (see the waits)
                if (is_gin(a, w)) {
                    int rows = 0, copies = 0;
                    for (int el = 0; el < EL; ++el) rows += m.cnt[w][el];
                    for (int u = 0; u < a.T * TOPK; ++u) copies += a.topk_ids[u] >= 0 && a.topk_ids[u] / EL == w;
                    gin_cum(ctr)[w] += 1;
                    gin_cum(ctr)[kMaxWorld + w] += rows;
                    gin_cum(ctr)[2 * kMaxWorld + w] += copies;
                }
            if (a.gin.mask)
                for (int u = 0; u < m.tiles_prefix[EL]; ++u) ctr[kCtrTile + u] = 0;
#endif
            ctr[kCtrDispatch] = 0;
            ctr[kCtrQueue] = 0;
            ctr[kCtrL2Done] = 0;
            for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
            __threadfence();
            ctr[kCtrExit] = 0;
        }
    }
}

// ------------------------------------------------------------------------------------------------ host
struct State {
    int rank = 0, world = 0, EL = 0, TCAP = 0, H = 0, I = 0, TOPK = 0;
    void *local_alloc = nullptr;  // every peer-visible buffer: one cudaMalloc (P2P) or the NCCL window
    size_t off_pool, off_rflag, off_comb, off_cflag, off_route, off_rtflag;
    size_t st_route, st_row, st_comb, off_rcomb, st_cidx, total;  // GIN staging and receive regions (window only)
    void *peer_base[kMaxWorld] = {};  // nullptr: a GIN peer
    uint32_t gin_mask = 0;
    const void *gin_comm = nullptr;
    void *gin_win = nullptr;
    int gin_nctx = 0;
    uint8_t *inter = nullptr;
    uint32_t *inter_s = nullptr;
    int *ctr = nullptr;
    int grid[kMaxNt + 1] = {};  // per tile width (NT)
};
static State g;

template <int NT>
static int grid_for(int sms) {
    int occ = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sm120_fp4_mega_moe_kernel<NT>, kWarps * 32, 0));
    return occ * sms;
}

static size_t align_up(size_t v) { return (v + 255) & ~size_t(255); }

// the buffer's layout (the same on every rank); with `window`, the GIN staging regions follow
static void layout(int rank, int world, int EL, int TCAP, int H, int I, int TOPK, bool window) {
    TORCH_CHECK(world <= kMaxWorld && EL <= kMaxEl && TOPK <= 8, "world / experts / top-k out of range");
    TORCH_CHECK(H % kRowsPerItem == 0 && I % kRowsPerItem == 0 && H % (16 * kSfVec) == 0,
                "unsupported hidden / intermediate size");
    g.rank = rank, g.world = world, g.EL = EL, g.TCAP = TCAP, g.H = H, g.I = I, g.TOPK = TOPK;
    const size_t rows = size_t(EL) * world * TCAP;
    size_t o = 0;
    g.off_pool = o; o = align_up(o + rows * pool_rx(H));
    g.off_rflag = o; o = align_up(o + size_t(world) * 4);
    g.off_comb = o; o = align_up(o + size_t(TCAP) * TOPK * H * 2);
    g.off_cflag = o; o = align_up(o + size_t(world) * 4);
    g.off_route = o; o = align_up(o + size_t(world) * EL * 4);
    g.off_rtflag = o; o = align_up(o + size_t(world) * 4);
    if (window) {
        TORCH_CHECK(EL + world * TCAP * TOPK / kTile <= kMaxTiles, "the NIC transport takes at most 64 tokens per rank");
        g.st_route = o; o = align_up(o + size_t(world) * EL * 4);
        g.st_row = o; o = align_up(o + size_t(TCAP) * TOPK * pool_rx(H));
        g.st_comb = o; o = align_up(o + size_t(world) * TCAP * TOPK * H * 2);
        g.off_rcomb = o; o = align_up(o + size_t(world) * TCAP * TOPK * H * 2);
        g.st_cidx = o; o = align_up(o + size_t(TCAP) * TOPK * 4);
    }
    g.total = o;
}

// the rank-local buffers and the grid sizes
static void setup_local() {
    const size_t rows = size_t(g.EL) * g.world * g.TCAP;
    CHECK_CUDA(cudaMalloc(&g.inter, rows * g.I / kXB));
    CHECK_CUDA(cudaMalloc(&g.inter_s, rows * (g.I / kSfWordK) * 4));
    CHECK_CUDA(cudaMalloc(&g.ctr, kCtrInts * 4));
    CHECK_CUDA(cudaMemset(g.ctr, 0, kCtrInts * 4));
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    const int sms = prop.multiProcessorCount;
    g.grid[1] = grid_for<1>(sms), g.grid[2] = grid_for<2>(sms);
}

// P2P transport: one cudaMalloc, shared over cudaIpc; returns this rank's handle bytes (open_peers takes everyone's)
torch::Tensor init(int rank, int world, int EL, int TCAP, int H, int I, int TOPK) {
    layout(rank, world, EL, TCAP, H, I, TOPK, false);
    CHECK_CUDA(cudaMalloc(&g.local_alloc, g.total));
    CHECK_CUDA(cudaMemset(g.local_alloc, 0, g.total));
    setup_local();
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

// window transport: bytes of the NCCL window to allocate
int64_t window_bytes(int rank, int world, int EL, int TCAP, int H, int I, int TOPK) {
    layout(rank, world, EL, TCAP, H, I, TOPK, true);
    return int64_t(g.total);
}

// window transport: `base` this rank's NCCL window (zeroed), `peers` every rank's load/store pointer to its window
// (0: reached over GIN, bit set in `mask`), `comm` the device ncclDevComm *, `win` the ncclWindow_t
void init_window(int rank, int world, int EL, int TCAP, int H, int I, int TOPK, int64_t base,
                 std::vector<int64_t> peers, int64_t mask, int64_t comm, int64_t win, int nctx) {
    layout(rank, world, EL, TCAP, H, I, TOPK, true);
    TORCH_CHECK(int(peers.size()) == world, "one peer pointer per rank");
    TORCH_CHECK(mask == 0 || (SM120_FP4_GIN && comm != 0 && nctx > 0), "GIN peers need the GIN build and a device "
                "communicator with GIN contexts");
    g.local_alloc = reinterpret_cast<void *>(base);
    for (int w = 0; w < world; ++w) {
        TORCH_CHECK(((mask >> w) & 1) == (peers[w] == 0), "rank ", w, ": a GIN peer has no pointer, a P2P peer has one");
        g.peer_base[w] = w == rank ? g.local_alloc : reinterpret_cast<void *>(peers[w]);
    }
    g.gin_mask = uint32_t(mask), g.gin_comm = reinterpret_cast<const void *>(comm);
    g.gin_win = reinterpret_cast<void *>(win), g.gin_nctx = nctx;
    setup_local();
}

int grid_size(int nt) { return g.grid[nt]; }

std::vector<int64_t> counters() {
    std::vector<int> h(kCtrInts);
    CHECK_CUDA(cudaMemcpy(h.data(), g.ctr, kCtrInts * 4, cudaMemcpyDeviceToHost));
    return std::vector<int64_t>(h.begin(), h.end());
}

void run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out, torch::Tensor w13,
         torch::Tensor ws13, torch::Tensor w2, torch::Tensor ws2, c10::optional<torch::Tensor> gscale,
         double swiglu_limit, int64_t tile_tokens, c10::optional<torch::Tensor> stamps, bool w13_up_first) {
    Args a{};
    a.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
    a.topk_ids = topk_ids.data_ptr<int>();
    a.topk_w = topk_w.data_ptr<float>();
    a.out = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    a.w13 = reinterpret_cast<const uint4 *>(w13.data_ptr());
    a.ws13 = reinterpret_cast<const uint32_t *>(ws13.data_ptr());
    a.w2 = reinterpret_cast<const uint4 *>(w2.data_ptr());
    a.ws2 = reinterpret_cast<const uint32_t *>(ws2.data_ptr());
    if (kNv) {
        TORCH_CHECK(gscale.has_value() && gscale->scalar_type() == torch::kFloat32 && gscale->numel() == 2 + 2 * g.EL,
                    "NVFP4 needs the packed global scales [2 + 2 * local experts] fp32");
        a.gscale = gscale->data_ptr<float>();
    }
    a.inter = g.inter;
    a.inter_s = g.inter_s;
    a.ctr = g.ctr;
    a.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long *>(stamps->data_ptr()) : nullptr;
    for (int w = 0; w < g.world; ++w) {
        if (!g.peer_base[w]) continue;  // GIN peer
        uint8_t *b = static_cast<uint8_t *>(g.peer_base[w]);
        a.peers.pool[w] = b + g.off_pool;
        a.peers.recv_flag[w] = reinterpret_cast<int *>(b + g.off_rflag);
        a.peers.comb[w] = reinterpret_cast<__nv_bfloat16 *>(b + g.off_comb);
        a.peers.comb_flag[w] = reinterpret_cast<int *>(b + g.off_cflag);
        a.peers.route[w] = reinterpret_cast<int *>(b + g.off_route);
        a.peers.route_flag[w] = reinterpret_cast<int *>(b + g.off_rtflag);
    }
    if (g.gin_mask) {
        uint8_t *b = static_cast<uint8_t *>(g.local_alloc);
        a.gin = Gin{g.gin_comm, g.gin_win, g.gin_mask, g.gin_nctx, g.off_pool, g.off_route, g.off_rcomb, g.st_route,
                    g.st_row, g.st_comb, reinterpret_cast<int *>(b + g.st_route), b + g.st_row,
                    reinterpret_cast<__nv_bfloat16 *>(b + g.st_comb), reinterpret_cast<__nv_bfloat16 *>(b + g.off_rcomb),
                    reinterpret_cast<int *>(b + g.st_cidx)};
    }
    a.T = int(x.size(0));
    TORCH_CHECK(a.T <= g.TCAP, "T exceeds TCAP");
    a.H = g.H, a.I = g.I, a.EL = g.EL, a.W = g.world, a.TOPK = g.TOPK, a.TCAP = g.TCAP, a.rank = g.rank;
    a.swiglu_limit = float(swiglu_limit);
    TORCH_CHECK(!w13_up_first || kRm, "w13_up_first: row-major weights ('w4a8_rm') only");
    a.w13_up_first = int(w13_up_first);
    // tokens per item: 0 = from the expected tokens per local expert (this rank's T, uniform routing)
    int nt = int(tile_tokens) / kTile;
    if (tile_tokens <= 0) {
        const float per_expert = float(a.T) * a.TOPK / a.EL;
        // swept on 6000D EP8, cold L2, 1-64 tokens per rank, every format: 8 while one sub-tile holds an expert's
        // tokens, 16 beyond
        nt = per_expert > 8.f ? 2 : 1;
    }
    TORCH_CHECK(nt >= 1 && nt <= kMaxNt && (tile_tokens <= 0 || tile_tokens == nt * kTile), "tile_tokens: 8 or 16");
    auto stream = c10::cuda::getCurrentCUDAStream();
    switch (nt) {
        case 1: sm120_fp4_mega_moe_kernel<1><<<g.grid[1], kWarps * 32, 0, stream>>>(a); break;
        default: sm120_fp4_mega_moe_kernel<2><<<g.grid[2], kWarps * 32, 0, stream>>>(a); break;
    }
    CHECK_CUDA(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init);
    m.def("open_peers", &open_peers);
    m.def("window_bytes", &window_bytes);
    m.def("init_window", &init_window);
    m.def("grid_size", &grid_size);
    m.def("counters", &counters);
    m.def("run", &run);
    m.attr("nvfp4") = bool(kNv);
}
