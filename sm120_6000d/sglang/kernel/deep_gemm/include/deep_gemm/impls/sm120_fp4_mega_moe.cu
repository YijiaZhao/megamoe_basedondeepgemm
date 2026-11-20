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
 * stores. A payload is staged in this rank's window first, then put to the peer's receive region: the token rows
 * (values, scales, metadata) for one GIN peer packed in (its local expert, slot) order and put once, after the whole
 * dispatch (one put per row was bound by the NIC's message rate); the partials of a finished FC2 tile go back as one
 * put per GIN source, its rows packed the same way. Each put adds its rows (1 for the routing
 * counts) to a signal of the receiver, one per source and kind; the receiver knows how many rows to expect and waits
 * for the cumulative signal instead of a flag.
 *
 * Which pairs use P2P and which GIN is a per-pair transport table (deep_gemm/mega/sm120_fp4_transport.py, the rules of
 * FlashInfer's Cake SM120 MegaMoE); the GIN build also carries Cake's setup-time transport probe (`probe_round`), which
 * times one all-to-all round over a candidate table in the window's pool (inbox) and GIN staging (source) regions.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
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
constexpr int kTile = 8;                                // tokens per MMA (N); an item takes NT (1, 2 or 4) of them
constexpr int kMaxNt = 4;
constexpr size_t kPf0Bytes = size_t(64) << 20;          // t=0 L2 prefetch budget over the locally known experts
constexpr int kUnroll = 4;
// The widest group and the work-item table are build parameters: the default build takes up to 16 ranks; sm120_fp4.py
// builds groups of 17 to 32 ranks with -DSM120_FP4_MAX_WORLD=32 -DSM120_FP4_MAX_TILES=6400 (EP32: 4 hosts x 8 GPUs).
#ifndef SM120_FP4_MAX_WORLD
#define SM120_FP4_MAX_WORLD 16
#endif
#ifndef SM120_FP4_MAX_TILES
#define SM120_FP4_MAX_TILES 2304
#endif
constexpr int kMaxWorld = SM120_FP4_MAX_WORLD;
static_assert(kMaxWorld == 16 || kMaxWorld == 32, "builds for up to 16 and up to 32 ranks");
constexpr int kMaxEl = 4096 / kMaxWorld;  // local experts: 256 (128 in the 32-rank build), Map::cnt keeps its size
constexpr uint64_t kSpinNs = 30000000000ull;

constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4, kErrRouterWait = 5;
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4, kCtrEpoch = 5;
constexpr int kCtrRoute = 6;  // CTAs done with the in-kernel router (Route)
constexpr int kMaxRouteE = 384;  // in-kernel router: experts (logical)
constexpr int kCtrFlag = 8, kCtrL1 = 16;  // kCtrFlag: local copies (epoch) of the route / rows / combine flags
// GIN: FC2 items done per tile (reset per call), cumulative signal totals received per kind and source (u64, kept)
// sum over local experts of ceil(rows / 8): EL + W * TCAP * TOPK / 8; 2304: TCAP <= 64 at 16 ranks and top-8; the
// 32-rank build's 6400: DeepSeek-V4.1's top-6 at EP32 up to 266 tokens per rank
constexpr int kMaxTiles = SM120_FP4_MAX_TILES;
constexpr int kCtrTile = kCtrL1 + kMaxEl, kCtrCum = kCtrTile + kMaxTiles;
constexpr int kCtrInts = kCtrCum + 2 * 3 * kMaxWorld;
static_assert(kCtrCum % 2 == 0, "the cumulative GIN totals must be 8-byte aligned");
constexpr int kSigRoute = 0, kSigRows = kMaxWorld, kSigComb = 2 * kMaxWorld;  // GIN signal ids, + source rank
// transport probe: GIN data signals [kSigProbe, + kMaxWorld), barrier signals [kSigProbe + kMaxWorld, + kMaxWorld)
// (sm120_fp4_mega_moe_gin.cu: 5 kMaxWorld); the window's probe LSA signal slots: data [0, kMaxWorld), barrier
// [kMaxWorld, 2 kMaxWorld) by source rank, kProbeSigStride u32 apart
constexpr int kSigProbe = 3 * kMaxWorld, kProbeSigStride = 32;

// a pool row: H/kXB E2M1 / E4M3 values, H/kSfVec scale bytes, then 16 bytes holding the int source token*8 + k
__host__ __device__ __forceinline__ int pool_rx(int H) { return H / kXB + H / kSfVec + 16; }

struct Peers {
    uint8_t *pool[kMaxWorld];       // [EL][W][TCAP] pool rows                    (dest-owned)
    int *recv_flag[kMaxWorld];      // [W] epoch
    __nv_bfloat16 *comb[kMaxWorld]; // [TCAP][TOPK][H]                    (src-owned)
    int *comb_flag[kMaxWorld];      // [W] epoch
    int *route[kMaxWorld];          // [2][W][EL]: tokens each source rank sends to each expert, by epoch parity (dest-owned)
    int *route_flag[kMaxWorld];     // [W] epoch
};

// NIC transport: window offsets of the receive regions (the P2P layout) and of this rank's staging regions, and the
// staging pointers. mask 0 (always without SM120_FP4_GIN): every peer over P2P.
struct Gin {
    const void *comm;  // ncclDevComm const * (device memory)
    void *win;         // ncclWindow_t of the whole buffer
    uint32_t mask;     // bit w: rank w is reached with GIN puts
    int nctx;          // GIN contexts: CTA b posts on context b % nctx; a signal is the sum over them
    size_t off_pool, off_route, off_rcomb, st_route, st_row, st_comb, off_rrow;
    int *s_route;           // [2][W][EL] this rank's token counts for each GIN peer's experts, by epoch parity
    uint8_t *s_row;         // [W][TCAP*TOPK] pool rows of the copies to each GIN peer, packed (its local expert, slot)
    const uint8_t *r_row;   // [W][TCAP*TOPK] pool rows from each GIN source, packed the same way
    __nv_bfloat16 *s_comb;  // [W][TCAP*TOPK][H] partials for each GIN source, packed (local expert, slot)
    __nv_bfloat16 *r_comb;  // [W][TCAP*TOPK][H] partials from each GIN expert rank, packed the same way
    int *cidx;              // [TCAP][TOPK] packed index at the GIN expert rank of copy (t, k)
};

// In-kernel router (DeepSeek-V4.1, SGLang's moe_fused_gate with sqrtsoftplus via log1p): from the gate logits, the
// routes this call's topk_ids / topk_w are then written from (CTA b routes token b with one warp, the others wait).
// logits [S][T][E] fp32, the S partial sums added in order; score = sqrt(l > 20 ? l : log1p(exp(l))), ranked by
// score + bias (bias_alt where input_ids == alt_id; NaN first), top-k with the lowest id on ties, weight = score,
// renormalized by sum (+ eps, or the zero-sum guard), times scale when apply_scale; EPLB (l2p [E][R], ascending, -1
// padded): row t takes replica t % n of n (SGLang's 'dynamic' dispatch).
struct Route {
    const float *logits;  // null: topk_ids / topk_w are the caller's
    const float *bias, *bias_alt;
    const int64_t *input_ids;
    int64_t alt_id;
    const int *l2p;
    int *ids;
    float *w;
    int S, E, R, renorm, apply_scale;
    float eps, scale;
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
    Route route;
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
// the puts spread over the contexts' queue pairs by CTA and peer: the puts one CTA posts to several peers at once (a
// thread per peer) go out in parallel instead of queueing on one context
__device__ __forceinline__ void gin_put(const Args &a, int peer, size_t dst_off, size_t src_off, size_t bytes, int sig,
                                        int add) {
    const ncclDevComm &c = *static_cast<const ncclDevComm *>(a.gin.comm);
    const ncclWindow_t win = static_cast<ncclWindow_t>(a.gin.win);
    ncclGin(c, (int(blockIdx.x) + peer) % a.gin.nctx)
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

#if SM120_FP4_A8  // the activation staging (cp.async) and the row-major paths: W4A8 builds
__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}
__device__ __forceinline__ void cp_async8(void *dst, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// activation staging of item_gemm_rm: per K step (kUnroll K64 chunks) a token's kStageB value bytes, rows kStageRow
// apart (the pad spreads the 8 token rows a warp reads at once over the banks), and its kStageW scale words
constexpr int kStageB = kUnroll * 64, kStageRow = kStageB + 32, kStageW = kUnroll / 2;

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

// item_gemm_rm for single-N8 items (NT 1): no staging and no CTA barrier per K step; each
// warp reads its lane's token (xrow / xsrow: token g's value row / scale words) where it uses it, and loads the
// weights one K step ahead, as item_gemm's single-N8 path. Only sub-tile 0 is computed; the others stay zero.
template <int NT>
__device__ __forceinline__ void item_gemm_rm_direct(float (&acc)[kNbPerWarp][NT][4], const uint8_t *W, const uint32_t *WS,
                                                    int nb0, int rpb, int roff, int K, const uint8_t *xrow,
                                                    const uint32_t *xsrow, int lane) {
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
    constexpr int kSw = kUnroll / 2, kQ = kUnroll / 2;
    const uint2 *xp = reinterpret_cast<const uint2 *>(xrow) + c;
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
    load(0, wq, sfa);
#pragma unroll 2
    for (int k0 = 0; k0 < kch; k0 += kUnroll) {
        if (k0 + kUnroll < kch) load(k0 + kUnroll, wn, sfn);
        uint32_t wv[kNbPerWarp][kUnroll][2][2];  // [block][K64 chunk][K32 step][row g / g + 8]
#pragma unroll
        for (int p = 0; p < kQ; ++p)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    uint32_t v[4] = {wq[j][p][r].x, wq[j][p][r].y, wq[j][p][r].z, wq[j][p][r].w};
                    quad_transpose(v, c);
#pragma unroll
                    for (int b = 0; b < 4; ++b) wv[j][p * 2 + b / 2][b % 2][r] = v[b];
                }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int kc = k0 + u;
            const uint32_t xsf = xsrow[kc / 2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint16_t sel = uint16_t((kc * 2 + h) & 3);
                const uint2 x = xp[(kc * 2 + h) * 4];  // bytes 8c..8c+7 of the K32 step
                const uint32_t b0 = __byte_perm(x.x, x.y, 0x6420u), b1 = __byte_perm(x.x, x.y, 0x7531u);
#pragma unroll
                for (int j = 0; j < kNbPerWarp; ++j) {
                    uint32_t wa[4];
                    expand(wv[j][u][h][0], wa[0], wa[2]);
                    expand(wv[j][u][h][1], wa[1], wa[3]);
                    mma_w4a8(acc[j][0], wa, b0, b1, sfa[j][u / 2], sel, xsf, sel);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
            for (int p = 0; p < kQ; ++p) wq[j][p][0] = wn[j][p][0], wq[j][p][1] = wn[j][p][1];
#pragma unroll
            for (int s = 0; s < kSw; ++s) sfa[j][s] = sfn[j][s];
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
// The item's tokens are the same for its 8 warps: the whole CTA stages their activations (xr / xs: token tk's value
// row / scale words, the item's live sub-tiles only) one K step ahead with cp.async into sx / ss ([2][kT] rows, double
// buffered), and every warp reads them from shared memory instead of each loading them from global memory.
// Every thread of the CTA calls it.
template <int NT>
__device__ __forceinline__ void item_gemm_rm(float (&acc)[kNbPerWarp][NT][4], const uint8_t *W, const uint32_t *WS,
                                             int nb0, int rpb, int roff, int K, const uint8_t *const *xr,
                                             const uint32_t *const *xs, uint8_t *sx, uint32_t *ss, int lane, int nsub) {
    constexpr int kT = kTile * NT;
    const int g = lane / 4, c = lane % 4;
    if constexpr (NT == 1) {  // single N8: each warp reads its activations itself, warps unsynchronized
        item_gemm_rm_direct<NT>(acc, W, WS, nb0, rpb, roff, K, xr[g], xs[g], lane);
        return;
    }
    const int ntk = (NT == 1 ? 1 : nsub) * kTile;
    auto stage = [&](int k0, int b) {
        for (int u = threadIdx.x; u < ntk * (kStageB / 16); u += blockDim.x) {
            const int tk = u / (kStageB / 16), q = u % (kStageB / 16);
            cp_async16(sx + (b * kT + tk) * kStageRow + q * 16, xr[tk] + k0 * 64 + q * 16);
        }
        for (int tk = threadIdx.x; tk < ntk; tk += blockDim.x)
            cp_async8(ss + (b * kT + tk) * kStageW, xs[tk] + k0 / 2);
        cp_async_commit();
    };
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
    stage(0, 0);
    if constexpr (kDb) load(0, wq, sfa);
#pragma unroll 2
    for (int k0 = 0; k0 < kch; k0 += kUnroll) {
        const int buf = (k0 / kUnroll) & 1;
        if (k0 + kUnroll < kch) {
            stage(k0 + kUnroll, buf ^ 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        if constexpr (kDb) {
            if (k0 + kUnroll < kch) load(k0 + kUnroll, wn, sfn);
        } else {
            load(k0, wq, sfa);
        }
        __syncthreads();  // this step's activations staged by every thread
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
            for (int s = 0; s < NT; ++s)
                sfb[s] = NT == 1 || s < nsub ? ss[(buf * kT + s * kTile + g) * kStageW + (kc - k0) / 2] : 0u;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int step = kc * 2 + h;
                const uint16_t sel = uint16_t(step & 3);
                uint32_t b0[NT], b1[NT];  // activation elements 8c + {0, 2, 4, 6} / 8c + {1, 3, 5, 7}
#pragma unroll
                for (int s = 0; s < NT; ++s) {
                    if (NT > 1 && s >= nsub) {  // past the tile's tokens (warp-uniform)
                        b0[s] = b1[s] = 0u;
                        continue;
                    }
                    const uint2 x = reinterpret_cast<const uint2 *>(sx + (buf * kT + s * kTile + g) * kStageRow)
                        [((kc - k0) * 2 + h) * 4 + c];
                    b0[s] = __byte_perm(x.x, x.y, 0x6420u);
                    b1[s] = __byte_perm(x.x, x.y, 0x7531u);
                }
#pragma unroll
                for (int j = 0; j < kNbPerWarp; ++j) {
                    uint32_t wa[4];  // even nibbles (elements 8c + 2i) of rows g / g + 8, then the odd ones
                    expand(wv[j][u][h][0], wa[0], wa[2]);
                    expand(wv[j][u][h][1], wa[1], wa[3]);
#pragma unroll
                    for (int s = 0; s < NT; ++s)
                        if (NT == 1 || s < nsub) mma_w4a8(acc[j][s], wa, b0[s], b1[s], sfa[j][u / 2], sel, sfb[s], sel);
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
        __syncthreads();  // every warp is done with this buffer before the next step stages into it
    }
}

// item_gemm for packed W4A8 weights on items of 16 or 32 tokens (NT 2 / 4), the way item_gemm_rm runs them: the CTA
// stages the item's activations (xr / xs, its live sub-tiles only) into sx / ss one K step ahead, the weights are
// loaded one K step ahead into a second register buffer, and the sub-tiles past the item's tokens (nsub, CTA-uniform)
// take no MMA. The packed fragments need no transpose: lane L's 16 bytes of a K64 chunk are its A registers of the
// chunk's two K32 steps (`expand`), its activation words step * 8 + c and + 4 in the natural order. Every thread of the
// CTA calls it.
template <int NT>
__device__ __forceinline__ void item_gemm_pk(float (&acc)[kNbPerWarp][NT][4], const uint4 *W, const uint32_t *WS,
                                             int nb0, int K, const uint8_t *const *xr, const uint32_t *const *xs,
                                             uint8_t *sx, uint32_t *ss, int lane, int nsub) {
    static_assert(NT > 1, "single-N8 items take item_gemm");
    constexpr int kT = kTile * NT;
    const int g = lane / 4, c = lane % 4;
    const int ntk = nsub * kTile;
    auto stage = [&](int k0, int b) {
        for (int u = threadIdx.x; u < ntk * (kStageB / 16); u += blockDim.x) {
            const int tk = u / (kStageB / 16), q = u % (kStageB / 16);
            cp_async16(sx + (b * kT + tk) * kStageRow + q * 16, xr[tk] + k0 * 64 + q * 16);
        }
        for (int tk = threadIdx.x; tk < ntk; tk += blockDim.x)
            cp_async8(ss + (b * kT + tk) * kStageW, xs[tk] + k0 / 2);
        cp_async_commit();
    };
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
    constexpr int kSw = kUnroll / 2;  // scale words per unrolled step
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
            for (int j = 0; j < kNbPerWarp; ++j) sf[j][q] = __ldg(sp[j] + (k0 / 2 + q) * 16);
    };
    stage(0, 0);
    load(0, wv, sfa);
#pragma unroll 2
    for (int k0 = 0; k0 < kch; k0 += kUnroll) {
        const int buf = (k0 / kUnroll) & 1;
        if (k0 + kUnroll < kch) {
            stage(k0 + kUnroll, buf ^ 1);
            cp_async_wait<1>();
            load(k0 + kUnroll, wn, sfn);
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();  // this step's activations staged by every thread
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int kc = k0 + u;
            uint32_t sfb[NT];
#pragma unroll
            for (int s = 0; s < NT; ++s)
                sfb[s] = s < nsub ? ss[(buf * kT + s * kTile + g) * kStageW + (kc - k0) / 2] : 0u;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int step = kc * 2 + h;
                const uint16_t sel = uint16_t(step & 3);
                uint32_t b0[NT], b1[NT];
#pragma unroll
                for (int s = 0; s < NT; ++s) {
                    if (s >= nsub) {  // past the tile's tokens (CTA-uniform)
                        b0[s] = b1[s] = 0u;
                        continue;
                    }
                    const uint8_t *row = sx + (buf * kT + s * kTile + g) * kStageRow;  // token g of sub-tile s
                    const uint32_t *x = reinterpret_cast<const uint32_t *>(row) + ((kc - k0) * 2 + h) * 8;
                    b0[s] = x[c];
                    b1[s] = x[4 + c];
                }
#pragma unroll
                for (int j = 0; j < kNbPerWarp; ++j) {
                    uint32_t wa[4];
                    expand(h ? wv[j][u].z : wv[j][u].x, wa[0], wa[1]);
                    expand(h ? wv[j][u].w : wv[j][u].y, wa[2], wa[3]);
#pragma unroll
                    for (int s = 0; s < NT; ++s)
                        if (s < nsub) mma_w4a8(acc[j][s], wa, b0[s], b1[s], sfa[j][u / 2], sel, sfb[s], sel);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) wv[j][u] = wn[j][u];
#pragma unroll
            for (int q = 0; q < kSw; ++q) sfa[j][q] = sfn[j][q];
        }
        __syncthreads();  // every warp is done with this buffer before the next step stages into it
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

// (value, -id) order: larger value first, then the lower expert id
__device__ __forceinline__ bool route_better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

// one warp: token t's top-k of the in-kernel router into r.ids / r.w (lane k: the k-th pick)
__device__ __forceinline__ void route_token(const Route &r, int t, int T, int K, int lane) {
    constexpr int kPer = kMaxRouteE / 32;  // experts per lane (e = lane + 32 j)
    const float *bias = r.bias_alt && r.input_ids && r.input_ids[t] == r.alt_id ? r.bias_alt : r.bias;
    float lg[kPer], rank[kPer], score[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) lg[j] = 0.f;
    for (int s = 0; s < r.S; ++s) {  // the slices in order; per slice the lane's experts loaded together
        const float *ps = r.logits + (size_t(s) * T + t) * r.E + lane;
        float v[kPer];
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[j] = lane + 32 * j < r.E ? __ldcg(ps + 32 * j) : 0.f;
#pragma unroll
        for (int j = 0; j < kPer; ++j) lg[j] += v[j];
    }
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int e = lane + 32 * j;
        rank[j] = -INFINITY;
        score[j] = 0.f;
        if (e < r.E) {
            const float l = lg[j];
            score[j] = sqrtf(l > 20.f ? l : log1pf(expf(l)));
            const float v = score[j] + bias[e];
            rank[j] = v == v ? v : INFINITY;  // NaN ranks first, as torch.topk
        }
    }
    int sel = -1;
    float sw = 0.f, sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            if (rank[j] != -INFINITY && route_better(rank[j], lane + 32 * j, bv, bi)) bv = rank[j], bi = lane + 32 * j;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (route_better(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        float w = 0.f;
        if (lane == bi % 32) {
#pragma unroll
            for (int j = 0; j < kPer; ++j)
                if (j == bi / 32) w = score[j], rank[j] = -INFINITY;
        }
        w = __shfl_sync(0xffffffffu, w, bi % 32);
        sum += w;
        if (lane == k) sel = bi, sw = w;
    }
    {  // moe_fused_gate's tl.sum over its K slots (lanes, xor 2 / 1 / 4) and its division (div.full.f32)
        float v = lane < K ? sw : 0.f;
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        sum = __shfl_sync(0xffffffffu, v, 0);
    }
    if (r.renorm) {
        const float den = r.eps > 0.f ? sum + r.eps : (sum > 0.f ? sum : 1.f);
        asm("div.full.f32 %0, %1, %2;" : "=f"(sw) : "f"(sw), "f"(den));
    }
    if (r.apply_scale) sw *= r.scale;
    if (lane < K) {
        if (r.l2p) {
            const int *cand = r.l2p + size_t(sel) * r.R;
            int n = 0;
            while (n < r.R && cand[n] >= 0) ++n;
            sel = cand[n > 1 ? t % n : 0];
        }
        r.ids[t * K + lane] = sel;
        r.w[t * K + lane] = sw;
    }
}

template <int NT>  // tokens per item: NT x 8
// two CTAs per SM (the grid is sized from the occupancy): the register budget is 128, which the unrolled weight loop
// uses to keep loads in flight
__global__ void __launch_bounds__(kWarps * 32, 2) sm120_fp4_mega_moe_kernel(Args a) {
    constexpr int kT = kTile * NT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
    __shared__ Map m;
    __shared__ int s_item, s_ok, s_epoch, s_last;
    // L1: [tok][64 inter] / L2: [tok][128 hidden] fp32 / dispatch: one FP4 token row (H/2 + H/kSfVec bytes).
    // W4A8: the same bytes hold item_gemm_rm's / item_gemm_pk's activation staging while an item's GEMM runs (the
    // epilogues use s_tile only after it)
#if SM120_FP4_A8
    constexpr int kSx = 2 * kT * kStageRow, kSxs = kSx + 2 * kT * kStageW * 4;
    constexpr int kRaw = kT * kRowsPerItem * 4 > kSxs ? kT * kRowsPerItem * 4 : kSxs;
#else
    constexpr int kRaw = kT * kRowsPerItem * 4;
#endif
    __shared__ __align__(16) uint8_t s_raw[kRaw];
    auto s_tile = reinterpret_cast<float (*)[kRowsPerItem]>(s_raw);
#if SM120_FP4_A8
    uint8_t *const s_sx = s_raw;
    uint32_t *const s_ss = reinterpret_cast<uint32_t *>(s_raw + kSx);
    __shared__ const uint8_t *s_xr[kT];  // the item's token rows: values / scale words
    __shared__ const uint32_t *s_xs[kT];
#endif

    unsigned long long st_[16] = {gtime()};  // 0 start 1 pushed 2 flags 3 work end 4 comb flags 5 end 6 L1-wait ns 7 items 8 route-push 9 route-ok 10 map 11 pf 12 flag-sent 13 last FC1 14 first FC2 15 comb release
    if (threadIdx.x == 0) {
        s_epoch = *reinterpret_cast<volatile int *>(a.ctr + kCtrEpoch) + 1;
        s_ok = 1;  // cleared by a timed-out wait
    }
    __syncthreads();
    const int epoch = s_epoch;
    int *ctr = a.ctr;
    if (a.route.logits && a.T > 0) {  // in-kernel router: CTA b routes tokens b, b + grid, ...; every CTA then waits
        for (int t = blockIdx.x; t < a.T; t += gridDim.x)
            if (warp == 0) route_token(a.route, t, a.T, a.TOPK, lane);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (int(blockIdx.x) < a.T) {
                __threadfence();
                atomicAdd(ctr + kCtrRoute, 1);
            }
            // the routing CTAs wait on nothing before their count: no deadlock even when not all CTAs are resident
            if (!wait_ge(ctr + kCtrRoute, min(a.T, int(gridDim.x)), false)) atomicExch(ctr + kCtrErr, kErrRouterWait);
        }
        __syncthreads();
    }
#if SM120_FP4_GIN_DEVICE
    __shared__ int s_ncopy[kMaxWorld];  // this rank's copies to each rank (GIN signal counts), counted by the whole CTA
    if (a.gin.mask) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) s_ncopy[w] = 0;
        __syncthreads();
        for (int u = threadIdx.x; u < a.T * a.TOPK; u += blockDim.x)
            if (a.topk_ids[u] >= 0) atomicAdd(&s_ncopy[a.topk_ids[u] / a.EL], 1);
        __syncthreads();
    }
#endif
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
        // the counts go to the half of the peer's route region this call's epoch parity picks: a peer that sends this
        // rank nothing need not wait for it, finishes the call and posts its next call's counts while this rank may not
        // have read these yet (it cannot get two calls ahead: the next call waits for every rank's counts)
        const int rhalf = (epoch & 1) * W * EL;
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x) {
            const int d = u / EL;
            *(is_gin(a, d) ? a.gin.s_route + rhalf + u : a.peers.route[d] + rhalf + a.rank * EL + u % EL) = m.cnt[d][u % EL];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x < W) {  // a thread per peer
            const int d = threadIdx.x;
#if SM120_FP4_GIN_DEVICE
            if (is_gin(a, d))
                gin_put(a, d, a.gin.off_route + (size_t(rhalf) + size_t(a.rank) * EL) * 4,
                        a.gin.st_route + (size_t(rhalf) + size_t(d) * EL) * 4, EL * 4, kSigRoute + a.rank, 1);
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
        __shared__ int s_rowk[8], s_dstk[8], s_pk[8];
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
                    if (lane == 0) a.gin.cidx[t * TOPK + warp] = s_pk[warp] = pre + slot;
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
                        uint8_t *rec = is_gin(a, dst) ? a.gin.s_row + (size_t(dst) * TCAP * TOPK + s_pk[warp]) * pool_rx(H)
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
                    // a GIN copy goes to its packed staging row, put with the peer's others after the dispatch
                    uint8_t *dx = is_gin(a, s_dstk[k])
                                      ? a.gin.s_row + (size_t(s_dstk[k]) * TCAP * TOPK + s_pk[k]) * pool_rx(H)
                                      : a.peers.pool[s_dstk[k]] + size_t(s_rowk[k]) * pool_rx(H);
                    uint8_t *ds = dx + H / kXB;
                    if (q < nq)
                        reinterpret_cast<uint4 *>(dx)[oq + q] = reinterpret_cast<const uint4 *>(s_q)[q];
                    else
                        reinterpret_cast<uint4 *>(ds)[os + q - nq] = reinterpret_cast<const uint4 *>(s_sc)[q - nq];
                }
                __syncthreads();
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    st_[1] = gtime();
    if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
    __syncthreads();
    if (s_item && threadIdx.x < W) {  // last CTA out of the dispatch: rows fenced -> flag, or the GIN peer's rows
        const int d = threadIdx.x;
        if (!is_gin(a, d)) {
            st_release_sys(a.peers.recv_flag[d] + a.rank, epoch);
        } else {
#if SM120_FP4_GIN_DEVICE
            const int n = s_ncopy[d];
            if (n > 0)
                gin_put(a, d, a.gin.off_rrow + size_t(a.rank) * TCAP * TOPK * pool_rx(H),
                        a.gin.st_row + size_t(d) * TCAP * TOPK * pool_rx(H), size_t(n) * pool_rx(H), kSigRows + a.rank, n);
#endif
        }
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
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x)
        m.cnt[u / EL][u % EL] = __ldcv(a.peers.route[a.rank] + (epoch & 1) * W * EL + u);
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
        // token row i of source w for local expert el: the pool, or the packed rows from a GIN source
        auto row_ptr = [&](int el, int w, int i) -> const uint8_t * {
            if (!is_gin(a, w)) return pool + ((size_t(el) * W + w) * TCAP + i) * RX;
            int pk = i;
            for (int e2 = 0; e2 < el; ++e2) pk += m.cnt[w][e2];
            return a.gin.r_row + (size_t(w) * TCAP * TOPK + pk) * RX;
        };
        auto meta_of = [&](int el, int w, int i) { return *reinterpret_cast<const int *>(row_ptr(el, w, i) + H / kXB + H / kSfVec); };
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
#if SM120_FP4_RM
                if (threadIdx.x < kT) {  // token tk's row (past the tile's tokens: its first token)
                    int w, i;
                    token_src(m, el, tile * kT + (int(threadIdx.x) < ntok ? int(threadIdx.x) : 0), W, w, i);
                    const uint8_t *row = row_ptr(el, w, i);
                    s_xr[threadIdx.x] = row;
                    s_xs[threadIdx.x] = reinterpret_cast<const uint32_t *>(row + H / kXB);
                }
                __syncthreads();
                // gate rows: block row g, up rows: g + 8 (I rows apart, before or after the gate rows)
                item_gemm_rm<NT>(acc, reinterpret_cast<const uint8_t *>(a.w13 + el * w13_expert) +
                                          (a.w13_up_first ? size_t(I) * (H / 2) : 0),
                                 a.ws13 + el * ws13_expert, nb0, 8, a.w13_up_first ? -I : I, H, s_xr, s_xs, s_sx, s_ss,
                                 lane, (ntok + kTile - 1) / kTile);
#else
#if SM120_FP4_A8
                if constexpr (NT > 1) {  // packed W4A8, 16 / 32-token items: staged, empty sub-tiles skipped
                    if (threadIdx.x < kT) {
                        int w, i;
                        token_src(m, el, tile * kT + (int(threadIdx.x) < ntok ? int(threadIdx.x) : 0), W, w, i);
                        const uint8_t *row = row_ptr(el, w, i);
                        s_xr[threadIdx.x] = row;
                        s_xs[threadIdx.x] = reinterpret_cast<const uint32_t *>(row + H / kXB);
                    }
                    __syncthreads();
                    item_gemm_pk<NT>(acc, a.w13 + el * w13_expert, a.ws13 + el * ws13_expert, nb0, H, s_xr, s_xs, s_sx,
                                     s_ss, lane, (ntok + kTile - 1) / kTile);
                } else
#endif
                {
#pragma unroll
                    for (int s = 0; s < NT; ++s) {
                        int w, i;
                        token_src(m, el, js[s], W, w, i);
                        const uint8_t *row = row_ptr(el, w, i);
                        xr[s] = reinterpret_cast<const uint32_t *>(row);
                        xs[s] = reinterpret_cast<const uint32_t *>(row + H / kXB);
                    }
                    item_gemm<NT>(acc, a.w13 + el * w13_expert, a.ws13 + el * ws13_expert, nb0, H, xr, xs, lane);
                }
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
#if SM120_FP4_RM
                if (threadIdx.x < kT) {
                    const size_t r = size_t(el) * TOKE + tile * kT + (int(threadIdx.x) < ntok ? int(threadIdx.x) : 0);
                    s_xr[threadIdx.x] = a.inter + r * (I / kXB);
                    s_xs[threadIdx.x] = a.inter_s + r * (I / kSfWordK);
                }
                __syncthreads();
                item_gemm_rm<NT>(acc, reinterpret_cast<const uint8_t *>(a.w2 + el * w2_expert), a.ws2 + el * ws2_expert,
                                 nb0, 16, 8, I, s_xr, s_xs, s_sx, s_ss, lane, (ntok + kTile - 1) / kTile);
#else
#if SM120_FP4_A8
                if constexpr (NT > 1) {
                    if (threadIdx.x < kT) {
                        const int tk = int(threadIdx.x) < ntok ? int(threadIdx.x) : 0;
                        const size_t r = size_t(el) * TOKE + tile * kT + tk;
                        s_xr[threadIdx.x] = a.inter + r * (I / kXB);
                        s_xs[threadIdx.x] = a.inter_s + r * (I / kSfWordK);
                    }
                    __syncthreads();
                    item_gemm_pk<NT>(acc, a.w2 + el * w2_expert, a.ws2 + el * ws2_expert, nb0, I, s_xr, s_xs, s_sx,
                                     s_ss, lane, (ntok + kTile - 1) / kTile);
                } else
#endif
                {
#pragma unroll
                    for (int s = 0; s < NT; ++s) {
                        const size_t r = size_t(el) * TOKE + js[s];
                        xr[s] = reinterpret_cast<const uint32_t *>(a.inter + r * (I / kXB));
                        xs[s] = a.inter_s + r * (I / kSfWordK);
                    }
                    item_gemm<NT>(acc, a.w2 + el * w2_expert, a.ws2 + el * ws2_expert, nb0, I, xr, xs, lane);
                }
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
                    const int meta = meta_of(el, w, i);
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
#if SM120_FP4_GIN_DEVICE
            const int n = s_ncopy[w];  // this rank's copies at w: their partial rows
#else
            const int n = 0;
#endif
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
                    int rows = 0;
                    for (int el = 0; el < EL; ++el) rows += m.cnt[w][el];
                    gin_cum(ctr)[w] += 1;
                    gin_cum(ctr)[kMaxWorld + w] += rows;
                    gin_cum(ctr)[2 * kMaxWorld + w] += s_ncopy[w];
                }
            if (a.gin.mask)
                for (int u = 0; u < m.tiles_prefix[EL]; ++u) ctr[kCtrTile + u] = 0;
#endif
            ctr[kCtrDispatch] = 0;
            ctr[kCtrRoute] = 0;
            ctr[kCtrQueue] = 0;
            ctr[kCtrL2Done] = 0;
            for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
            __threadfence();
            ctr[kCtrExit] = 0;
        }
    }
}


// ------------------------------------------------------------------------------------------------ transport probe
// Ported from FlashInfer's Cake SM120 MegaMoE (cake_sm120_megamoe_binding.cu, `probe` namespace; Apache-2.0): one CTA
// per destination; LSA destinations stream 8 KB blocks with a 3-stage cp.async.bulk ring, GIN destinations issue
// chunked ncclGin puts from one elected thread. The last CTA (blockIdx.x == world - 1) is the waiter: it first runs a
// device-side rendezvous with every peer it talks to this round (a 4-byte ping on a dedicated barrier signal over that
// pair's own transport), takes a %globaltimer stamp, releases the senders through a device-global `go` flag, and
// stamps again once every sender CTA finished and every receive signal arrived. Inter-rank launch skew is therefore
// outside the measured window.
namespace probe {

constexpr int kSelf = 0, kLsa = 1, kGin = 2;
constexpr int kStageBytes = 8192, kStages = 3, kCopyWarps = 4;  // 4 copier warps x 3 stages x 8 KB per destination
constexpr int kBlockWarps = 8;

struct ProbeArgs {
    const void *comm;               // ncclDevComm const *
    const unsigned char *src;       // local GIN staging region (window)
    void *win;                      // ncclWindow_t of the whole buffer
    size_t src_off, dst_off0;       // window offsets: source region, inbox (pool) region
    unsigned long long dst_ptr[kMaxWorld];   // LSA-mapped peer inbox (0 = unmapped)
    unsigned long long peer_sig[kMaxWorld];  // LSA-mapped peer probe signal slots
    uint32_t *my_sig;                        // local probe signal slots
    int row[kMaxWorld];                      // transport per destination
    int rank, world;
    unsigned long long bytes_per_dest, dst_slot_bytes;
    uint32_t epoch;                          // expected cumulative LSA signal value after this round
    unsigned chunk_bytes;                    // GIN put granularity
    const unsigned long long *gin_targets;          // per-source cumulative data-signal target
    const unsigned long long *gin_barrier_targets;  // per-source cumulative barrier target
    unsigned int *go_flag, *done_ctas;              // device scratch, host-zeroed each round
    unsigned long long *elapsed_ns_out;
};

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void st_release_sys_u32(uint32_t *p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu_u32(uint32_t *p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

extern __shared__ __align__(1024) unsigned char probe_smem[];

__global__ void probe_send(ProbeArgs a) {
#if SM120_FP4_GIN_DEVICE
    const ncclDevComm &comm = *static_cast<const ncclDevComm *>(a.comm);
    const ncclWindow_t win = static_cast<ncclWindow_t>(a.win);
    const int sig_data = kSigProbe, sig_bar = kSigProbe + kMaxWorld;
    const unsigned long long dst_off = static_cast<unsigned long long>(a.rank) * a.dst_slot_bytes;
    if (blockIdx.x == static_cast<unsigned>(a.world) - 1) {
        const int w = threadIdx.x / 32;
        __shared__ int recv_done;  // every warp finishes its receive waits before warp 0 stamps t1
        if (threadIdx.x == 0) recv_done = 0;
        __syncthreads();
        if (threadIdx.x % 32) return;
        unsigned long long t0 = 0;
        if (w == 0) {
            for (int p = 0; p < a.world; ++p) {  // rendezvous 1: ping every peer of this round over its transport
                if (p == a.rank) continue;
                if (a.row[p] == kLsa) {
                    uint32_t *dst4 = reinterpret_cast<uint32_t *>(reinterpret_cast<unsigned char *>(a.dst_ptr[p]) + dst_off);
                    *dst4 = *reinterpret_cast<const uint32_t *>(a.src);
                    __threadfence_system();
                    st_release_sys_u32(reinterpret_cast<uint32_t *>(a.peer_sig[p]) + (kMaxWorld + a.rank) * kProbeSigStride,
                                       a.epoch);
                } else if (a.row[p] == kGin) {
                    ncclGin gin(comm, 0);
                    gin.put(ncclTeamWorld(comm), p, win, a.dst_off0 + dst_off, win, a.src_off, 4,
                            ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(sig_bar + a.rank), 1}, ncclGin_None{},
                            ncclCoopThread());
                }
            }
            for (int p = 0; p < a.world; ++p) {  // rendezvous 2: wait for their pings
                if (p == a.rank) continue;
                if (a.row[p] == kLsa) {
                    const uint32_t *sig = a.my_sig + (kMaxWorld + p) * kProbeSigStride;
                    while (static_cast<int32_t>(ld_acquire_sys_u32(sig) - a.epoch) < 0) {
                    }
                } else if (a.row[p] == kGin) {
                    ncclGin gin(comm, 0);
                    gin.waitSignal(ncclCoopThread(), static_cast<ncclGinSignal_t>(sig_bar + p), a.gin_barrier_targets[p],
                                   64, cuda::memory_order_acquire);
                }
            }
            t0 = gtime();
            __threadfence();
            st_release_gpu_u32(a.go_flag, 1u);
        }
        for (int src = w; src < a.world; src += kBlockWarps) {  // receive waits: warp w owns sources w, w + 8, ...
            if (src == a.rank) continue;
            if (a.row[src] == kLsa) {
                const uint32_t *sig = a.my_sig + src * kProbeSigStride;
                while (static_cast<int32_t>(ld_acquire_sys_u32(sig) - a.epoch) < 0) {
                }
            } else if (a.row[src] == kGin) {
                ncclGin gin(comm, 0);
                gin.waitSignal(ncclCoopThread(), static_cast<ncclGinSignal_t>(sig_data + src), a.gin_targets[src], 64,
                               cuda::memory_order_acquire);
            }
        }
        atomicAdd(&recv_done, 1);
        if (w == 0) {
            while (atomicAdd(&recv_done, 0) < kBlockWarps) {
            }
            while (atomicAdd(a.done_ctas, 0u) < static_cast<unsigned>(a.world - 1)) {
            }
            *a.elapsed_ns_out = gtime() - t0;
        }
        return;
    }

    const int dest = (a.rank + 1 + blockIdx.x) % a.world;
    const int kind = a.row[dest];
    const int w = threadIdx.x / 32;
    __shared__ int done;
    if (threadIdx.x == 0) done = 0;
    __syncthreads();
    if (threadIdx.x % 32) return;
    if (w >= kCopyWarps) return;
    while (ld_acquire_gpu_u32(a.go_flag) == 0) {
    }  // the first transfer waits for the rendezvous
    if (kind == kLsa) {
        unsigned char *my = probe_smem + static_cast<size_t>(w) * kStages * kStageBytes;
        uint64_t *bars = reinterpret_cast<uint64_t *>(probe_smem + static_cast<size_t>(kCopyWarps) * kStages * kStageBytes) +
                         w * kStages;
        for (int st = 0; st < kStages; ++st)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(&bars[st])), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        __threadfence_block();
        unsigned char *dst = reinterpret_cast<unsigned char *>(a.dst_ptr[dest]) + dst_off;
        unsigned n = 0;
        for (unsigned long long off = static_cast<unsigned long long>(w) * kStageBytes; off < a.bytes_per_dest;
             off += static_cast<unsigned long long>(kCopyWarps) * kStageBytes, ++n) {
            const unsigned len = static_cast<unsigned>(min(static_cast<unsigned long long>(kStageBytes), a.bytes_per_dest - off));
            const int st = n % kStages;
            const unsigned phase = (n / kStages) & 1;
            unsigned char *buf = my + static_cast<size_t>(st) * kStageBytes;
            if (n >= static_cast<unsigned>(kStages)) asm volatile("cp.async.bulk.wait_group.read 2;" ::: "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bars[st])), "r"(len)
                         : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                             smem_u32(buf)),
                         "l"(a.src + off), "r"(len), "r"(smem_u32(&bars[st]))
                         : "memory");
            asm volatile(
                "{\n .reg .pred p;\n PROBE_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n "
                "@!p bra PROBE_WAIT;\n}" ::"r"(smem_u32(&bars[st])),
                "r"(phase)
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst + off), "r"(smem_u32(buf)),
                         "r"(len)
                         : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        __threadfence_system();
        atomicAdd(&done, 1);
        if (w == 0) {
            while (atomicAdd(&done, 0) < kCopyWarps) {
            }
            __threadfence_system();
            st_release_sys_u32(reinterpret_cast<uint32_t *>(a.peer_sig[dest]) + a.rank * kProbeSigStride,
                               a.epoch);
            atomicAdd(a.done_ctas, 1u);
        }
    } else if (kind == kGin) {
        if (w == 0) {
            ncclGin gin(comm, 0);
            for (unsigned long long off = 0; off < a.bytes_per_dest; off += a.chunk_bytes) {
                const size_t len = static_cast<size_t>(min(static_cast<unsigned long long>(a.chunk_bytes), a.bytes_per_dest - off));
                __threadfence_system();
                gin.put(ncclTeamWorld(comm), dest, win, a.dst_off0 + dst_off + off, win, a.src_off + off, len,
                        ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(sig_data + a.rank), 1}, ncclGin_None{},
                        ncclCoopThread());
            }
            atomicAdd(a.done_ctas, 1u);
        }
    } else if (w == 0) {
        atomicAdd(a.done_ctas, 1u);  // SELF this round: still counts, so the waiter's gate is satisfiable
    }
#endif
}

__global__ void probe_read_gin_signals(const void *comm, int signal_base, int world, unsigned long long *out) {
#if SM120_FP4_GIN_DEVICE
    if (threadIdx.x >= static_cast<unsigned>(world)) return;
    ncclGin gin(*static_cast<const ncclDevComm *>(comm), 0);
    out[threadIdx.x] = gin.readSignal(static_cast<ncclGinSignal_t>(signal_base + threadIdx.x), 64, cuda::memory_order_acquire);
#endif
}

}  // namespace probe

// ------------------------------------------------------------------------------------------------ host
struct State {
    int rank = 0, world = 0, EL = 0, TCAP = 0, H = 0, I = 0, TOPK = 0;
    void *local_alloc = nullptr;  // every peer-visible buffer: one cudaMalloc (P2P) or the NCCL window
    size_t off_pool, off_rflag, off_comb, off_cflag, off_route, off_rtflag;
    size_t st_route, st_row, st_comb, off_rcomb, st_cidx, off_rrow, off_psig, total;  // GIN staging / receive, probe
                                                                           // (window only)
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
    g.off_route = o; o = align_up(o + 2 * size_t(world) * EL * 4);  // two halves, by epoch parity
    g.off_rtflag = o; o = align_up(o + size_t(world) * 4);
    if (window) {
        TORCH_CHECK(EL + world * TCAP * TOPK / kTile <= kMaxTiles,
                    "too many tokens per rank for the NIC transport: local experts + world * tokens * top-k / 8 must be at most ",
                    kMaxTiles);
        g.st_route = o; o = align_up(o + 2 * size_t(world) * EL * 4);
        g.st_row = o; o = align_up(o + size_t(world) * TCAP * TOPK * pool_rx(H));
        g.st_comb = o; o = align_up(o + size_t(world) * TCAP * TOPK * H * 2);
        g.off_rcomb = o; o = align_up(o + size_t(world) * TCAP * TOPK * H * 2);
        g.st_cidx = o; o = align_up(o + size_t(TCAP) * TOPK * 4);
        g.off_rrow = o; o = align_up(o + size_t(world) * TCAP * TOPK * pool_rx(H));
        g.off_psig = o; o = align_up(o + size_t(2) * kMaxWorld * kProbeSigStride * 4);
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
    g.grid[1] = grid_for<1>(sms), g.grid[2] = grid_for<2>(sms), g.grid[4] = grid_for<4>(sms);
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
         double swiglu_limit, int64_t tile_tokens, c10::optional<torch::Tensor> stamps, bool w13_up_first,
         c10::optional<torch::Tensor> logits, c10::optional<torch::Tensor> rbias, c10::optional<torch::Tensor> rbias_alt,
         c10::optional<torch::Tensor> input_ids, int64_t alt_id, c10::optional<torch::Tensor> l2p, bool renorm,
         double eps, bool apply_scale, double rscale) {
    // logits ([S, T, E] fp32): the in-kernel router (Route) writes this call's routes into topk_ids / topk_w first
    Args a{};
    if (logits.has_value()) {
        TORCH_CHECK(logits->scalar_type() == torch::kFloat32 && logits->dim() == 3 && logits->is_contiguous() &&
                        logits->size(1) == x.size(0) && logits->size(2) <= kMaxRouteE,
                    "router logits: [slices, T, E <= 384] fp32");
        TORCH_CHECK(rbias.has_value() && rbias->scalar_type() == torch::kFloat32 && rbias->numel() == logits->size(2));
        TORCH_CHECK(!rbias_alt.has_value() || (rbias_alt->scalar_type() == torch::kFloat32 && input_ids.has_value() &&
                                                input_ids->scalar_type() == torch::kInt64));
        TORCH_CHECK(!l2p.has_value() || (l2p->scalar_type() == torch::kInt32 && l2p->dim() == 2 &&
                                          l2p->size(0) == logits->size(2) && l2p->is_contiguous()));
        Route &r = a.route;
        r.logits = logits->data_ptr<float>();
        r.S = int(logits->size(0)), r.E = int(logits->size(2));
        r.bias = rbias->data_ptr<float>();
        r.bias_alt = rbias_alt.has_value() ? rbias_alt->data_ptr<float>() : nullptr;
        r.input_ids = rbias_alt.has_value() ? input_ids->data_ptr<int64_t>() : nullptr;
        r.alt_id = alt_id;
        r.l2p = l2p.has_value() ? l2p->data_ptr<int>() : nullptr;
        r.R = l2p.has_value() ? int(l2p->size(1)) : 1;
        r.ids = topk_ids.data_ptr<int>();
        r.w = topk_w.data_ptr<float>();
        r.renorm = renorm, r.eps = float(eps), r.apply_scale = apply_scale, r.scale = float(rscale);
    }
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
                    g.st_row, g.st_comb, g.off_rrow, reinterpret_cast<int *>(b + g.st_route), b + g.st_row,
                    b + g.off_rrow, reinterpret_cast<__nv_bfloat16 *>(b + g.st_comb),
                    reinterpret_cast<__nv_bfloat16 *>(b + g.off_rcomb), reinterpret_cast<int *>(b + g.st_cidx)};
    }
    a.T = int(x.size(0));
    TORCH_CHECK(a.T <= g.TCAP, "T exceeds TCAP");
    a.H = g.H, a.I = g.I, a.EL = g.EL, a.W = g.world, a.TOPK = g.TOPK, a.TCAP = g.TCAP, a.rank = g.rank;
    a.swiglu_limit = float(swiglu_limit);
    TORCH_CHECK(!w13_up_first || kRm, "w13_up_first: row-major weights ('w4a8_rm') only");
    a.w13_up_first = int(w13_up_first);
    // tokens per item: 0 = automatic, from the expected tokens per local expert (this rank's T, uniform routing), the
    // best width under uniform routing (6000D EP8, DSV4.1 and README shapes, every W4A8 format): 8 below 8 per expert,
    // 16 from 8 (32 is no faster there: README shape at 96 per rank 331 vs 343 us). Skewed routing favours wider items
    // earlier: sm120_fp4.py's DG_SM120_MEGA_MOE_TILE_SPLITS passes tile_tokens from other thresholds (README.md).
    // 8-token items take the unsynchronized single-N8 paths; W4A8's wider items stage the activations and skip their
    // empty sub-tiles (item_gemm_rm / item_gemm_pk).
    int nt = int(tile_tokens) / kTile;
    if (tile_tokens <= 0) nt = float(a.T) * a.TOPK / a.EL >= 8.f ? 2 : 1;
    TORCH_CHECK((nt == 1 || nt == 2 || nt == 4) && (tile_tokens <= 0 || tile_tokens == nt * kTile),
                "tile_tokens: 8, 16 or 32");
    auto stream = c10::cuda::getCurrentCUDAStream();
    switch (nt) {
        case 1: sm120_fp4_mega_moe_kernel<1><<<g.grid[1], kWarps * 32, 0, stream>>>(a); break;
        case 4: sm120_fp4_mega_moe_kernel<4><<<g.grid[4], kWarps * 32, 0, stream>>>(a); break;
        default: sm120_fp4_mega_moe_kernel<2><<<g.grid[2], kWarps * 32, 0, stream>>>(a); break;
    }
    CHECK_CUDA(cudaGetLastError());
}


// window regions the transport probe uses: [pool (inbox) offset, bytes, GIN staging (source) offset, bytes, probe
// signal slots offset] (window layout, after window_bytes / init_window)
std::vector<int64_t> probe_regions() {
    return {int64_t(g.off_pool), int64_t(g.off_rflag - g.off_pool), int64_t(g.st_comb), int64_t(g.off_rcomb - g.st_comb),
            int64_t(g.off_psig)};
}

// One probe round (see `probe`): `row` [world] transport per destination, `dst_ptrs` / `sig_ptrs` [world] the peers'
// LSA-mapped inbox / probe-signal pointers (0 = unmapped), `gin_targets` / `gin_barrier_targets` CUDA int64 [world]
// cumulative expectations, `scratch` CUDA int64 [3] (go, done, elapsed; zeroed here). Returns {device ms, host ms}.
std::vector<double> probe_round(int64_t comm, int64_t win, int64_t src, int64_t src_off, int64_t dst_off0,
                                std::vector<int64_t> dst_ptrs, std::vector<int64_t> sig_ptrs, int64_t my_sig,
                                std::vector<int64_t> row, int64_t rank, int64_t world, int64_t bytes_per_dest,
                                int64_t dst_slot_bytes, int64_t epoch, int64_t chunk_bytes, torch::Tensor gin_targets,
                                torch::Tensor gin_barrier_targets, torch::Tensor scratch) {
    using namespace probe;
    TORCH_CHECK(SM120_FP4_GIN, "the transport probe needs the GIN build");
    TORCH_CHECK(world >= 2 && world <= kMaxWorld && int64_t(row.size()) == world && int64_t(dst_ptrs.size()) == world &&
                    int64_t(sig_ptrs.size()) == world, "probe arguments do not match world");
    TORCH_CHECK(chunk_bytes > 0 && chunk_bytes <= UINT32_MAX, "probe chunk_bytes out of range");
    ProbeArgs a{};
    a.comm = reinterpret_cast<const void *>(comm);
    a.win = reinterpret_cast<void *>(win);
    a.src = reinterpret_cast<const unsigned char *>(src);
    a.src_off = size_t(src_off), a.dst_off0 = size_t(dst_off0);
    for (int i = 0; i < world; ++i) {
        a.dst_ptr[i] = static_cast<unsigned long long>(dst_ptrs[i]);
        a.peer_sig[i] = static_cast<unsigned long long>(sig_ptrs[i]);
        a.row[i] = int(row[i]);
        TORCH_CHECK(a.row[i] != kLsa || (dst_ptrs[i] && sig_ptrs[i]), "LSA destination ", i, " is not mapped");
    }
    a.my_sig = reinterpret_cast<uint32_t *>(my_sig);
    a.rank = int(rank), a.world = int(world);
    a.bytes_per_dest = static_cast<unsigned long long>(bytes_per_dest);
    a.dst_slot_bytes = static_cast<unsigned long long>(dst_slot_bytes);
    a.epoch = static_cast<uint32_t>(epoch);
    a.chunk_bytes = static_cast<unsigned>(chunk_bytes);
    a.gin_targets = reinterpret_cast<const unsigned long long *>(gin_targets.data_ptr());
    a.gin_barrier_targets = reinterpret_cast<const unsigned long long *>(gin_barrier_targets.data_ptr());
    auto *sc = reinterpret_cast<unsigned long long *>(scratch.data_ptr());
    CHECK_CUDA(cudaMemset(sc, 0, 3 * sizeof(unsigned long long)));
    a.go_flag = reinterpret_cast<unsigned int *>(sc);
    a.done_ctas = reinterpret_cast<unsigned int *>(sc + 1);
    a.elapsed_ns_out = sc + 2;
    const size_t smem = size_t(kCopyWarps) * kStages * kStageBytes + kCopyWarps * kStages * sizeof(uint64_t) + 1024;
    CHECK_CUDA(cudaFuncSetAttribute(probe_send, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventRecord(e0, stream));
    probe_send<<<unsigned(world), 32 * kBlockWarps, smem, stream>>>(a);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(e1, stream));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    cudaError_t status;
    while ((status = cudaStreamQuery(stream)) == cudaErrorNotReady) {
        TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "transport probe round timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK_CUDA(status);
    float host_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&host_ms, e0, e1));
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    unsigned long long ns = 0;
    CHECK_CUDA(cudaMemcpy(&ns, sc + 2, sizeof(ns), cudaMemcpyDeviceToHost));
    return {double(ns) / 1e6, double(host_ms)};
}

// the cumulative GIN signal values [base, base + world) (probe targets start from them)
std::vector<int64_t> probe_read_gin_signals(int64_t comm, int64_t base, int64_t world) {
    TORCH_CHECK(SM120_FP4_GIN, "the transport probe needs the GIN build");
    auto out = torch::zeros({world}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    probe::probe_read_gin_signals<<<1, unsigned((world + 31) / 32 * 32), 0, stream>>>(
        reinterpret_cast<const void *>(comm), int(base), int(world), reinterpret_cast<unsigned long long *>(out.data_ptr()));
    CHECK_CUDA(cudaGetLastError());
    auto h = out.cpu();
    return std::vector<int64_t>(h.data_ptr<int64_t>(), h.data_ptr<int64_t>() + world);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init);
    m.def("open_peers", &open_peers);
    m.def("window_bytes", &window_bytes);
    m.def("init_window", &init_window);
    m.def("grid_size", &grid_size);
    m.def("counters", &counters);
    m.def("run", &run);
    m.def("probe_regions", &probe_regions);
    m.def("probe_round", &probe_round);
    m.def("probe_read_gin_signals", &probe_read_gin_signals);
    m.attr("probe_signal_base") = kSigProbe;
    m.attr("probe_signal_stride") = kProbeSigStride;
    m.attr("max_world") = kMaxWorld;
    m.attr("max_local_experts") = kMaxEl;
    m.attr("max_tiles") = kMaxTiles;
    m.attr("nvfp4") = bool(kNv);
}
