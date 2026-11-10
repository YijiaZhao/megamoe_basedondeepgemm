/**
 * SM120 decode MegaMoE, W4A4 (FP4 weights x FP4 activations) and W4A8 (MXFP4 weights x MXFP8 activations), EP over
 * PCIe P2P, one persistent kernel per rank.
 * Same dataflow as the W4A8 kernel (sm120_mxfp4_mega_moe.cu, unchanged): dispatch (FP4 quant + P2P push) -> FC1 +
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
 * Every spin wait times out after kSpinNs and reports through the error word instead of hanging the GPU.
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
constexpr bool kNv = SM120_FP4_NV;
constexpr bool kA8 = SM120_FP4_A8;
static_assert(!(kNv && kA8));
constexpr int kXB = kA8 ? 1 : 2;       // activation values per byte
constexpr int kSfVec = kNv ? 16 : 32;  // values per block scale
constexpr int kSfWordK = 4 * kSfVec;   // K covered by one u32 of scales
constexpr int kQW = kSfVec / (4 * kXB);  // u32 per quantized scale group

constexpr int kWarps = 8;
constexpr int kNbPerWarp = 1;
constexpr int kRowsPerItem = kWarps * kNbPerWarp * 16;  // 128 weight rows per work item
constexpr int kTile = 8;                                // tokens per MMA (N); an item takes NT of them
constexpr int kMaxNt = 4;
constexpr size_t kPf0Bytes = size_t(64) << 20;          // t=0 L2 prefetch budget over the locally known experts
#ifndef SM120_FP4_L2KEEP_MB
#define SM120_FP4_L2KEEP_MB 110
#endif
constexpr size_t kL2Keep = size_t(SM120_FP4_L2KEEP_MB) << 20;  // active weights up to this size stay in L2 between calls (112 MB)
constexpr int kUnroll = 4;
constexpr int kMaxWorld = 8;
constexpr int kMaxEl = 256;
constexpr uint64_t kSpinNs = 30000000000ull;

constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4;
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4, kCtrEpoch = 5;
constexpr int kCtrFlag = 8, kCtrL1 = 16;  // kCtrFlag: local copies (epoch) of the route / rows / combine flags
constexpr int kCtrInts = kCtrL1 + kMaxEl;

struct Peers {
    uint8_t *pool_x[kMaxWorld];     // [EL][W][TCAP][H/kXB] E2M1 / E4M3  (dest-owned)
    uint32_t *pool_xs[kMaxWorld];   // [EL][W][TCAP][H/kSfWordK] scales
    int *pool_meta[kMaxWorld];      // [EL][W][TCAP] src token*8 + k
    int *recv_flag[kMaxWorld];      // [W] epoch
    __nv_bfloat16 *comb[kMaxWorld]; // [TCAP][TOPK][H]                    (src-owned)
    int *comb_flag[kMaxWorld];      // [W] epoch
    int *route[kMaxWorld];          // [W][EL]: tokens each source rank sends to each expert (dest-owned)
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
    const float *gscale;      // NVFP4: [0] FC1 input, [1] FC2 input, [2, 2+EL) W13, [2+EL, 2+2EL) W2; MXFP4: unused
    uint8_t *inter;           // [EL][W*TCAP][I/kXB] E2M1 / E4M3 (local)
    uint32_t *inter_s;        // [EL][W*TCAP][I/kSfWordK]
    int *ctr;                 // kCtrInts
    unsigned long long *stamps;  // optional [grid][16] globaltimer stamps (debug)
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
// flag store; every caller has fenced its data with __threadfence_system first (a release store costs ~1 us more)
__device__ __forceinline__ void st_release_sys(int *p, int v) { *reinterpret_cast<volatile int *>(p) = v; }
__device__ __forceinline__ bool wait_ge(const int *p, int v, bool sys) {
    const uint64_t t0 = gtime();
    while ((sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) < v)
        if (gtime() - t0 > kSpinNs) return false;
    return true;
}

// Wait for every peer's flag (written over PCIe). One CTA polls them, a thread per peer, and republishes on a local
// word that the other CTAs wait on: with every CTA polling, the loads delay the flags' arrival by microseconds.
__device__ __forceinline__ bool wait_peers(const int *flags, int W, int epoch, int *local, bool poller) {
    if (!poller) return threadIdx.x != 0 || wait_ge(local, epoch, false);
    if (threadIdx.x >= 32) return true;
    const bool ok = int(threadIdx.x) >= W || wait_ge(flags + threadIdx.x, epoch, true);
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
// s*8 + 2c, 2c+1. Each weight fragment is loaded once and feeds NT MMAs. PLAIN: cached weight loads (the weights stay
// in L2 for the next call); otherwise streaming loads, which read faster once the active weights exceed L2.
// xrow[s]: this lane's token row of sub-tile s (K/kXB bytes, as words), xsrow[s]: its scale words (K/kSfWordK).
template <int NT, bool PLAIN>
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
    for (int k0 = 0; k0 < kch; k0 += kUnroll) {
        uint4 wv[kNbPerWarp][kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) wv[j][u] = PLAIN ? wp[j][(k0 + u) * 32] : __ldcs(wp[j] + (k0 + u) * 32);
        uint32_t sfa[kNbPerWarp][kSw];
#pragma unroll
        for (int q = 0; q < kSw; ++q)
#pragma unroll
            for (int j = 0; j < kNbPerWarp; ++j) sfa[j][q] = __ldg(sp[j] + (k0 * kSw / kUnroll + q) * 16);
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
    }
}

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
    int lag;    // queue: FC2 of tile t follows FC1 of tile t + lag
    int plain;  // the active experts' weights fit kL2Keep: cached loads
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
__global__ void __launch_bounds__(kWarps * 32) sm120_fp4_mega_moe_kernel(Args a) {
    constexpr int kT = kTile * NT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
    __shared__ Map m;
    __shared__ int s_item, s_ok, s_epoch;
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
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x)
            a.peers.route[u / EL][a.rank * EL + u % EL] = m.cnt[u / EL][u % EL];
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x < W) st_release_sys(a.peers.route_flag[threadIdx.x] + a.rank, epoch);  // a thread per peer
        st_[8] = gtime();
    }

    // ---------------- t=0: pull the locally known experts' FC1 weights into L2 (budget kPf0Bytes), on the CTAs that
    // do not dispatch (unless few are left): the rows and their flags must not wait for it
    const int nfree = int(gridDim.x) - ndisp;
    const bool pf_all = nfree * 4 < int(gridDim.x);
    if (kPf0Bytes > 0 && EL <= 64 && (pf_all || int(blockIdx.x) >= ndisp)) {
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
            const size_t first = size_t(pf_all ? blockIdx.x : blockIdx.x - ndisp) * blockDim.x + threadIdx.x;
            for (size_t l = first; l < lines; l += size_t(pf_all ? gridDim.x : nfree) * blockDim.x) {
                const int ki = int(l / (per / 128));
                unsigned long long rest = known;
                for (int q = 0; q < ki; ++q) rest &= rest - 1;
                const int el = __ffsll(rest) - 1;
                const char *p = reinterpret_cast<const char *>(a.w13) + el * per + (l % (per / 128)) * 128;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
            }
        }
    }

    // ---------------- dispatch: one CTA per token: quantize once to FP4 in smem, push TOPK copies with every thread
    {
        uint8_t *s_q = reinterpret_cast<uint8_t *>(&s_tile[0][0]);  // [H/kXB] values, then [H/kSfVec] scale bytes
        uint8_t *s_sc = s_q + H / kXB;
        __shared__ int s_rowk[8], s_dstk[8];
        for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
            for (int gi = threadIdx.x; gi < H / kSfVec; gi += blockDim.x) {  // one thread per scale group
                const uint4 *s4 = reinterpret_cast<const uint4 *>(a.x + size_t(t) * H + gi * kSfVec);
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
            if (warp < TOPK) {  // warp k: destination row of copy k (slot = tokens t' < t routed to the same expert)
                const int e = a.topk_ids[t * TOPK + warp];  // < 0: padding token, nothing to send
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
            const int nq = H / kXB / 16, per = nq + H / kSfVec / 16;  // uint4 per copy: data + scales
            for (int u = threadIdx.x; u < TOPK * per; u += blockDim.x) {
                const int k = u / per, q = u % per;
                if (s_rowk[k] < 0) continue;
                const size_t row = size_t(s_rowk[k]);
                const int dst = s_dstk[k];
                if (q < nq)
                    reinterpret_cast<uint4 *>(a.peers.pool_x[dst] + row * (H / kXB))[q] =
                        reinterpret_cast<const uint4 *>(s_q)[q];
                else
                    reinterpret_cast<uint4 *>(a.peers.pool_xs[dst] + row * (H / kSfWordK))[q - nq] =
                        reinterpret_cast<const uint4 *>(s_sc)[q - nq];
            }
            __syncthreads();
        }
    }
    __threadfence_system();
    __syncthreads();
    st_[1] = gtime();
    if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
    __syncthreads();
    if (s_item && threadIdx.x < W) {  // last CTA out of the dispatch: every CTA has fenced its rows -> flag
        st_release_sys(a.peers.recv_flag[threadIdx.x] + a.rank, epoch);
        st_[12] = gtime();
    }
    __syncthreads();

    // ---------------- per-expert map from the exchanged token counts (same slot order as the dispatch)
    const bool poller = blockIdx.x == gridDim.x - 1;
    if (!wait_peers(a.peers.route_flag[a.rank], W, epoch, ctr + kCtrFlag, poller)) {
        atomicExch(ctr + kCtrErr, kErrRouteWait);
        s_ok = 0;
    }
    __syncthreads();
    st_[9] = gtime();
    if (!s_ok) goto exit_kernel;
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = __ldcv(a.peers.route[a.rank] + u);
    __syncthreads();
    if (warp == 0) {
        int tiles_sum = 0, tiles_max = 0, nact = 0;
        for (int base = 0; base < EL; base += 32) {
            const int el = base + lane;
            int n = 0;
            if (el < EL)
                for (int w = 0; w < W; ++w) n += m.cnt[w][el];
            nact += __popc(__ballot_sync(0xffffffffu, n > 0));
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
            m.plain = size_t(nact) * (size_t(3 * I) * H / 2 + size_t(3 * I) * H / kSfVec) <= kL2Keep;
        }
    }
    __syncthreads();
    st_[10] = gtime();
    {  // this CTA's first item is static: pull this warp's share of its weights and scales into L2 now. The item is
        // FC1 while the queue head lasts; with few tokens the later CTAs start on FC2 items, whose weights then arrive
        // while FC1 runs. Skipped when the weights stay in L2 between calls (it would only delay the rows wait)
        const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
        const int ntiles = m.tiles_prefix[EL];
        const int item = first_item(blockIdx.x, gridDim.x);
        if (!m.plain && item < ntiles * (nblk1 + nblk2)) {
            bool l1;
            int tile_g, rb;
            item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
            int el = 0;
            while (m.tiles_prefix[el + 1] <= tile_g) ++el;
            const int K = l1 ? H : I;
            const size_t nb = size_t(el) * (l1 ? 2 * I / 16 : H / 16) + (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
            const char *wb = reinterpret_cast<const char *>((l1 ? a.w13 : a.w2) + nb * (K / 64) * 32);
            const char *sb = reinterpret_cast<const char *>((l1 ? a.ws13 : a.ws2) + nb * (K / kSfWordK) * 16);
            for (int off = lane * 128; off < kNbPerWarp * (K / 64) * 512; off += 32 * 128)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(wb + off));
            for (int off = lane * 128; off < kNbPerWarp * (K / kSfWordK) * 64; off += 32 * 128)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(sb + off));
        }
    }
    st_[11] = gtime();

    // ---------------- wait for every source rank's rows
    if (!wait_peers(a.peers.recv_flag[a.rank], W, epoch, ctr + kCtrFlag + 1, poller)) {
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
        const uint8_t *pool_x = a.peers.pool_x[a.rank];
        const uint32_t *pool_xs = a.peers.pool_xs[a.rank];
        const int *pool_meta = a.peers.pool_meta[a.rank];
        const int g = lane / 4, c = lane % 4;
        const size_t w13_expert = size_t(2 * I / 16) * (H / 64) * 32, ws13_expert = size_t(2 * I / 16) * (H / kSfWordK) * 16;
        const size_t w2_expert = size_t(H / 16) * (I / 64) * 32, ws2_expert = size_t(H / 16) * (I / kSfWordK) * 16;
        const int TOKE = W * TCAP;
        const bool plain = m.plain;

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
                    xr[s] = reinterpret_cast<const uint32_t *>(pool_x + row * (H / kXB));
                    xs[s] = pool_xs + row * (H / kSfWordK);
                }
                if (plain)
                    item_gemm<NT, true>(acc, a.w13 + el * w13_expert, a.ws13 + el * ws13_expert, nb0, H, xr, xs, lane);
                else
                    item_gemm<NT, false>(acc, a.w13 + el * w13_expert, a.ws13 + el * ws13_expert, nb0, H, xr, xs, lane);
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
                if (plain)
                    item_gemm<NT, true>(acc, a.w2 + el * w2_expert, a.ws2 + el * ws2_expert, nb0, I, xr, xs, lane);
                else
                    item_gemm<NT, false>(acc, a.w2 + el * w2_expert, a.ws2 + el * ws2_expert, nb0, I, xr, xs, lane);
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
                // push: token tk of this tile -> its source rank's combine slot, kRowsPerItem bf16 in 16 B stores
                constexpr int kVec = kRowsPerItem / 8;
                for (int u = threadIdx.x; u < ntok * kVec; u += blockDim.x) {
                    const int tk = u / kVec, q = u % kVec;
                    int w, i;
                    token_src(m, el, tile * kT + tk, W, w, i);
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
                if (threadIdx.x == 0) s_item = atomicAdd(ctr + kCtrL2Done, 1) == n2 - 1;
                __syncthreads();
                if (s_item && threadIdx.x < W) {  // this rank's last FC2 item: tell every source rank
                    st_release_sys(a.peers.comb_flag[threadIdx.x] + a.rank, epoch);
                    st_[15] = gtime();
                }
            }
            __syncthreads();
        }
        st_[3] = gtime();
        if (n2 == 0 && blockIdx.x == 0 && threadIdx.x < W)  // no local work: still release every source
            st_release_sys(a.peers.comb_flag[threadIdx.x] + a.rank, epoch);
    }

    // ---------------- combine: wait for every expert rank, out[t] = sum_k w[t][k] * partial[t][k]
    if (!wait_peers(a.peers.comb_flag[a.rank], W, epoch, ctr + kCtrFlag + 2, poller)) {
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

// ------------------------------------------------------------------------------------------------ host
struct State {
    int rank = 0, world = 0, EL = 0, TCAP = 0, H = 0, I = 0, TOPK = 0;
    void *local_alloc = nullptr;  // one cudaMalloc holding every peer-visible buffer
    size_t off_pool_x, off_pool_xs, off_meta, off_rflag, off_comb, off_cflag, off_route, off_rtflag, total;
    void *peer_base[kMaxWorld] = {};
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

// returns this rank's cudaIpcMemHandle bytes
torch::Tensor init(int rank, int world, int EL, int TCAP, int H, int I, int TOPK) {
    TORCH_CHECK(world <= kMaxWorld && EL <= kMaxEl && TOPK <= 8, "world / experts / top-k out of range");
    TORCH_CHECK(H % kRowsPerItem == 0 && I % kRowsPerItem == 0 && H % (16 * kSfVec) == 0 &&
                    H / kXB + H / kSfVec <= kTile * kRowsPerItem * 4,
                "unsupported hidden / intermediate size");
    g.rank = rank, g.world = world, g.EL = EL, g.TCAP = TCAP, g.H = H, g.I = I, g.TOPK = TOPK;
    const size_t rows = size_t(EL) * world * TCAP;
    size_t o = 0;
    g.off_pool_x = o; o = align_up(o + rows * H / kXB);
    g.off_pool_xs = o; o = align_up(o + rows * (H / kSfWordK) * 4);
    g.off_meta = o; o = align_up(o + rows * 4);
    g.off_rflag = o; o = align_up(o + size_t(world) * 4);
    g.off_comb = o; o = align_up(o + size_t(TCAP) * TOPK * H * 2);
    g.off_cflag = o; o = align_up(o + size_t(world) * 4);
    g.off_route = o; o = align_up(o + size_t(world) * EL * 4);
    g.off_rtflag = o; o = align_up(o + size_t(world) * 4);
    g.total = o;
    CHECK_CUDA(cudaMalloc(&g.local_alloc, g.total));
    CHECK_CUDA(cudaMemset(g.local_alloc, 0, g.total));
    CHECK_CUDA(cudaMalloc(&g.inter, rows * I / kXB));
    CHECK_CUDA(cudaMalloc(&g.inter_s, rows * (I / kSfWordK) * 4));
    CHECK_CUDA(cudaMalloc(&g.ctr, kCtrInts * 4));
    CHECK_CUDA(cudaMemset(g.ctr, 0, kCtrInts * 4));
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    const int sms = prop.multiProcessorCount;
    g.grid[1] = grid_for<1>(sms), g.grid[2] = grid_for<2>(sms), g.grid[3] = grid_for<3>(sms), g.grid[4] = grid_for<4>(sms);
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

int grid_size(int nt) { return g.grid[nt]; }

std::vector<int64_t> counters() {
    std::vector<int> h(kCtrInts);
    CHECK_CUDA(cudaMemcpy(h.data(), g.ctr, kCtrInts * 4, cudaMemcpyDeviceToHost));
    return std::vector<int64_t>(h.begin(), h.end());
}

void run(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out, torch::Tensor w13,
         torch::Tensor ws13, torch::Tensor w2, torch::Tensor ws2, c10::optional<torch::Tensor> gscale,
         double swiglu_limit, int64_t tile_tokens, c10::optional<torch::Tensor> stamps) {
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
        uint8_t *b = static_cast<uint8_t *>(g.peer_base[w]);
        a.peers.pool_x[w] = b + g.off_pool_x;
        a.peers.pool_xs[w] = reinterpret_cast<uint32_t *>(b + g.off_pool_xs);
        a.peers.pool_meta[w] = reinterpret_cast<int *>(b + g.off_meta);
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
    // tokens per item: 0 = from the expected tokens per local expert (this rank's T, uniform routing)
    int nt = int(tile_tokens) / kTile;
    if (tile_tokens <= 0) {
        const float per_expert = float(a.T) * a.TOPK / a.EL;
        nt = max(1, min(kMaxNt, int(ceilf(per_expert / 4.f))));  // swept on 6000D EP8: 8 / 16 / 24 / 32 at T 16 / 32 / 64 / 128
    }
    TORCH_CHECK(nt >= 1 && nt <= kMaxNt && (tile_tokens <= 0 || tile_tokens == nt * kTile), "tile_tokens: 8, 16, 24 or 32");
    auto stream = c10::cuda::getCurrentCUDAStream();
    switch (nt) {
        case 1: sm120_fp4_mega_moe_kernel<1><<<g.grid[1], kWarps * 32, 0, stream>>>(a); break;
        case 2: sm120_fp4_mega_moe_kernel<2><<<g.grid[2], kWarps * 32, 0, stream>>>(a); break;
        case 3: sm120_fp4_mega_moe_kernel<3><<<g.grid[3], kWarps * 32, 0, stream>>>(a); break;
        default: sm120_fp4_mega_moe_kernel<4><<<g.grid[4], kWarps * 32, 0, stream>>>(a); break;
    }
    CHECK_CUDA(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init);
    m.def("open_peers", &open_peers);
    m.def("grid_size", &grid_size);
    m.def("counters", &counters);
    m.def("run", &run);
    m.attr("nvfp4") = bool(kNv);
}
