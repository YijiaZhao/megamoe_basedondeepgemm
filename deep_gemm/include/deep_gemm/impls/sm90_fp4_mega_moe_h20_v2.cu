/**
 * H20 (SM90) decode MegaMoE, the SM120 kernel's dataflow (sm120_fp4_mega_moe.cu) on Hopper: MXFP4 weights x FP8
 * activations, EP over NVLink P2P, one persistent kernel per rank with several independent CTAs per SM.
 * Dataflow: routing counts -> dispatch (FP8 rows + K128 fp32 scales pushed to the owners' expert pools) -> FC1 +
 * SwiGLU (swap-AB mma.sync m16n8k32 e4m3) -> FP8 intermediate (UE8M0 scale per 128) -> FC2 -> bf16 partials pushed
 * back to the source rank -> sum over the top-k. Epoch-tagged flags, no cross-rank barriers.
 *
 * Weights: the fused MXFP4 dense tiles of deep_gemm.transform_mxfp4_weights_for_mega_moe_fused, (E, N/256, K/128,
 * 256 rows, 80 B): 64 B FP4 in RF fragment order + word transpose (16 B chunk c = word c of the four K32 steps),
 * bytes 64..67 the per-K32 relative scale index of the 128-entry LUT (fp4_fused_dequant.cuh); W13 rows interleave
 * 8 gate / 8 up, so a 16-row MMA block holds 8 SwiGLU pairs (acc rows g / g+8); the per-row factor 2^e_ref is applied
 * in the epilogue. The m16n8k32 A fragment of lane (g, c) for K32 step s is exactly word s of chunk c of rows g, g+8,
 * decoded by `dq` (prmt LUT, 7 ALU ops per 8 values).
 * A work item is one 256-row weight tile x one tile of 8 tokens of one local expert. CTA = 2 math warpgroups + 1
 * producer warp, H20V2_CPS CTAs per SM: the producer streams the tile's K128 blocks (20 KB each, cp.async.bulk) into a
 * ring of H20V2_STAGES smem slots together with the K128 block of the item's 8 token rows (one 2D TMA box, 128B
 * swizzle = the wgmma B operand; the rows are first compacted per expert); warpgroup w
 * decodes rows 128w..128w+127 (two m64 tiles) and runs RS wgmma m64n8k32 per K128 block. 16-row block
 * b = 8w + 4j + (warp % 4) of m64 tile j: L1 output columns 8b..8b+7 (the item's 128 = one FC2 input scale group).
 */
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <deep_gemm/quantization/fp4_fused_dequant.cuh>

#define CHECK_CUDA(x)                                                                      \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        TORCH_CHECK(e_ == cudaSuccess, "CUDA error: ", cudaGetErrorString(e_), " @ ", #x); \
    } while (0)

#ifndef H20V2_CPS
#define H20V2_CPS 2  // CTAs per SM
#endif
#ifndef H20V2_STAGES
#define H20V2_STAGES 4  // smem ring slots (one K128 block of a weight tile + of the 8 token rows each)
#endif

#ifndef H20V2_LAG
#define H20V2_LAG 2  // FC1 / FC2 interleave distance in CTA waves of FC1 items (0: every FC1 item first)
#endif
#ifndef H20V2_PFL2
#define H20V2_PFL2 8  // weight blocks of the first item pulled into L2 beyond the ring while the rows are in flight
#endif

constexpr int kWarps = 8;  // math warps; warp kWarps is the weight producer
constexpr int kMathThreads = kWarps * 32;
constexpr int kThreads = kMathThreads + 64;  // + the producer warp (kWarps) and the epilogue warp (kWarps + 1)
constexpr int kStages = H20V2_STAGES;
constexpr int kBlkBytes = 256 * 80;                // one K128 block of a weight tile
constexpr int kStageBytes = kBlkBytes + 1024;       // + the K128 block of the item's 8 token rows
constexpr int kNbPerWarp = 2;
constexpr int kRowsPerItem = kWarps * kNbPerWarp * 16;  // 256 = one packed weight tile
constexpr int kTile = 8;
constexpr int kMaxNt = 1;
constexpr int kMaxT = kTile * kMaxNt;
constexpr int kMaxWorld = 16;
constexpr int kMaxEl = 128;
constexpr int kMaxKb = 32;  // K <= 4096
constexpr int kTopkMax = 8;
constexpr uint64_t kSpinNs = 30000000000ull;

constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4;
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4, kCtrEpoch = 5;
constexpr int kCtrCompact = 6;  // CTAs done compacting the pool rows
constexpr int kCtrFlag = 8, kCtrL1 = 16;
constexpr int kMaxTiles = 2048;               // expert tiles of 8 tokens per call
constexpr int kCtrTileX = kCtrL1 + kMaxEl;     // [tile]: epoch once the tile's rows are compacted (never reset)
constexpr int kCtrInts = kCtrTileX + kMaxTiles;

// a pool row: H E4M3 values, H/128 fp32 scales, then 16 bytes: int source token*8 + k, float top-k weight
__host__ __device__ __forceinline__ int pool_rx(int H) { return H + H / 32 + 16; }

struct Peers {
    uint8_t *pool[kMaxWorld];        // [EL][W][TCAP] pool rows (dest-owned)
    int *recv_flag[kMaxWorld];       // [W] epoch
    __nv_bfloat16 *comb[kMaxWorld];  // [TCAP][TOPK][H] (src-owned)
    int *comb_flag[kMaxWorld];       // [W] epoch
    int *route[kMaxWorld];           // [2][W][EL] by epoch parity (dest-owned)
    int *route_flag[kMaxWorld];      // [W] epoch
};

struct Args {
    const uint8_t *x;          // [T][H] E4M3
    const float *x_sf;         // [T][H/128]
    const int64_t *topk_ids;   // [T][TOPK] global expert ids (< 0: padding)
    const float *topk_w;       // [T][TOPK]
    __nv_bfloat16 *out;        // [T][H]
    const uint8_t *w13, *w2;   // fused dense tiles
    const float *rs13, *rs2;   // [EL][2I], [EL][H] per-row factors
    uint8_t *inter;            // [EL][W*TCAP][I] E4M3
    float *inter_s;            // [EL][W*TCAP][I/128]
    uint8_t *xd;               // [EL][W*TCAP][H] E4M3: the pool rows compacted per expert (expert-local token order)
    float *xsd;                // [EL][W*TCAP][H/128]
    int4 *tinfo;               // [EL][W*TCAP]: source rank, source token*8 + k, top-k weight bits
    int *ctr;
    unsigned long long *stamps;
    Peers peers;
    int T, H, I, EL, W, TOPK, TCAP, rank;
    float clamp;
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
__device__ __forceinline__ void st_flag(int *p, int v) { *reinterpret_cast<volatile int *>(p) = v; }
__device__ __forceinline__ bool wait_ge(const int *p, int v, bool sys) {
    const uint64_t t0 = gtime();
    while ((sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) < v)
        if (gtime() - t0 > kSpinNs) return false;
    return true;
}
// one CTA polls every peer's flag (a thread per peer) and republishes on a local word the others wait on
__device__ __forceinline__ bool wait_peers(const int *flags, int W, int epoch, int *local, bool poller) {
    if (!poller) return threadIdx.x != 0 || wait_ge(local, epoch, false);
    if (threadIdx.x >= 32) return true;
    bool ok = true;
    if (int(threadIdx.x) < W) ok = wait_ge(flags + threadIdx.x, epoch, true);
    if (__all_sync(0xffffffffu, ok) && threadIdx.x == 0) atomicExch(local, epoch);
    return ok;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}
// one packed word (RF order, braided) -> E4M3 bytes of K 4c..4c+3 (x) and 16+4c.. (y), the K32 group's scale folded in
__device__ __forceinline__ uint2 dq(uint32_t p, uint2 lut) {
    const uint32_t m = p & 0x77777777u;
    uint32_t hi = prmt(lut.x, lut.y, m), lo = prmt(lut.x, lut.y, m >> 16);
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(hi) : "r"(p));
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(lo) : "r"(p << 4));
    return make_uint2(hi, lo);
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldcs4(const void *p) {
    uint4 r;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t ldcs1(const void *p) {
    uint32_t r;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void bar_init(uint64_t *b, int n) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n));
}
__device__ __forceinline__ void bar_arrive(uint64_t *b) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t *b, uint32_t ph) {
    asm volatile("{\n .reg .pred P;\n W: mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n @!P bra W;\n}" ::"r"(smem_u32(b)),
                 "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void *dst, const void *src, uint32_t bytes, uint64_t *b) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
// the math warps only (the producer warp runs ahead on the item queue)
__device__ __forceinline__ void math_sync() { asm volatile("bar.sync 1, %0;" ::"n"(kMathThreads) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t &r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float &r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void wgmma_rs(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// D = A B (the accumulator input is ignored: no zeroing moves)
__device__ __forceinline__ void wgmma_rs_first(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// K-major 128B-swizzled B tile (8 token rows of 128 B per K128 block, blocks 1 KB apart)
__device__ __forceinline__ uint64_t b_desc(const void *p) {
    return uint64_t((smem_u32(p) >> 4) & 0x3fff) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}

__device__ __forceinline__ void expect_tx(uint64_t *b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_nx(void *dst, const void *src, uint32_t bytes, uint64_t *b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void tma_2d(void *dst, const CUtensorMap *tm, int c0, int c1, uint64_t *b) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
                     smem_u32(dst)), "l"(tm), "r"(c0), "r"(c1), "r"(smem_u32(b)) : "memory");
}

// warpgroup GEMM over one 256-row item: fin[j] = rows 128*wg + 64*j + 16*(warp % 4) + {g, g+8}, tokens 2c, 2c+1.
// Consumes KB ring stages (weights + B block; stage counter st, advanced by KB), releasing each once its weight words
// are in registers and its wgmmas are done. xs: the item's 8 token scale rows (stride KB).
__device__ __forceinline__ void item_gemm(float (&fin)[2][4], int KB, const uint8_t *ring, uint64_t *full,
                                          uint64_t *empty, int &st, const float *xs, const uint2 *lut, int warp,
                                          int lane) {
    const int g = lane / 4, c = lane % 4, wg = warp / 4, wi = warp % 4;
#pragma unroll
    for (int j = 0; j < 2; ++j) fin[j][0] = fin[j][1] = fin[j][2] = fin[j][3] = 0.f;
    const float *xs0 = xs + (2 * c) * KB;  // this lane's two tokens' K128 scales (xs0[KB]: the second token's)
    const uint32_t row_off = (wg * 128 + wi * 16 + g) * 80;  // + j * 64 rows; + 8 rows
    const char *lutb = reinterpret_cast<const char *>(lut);
    int d = st % kStages;
    uint32_t ph = (st / kStages) & 1;
#pragma unroll 1
    for (int kb = 0; kb < KB; ++kb, ++st) {
        const float sc0 = __ldg(xs0), sc1 = __ldg(xs0 + KB);
        ++xs0;
        bar_wait(full + d, ph);
        const uint8_t *sl = ring + d * kStageBytes;
        uint4 w[2][2];
        uint32_t sw[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint8_t *r0 = sl + row_off + j * 64 * 80;
            w[j][0] = *reinterpret_cast<const uint4 *>(r0 + c * 16);
            w[j][1] = *reinterpret_cast<const uint4 *>(r0 + 640 + c * 16);
            sw[j][0] = *reinterpret_cast<const uint32_t *>(r0 + 64);
            sw[j][1] = *reinterpret_cast<const uint32_t *>(r0 + 704);
        }
        uint32_t af[2][4][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t x0 = q == 0 ? w[j][0].x : q == 1 ? w[j][0].y : q == 2 ? w[j][0].z : w[j][0].w;
                const uint32_t x1 = q == 0 ? w[j][1].x : q == 1 ? w[j][1].y : q == 2 ? w[j][1].z : w[j][1].w;
                // scale bytes hold the LUT index x 8 (byte offset of its uint2 entry; see h20_v2.py)
                const uint2 d0 = dq(x0, *reinterpret_cast<const uint2 *>(lutb + __byte_perm(sw[j][0], 0u, 0x4440u + q)));
                const uint2 d1 = dq(x1, *reinterpret_cast<const uint2 *>(lutb + __byte_perm(sw[j][1], 0u, 0x4440u + q)));
                af[j][q][0] = d0.x, af[j][q][1] = d1.x, af[j][q][2] = d0.y, af[j][q][3] = d1.y;
            }
        float acc[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int i = 0; i < 4; ++i) fence_r(af[j][q][i]);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t desc = b_desc(sl + kBlkBytes);
#pragma unroll
        for (int j = 0; j < 2; ++j) wgmma_rs_first(acc[j], af[j][0], desc);
#pragma unroll
        for (int q = 1; q < 4; ++q)
#pragma unroll
            for (int j = 0; j < 2; ++j) wgmma_rs(acc[j], af[j][q], desc + uint64_t(q * 2));
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) fence_f(acc[j][i]);
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) fence_f(acc[j][i]);
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int i = 0; i < 4; ++i) fence_r(af[j][q][i]);
        __syncwarp();
        if (lane == 0) bar_arrive(empty + d);  // the B block is read by the wgmmas: release after the wait
        if (++d == kStages) d = 0, ph ^= 1u;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fin[j][0] += acc[j][0] * sc0;
            fin[j][1] += acc[j][1] * sc1;
            fin[j][2] += acc[j][2] * sc0;
            fin[j][3] += acc[j][3] * sc1;
        }
    }
}

struct Map {
    int cnt[kMaxWorld][kMaxEl];
    int ne[kMaxEl];
    int tok_prefix[kMaxEl + 1];
    int tiles_prefix[kMaxEl + 1];
    int lag;
};

// the expert whose [prefix[el], prefix[el + 1]) holds v (binary search over EL + 1 ascending entries)
__device__ __forceinline__ int find_expert(const int *prefix, int EL, int v) {
    int lo = 0, hi = EL - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (prefix[mid] <= v) lo = mid; else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ void token_src(const Map &m, int el, int j, int W, int &w, int &i) {
    w = 0;
    i = j;
    while (w < W - 1 && i >= m.cnt[w][el]) {
        i -= m.cnt[w][el];
        ++w;
    }
}

// queue order: FC1 of the first `lag` tiles, then FC1 of tile t + lag followed by FC2 of tile t, then the last FC2s
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
__device__ __forceinline__ int first_item(unsigned block, unsigned grid) { return int(grid - 1 - block); }

template <int NT>
__global__ void __launch_bounds__(kThreads, H20V2_CPS)
    h20_v2_mega_moe_kernel(Args a, const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_inter) {
    static_assert(NT == 1, "8 tokens per item");
    constexpr int kT = kTile * NT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
    __shared__ Map m;
    __shared__ int s_item, s_ok, s_epoch;
    __shared__ uint2 s_lut[128];
    __shared__ uint64_t s_full[kStages], s_empty[kStages];
    __shared__ uint64_t s_mfull[2], s_mempty[2];  // item mailbox: producer -> math warps
    __shared__ int s_mail[2];
    // math -> epilogue warp hand-off, two slots: L1 [tok][128] fp32 SwiGLU outputs / L2 [tok][256] bf16 partials,
    // the tokens' info and the item (expert, tile, rb | l1 << 16, valid tokens; expert < 0: no more items)
    __shared__ __align__(16) uint8_t s_ebuf[2][kTile * kRowsPerItem * 2];
    __shared__ int4 s_einfo[2][kMaxT], s_eitem[2];
    __shared__ uint64_t s_efull[2], s_eempty[2];
    // dynamic: ring of kStages x (weight K128 block 20 KB | token K128 block 1 KB, 128B swizzle)
    extern __shared__ __align__(1024) uint8_t ring[];

    // stamps (thread 0, globaltimer ns): 0 start 1 dispatched 10 map 2 rows in 3 work end 4 combine flags 5 end;
    // per-CTA sums: 6 FC1 wait 8 B tile 9 GEMM 11 epilogue; 7 items, 12 first item, 13 last item end
    unsigned long long st_[16] = {gtime()};
    if (threadIdx.x < 128) s_lut[threadIdx.x] = deep_gemm::nvfp4::kE2M1AndE8M0RelToFp8Lut[threadIdx.x];
    if (threadIdx.x == 0) {
        s_epoch = *reinterpret_cast<volatile int *>(a.ctr + kCtrEpoch) + 1;
        s_ok = 1;
        for (int d = 0; d < kStages; ++d) bar_init(s_full + d, 1), bar_init(s_empty + d, kWarps);
        for (int q = 0; q < 2; ++q) bar_init(s_mfull + q, 1), bar_init(s_mempty + q, kWarps);
        for (int q = 0; q < 2; ++q) bar_init(s_efull + q, 1), bar_init(s_eempty + q, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    int st = 0;  // weight ring stages issued (producer) / consumed (math warps)
    const int epoch = s_epoch;
    int *ctr = a.ctr;

    // ---------------- routing counts first (tiny): every rank learns its experts' load before the rows land
    const int ndisp = min(a.T + 1, int(gridDim.x));
    if (blockIdx.x == ndisp - 1) {
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = 0;
        __syncthreads();
        for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
            const int e = int(a.topk_ids[u]);
            if (e >= 0) atomicAdd(&m.cnt[e / EL][e % EL], 1);
        }
        __syncthreads();
        const int rhalf = (epoch & 1) * W * EL;
        for (int u = threadIdx.x; u < W * EL; u += blockDim.x) {
            const int d = u / EL;
            a.peers.route[d][rhalf + a.rank * EL + u % EL] = m.cnt[d][u % EL];
        }
        __threadfence_system();
        __syncthreads();
        if (int(threadIdx.x) < W) st_flag(a.peers.route_flag[threadIdx.x] + a.rank, epoch);
    }

    // ---------------- dispatch: one CTA per token, every thread pushes 16 B pieces of the row to all TOPK copies
    {
        __shared__ int s_rowk[kTopkMax], s_dstk[kTopkMax];
        const int RX = pool_rx(H), nq = RX / 16 - 1;  // data + scales; the last 16 B: meta
        for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
            if (warp < TOPK) {
                const int e = int(a.topk_ids[t * TOPK + warp]);
                int slot = 0;
                for (int u = lane; u < t * TOPK; u += 32) slot += (int(a.topk_ids[u]) == e);
#pragma unroll
                for (int o = 16; o; o >>= 1) slot += __shfl_xor_sync(0xffffffffu, slot, o);
                if (lane == 0) {
                    if (e < 0) {
                        s_rowk[warp] = -1;
                    } else {
                        const int dst = e / EL, el = e % EL;
                        const int row = (el * W + a.rank) * TCAP + slot;
                        s_rowk[warp] = row, s_dstk[warp] = dst;
                        uint8_t *rec = a.peers.pool[dst] + size_t(row) * RX;
                        const float wk = a.topk_w[t * TOPK + warp];
                        *reinterpret_cast<int4 *>(rec + RX - 16) = make_int4(t * 8 + warp, __float_as_int(wk), 0, 0);
                    }
                }
            }
            __syncthreads();
            const uint4 *xs4 = reinterpret_cast<const uint4 *>(a.x + size_t(t) * H);
            const uint4 *ss4 = reinterpret_cast<const uint4 *>(a.x_sf + size_t(t) * (H / 128));
            for (int q = threadIdx.x; q < nq; q += blockDim.x) {
                const uint4 v = q < H / 16 ? __ldg(xs4 + q) : __ldg(ss4 + q - H / 16);
#pragma unroll
                for (int k = 0; k < kTopkMax; ++k) {
                    if (k >= TOPK || s_rowk[k] < 0) continue;
                    reinterpret_cast<uint4 *>(a.peers.pool[s_dstk[k]] + size_t(s_rowk[k]) * RX)[q] = v;
                }
            }
            __syncthreads();
        }
    }
    __threadfence_system();
    __syncthreads();
    st_[1] = gtime();
    if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
    __syncthreads();
    if (s_item && int(threadIdx.x) < W) st_flag(a.peers.recv_flag[threadIdx.x] + a.rank, epoch);
    __syncthreads();

    // ---------------- per-expert map from the exchanged counts
    const bool poller = blockIdx.x == gridDim.x - 1;
    if (!wait_peers(a.peers.route_flag[a.rank], W, epoch, ctr + kCtrFlag, poller)) {
        atomicExch(ctr + kCtrErr, kErrRouteWait);
        s_ok = 0;
    }
    __syncthreads();
    if (!s_ok) goto exit_kernel;
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x)
        m.cnt[u / EL][u % EL] = __ldcv(a.peers.route[a.rank] + (epoch & 1) * W * EL + u);
    __syncthreads();
    if (warp == 0) {
        int tiles_sum = 0, tiles_max = 0, tok_sum = 0;
        for (int base = 0; base < EL; base += 32) {
            const int el = base + lane;
            int n = 0;
            if (el < EL)
                for (int w = 0; w < W; ++w) n += m.cnt[w][el];
            int incl = (n + kT - 1) / kT, tincl = n;
            tiles_max = max(tiles_max, incl);
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, tincl, o);
                if (lane >= o) tincl += v;
            }
            if (el < EL) m.tok_prefix[el + 1] = tok_sum + tincl;
            tok_sum += __shfl_sync(0xffffffffu, tincl, 31);
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
        if (lane == 0) {
            m.tiles_prefix[0] = 0;
            m.tok_prefix[0] = 0;
            m.lag = H20V2_LAG == 0 ? tiles_sum
                                   : min(tiles_sum, H20V2_LAG * ((int(gridDim.x) + 2 * I / kRowsPerItem - 1) / (2 * I / kRowsPerItem)) +
                                                        tiles_max);
        }
    }
    __syncthreads();
    st_[10] = gtime();
    {
        const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
        const int ntiles = m.tiles_prefix[EL];
        const int n1 = ntiles * nblk1, n2 = ntiles * nblk2;
        const uint8_t *pool = a.peers.pool[a.rank];
        const int RX = pool_rx(H);
        const int g = lane / 4, c = lane % 4;
        const int TOKE = W * TCAP;

        if (warp == kWarps) {  // producer: claims the items, streams weight blocks + token blocks into the ring
            if (lane == 0) {
                int item = first_item(blockIdx.x, gridDim.x), pst = st;
                for (int n = 0;; ++n) {
                    const int q = n & 1;
                    if (n >= 2) bar_wait(s_mempty + q, ((n >> 1) - 1) & 1);
                    s_mail[q] = item;
                    bar_arrive(s_mfull + q);
                    if (item >= n1 + n2) break;
                    bool l1;
                    int tile_g, rb;
                    item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
                    const int el = find_expert(m.tiles_prefix, EL, tile_g);
                    const int tile = tile_g - m.tiles_prefix[el];
                    const int KB = (l1 ? H : I) / 128, N = l1 ? 2 * I : H;
                    const uint8_t *tbase = (l1 ? a.w13 : a.w2) + (size_t(el) * (N / 256) + rb) * size_t(KB) * kBlkBytes;
                    const int row0 = el * TOKE + tile * kT;
                    const CUtensorMap *tm = l1 ? &tm_x : &tm_inter;
                    // token rows: compacted by every CTA (FC1) / written by every FC1 item of the tile (FC2). The
                    // weight block of a stage goes out first, its token block once the rows are ready
                    bool b_ready = false;
                    if (n == 0)  // first item: the blocks past the ring into L2 too (DRAM idles until the rows land)
                        for (int kb = kStages; kb < min(KB, kStages + H20V2_PFL2); ++kb)
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(tbase + size_t(kb) * kBlkBytes),
                                         "r"(uint32_t(kBlkBytes)) : "memory");
                    for (int kb = 0; kb < KB; ++kb, ++pst) {
                        const int d = pst % kStages, use = pst / kStages;
                        if (use > 0) bar_wait(s_empty + d, (use - 1) & 1);
                        uint8_t *dst = ring + size_t(d) * kStageBytes;
                        expect_tx(s_full + d, kStageBytes);
                        bulk_nx(dst, tbase + size_t(kb) * kBlkBytes, kBlkBytes, s_full + d);
                        if (!b_ready) {
                            if (!(l1 ? wait_ge(ctr + kCtrTileX + tile_g, epoch, false)
                                     : wait_ge(ctr + kCtrL1 + el, ((m.ne[el] + kT - 1) / kT) * nblk1, false)))
                                atomicExch(ctr + kCtrErr, kErrL1Wait);
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            b_ready = true;
                        }
                        tma_2d(dst + kBlkBytes, tm, kb * 128, row0, s_full + d);
                    }
                    item = int(gridDim.x) + atomicAdd(ctr + kCtrQueue, 1);
                }
            }
            __syncwarp();
        } else if (warp == kWarps + 1) {
            // ---------------- epilogue warp: L1 -> FP8 FC2 input + scale, L2 -> partials pushed to the source ranks
            int l2_items = 0;
            for (int n = 0;; ++n) {
                const int es = n & 1;
                bar_wait(s_efull + es, (n >> 1) & 1);
                const int4 it = s_eitem[es];
                if (it.x < 0) break;
                const int el = it.x, tile = it.y, rb = it.z & 0xffff, ntok = it.w;
                if (it.z >> 16) {
                    // per token, the item's 128 outputs = one K128 group, UE8M0 scale ceil(log2(amax / 448))
                    const float *buf = reinterpret_cast<const float *>(s_ebuf[es]);
                    for (int tk = 0; tk < ntok; ++tk) {
                        const float4 v4 = reinterpret_cast<const float4 *>(buf + tk * 128)[lane];
                        const float v[4] = {v4.x, v4.y, v4.z, v4.w};
                        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
                        for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                        const int bits = __float_as_int(amax * (1.f / 448.f));
                        int e = amax > 0.f ? ((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0) : -127;
                        e = max(-126, min(127, e));
                        const float inv = __int_as_float((127 - e) << 23), sf = __int_as_float((127 + e) << 23);
                        uint32_t word = 0;
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            word |= uint32_t(__nv_cvt_float_to_fp8(v[q] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * q);
                        const size_t r = size_t(el) * TOKE + tile * kT + tk;
                        reinterpret_cast<uint32_t *>(a.inter + r * I + rb * 128)[lane] = word;
                        if (lane == 0) a.inter_s[r * (I / 128) + rb] = sf;
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");  // FC2 reads these rows with TMA
                    __threadfence();
                    __syncwarp();
                    if (lane == 0) atomicAdd(ctr + kCtrL1 + el, 1);
                } else {
                    // token tk -> its source rank's combine slot, 256 bf16 (16 B per lane)
                    const __nv_bfloat16 *buf = reinterpret_cast<const __nv_bfloat16 *>(s_ebuf[es]);
                    for (int tk = 0; tk < ntok; ++tk) {
                        const int4 inf = s_einfo[es][tk];
                        const uint4 v = reinterpret_cast<const uint4 *>(buf + tk * 256)[lane];
                        __nv_bfloat16 *dst = a.peers.comb[inf.x] + (size_t(inf.y / 8) * TOPK + inf.y % 8) * H +
                                             rb * kRowsPerItem + lane * 8;
                        *reinterpret_cast<uint4 *>(dst) = v;
                    }
                    ++l2_items;
                }
                __syncwarp();
                if (lane == 0) bar_arrive(s_eempty + es);
            }
            // this CTA's partials, then its FC2 count; the CTA that completes n2 tells every source rank
            __threadfence_system();
            __syncwarp();
            int done = 0;
            if (lane == 0) {
                const int old = atomicAdd(ctr + kCtrL2Done, l2_items);
                done = old < n2 && old + l2_items >= n2;
            }
            done = __shfl_sync(0xffffffffu, done, 0);
            if (done && lane < W) st_flag(a.peers.comb_flag[lane] + a.rank, epoch);
        } else {
            // ---------------- wait for every source rank's rows (the producer streams weights meanwhile)
            if (!wait_peers(a.peers.recv_flag[a.rank], W, epoch, ctr + kCtrFlag + 1, poller)) {
                atomicExch(ctr + kCtrErr, kErrDispatchWait);
                s_ok = 0;
            }
            math_sync();
            st_[2] = gtime();
            if (!s_ok) goto exit_kernel;
            // ---------------- compact the pool rows per expert (expert-local token order): the TMA B operand. Tile t
            // (8 rows) goes to CTA t % grid, which then posts the tile's epoch flag; FC1 items of a tile wait for
            // that flag only (producer: before its B TMA; math warps: before the tile's token info / scales)
            {
                const int nq = RX / 16;
                for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
                    const int el = find_expert(m.tiles_prefix, EL, t);
                    const int j0 = (t - m.tiles_prefix[el]) * kT, nrow = min(kT, m.ne[el] - j0);
                    for (int u = threadIdx.x; u < nrow * nq; u += kMathThreads) {
                        const int jr = u / nq, qq = u % nq, j = j0 + jr;
                        int w, i;
                        token_src(m, el, j, W, w, i);
                        const uint4 v = *reinterpret_cast<const uint4 *>(pool + ((size_t(el) * W + w) * TCAP + i) * RX + qq * 16);
                        const size_t dr = size_t(el) * TOKE + j;
                        if (qq < H / 16)
                            reinterpret_cast<uint4 *>(a.xd + dr * H)[qq] = v;
                        else if (qq < nq - 1)
                            reinterpret_cast<uint4 *>(a.xsd + dr * (H / 128))[qq - H / 16] = v;
                        else
                            a.tinfo[dr] = make_int4(w, int(v.x), int(v.y), 0);
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    __threadfence();
                    math_sync();
                    if (threadIdx.x == 0) st_flag(ctr + kCtrTileX + t, epoch);
                }
            }
            int nitems = 0;
            for (int n = 0;; ++n) {
                const int q = n & 1;
                bar_wait(s_mfull + q, (n >> 1) & 1);
                const int item = s_mail[q];
                __syncwarp();
                if (lane == 0) bar_arrive(s_mempty + q);
                if (item >= n1 + n2) break;
                st_[7] += 1;
                unsigned long long ts = gtime();
                if (!st_[12]) st_[12] = ts;
                bool l1;
                int tile_g, rb;
                item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
                const int el = find_expert(m.tiles_prefix, EL, tile_g);
                const int tile = tile_g - m.tiles_prefix[el];
                const int ntok = min(kT, m.ne[el] - tile * kT);
                const int KB = (l1 ? H : I) / 128;
                const int row0 = el * TOKE + tile * kT;
                // the tile's token info / scales: compacted (FC1) / written by every FC1 item of the tile (FC2)
                if (threadIdx.x == 0 && !(l1 ? wait_ge(ctr + kCtrTileX + tile_g, epoch, false)
                                             : wait_ge(ctr + kCtrL1 + el, ((m.ne[el] + kT - 1) / kT) * nblk1, false)))
                    atomicExch(ctr + kCtrErr, kErrL1Wait);
                math_sync();
                // token info for the epilogue warp and this lane's two top-k weights (L1), loaded before the GEMM
                const int4 info = int(threadIdx.x) < kT ? a.tinfo[row0 + threadIdx.x] : make_int4(0, 0, 0, 0);
                const float wk0 = l1 ? __int_as_float(a.tinfo[row0 + 2 * c].z) : 0.f;
                const float wk1 = l1 ? __int_as_float(a.tinfo[row0 + 2 * c + 1].z) : 0.f;
                { const unsigned long long t2 = gtime(); st_[6] += t2 - ts; ts = t2; }
                const float *xs = l1 ? a.xsd + size_t(row0) * (H / 128) : a.inter_s + size_t(row0) * (I / 128);
                float acc[2][4];
                item_gemm(acc, KB, ring, s_full, s_empty, st, xs, s_lut, warp, lane);
                { const unsigned long long t2 = gtime(); st_[9] += t2 - ts; ts = t2; }
                // hand off to the epilogue warp (slot n & 1, free once it finished item n - 2)
                const int es = n & 1;
                if (n >= 2) bar_wait(s_eempty + es, ((n >> 1) - 1) & 1);
                if (l1) {
                    // SwiGLU: block b, rows g (gate) / g+8 (up) -> output col b*8 + g; tokens 2c, 2c+1
                    float *buf = reinterpret_cast<float *>(s_ebuf[es]);
                    const float *rs = a.rs13 + size_t(el) * 2 * I + rb * kRowsPerItem;
#pragma unroll
                    for (int j = 0; j < kNbPerWarp; ++j) {
                        const int b = (warp >> 2) * 8 + j * 4 + (warp & 3), col = b * 8 + g;
                        const float sg = __ldg(rs + b * 16 + g), su = __ldg(rs + b * 16 + 8 + g);
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int tk = 2 * c + h;
                            float v = 0.f;
                            if (tk < ntok) {
                                const float gate = fminf(acc[j][h] * sg, a.clamp);
                                const float up = fminf(fmaxf(acc[j][2 + h] * su, -a.clamp), a.clamp);
                                v = gate / (1.f + __expf(-gate)) * up * (h ? wk1 : wk0);
                            }
                            buf[tk * 128 + col] = v;
                        }
                    }
                } else {
                    __nv_bfloat16 *buf = reinterpret_cast<__nv_bfloat16 *>(s_ebuf[es]);
                    const float *rs = a.rs2 + size_t(el) * H + rb * kRowsPerItem;
#pragma unroll
                    for (int j = 0; j < kNbPerWarp; ++j) {
                        const int b = (warp >> 2) * 8 + j * 4 + (warp & 3);
                        const float s0 = __ldg(rs + b * 16 + g), s1 = __ldg(rs + b * 16 + 8 + g);
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            buf[(2 * c + h) * 256 + b * 16 + g] = __float2bfloat16_rn(acc[j][h] * s0);
                            buf[(2 * c + h) * 256 + b * 16 + 8 + g] = __float2bfloat16_rn(acc[j][2 + h] * s1);
                        }
                    }
                }
                if (int(threadIdx.x) < kT) s_einfo[es][threadIdx.x] = info;
                math_sync();
                if (threadIdx.x == 0) {
                    s_eitem[es] = make_int4(el, tile, rb | (l1 ? 1 << 16 : 0), ntok);
                    bar_arrive(s_efull + es);
                }
                ++nitems;
                st_[13] = gtime();
                st_[11] += st_[13] - ts;
            }
            if (threadIdx.x == 0) {  // no more items for the epilogue warp
                const int es = nitems & 1;
                if (nitems >= 2) bar_wait(s_eempty + es, ((nitems >> 1) - 1) & 1);
                s_eitem[es] = make_int4(-1, 0, 0, 0);
                bar_arrive(s_efull + es);
            }
        }
        st_[3] = gtime();
        if (n2 == 0 && blockIdx.x == 0 && int(threadIdx.x) < W) st_flag(a.peers.comb_flag[threadIdx.x] + a.rank, epoch);
    }

    // ---------------- combine: out[t] = sum_k partial[t][k] (the top-k weight is folded into FC1)
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
                if (a.topk_ids[t * TOPK + k] < 0) continue;
                const uint4 u = *reinterpret_cast<const uint4 *>(comb + (size_t(t) * TOPK + k) * H + h8);
                const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float2 f = __bfloat1622float2(b2[r]);
                    s[2 * r] += f.x;
                    s[2 * r + 1] += f.y;
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
            ctr[kCtrCompact] = 0;
            for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
            __threadfence();
            ctr[kCtrExit] = 0;
        }
    }
}

// ------------------------------------------------------------------------------------------------ host
namespace {
struct State {
    int rank = 0, world = 0, EL = 0, TCAP = 0, H = 0, I = 0, TOPK = 0;
    size_t off_pool = 0, off_rflag = 0, off_comb = 0, off_cflag = 0, off_route = 0, off_rtflag = 0, total = 0;
    void *local_alloc = nullptr;
    void *peer_base[kMaxWorld] = {};
    uint8_t *inter = nullptr;
    float *inter_s = nullptr;
    int *ctr = nullptr;
    uint8_t *xd = nullptr;
    float *xsd = nullptr;
    int4 *tinfo = nullptr;
    CUtensorMap tm_x, tm_inter;
    int grid[kMaxNt + 1] = {};
} g;
}  // namespace

static int dyn_smem(int) { return kStages * kStageBytes; }
// [rows][cols] E4M3, box = 8 rows x 128 B, 128B swizzle (the wgmma B operand of one K128 block)
static CUtensorMap make_rows_map(void *base, int rows, int cols) {
    static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
    if (!enc) {
        cudaDriverEntryPointQueryResult q;
        CHECK_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void **>(&enc), cudaEnableDefault, &q));
        TORCH_CHECK(q == cudaDriverEntryPointSuccess && enc, "cuTensorMapEncodeTiled not found");
    }
    CUtensorMap m;
    const cuuint64_t dims[2] = {cuuint64_t(cols), cuuint64_t(rows)};
    const cuuint64_t strides[1] = {cuuint64_t(cols)};
    const cuuint32_t box[2] = {128, uint32_t(kTile)}, es[2] = {1, 1};
    const CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(r));
    return m;
}
template <int NT>
static int grid_for(int sms, int H) {
    CHECK_CUDA(cudaFuncSetAttribute(h20_v2_mega_moe_kernel<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem(H)));
    int occ = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, h20_v2_mega_moe_kernel<NT>, kThreads, dyn_smem(H)));
    TORCH_CHECK(occ >= 1, "h20_v2: no resident CTA");
    return occ * sms;
}
static size_t align_up(size_t v) { return (v + 255) & ~size_t(255); }

torch::Tensor init(int64_t rank, int64_t world, int64_t EL, int64_t TCAP, int64_t H, int64_t I, int64_t TOPK) {
    TORCH_CHECK(world <= kMaxWorld && EL <= kMaxEl && TOPK <= kTopkMax, "world / experts / top-k out of range");
    TORCH_CHECK(EL + world * TCAP * TOPK / kTile <= kMaxTiles, "too many expert tiles per call");
    TORCH_CHECK(H % kRowsPerItem == 0 && (2 * I) % kRowsPerItem == 0 && H / 128 <= kMaxKb && I / 128 <= kMaxKb,
                "unsupported hidden / intermediate size");
    g.rank = int(rank), g.world = int(world), g.EL = int(EL), g.TCAP = int(TCAP), g.H = int(H), g.I = int(I),
    g.TOPK = int(TOPK);
    const size_t rows = size_t(EL) * world * TCAP;
    size_t o = 0;
    g.off_pool = o; o = align_up(o + rows * pool_rx(int(H)));
    g.off_rflag = o; o = align_up(o + size_t(world) * 4);
    g.off_comb = o; o = align_up(o + size_t(TCAP) * TOPK * H * 2);
    g.off_cflag = o; o = align_up(o + size_t(world) * 4);
    g.off_route = o; o = align_up(o + 2 * size_t(world) * EL * 4);
    g.off_rtflag = o; o = align_up(o + size_t(world) * 4);
    g.total = o;
    CHECK_CUDA(cudaMalloc(&g.local_alloc, g.total));
    CHECK_CUDA(cudaMemset(g.local_alloc, 0, g.total));
    CHECK_CUDA(cudaMalloc(&g.inter, rows * I));
    CHECK_CUDA(cudaMalloc(&g.inter_s, rows * (I / 128) * 4));
    CHECK_CUDA(cudaMalloc(&g.xd, rows * H));
    CHECK_CUDA(cudaMalloc(&g.xsd, rows * (H / 128) * 4));
    CHECK_CUDA(cudaMalloc(&g.tinfo, rows * sizeof(int4)));
    g.tm_x = make_rows_map(g.xd, int(rows), int(H));
    g.tm_inter = make_rows_map(g.inter, int(rows), int(I));
    CHECK_CUDA(cudaMalloc(&g.ctr, kCtrInts * 4));
    CHECK_CUDA(cudaMemset(g.ctr, 0, kCtrInts * 4));
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    g.grid[1] = grid_for<1>(prop.multiProcessorCount, int(H));
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

std::vector<int64_t> grid_sizes() { return {g.grid[1]}; }

std::vector<int64_t> counters() {
    std::vector<int> h(kCtrInts);
    CHECK_CUDA(cudaMemcpy(h.data(), g.ctr, kCtrInts * 4, cudaMemcpyDeviceToHost));
    return std::vector<int64_t>(h.begin(), h.end());
}

void run(torch::Tensor x, torch::Tensor x_sf, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor out,
         torch::Tensor w13, torch::Tensor rs13, torch::Tensor w2, torch::Tensor rs2, double clamp, int64_t tile_tokens,
         c10::optional<torch::Tensor> stamps) {
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt64 && topk_w.scalar_type() == torch::kFloat32 &&
                x_sf.scalar_type() == torch::kFloat32, "topk_ids int64, topk_w / x_sf fp32");
    Args a{};
    a.x = reinterpret_cast<const uint8_t *>(x.data_ptr());
    a.x_sf = x_sf.data_ptr<float>();
    a.topk_ids = topk_ids.data_ptr<int64_t>();
    a.topk_w = topk_w.data_ptr<float>();
    a.out = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    a.w13 = reinterpret_cast<const uint8_t *>(w13.data_ptr());
    a.w2 = reinterpret_cast<const uint8_t *>(w2.data_ptr());
    a.rs13 = rs13.data_ptr<float>();
    a.rs2 = rs2.data_ptr<float>();
    a.inter = g.inter;
    a.inter_s = g.inter_s;
    a.xd = g.xd, a.xsd = g.xsd, a.tinfo = g.tinfo;
    a.ctr = g.ctr;
    a.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long *>(stamps->data_ptr()) : nullptr;
    for (int w = 0; w < g.world; ++w) {
        uint8_t *b = static_cast<uint8_t *>(g.peer_base[w]);
        a.peers.pool[w] = b + g.off_pool;
        a.peers.recv_flag[w] = reinterpret_cast<int *>(b + g.off_rflag);
        a.peers.comb[w] = reinterpret_cast<__nv_bfloat16 *>(b + g.off_comb);
        a.peers.comb_flag[w] = reinterpret_cast<int *>(b + g.off_cflag);
        a.peers.route[w] = reinterpret_cast<int *>(b + g.off_route);
        a.peers.route_flag[w] = reinterpret_cast<int *>(b + g.off_rtflag);
    }
    a.T = int(x.size(0));
    TORCH_CHECK(a.T <= g.TCAP, "T exceeds TCAP");
    a.H = g.H, a.I = g.I, a.EL = g.EL, a.W = g.world, a.TOPK = g.TOPK, a.TCAP = g.TCAP, a.rank = g.rank;
    a.clamp = float(clamp);
    TORCH_CHECK(tile_tokens <= 0 || tile_tokens == kTile, "tile_tokens: 8");
    auto stream = c10::cuda::getCurrentCUDAStream();
    h20_v2_mega_moe_kernel<1><<<g.grid[1], kThreads, dyn_smem(g.H), stream>>>(a, g.tm_x, g.tm_inter);
    CHECK_CUDA(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init);
    m.def("open_peers", &open_peers);
    m.def("grid_sizes", &grid_sizes);
    m.def("counters", &counters);
    m.def("run", &run);
}
