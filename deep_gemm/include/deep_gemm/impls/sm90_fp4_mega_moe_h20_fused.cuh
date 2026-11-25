#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <deep_gemm/impls/fable_cc_select.cuh>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier_fused.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe_fused.cuh>
#include <deep_gemm/layout/h20_ep16_gin.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe_fused.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/quantization/fp4_fused_dequant.cuh>

// Two-node EP16 inside the kernel (deep_gemm/mega/ep16.py, transport 'gin'): the launch is one
// node of 8 NVLink ranks and 192 experts; rows [0, T) are this rank's tokens, rows [T, 2T) the
// rail peer's (same local GPU on the other node), which arrive through the NIC with one NCCL GIN
// put; the peer's combined rows go back the same way, one put per row. NCCL's device headers only
// compile in the device pass.
#ifndef DG_H20_EP16_GIN
#define DG_H20_EP16_GIN 0
#endif
#if DG_H20_EP16_GIN && defined(__CUDA_ARCH__)
#include <nccl_device.h>
#define DG_H20_EP16_GIN_DEVICE 1
#else
#define DG_H20_EP16_GIN_DEVICE 0
#endif

namespace deep_gemm {

namespace nvfp4 {

__device__ __forceinline__ uint2 dequant_mode2_nibble_word(
        const uint32_t packed, const uint2& lut) {
    const uint32_t magnitude_selectors = packed & 0x77777777u;
    uint32_t out_hi =
        byte_perm_unchecked(lut.x, lut.y, magnitude_selectors);
    uint32_t out_lo =
        byte_perm_unchecked(lut.x, lut.y, magnitude_selectors >> 16);
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;"
        : "+r"(out_hi) : "r"(packed));
    const uint32_t shifted = packed << 4;
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;"
        : "+r"(out_lo) : "r"(shifted));
    return make_uint2(out_hi, out_lo);
}

// `kPer32Scale` (MXFP4): one scale byte per 32 K values, i.e. one LUT row per
// 16-byte quad, stored in the first 4 bytes of the scale word. NVFP4 uses two
// scale bytes per quad (per 16 K values).
//
// MXFP4 rows are additionally stored in WGMMA "RF fragment" order (host
// `_mxfp4_rf_fragment_order`): within the 16-byte quad of K32 group g, word c
// (c = 0..3) holds K = g*32 + 4c + {0..3} in its "hi" braid slots and
// K = g*32 + 16 + 4c + {0..3} in its "lo" slots. That is exactly the A
// fragment thread `lane % 4 == c` needs for an m64nNk32 RS WGMMA, so the
// swapAB tiers decode one uint4 per row straight into registers. The SMEM
// tile decoders below gather (q0.x, q1.x, q2.x, q3.x) for K[0..16) and
// (q0.y, q1.y, q2.y, q3.y) for K[16..32) so the decoded tile is unchanged.
// RF-ordered MXFP4 rows are additionally word-transposed on the host
// (`_fused_word_transpose`): 16-byte chunk c holds word c of K32 groups 0..3,
// so the RS loader fetches all four K32 steps of thread `lane % 4 == c` with a
// single uint4. The SS tile decoders load the whole 64-byte row anyway, so
// they undo the transpose in registers (pure renaming after unrolling): after
// this, `quads[g]` holds words c = 0..3 of K32 group g, i.e. word (g, c) is
// read from byte offset c*16 + g*4.
__device__ __forceinline__ void transpose_rf_quads(uint4 (&quads)[4]) {
    const uint4 t0 = quads[0], t1 = quads[1], t2 = quads[2], t3 = quads[3];
    quads[0] = make_uint4(t0.x, t1.x, t2.x, t3.x);
    quads[1] = make_uint4(t0.y, t1.y, t2.y, t3.y);
    quads[2] = make_uint4(t0.z, t1.z, t2.z, t3.z);
    quads[3] = make_uint4(t0.w, t1.w, t2.w, t3.w);
}

template <bool kRFOrder>
__device__ __forceinline__ void store_decoded_quad(
        uint8_t* __restrict__ fp8_dst,
        const uint2& q0, const uint2& q1, const uint2& q2, const uint2& q3,
        const uint32_t k_offset0, const uint32_t k_offset1,
        const uint32_t row_swizzle) {
    if constexpr (kRFOrder) {
        *reinterpret_cast<uint4*>(fp8_dst + (k_offset0 ^ row_swizzle)) =
            make_uint4(q0.x, q1.x, q2.x, q3.x);
        *reinterpret_cast<uint4*>(fp8_dst + (k_offset1 ^ row_swizzle)) =
            make_uint4(q0.y, q1.y, q2.y, q3.y);
    } else {
        *reinterpret_cast<uint4*>(fp8_dst + (k_offset0 ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);
        *reinterpret_cast<uint4*>(fp8_dst + (k_offset1 ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

template <bool kQuadILP = false, bool kPer32Scale = false>
__device__ __forceinline__ void dequant_mode2_nibble_row_regs(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint2& scale_words,
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem) {
#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        uint2 lut0, lut1;
        if constexpr (kPer32Scale) {
            const uint32_t scale =
                (scale_words.x >> (quad_i * 8)) & 0x7fu;
            lut0 = lut_smem[scale];
            lut1 = lut0;
        } else {
            const uint32_t scale_word =
                quad_i < 2 ? scale_words.x : scale_words.y;
            const uint32_t scale0 =
                (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;
            const uint32_t scale1 =
                (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            lut0 = lut_smem[scale0];
            lut1 = lut_smem[scale1];
        }

        const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
        const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
        if constexpr (!kQuadILP && !kPer32Scale) {
            *reinterpret_cast<uint4*>(
                fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }

        const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
        const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
        if constexpr (kPer32Scale) {
            // RF-ordered row: both 16B chunks depend on all four words.
            store_decoded_quad<true>(fp8_dst, q0, q1, q2, q3,
                                     scale_i0 * 16, scale_i1 * 16, row_swizzle);
        } else {
            if constexpr (kQuadILP) {
                *reinterpret_cast<uint4*>(
                    fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                    make_uint4(q0.x, q0.y, q1.x, q1.y);
            }
            *reinterpret_cast<uint4*>(
                fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kQuadILP = false, bool kPer32Scale = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    if constexpr (kPer32Scale)
        transpose_rf_quads(fp4_quads);
    const uint2 scale_words =
        *reinterpret_cast<const uint2*>(row_ptr + 64);
    dequant_mode2_nibble_row_regs<kQuadILP, kPer32Scale>(
        smem_b + row * 128, fp4_quads, scale_words,
        (row & 7u) << 4, lut_smem);
}

// Threads 0-127 and 128-255 each decode one K64 half of the same N128 tile,
// allowing the two M64 warpgroups to reuse the decoded weights.
template <bool kPer32Scale = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble_split_m(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t thread_idx,
        const uint2* __restrict__ lut_smem) {
    const uint32_t row = thread_idx & 127u;
    const uint32_t k_half_idx = thread_idx >> 7;
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80u;
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr + k_half_idx * 32u);
    // NVFP4: 4 scale bytes per K64 half (word `k_half_idx`).
    // MXFP4: 2 scale bytes per K64 half, both halves live in word 0.
    const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(
        row_ptr + 64u + (kPer32Scale ? 0u : k_half_idx * sizeof(uint32_t)));
    uint8_t* __restrict__ fp8_dst = smem_b + row * 128u;
    const uint32_t row_swizzle = (row & 7u) << 4;

    #pragma unroll
    for (uint32_t quad_i = 0; quad_i < 2; ++ quad_i) {
        uint4 q;
        if constexpr (kPer32Scale) {
            // Word-transposed RF row: word c of K32 group g lives at byte
            // c*16 + g*4, so gather the four words of this half's group.
            const uint32_t g = k_half_idx * 2u + quad_i;
            const uint32_t* __restrict__ words =
                reinterpret_cast<const uint32_t*>(row_ptr + g * 4u);
            q = make_uint4(words[0], words[4], words[8], words[12]);
        } else {
            q = fp4_src[quad_i];
        }
        const uint32_t scale_i0 = quad_i * 2u;
        const uint32_t scale_i1 = scale_i0 + 1u;
        uint2 lut0, lut1;
        if constexpr (kPer32Scale) {
            const uint32_t scale =
                (scale_word >> ((k_half_idx * 2u + quad_i) * 8u)) & 0x7fu;
            lut0 = lut_smem[scale];
            lut1 = lut0;
        } else {
            const uint32_t scale0 = (scale_word >> (scale_i0 * 8u)) & 0x7fu;
            const uint32_t scale1 = (scale_word >> (scale_i1 * 8u)) & 0x7fu;
            lut0 = lut_smem[scale0];
            lut1 = lut_smem[scale1];
        }
        const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
        const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
        const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
        const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
        const uint32_t k_offset0 =
            k_half_idx * 64u + scale_i0 * 16u;
        const uint32_t k_offset1 =
            k_half_idx * 64u + scale_i1 * 16u;
        store_decoded_quad<kPer32Scale>(fp8_dst, q0, q1, q2, q3,
                                        k_offset0, k_offset1, row_swizzle);
    }
}

__device__ __forceinline__ uint2 dequant_braided_selector_word(
        const uint32_t braided, const uint2& lut) {
    const uint32_t sel0 = braided & 0x00007777u;
    const uint32_t sel1 = (braided >> 16) & 0x00007777u;
    uint32_t out0 = byte_perm_unchecked(lut.x, lut.y, sel0);
    uint32_t out1 = byte_perm_unchecked(lut.x, lut.y, sel1);
    out0 |= braided & 0x80808080u;
    out1 |= (braided << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

template <bool kRFOrder = false>
__device__ __forceinline__ void dequant_braided_quad(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        const int scale_i0,
        const uint32_t row_swizzle) {
    const uint2 q0 = dequant_braided_selector_word(q.x, lut0);
    const uint2 q1 = dequant_braided_selector_word(q.y, lut0);
    if constexpr (!kRFOrder) {
        *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);
    }

    const uint2 q2 = dequant_braided_selector_word(q.z, lut1);
    const uint2 q3 = dequant_braided_selector_word(q.w, lut1);
    if constexpr (kRFOrder) {
        store_decoded_quad<true>(fp8_dst, q0, q1, q2, q3,
                                 scale_i0 * 16, (scale_i0 + 1) * 16, row_swizzle);
    } else {
        *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

template <bool kRFOrder = false>
__device__ __forceinline__ void dequant_braided_quad_ilp(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        const int scale_i0,
        const uint32_t row_swizzle) {
    // Expose all four independent PRMT chains together so ptxas can overlap
    // their selector/sign work before either 128-bit shared-memory store.
    const uint32_t q0_sel0 = q.x & 0x00007777u;
    const uint32_t q0_sel1 = (q.x >> 16) & 0x00007777u;
    const uint32_t q1_sel0 = q.y & 0x00007777u;
    const uint32_t q1_sel1 = (q.y >> 16) & 0x00007777u;
    const uint32_t q2_sel0 = q.z & 0x00007777u;
    const uint32_t q2_sel1 = (q.z >> 16) & 0x00007777u;
    const uint32_t q3_sel0 = q.w & 0x00007777u;
    const uint32_t q3_sel1 = (q.w >> 16) & 0x00007777u;

    uint32_t q0_out0 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel0);
    uint32_t q0_out1 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel1);
    uint32_t q1_out0 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel0);
    uint32_t q1_out1 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel1);
    uint32_t q2_out0 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel0);
    uint32_t q2_out1 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel1);
    uint32_t q3_out0 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel0);
    uint32_t q3_out1 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel1);

    q0_out0 |= q.x & 0x80808080u;
    q0_out1 |= (q.x << 4) & 0x80808080u;
    q1_out0 |= q.y & 0x80808080u;
    q1_out1 |= (q.y << 4) & 0x80808080u;
    q2_out0 |= q.z & 0x80808080u;
    q2_out1 |= (q.z << 4) & 0x80808080u;
    q3_out0 |= q.w & 0x80808080u;
    q3_out1 |= (q.w << 4) & 0x80808080u;

    store_decoded_quad<kRFOrder>(
        fp8_dst,
        make_uint2(q0_out0, q0_out1), make_uint2(q1_out0, q1_out1),
        make_uint2(q2_out0, q2_out1), make_uint2(q3_out0, q3_out1),
        scale_i0 * 16, (scale_i0 + 1) * 16, row_swizzle);
}

template <int kQuad, bool kQuadIlp, bool kPer32Scale = false>
__device__ __forceinline__ void dequant_braided_quad_lut_window(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint32_t scale_word_lo,
        const uint32_t scale_word_hi,
        const uint2* __restrict__ lut_smem,
        const uint2 lut0,
        const uint2 lut1,
        const uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        if constexpr (kPer32Scale) {
            const uint32_t next_scale =
                (scale_word_lo >> ((kQuad + 1) * 8)) & 0x7fu;
            next_lut0 = lut_smem[next_scale];
            next_lut1 = next_lut0;
        } else {
            constexpr int kNextScaleI0 = (kQuad + 1) * 2;
            constexpr int kNextScaleI1 = kNextScaleI0 + 1;
            const uint32_t next_scale_word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
            const uint32_t next_scale0 =
                (next_scale_word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
            const uint32_t next_scale1 =
                (next_scale_word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
            next_lut0 = lut_smem[next_scale0];
            next_lut1 = lut_smem[next_scale1];
        }
    }

    if constexpr (kQuadIlp) {
        dequant_braided_quad_ilp<kPer32Scale>(
            fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
    } else {
        dequant_braided_quad<kPer32Scale>(
            fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
    }

    if constexpr (kQuad + 1 < 4) {
        dequant_braided_quad_lut_window<kQuad + 1, kQuadIlp, kPer32Scale>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut_smem,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <bool kQuadIlp = false, bool kPer32Scale = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_braided_lut_window(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    if constexpr (kPer32Scale)
        transpose_rf_quads(fp4_quads);

    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint2 lut0 = lut_smem[scale_words.x & 0x7fu];
    const uint2 lut1 = kPer32Scale ? lut0 : lut_smem[(scale_words.x >> 8) & 0x7fu];
    dequant_braided_quad_lut_window<0, kQuadIlp, kPer32Scale>(
        smem_b + row * 128, fp4_quads, scale_words.x, scale_words.y,
        lut_smem, lut0, lut1, (row & 7u) << 4);
}

// QoQ W4A8 (uint4 code, per-row/K128 zero point z and integer s2, per-row
// bf16 s1): decode one 80-byte packed row (64B codes + [s2, z] at bytes 64/65)
// into 128 int8 bytes = (code - z) in [-15, 15]. "SHIFTXOR": spread nibbles
// with shift/mask, then per-byte subtract z with a borrow guard
// ((x | 0x80) - z) ^ 0x80 == (x - z) mod 256 lane-wise. s2 is applied at the
// per-K128 promote, s1 in the epilogue. QoQ rows are stored in the same RF
// fragment order + word transpose as MXFP4 (host `_mxfp4_rf_fragment_order`,
// `_fused_word_transpose`; plain nibbles, no braid): after `transpose_rf_quads`,
// word c of K32 group g holds K = g*32 + 4c + [0..4) in its high nibbles and
// K = g*32 + 16 + 4c + [0..4) in its low nibbles, so the decoded high words of
// c = 0..3 form K[0..16) and the low words K[16..32) of the group. The swapAB
// tiers decode the same words straight into RS A fragments (`decode_stage_rf`).
__device__ __forceinline__ void dequant_smem_b_from_packed_qoq_shiftxor(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80u;
    const uint4* __restrict__ src = reinterpret_cast<const uint4*>(row_ptr);
    const uint32_t zz = static_cast<uint32_t>(row_ptr[65]) * 0x01010101u;
    uint8_t* __restrict__ dst = smem_b + row * 128u;
    const uint32_t row_swizzle = (row & 7u) << 4;
    uint4 quads[4] = {src[0], src[1], src[2], src[3]};
    transpose_rf_quads(quads);
    #pragma unroll
    for (uint32_t g = 0; g < 4; ++ g) {
        const uint4 q = quads[g];
        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
        uint32_t hi[4], lo[4];
        #pragma unroll
        for (uint32_t c = 0; c < 4; ++ c) {
            hi[c] = ((((w[c] >> 4) & 0x0f0f0f0fu) | 0x80808080u) - zz) ^ 0x80808080u;
            lo[c] = (((w[c] & 0x0f0f0f0fu) | 0x80808080u) - zz) ^ 0x80808080u;
        }
        *reinterpret_cast<uint4*>(dst + ((g * 32u) ^ row_swizzle)) =
            make_uint4(hi[0], hi[1], hi[2], hi[3]);
        *reinterpret_cast<uint4*>(dst + ((g * 32u + 16u) ^ row_swizzle)) =
            make_uint4(lo[0], lo[1], lo[2], lo[3]);
    }
}

}  // namespace nvfp4

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    float kActivationClamp,
    bool kFastMath,
    bool kSwapABRequested,
    bool kSingleActiveDispatchWarp,
    bool kUseMode2RowDecoder,
    bool kUseInterleavedScheduler,
    bool kMXFP4 = false,
    uint32_t kPrefetchWeightKBlocks = 0,
    bool kSwapPipelineDecode = true,
    bool kDistributedExpertBcast = true,
    bool kQoQ = false,
    // Weights stored as dense (E, N/BLOCK_N, K/BLOCK_K, BLOCK_N, 80 B) tiles:
    // the B loader streams one 1D bulk copy (BLOCK_N*80 B) per stage from
    // `l{1,2}_weights_ptr` instead of the 2D TMA box (H20: 2D box is
    // TMA-issue bound, ~0.4us/stage; 1D bulk ~0.18us/stage at 7 in flight).
    bool kDenseWeightTiles = false,
    // Host-selected half-tile tasks (128-row tasks + intra-CTA K-split) on the
    // BM8 MXFP4 RF swapAB path; see `kHalfTileTasks` in the body. Ignored by
    // every other tier.
    bool kHalfTileTasksRequested = false,
    // Host-selected L1 split-K tasks (two K halves of one (expert, n_block) task
    // on two SMs, cross-CTA fp32 reduction through the workspace) on the BM8
    // MXFP4 RF swapAB path; see `kSplitKL1` in the body. Ignored by every other
    // tier. Env DG_FP4_SPLITK_L1 (default 1).
    bool kSplitKL1Requested = false,
    // Host-selected L2 half-row tasks (128-row L2 tasks, the two math WGs each own
    // 64 rows over the full K, no cross-WG reduction) on the BM8 MXFP4 RF swapAB
    // path; see `kL2HalfRowTasks` in the body. Ignored by every other tier.
    // Env DG_FP4_L2_HALFROW (default 0: measured slower on H20, see the body).
    bool kL2HalfRowTasksRequested = false,
    // Host-selected L2 split-K tasks (the L2 tasks of the last partial L2 wave run
    // as two K ranges on two SMs, same cross-CTA fp32 reduction as kSplitKL1) on
    // the BM8 MXFP4 RF swapAB path; see `kSplitKL2` in the body. Env DG_FP4_SPLITK_L2
    // (default 0: measured neutral/slower on H20, see the host).
    // 0 off; 2 or 3 = number of K ranges per tail task.
    uint32_t kSplitKL2Ways = 0,
    // Host-selected stream-K (tiny M: the (task, K128 block) units of a phase are
    // split into kNumSMs contiguous ranges, n-way cross-CTA fp32 reduction per
    // tile) on the BM8 MXFP4/QoQ RF swapAB path; see `kStreamK` in the body. Env
    // DG_FP4_STREAMK / DG_FP4_STREAMK_MAX_M.
    bool kStreamKRequested = false,
    // Host-selected fast NVLink-barrier epilogue (SM0 publishes completion through
    // one word instead of a second grid-wide sync); see `kNvlFastEpilogue` in the
    // body. Env DG_FP4_NVL_FAST_EPI (default 0: within noise on H20, see the host).
    bool kNvlFastEpilogueRequested = false,
    // Host-selected fine-grained combine (per-token NVLink arrival counters replace
    // the combine NVLink barrier); see `kFineCombine` in the body. Env
    // DG_FP4_FINE_COMBINE (default 1).
    bool kFineCombineRequested = true,
    // Host-selected push dispatch (tiny M): the source rank writes each routed
    // token row + SF + weight + metadata straight into the destination rank's
    // pool over NVLink during routing (row = remote atomic ticket on the
    // destination's per-expert count), so no pull round trip follows NVLink
    // barrier #1; see `kPushDispatch` in the body. Env DG_FP4_PUSH_DISPATCH /
    // DG_FP4_PUSH_DISPATCH_MAX_M. `kPushMaxTokensPerRank` bounds the rows one rank
    // can send to one expert (== its local token count) and sizes the per-expert
    // pool stride.
    bool kPushDispatchRequested = false,
    uint32_t kPushMaxTokensPerRank = 2,
    // Lean routing (host env DG_FP4_LEAN_ROUTING): see the dispatch prologue in
    // the body (`kLeanRouting` / `kLeanPush`).
    bool kLeanRouting = true,
    // Push DONE flags (host env DG_FP4_PUSH_DONE_FLAGS, default 1; lean push only):
    // NVLink barrier #1 is replaced by one release.sys DONE signal per source rank
    // into every destination's DONE count; see `kPushDoneFlags` in the body.
    bool kPushDoneFlagsRequested = true,
    // QoQ inline s2 (host env DG_FP4_QOQ_INLINE_S2, default 1): fold the per-(row,
    // K128) integer s2 into the int8 weight at RF decode time and accumulate the
    // whole L1 task K range in one int32 set (see `kInlineS2` in the body).
    bool kQoQInlineS2 = false,
    // QoQ inline s2 A-fragment register buffers (host env DG_FP4_QIS2_FRAGS, 2|3|4,
    // default 2): the wgmma.wait_group lag before a fragment buffer is re-decoded
    // is kQoQInlineS2Frags - 1 groups (+32 regs per extra buffer).
    uint32_t kQoQInlineS2Frags = 2,
    // QoQ inline s2 interleaved issue (host env DG_FP4_QIS2_ILV): one commit group per
    // K32 step (2 wgmma) and the next block's K32-step decode between the groups, so
    // the ALU decode runs while the tensor pipe drains instead of after it.
    bool kQoQInlineS2Ilv = false,
    // QoQ inline s2, 2-buffer loop (host env DG_FP4_QIS2_PREFETCH_PACKED): load the
    // next block's packed words before the wgmma wait that frees its fragment buffer.
    bool kQoQInlineS2PrefetchPacked = true,
    // Generic 2-K-block RF loop (host env DG_FP4_RF_PREFETCH_PACKED): k+1 barrier check
    // and next block-0 packed LDS before the wait<1> that frees frag[0].
    bool kRFPrefetchPacked = false,
    // Debug (host env DG_FP4_POOL_STRIDE_DEBUG, pull dispatch only): address the
    // token pool with the push-dispatch fixed per-expert stride while keeping the
    // pull protocol; see `kStridedPool` in the body.
    bool kStridedPoolDebug = false,
    // Wide tasks (host env DG_FP4_L1_BN / DG_FP4_L2_BN = 512, gated by
    // DG_FP4_BN512_MIN_M / DG_FP4_BN512_MAX_M on the global token count): packed
    // 256-row weight tiles per L1 / L2 task (1 or 2). See `kWideTiles` in the body.
    uint32_t kL1TaskTiles = 1,
    uint32_t kL2TaskTiles = 1
>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_nvfp4_mega_moe_h200_fused_impl(
        void* y,
        int* cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<8> sym_buffer,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
        // Raw fused weight bytes; only read when kDenseWeightTiles (MXFP4/QoQ).
        const void* __restrict__ l1_weights_ptr,
        const void* __restrict__ l2_weights_ptr,
        // NVFP4: optional per-expert scale [E].
        // MXFP4: required per-(expert, weight row) scale [E, N] = 2^e_ref * global.
        const float* __restrict__ l1_global_scales,
        const float* __restrict__ l2_global_scales,
        unsigned long long* __restrict__ phase_stamps,
        // DG_FE_SELECT_IN_MEGA=1: the Fable cc frontend's compact [token][384] u32 key array (nullptr = topk from the frontend)
        const uint32_t* __restrict__ fe_keys,
        const __grid_constant__ H20Ep16Gin ep16_gin) {
    // H3072/I1280 (default) or H4096/I1536 (host: SM90FP4H20FusedShape::is_supported_h200_shape)
#ifndef DG_H20_HIDDEN
#define DG_H20_HIDDEN 3072
#endif
#ifndef DG_H20_INTERMEDIATE
#define DG_H20_INTERMEDIATE 1280
#endif
    constexpr uint32_t kHidden = DG_H20_HIDDEN;
    constexpr uint32_t kIntermediateHidden = DG_H20_INTERMEDIATE;
    // Experts across the kNumRanks NVLink ranks of this launch: 384 (EP8), or 192 when
    // the launch is one node of a two-node EP16 (24 per rank; see deep_gemm/mega/ep16.py).
#ifndef DG_H20_NUM_EXPERTS
#define DG_H20_NUM_EXPERTS 384
#endif
    constexpr uint32_t kNumExperts = DG_H20_NUM_EXPERTS;
    constexpr uint32_t kNumTopk = 8;
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumDispatchThreads = 64;
    constexpr uint32_t kNumNonEpilogueThreads = 64;
    constexpr uint32_t kNumEpilogueThreads = 256;
    // Grid size: the device's 78 SMs, or fewer when SMs are kept free for a
    // concurrent kernel (EP16 overlap, see deep_gemm/mega/ep16.py).
#ifndef DG_H20_NUM_SMS
#define DG_H20_NUM_SMS 78
#endif
    constexpr uint32_t kNumSMs = DG_H20_NUM_SMS;
    constexpr uint32_t kNumRanks = 8;
    constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <deep_gemm/impls/sm90_fp4_mega_moe_h20_fused_body.inl>
}


}  // namespace deep_gemm

#pragma clang diagnostic pop
