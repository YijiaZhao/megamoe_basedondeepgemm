#pragma once

#include <algorithm>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe_fused.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/h20_ep16_gin.cuh>

#include "../heuristics/sm90_fp4_mega_moe_h20_fused.hpp"

namespace deep_gemm {

class SM90FP4H20FusedRuntime final
    : public LaunchRuntime<SM90FP4H20FusedRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        float activation_clamp;
        bool fast_math;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
        bool use_interleaved_scheduler;
        bool mxfp4;
        int prefetch_weight_k_blocks;
        bool swap_pipeline_decode;
        bool distributed_expert_bcast;
        bool qoq;
        bool dense_weight_tiles;
        bool half_tile_tasks;
        bool split_k_l1;
        bool l2_half_row_tasks;
        uint32_t split_k_l2_ways;  // 0 off, 2 or 3 K ranges per L2 tail task
        bool stream_k;
        bool nvl_fast_epilogue;
        bool fine_combine;
        bool push_dispatch;
        int push_max_tokens_per_rank;
        bool lean_routing;
        bool push_done_flags;
        bool qoq_inline_s2;
        int qoq_inline_s2_frags;
        bool qoq_inline_s2_ilv;
        bool qoq_inline_s2_prefetch_packed;
        bool rf_prefetch_packed;
        bool strided_pool_debug;
        int l1_task_tiles;
        int l2_task_tiles;
        int num_experts;
        int num_sms;
        // Two-node EP16 over NCCL GIN (0: off, else the NCCL GIN type of the rail communicator)
        int ep16_gin_type;
        SM90FP4H20FusedConfig config;

        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const void* l1_weights_ptr;
        const void* l2_weights_ptr;
        const float* l1_global_scales;
        const float* l2_global_scales;
        unsigned long long* phase_stamps;
        const uint32_t* fe_keys;
        H20Ep16Gin ep16_gin;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const char* kernel_symbol = args.qoq ? "sm90_qoq_mega_moe_h20_fused_impl" :
            (args.mxfp4 ? "sm90_mxfp4_mega_moe_h20_fused_impl" :
                          "sm90_nvfp4_mega_moe_h200_fused_impl");
        const std::string kernel_header = fmt::format(
            "#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1\n"
            "#define DG_H20_NUM_EXPERTS {}\n"
            "#define DG_H20_NUM_SMS {}\n"
            "{}"
            "{}"
            "#define sm90_nvfp4_mega_moe_h200_fused_impl {}\n"
            "#include <deep_gemm/impls/sm90_fp4_mega_moe_h20_fused.cuh>",
            args.num_experts,
            args.num_sms,
            get_env<int>("DG_FP4_SPIN_TIMEOUT", 0) != 0 ? "#define DG_FUSED_SPIN_TIMEOUT 1\n" : "",
            // NCCL GIN: keep only the rail communicator's backend (NCCL_GIN_TYPE_PROXY 2, GDAKI 3);
            // the NCCL include directory comes from DG_JIT_EXTRA_FLAGS (deep_gemm/mega/ep16.py)
            args.ep16_gin_type == 0 ? std::string() : fmt::format(
                "#define DG_H20_EP16_GIN 1\n#define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE\n"
                "#define NCCL_GIN_GPI_ENABLE 0\n#define NCCL_GIN_PROXY_ENABLE {}\n#define NCCL_GIN_GDAKI_ENABLE {}\n",
                int(args.ep16_gin_type == 2), int(args.ep16_gin_type == 3)),
            kernel_symbol);
        const std::string policy_template_args = fmt::format(
            "/* kSwapABRequested */ {},\n"
            "        /* kSingleActiveDispatchWarp */ {},\n"
            "        /* kUseMode2RowDecoder */ {},\n"
            "        /* kUseInterleavedScheduler */ {},\n"
            "        /* kMXFP4 */ {},\n"
            "        /* kPrefetchWeightKBlocks */ {},\n"
            "        /* kSwapPipelineDecode */ {},\n"
            "        /* kDistributedExpertBcast */ {},\n"
            "        /* kQoQ */ {},\n"
            "        /* kDenseWeightTiles */ {},\n"
            "        /* kHalfTileTasksRequested */ {},\n"
            "        /* kSplitKL1Requested */ {},\n"
            "        /* kL2HalfRowTasksRequested */ {},\n"
            "        /* kSplitKL2Requested */ {},\n"
            "        /* kStreamKRequested */ {},\n"
            "        /* kNvlFastEpilogueRequested */ {},\n"
            "        /* kFineCombineRequested */ {},\n"
            "        /* kPushDispatchRequested */ {},\n"
            "        /* kPushMaxTokensPerRank */ {},\n"
            "        /* kLeanRouting */ {},\n"
            "        /* kPushDoneFlagsRequested */ {},\n"
            "        /* kQoQInlineS2 */ {},\n"
            "        /* kQoQInlineS2Frags */ {},\n"
            "        /* kQoQInlineS2Ilv */ {},\n"
            "        /* kQoQInlineS2PrefetchPacked */ {},\n"
            "        /* kRFPrefetchPacked */ {},\n"
            "        /* kStridedPoolDebug */ {},\n"
            "        /* kL1TaskTiles */ {},\n"
            "        /* kL2TaskTiles */ {}",
            args.swap_ab ? "true" : "false",
            args.single_active_dispatch_warp ? "true" : "false",
            args.use_mode2_row_decoder ? "true" : "false",
            args.use_interleaved_scheduler ? "true" : "false",
            args.mxfp4 ? "true" : "false",
            args.prefetch_weight_k_blocks,
            args.swap_pipeline_decode ? "true" : "false",
            args.distributed_expert_bcast ? "true" : "false",
            args.qoq ? "true" : "false",
            args.dense_weight_tiles ? "true" : "false",
            args.half_tile_tasks ? "true" : "false",
            args.split_k_l1 ? "true" : "false",
            args.l2_half_row_tasks ? "true" : "false",
            args.split_k_l2_ways,
            args.stream_k ? "true" : "false",
            args.nvl_fast_epilogue ? "true" : "false",
            args.fine_combine ? "true" : "false",
            args.push_dispatch ? "true" : "false",
            args.push_max_tokens_per_rank,
            args.lean_routing ? "true" : "false",
            args.push_done_flags ? "true" : "false",
            args.qoq_inline_s2 ? "true" : "false",
            args.qoq_inline_s2_frags,
            args.qoq_inline_s2_ilv ? "true" : "false",
            args.qoq_inline_s2_prefetch_packed ? "true" : "false",
            args.rf_prefetch_packed ? "true" : "false",
            args.strided_pool_debug ? "true" : "false",
            args.l1_task_tiles,
            args.l2_task_tiles);
        return fmt::format(R"(
{}

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        /* kNumMaxTokensPerRank */ {},
        /* kNumExpertsPerWave */ {},
        /* BLOCK_M */ {},
        /* BLOCK_N */ {},
        /* kNumMaxPoolTokens */ {},
        /* kNumPaddedSFPoolTokens */ {},
        /* kNumStages */ {},
        /* kActivationClamp */ {},
        /* kFastMath */ {},
        {}
    >);
}};
)",
            kernel_header,
            kernel_symbol,
            args.num_max_tokens_per_rank,
            args.config.num_experts_per_wave,
            args.config.block_m,
            args.config.block_n,
            args.config.num_max_pool_tokens,
            args.config.num_padded_sf_pool_tokens,
            args.config.num_stages,
            to_string(args.activation_clamp),
            args.fast_math ? "true" : "false",
            policy_template_args);
    }

    static void launch_impl(
            const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l1_weights_ptr,
            args.l2_weights_ptr,
            args.l1_global_scales,
            args.l2_global_scales,
            args.phase_stamps,
            args.fe_keys,
            args.ep16_gin));
    }
};

static void sm90_fp4_h20_fused_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> l1_global_scales,
    const std::optional<torch::Tensor> l2_global_scales,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& mxfp4 = false,
    const std::optional<torch::Tensor>& phase_stamps = std::nullopt,
    const bool& qoq = false,
    // DG_FE_SELECT_IN_MEGA=1: the Fable cc frontend's compact [token][384] u32 key array
    const std::optional<torch::Tensor>& fe_keys = std::nullopt,
    // Two-node EP16 over NCCL GIN: {ncclDevComm*, ncclWindow_t, window base, node, max own rows,
    // GIN type[, GIN contexts]} (deep_gemm/mega/ep16.py), empty = off. num_tokens is then 2T (own + peer rows).
    const std::vector<int64_t>& ep16_gin = {}
) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int num_experts = num_experts_per_rank * num_ranks;
    const int num_sms = device_runtime->get_num_sms();
    const int num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    // Wide (512-row, two packed tiles) L1 / L2 tasks (kernel `kWideTiles`, template
    // kL1TaskTiles / kL2TaskTiles): DG_FP4_L1_BN / DG_FP4_L2_BN in {256, 512} (default
    // L1 512, L2 256 since the 2026-09-10 A/B below), applied only to launches whose
    // global token upper bound (tokens per rank x ranks) lies in [DG_FP4_BN512_MIN_M,
    // DG_FP4_BN512_MAX_M] (default 16..16: the customer's M=16 = 2 rows per rank;
    // M=2/4/8 (1 row per rank -> bound 8) and M=128/512 are untouched) and to the BM8
    // MXFP4/QoQ RF swapAB dense tier (checked below). At M=16 the BN256 L1 phase is
    // 160 tasks on 78 SMs (2.05 waves + tail split-K) and L2 192 (2.5 waves); a wide
    // task covers two adjacent packed tiles (4 x 64-row halves per WG, one K128 block
    // per stage, RF-loop unit = (K-block, half pair): same registers, 0 spill, same
    // per-K128 promote) so the L1 phase is 80 tasks == one wave + a 2-task tail (80 >
    // 78 SMs) that is split THREE ways (kernel `kNumL1KSplits` == 3 for wide L1).
    // H20 .7 2026-09-10 (tip 8a51244, 8 ranks, 1830 MHz), rank-0 phase-stamp probe, us:
    // the wide L1 task costs 40.9 vs 21.1 (MXFP4; QoQ 38.1 vs 20.6) == 2.0x, i.e. the
    // per-stage cost (1351 ns head-to-head, decode 791/732 + issue 444/350 ns) has NO
    // fixed part to amortise, so the L1 phase is 48.6 vs 46.6 (one 40.9 us wave + tail
    // vs 2.05 waves); the wide L2 task is 14.9 vs 8.2 us and 96 tasks make 2 waves, so
    // L2 wide loses (kernel end 76.5 vs 74.0 MXFP4, 71.9 vs 68.6 QoQ). Customer method
    // (official capture, Mega-only fused M16, GPU0 last-3 median, 5 independent passes
    // interleaved off/on/L1-only, median across passes [skew-free min-over-device
    // median]), MXFP4 / QoQ:
    //   L1 256 L2 256 (off)   84.1 / 80.4   [78.0 / 74.7]
    //   L1 512 L2 512 (on)    83.1 / 80.1   [80.1 / 78.6]
    //   L1 512 L2 256         82.6 / 77.4   [76.5 / 73.7]   <- default
    // L1-only wins on both quants with both methods (-1.5 / -3.0 customer, -1.5 / -1.0
    // skew-free): the L1 phase is neutral, the gain is the shorter L1->L2 dependency
    // chain (one wide L1 task publishes two L2 K128 blocks at once) and 2 fewer
    // straggler tasks. Numerics identical (T=2 -> M16 mxfp4 0.99999 / qoq 0.99993,
    // gate T=8/8/16, mxfp4 128/512, 200-iter graph-replay stress clean).
    // DG_FP4_L1_BN=256 restores the previous schedule; DG_FP4_L2_BN=512 is the
    // measured-worse L2 variant.
    const int wide_bn_l1_env = get_env<int>("DG_FP4_L1_BN", 512);
    const int wide_bn_l2_env = get_env<int>("DG_FP4_L2_BN", 256);
    DG_HOST_ASSERT((wide_bn_l1_env == 256 || wide_bn_l1_env == 512) &&
                   (wide_bn_l2_env == 256 || wide_bn_l2_env == 512));
    const int num_global_tokens_upper = num_tokens * num_ranks;
    const bool wide_m_ok = num_global_tokens_upper >= get_env<int>("DG_FP4_BN512_MIN_M", 16) &&
                           num_global_tokens_upper <= get_env<int>("DG_FP4_BN512_MAX_M", 16);
    const bool wide_request = (mxfp4 || qoq) && wide_m_ok &&
                              (wide_bn_l1_env == 512 || wide_bn_l2_env == 512);
    const SM90FP4H20FusedInput heuristic_input {
        num_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        /* rf_decode */ mxfp4 || qoq,
        /* wide_tiles */ wide_request,
    };
    const auto plan = select_sm90_nvfp4_h200_fused(heuristic_input);
    const auto& config = plan.config;
    using KernelConfig = SM90FP4H20FusedConfig;
    DG_HOST_ASSERT(num_experts_per_rank % config.num_experts_per_wave == 0);
    DG_HOST_ASSERT((config.block_m == 8 || config.block_m == 16 ||
                    config.block_m == 24 || config.block_m == 64 ||
                    config.block_m == 128));
    DG_HOST_ASSERT(config.block_n == 128 || config.block_n == 256);
    DG_HOST_ASSERT(plan.swap_ab == (num_tokens <= 64));
    // MXFP4/QoQ fused weights are packed as dense 256 x BK128 tiles (80 B rows,
    // kSM90FusedPackedTileN) and loaded with one 1D bulk copy per stage. A
    // BLOCK_N=128 kernel tile is one contiguous half of a packed tile.
    const bool dense_weight_tiles = mxfp4 || qoq;
    if (dense_weight_tiles) {
        DG_HOST_ASSERT(256 % config.block_n == 0);
        DG_HOST_ASSERT(l1_weights.is_contiguous() && l2_weights.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(l1_weights.data_ptr()) % 16 == 0);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(l2_weights.data_ptr()) % 16 == 0);
        DG_HOST_ASSERT(l1_weights.size(2) == (hidden / KernelConfig::kBlockK) * kSM90NVFP4BStoragePerKBlock);
        DG_HOST_ASSERT(l2_weights.size(2) == (intermediate_hidden / KernelConfig::kBlockK) * kSM90NVFP4BStoragePerKBlock);
    }
    // QoQ W4A8 is implemented for the swapAB tiers only (<= 64 tokens per rank).
    DG_HOST_ASSERT(!qoq || plan.swap_ab);
    // Wide tasks: BM8 RF swapAB dense tier only (the heuristic already chose one
    // K-block per stage for this launch when `wide_request`).
    const bool wide_tier = wide_request && plan.swap_ab && config.block_m == 8 &&
        config.block_n == 256 && plan.use_interleaved_scheduler && dense_weight_tiles;
    const int l1_task_tiles = (wide_tier && wide_bn_l1_env == 512) ? 2 : 1;
    const int l2_task_tiles = (wide_tier && wide_bn_l2_env == 512) ? 2 : 1;
    const bool wide_tiles = l1_task_tiles > 1 || l2_task_tiles > 1;
    // A wide request only changes the heuristic's BM8 stage depth, so a non-BM8 plan
    // (larger M) simply ignores it; a BM8 plan of a MXFP4/QoQ host always qualifies.
    DG_HOST_ASSERT(!wide_request || wide_tier || config.block_m != 8);

    // Half-tile tasks (kernel `kHalfTileTasks`, master gate
    // fused_layout::kSM90FusedHalfTileTasks): the BM8 MXFP4 RF swapAB tier
    // schedules 128-row tasks (intra-CTA K-split), so the L1 output store box and
    // the L2 activation-scale granularity follow the 128-row task N.
    // H20 A/B (2026-09-09): wins only when the L1 tasks fit one wave (M=2 global
    // tokens: 51 -> 44.5 us); at M=8/16 the per-K-block RF loop cost does not halve
    // with the task, so it loses (65 -> 68.6, 90 -> 101 us). Default OFF; the token
    // count per rank cannot tell M=2 from M=8 (both 1 token/rank), so this is an
    // explicit knob rather than a tier rule.
    const bool half_tile_tasks = fused_layout::kSM90FusedHalfTileTasks &&
        mxfp4 && plan.swap_ab && config.block_m == 8 && !wide_tiles &&
        get_env<int>("DG_FP4_HALF_TILE", 0) != 0;
    // L1 split-K tasks (kernel `kSplitKL1`): the BM8 MXFP4 RF swapAB tier claims
    // the L1 tasks of the last partial L1 wave as two K halves on two SMs (cross-CTA fp32
    // reduction through fused_layout::Workspace scratch, 5 MB, bounded to
    // kSM90SplitKL1MaxPoolBlocks pool blocks; larger launches fall back in-kernel).
    // The per-WG math shape is unchanged, so unlike half-tile tasks the per-task
    // latency really halves. H20 A/B (2026-09-09, phase stamps, kernel end us,
    // ON vs same-day OFF): M=2 47.4-48.9 vs 50.7; M=8 62.8-64.9 vs 65.3; M=16
    // 86.7-88.6 vs 89.2 (L1 phase -4.5 / -12 us at M=8 / 16, partly given back to
    // a longer L2 tail). Default ON for that tier; DG_FP4_SPLITK_L1=0 disables.
    // Exclusive with half-tile tasks. No effect on TMA boxes / SF granularity.
    const bool split_k_l1 = (mxfp4 || qoq) && plan.swap_ab && config.block_m == 8 &&
        !half_tile_tasks && plan.use_interleaved_scheduler &&
        get_env<int>("DG_FP4_SPLITK_L1", 1) != 0;
    // L2 half-row tasks (kernel `kL2HalfRowTasks`): the BM8 MXFP4 RF swapAB tier
    // schedules 128-row L2 tasks (24 per M block instead of 12) and the two math
    // WGs each own 64 of those rows over the full K (own accumulators + epilogue,
    // no reduction), so the per-WG per-stage work and the L2 task latency halve.
    // Targets the post-split-K critical tail (L2 tasks quantising on 78 SMs).
    // L1 tasks / TMA boxes / SF granularity are untouched. Exclusive with
    // half-tile tasks.
    // H20 A/B (2026-09-09, 8 ranks, phase stamps, back-to-back ON vs OFF): the L2
    // task does NOT get shorter with half the rows per WG (per-task probe 6.5 us
    // both ways: the RF stage chain is latency-bound, as the half-tile experiment
    // already found per K-block), so doubling the task count doubles the L2 tail
    // (M16 last L2 - last L1: 26 vs 10.6 us) and the ON build also slows the L1
    // task (29 vs 22.7 us; one ON cubin spills, STACK 56). Kernel end M8 73.3-73.9
    // vs 62.9-63.8, M16 115.8-116.0 vs 87.1-99.9. Default OFF; DG_FP4_L2_HALFROW=1
    // enables it (numerics verified: T=2/8/16/128/512 + QoQ pass).
    const bool l2_half_row_tasks = mxfp4 && plan.swap_ab && config.block_m == 8 && !wide_tiles &&
        !half_tile_tasks && plan.use_interleaved_scheduler && dense_weight_tiles &&
        get_env<int>("DG_FP4_L2_HALFROW", 0) != 0;
    // L2 split-K tasks (kernel `kSplitKL2`): the L2 tasks of the last L2 claim
    // batch (the batch that runs after the last L1 wave: (num_l2 - first batch)
    // % 78, M=8: 22 of 96, M=2: all 24; M=16: 44 -> 88 halves do not fit, unsplit)
    // run as two K-range halves (K-blocks [0, 4) publisher / [4, 10) finisher) with
    // the kSplitKL1 protocol on separate scratch slots (workspace +6 MB).
    // H20 A/B (2026-09-09, 3 interleaved reps, rank-0 stamps): NO gain. The L2 tail
    // (last L2 - last L1) is a dependency-latency chain after the last L1 notify
    // (poll + TMA + last stage + epilogue + NVLink scatter, ~5 us at M=8) plus, at
    // M=16, two full L2 waves gated on the last L1 wave (122 tasks / 78 SMs), and
    // a 2-way K split shortens neither: M=8 tail 4.7 vs 4.9 us (kernel end 63.9 vs
    // 64.4), M=16 11.6 vs 10.7 (88.9 vs 87.3), M=2 7.8 vs 6.6 (50.8 vs 48.2).
    // 2026-09-10 per-CTA task log (M=8, rank 0): the tail is NOT the dependency
    // chain but occupancy: 78 CTAs each run one 18.2 us L1 task, then 96 L2 tasks of
    // 7.1 us over 78 SMs put a second full L2 task on 22 CTAs (46.8 -> 53.7 us) while
    // 56 CTAs idle ~7 us; the 2-way split leaves a 3-stage finisher (~4.3 us), hence
    // ~0.7 us. DG_FP4_SPLITK_L2=3: the tail tasks run as THREE stage-aligned K ranges
    // (1/2/2 stages, the finisher last and longest, n-1 publisher slots per task),
    // 66 segments <= 78 SMs at M=8 (M=16: 44 * 3 > 78, unsplit as before).
    // DG_FP4_SPLITK_L2=1|2 keeps the 2-way split; 0 off. Numerics verified with =3
    // (T=2/8/8/16 cos_min 0.99998-0.99999, 128/512 0.99988, QoQ 8 0.99993).
    // H20 2026-09-10, =3 vs 0, skew-free (DG_PROFILE_HOST_BARRIER=1) nsys
    // min-over-devices, 2 passes: rank-0 last-L2 - last-L1 does drop 7.2/6.5 ->
    // 5.8/5.6 us at M=8, but the kernel gets SLOWER: MXFP4 M8 54.4/54.3 -> 55.8/55.8,
    // M2 41.9/41.4 -> 43.0/42.8, M16 77.7/77.8 -> 79.6/80.6 (unsplit there, 44 x 3 >
    // 78), QoQ M8 54.1/53.9 -> 54.0/53.9 (knob is MXFP4-only). The stamps show why:
    // after the last L2 the critical path is the combine (slot 6 -> 7 = 3.7-4.3 us:
    // ONE warp on SM0 sums the token's 8 slices after SM0's own L2 tasks), and the
    // 3-way reduce lengthens the finisher epilogue (1.8 -> 2.6 us) without moving
    // the combine end. Default stays 0. Exclusive with L2 half-row tasks.
    const int split_k_l2_env = get_env<int>("DG_FP4_SPLITK_L2", 0);
    const bool split_k_l2 = mxfp4 && plan.swap_ab && config.block_m == 8 && !wide_tiles &&
        !half_tile_tasks && !l2_half_row_tasks && plan.use_interleaved_scheduler &&
        dense_weight_tiles && split_k_l2_env != 0;
    const uint32_t split_k_l2_ways = split_k_l2_env >= 3 ? 3u : 2u;
    // QoQ inline s2 (kernel `kQoQInlineS2`, DG_FP4_QOQ_INLINE_S2 default 1) on every QoQ swapAB
    // launch. The 2026-09-11 T=32 / T=64 failures (rows 8.. of a BM16 / BM24 block with > 8 valid
    // rows got token group 0's sums) were a `promote_task_rf` accumulator index (kernel, fixed).
    const bool qoq_inline_s2 = qoq && get_env<int>("DG_FP4_QOQ_INLINE_S2", 1) != 0;
    // Stream-K (kernel `kStreamK`): for tiny M the (task, K128 block) units of the
    // L1 and of the L2 phase are split into 78 contiguous near-equal unit ranges (one
    // per SM, task-major) with an n-way cross-CTA fp32 reduction per tile through
    // the split-K scratch, replacing the wave scheduler and both tail splits.
    // DG_FP4_STREAMK=1 (default 1 since 2026-09-10, see below) requests it when the
    // launch's global token count upper bound (num_tokens per rank x ranks; every
    // 1-token/rank launch, i.e. the customer's M=2 and M=4 as well as M=8, gives 8)
    // is <= DG_FP4_STREAMK_MAX_M (default 8), so M=128/512 launches are untouched.
    // The host cannot tell M=2/4 from M=8 (a MAX_M of 4 would switch stream-K off
    // exactly where it wins); the scheduler's in-kernel gate does that: it activates
    // stream-K only when the launch has fewer L1 tasks than SMs (idle SMs to fill),
    // which holds for M=2 (16-20 tasks) and M=4 (~40) but not for M=8 (80 tasks) or
    // M=16, which stay on the wave scheduler + tail split-K even with the knob on.
    // H20 A/B (2026-09-09, 8 ranks, rank-0 phase stamps, skew-corrected kernel end =
    // slot 7 - slot 16, ON x2 / OFF, TINYM off):
    //   M=2  L1 phase 13.4/12.7 vs 15.6 us, L2 tail 6.9/7.0 vs 8.1, end 36.4/35.8 vs 38.7
    //   M=8  L1 25.7/25.2 vs 29.5, L2 14.0/14.2 vs 7.1, end 55.6/55.8 vs 45.6 (worse)
    //   M=16 L1 52.3/52.1 vs 46.7, L2 23.7/23.3 vs 12.7, end 88.1/86.9 vs 67.2 (worse)
    // Per-task probe: a 6-K-block L1 segment costs 12.5 us (M=2; ~8 us fixed + 0.7
    // us/block), a full 24-block task 23-25 us, so with >= 1 full L1 wave the L1
    // phase is bounded by the per-SM stage chain (~1 us/K-block incl. fills), not
    // by idle SMs, and running every CTA's L1 range before its L2 range removes the
    // L1/L2 overlap of the wave scheduler (L2 tail doubles). Hence the in-kernel gate.
    // Re-validated on the push-dispatch-default tip (2026-09-10, 780a9f1): corr T=2/8/8/16
    // mxfp4 0.99998 / qoq 0.99993, owner-rank M=2/M=4 launches (stream-K active) mxfp4
    // 0.999997 / qoq 0.99994 (== knob 0 to 1e-7), 200-iter graph-replay stress M=2/M=4 clean.
    // CUSTOMER METHOD (official capture, GPU0 median of the last 3 spans, 1830 MHz),
    // 5 independent captures per knob, median [min..max] across captures, us; a
    // capture's number is kept even when its last-3 start skew is > 20 us (the median
    // absorbs it; both knobs had 1-2 such captures per point):
    //   Mega-only fused          knob 0                     knob 1
    //     MXFP4 M2   49.70 [47.62..89.41]   42.94 [39.17..96.54]   -6.75
    //     MXFP4 M4   60.77 [48.22..83.14]   57.41 [49.38..98.40]   -3.36
    //     QoQ   M2   45.12 [44.54..47.20]   44.06 [37.66..99.39]   -1.06
    //     QoQ   M4   48.26 [46.53..93.28]   48.03 [46.82..58.72]   -0.22
    //   E2E fused (frontend + mega, graph)
    //     MXFP4 M2   84.51 [75.36..175.58]  87.87 [84.29..90.46]   +3.36
    //     MXFP4 M4   85.41 [76.93..91.20]   90.56 [77.15..94.37]   +5.15
    //     QoQ   M2   84.22 [74.27..86.88]   88.90 [76.32..108.29]  +4.67
    //     QoQ   M4   83.10 [75.30..84.83]   84.45 [76.74..132.45]  +1.34
    //   M=8 gate check (wave path), knob 1 x3 captures (p1/p2/p3, median) vs the two
    //   official knob-0 matrices on the same kernel (fuse_customer_k0_p1/p2):
    //     Mega MXFP4 101.9(skew 48)/58.6/68.96 -> 68.96  vs 62.3/62.9
    //     Mega QoQ    62.1/54.7/56.96          -> 56.96  vs 72.2/60.4
    //     E2E  MXFP4  78.8/89.8/78.8           -> 78.8   vs 92.4/78.3
    //     E2E  QoQ    86.3/77.8/82.9           -> 82.9   vs 87.8/81.2
    //   i.e. within the capture-to-capture spread: M=8 is unchanged by the knob.
    // Default flipped to 1 on the Mega-only medians (wins at both M for both quants);
    // note the E2E fused medians moved the other way (+1.3..+5.2 us, 4/4 points) with
    // the same captures -- to be understood before the E2E number is quoted with the
    // knob on. DG_FP4_STREAMK=0 restores the wave scheduler at every M.
    const bool stream_k = (mxfp4 || qoq) && plan.swap_ab && config.block_m == 8 && !wide_tiles &&
        !half_tile_tasks && !l2_half_row_tasks && plan.use_interleaved_scheduler &&
        dense_weight_tiles && get_env<int>("DG_FP4_STREAMK", 1) != 0 &&
        num_global_tokens_upper <= get_env<int>("DG_FP4_STREAMK_MAX_M", 8);
    // Push dispatch (kernel `kPushDispatch`): for <= DG_FP4_PUSH_DISPATCH_MAX_M
    // (default 16) global tokens the source rank pushes each routed row (3 KB token +
    // per-K128 SF + top-k weight + source metadata) into the destination rank's pool
    // during routing (row position = remote atomic ticket on the destination's
    // per-expert count), so the post-barrier-#1 pull round trip disappears and the
    // first math task starts right after the barrier. The pool is addressed with a
    // fixed stride of ceil(num_ranks * tokens_per_rank / BLOCK_M) blocks per local
    // expert (2 at M <= 16), which must fit the token pool and the split-K / tiny-M
    // slot count (kSM90SplitKL1MaxPoolBlocks); larger launches keep the pull path.
    // Default ON (DG_FP4_PUSH_DISPATCH=0 = pull). H20 A/B (2026-09-09, phase stamps,
    // skew-corrected kernel end = slot 7 - slot 16, us, push x2 vs pull): mxfp4 M=2
    // 32.9/32.6 vs 37.1, M=8 41.1/42.8 + 42.9/42.9 vs 43.3/44.2, M=16 62.6/63.8 vs
    // 67.0; qoq M=8 41.2/41.0 vs 42.3/43.5. The win needed three fixes on top of the
    // protocol: one routed row per dispatch warp (the 4-token packing serialised the
    // remote stores: routing +5 us at M=16), no per-task arrival spin (release/acquire
    // on the count completeness word), and no weight prefetch (it competed with the
    // stage fills once the pool wait dropped from ~7 to ~2 us). The pool layout
    // itself is free (DG_FP4_POOL_STRIDE_DEBUG=1 under pull: end within 0.1-1 us).
    const int push_max_m = get_env<int>("DG_FP4_PUSH_DISPATCH_MAX_M", 16);
    const int push_max_tokens_per_rank = std::max(1, (push_max_m + num_ranks - 1) / num_ranks);
    const int push_blocks_per_expert =
        (num_ranks * push_max_tokens_per_rank + config.block_m - 1) / config.block_m;
    const bool push_dispatch = plan.use_interleaved_scheduler &&
        get_env<int>("DG_FP4_PUSH_DISPATCH", 1) != 0 &&
        num_global_tokens_upper <= push_max_m && num_tokens <= push_max_tokens_per_rank &&
        num_experts_per_rank * push_blocks_per_expert * config.block_m <= config.num_max_pool_tokens &&
        num_experts_per_rank * push_blocks_per_expert <=
            static_cast<int>(fused_layout::kSM90SplitKL1MaxPoolBlocks);
    // Debug knob (kernel `kStridedPoolDebug`, DG_FP4_POOL_STRIDE_DEBUG=1): keep the
    // PULL protocol but address the pool with the push fixed per-expert stride, to
    // isolate the layout's cost from the push protocol's in the phase-stamp probe.
    // Same fit conditions as push dispatch.
    const bool strided_pool_debug = !push_dispatch && plan.use_interleaved_scheduler &&
        get_env<int>("DG_FP4_POOL_STRIDE_DEBUG", 0) != 0 &&
        num_global_tokens_upper <= push_max_m && num_tokens <= push_max_tokens_per_rank &&
        num_experts_per_rank * push_blocks_per_expert * config.block_m <= config.num_max_pool_tokens &&
        num_experts_per_rank * push_blocks_per_expert <=
            static_cast<int>(fused_layout::kSM90SplitKL1MaxPoolBlocks);
    // Push DONE flags (kernel `kPushDoneFlags`, DG_FP4_PUSH_DONE_FLAGS, default 1, lean
    // push only): NVLink barrier #1 (78-CTA grid sync -> SM0 sys-scope signal to 8 ranks
    // -> SM0 wait -> grid-wide completion) is replaced by a per-rank DONE count: the
    // last CTA of a rank to finish its pushes (atomic CTA arrival ticket) red.release.sys-
    // adds 1 into every rank's DONE word; the destination's task producers and SM e's
    // count publisher wait for 8 arrivals with ld.acquire.sys (target = ranks * (launch
    // epoch + 1), epoch bumped in the workspace cleanup). Same information as barrier
    // #1 (all of a rank's rows and tickets landed) with one NVLink hop instead of a
    // grid sync + hop + grid sync, and no SM0 serialisation. The kNumRanks signals
    // are issued by kNumRanks lanes at once: serially from one thread each
    // release.sys drained the SM's NVLink stores (~1.5 us) and put +8 us on the
    // kernel. H20 A/B (2026-09-09, phase-stamp probe under nsys, skew-free
    // min-over-devices kernel duration, us, knob 1 vs 0, same session): mxfp4 M=2
    // 40.3 vs 41.2, M=8 53.3 vs 54.4, M=16 75.3 vs 76.4, qoq M=8 53.0 vs 54.1 (-1 us:
    // rank-0 push issued -> data complete 2.9 vs 3.7 us, first math 8.2 vs 9.6).
    // Numerics unchanged (mxfp4/qoq T=2/8/8/16 x2, mxfp4 128/512, 200-iter
    // graph-replay stress).
    const bool push_done_flags = push_dispatch &&
        get_env<int>("DG_FP4_LEAN_ROUTING", 1) != 0 &&
        get_env<int>("DG_FP4_PUSH_DONE_FLAGS", 1) != 0;
    // Fast NVLink-barrier epilogue (kernel `kNvlFastEpilogue`, needs the
    // distributed expert bcast so the first barrier has a prologue grid sync):
    // the two barriers with an epilogue (before dispatch pull, before combine)
    // replace their second grid-wide sync with one SM0-written completion word.
    // H20 A/B (2026-09-09, 3 interleaved reps): within noise (the epilogue grid
    // sync is cheap when SM0 arrives last: combine-barrier segment 5.8 vs 6.3 us at
    // M=8, 5.5 vs 4.8 at M=16; kernel end M=8 64.0 vs 64.4, M=16 87.3 vs 87.3,
    // M=2 47.3 vs 48.2). Default OFF; DG_FP4_NVL_FAST_EPI=1 enables (numerics
    // verified: T=2/8/16/128/512 + QoQ pass).
    const bool nvl_fast_epilogue = get_env<int>("DG_FP4_NVL_FAST_EPI", 0) != 0;
    // Fine-grained combine (kernel `kFineCombine`): the combine NVLink barrier is
    // replaced by per-token arrival counters (each L2 task red.release.sys-adds 1
    // per scattered token row into the destination rank's counter; the combine
    // warp of a token spins on its own counter), so a rank's combine overlaps the
    // other ranks' L2 tails. The sys-scope release fence is issued by dispatch warp
    // 0 through a per-CTA mailbox (a fence.sys in the math warps cost ~1.5-2 us per
    // L2 task on H20 and pushed the last L2 task end out by 2-4 us at M=8/16).
    // H20 A/B (2026-09-09, phase stamps rank 0, median us, ON x2 vs OFF, same
    // session): kernel end M=2 49.9/47.3 vs 47.8, M=8 58.3/58.1 vs 61.6, M=16
    // 82.9/82.4 vs 88.4; last L2 task end -> kernel end M=8 2.4/2.0 vs 7.8, M=16
    // 3.9/3.9 vs 8.1; CUDA-event wall M=2 58.7/56.8 vs 56.9, M=8 72.7/73.9 vs 73.2,
    // M=16 96.0/95.2 vs 97.0. Default ON; DG_FP4_FINE_COMBINE=0 restores the
    // barrier path (numerics identical: T=2/8/16/128/512 + QoQ, 200-iter graph
    // replay stress clean).
    const bool fine_combine = get_env<int>("DG_FP4_FINE_COMBINE", 1) != 0;
    const int task_block_n = half_tile_tasks ? config.block_n / 2 : config.block_n * l1_task_tiles;
    constexpr int kL1ScaleGranK = 128;
    // L2 activation scale granularity (kernel kL2ActsSFGranK): per 64 on the BM128/BN128
    // split-M tier and with half-tile tasks, per K128 otherwise; a wide L1 task publishes
    // two K128 groups (one per WG), so it is not task_block_n / 2 any more.
    const int l2_scale_gran_k = (half_tile_tasks || config.block_n == 128) ? 64 : 128;
    const auto tensor_map_l1_acts = make_tma_2d_desc(
        l1_acts, hidden, config.num_max_pool_tokens,
        KernelConfig::kBlockK, config.block_m,
        static_cast<int>(l1_acts.stride(-2)), KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_acts_sf,
        config.num_padded_sf_pool_tokens, hidden,
        config.block_m, kL1ScaleGranK, 1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, static_cast<int>(l1_weights.size(2)),
        num_experts_per_rank * intermediate_hidden * 2,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l1_weights.stride(-2)), 0);

    const int l1_output_store_block_n = task_block_n / 2;
    const auto tensor_map_l1_output = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_max_pool_tokens,
        l1_output_store_block_n, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), 0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_max_pool_tokens,
        KernelConfig::kBlockK, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l2_acts_sf,
        config.num_padded_sf_pool_tokens, intermediate_hidden,
        config.block_m, l2_scale_gran_k, 1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights, static_cast<int>(l2_weights.size(2)),
        num_experts_per_rank * hidden,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l2_weights.stride(-2)), 0);
    int* cumulative_stats_ptr = cumulative_local_expert_recv_stats.has_value() ?
        cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ?
        l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ?
        l2_global_scales->data_ptr<float>() : nullptr;

    const SM90FP4H20FusedRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .swap_ab = plan.swap_ab,
        .use_mode2_row_decoder = plan.use_mode2_row_decoder,
        .single_active_dispatch_warp = plan.single_active_dispatch_warp,
        .use_interleaved_scheduler = plan.use_interleaved_scheduler,
        .mxfp4 = mxfp4,
        // Env knob (default 8): number of BK128 weight rows per task prefetched
        // into L2 cache while waiting for activations; 0 disables.
        // H20 A/B (2026-09-09, phase stamps): prefetching 8 K-blocks per task is
        // ~neutral at M<=8 but pollutes L2 / steals HBM at M=16 (-6us L1 phase
        // when disabled), so default it off from 16 tokens up.
        // Push dispatch: the first stage waits ~2 us for the pool instead of ~7,
        // so the prefetch competes with the stage fills (M=8 H20: first-stage wait
        // 7.3 vs 4.7 us, per-task L1 25.2 vs 24.2 us); default 0.
        .prefetch_weight_k_blocks = get_env<int>("DG_FP4_PREFETCH_KBLOCKS",
                                                 (num_tokens >= 16 || push_dispatch) ? 0 : 8),
        // DG_FP4_SWAP_PIPE stays OFF (no gain on H200 09-02 nor H20 09-09).
        // DG_FP4_DIST_BCAST defaults ON since the H20 09-09 A/B.
        //   DG_FP4_SWAP_PIPE=1   overlap decode(k+1) with WGMMA(k) in swapAB tiles
        //   DG_FP4_DIST_BCAST=1  spread the dispatch expert-count broadcast over all SMs
        // MXFP4/QoQ swapAB tiers use the RF-decode serial loop (kRFDecode).
        .swap_pipeline_decode = !qoq && !mxfp4 && get_env<int>("DG_FP4_SWAP_PIPE", 0) != 0,
        // H20 A/B (2026-09-09): spreading the expert-count broadcast over all
        // SMs removes ~10us of SM0-serial sys-scope atomics (M=2: 62->52us).
        .distributed_expert_bcast = get_env<int>("DG_FP4_DIST_BCAST", 1) != 0,
        .qoq = qoq,
        // MXFP4/QoQ hosts pack dense (E, N/256, K/128, 256, 80 B) weight tiles
        // (deep_gemm/quantization_{mxfp4,qoq}_fused.py); the B loader then does
        // one 1D bulk copy per stage instead of a TMA-issue-bound 2D box.
        // NVFP4 keeps the row-major fused layout + 2D TMA.
        .dense_weight_tiles = dense_weight_tiles,
        .half_tile_tasks = half_tile_tasks,
        .split_k_l1 = split_k_l1,
        .l2_half_row_tasks = l2_half_row_tasks,
        .split_k_l2_ways = split_k_l2 ? split_k_l2_ways : 0u,
        .stream_k = stream_k,
        .nvl_fast_epilogue = nvl_fast_epilogue,
        .fine_combine = fine_combine,
        .push_dispatch = push_dispatch,
        .push_max_tokens_per_rank = push_max_tokens_per_rank,
        // Lean routing (kernel `kLeanRouting`, DG_FP4_LEAN_ROUTING, default 1): the
        // dispatch warps only atomically publish experts with a non-zero local count
        // (the per-CTA arrival high word is replaced by the constant the scheduler
        // expects, the grid sync before the broadcast already orders the counts);
        // with push dispatch the local send counts, the extra grid sync and the
        // cross-rank count broadcast disappear entirely (the destination finalises
        // its own counts after NVLink barrier #1). DG_FP4_LEAN_ROUTING=0 = old path.
        .lean_routing = get_env<int>("DG_FP4_LEAN_ROUTING", 1) != 0,
        .push_done_flags = push_done_flags,
        // QoQ inline s2 (kernel `kQoQInlineS2`, DG_FP4_QOQ_INLINE_S2, default 1): the
        // BM8 QoQ RF swapAB L1 loop folds the per-(row, K128) integer s2 into the
        // int8 weight at decode time and accumulates the whole task K range in one
        // int32 set (one promote by the per-token activation scale at task end)
        // instead of a per-K128 promote (which forced an accumulator readout /
        // tensor-pipe drain per block). 0 = the per-block promote path.
        .qoq_inline_s2 = qoq_inline_s2,
        // DG_FP4_QIS2_FRAGS (2|3|4, default 2): A-fragment register buffers of the
        // inline s2 loop (wait_group lag = frags - 1); 3/4 cost +32/+64 regs.
        .qoq_inline_s2_frags = std::clamp(get_env<int>("DG_FP4_QIS2_FRAGS", 2), 2, 4),
        // DG_FP4_QIS2_ILV (default 0): interleave the next block's per-K32 decode
        // between one-K32-step commit groups (two frag buffers, lag 4 groups).
        // Off: ptxas still serialises that loop (C7513) so it measured 2675 ns per
        // stage vs 1250-1290 for the 2-buffer loop (H20 2026-09-09).
        .qoq_inline_s2_ilv = qoq && get_env<int>("DG_FP4_QIS2_ILV", 0) != 0,
        // DG_FP4_QIS2_PREFETCH_PACKED (default 1): in the 2-buffer inline-s2 loop, issue
        // the LDS of the next block's packed weight words before the wgmma wait that
        // frees its fragment buffer, so the smem latency overlaps the wait. H20 probe
        // (2 passes, skew-corrected kernel end, us): M16 64.6/61.5 vs 67.7/67.4 off,
        // M8 42.1/40.5 vs 40.9/41.2 (L1 phase M16 41.7/42.2 vs 44.0/44.1).
        .qoq_inline_s2_prefetch_packed = qoq && get_env<int>("DG_FP4_QIS2_PREFETCH_PACKED", 1) != 0,
        // DG_FP4_RF_PREFETCH_PACKED (default 0): same packed-word prefetch for the generic
        // 2-K-block RF loop (MXFP4 LUT decode / QoQ per-block promote): the k+1 barrier
        // check and the next block-0 packed LDS move ahead of the wait<1> that frees
        // frag[0]. Numerics unchanged (only the load is hoisted). H20 MXFP4 probe (2
        // passes): a loss — stage 1399/1406 vs 1321/1253 ns (M8), 1312/1311 vs 1261/1317
        // (M16); skew-corrected end M8 44.5/43.7 vs 43.2/41.1 us, M16 69.7/69.9 vs
        // 63.7/64.3 (the hoisted k+1 barrier check exposes the wait: slot 17 654 vs 444
        // ns) — so off. The L1 loop is loader/HBM-bound at these token counts.
        .rf_prefetch_packed = get_env<int>("DG_FP4_RF_PREFETCH_PACKED", 0) != 0,
        .strided_pool_debug = strided_pool_debug,
        .l1_task_tiles = l1_task_tiles,
        .l2_task_tiles = l2_task_tiles,
        .num_experts = num_experts,
        .num_sms = num_sms,
        .ep16_gin_type = ep16_gin.empty() ? 0 : static_cast<int>(ep16_gin[5]),
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l1_weights_ptr = l1_weights.data_ptr(),
        .l2_weights_ptr = l2_weights.data_ptr(),
        .l1_global_scales = l1_global_scales_ptr,
        .l2_global_scales = l2_global_scales_ptr,
        .phase_stamps = phase_stamps.has_value() ?
            reinterpret_cast<unsigned long long*>(phase_stamps->data_ptr()) : nullptr,
        // DG_FE_SELECT_IN_MEGA=1: Fable cc frontend compact key array ([token][384] u32)
        .fe_keys = fe_keys.has_value() ? reinterpret_cast<const uint32_t*>(fe_keys->data_ptr()) : nullptr,
        .ep16_gin = ep16_gin.empty() ? H20Ep16Gin{} : H20Ep16Gin{
            reinterpret_cast<const void*>(ep16_gin[0]), reinterpret_cast<void*>(ep16_gin[1]),
            reinterpret_cast<uint8_t*>(ep16_gin[2]), static_cast<uint32_t>(ep16_gin[3]),
            static_cast<uint32_t>(num_tokens / 2), static_cast<uint32_t>(ep16_gin[4]),
            static_cast<uint32_t>(ep16_gin.size() > 6 ? ep16_gin[6] : 1)},
        .launch_args = LaunchArgs(num_sms, KernelConfig::kNumThreads, config.smem_size)
    };

    const auto code = SM90FP4H20FusedRuntime::generate(args);
    if (phase_stamps.has_value()) {
        DG_HOST_ASSERT(phase_stamps->scalar_type() == torch::kInt64 && phase_stamps->numel() >= 8);
    }
    const std::string kernel_name = std::string(qoq ? "sm90_qoq" : (mxfp4 ? "sm90_mxfp4" : "sm90_nvfp4")) +
        (plan.use_interleaved_scheduler ?
            "_h200_fused_interleaved" :
            (plan.use_mode2_row_decoder ?
                "_h200_fused_mode2_row" :
                "_h200_fused_lut_window")) + (push_dispatch ? "_push" : "") +
        (push_done_flags ? "_pdf" : "") +
        (strided_pool_debug ? "_stridedbg" : "") +
        (get_env<int>("DG_FP4_LEAN_ROUTING", 1) != 0 ? "_lean" : "") +
        (qoq_inline_s2 ? "_qis2" : "") +
        ((qoq && std::clamp(get_env<int>("DG_FP4_QIS2_FRAGS", 2), 2, 4) != 2) ?
            fmt::format("f{}", std::clamp(get_env<int>("DG_FP4_QIS2_FRAGS", 2), 2, 4)) : "") +
        ((qoq && get_env<int>("DG_FP4_QIS2_ILV", 0) != 0) ? "_ilv" : "") +
        ((qoq && get_env<int>("DG_FP4_QIS2_PREFETCH_PACKED", 1) == 0) ? "_nopf" : "") +
        (get_env<int>("DG_FP4_RF_PREFETCH_PACKED", 0) != 0 ? "_rfpf" : "") +
        (wide_tiles ? fmt::format("_bn{}x{}", 256 * l1_task_tiles, 256 * l2_task_tiles) : "") +
        (ep16_gin.empty() ? "" : fmt::format("_ep16gin{}", ep16_gin[5]));
    const auto runtime = compiler->build(kernel_name, code);
    SM90FP4H20FusedRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
