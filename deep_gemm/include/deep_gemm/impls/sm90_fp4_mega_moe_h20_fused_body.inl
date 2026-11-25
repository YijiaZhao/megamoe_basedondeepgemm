// Independent SM90 NVFP4 MegaMoE fused kernel body.
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(BLOCK_M == 8 || BLOCK_M == 16 ||
                     BLOCK_M == 24 || BLOCK_M == 64 || BLOCK_M == 128,
                     "H200 fused kernel requires BM8/BM16/BM24/BM64/BM128");
    // BM8: 4..7 stages of one K128 block, or 2..4 stages of two K128 blocks
    // (kKBlocksPerStage, checked precisely below once kRFDecode is known).
    DG_STATIC_ASSERT((BLOCK_M == 8 && kNumStages >= 2 && kNumStages <= 7) ||
                     ((BLOCK_M == 16 || BLOCK_M == 24) && kNumStages == 3) ||
                     (BLOCK_M == 64 && kNumStages == 3) ||
                     (BLOCK_M == 128 && kNumStages == 6),
                     "Unexpected H200 pipeline depth");
    DG_STATIC_ASSERT((BLOCK_M == 128) == (BLOCK_N == 128),
                     "BM128 is paired with the BN128 split-M topology");
    DG_STATIC_ASSERT(!kSwapABRequested || BLOCK_M <= 24,
                     "swap-AB is only selected through the M64 bucket");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Optional phase timestamps (globaltimer, ns). Slots:
    //   0 min kernel entry | 1 max dispatch routing done | 2 max dispatch pull done
    //   3 min first math task | 4 max last L1 task end | 5 max last L2 task end
    //   6 max after combine NVLink barrier | 7 max combine end | 16 max DG_FE_SELECT_IN_MEGA select done
    //   EP16 GIN: 48 max input put issued | 49 max peer input rows seen | 50 max peer rows copied in
    //   51 min first own-row wait for the peer's partial | 52 max last peer partial seen
    //   53 min first combined peer row put | 54 max last combined peer row put
    //   55 max own rows packed into the send region (per packing warp)
    const auto stamp_min = [&](const uint32_t slot) {
        if (phase_stamps != nullptr) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            atomicMin(phase_stamps + slot, t);
        }
    };
    const auto stamp_max = [&](const uint32_t slot) {
        if (phase_stamps != nullptr) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            atomicMax(phase_stamps + slot, t);
        }
    };
    // Steady-state accumulators (SM 0 only): slot 13 += time from entry to
    // after the first NVLink barrier, slot 15 += entry-to-end, slot 14 += 1.
    unsigned long long t_kernel_entry = 0;
    if (phase_stamps != nullptr)
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_kernel_entry));
    const auto stamp_accumulate = [&](const uint32_t slot) {
        if (phase_stamps != nullptr) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            atomicAdd(phase_stamps + slot, t - t_kernel_entry);
        }
    };
    if (warp_idx == 0 and cute::elect_one_sync()) {
        stamp_min(0);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing. The framework reserves
    // per-64 SF capacity; this per-128 path uses its first half
    // as a dense per-128 layout so no framework allocation change is needed.
    // =====================================================================
    const auto workspace = fused_layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = fused_layout::Data(kHidden);
    constexpr auto bf16_token_layout             = fused_layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = fused_layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = fused_layout::Data(kHidden / 32, false);
    // Physical per-64 capacity: logical per-128 scales occupy the first half.
    constexpr auto fp8_intermediate_sf_layout    = fused_layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = fused_layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = fused_layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = fused_layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = fused_layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = fused_layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = fused_layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = fused_layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = fused_layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = fused_layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = fused_layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = fused_layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = fused_layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = fused_layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    using task_info_t = fused_sched::TaskInfo;
    // Half-tile tasks (kHalfTileTasks; gate DG_FUSED_HALF_TILE_TASKS, default ON,
    // BM8 MXFP4 RF swapAB only == the 2-K-block-per-stage path): a task covers 128
    // of the 256 rows of a packed weight tile (sub-tile n_block_idx % 2) over the
    // full K, so task counts double (L1 10 -> 20, L2 12 -> 24 per M block) and the
    // per-task latency halves (80 L1 tasks on 78 SMs no longer cost two full task
    // lengths). Inside the CTA the two math WGs split K instead of N: WG w decodes
    // and multiplies K-block w of every stage on the SAME 128 rows (2 halves of 64,
    // identical register shape), WG1 hands its promoted partial sums to WG0 through
    // smem and WG0 alone runs the epilogue (64 SwiGLU columns for L1, 128 BF16
    // columns for L2). Weight tile layout (256 x 80 B) is unchanged; the B loader
    // fetches the two 10 KB sub-tiles of a stage with two bulk copies.
    // H20 probe (2026-09-09, 8 ranks, 78 SMs): the per-WG K-loop costs ~740 ns per
    // K128 block whatever the grouping (RF decode 349 + RS-WGMMA issue/drain ~390),
    // so a half task is ~9.9 us instead of ~16 us and 160 tasks quantise onto 78
    // SMs as badly as 80 do: M=2 (single wave) kernel end 51 -> 44.5 us, but M=8
    // 65 -> 68.6 and M=16 90 -> 101-103 us. The host therefore keeps it OFF by
    // default (DG_FP4_HALF_TILE=1 enables it).
    constexpr bool kHalfTileTasks =
        fused_layout::kSM90FusedHalfTileTasks && kHalfTileTasksRequested &&
        kSwapABRequested && kMXFP4 && BLOCK_M == 8 && kL1TaskTiles == 1 && kL2TaskTiles == 1;
    // Wide tasks (kWideTiles; host env DG_FP4_L1_BN / DG_FP4_L2_BN = 512, gated by
    // DG_FP4_BN512_MIN_M / _MAX_M on the global token count, default 16..16): an L1
    // (resp. L2) task covers kL1TaskTiles (kL2TaskTiles) ADJACENT packed 256-row
    // weight tiles (2 -> 512 rows) over the whole K, so the task count of the phase
    // halves (M=16: L1 ~160 -> ~80 tasks = one wave on 78 SMs instead of 2.05, L2
    // ~180 -> ~90). WG w owns packed tile 2n + w (4 x 64-row halves): for L1 that is
    // 128 SwiGLU columns == exactly one L2 activation-scale group (own per-token
    // amax, no cross-WG share), for L2 256 hidden columns. The pipeline carries ONE
    // K128 block per stage (kKBlocksPerStage == 1: 1 KB A + 2 x 20 KB B, 4 stages ==
    // the same 8 K-block bytes in flight as 2 blocks x 4 stages of BN256) and the RF
    // loop's unit is (K-block, 64-row half PAIR) instead of (K-block), so the 2-unit
    // loops, the fragment buffers (frag[2][2 halves]) and the accumulator sets are
    // exactly those of today's 2-K-block stage: same registers, same per-(K128, 256
    // rows) work per stage, same per-K128 promote (numerics identical). Only the BM8
    // MXFP4 / QoQ RF swapAB tier with dense tiles supports it; every other tier keeps
    // one packed tile per task.
    constexpr bool kWideTiles = (kL1TaskTiles > 1 || kL2TaskTiles > 1) &&
        kSwapABRequested && (kMXFP4 || kQoQ) && BLOCK_M == 8 && BLOCK_N == 256 &&
        kUseInterleavedScheduler && kDenseWeightTiles && kNumEpilogueWarpgroups == 2;
    DG_STATIC_ASSERT(kL1TaskTiles == 1 || (kWideTiles && kL1TaskTiles == 2),
                     "Wide L1 tasks: 2 packed tiles per task on the BM8 RF swapAB dense tier only");
    DG_STATIC_ASSERT(kL2TaskTiles == 1 || (kWideTiles && kL2TaskTiles == 2),
                     "Wide L2 tasks: 2 packed tiles per task on the BM8 RF swapAB dense tier only");
    constexpr uint32_t kL1Tiles = kWideTiles ? kL1TaskTiles : 1u;
    constexpr uint32_t kL2Tiles = kWideTiles ? kL2TaskTiles : 1u;
    // N extent of one scheduled task (== the weight tile N unless half-tile / wide tasks).
    constexpr uint32_t TASK_BLOCK_N = kHalfTileTasks ? BLOCK_N / 2 : BLOCK_N * kL1Tiles;
    // L1 split-K tasks (kSplitKL1; host env DG_FP4_SPLITK_L1, default ON for the BM8
    // MXFP4 RF swapAB tier == the 2-K-block-per-stage path; every other tier is
    // untouched). The L1 (expert, n_block) tasks of the last partial L1 wave (the
    // `num_l1_tasks % kNumSMs` stragglers; M=8: 2 of 80, M=16: 4 of 160) are each
    // scheduled as two adjacent task indices k_half = 0/1 covering K-blocks
    // [12*k_half, 12*k_half + 12) (6 stages), so the two halves run concurrently
    // on two SMs and the straggler wave takes ~half a task length. Splitting every
    // task instead was measured neutral (H20 09-09: 3 waves of ~11.3 us halves ==
    // 2 waves of ~17 us tasks; ~3.8 us fixed cost per task). The per-WG math
    // shape is identical to the unsplit path; `final_accum` already carries the
    // per-K-block activation SF at promote time, so the two partials are directly
    // summable. Reduction (before the L1 epilogue): half 0 publishes its partial to
    // a global scratch slot (fused_layout::Workspace, 8 KB) and releases a
    // per-(pool_block, n_block) flag, skipping the epilogue (no L1-ready notify);
    // half 1 acquires the flag, adds the slot and runs the normal epilogue +
    // notify. The scheduler only splits when the launch's pool
    // block count fits the scratch (kSM90SplitKL1MaxPoolBlocks); L2 is unchanged.
    constexpr bool kSplitKL1 =
        kSplitKL1Requested && kSwapABRequested && (kMXFP4 || kQoQ) && BLOCK_M == 8 &&
        !kHalfTileTasks && kUseInterleavedScheduler && kDenseWeightTiles && BLOCK_N == 256;
    // Wide L1 tasks: M=16 gives 16 pool blocks x 5 = 80 tasks on 78 SMs, so the 2-task
    // straggler wave is split THREE ways (8 K128 blocks each, ~1/3 of a task) instead
    // of two; the two publishers' 16 KB partials use the (idle with wide tasks) L2
    // split-K scratch, see `slot_of` in the reduction.
    constexpr uint32_t kNumL1KSplits = kSplitKL1 ?
        ((kWideTiles && kL1Tiles > 1) ? 3u : fused_layout::kSM90SplitKL1NumKSplits) : 1u;
    DG_STATIC_ASSERT(kNumL1KSplits <= 1u + fused_layout::kSM90SplitKL2MaxPublishers,
                     "Wide L1 3-way split: publisher partials live in the L2 split-K slots");
    // L2 half-row tasks (kL2HalfRowTasks; host env DG_FP4_L2_HALFROW, default OFF;
    // only the BM8 MXFP4 RF swapAB tier == the 2-K-block-per-stage path with dense
    // BN256 tiles and the interleaved scheduler can enable it; exclusive with
    // kHalfTileTasks; every other tier is untouched). After L1 split-K the L2 tasks are the critical tail
    // (192 tasks of ~6.5 us quantise badly on 78 SMs: M16 last L2 - last L1 = 11 us).
    // An L2 task covers 128 rows (one 10 KB half of a packed weight tile, the BN128
    // sub-tile addressing of the dense-tile work) instead of 256, and the two math
    // WGs split those 128 rows: WG w decodes and multiplies rows [64w, 64w + 64) for
    // BOTH K-blocks of every stage (one 64-row weight half per WG, 8 RS WGMMAs per
    // stage), keeps its own accumulators and runs its own epilogue for its 64
    // hidden rows (no cross-WG reduction). Per-WG per-stage work halves, so the
    // task takes ~half the time; the L2 task count doubles (12 -> 24 per M block).
    // L1 tasks are unchanged (256 rows, 10 per M block); the L1 -> L2 K-block
    // dependency is unchanged (the L2 task N does not affect its K mapping).
    // H20 probe (2026-09-09): the premise does not hold. The per-WG stage cost is a
    // latency chain (smem load -> LUT -> decode -> RS WGMMA -> drain -> promote), so
    // an L2 task still takes ~6.5 us with 64 rows per WG (same as with 128) and the
    // doubled task count doubles the L2 tail; the ON build also slowed the L1 task
    // (29 vs 22.7 us, register spills). Kept as an OFF-by-default knob (numerics
    // verified); see the host for the numbers.
    constexpr bool kL2HalfRowTasks =
        kL2HalfRowTasksRequested && kSwapABRequested && kMXFP4 && BLOCK_M == 8 &&
        !kHalfTileTasks && !kWideTiles && kUseInterleavedScheduler && kDenseWeightTiles &&
        BLOCK_N == 256 && kNumEpilogueWarpgroups == 2;
    // N extent of one scheduled L2 task (L1 tasks use TASK_BLOCK_N).
    constexpr uint32_t TASK_BLOCK_N_L2 = kL2HalfRowTasks ? BLOCK_N / 2 :
        (kHalfTileTasks ? TASK_BLOCK_N : BLOCK_N * kL2Tiles);
    // L2 split-K tasks (kSplitKL2; host env DG_FP4_SPLITK_L2): the L2 (expert,
    // n_block) tasks of the last partial L2 wave (`num_l2_tasks % kNumSMs`
    // stragglers; M=8: 18 of 96, M=16: 36 of 192) are each scheduled as two
    // adjacent task indices covering whole 2-K128-block stages: K half 0 =
    // K-blocks [0, 4) (2 stages, PUBLISHER), K half 1 = [4, 10) (3 stages,
    // FINISHER, claimed one index later), so the finisher's flag wait is ~free.
    // Same protocol/scratch format as kSplitKL1 (separate L2 slot index space);
    // the L2 epilogue (BF16 x scale, NVLink scatter) runs on the finisher only.
    // Each half's TMA producer waits only for the L1 bits of its own K-blocks.
    // kSplitKL2Ways == 3: three stage-aligned K ranges (L2: 1/2/2 stages, blocks [0,2)
    // / [2,6) / [6,10)); the last range is the finisher, the others publish their fp32
    // partial to their own slot and red.add the task flag; the finisher waits for
    // n - 1 arrivals and sums the slots in publisher order (see the epilogue).
    constexpr bool kSplitKL2 =
        kSplitKL2Ways != 0 && kSwapABRequested && kMXFP4 && BLOCK_M == 8 &&
        !kHalfTileTasks && !kL2HalfRowTasks && !kWideTiles && kUseInterleavedScheduler &&
        kDenseWeightTiles && BLOCK_N == 256;
    constexpr uint32_t kNumL2KSplits = kSplitKL2 ? (kSplitKL2Ways >= 3 ? 3u : 2u) : 1u;
    DG_STATIC_ASSERT(kNumL2KSplits <= 1u + fused_layout::kSM90SplitKL2MaxPublishers,
                     "L2 split-K needs one partial slot per publisher");
    // Stream-K (kStreamK; host env DG_FP4_STREAMK, gated by DG_FP4_STREAMK_MAX_M on
    // the global token count): for tiny M the L1 phase is ~16 us per wave of
    // (expert, n_block) tasks over the full K (24 K128 blocks) while most of the 78
    // SMs idle (M=2: 16 tasks) or a 2-task second wave doubles it (M=8: 80 tasks).
    // Instead, all (task, K-block) units of a phase are split into 78 contiguous,
    // near-equal unit ranges (task-major, K inner; scheduler
    // `claim_next_streamk_task`), so every SM streams ~1/78 of the phase's weights
    // and the phase takes ~total_units / 78 K-block times. A range is published as
    // consecutive segments (task + K-block sub-range: the K loop / RS pipeline is
    // unchanged, partial last stages are already supported). Every segment of a
    // tile with > 1 contributor stores its promoted fp32 partial to its worker's
    // per-phase slot and takes an acq_rel ticket on the tile's arrival counter (the
    // split-K flag); the last arriver (ticket == splits - 1) sums the partials in
    // split order (deterministic: fixed order regardless of who finishes) and runs
    // the normal epilogue (L1: SwiGLU + notify; L2: scatter + arrival counters). The
    // L2 activation loader keeps its per-stage wait on the L1 bits of its own
    // K-blocks. Units are whole 2-K-block stages (the RF loop consumes full
    // stages), so only the 2-blocks-per-stage tier qualifies (L2: 10 % 2 == 0).
    // Same tier gate as kSplitKL2 otherwise; the wave scheduler (and its tail
    // splits) remains the fallback when the pool block count exceeds the scratch.
    constexpr bool kStreamK =
        kStreamKRequested && kSwapABRequested && (kMXFP4 || kQoQ) && BLOCK_M == 8 &&
        !kHalfTileTasks && !kL2HalfRowTasks && !kWideTiles && kUseInterleavedScheduler &&
        kDenseWeightTiles && BLOCK_N == 256 && !kWideTiles;
    // Fast NVLink-barrier epilogue (kNvlFastEpilogue; host env DG_FP4_NVL_FAST_EPI):
    // see fused_comm::nvlink_barrier. Needs the first barrier (before dispatch
    // pull) to have a prologue grid sync, i.e. kDistributedExpertBcast, so that
    // SM0's first write of the done word is ordered after every CTA's kernel-start
    // snapshot of it.
    constexpr bool kNvlFastEpilogue = kNvlFastEpilogueRequested && kDistributedExpertBcast;
    // Fine-grained combine (host env DG_FP4_FINE_COMBINE, default 1): the combine
    // NVLink barrier (#2) is replaced by per-(destination rank, token) arrival
    // counters (`Workspace::get_combine_arrival_count_ptr`). Every L2 task, after its
    // CTA-wide post-scatter sync, red.release.sys-adds 1 per valid token row to the
    // row's destination counter (NVLink for remote destinations); a combine warp
    // spins (ld.acquire.sys) on its token's counter until it equals
    // popc(valid topk slots) * kNumRoutedL2BlockNs, then resets it and reads the
    // partials. The barrier's second job (all local math tasks done before the
    // dispatch warps clean the workspace) moves to a dispatch-warp grid sync, so
    // the combine warps never wait for other CTAs. Pool reuse across launches is
    // protected by NVLink barrier #1 of the next launch (a remote rank only starts
    // scattering after every CTA of every rank has entered that launch) and #3
    // (workspace cleanup), exactly as before.
    constexpr bool kFineCombine = kFineCombineRequested;
    // Dynamic combine token claim (fine combine only): instead of the static map
    // token -> (sm_idx * warps + warp), a
    // combine warp that has finished its CTA's math tasks claims tokens from a
    // per-launch ticket (`Workspace::get_combine_ticket_ptr`, atom.add.gpu by lane 0)
    // until it draws a ticket >= num_tokens. The ticket is double-buffered by launch
    // parity (`get_combine_epoch_ptr`, read once by every combine warp at role
    // start): SM0's cleanup of launch N (after the dispatch grid sync, i.e. after
    // every CTA's epilogue warps posted DONE, hence after their epoch read) zeroes
    // the word of parity N+1 and bumps the epoch. The zeroed word was last used by
    // launch N-1, whose kernel completed before N started on this rank (same
    // stream), so no late claimer of N-1 can observe the reset; launch N's own word
    // is only zeroed by the cleanup of N+1, after N completed.
    constexpr bool kCombineDynamic = kFineCombine;
    // Push dispatch (kPushDispatch; host env DG_FP4_PUSH_DISPATCH, see the host for
    // the default, gated by DG_FP4_PUSH_DISPATCH_MAX_M on the global token count,
    // default 16). Pull model
    // (default for larger M): after NVLink barrier #1 the dispatch warps pull every
    // received row (TMA over NVLink, ~6.7 us round trip + local store) before the
    // first math task can start. Push model: during routing the SOURCE rank takes a
    // remote atomic ticket on the destination's per-expert count (the same
    // `expert_recv_count_sum` word the broadcast finalises; its low 32 bits now grow
    // by 1 per row, the broadcast adds only the SM-count high word), so the row's
    // position inside the expert is known before any total is, and writes the 3 KB
    // row + per-K128 SF + top-k weight + source metadata straight into the
    // destination's pool with plain 16 B stores. Since the destination's dense
    // prefix-sum pool offsets are unknown to the sender, the pool is addressed with a
    // FIXED stride of kPushBlocksPerExpert blocks per local expert (rows inside an
    // expert stay packed: block e * stride + row / BLOCK_M); the scheduler keeps
    // dense task indices and only remaps `pool_block_idx` (see
    // InterleavedMegaMoEScheduler::create_task). Visibility: the pushes precede the
    // sender's grid sync + SM0's release.sys barrier signal exactly like the top-k
    // index writes of the pull model; after barrier #1 SM e's dispatch warp 0
    // finalises expert e's count with atom.release.gpu (the scheduler's acquire poll
    // of that word then covers the rows, so lean-push L1 loaders skip the per-task
    // arrival spin) and still publishes the arrival counts (tiny-M path). Pool reuse: a rank
    // can only push launch N+1 rows after NVLink barrier #3 of launch N, which every
    // rank reaches after ALL its math tasks and its workspace cleanup, so no
    // destination still reads launch N's L1 pool / metadata or has counters pending.
    constexpr bool kPushDispatch = kPushDispatchRequested && kUseInterleavedScheduler;
    // Lean push (see the dispatch prologue): no cross-rank count broadcast, the
    // destination finalises its counts after NVLink barrier #1.
    constexpr bool kLeanPush = kLeanRouting && kPushDispatch;
    // Push DONE flags (kPushDoneFlags; host env DG_FP4_PUSH_DONE_FLAGS, default 1):
    // barrier #1 is not executed. Sender: after a CTA's dispatch warps issued their
    // last pushed row, thread 0 takes a CTA arrival ticket (atom.acq_rel.gpu, after the
    // CTA barrier: releases this CTA's remote stores, acquires the earlier CTAs'); the
    // last CTA resets the ticket and red.release.sys-adds 1 into every rank's DONE
    // count (its own included; one lane per rank, one warp-wide sys fence). Receiver: the DONE target of a launch is
    // kNumRanks * (epoch + 1) where `epoch` counts this rank's completed flag
    // launches (bumped by SM0 in the workspace cleanup, i.e. after every reader of
    // this launch: the task producers fetch the counts before any math task exists,
    // SM e's publisher runs before SM e's epilogue warps join the math). Waiters:
    // (1) every CTA's task producer (ld.acquire.sys of the DONE count, then the final
    // low words of the 6 sums; the acquire covers the rows for the lean-push loaders,
    // which skip the per-task arrival spin), (2) SM e's dispatch warp 0, which then
    // publishes the completeness high word + L1 arrival counts exactly as before (the
    // tiny-M loaders and the comm-window L2 prefetch poll them). Everything else
    // barrier #1 ordered under lean push (nothing: no send counts, no broadcast, no
    // top-k index writes) is untouched; the pool-reuse argument rests on barrier #3.
    // The DONE count is monotonic (never reset) and a launch N+1 signal can only be
    // issued by a rank that passed barrier #3 of launch N, i.e. after every rank read
    // launch N's target, so early (skewed) arrivals for N+1 are counted correctly.
    constexpr bool kPushDoneFlags = kPushDoneFlagsRequested && kLeanPush;
    // Strided pool layout: implied by push dispatch; `kStridedPoolDebug` (host env
    // DG_FP4_POOL_STRIDE_DEBUG=1, pull dispatch only) forces the same fixed-stride
    // pool addressing under the PULL protocol, to separate the layout's cost from
    // the push protocol's (tickets / arrival publish) in the phase-stamp probe.
    constexpr bool kStridedPool = kUseInterleavedScheduler && (kPushDispatch || kStridedPoolDebug);
    constexpr uint32_t kPushBlocksPerExpert = kStridedPool ?
        math::constexpr_ceil_div(kNumRanks * kPushMaxTokensPerRank, BLOCK_M) : 0u;
    DG_STATIC_ASSERT(!kStridedPool ||
                     kNumExpertsPerRank * kPushBlocksPerExpert * BLOCK_M <= kNumMaxPoolTokens,
                     "Push dispatch: strided pool must fit the token pool");
    DG_STATIC_ASSERT(!kStridedPool ||
                     kNumExpertsPerRank * kPushBlocksPerExpert <= fused_layout::kSM90SplitKL1MaxPoolBlocks,
                     "Push dispatch: strided pool block indices must fit the split-K / tiny-M slots");
    constexpr uint32_t kNumRoutedL1BlockNs = L1_SHAPE_N / TASK_BLOCK_N;
    constexpr uint32_t kNumRoutedL2BlockNs = L2_SHAPE_N / TASK_BLOCK_N_L2;
    using interleaved_scheduler_t = fused_sched::InterleavedMegaMoEScheduler<
        BLOCK_M, TASK_BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumSMs, kNumRanks,
        kNumL1KSplits, fused_layout::kSM90SplitKL1MaxPoolBlocks,
        math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
        /* phase-specific task N: L1 256-row tasks, L2 TASK_BLOCK_N_L2-row tasks */
        kNumRoutedL1BlockNs, kNumRoutedL2BlockNs,
        kNumL2KSplits, kStreamK, /* stream-K unit == one 2-K-block stage */ 2u,
        L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K, kPushBlocksPerExpert>;
    constexpr bool kSplitMDecodedWeightReuse =
        BLOCK_M == 128 && BLOCK_N == 128 && kNumEpilogueWarpgroups == 2;
    constexpr uint32_t WG_BLOCK_M =
        kSplitMDecodedWeightReuse ? BLOCK_M / 2 : BLOCK_M;
    // Per-WG L1 rows: the WGs split the task N (wide L1 tasks: WG w owns packed tile
    // 2n + w, i.e. 256 rows = 4 halves).
    constexpr uint32_t WG_BLOCK_N =
        kSplitMDecodedWeightReuse ? BLOCK_N : (kHalfTileTasks ? BLOCK_N / 2 : TASK_BLOCK_N / 2);
    constexpr uint32_t L1_OUT_BLOCK_N = TASK_BLOCK_N / 2;  // post-SwiGLU task N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    // Rows per WG of an L2 task: 64 (one weight half) with L2 half-row tasks,
    // otherwise half the L2 task N (two halves; four with wide L2 tasks).
    // (BM128 split-M: each WG covers the whole BN128 tile, as for L1.)
    constexpr uint32_t L2_WG_BLOCK_N = kL2HalfRowTasks ? 64u :
        (kSplitMDecodedWeightReuse ? BLOCK_N : TASK_BLOCK_N_L2 / 2);
    // Capacity of the swapAB accumulator layout (64-row halves per WG, max over the
    // two phases); each phase uses its own kWGHalves <= this.
    constexpr uint32_t kSwapABWeightHalves = (WG_BLOCK_N > L2_WG_BLOCK_N ? WG_BLOCK_N : L2_WG_BLOCK_N) / 64;
    constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
    DG_STATIC_ASSERT(!kSwapABRequested || WG_L1_OUT_BLOCK_N == 64 || (kWideTiles && WG_L1_OUT_BLOCK_N == 128),
                     "swapAB expects 64 (or, wide tasks, 128) L1 output columns per WG");
    // Both dispatch warps participate in CTA-wide barriers. Selected plans may
    // use one warp for routing and token pulls, leaving the other warp's send
    // buffer available for an additional GEMM stage.
    constexpr uint32_t kNumActiveDispatchWarps =
        kSingleActiveDispatchWarp ? 1u : kNumDispatchWarps;
    constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
    constexpr bool kQuadDequantIlp = BLOCK_M == 8;
    // QoQ W4A8: int8 activations, (code - z) int8 weights, int32 IGMMA
    // accumulators promoted per K128 by s2[row] * s_act[token]; per-row s1 is
    // applied through the same [E, N] epilogue scale path as MXFP4's 2^e_ref.
    constexpr bool kPerRowEpilogueScale = kMXFP4 || kQoQ;
    DG_STATIC_ASSERT(!kQoQ || kSwapABRequested,
                     "QoQ currently supports the swapAB (<= 64 tokens per rank) tiers only");
    DG_STATIC_ASSERT(!(kQoQ && kSwapPipelineDecode),
                     "QoQ uses the serial swapAB main loop");
    using swap_accum_t = std::conditional_t<kQoQ, int32_t, float>;
    // RF decode (swapAB MXFP4 and QoQ tiers): each thread decodes its own WGMMA A
    // fragment straight from the packed-B rows (host stores MXFP4/QoQ rows in RF
    // fragment order, see `store_decoded_quad`) and issues RS-form WGMMAs.
    // No decoded SMEM tile, no per-stage STS, no per-WG decode barrier. QoQ uses
    // the int8 RS atoms (int32 accumulators, s2[row] * s_act[token] promote).
    DG_STATIC_ASSERT(!(kMXFP4 && kQoQ), "MXFP4 and QoQ are exclusive");
    constexpr bool kRFDecode = kSwapABRequested && (kMXFP4 || kQoQ);
    DG_STATIC_ASSERT(!(kRFDecode && kSwapPipelineDecode),
                     "RF decode is only implemented for the serial swapAB main loop");
    // K128 blocks per pipeline stage. The tiny-M (BM8) RF swapAB path carries two
    // consecutive K128 blocks per stage: H20 probes put the fixed per-stage skeleton
    // (mbarrier check + wgmma drain + arrive/loop) at ~540 ns of a ~740 ns
    // single-block stage, so amortising it over two K-blocks cuts the per-K128 cost
    // (741 -> ~620 ns). Wide tasks (2 x 20 KB packed B per K-block) and every other
    // path keep one K128 block per stage.
    constexpr uint32_t kKBlocksPerStage = (kRFDecode && BLOCK_M == 8 && !kWideTiles) ? 2u : 1u;
    DG_STATIC_ASSERT(BLOCK_M != 8 ||
                     (kKBlocksPerStage == 2 ? (kNumStages >= 2 && kNumStages <= 4)
                                            : (kNumStages >= 4 && kNumStages <= 7)),
                     "BM8 pipeline depth: 4..7 (1 K-block/stage) or 2..4 (2 K-blocks/stage)");
    // Loaders and math step `kKBlocksPerStage` K-blocks per stage; the last stage
    // of a task may be PARTIAL (min(kKBlocksPerStage, remaining) blocks). The RF main loop keeps
    // "frag[0] holds block 0 of a stage", which needs every task K-block count to be
    // even (L1 24 / 12 per split-K half, L2 10 / 4 + 6 per split-K half).
    DG_STATIC_ASSERT((L1_SHAPE_K / BLOCK_K) % 2 == 0 && (L2_SHAPE_K / BLOCK_K) % 2 == 0 &&
                     ((L1_SHAPE_K / BLOCK_K) / 2) % kKBlocksPerStage == 0,
                     "Even K-block counts (L1 whole stages) are required");
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 || !kSplitMDecodedWeightReuse,
                     "Multi-K-block stages are not implemented for the BM128 split-M path");
    DG_STATIC_ASSERT(!kHalfTileTasks ||
                     (kRFDecode && kKBlocksPerStage == 2 && kNumEpilogueWarpgroups == 2 &&
                      BLOCK_N == 256 && kDenseWeightTiles),
                     "Half-tile tasks: RF decode, 2 K-blocks per stage (one per WG), dense BN256 tiles");
    DG_STATIC_ASSERT(!kL2HalfRowTasks ||
                     (kRFDecode && kKBlocksPerStage == 2 && TASK_BLOCK_N == 256 &&
                      L2_SHAPE_N % TASK_BLOCK_N_L2 == 0 && L2_WG_BLOCK_N == 64),
                     "L2 half-row tasks: RF decode, 2 K-blocks per stage, full-tile L1 tasks, 128-row L2 tasks");
    DG_STATIC_ASSERT(!kSplitKL1 ||
                     (kRFDecode && (kKBlocksPerStage >= 2 || kWideTiles) && kNumEpilogueWarpgroups == 2 &&
                      ((L1_SHAPE_K / BLOCK_K) % (kNumL1KSplits * kKBlocksPerStage)) == 0 &&
                      (L1_SHAPE_N / TASK_BLOCK_N) * kL1Tiles <= fused_layout::kSM90SplitKL1NumL1BlockNs),
                     "Split-K L1: RF decode, multi-K-block stages, K halves of whole stages, L1 N-blocks fit the slots");
    DG_STATIC_ASSERT(!kSplitKL2 ||
                     (kRFDecode && kKBlocksPerStage >= 2 && kNumEpilogueWarpgroups == 2 &&
                      (L2_SHAPE_K / BLOCK_K) >= 2 * kKBlocksPerStage &&
                      L2_SHAPE_N / TASK_BLOCK_N_L2 <= fused_layout::kSM90SplitKL2NumL2BlockNs &&
                      L2_WG_BLOCK_N == WG_BLOCK_N),
                     "Split-K L2: RF decode, multi-K-block stages, >= 2 whole stages, L2 N-blocks fit the slots, full-row L2 tasks");
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    // A and B are CTA-local in the fixed cluster-size-one plan.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    // Half-tile tasks publish 64 L1 output columns per task, each with its own
    // per-token scale (like the BN128 split-M tier).
    constexpr uint32_t kL2ActsSFGranK =
        (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 64u : 128u;
    DG_STATIC_ASSERT(kSplitMDecodedWeightReuse || kHalfTileTasks ||
                     WG_L1_OUT_BLOCK_N < kL2ActsSFGranK || WG_L1_OUT_BLOCK_N == kL2ActsSFGranK,
                     "split-N warpgroups must share one L2 activation scale or own whole groups");
    // L2 activation-scale groups per L1 task (1; wide L1 tasks: 2, one per WG).
    constexpr uint32_t kL1SFGroups = L1_OUT_BLOCK_N / kL2ActsSFGranK;
    DG_STATIC_ASSERT(kL1SFGroups == 1 || (kL1SFGroups == 2 && WG_L1_OUT_BLOCK_N == kL2ActsSFGranK),
                     "L1 task SF groups: one shared by both WGs, or one per WG");
    // L1 -> L2 data dependency is per block: L1 N-block `n` (gate/up
    // interleaved) publishes L1-output columns [n * L1_OUT_BLOCK_N, +L1_OUT_BLOCK_N)
    // plus the matching per-token activation scales, which is exactly what L2
    // K-block(s) `n * kNumL2KBlocksPerL1Block / kNumL1BlocksPerL2KBlock` consume
    // (wide L1 tasks: one L1 N-block feeds two L2 K128 blocks). The L2 A-loader
    // waits per K-block on the readiness bits of only those L1 N-blocks.
    constexpr uint32_t kNumL2KBlocks = L2_SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL1BlocksPerL2KBlock = L1_OUT_BLOCK_N >= BLOCK_K ? 1u : BLOCK_K / L1_OUT_BLOCK_N;
    constexpr uint32_t kNumL2KBlocksPerL1Block = L1_OUT_BLOCK_N >= BLOCK_K ? L1_OUT_BLOCK_N / BLOCK_K : 1u;
    DG_STATIC_ASSERT((BLOCK_K % L1_OUT_BLOCK_N == 0 || L1_OUT_BLOCK_N % BLOCK_K == 0) &&
                     kNumRoutedL1BlockNs * kNumL2KBlocksPerL1Block == kNumL2KBlocks * kNumL1BlocksPerL2KBlock &&
                     kNumRoutedL1BlockNs <= 64,
                     "L1 output N-blocks must tile the L2 K dimension exactly");
    DG_STATIC_ASSERT(L1_OUT_BLOCK_N % kL2ActsSFGranK == 0,
                     "L2 activation scale groups must not straddle L1 output blocks");
    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumActiveDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_NVFP4_LUT_SIZE =
        math::constexpr_align<uint32_t>(128u * sizeof(uint2), kSharedMemoryAlignment);
    // RF decode: per-lane replicated LUT (128 entries x 32 lanes x 8 B = 32 KB).
    // Lane l reads entry i at [i*32 + l], so a warp's 32 gathers always hit 32
    // distinct 8 B slots of one 256 B row -> bank-conflict-free regardless of i.
    constexpr uint32_t SMEM_RF_LUT_REP_SIZE = 0u;  // replicated LUT measured: no gain (bank conflicts not the limiter)
    // One BM x K128 FP8 A tile (128 B-swizzled rows). With two K-blocks per stage
    // the tiles sit back to back; each is 1 KB (BM8) so the 1024 B swizzle
    // period and the WGMMA descriptor base alignment are preserved.
    constexpr uint32_t SMEM_A_SIZE_PER_KBLOCK = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = kKBlocksPerStage * SMEM_A_SIZE_PER_KBLOCK;
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 || SMEM_A_SIZE_PER_KBLOCK % 1024 == 0,
                     "Per-K-block A tiles must keep the 1024 B swizzle period");
    // RF decode feeds WGMMA A straight from registers; no decoded-B tile.
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        kRFDecode ? 0u : LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // BM128 split-M alternates two decoded-B slots. This lets one WG begin
    // decoding K+1 after its K WGMMA completes without overwriting the slot
    // that the paired WG may still be consuming.
    // swapAB BM8 (tiny-M): every WG decodes exactly the 128 rows it consumes
    // and drains its WGMMAs (wait<0>) before decoding the next stage, so two
    // decoded slots suffice; this frees ~64 KB for deeper packed-B prefetch.
    constexpr bool kDoubleBufferDecodedB =
        kSplitMDecodedWeightReuse || (kSwapABRequested && BLOCK_M == 8);
    constexpr uint32_t kNumDecodedBStages =
        kDoubleBufferDecodedB ? 2u : kNumStages;
    constexpr uint32_t B_LOAD_BYTES_PER_ROW = 80u;
    // Half-tile tasks: a K-block of a task is one 128-row (10 KB) sub-tile; wide
    // tasks: two adjacent packed tiles (40 KB). The stage slot is sized for the
    // larger of the L1 and L2 task K-blocks (they differ only when exactly one
    // phase runs wide tasks); each phase addresses its K-blocks with its own size.
    constexpr uint32_t SMEM_PACKED_B_L1_SIZE_PER_KBLOCK =
        TASK_BLOCK_N * B_LOAD_BYTES_PER_ROW * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_KBLOCK =
        (TASK_BLOCK_N > TASK_BLOCK_N_L2 ? TASK_BLOCK_N : TASK_BLOCK_N_L2) * B_LOAD_BYTES_PER_ROW * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE =
        kKBlocksPerStage * SMEM_PACKED_B_SIZE_PER_KBLOCK;
    // Dense weight tiles: one (BLOCK_N x 80 B) tile == exactly one packed-B stage,
    // fetched with a single `cp.async.bulk` (needs 16 B size/address alignment).
    DG_STATIC_ASSERT(!kDenseWeightTiles || (kMXFP4 || kQoQ),
                     "Dense weight tiles are only packed by the MXFP4/QoQ hosts");
    DG_STATIC_ASSERT(SMEM_PACKED_B_SIZE_PER_STAGE % 16 == 0, "Bulk copy size must be 16 B aligned");
    DG_STATIC_ASSERT(SMEM_PACKED_B_L1_SIZE_PER_KBLOCK == TASK_BLOCK_N * 80u, "Unexpected packed-B K-block size");
    // Host packer tile height (rows); a BLOCK_N < 256 kernel tile is a contiguous
    // BLOCK_N*80 B slice of the 256*80 B packed tile, a wide task spans kL1Tiles
    // (kL2Tiles) whole packed tiles (adjacent tile rows of the dense layout).
    constexpr uint32_t kPackedTileN = 256u;
    constexpr uint32_t kPackedTileBytes = kPackedTileN * B_LOAD_BYTES_PER_ROW;
    DG_STATIC_ASSERT(kPackedTileN % TASK_BLOCK_N == 0 || TASK_BLOCK_N == kPackedTileN * kL1Tiles,
                     "Task BLOCK_N must divide the packed tile height or span whole packed tiles");
    constexpr uint32_t kSubTilesPerPacked = TASK_BLOCK_N >= kPackedTileN ? 1u : kPackedTileN / TASK_BLOCK_N;
    // L2 tasks: with L2 half-row tasks a K-block is one 128-row (10 KB) sub-tile
    // (the upper/lower half of the 20 KB packed tile); the stage slot is sized for
    // L1 (2 x 20 KB) and L2 simply uses its first 2 x 10 KB.
    DG_STATIC_ASSERT(kPackedTileN % TASK_BLOCK_N_L2 == 0 || TASK_BLOCK_N_L2 == kPackedTileN * kL2Tiles,
                     "L2 task BLOCK_N must divide the packed tile height or span whole packed tiles");
    constexpr uint32_t kL2SubTilesPerPacked = TASK_BLOCK_N_L2 >= kPackedTileN ? 1u : kPackedTileN / TASK_BLOCK_N_L2;
    constexpr uint32_t SMEM_PACKED_B_L2_SIZE_PER_KBLOCK = TASK_BLOCK_N_L2 * B_LOAD_BYTES_PER_ROW;
    DG_STATIC_ASSERT(SMEM_PACKED_B_L2_SIZE_PER_KBLOCK <= SMEM_PACKED_B_SIZE_PER_KBLOCK &&
                     SMEM_PACKED_B_L2_SIZE_PER_KBLOCK % 16 == 0,
                     "L2 packed-B K-block must fit the L1-sized stage slot (16 B aligned)");
    // Two K-blocks per stage are fetched with ONE bulk copy, which needs the
    // consecutive k tiles of an (expert, n_block) to be contiguous: dense layout
    // with the kernel tile == the packed tile (BN256).
    // (Half-tile tasks fetch the two 10 KB sub-tiles with two bulk copies instead.)
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 ||
                     (kDenseWeightTiles && (kSubTilesPerPacked == 1 || kHalfTileTasks)),
                     "Multi-K-block stages need contiguous dense BN256 weight tiles");
    // L1 and L2 each consume one per-128 activation scale per row and K tile.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAGroups =
        (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 2u : 1u;
    // One kL2SFAHalfStride slot per (K-block, SF group).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kKBlocksPerStage * kNumL2SFAGroups * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = kSwapABRequested ?
        BLOCK_M * TASK_BLOCK_N_L2 * sizeof(nv_bfloat16) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SLOTS =
        kNumEpilogueWarpgroups * BLOCK_M;
    constexpr uint32_t SMEM_CD_L1_SWAP_AMAX_SLOTS = kSwapABRequested ?
        BLOCK_M * kNumEpilogueWarps : 0u;
    constexpr uint32_t SMEM_CD_L1_EXTRA_FLOAT_SLOTS =
        SMEM_CD_L1_SHARED_SF_SLOTS > SMEM_CD_L1_SWAP_AMAX_SLOTS ?
        SMEM_CD_L1_SHARED_SF_SLOTS : SMEM_CD_L1_SWAP_AMAX_SLOTS;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SIZE =
        SMEM_CD_L1_EXTRA_FLOAT_SLOTS * sizeof(float);
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE + SMEM_CD_L1_SHARED_SF_SIZE;
    constexpr uint32_t SMEM_CD_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);
    // Half-tile tasks: WG1 -> WG0 accumulator hand-off, [element][128 threads] floats
    // (the swapAB accumulators a thread actually uses: 2 halves x token chunks x 4).
    constexpr uint32_t kKSplitReduceElems = kSwapABWeightHalves * kSwapABTokenChunks * 4u;
    constexpr uint32_t SMEM_KSPLIT_REDUCE_SIZE =
        kHalfTileTasks ? 128u * kKSplitReduceElems * static_cast<uint32_t>(sizeof(float)) : 0u;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE + SMEM_RF_LUT_REP_SIZE +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE;

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = fused_layout::Buffer(
        fp8_token_layout, kNumActiveDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
    auto smem_nvfp4_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));

    auto smem_rf_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE));
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE +
                     SMEM_RF_LUT_REP_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l1_shared_sf =
        math::advance_ptr<float>(smem_cd_base, SMEM_CD_OUTPUT_BASE_SIZE);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        const uint32_t decoded_stage =
            kDoubleBufferDecodedB ? (i & 1u) : i;
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base,
            SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            decoded_stage * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE +
            i * SMEM_PACKED_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    // Barriers live after SF.
    auto smem_ksplit_reduce = reinterpret_cast<float*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + SMEM_KSPLIT_REDUCE_SIZE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    constexpr uint32_t kNumBaseBarriers =
        kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
    auto task_info_full_barriers = barrier_start_ptr + kNumBaseBarriers;
    auto task_info_empty_barriers = task_info_full_barriers +
        interleaved_scheduler_t::kNumScheduleStages;
    auto task_infos = reinterpret_cast<task_info_t*>(
        task_info_empty_barriers +
        interleaved_scheduler_t::kNumScheduleStages);
    // Stream-K: CTA-wide broadcast of the tile arrival ticket (epilogue thread 0 ->
    // all epilogue threads), 16 B after the task mailbox.
    auto smem_streamk_ticket = reinterpret_cast<uint32_t*>(
        task_infos + interleaved_scheduler_t::kNumScheduleStages);
    constexpr uint32_t kInterleavedSchedulerSMEMBytes =
        2 * interleaved_scheduler_t::kNumScheduleStages * sizeof(Barrier) +
        interleaved_scheduler_t::kNumScheduleStages * sizeof(task_info_t) + 16u;
    DG_STATIC_ASSERT(
        kInterleavedSchedulerSMEMBytes ==
            fused_layout::kSM90InterleavedSchedulerSMEMBytes,
        "Host and device scheduler shared-memory layouts disagree");
    constexpr uint32_t kInterleavedSMEMEnd =
        SMEM_BEFORE_BARRIER_SIZE + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        SMEM_KSPLIT_REDUCE_SIZE +
        kNumBaseBarriers * sizeof(Barrier) +
        kInterleavedSchedulerSMEMBytes;
    DG_STATIC_ASSERT(!kUseInterleavedScheduler || kInterleavedSMEMEnd <= 232448,
                     "Interleaved scheduler exceeds the SM90 shared-memory capacity");
    // The BM8 hosts launch with the full 232448 B; the multi-K-block layout must fit
    // regardless of the scheduler variant: 2 blocks x 4 stages = 4 x (2048 + 40960 +
    // 256) = 173056 B of stages plus the fixed regions and barriers.
    DG_STATIC_ASSERT(kKBlocksPerStage == 1 || kInterleavedSMEMEnd <= 232448,
                     "Multi-K-block pipeline exceeds the SM90 shared-memory capacity");

    // =====================================================================
    // Initialization
    // =====================================================================
    // QoQ decodes with plain ALU ops (shift/mask/subtract, see `decode_stage_rf`),
    // so it has no LUT to stage; only the FP4 paths fill the 1 KB LUT.
    if constexpr (!kQoQ) {
        if (thread_idx < 64) {
            reinterpret_cast<uint4*>(smem_nvfp4_lut)[thread_idx] =
                kMXFP4 ?
                    reinterpret_cast<const uint4*>(deep_gemm::nvfp4::kE2M1AndE8M0RelToFp8Lut)[thread_idx] :
                    reinterpret_cast<const uint4*>(deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut)[thread_idx];
        }
    }
    // Per-lane replicated RF LUT: only when its region is actually allocated
    // (SMEM_RF_LUT_REP_SIZE, currently 0: the RF decoders gather from the 1 KB LUT
    // above). With a 0-byte region the 32 KB fill would land in the GEMM stage
    // area as dead stores in the prologue of every CTA.
    if constexpr (kRFDecode && kMXFP4 && SMEM_RF_LUT_REP_SIZE > 0) {
        const uint2* lut_src = reinterpret_cast<const uint2*>(deep_gemm::nvfp4::kE2M1AndE8M0RelToFp8Lut);
        for (uint32_t i = thread_idx; i < 128u * 32u; i += blockDim.x)
            smem_rf_lut[i] = lut_src[i >> 5];
    }

    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Producer arrivals: A(+SFA) + B(TMA+SFB). SFB is copied with
                // cp.async.bulk and counted as B-loader transaction bytes, so
                // it does not need a separate producer arrival.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
            if constexpr (kUseInterleavedScheduler) {
                #pragma unroll
                for (uint32_t i = 0;
                     i < interleaved_scheduler_t::kNumScheduleStages;
                     ++ i) {
                    task_info_full_barriers[i].init(1);
                    task_info_empty_barriers[i].init(kNumEpilogueWarps);
                }
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    // PDL (library-wide DG_PDL -> LaunchArgs::enable_pdl): launched with programmatic
    // stream serialization this grid may start while the Fable frontend is still
    // running. Everything above touches only SMEM, m-barriers, TMA descriptors and
    // the constant LUT; from here on we read frontend outputs (topk_idx, x, x_sf),
    // so every thread blocks until the prerequisite grid has completed and its
    // memory is visible. No-op when launched without the attribute.
    #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;" ::: "memory");
    #endif
    // DG_FE_SELECT_IN_MEGA=1 (fe_keys != nullptr): the frontend ended after its router CTAs
    // stored the 384 keys per token; one otherwise idle prologue warp per token (warps 4..,
    // warps 0..2 are busy with the smem / m-barrier init above) selects the top-8 + softmax
    // and writes THIS rank's topk_idx / topk_weights (every CTA writes the same values; the
    // dispatch warps read them after the __syncthreads below). Frontend keys are complete
    // because the frontend kernel finished before this grid (or before griddepcontrol.wait).
    if (fe_keys != nullptr and warp_idx >= 4 and warp_idx - 4 < num_tokens and warp_idx - 4 < 8) {
        const uint32_t t = warp_idx - 4;
        const uint32_t* keys_t = fe_keys + t * kNumExperts;
        if (__ldcg(keys_t) == 0u) {
            // Zero first key = unrouted token (a real logit key is never 0: its low 16 bits are 0xFFFF - expert).
            // Written by the profiling driver's forced-balanced routing (DG_PROFILE_FORCE_BALANCED, the Mega-only
            // scope's inactive rows) and by the Fable cc frontend for an all-zero hidden row (DG_FE_ZERO_ROW_UNROUTED,
            // the E2E harness's padding rows; x = 0 -> y = 0 exactly): topk_idx -1 / weights 0, which the dispatch
            // treats as "no expert" and the combine as an all-zero output row (empty slot mask -> zeros stored).
            if (lane_idx < kNumTopk) {
                input_topk_idx_buffer.get_base_ptr<int64_t>()[t * kNumTopk + lane_idx] = -1;
                input_topk_weights_buffer.get_base_ptr<float>()[t * kNumTopk + lane_idx] = 0.0f;
            }
        } else {
            fable_cc::select_topk8_compact384(keys_t, static_cast<int>(lane_idx), static_cast<int>(t),
                                              input_topk_idx_buffer.get_base_ptr<int64_t>(),
                                              input_topk_weights_buffer.get_base_ptr<float>());
        }
        if (thread_idx == 4 * 32) stamp_max(16);       // slot 16: max FE-select-in-Mega done
    }
    // topk_idx / topk_weights loads: read-only (__ldg, ld.global.nc) when the frontend produced
    // them; under DG_FE_SELECT_IN_MEGA the prologue warps of THIS kernel wrote them, so the
    // non-coherent path may return stale data -> ld.global.cg (L2-coherent) instead.
    // EP16 GIN: the peer's rows [T, 2T) are written by this kernel too (copied in from the NIC
    // receive region), and every top-k entry is a global expert, mapped to this node's 192 on read
    // (-1 when it lives on the other node, whose launch computes it).
    constexpr bool kEp16Gin = DG_H20_EP16_GIN;
    const auto ld_topk_idx = [&](const int64_t* p) -> int64_t {
        if constexpr (kEp16Gin) {
            const int64_t e = static_cast<int64_t>(__ldcg(reinterpret_cast<const long long*>(p)));
            const int64_t first = static_cast<int64_t>(ep16_gin.node) * kNumExperts;
            return (e >= first and e < first + kNumExperts) ? e - first : -1;
        }
        return fe_keys != nullptr ? static_cast<int64_t>(__ldcg(reinterpret_cast<const long long*>(p))) : __ldg(p);
    };
    const auto ld_topk_weight = [&](const float* p) -> float {
        return (kEp16Gin or fe_keys != nullptr) ? __ldcg(p) : __ldg(p);
    };
    // EP16 GIN window regions (see H20Ep16Gin) and this launch's signal target: the epoch word
    // counts this rank's launches (SM0's cleanup adds 1 after every CTA has read it here), each
    // launch of the peer adds 1 to every signal.
    constexpr uint32_t kEp16RowBytes = kHidden + kHidden / 128 * 4 + kNumTopk * 12;
    constexpr uint32_t kEp16YBytes = kHidden * sizeof(nv_bfloat16);
    const uint64_t ep16_in_send = 0;
    const uint64_t ep16_in_recv = static_cast<uint64_t>(ep16_gin.max_own) * kEp16RowBytes;
    const uint64_t ep16_y_send = 2ull * ep16_gin.max_own * kEp16RowBytes;
    const uint64_t ep16_y_recv = ep16_y_send + static_cast<uint64_t>(ep16_gin.max_own) * kEp16YBytes;
    const uint64_t ep16_epoch_off = ep16_y_recv + static_cast<uint64_t>(ep16_gin.max_own) * kEp16YBytes;  // + 4: send counter
    const uint64_t ep16_target = kEp16Gin ?
        static_cast<uint64_t>(ptx::ld_volatile(reinterpret_cast<uint32_t*>(ep16_gin.base + ep16_epoch_off))) + 1ull : 0ull;
#if DG_H20_EP16_GIN_DEVICE
    const auto ep16_put = [&](const uint64_t dst_off, const uint64_t src_off, const uint64_t bytes, const uint32_t sig) {
        const ncclDevComm& c = *static_cast<const ncclDevComm*>(ep16_gin.comm);
        const ncclWindow_t win = static_cast<ncclWindow_t>(ep16_gin.win);
        // Signal s on GIN context s % contexts (its own QP), the waiter reads it on the same one
        ncclGin(c, static_cast<int>(sig % ep16_gin.num_contexts))
            .put(ncclTeamWorld(c), static_cast<int>(1u - ep16_gin.node), win, dst_off, win, src_off, bytes,
                          ncclGin_SignalInc{static_cast<ncclGinSignal_t>(sig)});
    };
    const auto ep16_wait = [&](const uint32_t sig, const uint32_t tag) {
        const ncclDevComm& c = *static_cast<const ncclDevComm*>(ep16_gin.comm);
        DG_SPIN_WHILE(ncclGin(c, static_cast<int>(sig % ep16_gin.num_contexts)).readSignal(static_cast<ncclGinSignal_t>(sig)) < ep16_target, tag);
    };
#else
    const auto ep16_put = [&](const uint64_t, const uint64_t, const uint64_t, const uint32_t) {};
    const auto ep16_wait = [&](const uint32_t, const uint32_t) {};
#endif
    const uint32_t ep16_num_own = kEp16Gin ? ep16_gin.num_own : num_tokens;
    // Fast NVLink-barrier epilogue: every thread snapshots the done count BEFORE
    // the kernel-start __syncthreads (see fused_comm::nvlink_barrier for why this
    // is race-free: SM0's first write of the word this launch is ordered after the
    // first barrier's prologue grid sync, which every CTA reaches after this sync).
    const uint32_t nvl_done_base = kNvlFastEpilogue ?
        ptx::ld_volatile(workspace.get_nvl_done_count_ptr()) : 0u;
    __syncthreads();
    if (thread_idx == 0) stamp_max(12);

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = fused_sched::MegaMoEScheduler<
        BLOCK_M, TASK_BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, 1>(workspace);
    auto interleaved_scheduler = interleaved_scheduler_t(
        workspace,
        task_info_full_barriers,
        task_info_empty_barriers,
        task_infos);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        k_block_idx += kKBlocksPerStage;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    constexpr uint32_t kSplitMDecodeBarrierIdx          = 8;
    constexpr uint32_t kKSplitReduceBarrierIdx          = 9;  // kHalfTileTasks WG1 -> WG0

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        kUseInterleavedScheduler ? 64 : 40;
    constexpr uint32_t kNumEpilogueRegisters    = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    // `first_worker_idx` of a task that is NOT a stream-K segment (wave-scheduler task
    // of a launch where stream-K is compiled in but inactive, e.g. >= kNumSMs L1 tasks
    // under unbalanced routing): its split-K tail halves must use the split-K
    // publisher/finisher protocol, not the stream-K per-worker slots (a genuine
    // stream-K segment may have first worker 0, so 0 cannot mark "not stream-K").
    constexpr uint32_t kNotStreamKWorker = 0xffffffffu;

    const auto for_each_static_selected_block = [&](auto&& func) {
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(),
                          block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == fused_sched::BlockPhase::None)
                break;
            if (block_phase == fused_sched::BlockPhase::Linear1) {
                func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear1>{},
                     local_expert_idx, L1_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
                     scheduler.get_current_pool_block_offset() + m_block_idx,
                     scheduler.template get_valid_m<false>(), 0u, 1u, 0u, kNotStreamKWorker);
            } else {
                func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear2>{},
                     local_expert_idx, L2_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
                     scheduler.get_current_pool_block_offset() + m_block_idx,
                     scheduler.template get_valid_m<false>(), 0u, 1u, 0u, kNotStreamKWorker);
            }
        }
    };

    // K-block range of K split `k_split_idx` (of `num_k_splits`) of a task with
    // `total_k_blocks` K-blocks: whole stages (kKBlocksPerStage blocks) split as evenly
    // as possible with the LAST splits taking the extra stages (the finisher never
    // waits for a longer publisher): 2-way L1 24 -> 12 / 12, L2 10 -> 4 / 6 (as
    // before), 3-way L2 10 -> 2 / 4 / 4.
    const auto get_split_k_num_stages = [](const uint32_t& total_stages,
                                           const uint32_t& k_split_idx,
                                           const uint32_t& num_k_splits) -> uint32_t {
        const uint32_t base = total_stages / num_k_splits, rem = total_stages % num_k_splits;
        return base + (k_split_idx + rem >= num_k_splits ? 1u : 0u);
    };
    const auto get_split_k_num_blocks = [&](const uint32_t& total_k_blocks,
                                            const uint32_t& k_split_idx,
                                            const uint32_t& num_k_splits) -> uint32_t {
        if (num_k_splits == 1u)
            return total_k_blocks;
        return get_split_k_num_stages(total_k_blocks / kKBlocksPerStage, k_split_idx, num_k_splits) *
               kKBlocksPerStage;
    };
    const auto get_split_k_block_begin = [&](const uint32_t& total_k_blocks,
                                             const uint32_t& k_split_idx,
                                             const uint32_t& num_k_splits) -> uint32_t {
        uint32_t begin = 0;
        for (uint32_t s = 0; s < k_split_idx; ++ s)
            begin += get_split_k_num_blocks(total_k_blocks, s, num_k_splits);
        return begin;
    };
    const auto invoke_interleaved_task = [&](const task_info_t& task_info,
                                              auto&& func) {
        // Split-K tasks (L1 tail: kSplitKL1, L2 tail: kSplitKL2) cover the K-block
        // range of their K split; every other task covers the whole K.
        // Stream-K segments carry an explicit K-block range (all tasks of a
        // stream-K launch are segments, most of them the whole K).
        // (Marker: only stream-K segments carry a non-zero K-block end; every warp
        // owns its own scheduler object and only the producer knows the mode.)
        const bool is_streamk = kStreamK && task_info.get_k_block_end() != 0u;
        if (task_info.block_phase == fused_sched::BlockPhase::Linear1) {
            const uint32_t num_k_splits = (kSplitKL1 || kStreamK) ? task_info.get_num_k_splits() : 1u;
            const uint32_t k_split_idx = (kSplitKL1 || kStreamK) ? task_info.get_k_split_idx() : 0u;
            const uint32_t num_k_blocks = is_streamk ?
                task_info.get_k_block_end() - task_info.get_k_block_begin() :
                get_split_k_num_blocks(L1_SHAPE_K / BLOCK_K, k_split_idx, num_k_splits);
            const uint32_t k_block_begin = is_streamk ? task_info.get_k_block_begin() :
                get_split_k_block_begin(L1_SHAPE_K / BLOCK_K, k_split_idx, num_k_splits);
            func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear1>{},
                 task_info.local_expert_idx, num_k_blocks,
                 task_info.m_block_idx, task_info.n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m,
                 k_split_idx, num_k_splits, k_block_begin,
                 is_streamk ? task_info.get_first_worker_idx() : kNotStreamKWorker);
        } else {
            const uint32_t num_k_splits = (kSplitKL2 || kStreamK) ? task_info.get_num_k_splits() : 1u;
            const uint32_t k_split_idx = (kSplitKL2 || kStreamK) ? task_info.get_k_split_idx() : 0u;
            const uint32_t num_k_blocks = is_streamk ?
                task_info.get_k_block_end() - task_info.get_k_block_begin() :
                get_split_k_num_blocks(L2_SHAPE_K / BLOCK_K, k_split_idx, num_k_splits);
            const uint32_t k_block_begin = is_streamk ? task_info.get_k_block_begin() :
                get_split_k_block_begin(L2_SHAPE_K / BLOCK_K, k_split_idx, num_k_splits);
            func(std::integral_constant<fused_sched::BlockPhase, fused_sched::BlockPhase::Linear2>{},
                 task_info.local_expert_idx, num_k_blocks,
                 task_info.m_block_idx, task_info.n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m,
                 k_split_idx, num_k_splits, k_block_begin,
                 is_streamk ? task_info.get_first_worker_idx() : kNotStreamKWorker);
        }
    };

    const auto for_each_published_block = [&](auto&& func) {
        task_info_t task_info;
        while (interleaved_scheduler.get_published_task(task_info))
            invoke_interleaved_task(task_info, func);
    };

    const auto produce_interleaved_blocks = [&](auto&& func) {
        if constexpr (kPushDoneFlags) {
            interleaved_scheduler.fetch_expert_recv_count(
                workspace.get_push_done_count_ptr(),
                static_cast<int>(kNumRanks * (ptx::ld_volatile(workspace.get_push_epoch_ptr()) + 1u)));
        } else {
            interleaved_scheduler.fetch_expert_recv_count();
        }
        while (true) {
            interleaved_scheduler.wait_task_slot_empty();
            const auto task_info = interleaved_scheduler.claim_next_task();
            interleaved_scheduler.publish_task(task_info);
            if (!task_info.is_valid())
                break;
            invoke_interleaved_task(task_info, func);
        }
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if constexpr (kPushDoneFlags) {
                // Next launch's DONE target (see `kPushDoneFlags`)
                if (thread_idx == 0)
                    *workspace.get_push_epoch_ptr() = ptx::ld_volatile(workspace.get_push_epoch_ptr()) + 1u;
            }
            if constexpr (kEp16Gin) {
                // Next launch's GIN signal target (every CTA read this launch's before the grid sync)
                if (thread_idx == 0)
                    *reinterpret_cast<uint32_t*>(ep16_gin.base + ep16_epoch_off) = static_cast<uint32_t>(ep16_target);
            }
            if constexpr (kCombineDynamic) {
                // Next dynamic-combine launch's ticket word (see `kCombineDynamic`)
                if (thread_idx == 0) {
                    const auto epoch = ptx::ld_volatile(workspace.get_combine_epoch_ptr());
                    *workspace.get_combine_ticket_ptr(epoch + 1u) = 0u;
                    *workspace.get_combine_epoch_ptr() = epoch + 1u;
                }
            }
            if constexpr (kUseInterleavedScheduler) {
                if (thread_idx == 0) {
                    *workspace.get_l1_task_count_ptr() = 0;
                    *workspace.get_l2_task_count_ptr() = 0;
                }
                if constexpr (kSplitKL1 || kSplitKL2 || kStreamK) {
                    // L1 and L2 flag slots are contiguous (L1 first)
                    for (uint32_t i = thread_idx; i < fused_layout::kSM90SplitKNumSlots; i += kNumDispatchThreads)
                        *workspace.get_splitk_l1_flag_ptr(0, i) = 0;
                }
            }
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = kStridedPool ?
                    i * kPushBlocksPerExpert : scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            if (warp_idx < kNumActiveDispatchWarps) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            ld_topk_idx(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                        if (expert_idx >= 0)
                            process(i * kNumTopk + lane_idx, expert_idx);
                    }
                    __syncwarp();
                }
            }
        };

        if constexpr (kEp16Gin) {
            // The pull path only: push dispatch (<= 16 global tokens) reads the rows before routing.
            DG_STATIC_ASSERT(!kPushDispatch, "EP16 GIN needs the pull dispatch");
            DG_STATIC_ASSERT(kNumDispatchThreads % 32 == 0, "Invalid dispatch threads for the EP16 row items");
            constexpr uint32_t kSFBytes = kHidden / 128 * 4;
            constexpr uint32_t kRowUint4 = kHidden / 16;
            constexpr uint32_t kRowSF = kSFBytes / 4;
            constexpr uint32_t kGridThreads = kNumSMs * kNumDispatchThreads;
            const uint32_t gtid = sm_idx * kNumDispatchThreads + thread_idx;
            const uint32_t num_x_items = ep16_num_own * kRowUint4;
            const uint32_t num_sf_items = ep16_num_own * kRowSF;
            const uint32_t num_topk_items = ep16_num_own * kNumTopk;
            // Send: the own rows [0, T) packed into in_send as grid-flattened items (x 16-byte chunks,
            // SF, top-k, weights), at most a few per dispatch thread of the first CTAs, so the pack is one
            // round of loads instead of a serial walk over the rows on one CTA. Every packing warp
            // releases and counts in; the last one puts the T rows into the peer's in_recv (signal 0)
            // and resets the counter (the next launch's warps start after this kernel).
            {
                const uint32_t num_items = num_x_items + num_sf_items + 2 * num_topk_items;
                const uint32_t num_send_warps = math::ceil_div(num_items < kGridThreads ? num_items : kGridThreads, 32u);
                if (gtid / 32 < num_send_warps) {
                    for (uint32_t i = gtid; i < num_items; i += kGridThreads) {
                        if (i < num_x_items) {
                            const uint32_t t = i / kRowUint4, c = i % kRowUint4;
                            reinterpret_cast<uint4*>(ep16_gin.base + ep16_in_send + static_cast<uint64_t>(t) * kEp16RowBytes)[c] =
                                __ldcg(input_token_buffer.get_data_buffer(t).get_base_ptr<uint4>() + c);
                        } else if (i < num_x_items + num_sf_items) {
                            const uint32_t j = i - num_x_items, t = j / kRowSF, s = j % kRowSF;
                            reinterpret_cast<float*>(ep16_gin.base + ep16_in_send + static_cast<uint64_t>(t) * kEp16RowBytes + kHidden)[s] =
                                __ldcg(input_sf_buffer.get_data_buffer(t).get_base_ptr<float>() + s);
                        } else if (i < num_x_items + num_sf_items + num_topk_items) {
                            const uint32_t j = i - num_x_items - num_sf_items, t = j / kNumTopk;
                            reinterpret_cast<long long*>(ep16_gin.base + ep16_in_send + static_cast<uint64_t>(t) * kEp16RowBytes + kHidden + kSFBytes)[j % kNumTopk] =
                                __ldcg(reinterpret_cast<const long long*>(input_topk_idx_buffer.get_base_ptr<int64_t>()) + j);
                        } else {
                            const uint32_t j = i - num_x_items - num_sf_items - num_topk_items, t = j / kNumTopk;
                            reinterpret_cast<float*>(ep16_gin.base + ep16_in_send + static_cast<uint64_t>(t) * kEp16RowBytes + kHidden + kSFBytes + kNumTopk * 8)[j % kNumTopk] =
                                __ldcg(input_topk_weights_buffer.get_base_ptr<float>() + j);
                        }
                    }
                    __syncwarp();
                    if (lane_idx == 0) {
                        stamp_max(55);
                        __threadfence_system();
                        auto* counter = reinterpret_cast<uint32_t*>(ep16_gin.base + ep16_epoch_off + 4);
                        if (atomicAdd(counter, 1u) + 1u == num_send_warps) {
                            *counter = 0u;
                            __threadfence_system();
                            ep16_put(ep16_in_recv, ep16_in_send, static_cast<uint64_t>(ep16_num_own) * kEp16RowBytes, 0);
                            stamp_max(48);
                        }
                    }
                    __syncwarp();
                }
            }
            // Receive x / SF: the peer rows from in_recv into the token buffer rows [T, 2T), the same
            // grid-flattened items; the NVLink ranks pull them after barrier #1.
            {
                const uint32_t num_items = num_x_items + num_sf_items;
                if (gtid - lane_idx < num_items) {
                    if (lane_idx == 0) {
                        ep16_wait(0, 5001);
                        stamp_max(49);
                    }
                    __syncwarp();
                    for (uint32_t i = gtid; i < num_items; i += kGridThreads) {
                        if (i < num_x_items) {
                            const uint32_t t = i / kRowUint4, c = i % kRowUint4;
                            input_token_buffer.get_data_buffer(ep16_num_own + t).get_base_ptr<uint4>()[c] =
                                __ldcv(reinterpret_cast<const uint4*>(ep16_gin.base + ep16_in_recv + static_cast<uint64_t>(t) * kEp16RowBytes) + c);
                        } else {
                            const uint32_t j = i - num_x_items, t = j / kRowSF, s = j % kRowSF;
                            input_sf_buffer.get_data_buffer(ep16_num_own + t).get_base_ptr<float>()[s] =
                                __ldcv(reinterpret_cast<const float*>(ep16_gin.base + ep16_in_recv + static_cast<uint64_t>(t) * kEp16RowBytes + kHidden) + s);
                        }
                    }
                    __syncwarp();
                    if (lane_idx == 0)
                        stamp_max(50);
                }
            }
            // Receive top-k / weights of the peer rows this warp routes (`read_topk_idx`'s row and lane
            // mapping, so each lane reads back its own store): one load per lane.
            if (warp_idx < kNumActiveDispatchWarps) {
                for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
                    const uint32_t t = i + lane_idx / kNumTopk;
                    const bool peer = lane_idx < kNumActivateLanes and t >= ep16_num_own and t < num_tokens;
                    if (__any_sync(0xffffffff, peer)) {
                        if (lane_idx == 0)
                            ep16_wait(0, 5003);
                        __syncwarp();
                        if (peer) {
                            const uint8_t* src = ep16_gin.base + ep16_in_recv + static_cast<uint64_t>(t - ep16_num_own) * kEp16RowBytes +
                                                 kHidden + kSFBytes;
                            const uint32_t k = lane_idx % kNumTopk;
                            input_topk_idx_buffer.get_base_ptr<int64_t>()[t * kNumTopk + k] =
                                __ldcv(reinterpret_cast<const long long*>(src) + k);
                            input_topk_weights_buffer.get_base_ptr<float>()[t * kNumTopk + k] =
                                __ldcv(reinterpret_cast<const float*>(src + kNumTopk * 8) + k);
                        }
                    }
                    __syncwarp();
                }
            }
        }

        // Lean routing (kLeanRouting; host env DG_FP4_LEAN_ROUTING, default 1).
        // Baseline: every CTA adds `(1 << 32) | count` to ALL 384 `expert_send_count`
        // words (78 x 384 = 30k serialised L2 atomics, ~5 us at any M, most of them
        // count 0); the high word (CTA arrivals) is forwarded by the broadcast into
        // `expert_recv_count_sum`, whose high word the destination's scheduler polls
        // for kNumSMs * kNumRanks (completeness). The grid sync before the broadcast
        // already guarantees every CTA's count landed, so the per-CTA arrival word is
        // redundant: lean routing adds only non-zero counts (no high word) and the
        // broadcast supplies the constant kNumSMs high word instead. Push dispatch
        // (kLeanPush) needs neither the local send counts (only the broadcast read
        // them), nor the extra grid sync, nor the cross-rank broadcast: the per-row
        // remote tickets already carry the low word, all of them are ordered before
        // this rank's barrier #1 signal by the barrier's prologue grid sync, and the
        // DESTINATION adds the kNumSMs * kNumRanks high word to its own 48 sums after
        // barrier #1 (one local atomic per expert). Data seen by the scheduler /
        // dispatch (`expert_recv_count`, `expert_recv_count_sum`) is unchanged.
        if constexpr (!kLeanPush) {
            // Count tokens per expert
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                atomicAdd_block(smem_expert_count + expert_idx, 1);
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Stake out per-expert SM offsets via global atomic
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t count = smem_expert_count[i];
                if constexpr (kLeanRouting) {
                    if (count != 0)
                        smem_expert_count[i] = static_cast<uint32_t>(ptx::atomic_add(
                            workspace.get_expert_send_count_ptr(i), static_cast<uint64_t>(count)));
                } else {
                    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(count);
                    smem_expert_count[i] = static_cast<uint32_t>(
                        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
                }
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        }
        if (thread_idx == 0) stamp_max(8);

        if constexpr (kPushDispatch) {
            // Push: one routed ROW (token, top-k slot) per dispatch warp, all rows of
            // the rank in flight across the grid at once (M <= 16 global tokens means
            // <= 16 rows per rank, so the old 4-tokens-per-warp packing serialised up
            // to 32 x 3 KB of remote stores behind one warp: +5 us in the routing
            // phase at M=16). Lane 0 takes the remote ticket first so its NVLink
            // round trip overlaps the local row / SF loads; the warp then streams the
            // row (16 B per lane per store) into the destination pool.
            if (warp_idx < kNumActiveDispatchWarps) {
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                constexpr uint32_t kNumTokenChunksPerLane = kHidden / (16 * 32);
                DG_STATIC_ASSERT(kHidden % 512 == 0 and kNumSFFloats <= 32, "Invalid token / SF shape");
                const uint32_t num_rows = num_tokens * kNumTopk;
                for (uint32_t r = sm_idx * kNumActiveDispatchWarps + warp_idx; r < num_rows;
                     r += kNumSMs * kNumActiveDispatchWarps) {
                    const uint32_t src_token_idx = r / kNumTopk, src_topk_idx = r % kNumTopk;
                    const int expert_idx = static_cast<int>(
                        ld_topk_idx(input_topk_idx_buffer.get_base_ptr<int64_t>() + r));
                    if (expert_idx < 0)
                        continue;
                    const uint32_t dr = static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
                    const uint32_t de = static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank;
                    uint32_t row_idx = 0;
                    if (lane_idx == 0)
                        row_idx = static_cast<uint32_t>(ptx::atomic_add_sys(
                            sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(de), dr), 1ull));
                    const auto* src_token = input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint4>();
                    uint4 row[kNumTokenChunksPerLane];
                    #pragma unroll
                    for (uint32_t c = 0; c < kNumTokenChunksPerLane; ++ c)
                        row[c] = __ldg(src_token + c * 32 + lane_idx);
                    const float sf = lane_idx < kNumSFFloats ?
                        __ldg(input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>() + lane_idx) : 0.0f;
                    const float weight = ld_topk_weight(input_topk_weights_buffer.get_base_ptr<float>() + r);
                    row_idx = __shfl_sync(0xffffffff, row_idx, 0);
                    DG_TRAP_ONLY_DEVICE_ASSERT(row_idx < kPushBlocksPerExpert * BLOCK_M);
                    const uint32_t pool_token_idx = de * kPushBlocksPerExpert * BLOCK_M + row_idx;
                    auto* dst_token = sym_buffer.map(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr<uint4>(), dr);
                    #pragma unroll
                    for (uint32_t c = 0; c < kNumTokenChunksPerLane; ++ c)
                        dst_token[c * 32 + lane_idx] = row[c];
                    if (lane_idx < kNumSFFloats)
                        sym_buffer.map(l1_sf_buffer.get_base_ptr<float>(), dr)
                            [lane_idx * kNumPaddedSFPoolTokens + pool_token_idx] = sf;
                    if (lane_idx == 0) {
                        *sym_buffer.map(l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>(), dr) = weight;
                        *sym_buffer.map(workspace.get_token_src_metadata_ptr(pool_token_idx), dr) =
                            {static_cast<uint32_t>(sym_buffer.rank_idx), src_token_idx, src_topk_idx};
                    }
                    __syncwarp();
                }
            }
            if (thread_idx == 0) stamp_max(9);  // push issued
            if constexpr (kPushDoneFlags) {
                // CTA arrival ticket; the last CTA signals DONE to every rank. The
                // kNumRanks release.sys signals are issued by kNumRanks lanes of warp 0
                // at once (one warp-wide sys fence), not serially by one thread: each
                // release.sys drains this SM's outstanding NVLink stores (~1.5 us on
                // H20), and 8 in a row put +12 us on the critical path (probe slot 10).
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
                if (warp_idx == 0) {
                    DG_STATIC_ASSERT(kNumRanks <= 32, "Too many ranks for one signalling warp");
                    uint32_t arrived = 0;
                    if (lane_idx == 0)
                        arrived = ptx::atomic_add_acq_rel(workspace.get_push_cta_arrival_ptr(), 1u);
                    arrived = __shfl_sync(0xffffffff, arrived, 0);
                    if (arrived == kNumSMs - 1) {
                        if (lane_idx == 0)
                            *workspace.get_push_cta_arrival_ptr() = 0;
                        if (lane_idx < kNumRanks)
                            ptx::red_add_rel_sys(sym_buffer.map(workspace.get_push_done_count_ptr(), lane_idx), 1);
                    }
                    __syncwarp();
                }
            }
        } else {
            // Write source token-topk indices to remote ranks
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
                const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                    expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            });
            if (thread_idx == 0) stamp_max(9);
        }

        // Lean push: no cross-CTA total is needed before barrier #1 (the barrier's
        // prologue grid sync orders the pushes), so the routing grid sync is dropped.
        if constexpr (!kLeanPush) {
            fused_comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
            );
        }
        if (thread_idx == 0) stamp_max(10);

        const auto broadcast_expert_status = [&](const uint32_t& i) {
            const auto dst_rank_idx = i / kNumExpertsPerRank;
            const auto dst_local_expert_idx = i % kNumExpertsPerRank;
            // Lean routing: the send count carries no arrival high word; supply the
            // constant the destination's scheduler expects (kNumSMs per source rank).
            const uint64_t expert_status = kLeanRouting ?
                ((static_cast<uint64_t>(kNumSMs) << 32) |
                 (*workspace.get_expert_send_count_ptr(i) & 0xffffffffull)) :
                *workspace.get_expert_send_count_ptr(i);
            *sym_buffer.map(
                workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                dst_rank_idx) = expert_status & 0xffffffff;
            // Push mode: the low word (row count) was already added by the per-row
            // tickets (all returned before this rank's grid sync), so add only the
            // high word that finalises the count for the destination's scheduler.
            ptx::atomic_add_sys(
                sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                kPushDispatch ? (expert_status & 0xffffffff00000000ull) : expert_status);
        };
        if constexpr (kLeanPush) {
            // No broadcast: the destination finalises its counts after barrier #1.
        } else if constexpr (kDistributedExpertBcast) {
            // Spread the per-expert cross-rank count updates over every SM
            // (<= 3 blocking sys-scope atomics per thread) instead of 12 serial
            // rounds on SM 0; the NVLink barrier below then needs its grid-sync
            // prologue so SM 0 only signals after all SMs have published.
            for (uint32_t i = sm_idx + thread_idx * kNumSMs; i < kNumExperts;
                 i += kNumSMs * kNumDispatchThreads)
                broadcast_expert_status(i);
        } else if (sm_idx == 0 and thread_idx < kNumActiveDispatchThreads) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumActiveDispatchThreads)
                broadcast_expert_status(i);
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        unsigned long long t_before_nvl_barrier = 0;
        if (thread_idx == 0) {
            stamp_max(11);
            if (phase_stamps != nullptr)
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_before_nvl_barrier));
        }

        if constexpr (!kPushDoneFlags) {
            fused_comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                kDistributedExpertBcast || kLeanPush, true,
                /* fast epilogue: barrier #1 of the launch */ kNvlFastEpilogue, nvl_done_base, 1u);
            if (thread_idx == 0) {
                stamp_max(1);
                if (sm_idx == 0) {
                    stamp_accumulate(13);
                    if (phase_stamps != nullptr) {
                        atomicAdd(phase_stamps + 14, 1ull);
                        unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                        atomicAdd(phase_stamps + 16, t - t_before_nvl_barrier);  // barrier wait incl. skew
                    }
                }
            }
        }

        if constexpr (kPushDispatch) {
            // Every rank's rows are in the local pool (ordered by barrier #1). SM e
            // (e < experts per rank) publishes expert e's L1 arrival counts, one block
            // per lane, with release.gpu: the loaders' acquire on the count then also
            // covers the remotely written rows (this thread acquired barrier #1).
            if (warp_idx == 0 and sm_idx < kNumExpertsPerRank) {
                if constexpr (kPushDoneFlags) {
                    // Wait for every rank's DONE (replaces barrier #1 for this publisher;
                    // slots 1 / 13 / 14 / 16 keep their meaning: SM0's wait incl. skew).
                    if (lane_idx == 0) {
                        const int target = static_cast<int>(
                            kNumRanks * (ptx::ld_volatile(workspace.get_push_epoch_ptr()) + 1u));
                        DG_SPIN_WHILE(static_cast<int>(ptx::ld_volatile(reinterpret_cast<const uint32_t*>(
                            workspace.get_push_done_count_ptr()))) - target < 0, 1094);
                        DG_SPIN_WHILE(ptx::ld_acq_sys(workspace.get_push_done_count_ptr()) - target < 0, 1095);
                        stamp_max(1);
                        if (sm_idx == 0) {
                            stamp_accumulate(13);
                            if (phase_stamps != nullptr) {
                                atomicAdd(phase_stamps + 14, 1ull);
                                unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                                atomicAdd(phase_stamps + 16, t - t_before_nvl_barrier);
                            }
                        }
                    }
                    __syncwarp();
                }
                uint32_t num_recv_tokens;
                if constexpr (kLeanPush) {
                    // Finalise expert `sm_idx`'s count: every rank's row tickets landed
                    // before its barrier #1 signal (acquired by this CTA), so add the
                    // completeness high word the scheduler polls for and take the
                    // total from the same atomic. RELEASE (gpu): the scheduler's
                    // acquire poll of this word then also covers the remotely written
                    // rows (this CTA acquired barrier #1), so the lean-push L1 loaders
                    // skip the per-task arrival-count spin (see the A loader).
                    uint32_t low = 0;
                    if (lane_idx == 0)
                        low = static_cast<uint32_t>(ptx::atomic_add_rel_gpu(
                            workspace.get_expert_recv_count_sum_ptr(sm_idx),
                            static_cast<uint64_t>(kNumSMs * kNumRanks) << 32));
                    num_recv_tokens = __shfl_sync(0xffffffff, low, 0);
                } else {
                    num_recv_tokens = static_cast<uint32_t>(
                        ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(sm_idx)));
                }
                const uint32_t num_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                if (lane_idx < num_blocks)
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(sm_idx * kPushBlocksPerExpert + lane_idx),
                        cute::min(num_recv_tokens - lane_idx * BLOCK_M, BLOCK_M));
                __syncwarp();
            }
            if (thread_idx == 0) stamp_max(2);  // pool ready
        }

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        if (!kPushDispatch and warp_idx < kNumActiveDispatchWarps) {
            uint32_t pull_mbarrier_phase = 0;
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
            const auto pull_mbarrier = dispatch_barriers[warp_idx];

            scheduler.fetch_expert_recv_count();

            constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
            int      current_expert_idx = -1;
            uint32_t stored_rank_count[kNumRanksPerLane] = {};
            uint32_t expert_start_idx = 0, expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumActiveDispatchWarps;
            for (uint32_t token_idx = sm_idx * kNumActiveDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }
                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection (identical to SM100)
                uint32_t current_rank_in_expert_idx;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                // TMA pull token data into SMEM
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                // Strided-pool debug: expert e's rows live at block e * stride (as
                // under push dispatch) instead of the dense prefix-sum block.
                const uint32_t pull_pool_block_offset = kStridedPool ?
                    static_cast<uint32_t>(current_expert_idx) * kPushBlocksPerExpert : expert_pool_block_offset;
                const uint32_t pool_token_idx =
                    pull_pool_block_offset * BLOCK_M + token_idx_in_expert;
                // Issue the remote top-k weight load together with the SF loads (it used to
                // wait behind the SF stores, adding a full NVLink round trip to the chain).
                float weight = 0.0f;
                if (lane_idx == 0)
                    weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + pool_token_idx] = remote_sf_ptr[j];
                }
                if (lane_idx == 0)
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
                __syncwarp();

                if (cute::elect_one_sync()) {

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(
                            pull_pool_block_offset + token_idx_in_expert / BLOCK_M),
                        1u);
                }
                __syncwarp();
            }
        }

        if (!kPushDispatch and thread_idx == 0) stamp_max(2);
        if constexpr (kFineCombine) {
            // Fine-grained combine signaller (dispatch warp 0): consume this CTA's
            // mailbox; for every finished L2 task release-add 1 per scattered token
            // row to the destination rank's counter (sys-scope fence here, off the
            // math warps), until the epilogue posts DONE (all math tasks finished).
            DG_STATIC_ASSERT(kNumSMs <= fused_layout::kSM90FineCombineMaxSMs, "Too many SMs for the combine mailboxes");
            if (warp_idx == 0) {
                auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                uint32_t consumed = ptx::ld_volatile(mailbox + 1);
                while (true) {
                    DG_SPIN_WHILE(ptx::ld_acq(mailbox) == consumed, 1214);
                    const uint32_t entry = ptx::ld_volatile(
                        mailbox + 4 + (consumed & (fused_layout::kSM90FineCombineRingSize - 1)));
                    __syncwarp();
                    if (entry == fused_layout::kSM90FineCombineDoneEntry)
                        break;
                    const uint32_t signal_pool_block_idx = entry & 0xffffffu, signal_valid_m = entry >> 24;
                    for (uint32_t row = lane_idx; row < signal_valid_m; row += 32) {
                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                            signal_pool_block_idx * BLOCK_M + row);
                        asm volatile("fence.acq_rel.sys;" ::: "memory");
                        ptx::red_add_rel_sys(
                            sym_buffer.map(workspace.get_combine_arrival_count_ptr(src_metadata.token_idx),
                                           src_metadata.rank_idx), 1);
                    }
                    __syncwarp();
                    ++ consumed;
                    if (lane_idx == 0)
                        ptx::st_rel_gpu(mailbox + 1, consumed);  // release: slot read done before reuse
                }
                // Terminal entry consumed too (keeps producer/consumer sequences aligned)
                ++ consumed;
                if (lane_idx == 0)
                    ptx::st_rel_gpu(mailbox + 1, consumed);
                __syncwarp();
            }
            // All dispatch warps: this CTA's math tasks are done (warp 0 saw DONE);
            // the cleanup below zeroes the L1 arrival counts / L2 arrival masks that
            // other CTAs' L1/L2 tasks still touch, so wait for every CTA's math tasks
            // (dispatch warps only; the combine warps are not held up).
            fused_comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); });
        } else {
            // Cleanup workspace, overlapping with combine
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }

        cleanup_workspace();
        fused_comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);
    } else if (warp_idx == kNumDispatchWarps) {
        // =====================================================================
        // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
        //   The two warps inside `kNumNonEpilogueThreads` load A + SFA and
        //   B + SFB, respectively.
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_a_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_split_idx,
                                     const uint32_t& num_k_splits,
                                     const uint32_t& k_block_begin,
                                     const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            // `k_block_begin`: first absolute K-block of this task (split-K K half 1
            // / stream-K segments start mid-K).
            const auto tensor_map_a_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. While waiting, warm L2 cache with
            // this task's weight rows beyond the smem stage depth (the B loader
            // is blocked on empty barriers after kNumStages), so the tile runs
            // from L2 once the activations arrive. Matters at tiny M where the
            // per-tile latency chain, not bandwidth, bounds the kernel.
            if (has_valid_m) {
                if constexpr (kPrefetchWeightKBlocks > 0) {
                    if (cute::elect_one_sync()) {
                        constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;
                        constexpr uint32_t kInFlightKBlocks = kNumStages * kKBlocksPerStage;
                        const uint32_t k_end = cute::min(num_k_blocks, kInFlightKBlocks + kPrefetchWeightKBlocks);
                        if constexpr (kDenseWeightTiles) {
                            // Dense tiles: k-blocks of one (expert, n_block) are
                            // contiguous, so prefetch whole 20 KB tiles in 1D.
                            constexpr uint32_t shape_k = kBlockIsL2 ? L2_SHAPE_K : L1_SHAPE_K;
                            const auto* weights_base = reinterpret_cast<const uint8_t*>(
                                kBlockIsL2 ? l2_weights_ptr : l1_weights_ptr);
                            constexpr uint32_t kPhaseSubTiles =
                                kBlockIsL2 ? kL2SubTilesPerPacked : kSubTilesPerPacked;
                            constexpr uint32_t kPhaseTiles = kBlockIsL2 ? kL2Tiles : kL1Tiles;
                            constexpr uint32_t kPhaseKBlockBytes =
                                kBlockIsL2 ? SMEM_PACKED_B_L2_SIZE_PER_KBLOCK : SMEM_PACKED_B_L1_SIZE_PER_KBLOCK;
                            const uint32_t tile_row = local_expert_idx * (shape_n / kPackedTileN) +
                                                      n_block_idx * kPhaseTiles / kPhaseSubTiles;
                            const auto* tiles = weights_base +
                                static_cast<size_t>(tile_row) * (shape_k / BLOCK_K) * kPackedTileBytes +
                                (n_block_idx % kPhaseSubTiles) * kPhaseKBlockBytes;
                            // Wide tasks: the kPhaseTiles packed tiles of a task are adjacent
                            // tile rows (one K range of tiles each).
                            for (uint32_t kb = kInFlightKBlocks; kb < k_end; ++ kb) {
                                #pragma unroll
                                for (uint32_t t = 0; t < kPhaseTiles; ++ t)
                                    ptx::tma_prefetch_1d(tiles + (static_cast<size_t>(t) * (shape_k / BLOCK_K) +
                                                                  (k_block_begin + kb)) * kPackedTileBytes,
                                                         kPhaseKBlockBytes / kPhaseTiles);
                            }
                        } else {
                            const auto tensor_map_b_ptr = kBlockIsL2 ?
                                &tensor_map_l2_weights : &tensor_map_l1_weights;
                            const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                            for (uint32_t kb = kInFlightKBlocks; kb < k_end; ++ kb)
                                cute::SM90_TMA_LOAD_2D::PREFETCH::copy(
                                    tensor_map_b_ptr, (k_block_begin + kb) * B_LOAD_BYTES_PER_ROW, n_idx);
                        }
                    }
                    __syncwarp();
                }
                if constexpr (!kBlockIsL2) {
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    // Probe slots 34/35 (SM0 loader, SM cycles / count): time spent in
                    // this spin per L1 task (push: waits for the post-barrier publish).
                    const bool arrival_probe_on = (phase_stamps != nullptr) && (sm_idx == 0) &&
                                                  (ptx::get_lane_idx() == 0);
                    const unsigned long long arrival_t0 = arrival_probe_on ? clock64() : 0ull;
                    // Lean push: every row of this launch was in the pool before the
                    // scheduler's acquire of the expert count's completeness word (released
                    // by the publishing CTA after it acquired NVLink barrier #1), so the
                    // per-task arrival spin (1.3-2.6 us on the first task, H20) is skipped.
                    if constexpr (!kLeanPush)
                        DG_SPIN_WHILE(ptx::ld_acq(ptr) != valid_m, 1329);
                    if (arrival_probe_on) {
                        atomicAdd(phase_stamps + 34, clock64() - arrival_t0);
                        atomicAdd(phase_stamps + 35, 1ull);
                    }
                    // Push dispatch: the rows were written by REMOTE ranks (weak stores
                    // over NVLink into this GPU's L2/HBM), never by this SM's generic proxy,
                    // so no proxy fence is needed before the TMA loads; the acquire above
                    // (release chain: pusher -> barrier #1 -> local publish) orders them.
                    // (A fence.proxy.async.global here cost ~2 us per L1 task on H20.)
                }
                // L2: no up-front wait for all L1 N-blocks; each stage below
                // waits only for the L1 blocks that produced its K-block(s).
            }
            // Cached snapshot of the L1 readiness mask (L2 tasks only).
            uint64_t l1_ready_mask = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if constexpr (kBlockIsL2) {
                    if (has_valid_m) {
                        // Bits of the L1 N-blocks feeding absolute K-blocks
                        // [k_block_begin + k_block_idx, + blocks in this stage).
                        constexpr uint32_t kBitsPerStage = kKBlocksPerStage * kNumL1BlocksPerL2KBlock;
                        DG_STATIC_ASSERT(kBitsPerStage < 64, "Stage readiness mask overflow");
                        // Wide L1 tasks (kNumL2KBlocksPerL1Block == 2): K-blocks k .. k+n-1 need the
                        // L1 N-blocks k/2 .. (k+n-1)/2.
                        const uint32_t first_kb = k_block_begin + k_block_idx;
                        const uint32_t last_kb = first_kb + cute::min(kKBlocksPerStage, num_k_blocks - k_block_idx) - 1u;
                        const uint32_t first_bit = (first_kb / kNumL2KBlocksPerL1Block) * kNumL1BlocksPerL2KBlock;
                        const uint32_t num_bits = (last_kb / kNumL2KBlocksPerL1Block) * kNumL1BlocksPerL2KBlock +
                                                  kNumL1BlocksPerL2KBlock - first_bit;
                        const uint64_t need = ((1ull << num_bits) - 1ull) << first_bit;
                        if ((l1_ready_mask & need) != need) {
                            const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                            DG_SPIN_WHILE(((l1_ready_mask = ptx::ld_acq_gpu(ptr)) & need) != need, 1357);
                        }
                        // L2 tail probe: 37 = max time the LAST stage's L1 inputs were seen ready
                        if (k_block_idx + kKBlocksPerStage >= num_k_blocks && ptx::get_lane_idx() == 0)
                            stamp_max(37);
                    }
                }

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                        const uint32_t m_idx = pool_block_idx * BLOCK_M;
                        const uint32_t k_idx = (k_block_begin + k_block_idx) * BLOCK_K;

                        if constexpr (!kBlockIsL2 || !kSplitMDecodedWeightReuse) {
                            // One A tile + one SFA row per K-block of the stage,
                            // a single expect-tx for the whole stage.
                            // L2 with per-64 activation scales (kHalfTileTasks): two SF
                            // rows per K128 block, slots (kb * kNumL2SFAGroups + g).
                            constexpr uint32_t kSFRowsPerKBlock =
                                kBlockIsL2 ? BLOCK_K / kL2ActsSFGranK : 1u;
                            DG_STATIC_ASSERT(kSFRowsPerKBlock <= kNumL2SFAGroups,
                                             "Not enough SFA slots per K-block");
                            // Partial last stage: only the remaining K-blocks are fetched
                            // (and counted in expect-tx).
                            const uint32_t num_stage_blocks =
                                cute::min(kKBlocksPerStage, num_k_blocks - k_block_idx);
                            #pragma unroll
                            for (uint32_t kb = 0; kb < kKBlocksPerStage; ++ kb) {
                                if (kb >= num_stage_blocks) break;
                                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                    tensor_map_a_ptr, full_barriers[stage_idx],
                                    smem_a[stage_idx] + kb * SMEM_A_SIZE_PER_KBLOCK / sizeof(a_dtype_t),
                                    k_idx + kb * BLOCK_K, m_idx, 1);
                                #pragma unroll
                                for (uint32_t g = 0; g < kSFRowsPerKBlock; ++ g) {
                                    tma::copy<BLOCK_M, 1, 0, float>(
                                        tensor_map_sfa_ptr, full_barriers[stage_idx],
                                        smem_sfa[stage_idx] + (kb * kNumL2SFAGroups + g) * kL2SFAHalfStride,
                                        m_idx, (k_block_begin + k_block_idx + kb) * kSFRowsPerKBlock + g, 1);
                                }
                            }
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                num_stage_blocks * (SMEM_A_SIZE_PER_KBLOCK +
                                                    kSFRowsPerKBlock * BLOCK_M * sizeof(float)));
                        } else {
                            // TMA load A
                            tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                                k_idx, m_idx, 1);
                            // BN128 L1 produces per-64 activation scales. L2
                            // consumes both scale groups for each BK128 tile.
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, (k_block_begin + k_block_idx) * 2, 1);
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + kL2SFAHalfStride,
                                m_idx, (k_block_begin + k_block_idx) * 2 + 1, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                        }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        };
        if constexpr (kUseInterleavedScheduler) {
            for_each_published_block(load_a_task);
        } else {
            for_each_static_selected_block(load_a_task);
        }

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_b_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_split_idx,
                                     const uint32_t& num_k_splits,
                                     const uint32_t& k_block_begin,
                                     const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            const auto tensor_map_b_ptr = kBlockIsL2 ?
                &tensor_map_l2_weights : &tensor_map_l1_weights;
            constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;
            constexpr uint32_t shape_k = kBlockIsL2 ? L2_SHAPE_K : L1_SHAPE_K;
            // Dense tiles (MXFP4/QoQ): packed tile (expert, n256_block, k_block) lives at
            // ((expert * n256_blocks + n256_block) * k_blocks + k_block) * (256 * 80 B);
            // a BLOCK_N=128 kernel tile is the contiguous upper/lower 10 KB half.
            // Per-phase sub-tile geometry (L2 half-row tasks: 128-row L2 sub-tiles).
            constexpr uint32_t kPhaseSubTiles =
                kBlockIsL2 ? kL2SubTilesPerPacked : kSubTilesPerPacked;
            // Wide tasks: packed tiles per task (adjacent tile rows, each a contiguous
            // K range of 20 KB tiles); tile t of a stage's K-block lands at +t * 20 KB.
            constexpr uint32_t kPhaseTiles = kBlockIsL2 ? kL2Tiles : kL1Tiles;
            constexpr uint32_t kPhaseKBlockBytes =
                kBlockIsL2 ? SMEM_PACKED_B_L2_SIZE_PER_KBLOCK : SMEM_PACKED_B_L1_SIZE_PER_KBLOCK;
            DG_STATIC_ASSERT(kPhaseTiles == 1 || kPhaseKBlockBytes == kPhaseTiles * kPackedTileBytes,
                             "Wide task K-block == kPhaseTiles whole packed tiles");
            const uint8_t* dense_tiles = nullptr;
            if constexpr (kDenseWeightTiles) {
                const uint32_t tile_row = local_expert_idx * (shape_n / kPackedTileN) +
                                          n_block_idx * kPhaseTiles / kPhaseSubTiles;
                dense_tiles = reinterpret_cast<const uint8_t*>(kBlockIsL2 ? l2_weights_ptr : l1_weights_ptr) +
                    static_cast<size_t>(tile_row) * (shape_k / BLOCK_K) * kPackedTileBytes +
                    (n_block_idx % kPhaseSubTiles) * kPhaseKBlockBytes;
            }
            constexpr size_t kTileRowStrideBytes = static_cast<size_t>(shape_k / BLOCK_K) * kPackedTileBytes;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Partial last stage: only the remaining K-blocks are fetched.
                const uint32_t num_stage_blocks =
                    cute::min(kKBlocksPerStage, num_k_blocks - k_block_idx);
                const uint32_t stage_bytes = num_stage_blocks * kPhaseKBlockBytes;
                if (cute::elect_one_sync()) {
                    if constexpr (kDenseWeightTiles) {
                        // One 1D bulk copy per stage (20 KB per K-block for BN256;
                        // 40 / 80 KB when kKBlocksPerStage == 2 / 4, tiles k..k+3 are
                        // adjacent in the dense layout); decoders see 80 B rows at row * 80.
                        if constexpr (kPhaseTiles > 1) {
                            // Wide tasks (one K-block per stage): one 20 KB bulk copy per
                            // packed tile of the task, tile t at +t * 20 KB of the stage slot.
                            DG_STATIC_ASSERT(kPhaseTiles == 1 || kKBlocksPerStage == 1,
                                             "Wide tasks: one K-block per stage");
                            #pragma unroll
                            for (uint32_t t = 0; t < kPhaseTiles; ++ t) {
                                ptx::tma_load_1d(
                                    reinterpret_cast<uint8_t*>(smem_packed_b[stage_idx]) + t * kPackedTileBytes,
                                    dense_tiles + t * kTileRowStrideBytes +
                                        static_cast<size_t>(k_block_begin + k_block_idx) * kPackedTileBytes,
                                    full_barriers[stage_idx], kPackedTileBytes);
                            }
                        } else if constexpr (kKBlocksPerStage == 1 || kPhaseSubTiles == 1) {
                            ptx::tma_load_1d(smem_packed_b[stage_idx],
                                        dense_tiles + static_cast<size_t>(k_block_begin + k_block_idx) * kPackedTileBytes,
                                        full_barriers[stage_idx], stage_bytes);
                        } else {
                            // Half-tile / L2 half-row tasks: the stage's K-blocks are 10 KB
                            // sub-tiles one packed tile (20 KB) apart -> one bulk copy each
                            // (k, k+1 at +0 / +10 KB of the stage slot), one expect-tx.
                            #pragma unroll
                            for (uint32_t kb = 0; kb < kKBlocksPerStage; ++ kb) {
                                if (kb >= num_stage_blocks) break;
                                ptx::tma_load_1d(
                                    reinterpret_cast<uint8_t*>(smem_packed_b[stage_idx]) +
                                        kb * kPhaseKBlockBytes,
                                    dense_tiles + static_cast<size_t>(k_block_begin + k_block_idx + kb) * kPackedTileBytes,
                                    full_barriers[stage_idx], kPhaseKBlockBytes);
                            }
                        }
                    } else {
                        const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                        // NVFP4 fused B+scale layout stores 64B packed FP4 + 8B
                        // UE4M3 scale + 8B zero padding per BK128 row.
                        const uint32_t k_idx = (k_block_begin + k_block_idx) * B_LOAD_BYTES_PER_ROW;
                        tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            smem_packed_b[stage_idx],
                            k_idx, n_idx, 1);
                    }
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        kDenseWeightTiles ? stage_bytes : SMEM_PACKED_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        };
        if constexpr (kUseInterleavedScheduler) {
            produce_interleaved_blocks(load_b_task);
        } else {
            for_each_static_selected_block(load_b_task);
        }

    } else {
        // =====================================================================
        // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if (lane_idx == 0)
                empty_barriers[s]->arrive();
        };

        const auto notify_l1_ready = [&](const uint32_t& ready_pool_block_idx,
                                         const uint32_t& ready_n_block_idx) {
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                ptx::red_or_rel_gpu(
                    workspace.get_l2_arrival_mask_ptr(ready_pool_block_idx),
                    1ull << ready_n_block_idx);
            }
            __syncwarp();
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        DG_STATIC_ASSERT(kSwapABRequested ||
                         (WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N),
                         "Split-N WGs must each run one M64N128 WGMMA per K-block");

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Fine-grained combine: this CTA's mailbox producer sequence (CTA-uniform;
        // only epilogue thread 0 writes). Baseline = the word's value left by the
        // previous launch (ordered by the kernel boundary; the consumer reads the
        // same baseline from its own word).
        uint32_t combine_mailbox_seq = 0u;
        // Dynamic combine: this launch's ticket word (parity of the epoch, read before
        // any math task so SM0's cleanup bump cannot be observed by this launch)
        uint32_t combine_ticket_parity = 0u;
        if constexpr (kCombineDynamic)
            combine_ticket_parity = ptx::ld_volatile(workspace.get_combine_epoch_ptr()) & 1u;
        if constexpr (kFineCombine)
            combine_mailbox_seq = *workspace.get_combine_mailbox_ptr(sm_idx);
        const auto run_math_task_impl = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_split_idx,
                                     const uint32_t& num_k_splits,
                                     const uint32_t& k_block_begin,
                                     const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            // Per-phase task geometry. L2 half-row tasks: 128-row tasks, WG w owns rows
            // [64w, 64w + 64) == ONE 64-row weight half (kWGHalves == 1) of the 10 KB
            // packed sub-tile per K-block; every other case keeps the 256-row task with
            // WG_BLOCK_N rows (two halves) per WG.
            constexpr uint32_t kTaskBlockN = kBlockIsL2 ? TASK_BLOCK_N_L2 : TASK_BLOCK_N;
            constexpr uint32_t kWGBlockN = kBlockIsL2 ? L2_WG_BLOCK_N : WG_BLOCK_N;
            constexpr uint32_t kWGHalves = kWGBlockN / 64;
            constexpr uint32_t kPackedBKBlockBytes =
                kBlockIsL2 ? SMEM_PACKED_B_L2_SIZE_PER_KBLOCK : SMEM_PACKED_B_L1_SIZE_PER_KBLOCK;
            DG_STATIC_ASSERT(kWGHalves >= 1 && kWGHalves <= kSwapABWeightHalves,
                             "Per-WG weight halves must fit the swapAB accumulator layout");
            // Packed tiles per task of this phase (wide tasks: 2) and the L2 SF groups
            // an L1 task publishes (wide L1 tasks: 2, one per WG).
            constexpr uint32_t kPhaseTiles = kBlockIsL2 ? kL2Tiles : kL1Tiles;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            // Half-tile tasks: both WGs work on the task's 128 rows (K-split), so
            // the per-WG N offsets are zero and only WG0 owns the epilogue.
            const uint32_t wg_n_idx =
                (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 0u : epilogue_wg_idx * kWGBlockN;
            const uint32_t wg_l1_out_n_idx =
                (kSplitMDecodedWeightReuse || kHalfTileTasks) ? 0u : epilogue_wg_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * kTaskBlockN + wg_n_idx;
            const uint32_t ksplit_kb = kHalfTileTasks ? epilogue_wg_idx : 0u;
            const bool is_epilogue_wg = !kHalfTileTasks || epilogue_wg_idx == 0;
            const uint32_t row_block_offset =
                kSplitMDecodedWeightReuse ? epilogue_wg_idx * WG_BLOCK_M : 0u;
            // Fine-grained combine: called by every epilogue thread right after the
            // CTA-wide sync that follows the L2 NVLink scatter of this task. Thread 0
            // posts (pool block, valid rows) to the CTA mailbox with st.release.gpu;
            // the bar.sync made every thread's scatter stores happen-before that
            // release, and the dispatch consumer's acquire + sys-scope release make
            // them visible to the remote combine warp (cumulativity).
            const auto signal_combine_arrivals = [&]() {
                if constexpr (kFineCombine) {
                    if (epilogue_thread_idx == 0 && valid_m > 0) {
                        auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                        DG_SPIN_WHILE(combine_mailbox_seq - ptx::ld_volatile(mailbox + 1) >= fused_layout::kSM90FineCombineRingSize, 1611);
                        mailbox[4 + (combine_mailbox_seq & (fused_layout::kSM90FineCombineRingSize - 1))] =
                            pool_block_idx | (valid_m << 24);
                        ptx::st_rel_gpu(mailbox, combine_mailbox_seq + 1);
                        ++ combine_mailbox_seq;
                    }
                }
            };
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            // NVFP4: one scale per expert. MXFP4: one scale per weight row
            // (2^e_ref folded with the global scale), indexed by the L2 output
            // column `n_idx + col` within this expert.
            const float l2_global_scale = (kPerRowEpilogueScale || l2_global_scales == nullptr) ?
                1.0f : __ldg(l2_global_scales + local_expert_idx);
            const float* __restrict__ l2_row_scales = kPerRowEpilogueScale ?
                l2_global_scales + local_expert_idx * L2_SHAPE_N + n_idx : nullptr;
            const auto l2_scale_at = [&](const uint32_t& col) -> float {
                if constexpr (kPerRowEpilogueScale)
                    return __ldg(l2_row_scales + col);
                else
                    return l2_global_scale;
            };
            const auto cast_l2_scaled_bf16_pair = [&](float x, float y, const uint32_t& col) -> uint32_t {
                if constexpr (kPerRowEpilogueScale) {
                    const float2 sc = __ldg(reinterpret_cast<const float2*>(l2_row_scales + col));
                    x *= sc.x;
                    y *= sc.y;
                } else {
                    x *= l2_global_scale;
                    y *= l2_global_scale;
                }
                return math::cast_into_bf16_and_pack(x, y);
            };

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            // 64 for M=64,N=128; the swapAB layout needs kSwapABHalfAccumPerThread per
            // 64-row half (wide tasks: 4 halves -> 128; only token chunk 0 is live at BM8).
            constexpr uint32_t kAccumPerThread =
                WGMMA::kNumAccum > kWGHalves * kSwapABHalfAccumPerThread ?
                WGMMA::kNumAccum : kWGHalves * kSwapABHalfAccumPerThread;
            float final_accum[kAccumPerThread] = {};

            // L2 swapAB epilogue for L2 N-block `l2_n_block_idx` from `final_accum`: BF16 x
            // per-row scale -> smem, 16-lane NVLink row scatter, CTA barrier, combine
            // mailbox entry. The L2 task path calls it with its own n_block_idx.
            const auto l2_epilogue_swap = [&](const uint32_t& l2_n_block_idx) {
                if constexpr (kSwapABRequested) {
                    const uint32_t l2_n_idx = l2_n_block_idx * TASK_BLOCK_N_L2 + wg_n_idx;
                    const float* __restrict__ l2_rows = kPerRowEpilogueScale ?
                        l2_global_scales + local_expert_idx * L2_SHAPE_N + l2_n_idx : nullptr;
                    const auto scale_at = [&](const uint32_t& col) -> float {
                        if constexpr (kPerRowEpilogueScale)
                            return __ldg(l2_rows + col);
                        else
                            return l2_global_scale;
                    };
                    // Each active warp scatters a contiguous group of up to 16 rows.
                    constexpr uint32_t kNumRowsPerWarp =
                        BLOCK_M == 8 ? 4u : 8u;
                    auto store_swap_bf16 = [&](const uint32_t& token, const uint32_t& col, const float& value) {
                        if (token < valid_m)
                            smem_cd_l2[token * TASK_BLOCK_N_L2 + wg_n_idx + col] =
                                __float2bfloat16_rn(value * scale_at(col));
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        // L2 half-row tasks: one 64-row half per WG (own accumulators,
                        // own 64 hidden rows [n_idx, n_idx + 64) of the combine row).
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            const uint32_t col_offset = half * 64u;
                            store_swap_bf16(token_0, col_offset + r_0, final_accum[accum_offset + 0]);
                            store_swap_bf16(token_0, col_offset + r_1, final_accum[accum_offset + 2]);
                            store_swap_bf16(token_1, col_offset + r_0, final_accum[accum_offset + 1]);
                            store_swap_bf16(token_1, col_offset + r_1, final_accum[accum_offset + 3]);
                        }
                    };

                    if (is_epilogue_wg) {
                        store_l2_swap_chunk(0);
                        if (valid_m > 8) {
                            #pragma unroll
                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                if (i < num_swap_token_chunks)
                                    store_l2_swap_chunk(i);
                            }
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    const uint32_t row_in_warp_block = lane_idx / 16;
                    const uint32_t lane_in_row = lane_idx % 16;
                    constexpr uint32_t kColsPerScatterLane = kWGBlockN / 16;
                    DG_STATIC_ASSERT(kWGBlockN % 16 == 0,
                                     "SwapAB L2 scatter expects an even lane partition");
                    DG_STATIC_ASSERT(kColsPerScatterLane == 4 || kColsPerScatterLane == 8 || kColsPerScatterLane == 16,
                                     "SwapAB L2 scatter supports WG_BLOCK_N=64, 128 or 256");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t token = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        if (token >= valid_m || !is_epilogue_wg) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                            pool_block_idx * BLOCK_M + token);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        auto smem_ptr = smem_cd_l2
                            + token * TASK_BLOCK_N_L2
                            + wg_n_idx
                            + lane_in_row * kColsPerScatterLane;
                        if constexpr (kColsPerScatterLane == 16) {
                            // Wide L2 tasks: 32 B per lane (two 16 B stores)
                            const auto packed_0 = *reinterpret_cast<uint4*>(smem_ptr);
                            const auto packed_1 = *reinterpret_cast<uint4*>(smem_ptr + 8);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                l2_n_idx * sizeof(nv_bfloat16) + lane_in_row * 2u * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed_0;
                            *sym_buffer.map(dst_ptr + 1, dst_rank_idx) = packed_1;
                        } else if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                l2_n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                l2_n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    signal_combine_arrivals();
                }
            };

            const auto decode_b_stage = [&](const uint32_t& decoded_stage) {
                if constexpr (kSplitMDecodedWeightReuse) {
                    // Both M64 consumers share the same decoded N128 tile.
                    // The two physical decoded slots remove the overwrite
                    // hazard; the trailing pair barrier publishes K+1 before
                    // either consumer issues its WGMMA.
                    DG_STATIC_ASSERT(kUseMode2RowDecoder,
                                     "BM128 split-M uses the cooperative Mode2 decoder");
                    deep_gemm::nvfp4::
                        dequant_smem_b_from_packed_mode2_nibble_split_m<kMXFP4>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    cutlass::arch::fence_view_async_shared();
                    asm volatile("bar.sync %0, %1;" : :
                                 "n"(kSplitMDecodeBarrierIdx), "n"(256) : "memory");
                } else {
                    if constexpr (kQoQ) {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_qoq_shiftxor(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx);
                    } else if constexpr (kUseMode2RowDecoder) {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_mode2_nibble<
                            kQuadDequantIlp, kMXFP4>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    } else {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_braided_lut_window<
                            kQuadDequantIlp, kMXFP4>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    }
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(
                        128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                }
            };
            // Stage-level probe (SM0 / thread 0 only; phase_stamps slots 17..22 + 30/31, SM cycles):
            //   17 full-barrier wait | 18 RF decode+LUT (both halves) | 19 wgmma arrive->drain
            //   21 #L1 stages | 22 head-to-head stage total (excl. last stage of a task)
            //   (kKBlocksPerStage == 2: all per *2-K128-block* stage; 18 = both decodes,
            //    19 = wait<1> + wait<0> drains; 21 counts 2-block stages, 12/L1 task,
            //    or 6 per L1 K-half task when kSplitKL1 is active)
            const bool kstage_probe_on =
                (phase_stamps != nullptr) && (sm_idx == 0) && (epilogue_thread_idx == 0);
            unsigned long long kstage_t_prev = 0;
            // Timing-only experiments (numerics invalid), bitmask in phase_stamps[24]:
            // 1 skip decode | 2 skip wgmma issue | 4 skip k+1 barrier check | 8 skip promote.
            const uint32_t kexp = (phase_stamps != nullptr) ?
                static_cast<uint32_t>(phase_stamps[24]) : 0u;
            // Bits 1/2/8 wrap wgmma issues and the writes/reads of wgmma registers in a
            // runtime (non-uniform to ptxas) branch; ptxas then serialises EVERY wgmma
            // of the kernel (C7518 "program dependence on compiler-inserted WG.DP in
            // divergent path": WARPGROUP.DEPBAR.LE gsb0, 0x0 after each IGMMA, measured
            // on H20 2026-09-09). They are compiled in only with -DDG_FP4_PROBE_EXP_GATES=1
            // (timing experiments on a serialised pipe); bit 4 (barrier) is always live.
#ifdef DG_FP4_PROBE_EXP_GATES
            constexpr bool kExpGates = DG_FP4_PROBE_EXP_GATES != 0;
#else
            constexpr bool kExpGates = false;
#endif
            const auto exp_skip = [&](const uint32_t& bit) -> bool {
                return kExpGates && (kexp & bit) != 0u;
            };
            const auto kstage_add = [&](const uint32_t slot, const unsigned long long& v) {
                if (kstage_probe_on) atomicAdd(phase_stamps + slot, v);
            };
            // Non-blocking mbarrier phase check. The blocking `wait()` costs ~250 ns
            // even when the phase has already completed (suspend/wake path), and the
            // k+1 tile is essentially always landed by the time the math warps ask
            // for it (probe slot 23), so test first and only block on a miss.
            const auto barrier_ready = [](Barrier* bar, const uint32_t& parity) -> bool {
                uint32_t ready = 0;
                asm volatile(
                    "{\n .reg .pred P;\n"
                    " mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n"
                    " selp.u32 %0, 1, 0, P;\n}"
                    : "=r"(ready)
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(parity)
                    : "memory");
                return ready != 0;
            };

            // Software-pipelined swapAB main loop (kSwapPipelineDecode): the WGMMAs
            // of stage k for both weight halves are issued asynchronously, stage
            // k+1 is decoded while the tensor cores run, then k is waited on,
            // promoted and released. The generic loop below decodes, runs and
            // waits serially per stage.
            // RF decode (kRFDecode, MXFP4/QoQ swapAB): dedicated per-stage loop shared by
            // L1 and L2 (both promote with the per-token K128 activation SF; the
            // MXFP4 row scale is folded into the FP8 fragments by the LUT, the QoQ
            // integer s2[row] (byte 64 of the packed row) rides the promote). Thread
            // (warp w, lane l) owns rows r_0 = 16w + l/4 and r_1 = r_0 + 8 of each
            // 64-row weight half; column c = l % 4 owns K = 4c..4c+3 and 16+4c..16+4c+3
            // of every K32 step, i.e. exactly word c of each 16-byte quad in the
            // RF-ordered packed row (one uint4 per row covers all four K32 steps).
            // Decoded bytes are identical to the SMEM tile decoder's (bit-exact vs SS).
            // Per stage: all smem loads for both halves first, then the 16 LUT gathers,
            // then the decode; both halves' eight m64nNk32 RS WGMMAs share ONE commit
            // group and one drain.
            if constexpr (kSwapABRequested && kRFDecode) {
                // RF loop unit: (K128 block, PAIR of 64-row halves). A WG with two halves
                // (BN256 tasks) has one pair, so unit == K-block; a wide-task WG (four
                // halves) has two pairs per K-block and one K-block per stage, so the
                // 2-unit loops below run unchanged with the same fragment / accumulator
                // registers (unit u: K-block u / kHalfPairs, half pair u % kHalfPairs).
                auto run_swap_ab_rf = [&]<uint32_t N_SWAP>() {
                    constexpr uint32_t kUnitHalves = kWGHalves >= 2u ? 2u : kWGHalves;
                    constexpr uint32_t kHalfPairs = kWGHalves / kUnitHalves;
                    constexpr uint32_t kUnitsPerStage = kKBlocksPerStage * kHalfPairs;
                    constexpr bool kWideUnits = kHalfPairs > 1u;
                    DG_STATIC_ASSERT(kWGHalves == kUnitHalves * kHalfPairs, "Halves must form whole pairs");
                    DG_STATIC_ASSERT(kUnitsPerStage <= 2, "RF loops: at most two (K-block, half pair) units per stage");
                    DG_STATIC_ASSERT(!kWideUnits || (kKBlocksPerStage == 1 && kHalfPairs == 2 && !kHalfTileTasks),
                                     "Wide tasks: one K-block per stage, two half pairs, no half-tile tasks");
                    // QoQ: int8 RS atoms have no N=24; pad to 32 like the SS path (extra
                    // token columns are masked by `token < valid_m`).
                    // QoQ inline s2 (kQoQInlineS2, host env DG_FP4_QOQ_INLINE_S2): L1 only
                    // (the L1 int8 activations carry ONE per-token scale repeated into
                    // every K128 SF slot; the L2 activations are re-quantised per
                    // (token, K128) by the L1 epilogue, so L2 keeps the per-block
                    // promote). The decode folds s2 into the int8 weight ((code - z) * s2
                    // fits int8: |.| <= 112 + s2/2 <= 120), the whole task K range
                    // accumulates in ONE int32 set (|sum| <= 3072 * 127 * 127 < 2^26) and
                    // the per-token activation scale is applied once at task end, so no
                    // per-K128 accumulator readout / tensor-pipe drain sits in the loop.
                    // 2-K-block stages only (the 1-/4-block loops keep the promote path).
                    // (Also the 1-unit loop of wide-L2-only launches, so the QoQ L1 numerics
                    // do not depend on the L2 task shape.)
                    constexpr bool kInlineS2 = kQoQ && kQoQInlineS2 && !kBlockIsL2 &&
                                               !kHalfTileTasks && kUnitsPerStage <= 2;
                    // The plain 2-fragment-buffer inline-s2 loop takes one knob (host env
                    // DG_FP4_QIS2_PREFETCH_PACKED): the next block's packed words are loaded
                    // (LDS) before the wgmma wait that frees its fragment buffer, decoded after it.
                    // Wide units: only the plain 2-buffer inline-s2 loop is implemented
                    // (its per-unit accumulator sets are per half pair); ILV / 3-4 frag
                    // buffers fall back to it.
                    constexpr bool kQIS2Plain = kInlineS2 && kUnitsPerStage == 2 &&
                                                (kWideUnits || (!kQoQInlineS2Ilv && kQoQInlineS2Frags == 2));
                    constexpr bool kQIS2Prefetch = kQIS2Plain && kQoQInlineS2PrefetchPacked;
                    using SwapRS = typename std::conditional_t<kQoQ,
                        mma::sm90::INT8MMARSSelector<(N_SWAP == 24 ? 32 : N_SWAP)>,
                        mma::sm90::FP8MMARSSelector<N_SWAP>>::type;
                    DG_STATIC_ASSERT(BLOCK_K / SwapRS::K == 4, "Expects 4 K32 steps per K128");
                    constexpr uint32_t kSwapAccum = SwapRS::kNumAccum;
                    // One accumulator set per K-block of a stage: the two K128 blocks
                    // carry different per-token activation scales, so they are promoted
                    // separately (the RS WGMMAs of both run concurrently).
                    // Half-tile tasks: each WG owns ONE K-block per stage (ksplit_kb).
                    // >2 K-blocks per stage: two accumulator sets alternate (acc[b & 1]);
                    // block b is promoted right after wait<1> following issue(b+1), which
                    // frees its set for block b+2 (4 sets = +32 regs would spill).
                    // Wide units: one set per half pair (inline s2: both accumulate the task).
                    constexpr uint32_t kNumAccKBlocks =
                        (kHalfTileTasks || kInlineS2) ? kHalfPairs : cute::min(kUnitsPerStage, 2u);
                    // Per-64 L2 activation scales (kHalfTileTasks): K32 steps {0,1} and
                    // {2,3} of a K128 block accumulate separately (same commit group)
                    // and are promoted with their own SF row.
                    constexpr uint32_t kSFGroups = (kBlockIsL2 && kL2ActsSFGranK == 64u) ? 2u : 1u;
                    constexpr uint32_t kK32PerSFGroup = 4u / kSFGroups;
                    // Two accumulator chains per (SF group, half): K32 steps alternate
                    // between them so the 4 dependent RS-WGMMAs of a K128 block become
                    // two independent 2-deep chains (tensor-core latency exposed once
                    // less per block); the chains are summed at promote time.
                    // Inline s2 measured 4 chains (8 independent streams, register-neutral
                    // vs the old 2 sets x 2 chains) at no gain (H20 09-09: M8 1413/1420 vs
                    // 1381/1501 ns per stage, M16 1558 vs 1547/1487), so the dependent
                    // RS-wgmma chain is not the stage floor; 2 chains kept.
                    constexpr uint32_t kAccChains = 2u;
                    swap_accum_t swap_accum[kNumAccKBlocks][kSFGroups][kUnitHalves][kAccChains][kSwapAccum];
                    // Inline s2 may rotate 3 or 4 A-fragment buffers (kQoQInlineS2Frags,
                    // DG_FP4_QIS2_FRAGS); every other loop uses exactly two.
                    constexpr uint32_t kFragBufs =
                        (kInlineS2 && !kQoQInlineS2Ilv && !kWideUnits && kUnitsPerStage == 2) ? kQoQInlineS2Frags : 2u;
                    uint32_t frag[kFragBufs][kUnitHalves][4][4];  // [buffer][half][k32 step][a0..a3]

                    const auto fence_accum = [&]() {
                        #pragma unroll
                        for (uint32_t kb = 0; kb < kNumAccKBlocks; ++ kb) {
                            #pragma unroll
                            for (uint32_t g = 0; g < kSFGroups; ++ g) {
                                #pragma unroll
                                for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t c = 0; c < kAccChains; ++ c) {
                                        #pragma unroll
                                        for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                            ptx::warpgroup_fence_operand(swap_accum[kb][g][h][c][i]);
                                    }
                                }
                            }
                        }
                    };
                    const auto fence_frag = [&](uint32_t (&f)[kUnitHalves][4][4]) {
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++ i)
                                    ptx::warpgroup_fence_operand(reinterpret_cast<float&>(f[h][k][i]));
                            }
                        }
                    };
                    // Decode both 64-row halves of K-block `kb` of `stage` into `f` (this
                    // thread's A fragments for the 4 K32 steps x 2 halves).
                    // Split in two so the plain inline-s2 loop can issue the loads before a
                    // wgmma wait (kQIS2Prefetch) and decode after it.
                    const auto load_packed_rf = [&](const uint32_t& stage, const uint32_t& unit,
                                                    uint4 (&w)[kUnitHalves][2], uint32_t (&sw)[kUnitHalves][2]) {
                        const uint32_t kb = unit / kHalfPairs, hp = unit % kHalfPairs;
                        const auto* packed_rows =
                            reinterpret_cast<const uint8_t*>(smem_packed_b[stage]) +
                            kb * kPackedBKBlockBytes;
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            const uint32_t row_0 = wg_n_idx + (hp * kUnitHalves + h) * 64u + r_0;
                            const uint32_t row_1 = row_0 + 8u;
                            w[h][0] = *reinterpret_cast<const uint4*>(
                                packed_rows + row_0 * 80u + col_idx * 16u);
                            w[h][1] = *reinterpret_cast<const uint4*>(
                                packed_rows + row_1 * 80u + col_idx * 16u);
                            sw[h][0] = *reinterpret_cast<const uint32_t*>(packed_rows + row_0 * 80u + 64u);
                            sw[h][1] = *reinterpret_cast<const uint32_t*>(packed_rows + row_1 * 80u + 64u);
                        }
                    };
                    const auto decode_words_rf = [&](const uint4 (&w)[kUnitHalves][2], const uint32_t (&sw)[kUnitHalves][2],
                                                     uint32_t (&f)[kUnitHalves][4][4]) {
                        if constexpr (kInlineS2) {
                            // QoQ inline s2: byte = (code - z) * s2 as int8, LiquidGEMM form
                            // (arXiv 2509.01229 Sec. 4 Eq. 9-12): with a = 128 - z * s2,
                            //   int8 = ((code * s2 + a) mod 256) XOR 0x80
                            // on all four byte lanes of a word at once (codes already in byte
                            // lanes after the nibble AND): one IMAD (word * s2 + a * 0x01010101)
                            // and one LOP3 XOR per 4 weights. Overflow-free by the packer's
                            // invariants (z = round(-min / s2), |w8| <= 112, s2 <= 15):
                            // z * s2 <= 112 + s2/2 < 128 so a in [8, 128] (no wrap), and each
                            // lane holds 128 + (code - z) * s2 in [8, 248] (no carry between
                            // lanes). Verified exhaustively on the host over every (code, z, s2)
                            // whose dequant fits int8 (bit-exact vs ((code - z) * s2) & 0xff).
                            // a is derived in-kernel from the meta bytes (s2 = byte 64,
                            // z = byte 65): 2 IMAD per row per K-block.
                            uint32_t s2v[kUnitHalves][2], a4[kUnitHalves][2];
                            #pragma unroll
                            for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                                #pragma unroll
                                for (uint32_t r = 0; r < 2; ++ r) {
                                    s2v[h][r] = sw[h][r] & 0xffu;
                                    const uint32_t z = (sw[h][r] >> 8u) & 0xffu;
                                    a4[h][r] = (0x80u - z * s2v[h][r]) * 0x01010101u;
                                }
                            }
                            const auto fold = [](const uint32_t& c, const uint32_t& s2,
                                                 const uint32_t& a) -> uint32_t {
                                return (c * s2 + a) ^ 0x80808080u;
                            };
                            #pragma unroll
                            for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                                #pragma unroll
                                for (uint32_t k = 0; k < 4; ++ k) {
                                    const uint32_t word_r0 = k == 0 ? w[h][0].x : k == 1 ? w[h][0].y :
                                                             k == 2 ? w[h][0].z : w[h][0].w;
                                    const uint32_t word_r1 = k == 0 ? w[h][1].x : k == 1 ? w[h][1].y :
                                                             k == 2 ? w[h][1].z : w[h][1].w;
                                    f[h][k][0] = fold((word_r0 >> 4) & 0x0f0f0f0fu, s2v[h][0], a4[h][0]);
                                    f[h][k][1] = fold((word_r1 >> 4) & 0x0f0f0f0fu, s2v[h][1], a4[h][1]);
                                    f[h][k][2] = fold(word_r0 & 0x0f0f0f0fu, s2v[h][0], a4[h][0]);
                                    f[h][k][3] = fold(word_r1 & 0x0f0f0f0fu, s2v[h][1], a4[h][1]);
                                }
                            }
                        } else if constexpr (kQoQ) {
                            // QoQ: pure ALU decode, no LUT gather. Word (K32 step k, column c)
                            // holds K 4c..4c+3 in its high nibbles and K 16+4c..16+4c+3 in its
                            // low nibbles (host `_mxfp4_rf_fragment_order`, plain nibbles, no
                            // braid); byte b <-> K +b. Same borrow-guarded per-byte subtract
                            // as `dequant_smem_b_from_packed_qoq_shiftxor`: (code - z) int8,
                            // bit-exact vs the SS tile decoder. z = byte 65 of the packed row.
                            uint32_t zz[kUnitHalves][2];
                            #pragma unroll
                            for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                                zz[h][0] = ((sw[h][0] >> 8u) & 0xffu) * 0x01010101u;
                                zz[h][1] = ((sw[h][1] >> 8u) & 0xffu) * 0x01010101u;
                            }
                            const auto zsub = [](const uint32_t& nib, const uint32_t& z) -> uint32_t {
                                return ((nib | 0x80808080u) - z) ^ 0x80808080u;
                            };
                            #pragma unroll
                            for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                                #pragma unroll
                                for (uint32_t k = 0; k < 4; ++ k) {
                                    const uint32_t word_r0 = k == 0 ? w[h][0].x : k == 1 ? w[h][0].y :
                                                             k == 2 ? w[h][0].z : w[h][0].w;
                                    const uint32_t word_r1 = k == 0 ? w[h][1].x : k == 1 ? w[h][1].y :
                                                             k == 2 ? w[h][1].z : w[h][1].w;
                                    // a0: (r_0, K 4c..), a1: (r_1, K 4c..), a2: (r_0, K 16+4c..), a3: (r_1, K 16+4c..)
                                    f[h][k][0] = zsub((word_r0 >> 4) & 0x0f0f0f0fu, zz[h][0]);
                                    f[h][k][1] = zsub((word_r1 >> 4) & 0x0f0f0f0fu, zz[h][1]);
                                    f[h][k][2] = zsub(word_r0 & 0x0f0f0f0fu, zz[h][0]);
                                    f[h][k][3] = zsub(word_r1 & 0x0f0f0f0fu, zz[h][1]);
                                }
                            }
                        } else {
                        uint2 lut[kUnitHalves][2][4];
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t r = 0; r < 2; ++ r) {
                                #pragma unroll
                                for (uint32_t k = 0; k < 4; ++ k)
                                    lut[h][r][k] = smem_nvfp4_lut[(sw[h][r] >> (k * 8u)) & 0x7fu];
                            }
                        }
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                const uint32_t word_r0 = k == 0 ? w[h][0].x : k == 1 ? w[h][0].y :
                                                         k == 2 ? w[h][0].z : w[h][0].w;
                                const uint32_t word_r1 = k == 0 ? w[h][1].x : k == 1 ? w[h][1].y :
                                                         k == 2 ? w[h][1].z : w[h][1].w;
                                const uint2 d0 = deep_gemm::nvfp4::dequant_mode2_nibble_word(word_r0, lut[h][0][k]);
                                const uint2 d1 = deep_gemm::nvfp4::dequant_mode2_nibble_word(word_r1, lut[h][1][k]);
                                // a0: (r_0, K 4c..), a1: (r_1, K 4c..), a2: (r_0, K 16+4c..), a3: (r_1, K 16+4c..)
                                f[h][k][0] = d0.x; f[h][k][1] = d1.x;
                                f[h][k][2] = d0.y; f[h][k][3] = d1.y;
                            }
                        }
                        }
                    };
                    const auto decode_stage_rf = [&](const uint32_t& stage, const uint32_t& unit,
                                                     uint32_t (&f)[kUnitHalves][4][4]) {
                        uint4 w[kUnitHalves][2];
                        uint32_t sw[kUnitHalves][2];
                        load_packed_rf(stage, unit, w, sw);
                        decode_words_rf(w, sw, f);
                    };
                    // One commit group for K-block `kb` of `stage`: 4 K32 steps into
                    // acc[0] with f[0], then into acc[1] with f[1]. The B (activation)
                    // descriptor addresses the kb-th 1 KB swizzled A tile of the stage.
                    const auto issue_stage_rf = [&](const uint32_t& stage, const uint32_t& unit,
                                                    uint32_t (&f)[kUnitHalves][4][4],
                                                    swap_accum_t (&acc)[kSFGroups][kUnitHalves][kAccChains][kSwapAccum]) {
                        const uint32_t kb = unit / kHalfPairs;
                        fence_accum();
                        fence_frag(f);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; ++ k) {
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage] + kb * (SMEM_A_SIZE_PER_KBLOCK / sizeof(a_dtype_t)) +
                                    k * SwapRS::K, 1);
                                // Inline s2: the set accumulates the whole task (zeroed at task
                                // start), so every group adds; otherwise the first K32 step of
                                // each chain overwrites (fresh set per K-block).
                                SwapRS::wgmma(f[h][k], desc_b,
                                              acc[k / kK32PerSFGroup][h][(k % kK32PerSFGroup) % kAccChains],
                                              kInlineS2 || (((k % kK32PerSFGroup) / kAccChains) > 0));
                            }
                        }
                        ptx::warpgroup_commit_batch();
                    };
                    // Promote K-block `kb` of `stage` with its per-token K128 activation SF.
                    // QoQ: also by the per-row/K128 integer s2 (byte 64 of the packed row,
                    // still resident in this stage); the two int32 chains are summed
                    // exactly, so the result is bit-exact vs the SS path's
                    // (scale * s2) * float(acc).
                    const auto promote_stage_rf = [&](const uint32_t& stage, const uint32_t& unit,
                                                      const swap_accum_t (&acc)[kSFGroups][kUnitHalves][kAccChains][kSwapAccum]) {
                        const uint32_t kb = unit / kHalfPairs, hp = unit % kHalfPairs;
                        #pragma unroll
                        for (uint32_t g = 0; g < kSFGroups; ++ g) {
                        const float* sfa = smem_sfa[stage] + (kb * kNumL2SFAGroups + g) * kL2SFAHalfStride;
                        #pragma unroll
                        for (uint32_t half = 0; half < kUnitHalves; ++ half) {
                            float s2_r0 = 1.0f, s2_r1 = 1.0f;
                            if constexpr (kQoQ) {
                                // s2[row] = byte 64 of the packed row (meta word bytes 64..67):
                                // one LDS.32 per row and an exact u8 -> float via the 2^23 magic
                                // constant. The previous generic u8 load + I2F pair (quarter-rate
                                // XU op right on the promote critical path) measured 475 ns per
                                // 2-block stage against 151 ns for the MXFP4 promote (slot 30).
                                const auto* packed_rows =
                                    reinterpret_cast<const uint8_t*>(smem_packed_b[stage]) + kb * kPackedBKBlockBytes;
                                const uint32_t m0 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                    packed_rows + (wg_n_idx + (hp * kUnitHalves + half) * 64u + r_0) * 80u + 64u));
                                const uint32_t m1 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                    packed_rows + (wg_n_idx + (hp * kUnitHalves + half) * 64u + r_1) * 80u + 64u));
                                s2_r0 = __uint_as_float(0x4B000000u | (m0 & 0xffu)) - 8388608.0f;
                                s2_r1 = __uint_as_float(0x4B000000u | (m1 & 0xffu)) - 8388608.0f;
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t accum_offset = (hp * kUnitHalves + half) * kSwapABHalfAccumPerThread + i * 4;
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                const auto acc_sum = [&](const uint32_t& j) -> float {
                                    swap_accum_t v = acc[g][half][0][j];
                                    #pragma unroll
                                    for (uint32_t c = 1; c < kAccChains; ++ c)
                                        v += acc[g][half][c][j];
                                    if constexpr (kQoQ) {
                                        // int32 -> float without I2F (quarter-rate on SM90; 32
                                        // per thread per K-block here). |v| <= 127 * 15 * 128 <
                                        // 2^22, so 1.5 * 2^23 + v is exact in fp32 and the
                                        // subtract recovers v exactly (bit-exact vs cvt.rn).
                                        return __int_as_float(0x4B400000 + v) - 12582912.0f;
                                    } else {
                                        return static_cast<float>(v);
                                    }
                                };
                                if constexpr (kQoQ) {
                                    // Branch-free: the token guards compiled to one BSSY/BRA/BSYNC
                                    // block per token pair, each with an exposed LDS -> FMUL -> FFMA
                                    // chain (SASS of the 09-09 build; slot-30 probe 525 ns/stage vs
                                    // 150 for MXFP4). Load both SFs unconditionally (the SFA stage
                                    // slot holds BLOCK_M >= token entries) and zero the scale of a
                                    // padded token instead: acc is a finite int32 sum, so 0 * acc == 0
                                    // and final_accum of the padded token stays untouched.
                                    const float raw_0 = ptx::ld_shared(sfa + token_0);
                                    const float raw_1 = ptx::ld_shared(sfa + token_1);
                                    const float scale_0 = token_0 < valid_m ? raw_0 : 0.0f;
                                    const float scale_1 = token_1 < valid_m ? raw_1 : 0.0f;
                                    final_accum[accum_offset + 0] += (scale_0 * s2_r0) * acc_sum(i * 4 + 0);
                                    final_accum[accum_offset + 2] += (scale_0 * s2_r1) * acc_sum(i * 4 + 2);
                                    final_accum[accum_offset + 1] += (scale_1 * s2_r0) * acc_sum(i * 4 + 1);
                                    final_accum[accum_offset + 3] += (scale_1 * s2_r1) * acc_sum(i * 4 + 3);
                                } else {
                                if (token_0 < valid_m) {
                                    const float scale_0 = ptx::ld_shared(sfa + token_0);
                                    final_accum[accum_offset + 0] += (scale_0 * s2_r0) * acc_sum(i * 4 + 0);
                                    final_accum[accum_offset + 2] += (scale_0 * s2_r1) * acc_sum(i * 4 + 2);
                                }
                                if (token_1 < valid_m) {
                                    const float scale_1 = ptx::ld_shared(sfa + token_1);
                                    final_accum[accum_offset + 1] += (scale_1 * s2_r0) * acc_sum(i * 4 + 1);
                                    final_accum[accum_offset + 3] += (scale_1 * s2_r1) * acc_sum(i * 4 + 3);
                                }
                                }
                            }
                        }
                        }
                    };

                    // QoQ inline s2: the per-token activation scale (constant along K, see
                    // kInlineS2) is read once from the first stage's SFA slot (0 for padded
                    // tokens) and applied to the int32 task sum at task end.
                    float act_scale[kSwapAccum / 4][2];
                    const auto capture_act_scale = [&](const uint32_t& stage) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const float raw_0 = ptx::ld_shared(smem_sfa[stage] + token_0);
                            const float raw_1 = ptx::ld_shared(smem_sfa[stage] + token_0 + 1);
                            act_scale[i][0] = token_0 < valid_m ? raw_0 : 0.0f;
                            act_scale[i][1] = token_0 + 1 < valid_m ? raw_1 : 0.0f;
                        }
                    };
                    const auto promote_task_rf = [&](const swap_accum_t (&acc)[kSFGroups][kUnitHalves][kAccChains][kSwapAccum],
                                                     const uint32_t& hp) {
                        #pragma unroll
                        for (uint32_t half = 0; half < kUnitHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t accum_offset = (hp * kUnitHalves + half) * kSwapABHalfAccumPerThread + i * 4;
                                #pragma unroll
                                for (uint32_t j = 0; j < 4; ++ j) {
                                    // Accumulator i * 4 + j holds token (i * 8 + col_idx * 2 + (j & 1))
                                    // of weight row (j < 2 ? r_0 : r_1), like `promote_stage_rf`.
                                    // (Indexing `acc[..][j]` here made every token group i > 0 of a
                                    // BM16 / BM24 block, i.e. rows 8.. of a block with > 8 valid
                                    // rows, re-use group 0's sums under their own activation scale:
                                    // the QoQ T=32 / T=64 cos_min < 0 failures of 2026-09-11.)
                                    swap_accum_t v = acc[0][half][0][i * 4 + j];
                                    #pragma unroll
                                    for (uint32_t c = 1; c < kAccChains; ++ c)
                                        v += acc[0][half][c][i * 4 + j];
                                    // |v| < 2^26 exceeds the 2^22 magic-constant range: one
                                    // exact I2F per element, 32 per thread per TASK.
                                    final_accum[accum_offset + j] += act_scale[i][j & 1] * static_cast<float>(v);
                                }
                            }
                        }
                    };
                    // Half-tile tasks use this one-K-block-per-WG loop too: WG w takes
                    // K-block ksplit_kb = w of every (2-K-block) stage.
                    if constexpr (kUnitsPerStage == 1 || kHalfTileTasks) {
                    if constexpr (kInlineS2) {
                        // One int32 set for the whole task (see the 2-unit inline-s2 loop).
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t c = 0; c < kAccChains; ++ c) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    swap_accum[0][0][h][c][i] = 0;
                            }
                        }
                    }
                    // Software pipeline: the 8 WGMMAs of stage k are issued
                    // asynchronously; while the tensor cores run, the math warps wait
                    // for stage k+1's full barrier (the loader runs kNumStages ahead)
                    // and decode its fragments into the other frag buffer. Only then
                    // is stage k drained, promoted and released. Frag buffer `fcur`
                    // stays live (fenced) until the drain, so its registers cannot be
                    // reused by the k+1 decode while the RS WGMMAs read them.
                    // Probe: 17 = exposed k+1 barrier wait, 18 = k+1 decode (overlapped
                    // with k's WGMMAs), 19 = exposed drain after the decode.
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (!kBlockIsL2)
                            { const unsigned long long kt_fw = clock64() - kt_head; kstage_add(17, kt_fw); kstage_add(36, kt_fw); kstage_add(37, 1ull); }  // 36/37: first-stage wait / count
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        if constexpr (kInlineS2)
                            capture_act_scale(stage_idx);
                        decode_stage_rf(stage_idx, ksplit_kb, frag[0]);
                    }
                    const auto stage_step = [&](uint32_t& k_block_idx,
                                                uint32_t (&fcur)[kUnitHalves][4][4],
                                                uint32_t (&fnext)[kUnitHalves][4][4]) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        if constexpr (!kBlockIsL2) {
                            kstage_add(21, 1ull);
                            if (k_block_idx > 0)
                                kstage_add(22, kt_head - kstage_t_prev);
                            kstage_t_prev = kt_head;
                        }
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, ksplit_kb, fcur, swap_accum[0]);
                        unsigned long long kt_b = clock64();
                        if (k_block_idx + kKBlocksPerStage < num_k_blocks) {
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            if ((kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    if constexpr (!kBlockIsL2)
                                        kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            const unsigned long long kt_a = clock64();
                            if constexpr (!kBlockIsL2)
                                kstage_add(17, kt_a - kt_b);
                            if (!exp_skip(1u))
                                decode_stage_rf(next_stage, ksplit_kb, fnext);
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        }
                        fence_accum();
                        ptx::warpgroup_wait<0>();
                        fence_frag(fcur);
                        kstage_add(19, clock64() - kt_b);
                        if constexpr (!kInlineS2) {
                            if (!exp_skip(8u))
                                promote_stage_rf(cur_stage, ksplit_kb, swap_accum[0]);
                        }
                        arrive_empty_barrier(cur_stage);
                        advance_pipeline(k_block_idx);
                    };
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        stage_step(k_block_idx, frag[0], frag[1]);
                        if (k_block_idx < num_k_blocks)
                            stage_step(k_block_idx, frag[1], frag[0]);
                    }
                    // Explicit drain on the K-loop exit path: without it ptxas' CFG analysis
                    // finds a path from an in-flight wgmma to the next task's accumulator
                    // zeroing / first decode and serialises every wgmma (C7518).
                    ptx::warpgroup_wait<0>();
                    if constexpr (kInlineS2) {
                        if (num_k_blocks > 0 && !exp_skip(8u))
                            promote_task_rf(swap_accum[0], 0u);
                    }
                    } else if constexpr (kUnitsPerStage == 2 && kInlineS2 && kQoQInlineS2Ilv && !kWideUnits) {
                    // QoQ inline s2, interleaved issue. The SASS audit (H20 2026-09-09) showed
                    // the stage is bound by tensor-pipe occupancy: 16 RS m64n8k32 per stage
                    // cost ~110 clk each per WG (two WGs share the pipe), and a wgmma issue
                    // blocks the warp while the pipe queue is full, so the block decode that
                    // followed a whole-block issue ran with the pipe idle. Here each K32 step
                    // is its own commit group G(n, k) (2 wgmma: both 64-row halves) and the
                    // K32 step k of the NEXT block is decoded right after G(n, k) is issued:
                    //   pending before wait = G(n-1, k..3) + G(n, 0..k) = 5 groups, so
                    //   wait<4> retires exactly G(n-1, k), the last reader of fnext[.][k].
                    // Stage s-1 is released after G(2s-1, 3) retired (block 2s, step 3).
                    // Slots: 31 issue windows, 19 wait<4>/wait<0>, 18 per-step decodes,
                    // 17 k+1 barrier check, 30 task promote, 22 head-to-head, 21 stages.
                    #pragma unroll
                    for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t c = 0; c < kAccChains; ++ c) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                swap_accum[0][0][h][c][i] = 0;
                        }
                    }
                    // Decode K32 step `k` of K-block `kb` of `stage` into f[.][k] (one LDS.32
                    // per row for the 4 codes x 2 nibble planes, meta reloaded per step).
                    const auto decode_kstep_rf = [&](const uint32_t& stage, const uint32_t& kb,
                                                     const uint32_t& k, uint32_t (&f)[kUnitHalves][4][4]) {
                        const auto* packed_rows =
                            reinterpret_cast<const uint8_t*>(smem_packed_b[stage]) + kb * kPackedBKBlockBytes;
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            const uint32_t row_0 = wg_n_idx + h * 64u + r_0;
                            const uint32_t row_1 = row_0 + 8u;
                            const uint32_t word_r0 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                packed_rows + row_0 * 80u + col_idx * 16u + k * 4u));
                            const uint32_t word_r1 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                packed_rows + row_1 * 80u + col_idx * 16u + k * 4u));
                            const uint32_t sw0 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(packed_rows + row_0 * 80u + 64u));
                            const uint32_t sw1 = ptx::ld_shared(reinterpret_cast<const uint32_t*>(packed_rows + row_1 * 80u + 64u));
                            const uint32_t s2_0 = sw0 & 0xffu, s2_1 = sw1 & 0xffu;
                            const uint32_t a4_0 = (0x80u - ((sw0 >> 8u) & 0xffu) * s2_0) * 0x01010101u;
                            const uint32_t a4_1 = (0x80u - ((sw1 >> 8u) & 0xffu) * s2_1) * 0x01010101u;
                            f[h][k][0] = (((word_r0 >> 4) & 0x0f0f0f0fu) * s2_0 + a4_0) ^ 0x80808080u;
                            f[h][k][1] = (((word_r1 >> 4) & 0x0f0f0f0fu) * s2_1 + a4_1) ^ 0x80808080u;
                            f[h][k][2] = ((word_r0 & 0x0f0f0f0fu) * s2_0 + a4_0) ^ 0x80808080u;
                            f[h][k][3] = ((word_r1 & 0x0f0f0f0fu) * s2_1 + a4_1) ^ 0x80808080u;
                        }
                    };
                    // Fence only the K32-step-k registers of a fragment buffer: the other steps
                    // are still being read by in-flight groups, and a fence (an asm output
                    // operand, i.e. a definition to ptxas) on a register an in-flight wgmma
                    // reads serialises the pipe (C7513 "non wgmma instructions defining input
                    // registers of a wgmma between start and end of the pipeline stage").
                    const auto fence_frag_k = [&](uint32_t (&f)[kUnitHalves][4][4], const uint32_t& k) {
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t i = 0; i < 4; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(f[h][k][i]));
                        }
                    };
                    // Issue the 4 K32-step groups of block (stage, kb) from fcur, decoding
                    // block (nstage, nkb) step by step into fnext behind wait<4>.
                    const auto block_step_ilv = [&](const uint32_t& stage, const uint32_t& kb,
                                                    uint32_t (&fcur)[kUnitHalves][4][4],
                                                    uint32_t (&fnext)[kUnitHalves][4][4],
                                                    const uint32_t& nstage, const uint32_t& nkb,
                                                    const bool& release_prev,
                                                    const uint32_t& prev_stage) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; ++ k) {
                            unsigned long long kt_a = clock64();
                            if (!exp_skip(2u)) {
                                fence_accum();
                                fence_frag_k(fcur, k);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage] + kb * (SMEM_A_SIZE_PER_KBLOCK / sizeof(a_dtype_t)) +
                                        k * SwapRS::K, 1);
                                    SwapRS::wgmma(fcur[h][k], desc_b, swap_accum[0][0][h][k % kAccChains], true);
                                }
                                ptx::warpgroup_commit_batch();
                            }
                            unsigned long long kt_b = clock64();
                            kstage_add(31, kt_b - kt_a);
                            // Unconditional (also on the task's last block, where fnext is
                            // decoded from a stale slot and never issued): a wgmma-register
                            // write inside a runtime branch makes ptxas serialise the pipe.
                            fence_accum();
                            ptx::warpgroup_wait<4>();
                            fence_frag_k(fnext, k);
                            if (release_prev && k == 3)
                                arrive_empty_barrier(prev_stage);
                            kt_a = clock64();
                            kstage_add(19, kt_a - kt_b);
                            if (!exp_skip(1u))
                                decode_kstep_rf(nstage, nkb, k, fnext);
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        }
                    };
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        { const unsigned long long kt_fw = clock64() - kt_head; kstage_add(17, kt_fw); kstage_add(36, kt_fw); kstage_add(37, 1ull); }  // 36/37: first-stage wait / count
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        capture_act_scale(stage_idx);
                        decode_stage_rf(stage_idx, 0, frag[0]);
                    }
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        const uint32_t prev_stage = cur_stage == 0 ? kNumStages - 1 : cur_stage - 1;
                        const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                        const uint32_t next_phase = phase ^ (next_stage == 0);
                        const bool last_stage = k_block_idx + kKBlocksPerStage >= num_k_blocks;
                        kstage_add(21, 1ull);
                        if (k_block_idx > 0)
                            kstage_add(22, kt_head - kstage_t_prev);
                        kstage_t_prev = kt_head;
                        // Block 0: its groups + the decode of block 1 (same stage).
                        block_step_ilv(cur_stage, 0, frag[0], frag[1], cur_stage, 1, k_block_idx > 0, prev_stage);
                        if (!last_stage) {
                            const unsigned long long kt_a = clock64();
                            if ((kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            kstage_add(17, clock64() - kt_a);
                        }
                        // Block 1: its groups + the decode of the next stage's block 0.
                        block_step_ilv(cur_stage, 1, frag[1], frag[0], next_stage, 0, false, prev_stage);
                        if (last_stage) {
                            const unsigned long long kt_a = clock64();
                            fence_accum();
                            ptx::warpgroup_wait<0>();
                            fence_frag(frag[0]);
                            fence_frag(frag[1]);
                            arrive_empty_barrier(cur_stage);
                            const unsigned long long kt_b = clock64();
                            kstage_add(19, kt_b - kt_a);
                            if (!exp_skip(8u))
                                promote_task_rf(swap_accum[0], 0u);
                            kstage_add(30, clock64() - kt_b);
                        }
                        advance_pipeline(k_block_idx);
                    }
                    // Explicit drain on the K-loop exit path: without it ptxas' CFG analysis
                    // finds a path from an in-flight wgmma to the next task's accumulator
                    // zeroing / first decode and serialises every wgmma (C7518).
                    ptx::warpgroup_wait<0>();
                    } else if constexpr (kUnitsPerStage == 2 && kInlineS2 && kFragBufs > 2) {
                    // QoQ inline s2 with kFragBufs (3 or 4) rotating A-fragment buffers:
                    // same schedule as the 2-buffer loop below, but block n decodes into
                    // frag[n % NF] and the group that last read that buffer is G(n - NF),
                    // so the retire before each decode is wait<NF - 1> (lag NF - 1 groups
                    // instead of 1). Stage t is released once G(2t + 1) retired, i.e. at
                    // the decode of block n with n - NF == 2t + 1 (odd) -> stage (n - NF) / 2;
                    // the stages still pending at task end are released after wait<0>.
                    // The stage loop is unrolled kRot times so the buffer indices are
                    // compile-time (period of n % NF over 2-block stages).
                    constexpr uint32_t NF = kFragBufs;
                    constexpr uint32_t kRot = (NF % 2u == 0u) ? NF / 2u : NF;
                    #pragma unroll
                    for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t c = 0; c < kAccChains; ++ c) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                swap_accum[0][0][h][c][i] = 0;
                        }
                    }
                    uint32_t s_in_task = 0, released = 0;
                    // Smem slot of task stage t (t <= current stage s_in_task).
                    const auto slot_of_stage = [&](const uint32_t& t) -> uint32_t {
                        const uint32_t back = s_in_task - t;
                        return stage_idx >= back ? stage_idx - back : stage_idx + kNumStages - back;
                    };
                    // Block n is about to be decoded: G(n - NF) has retired.
                    const auto maybe_release = [&](const uint32_t& n) {
                        if (n >= NF && ((n - NF) & 1u)) {
                            const uint32_t t = (n - NF) / 2u;
                            arrive_empty_barrier(slot_of_stage(t));
                            released = t + 1;
                        }
                    };
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        { const unsigned long long kt_fw = clock64() - kt_head; kstage_add(17, kt_fw); kstage_add(36, kt_fw); kstage_add(37, 1ull); }  // 36/37: first-stage wait / count
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        capture_act_scale(stage_idx);
                        decode_stage_rf(stage_idx, 0, frag[0]);
                    }
                    const auto stage_body = [&](const uint32_t& rot, uint32_t& k_block_idx) {
                        const uint32_t b0 = (2u * rot) % NF, b1 = (2u * rot + 1u) % NF, b2 = (2u * rot + 2u) % NF;
                        const uint32_t n0 = 2u * s_in_task, n1 = n0 + 1u, n2 = n0 + 2u;
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        kstage_add(21, 1ull);
                        if (k_block_idx > 0)
                            kstage_add(22, kt_head - kstage_t_prev);
                        kstage_t_prev = kt_head;
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, 0, frag[b0], swap_accum[0]);
                        unsigned long long kt_b = clock64();
                        kstage_add(31, kt_b - kt_head);
                        fence_accum();
                        ptx::warpgroup_wait<NF - 1>();
                        fence_frag(frag[b1]);
                        maybe_release(n1);
                        unsigned long long kt_a = clock64();
                        kstage_add(19, kt_a - kt_b);
                        if (!exp_skip(1u))
                            decode_stage_rf(cur_stage, 1, frag[b1]);
                        kt_b = clock64();
                        kstage_add(18, kt_b - kt_a);
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, 1, frag[b1], swap_accum[0]);
                        kt_a = clock64();
                        kstage_add(31, kt_a - kt_b);
                        if (k_block_idx + kKBlocksPerStage < num_k_blocks) {
                            fence_accum();
                            ptx::warpgroup_wait<NF - 1>();
                            fence_frag(frag[b2]);
                            maybe_release(n2);
                            kt_b = clock64();
                            kstage_add(19, kt_b - kt_a);
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            if ((kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            kt_a = clock64();
                            kstage_add(17, kt_a - kt_b);
                            if (!exp_skip(1u))
                                decode_stage_rf(next_stage, 0, frag[b2]);
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        } else {
                            fence_accum();
                            ptx::warpgroup_wait<0>();
                            #pragma unroll
                            for (uint32_t b = 0; b < NF; ++ b)
                                fence_frag(frag[b]);
                            for (uint32_t t = released; t <= s_in_task; ++ t)
                                arrive_empty_barrier(slot_of_stage(t));
                            kt_a = clock64();
                            kstage_add(19, kt_a - kt_b);
                            if (!exp_skip(8u))
                                promote_task_rf(swap_accum[0], 0u);
                            kstage_add(30, clock64() - kt_a);
                        }
                        advance_pipeline(k_block_idx);
                        ++ s_in_task;
                    };
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        #pragma unroll
                        for (uint32_t rot = 0; rot < kRot; ++ rot) {
                            if (k_block_idx < num_k_blocks)
                                stage_body(rot, k_block_idx);
                        }
                    }
                    // Explicit drain on the K-loop exit path: without it ptxas' CFG analysis
                    // finds a path from an in-flight wgmma to the next task's accumulator
                    // zeroing / first decode and serialises every wgmma (C7518).
                    ptx::warpgroup_wait<0>();
                    } else if constexpr (kUnitsPerStage == 2 && kInlineS2) {
                    // QoQ inline s2, two K128 blocks per stage, ONE int32 accumulator set,
                    // no per-block promote. Commit groups G(s,0), G(s,1) per stage; the
                    // tensor pipe is never drained inside the task, only lagged by one
                    // group (the two frag buffers bound the lag: an RS fragment buffer
                    // may be re-decoded only after the group that reads it retired):
                    //   issue G(s,0) [frag 0];
                    //   wait<1>  -> G(s-1,1) retired: frag 1 free, stage s-1's B tile
                    //               fully read -> release stage s-1;
                    //   decode blk1 -> frag 1; issue G(s,1);
                    //   wait<1>  -> G(s,0) retired: frag 0 free;
                    //   wait stage s+1 full; decode its blk0 -> frag 0 (overlaps G(s,1)).
                    // Task end: wait<0>, release the last stage, promote the task sum once.
                    // Probe (per 2-block stage): 17 = exposed k+1 barrier wait, 18 = both
                    // decodes, 19 = exposed wait<1>s, 31 = both issues, 30 = task-end
                    // promote (per stage average), 21 = stage count, 22 = head-to-head.
                    // Knob (kQIS2Prefetch, see its definition): with prefetch the packed
                    // words of the block to decode next are loaded before the wgmma wait
                    // that frees its fragment buffer (blk1: before the first wait<1>; next
                    // blk0: after the k+1 barrier check, before the second wait<1>).
                    {
                        #pragma unroll
                        for (uint32_t u = 0; u < kNumAccKBlocks; ++ u) {
                        #pragma unroll
                        for (uint32_t h = 0; h < kUnitHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t c = 0; c < kAccChains; ++ c) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    swap_accum[u][0][h][c][i] = 0;
                            }
                        }
                        }
                    }
                    uint4 pf_w[kUnitHalves][2];      // prefetched packed words (kQIS2Prefetch)
                    uint32_t pf_sw[kUnitHalves][2];
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        { const unsigned long long kt_fw = clock64() - kt_head; kstage_add(17, kt_fw); kstage_add(36, kt_fw); kstage_add(37, 1ull); }  // 36/37: first-stage wait / count
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        capture_act_scale(stage_idx);
                        decode_stage_rf(stage_idx, 0, frag[0]);
                    }
                    {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        const uint32_t prev_stage = cur_stage == 0 ? kNumStages - 1 : cur_stage - 1;
                        kstage_add(21, 1ull);
                        if (k_block_idx > 0)
                            kstage_add(22, kt_head - kstage_t_prev);
                        kstage_t_prev = kt_head;
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, 0, frag[0], swap_accum[0]);
                        unsigned long long kt_b = clock64();
                        kstage_add(31, kt_b - kt_head);
                        if constexpr (kQIS2Prefetch)
                            load_packed_rf(cur_stage, 1, pf_w, pf_sw);
                        // Retire the previous stage's block-1 group (issued before the k+1
                        // barrier wait and decode, so normally already complete): frag[1]
                        // is free and stage s-1's activation tile can be recycled.
                        fence_accum();
                        ptx::warpgroup_wait<1>();
                        fence_frag(frag[1]);
                        if (k_block_idx > 0)
                            arrive_empty_barrier(prev_stage);
                        unsigned long long kt_a = clock64();
                        kstage_add(19, kt_a - kt_b);
                        if (!exp_skip(1u)) {
                            if constexpr (kQIS2Prefetch)
                                decode_words_rf(pf_w, pf_sw, frag[1]);
                            else
                                decode_stage_rf(cur_stage, 1, frag[1]);
                        }
                        kt_b = clock64();
                        kstage_add(18, kt_b - kt_a);
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, 1, frag[1], swap_accum[1u % kNumAccKBlocks]);
                        kt_a = clock64();
                        kstage_add(31, kt_a - kt_b);
                        const bool has_next = k_block_idx + kKBlocksPerStage < num_k_blocks;
                        const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                        const uint32_t next_phase = phase ^ (next_stage == 0);
                        if constexpr (kQIS2Prefetch) {
                            // k+1 barrier first, then the next block-0 words are in flight
                            // while block 0's group retires.
                            if (has_next && (kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            kt_b = clock64();
                            kstage_add(17, kt_b - kt_a);
                            load_packed_rf(next_stage, 0, pf_w, pf_sw);
                            kt_a = clock64();
                            kstage_add(18, kt_a - kt_b);
                        }
                        // Retire block 0's group so frag[0] can take the next stage's block 0.
                        fence_accum();
                        ptx::warpgroup_wait<1>();
                        fence_frag(frag[0]);
                        kt_b = clock64();
                        kstage_add(19, kt_b - kt_a);
                        if constexpr (!kQIS2Prefetch) {
                            if (has_next && (kexp & 4u) == 0u) {
                                if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                    kstage_add(23, 1ull);
                                    full_barriers[next_stage]->wait(next_phase);
                                }
                            }
                            kt_a = clock64();
                            kstage_add(17, kt_a - kt_b);
                        } else {
                            kt_a = kt_b;
                        }
                        // Decoded unconditionally (stale slot on the task's last stage, never
                        // issued): a frag write in a runtime branch serialises the pipe (ptxas
                        // C7518), see `kExpGates`.
                        if (!exp_skip(1u)) {
                            if constexpr (kQIS2Prefetch)
                                decode_words_rf(pf_w, pf_sw, frag[0]);
                            else
                                decode_stage_rf(next_stage, 0, frag[0]);
                        }
                        kt_b = clock64();
                        kstage_add(18, kt_b - kt_a);
                        if (!has_next) {
                            // Last stage of the task: drain, release it, promote the task sum.
                            fence_accum();
                            ptx::warpgroup_wait<0>();
                            fence_frag(frag[1]);
                            arrive_empty_barrier(cur_stage);
                            kt_a = clock64();
                            kstage_add(19, kt_a - kt_b);
                            if (!exp_skip(8u)) {
                                #pragma unroll
                                for (uint32_t u = 0; u < kNumAccKBlocks; ++ u)
                                    promote_task_rf(swap_accum[u], u);
                            }
                            kstage_add(30, clock64() - kt_a);
                        }
                        advance_pipeline(k_block_idx);
                    }
                    }
                    // Explicit drain on the K-loop exit path: without it ptxas' CFG analysis
                    // finds a path from an in-flight wgmma to the next task's accumulator
                    // zeroing / first decode and serialises every wgmma (C7518).
                    ptx::warpgroup_wait<0>();
                    } else if constexpr (kUnitsPerStage == 2) {
                    // Two K128 blocks per stage, one commit group each, frag buffers at
                    // K-block granularity (frag[0] always holds block 0, frag[1] block 1):
                    //   issue(blk0 -> acc0); decode blk1 -> frag[1]; issue(blk1 -> acc1);
                    //   wait<1> (blk0 group done, frag[0] free);
                    //   wait stage k+1 full; decode its blk0 -> frag[0] (overlaps blk1 WGMMAs);
                    //   wait<0>; promote acc0 (SFA slot 0) + acc1 (SFA slot 1); release stage.
                    // Probe (per 2-block stage): 17 = exposed k+1 barrier wait, 18 = both
                    // decodes (blk1 of k + blk0 of k+1), 19 = exposed drains (wait<1> +
                    // wait<0>), 21 = stage count, 22 = head-to-head stage total.
                    if (num_k_blocks > 0) {
                        const unsigned long long kt_head = clock64();
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (!kBlockIsL2)
                            { const unsigned long long kt_fw = clock64() - kt_head; kstage_add(17, kt_fw); kstage_add(36, kt_fw); kstage_add(37, 1ull); }  // 36/37: first-stage wait / count
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        decode_stage_rf(stage_idx, ksplit_kb, frag[0]);
                    }
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                        const unsigned long long kt_head = clock64();
                        const uint32_t cur_stage = stage_idx;
                        if constexpr (!kBlockIsL2) {
                            kstage_add(21, 1ull);
                            if (k_block_idx > 0)
                                kstage_add(22, kt_head - kstage_t_prev);
                            kstage_t_prev = kt_head;
                        }
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, 0, frag[0], swap_accum[0]);
                        unsigned long long kt_b = clock64();
                        kstage_add(31, kt_b - kt_head);
                        if (!exp_skip(1u))
                            decode_stage_rf(cur_stage, 1, frag[1]);
                        unsigned long long kt_a = clock64();
                        kstage_add(18, kt_a - kt_b);
                        if (!exp_skip(2u))
                            issue_stage_rf(cur_stage, 1, frag[1], swap_accum[1]);
                        kt_b = clock64();
                        kstage_add(31, kt_b - kt_a);
                        kt_a = kt_b;
                        const bool has_next = k_block_idx + kKBlocksPerStage < num_k_blocks;
                        const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                        const uint32_t next_phase = phase ^ (next_stage == 0);
                        uint4 pf_w[kUnitHalves][2];      // kRFPrefetchPacked: next block-0 packed words
                        uint32_t pf_sw[kUnitHalves][2];
                        if constexpr (kRFPrefetchPacked) {
                            // k+1 barrier and the next block-0 packed LDS ahead of the wait<1>
                            // so the smem latency overlaps block 0's retirement.
                            if (has_next) {
                                if ((kexp & 4u) == 0u) {
                                    if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                        if constexpr (!kBlockIsL2)
                                            kstage_add(23, 1ull);
                                        full_barriers[next_stage]->wait(next_phase);
                                    }
                                }
                                kt_b = clock64();
                                if constexpr (!kBlockIsL2)
                                    kstage_add(17, kt_b - kt_a);
                                load_packed_rf(next_stage, 0, pf_w, pf_sw);
                                kt_a = kt_b;
                            }
                        }
                        // Block 0's group was issued before the block-1 decode; retire it
                        // so frag[0] can take the next stage's block 0.
                        fence_accum();
                        ptx::warpgroup_wait<1>();
                        fence_frag(frag[0]);
                        kt_b = clock64();
                        kstage_add(19, kt_b - kt_a);
                        if (has_next) {
                            if constexpr (!kRFPrefetchPacked) {
                                if ((kexp & 4u) == 0u) {
                                    if (!barrier_ready(full_barriers[next_stage], next_phase)) {
                                        if constexpr (!kBlockIsL2)
                                            kstage_add(23, 1ull);
                                        full_barriers[next_stage]->wait(next_phase);
                                    }
                                }
                                kt_a = clock64();
                                if constexpr (!kBlockIsL2)
                                    kstage_add(17, kt_a - kt_b);
                            } else {
                                kt_a = kt_b;
                            }
                            if (!exp_skip(1u)) {
                                if constexpr (kRFPrefetchPacked)
                                    decode_words_rf(pf_w, pf_sw, frag[0]);
                                else
                                    decode_stage_rf(next_stage, 0, frag[0]);
                            }
                            kt_b = clock64();
                            kstage_add(18, kt_b - kt_a);
                        }
                        fence_accum();
                        ptx::warpgroup_wait<0>();
                        fence_frag(frag[1]);
                        kt_a = clock64();
                        kstage_add(19, kt_a - kt_b);
                        if (!exp_skip(8u)) {
                            promote_stage_rf(cur_stage, ksplit_kb, swap_accum[0]);
                            promote_stage_rf(cur_stage, 1, swap_accum[1]);
                        }
                        // 30 = both promotes (QoQ: s2 byte loads + int32 -> float + scale).
                        kstage_add(30, clock64() - kt_a);
                        arrive_empty_barrier(cur_stage);
                        advance_pipeline(k_block_idx);
                    }
                    ptx::warpgroup_wait<0>();  // explicit drain on the loop exit path (ptxas C7518, see above)
                    }
                };
                if constexpr (BLOCK_M == 8) {
                    run_swap_ab_rf.template operator()<8>();
                } else if constexpr (BLOCK_M == 16) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_rf.template operator()<8>();
                    } else {
                        run_swap_ab_rf.template operator()<16>();
                    }
                } else if constexpr (BLOCK_M == 24) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_rf.template operator()<8>();
                    } else if (n_swap <= 16) {
                        run_swap_ab_rf.template operator()<16>();
                    } else {
                        run_swap_ab_rf.template operator()<24>();
                    }
                }
            } else if constexpr (kSwapABRequested && kSwapPipelineDecode) {
                auto run_swap_ab_pipelined = [&]<uint32_t N_SWAP>() {
                    using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                    constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                    float swap_accum[kWGHalves][kSwapAccum];

                    if (num_k_blocks > 0) {
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (kUseInterleavedScheduler)
                            interleaved_scheduler.release_task_info(lane_idx);
                        decode_b_stage(stage_idx);
                    }
                    for (uint32_t k_block_idx = 0;
                         k_block_idx < num_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        const uint32_t cur_stage = stage_idx;
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[half][i]);
                        }
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[cur_stage] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[cur_stage] + k * SwapWGMMA::K, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum[half], k);
                            }
                        }
                        ptx::warpgroup_commit_batch();

                        // Decode the next stage while the WGMMAs of this one run.
                        if (k_block_idx + 1 < num_k_blocks) {
                            const uint32_t next_stage = cur_stage == kNumStages - 1 ? 0 : cur_stage + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0);
                            full_barriers[next_stage]->wait(next_phase);
                            decode_b_stage(next_stage);
                        }

                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[half][i]);
                        }
                        ptx::warpgroup_wait<0>();

                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                if (token_0 < valid_m) {
                                    const float scale_0 = ptx::ld_shared(smem_sfa[cur_stage] + token_0);
                                    final_accum[accum_offset + 0] += scale_0 * swap_accum[half][i * 4 + 0];
                                    final_accum[accum_offset + 2] += scale_0 * swap_accum[half][i * 4 + 2];
                                }
                                if (token_1 < valid_m) {
                                    const float scale_1 = ptx::ld_shared(smem_sfa[cur_stage] + token_1);
                                    final_accum[accum_offset + 1] += scale_1 * swap_accum[half][i * 4 + 1];
                                    final_accum[accum_offset + 3] += scale_1 * swap_accum[half][i * 4 + 3];
                                }
                            }
                        }
                        arrive_empty_barrier(cur_stage);
                    }
                };
                if constexpr (BLOCK_M == 8) {
                    run_swap_ab_pipelined.template operator()<8>();
                } else if constexpr (BLOCK_M == 16) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_pipelined.template operator()<8>();
                    } else {
                        run_swap_ab_pipelined.template operator()<16>();
                    }
                } else if constexpr (BLOCK_M == 24) {
                    const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                    if (n_swap <= 8) {
                        run_swap_ab_pipelined.template operator()<8>();
                    } else if (n_swap <= 16) {
                        run_swap_ab_pipelined.template operator()<16>();
                    } else {
                        run_swap_ab_pipelined.template operator()<24>();
                    }
                }
            } else {
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks;
                 advance_pipeline(k_block_idx)) {
                const unsigned long long kt_head = clock64();
                full_barriers[stage_idx]->wait(phase);
                if constexpr (!kBlockIsL2) {
                    { const unsigned long long kt_fw = clock64() - kt_head; kstage_add(17, kt_fw); kstage_add(36, kt_fw); kstage_add(37, 1ull); }  // 36/37: first-stage wait / count
                    kstage_add(21, 1ull);
                    if (k_block_idx > 0)
                        kstage_add(22, kt_head - kstage_t_prev);
                    kstage_t_prev = kt_head;
                }
                if constexpr (kUseInterleavedScheduler) {
                    if (k_block_idx == 0)
                        interleaved_scheduler.release_task_info(lane_idx);
                }
                if constexpr (!kRFDecode)
                    decode_b_stage(stage_idx);

                // Read SF (must precede warpgroup_arrive)
                const float scale_a_0_lo =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                const float scale_a_1_lo =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                float scale_a_0_hi = 0.0f;
                float scale_a_1_hi = 0.0f;
                if constexpr (kBlockIsL2 && kSplitMDecodedWeightReuse) {
                    scale_a_0_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                }

                // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
                // expansion, so the WGMMA accumulator only needs activation SF.

                if constexpr (!kBlockIsL2) {
                    if constexpr (kSwapABRequested) {
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            // INT8 atoms have no N=24; pad to 32 (extra token columns are
                            // masked by `token < valid_m`, smem A rows beyond BLOCK_M are
                            // never consumed).
                            using SwapWGMMA = typename std::conditional_t<kQoQ,
                                mma::sm90::INT8MMASelector<(N_SWAP == 24 ? 32 : N_SWAP)>,
                                mma::sm90::FP8MMASelector<N_SWAP>>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            swap_accum_t swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t half = 0; half < kWGHalves; ++ half) {
                                {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                }

                                // QoQ: fold the per-row/K128 integer s2 (byte 64 of the
                                // packed row, still resident in this stage) into the promote.
                                float s2_r0 = 1.0f, s2_r1 = 1.0f;
                                if constexpr (kQoQ) {
                                    const auto* packed_rows = reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]);
                                    s2_r0 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_0) * 80u + 64u]);
                                    s2_r1 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_1) * 80u + 64u]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] += (scale_0 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 0]);
                                        final_accum[accum_offset + 2] += (scale_0 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 2]);
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] += (scale_1 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 1]);
                                        final_accum[accum_offset + 3] += (scale_1 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 3]);
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l1.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else {
                                run_swap_ab_l1.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else {
                                run_swap_ab_l1.template operator()<24>();
                            }
                        }
                    } else {
                        float accum[kAccumPerThread];
                        // Single per-128 K-block WGMMA group
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        // L1: gate/up alternate at gran=8 along N; each `i` block
                        // of 8 cols belongs entirely to one of {gate, up}, so .x
                        // and .y share the same scalar.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                        }
                    }
                } else {
                    if constexpr (kSwapABRequested) {
                        DG_STATIC_ASSERT(kL2ActsSFGranK == 128,
                                         "L2 swap-AB requires per-128 activation scales");
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename std::conditional_t<kQoQ,
                                mma::sm90::INT8MMASelector<(N_SWAP == 24 ? 32 : N_SWAP)>,
                                mma::sm90::FP8MMASelector<N_SWAP>>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            swap_accum_t swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t half = 0; half < kWGHalves; ++ half) {
                                {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                }
                                float s2_r0 = 1.0f, s2_r1 = 1.0f;
                                if constexpr (kQoQ) {
                                    const auto* packed_rows = reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]);
                                    s2_r0 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_0) * 80u + 64u]);
                                    s2_r1 = static_cast<float>(packed_rows[(wg_n_idx + half * 64u + r_1) * 80u + 64u]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset =
                                        half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] +=
                                            (scale_0 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 0]);
                                        final_accum[accum_offset + 2] +=
                                            (scale_0 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 2]);
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] +=
                                            (scale_1 * s2_r0) * static_cast<float>(swap_accum[i * 4 + 1]);
                                        final_accum[accum_offset + 3] +=
                                            (scale_1 * s2_r1) * static_cast<float>(swap_accum[i * 4 + 3]);
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l2.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else {
                                run_swap_ab_l2.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else {
                                run_swap_ab_l2.template operator()<24>();
                            }
                        }
                    } else {
                        float accum[kAccumPerThread];
                        const auto promote_l2_accum = [&](const float& scale_r0,
                                                          const float& scale_r1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_r0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_r0 * accum[i*4+1];
                                final_accum[i*4+2] += scale_r1 * accum[i*4+2];
                                final_accum[i*4+3] += scale_r1 * accum[i*4+3];
                            }
                        };
                        if constexpr (kSplitMDecodedWeightReuse) {
                            #pragma unroll
                            for (uint32_t sf_group = 0; sf_group < 2; ++ sf_group) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                    const uint32_t k_off =
                                        sf_group * (BLOCK_K / 2) + k * WGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();
                                if (sf_group == 0)
                                    promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                                else
                                    promote_l2_accum(scale_a_0_hi, scale_a_1_hi);
                            }
                            arrive_empty_barrier(stage_idx);
                        } else {
                            // One per-128 scale permits a single four-instruction
                            // WGMMA group and one accumulator promotion per K tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                    k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                            arrive_empty_barrier(stage_idx);
                            promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                        }
                    }
                }
            }
            }  // kSwapPipelineDecode

            // Skip epilogue when block is past valid M (the GEMM loop already
            // released its pipeline stages). Drain any prior L1 async store.
            if (valid_m == 0) {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            // Half-tile tasks: K-split hand-off. WG1 publishes its promoted partial
            // sums ([element][thread] floats, conflict-free), WG0 adds them and then
            // runs the epilogue alone; WG1 only keeps the CTA-wide barriers company.
            // WAR safety: WG1's next write happens after it passes this task's
            // epilogue-wide barriers, which WG0 reaches only after reading here.
            if constexpr (kHalfTileTasks) {
                const uint32_t tid_in_wg = epilogue_thread_idx & 127u;
                if (epilogue_wg_idx == 1) {
                    #pragma unroll
                    for (uint32_t h = 0; h < kWGHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                smem_ksplit_reduce[((h * kSwapABTokenChunks + i) * 4 + j) * 128u + tid_in_wg] =
                                    final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j];
                        }
                    }
                }
                asm volatile("bar.sync %0, %1;" : :
                             "n"(kKSplitReduceBarrierIdx), "n"(kNumEpilogueThreads) : "memory");
                if (epilogue_wg_idx == 0) {
                    #pragma unroll
                    for (uint32_t h = 0; h < kWGHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j] +=
                                    smem_ksplit_reduce[((h * kSwapABTokenChunks + i) * 4 + j) * 128u + tid_in_wg];
                        }
                    }
                }
            }

            // Split-K L1 (kSplitKL1): cross-CTA reduction of the two K halves of this
            // (pool_block, n_block). Roles are fixed by the K half so no counter
            // round-trip sits on the critical path: half 0 (PUBLISHER) stores its
            // promoted partials to the scratch slot ([element][256 threads] floats,
            // coalesced), one thread releases the ready flag after the CTA barrier
            // (bar.sync + release.gpu is cumulative over the other threads' stores,
            // as in CUTLASS's split-K semaphore) and the CTA moves on: no epilogue,
            // no L1-ready notify (every CTA-wide barrier of the task was passed by
            // all 256 threads). Half 1 (FINISHER) acquires the flag (its publisher
            // was claimed one index earlier and never waits on L2 progress, so the
            // wait is short and deadlock-free), resets it for the next launch, adds
            // the partner's slot and runs the normal epilogue + notify.
            // valid_m == 0 tasks skipped this on both halves (flag untouched).
            // Split-K L2 (kSplitKL2) uses the identical protocol on the L2 tail tasks
            // (own slot/flag index space): the L2 accumulator is also two 64-row
            // weight halves x 8 tokens per thread, the finisher runs the L2 epilogue
            // (BF16 x scale, NVLink scatter). Deadlock-free for the same reason: the
            // publisher (index i) only waits on L1-ready bits, the finisher (i + 1)
            // only on the publisher; by induction on the claim order every earlier
            // task completes.
            // Stream-K (kStreamK, active launches): n-way generalisation. Every
            // contributor of a > 1-split tile stores its partial to its own worker
            // slot (slot 0 if its segment starts at K-block 0, else slot 1: a worker
            // has at most one of each per phase), then thread 0 takes an acq_rel
            // ticket on the tile's arrival counter and broadcasts it through smem.
            // Non-last arrivers leave (no epilogue); the last one (ticket ==
            // splits - 1) resets the counter for the next launch, sums the partials
            // in split order (contributor i == worker first + i, slot i == 0 ? 0 : 1;
            // its own from registers) and runs the epilogue. The partial store of
            // the eventual finisher is wasted (8 KB) but keeps the protocol
            // role-free. Deadlock-free: no contributor waits on anything here.
            // L2 tail probe: 40 = max L2 epilogue start (K loop drained)
            if (kBlockIsL2 && epilogue_thread_idx == 0) stamp_max(40);
            // Stream-K protocol only for stream-K segments; a split-K TAIL task of a
            // wave-scheduled launch (stream-K compiled in but inactive: >= kNumSMs L1
            // tasks, i.e. >= 8 active local experts at <= 8 global tokens under real,
            // unbalanced routing) takes the split-K branch below. Before this
            // distinction such tails wrote their partials to stream-K worker slots
            // 0/1 shared by every concurrently running tail (wrong sums on the
            // highest-index experts of those ranks; the balanced correctness test
            // never reaches >= 78 L1 tasks with stream-K compiled in).
            const bool streamk_reduce = kStreamK && first_worker_idx != kNotStreamKWorker;
            if constexpr (kStreamK) {
                if (num_k_splits > 1 && streamk_reduce) {
                    constexpr uint32_t kNumPartialElems = kWGHalves * kSwapABTokenChunks * 4u;
                    DG_STATIC_ASSERT(!kStreamK ||
                                     kNumPartialElems * kNumEpilogueThreads * sizeof(float) ==
                                     fused_layout::kSM90SplitKL1PartialBytes,
                                     "Stream-K partial slot size mismatch");
                    DG_STATIC_ASSERT(!kStreamK || kNumSMs <= fused_layout::kSM90StreamKMaxSMs,
                                     "Stream-K per-worker slots do not cover the grid");
                    const uint32_t worker_idx = first_worker_idx + k_split_idx;
                    float* my_slot = workspace.get_streamk_scratch_ptr(
                        kBlockIsL2, worker_idx, k_split_idx == 0u ? 0u : 1u);
                    uint32_t* counter = kBlockIsL2 ?
                        workspace.get_splitk_l2_flag_ptr(pool_block_idx, n_block_idx) :
                        workspace.get_splitk_l1_flag_ptr(pool_block_idx, n_block_idx);
                    #pragma unroll
                    for (uint32_t h = 0; h < kWGHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                __stcg(my_slot + ((h * kSwapABTokenChunks + i) * 4 + j) * kNumEpilogueThreads + epilogue_thread_idx,
                                       final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j]);
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0) {
                        const uint32_t ticket = ptx::atomic_add_acq_rel(counter, 1u);
                        if (ticket == num_k_splits - 1u)
                            *counter = 0u;
                        *smem_streamk_ticket = ticket;
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    const uint32_t ticket = *smem_streamk_ticket;
                    if (ticket != num_k_splits - 1u)
                        return;  // another contributor finishes this tile
                    float partial_sum[kNumPartialElems];
                    #pragma unroll
                    for (uint32_t e = 0; e < kNumPartialElems; ++ e)
                        partial_sum[e] = 0.0f;
                    for (uint32_t c = 0; c < num_k_splits; ++ c) {
                        if (c == k_split_idx) {
                            #pragma unroll
                            for (uint32_t h = 0; h < kWGHalves; ++ h) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                                    #pragma unroll
                                    for (uint32_t j = 0; j < 4; ++ j)
                                        partial_sum[(h * kSwapABTokenChunks + i) * 4 + j] +=
                                            final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j];
                                }
                            }
                        } else {
                            const float* slot = workspace.get_streamk_scratch_ptr(
                                kBlockIsL2, first_worker_idx + c, c == 0u ? 0u : 1u);
                            #pragma unroll
                            for (uint32_t e = 0; e < kNumPartialElems; ++ e)
                                partial_sum[e] += __ldcg(slot + e * kNumEpilogueThreads + epilogue_thread_idx);
                        }
                    }
                    #pragma unroll
                    for (uint32_t h = 0; h < kWGHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j)
                                final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j] =
                                    partial_sum[(h * kSwapABTokenChunks + i) * 4 + j];
                        }
                    }
                }
            }
            if constexpr ((kSplitKL1 && !kBlockIsL2) || (kSplitKL2 && kBlockIsL2)) {
                if (num_k_splits > 1 && !streamk_reduce) {
                    constexpr uint32_t kNumPartialElems = kWGHalves * kSwapABTokenChunks * 4u;
                    // (Guarded: this discarded branch is not template-dependent.)
                    // Wide tasks: the partial spans kPhaseTiles adjacent 8 KB slots (the
                    // slot index space is per packed tile: n_block * kPhaseTiles).
                    DG_STATIC_ASSERT(!(kSplitKL1 || kSplitKL2) ||
                                     kNumPartialElems * kNumEpilogueThreads * sizeof(float) ==
                                     fused_layout::kSM90SplitKL1PartialBytes * kPhaseTiles,
                                     "Split-K partial slot size mismatch");
                    DG_STATIC_ASSERT(fused_layout::kSM90SplitKL1NumKSplits == 2,
                                     "L1 split-K default handshake: one publisher and one finisher half");
                    DG_STATIC_ASSERT(kBlockIsL2 || kNumL1KSplits == 2 || (kPhaseTiles > 1 && !kSplitKL2 && !kStreamK),
                                     "Wide L1 3-way split needs the L2 split-K scratch to be free");
                    // Publisher p (k_split_idx < num_k_splits - 1) owns partial slot p;
                    // the finisher (last split) sums slots 0 .. num_k_splits - 2.
                    // Wide L1 (3-way): publisher p's 16 KB partial = the two 8 KB publisher
                    // slots of L2 N-block (n_block * 2 + p) (<= 9 < 12), contiguous.
                    const auto slot_of = [&](const uint32_t& publisher_idx) -> float* {
                        if constexpr (kBlockIsL2)
                            return workspace.get_splitk_l2_scratch_ptr(pool_block_idx, n_block_idx, publisher_idx);
                        else if constexpr (kPhaseTiles > 1)
                            return workspace.get_splitk_l2_scratch_ptr(
                                pool_block_idx, n_block_idx * kPhaseTiles + publisher_idx, 0u);
                        else
                            return workspace.get_splitk_l1_scratch_ptr(pool_block_idx, n_block_idx);
                    };
                    float* slot = slot_of(k_split_idx);
                    uint32_t* flag = kBlockIsL2 ?
                        workspace.get_splitk_l2_flag_ptr(pool_block_idx, n_block_idx) :
                        workspace.get_splitk_l1_flag_ptr(pool_block_idx, n_block_idx);
                    if (k_split_idx + 1u < num_k_splits) {
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                                #pragma unroll
                                for (uint32_t j = 0; j < 4; ++ j)
                                    __stcg(slot + ((h * kSwapABTokenChunks + i) * 4 + j) * kNumEpilogueThreads + epilogue_thread_idx,
                                           final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j]);
                            }
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_thread_idx == 0)
                            ptx::red_add_rel(flag, 1u);
                        return;  // PUBLISHER: the finisher CTA completes this task
                    }
                    if (epilogue_thread_idx == 0) {
                        DG_SPIN_WHILE(ptx::ld_acq(flag) != num_k_splits - 1u, 2827);
                        *flag = 0u;
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    for (uint32_t pub = 0; pub + 1u < num_k_splits; ++ pub) {
                        const float* pslot = slot_of(pub);
                        #pragma unroll
                        for (uint32_t h = 0; h < kWGHalves; ++ h) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                                #pragma unroll
                                for (uint32_t j = 0; j < 4; ++ j)
                                    final_accum[h * kSwapABHalfAccumPerThread + i * 4 + j] +=
                                        __ldcg(pslot + ((h * kSwapABTokenChunks + i) * 4 + j) * kNumEpilogueThreads + epilogue_thread_idx);
                            }
                        }
                    }
                }
            }

            if constexpr (!kBlockIsL2) {
                const float l1_global_scale = (kPerRowEpilogueScale || l1_global_scales == nullptr) ?
                    1.0f : __ldg(l1_global_scales + local_expert_idx);
                // MXFP4 per-weight-row scales, indexed by the interleaved L1
                // weight row (gate/up interleaved by 8) within this expert.
                const float* __restrict__ l1_row_scales = kPerRowEpilogueScale ?
                    l1_global_scales + local_expert_idx * L1_SHAPE_N + n_idx : nullptr;
                const auto l1_scale_at = [&](const uint32_t& row) -> float {
                    if constexpr (kPerRowEpilogueScale)
                        return __ldg(l1_row_scales + row);
                    else
                        return l1_global_scale;
                };
                // L2 activation byte: FP8 E4M3 (FP4 paths) or INT8 (QoQ).
                const auto quantize_l1_out_byte = [](const float& v) -> uint8_t {
                    if constexpr (kQoQ) {
                        const int q = cute::min(cute::max(__float2int_rn(v), -127), 127);
                        return static_cast<uint8_t>(static_cast<int8_t>(q));
                    } else {
                        const __nv_fp8_e4m3 q(v);
                        return *reinterpret_cast<const uint8_t*>(&q);
                    }
                };
                if constexpr (kSwapABRequested) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    constexpr uint32_t reduce_warp_start = 0;
                    // Half-tile tasks: only WG0's warps hold the reduced accumulators.
                    constexpr uint32_t reduce_warp_count =
                        kHalfTileTasks ? kNumEpilogueWarps / 2 : kNumEpilogueWarps;
                    // L2 SF groups of this task (wide L1 tasks: 2, WG w owns group w): the
                    // per-token amax is reduced over the kWarpsPerSFGroup warps of a group.
                    constexpr uint32_t kWarpsPerSFGroup = reduce_warp_count / kL1SFGroups;
                    DG_STATIC_ASSERT(kL1SFGroups == 1 || kWarpsPerSFGroup == 4,
                                     "Wide L1 tasks: one SF group per WG (4 warps)");
                    const uint32_t sf_group_of_warp =
                        kL1SFGroups == 1 ? 0u : (epilogue_warp_idx - reduce_warp_start) / kWarpsPerSFGroup;
                    const uint32_t scale_token_thread = epilogue_thread_idx;
                    constexpr uint32_t scale_token_stride = kNumEpilogueThreads;
                    const uint32_t sf_base_k_idx =
                        n_block_idx * L1_OUT_BLOCK_N / kL2ActsSFGranK;
                    float swap_v0[kWGHalves][kSwapABTokenChunks] = {};
                    float swap_v1[kWGHalves][kSwapABTokenChunks] = {};

                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;

                        float v0_amax = 0.0f;
                        float v1_amax = 0.0f;
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            // swapAB: WGMMA rows are weight rows; r_0 = gate, r_1 = up.
                            const float gate_scale = l1_scale_at(half * 64u + r_0);
                            const float up_scale = l1_scale_at(half * 64u + r_1);
                            float v0 = 0.0f;
                            if (token_0 < valid_m) {
                                float g0 = final_accum[accum_offset + 0] * gate_scale;
                                float u0 = final_accum[accum_offset + 2] * up_scale;
                                clamp_gate(g0);
                                clamp_up(u0);
                                const float weight_0 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_0)
                                    .template get_base_ptr<float>();
                                v0 = silu(g0) * u0 * weight_0;
                                swap_v0[half][i] = v0;
                                v0_amax = cute::max(v0_amax, cute::abs(v0));
                            }

                            float v1 = 0.0f;
                            if (token_1 < valid_m) {
                                float g1 = final_accum[accum_offset + 1] * gate_scale;
                                float u1 = final_accum[accum_offset + 3] * up_scale;
                                clamp_gate(g1);
                                clamp_up(u1);
                                const float weight_1 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_1)
                                    .template get_base_ptr<float>();
                                v1 = silu(g1) * u1 * weight_1;
                                swap_v1[half][i] = v1;
                                v1_amax = cute::max(v1_amax, cute::abs(v1));
                            }
                        }

                        const float amax0 = math::warp_reduce<4, true>(
                            v0_amax, math::ReduceMax<float>());
                        const float amax1 = math::warp_reduce<4, true>(
                            v1_amax, math::ReduceMax<float>());
                        if (row_idx == 0) {
                            if (token_0 < valid_m)
                                smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
                            if (token_1 < valid_m)
                                smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    if (is_epilogue_wg) {
                        store_l1_swap_chunk(0);
                        if (valid_m > 8) {
                            #pragma unroll
                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                if (i < num_swap_token_chunks)
                                    store_l1_swap_chunk(i);
                            }
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    for (uint32_t token_group = scale_token_thread;
                         token_group < valid_m * kL1SFGroups;
                         token_group += scale_token_stride) {
                        const uint32_t token = token_group / kL1SFGroups;
                        const uint32_t sf_group = token_group % kL1SFGroups;
                        const uint32_t group_warp_start = reduce_warp_start + sf_group * kWarpsPerSFGroup;
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t w = 0; w < kWarpsPerSFGroup; ++ w)
                            amax = cute::max(
                                amax, smem_cd_l1_shared_sf[token * kNumEpilogueWarps + group_warp_start + w]);
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        if constexpr (kQoQ) {
                            // INT8 activation for L2: symmetric per-token scale amax/127.
                            sf_pair.x = sf_pair.y = amax * (1.0f / 127.0f);
                            // `__fdividef` (MUFU.RCP + FMUL), NOT `127.0f / amax`: the IEEE
                            // fp32 division emits a CALL to a slow-path subroutine, and ANY
                            // call in this function makes ptxas serialise every wgmma of the
                            // kernel (C7510 "wgmma pipeline crossing function boundary":
                            // WARPGROUP.DEPBAR.LE gsb0, 0x0 after each IGMMA). Measured on
                            // H20 (SASS audit 2026-09-09): 16 serialised m64n8k32 per stage.
                            sf_inv_pair.x = sf_inv_pair.y = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
                        } else {
                            math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        }

                        {
                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_idx = m_idx + token;
                            sf_base_ptr[(sf_base_k_idx + sf_group) * kNumPaddedSFPoolTokens + token_idx] =
                                sf_pair.x;
                        }
                        smem_cd_l1_shared_sf[token * kNumEpilogueWarps + group_warp_start] = sf_inv_pair.x;
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    #pragma unroll
                    for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                        if (!is_epilogue_wg) break;
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t half = 0; half < kWGHalves; ++ half) {
                            const uint32_t out_col_base =
                                wg_l1_out_n_idx + half * 32u + warp_idx_in_wg * 8 + row_idx;
                            const uint32_t sf_inv_slot = reduce_warp_start + sf_group_of_warp * kWarpsPerSFGroup;
                            if (token_0 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + sf_inv_slot];
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                    quantize_l1_out_byte(swap_v0[half][i] * sf_inv);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + sf_inv_slot];
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                    quantize_l1_out_byte(swap_v1[half][i] * sf_inv);
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    {
                        if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1,
                                out_n_idx,
                                m_idx);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                        ptx::tma_store_wait<0>();
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        notify_l1_ready(pool_block_idx, n_block_idx);
                    }
                } else {
                    // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;
                    // Layout in `final_accum`:
                    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                    //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                    //
                    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    constexpr uint32_t kNumSFGroups = 1;
                    float swiglu_r0[kNumPairs][2];
                    float swiglu_r1[kNumPairs][2];

                    // Per-row amax, one scale for each 64-col L1 output group.
                    float amax_r0[kNumSFGroups] = {};
                    float amax_r1[kNumSFGroups] = {};

                    // Compute SwiGLU + per-group amax.
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        const uint32_t sf_group = p / 8;

                        auto clamp_gate = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(x, kActivationClamp);
                        };
                        auto clamp_up = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                        };
                        // Weight rows: gate chunk covers interleaved rows
                        // 16p + col_idx*2 + {0,1}, up chunk the same + 8.
                        const float gs_c0 = l1_scale_at(gate * 8 + col_idx * 2);
                        const float gs_c1 = l1_scale_at(gate * 8 + col_idx * 2 + 1);
                        const float us_c0 = l1_scale_at(up * 8 + col_idx * 2);
                        const float us_c1 = l1_scale_at(up * 8 + col_idx * 2 + 1);
                        float g_r0_c0 = final_accum[gate*4 + 0] * gs_c0; clamp_gate(g_r0_c0);
                        float g_r0_c1 = final_accum[gate*4 + 1] * gs_c1; clamp_gate(g_r0_c1);
                        float g_r1_c0 = final_accum[gate*4 + 2] * gs_c0; clamp_gate(g_r1_c0);
                        float g_r1_c1 = final_accum[gate*4 + 3] * gs_c1; clamp_gate(g_r1_c1);
                        float u_r0_c0 = final_accum[up*4   + 0] * us_c0; clamp_up(u_r0_c0);
                        float u_r0_c1 = final_accum[up*4   + 1] * us_c1; clamp_up(u_r0_c1);
                        float u_r1_c0 = final_accum[up*4   + 2] * us_c0; clamp_up(u_r1_c0);
                        float u_r1_c1 = final_accum[up*4   + 3] * us_c1; clamp_up(u_r1_c1);

                        auto silu = [](float x) -> float {
                            const float e = kFastMath ? __expf(-x) : expf(-x);
                            const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                            return x * sig;
                        };

                        if (valid_r0) {
                            swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                            swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                            amax_r0[sf_group] = cute::max(
                                amax_r0[sf_group],
                                cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                        } else {
                            swiglu_r0[p][0] = 0.0f;
                            swiglu_r0[p][1] = 0.0f;
                        }
                        if (valid_r1) {
                            swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                            swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                            amax_r1[sf_group] = cute::max(
                                amax_r1[sf_group],
                                cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                        } else {
                            swiglu_r1[p][0] = 0.0f;
                            swiglu_r1[p][1] = 0.0f;
                        }
                    }


                    const float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r0)
                        .template get_base_ptr<float>() : 0.0f;
                    const float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r1)
                        .template get_base_ptr<float>() : 0.0f;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        swiglu_r0[p][0] *= weight_r0;
                        swiglu_r0[p][1] *= weight_r0;
                        swiglu_r1[p][0] *= weight_r1;
                        swiglu_r1[p][1] *= weight_r1;
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] *= cute::abs(weight_r0);
                        amax_r1[g] *= cute::abs(weight_r1);
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                    }

                    if (col_idx == 0) {
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0] = amax_r0[0];
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1] = amax_r1[0];
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if constexpr (kSplitMDecodedWeightReuse) {
                        amax_r0[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r0];
                        amax_r1[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r1];
                    } else {
                        amax_r0[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r0],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r0]);
                        amax_r1[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r1],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r1]);
                    }

                    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                        sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                    }

                    // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t sf_group = p / 8;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                        const uint32_t col = p * 8 + col_idx * 2;
                        auto* p0 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r0 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r1 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        if (valid_r0)
                            *p0 = r0_pair.__x;
                        if (valid_r1)
                            *p1 = r1_pair.__x;
                    }

                    // Write one physical L2-activation scale per 128 output columns.
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = m_idx + row_offset_r0;
                        const uint32_t token_r1 = m_idx + row_offset_r1;
                        const uint32_t base_k_sf_idx =
                            (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / kL2ActsSFGranK;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            const uint32_t sf_k_idx = base_k_sf_idx + g;
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r0)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0[g];
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r1)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1[g];
                        }
                    }

                    // Issue TMA store of the entire tile. Padding rows beyond
                    // `valid_m` are written with stale/garbage FP8 to the L1-output
                    // pool buffer, but they are never consumed downstream: the L2
                    // GEMM tile loads them, but its NVLink-scatter epilogue is
                    // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                    // padding rows can produce NaN accumulators that simply stay
                    // in registers (only valid rows are converted to BF16 and
                    // STSM'd into smem). Using TMA for partial tiles is a large
                    // win for low-batch / decode where every tile is partial.
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                if constexpr (kSwapABRequested) {
                    l2_epilogue_swap(n_block_idx);
                } else {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 64 || WG_BLOCK_N == 128,
                                     "Direct L2 scatter requires N64/N128");
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row, const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                                pool_block_idx * BLOCK_M + row_offset);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = cast_l2_scaled_bf16_pair(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1], col_lo);
                                const uint32_t packed_hi = cast_l2_scaled_bf16_pair(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1], col_hi);
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    signal_combine_arrivals();
                }
            }
        };
        unsigned long long ktask_prev_end = 0ull;
        // Per-CTA task log (probe buffers with >= 48 + 160 * 16 slots; see
        // tests/profile_fused_phase_stamps.py PROBE_TASKLOG): slot 48 + sm * 16 + {2t, 2t+1}
        // = task t's (start, end) in ns since this CTA's kernel entry (low 32 bits), start
        // word high bits = meta (bit 31 L2, 30..24 pool block, 23..16 n block, 15..8 k split,
        // 7..0 num k splits); + 14 = this CTA's absolute entry globaltimer, + 15 = task count.
        constexpr uint32_t kTaskLogBase = 64, kTaskLogPerCTA = 16, kTaskLogMaxTasks = 7, kTaskLogMaxSMs = 160;
        uint32_t task_log_seq = 0;
        const bool task_log_on = (phase_stamps != nullptr) && (epilogue_thread_idx == 0) &&
                                 (sm_idx < kTaskLogMaxSMs) && (phase_stamps[47] == 0x5441534bull);
        const auto run_math_task = [&](const auto& block_phase,
                                       const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks,
                                       const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                       const uint32_t& pool_block_idx,
                                       const uint32_t& valid_m,
                                       const uint32_t& k_split_idx,
                                       const uint32_t& num_k_splits,
                                       const uint32_t& k_block_begin,
                                       const uint32_t& first_worker_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == fused_sched::BlockPhase::Linear2;
            if (epilogue_thread_idx == 0) stamp_min(3);
            // L2 tail probe (all SMs, globaltimer): 36 = max L2 task math start
            if (kBlockIsL2 && epilogue_thread_idx == 0) stamp_max(36);
            // Per-task probe (SM0 thread0, SM cycles): 25/27 = L1 task time / count,
            // 26/28 = L2 task time / count, 29 = gap between consecutive tasks.
            // (kL2HalfRowTasks: an L2 task is 5 stages of 128 rows, 64 per WG.)
            const bool ktask_probe_on =
                (phase_stamps != nullptr) && (sm_idx == 0) && (epilogue_thread_idx == 0);
            unsigned long long kt_task0 = 0;
            if (ktask_probe_on) {
                kt_task0 = clock64();
                if (ktask_prev_end != 0ull) atomicAdd(phase_stamps + 29, kt_task0 - ktask_prev_end);
            }
            unsigned long long tl_start = 0ull;
            if (task_log_on) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tl_start));
            run_math_task_impl(block_phase, local_expert_idx, num_k_blocks,
                               m_block_idx, n_block_idx, pool_block_idx, valid_m,
                               k_split_idx, num_k_splits, k_block_begin, first_worker_idx);
            if (epilogue_thread_idx == 0) stamp_max(kBlockIsL2 ? 5 : 4);
            if (task_log_on) {
                unsigned long long tl_end; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tl_end));
                auto* log = phase_stamps + kTaskLogBase + sm_idx * kTaskLogPerCTA;
                if (task_log_seq < kTaskLogMaxTasks) {
                    const unsigned long long meta =
                        (kBlockIsL2 ? 1ull << 31 : 0ull) | (static_cast<unsigned long long>(pool_block_idx & 0x7fu) << 24) |
                        (static_cast<unsigned long long>(n_block_idx & 0xffu) << 16) |
                        (static_cast<unsigned long long>(k_split_idx & 0xffu) << 8) |
                        static_cast<unsigned long long>(num_k_splits & 0xffu);
                    log[2 * task_log_seq] = ((tl_start - t_kernel_entry) & 0xffffffffull) | (meta << 32);
                    log[2 * task_log_seq + 1] = (tl_end - t_kernel_entry) & 0xffffffffull;
                }
                log[14] = t_kernel_entry;
                log[15] = ++ task_log_seq;
            }
            if (ktask_probe_on) {
                const unsigned long long kt_task1 = clock64();
                if (valid_m > 0) {
                    atomicAdd(phase_stamps + (kBlockIsL2 ? 26 : 25), kt_task1 - kt_task0);
                    atomicAdd(phase_stamps + (kBlockIsL2 ? 28 : 27), 1ull);
                }
                // 32/33 = K-blocks (2 per stream-K unit) run by SM0 in L1 / L2
                atomicAdd(phase_stamps + (kBlockIsL2 ? 33 : 32), static_cast<unsigned long long>(num_k_blocks));
                ktask_prev_end = kt_task1;
            }
        };
        if constexpr (kUseInterleavedScheduler) {
            for_each_published_block(run_math_task);
        } else {
            for_each_static_selected_block(run_math_task);
        }

        // Fine-grained combine: tell the dispatch warps that this CTA's math tasks
        // are done (replaces the epilogue/dispatch pairing below).
        if constexpr (kFineCombine) {
            if (epilogue_thread_idx == 0) {
                auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                DG_SPIN_WHILE(combine_mailbox_seq - ptx::ld_volatile(mailbox + 1) >= fused_layout::kSM90FineCombineRingSize, 3402);
                mailbox[4 + (combine_mailbox_seq & (fused_layout::kSM90FineCombineRingSize - 1))] =
                    fused_layout::kSM90FineCombineDoneEntry;
                ptx::st_rel_gpu(mailbox, combine_mailbox_seq + 1);
            }
        }

        // ---------------- COMBINE ----------------
        // Barrier path: NVLink barrier first, signals remote ranks that this rank's
        // GEMM outputs (NVLink scatter targets) are fully written. Fine-grained
        // combine (kFineCombine) skips it: readiness is per token (see the loop).
        if constexpr (!kFineCombine) {
            fused_comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); },
                true, true,
                /* fast epilogue: barrier #2 of the launch (#3 = after cleanup, no epilogue) */
                kNvlFastEpilogue, nvl_done_base, 2u);
            if (epilogue_thread_idx == 0) stamp_max(6);
        }

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state (kFineCombine: the mailbox
        // DONE entry carries that information instead).
        if constexpr (!kFineCombine)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumChunks =
            (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        // Token assignment: dynamic ticket (kCombineDynamic) or static (SM, warp) stride
        const auto combine_ticket_ptr = workspace.get_combine_ticket_ptr(combine_ticket_parity);
        uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
        const auto next_combine_token = [&]() {
            if constexpr (kCombineDynamic) {
                // Cheap exit for the (many) warps that free up after every token has
                // been claimed: a plain load is served by L2 without the same-address
                // atomic serialisation (~1000 warps free up within a few us at tiny M).
                uint32_t ticket = 0;
                if (lane_idx == 0) {
                    ticket = ptx::ld_volatile(combine_ticket_ptr);
                    if (ticket < num_tokens)
                        ticket = ptx::atomic_add(combine_ticket_ptr, 1u);
                }
                token_idx = __shfl_sync(0xffffffff, ticket, 0);
            } else {
                token_idx += kNumSMs * kNumEpilogueWarps;
            }
        };
        if constexpr (kCombineDynamic)
            next_combine_token();
        for (; token_idx < num_tokens; next_combine_token()) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(ld_topk_idx(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            if constexpr (kFineCombine) {
                // Wait until every (topk slot, L2 N-block) slice of this token has
                // landed, then hand the counter back (see the layout header) and
                // order the generic-proxy acquire before the TMA (async-proxy) loads.
                if (lane_idx == 0) {
                    const auto counter_ptr = workspace.get_combine_arrival_count_ptr(token_idx);
                    const int target = static_cast<int>(__popc(total_mask) * kNumRoutedL2BlockNs);
                    DG_SPIN_WHILE(ptx::ld_acq_sys(counter_ptr) != target, 3474);
                    *counter_ptr = 0;
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    stamp_max(6);
                }
                __syncwarp();
            }
            // EP16 GIN: an own row adds the peer node's partial of the same token (its
            // experts there), which arrives as the peer's combined row `token_idx`.
            const bool ep16_peer_row = kEp16Gin and token_idx >= ep16_num_own;
            if (kEp16Gin and not ep16_peer_row) {
                if (lane_idx == 0) {
                    stamp_min(51);
                    ep16_wait(1 + token_idx, 5002);
                    stamp_max(52);
                }
                __syncwarp();
            }

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                if constexpr (kEp16Gin) {
                    const uint64_t row_off = static_cast<uint64_t>(ep16_peer_row ? token_idx - ep16_num_own : token_idx) * kEp16YBytes +
                                             chunk_byte_offset;
                    if (ep16_peer_row) {
                        // The peer's row: to y_send, put back after the last chunk
                        auto* dst = reinterpret_cast<uint4*>(ep16_gin.base + ep16_y_send + row_off);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                            uint4 casted;
                            auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                            dst[j * 32 + lane_idx] = casted;
                        }
                        continue;
                    }
                    // Own row: + the peer's partial
                    const auto* src = reinterpret_cast<const uint4*>(ep16_gin.base + ep16_y_recv + row_off);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = __ldcv(src + j * 32 + lane_idx);
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
            if (ep16_peer_row) {
                __syncwarp();
                if (lane_idx == 0) {
                    __threadfence_system();
                    const uint64_t row_off = static_cast<uint64_t>(token_idx - ep16_num_own) * kEp16YBytes;
                    stamp_min(53);
                    ep16_put(ep16_y_recv + row_off, ep16_y_send + row_off, kEp16YBytes, 1 + token_idx - ep16_num_own);
                    stamp_max(54);
                }
                __syncwarp();
            }
        }
        if (epilogue_thread_idx == 0) {
            stamp_max(7);
            if (sm_idx == 0) stamp_accumulate(15);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
