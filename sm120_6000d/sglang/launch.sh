#!/bin/bash
# DeepSeek-V4.1-Flash on SGLang 0.5.22 with the SM120 MegaMoE kernel, one 8-GPU RTX 6000D host.
# Starts lmsysorg/sglang:v0.5.22-cu130 unchanged, applies sglang_v0.5.22_dsv41.patch inside the container, mounts
# kernel/ (this directory's copy of the SM120 FP4 MegaMoE with the 'w4a8_rm' format) as the kernel source and launches
# the server. See README.md.
#   MODEL_DIR=/raid/models bash launch.sh                  # MegaMoE, PCIe P2P (Broadcom-switch hosts, e.g. .20)
#   MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable bash launch.sh   # MegaMoE, peers NCCL cannot map for P2P over the NICs (CX8aaS)
#   MODEL_DIR=/raid/models MEGA=0 bash launch.sh           # baseline: stock FlashInfer MoE (startup fix only)
#   MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 bash launch.sh   # + EPLB, 8 redundant experts
#   MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 FUSED_ROUTER=1 bash launch.sh   # + the kernel routes (fastest on one host)
#   two hosts, EP16, every main-model MoE call on FlashInfer's Cake (fastest), on both hosts, node rank 0 first:
#   MODEL_DIR=/raid/models TRANSPORT=lsa_where_reachable EPLB=1 CAKE=1 FI_DIR=<FlashInfer checkout> NNODES=2 NODE_RANK=0|1 \
#     MASTER=<node rank 0's address> bash launch.sh --mem-fraction-static 0.66
# Environment: MODEL_DIR (host dir holding the model), MODEL (dir name, default DeepSeek-V4.1-Flash-1017),
# NAME (container, default dsv41_sglang), PORT (default 30041), BUILD_DIR (host cache for the kernel build).
# PACKED (MEGA=1, default 0; 1 is not measured end to end): the main model's MoE weights in the kernel's packed 'w4a8' layout (in place at load;
# FlashInfer's prefill calls read a layer rebuilt row-major into a shared ~0.9 GB scratch); 0 keeps 'w4a8_rm'.
# DENSE_OPT (default 1 with MEGA=1, 0 with MEGA=0): block-FP8 linears on FlashInfer CUTLASS (--fp8-gemm-backend
# flashinfer_cutlass: the 32x32 ue8m0 weights served as MXFP8, unchanged; SGLANG_SM120_MXFP8_B12X_MIN_N=4096 moves
# the layers with >= 4096 output rows to FlashInfer's SM120 b12x kernel) and the grouped verify indexer
# (SGLANG_SM120_INDEXER_GROUP = draft tokens + 1); DENSE_OPT=0 keeps SGLang's Triton FP8 GEMM and stock indexer.
# CAKE=1 (MEGA=1): every MoE call of the main model (prefill and decode, CUDA graphs included; up to CAKE_MAX_TOKENS,
# default 2048 per rank = SGLang's per-rank prefill chunk) on FlashInfer's Cake SM120 MegaMoE; the DSPARK draft model's
# MoE stays on this repo's kernel; FI_DIR = a FlashInfer main checkout (with its
# submodules), installed over the image's FlashInfer at start; FI_CACHE (host dir) keeps its JIT builds. Sets PACKED=0.
# Two hosts (EP16 / DP16), on both at once, node rank 0 first: NNODES=2 NODE_RANK=0|1 MASTER=<node rank 0's address>
# (reachable from every host); DIST_PORT (its rendezvous port, default 29661), IFNAME (bootstrap interface, default bond0).
# EPLB=1: SGLang's EPLB places the main model's experts from eplb_map_r8_ep8.pt (EP8, 384 experts + 8 replicas) or, with
# NNODES=2, eplb_map_r16_ep16.pt (EP16, + 16 replicas); another
# placement: EPLB_MAP (host .pt, {'physical_to_logical_map'} or {'logical_count'}) and REDUNDANT (its replica count).
# FUSED_ROUTER=1 (MEGA=1, not with CAKE=1): the SM120 kernel routes from the gate logits itself (DeepSeek-V4.1's
# router: sqrtsoftplus, correction bias, bias_vl for image tokens, top-k, renormalize, routed scaling, EPLB replica
# pick) instead of SGLang's moe_fused_gate and EPLB remap (SGLANG_SM120_FUSED_ROUTER).
# Extra arguments are appended to the server command line.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the host directory that holds the model}
MODEL=${MODEL:-DeepSeek-V4.1-Flash-1017}
NAME=${NAME:-dsv41_sglang}
PORT=${PORT:-30041}
MEGA=${MEGA:-1}
TRANSPORT=${TRANSPORT:-p2p}
BUILD_DIR=${BUILD_DIR:-$HOME/.deep_gemm/sm120_mega_moe_sglang}
NNODES=${NNODES:-1}
WORLD=$((8 * NNODES))  # TP = attention DP = EP = every GPU; SGLang splits the chunked prefill over the DP ranks
mkdir -p "$BUILD_DIR"

ENVS=(-e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)
DEV=()
EPLB_MAP=${EPLB_MAP:-$([ "${EPLB:-0}" = 1 ] && echo "$HERE/eplb_map_r${WORLD}_ep${WORLD}.pt")}
EPLB_VOL=() EPLB_ARGS=""
if [ -n "$EPLB_MAP" ]; then
  EPLB_VOL=(-v "$EPLB_MAP":/opt/eplb_map.pt:ro)
  EPLB_ARGS="--init-expert-location /opt/eplb_map.pt --ep-num-redundant-experts ${REDUNDANT:-$WORLD}"
fi
MULTI=""
if [ "$NNODES" -gt 1 ]; then
  ENVS+=(-e NCCL_SOCKET_IFNAME=${IFNAME:-bond0} -e GLOO_SOCKET_IFNAME=${IFNAME:-bond0})
  MULTI="--nnodes $NNODES --node-rank ${NODE_RANK:?set NODE_RANK} --dist-init-addr ${MASTER:?set MASTER}:${DIST_PORT:-29661}"
  # baseline too: a TP16 shard of the shared expert (2304 / 16 = 144 columns) is not a multiple of the FP8 block (32)
  ENVS+=(-e SGLANG_SHARED_EXPERT_TP1=1)
fi
CAKE=${CAKE:-0}
[ "$CAKE" = 1 ] && PACKED=0
FI_VOL=() FI_SETUP=""
if [ "$CAKE" = 1 ]; then
  FI_DIR=${FI_DIR:?set FI_DIR to a FlashInfer main checkout for CAKE=1}
  FI_CACHE=${FI_CACHE:-$HOME/.cache/flashinfer_cake}
  mkdir -p "$FI_CACHE"
  FI_VOL=(-v "$FI_DIR":/opt/flashinfer -v "$FI_CACHE":/root/.cache/flashinfer)
  # the image's FlashInfer AOT caches belong to its release; FlashInfer main JIT-builds what it needs
  FI_SETUP="pip uninstall -y \$(pip list 2>/dev/null | awk '/^flashinfer-(jit-cache|cubin)/{print \$1}') >/dev/null &&
    rm -rf \$(python3 -c 'import site; print(site.getsitepackages()[0])')/flashinfer_jit_cache &&
    pip install -q --no-build-isolation --no-deps -e /opt/flashinfer &&"
  ENVS+=(-e SGLANG_SM120_CAKE_ALL=1 -e SGLANG_SM120_CAKE_PREFILL=1 -e SGLANG_SM120_CAKE_MAX_TOKENS=${CAKE_MAX_TOKENS:-2048})
fi
if [ "$MEGA" = 1 ]; then
  ENVS+=(-e SGLANG_SM120_MEGA_MOE=1 -e SGLANG_SHARED_EXPERT_TP1=1
         -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=128
         -e SGLANG_SM120_MEGA_MOE_DIR=/opt/sm120_megamoe -e DG_SM120_MEGA_MOE_BUILD_DIR=/b
         -e SGLANG_SM120_MEGA_MOE_TRANSPORT=$TRANSPORT -e SGLANG_SM120_MEGA_MOE_PACKED=${PACKED:-0}
         -e SGLANG_SM120_FUSED_ROUTER=${FUSED_ROUTER:-0})
  if [ "$TRANSPORT" != p2p ]; then
    # NCCL GIN needs the NCCL Device API, which needs cuMem (SGLang otherwise sets NCCL_CUMEM_ENABLE=0)
    ENVS+=(-e NCCL_CUMEM_ENABLE=1 -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7})
    DEV=(--device /dev/infiniband --cap-add IPC_LOCK)
  fi
fi
DSPARK_BLOCK=5
DENSE_ARGS=""
if [ "${DENSE_OPT:-$MEGA}" = 1 ]; then
  ENVS+=(-e SGLANG_SM120_INDEXER_GROUP=$((DSPARK_BLOCK + 1)) -e SGLANG_SM120_MXFP8_B12X_MIN_N=4096)
  DENSE_ARGS="--fp8-gemm-backend flashinfer_cutlass"
fi

docker run -d --name "$NAME" --gpus all --ipc host --network host --shm-size 64g -u root --ulimit memlock=-1 \
  "${DEV[@]}" "${ENVS[@]}" "${EPLB_VOL[@]}" "${FI_VOL[@]}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL_DIR":/models:ro -v "$HERE/kernel":/opt/sm120_megamoe:ro -v "$BUILD_DIR":/b \
  -v "$HERE/sglang_v0.5.22_dsv41.patch":/opt/sglang_v0.5.22_dsv41.patch:ro \
  --entrypoint bash lmsysorg/sglang:v0.5.22-cu130 -c "
    $FI_SETUP cd /sgl-workspace/sglang && git apply /opt/sglang_v0.5.22_dsv41.patch &&
    exec python3 -m sglang.launch_server --model-path /models/$MODEL --served-model-name dsv41 --trust-remote-code \
      --tp $WORLD --dp $WORLD --enable-dp-attention --ep-size $WORLD $MULTI --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
      --page-size 256 --mem-fraction-static 0.8 --watchdog-timeout 3600 \
      --chunked-prefill-size $((2048 * WORLD)) --cuda-graph-max-bs-decode 16 --max-running-requests $((16 * WORLD)) --context-length 180000 \
      --speculative-algorithm DSPARK --speculative-dspark-block-size $DSPARK_BLOCK --enable-dp-lm-head --swa-full-tokens-ratio 0.02 \
      --enable-metrics --enable-metrics-for-all-schedulers --incremental-streaming-output --stream-interval 1 \
      $EPLB_ARGS $DENSE_ARGS --host 0.0.0.0 --port $PORT $*"
echo "started $NAME (MEGA=$MEGA NNODES=$NNODES TRANSPORT=$TRANSPORT EPLB_MAP=${EPLB_MAP:-none} DENSE_OPT=${DENSE_OPT:-$MEGA} CAKE=$CAKE); logs: docker logs -f $NAME"
