#!/bin/bash
# DeepSeek-V4.1-Flash on SGLang 0.5.22 with the SM120 MegaMoE kernel, one 8-GPU RTX 6000D host.
# Starts lmsysorg/sglang:v0.5.22-cu130 unchanged, applies sglang_v0.5.22_dsv41.patch inside the container, mounts
# kernel/ (this directory's copy of the SM120 FP4 MegaMoE with the 'w4a8_rm' format) as the kernel source and launches
# the server. See README.md.
#   MODEL_DIR=/raid/models bash launch.sh                  # MegaMoE, PCIe P2P (Broadcom-switch hosts, e.g. .20)
#   MODEL_DIR=/raid/models TRANSPORT=auto bash launch.sh   # MegaMoE, cross-root-port peers over the NICs (CX8aaS)
#   MODEL_DIR=/raid/models MEGA=0 bash launch.sh           # baseline: stock FlashInfer MoE (startup fix only)
# Environment: MODEL_DIR (host dir holding the model), MODEL (dir name, default DeepSeek-V4.1-Flash-1017),
# NAME (container, default dsv41_sglang), PORT (default 30041), BUILD_DIR (host cache for the kernel build).
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
mkdir -p "$BUILD_DIR"

ENVS=(-e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)
DEV=()
if [ "$MEGA" = 1 ]; then
  ENVS+=(-e SGLANG_SM120_MEGA_MOE=1 -e SGLANG_SHARED_EXPERT_TP1=1
         -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=128
         -e SGLANG_SM120_MEGA_MOE_DIR=/opt/sm120_megamoe -e DG_SM120_MEGA_MOE_BUILD_DIR=/b
         -e SGLANG_SM120_MEGA_MOE_TRANSPORT=$TRANSPORT)
  if [ "$TRANSPORT" != p2p ]; then
    # NCCL GIN needs the NCCL Device API, which needs cuMem (SGLang otherwise sets NCCL_CUMEM_ENABLE=0)
    ENVS+=(-e NCCL_CUMEM_ENABLE=1 -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7})
    DEV=(--device /dev/infiniband --cap-add IPC_LOCK)
  fi
fi

docker run -d --name "$NAME" --gpus all --ipc host --network host --shm-size 64g -u root --ulimit memlock=-1 \
  "${DEV[@]}" "${ENVS[@]}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL_DIR":/models:ro -v "$HERE/kernel":/opt/sm120_megamoe:ro -v "$BUILD_DIR":/b \
  -v "$HERE/sglang_v0.5.22_dsv41.patch":/opt/sglang_v0.5.22_dsv41.patch:ro \
  --entrypoint bash lmsysorg/sglang:v0.5.22-cu130 -c "
    cd /sgl-workspace/sglang && git apply /opt/sglang_v0.5.22_dsv41.patch &&
    exec python3 -m sglang.launch_server --model-path /models/$MODEL --served-model-name dsv41 --trust-remote-code \
      --tp 8 --dp 8 --enable-dp-attention --ep-size 8 --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
      --page-size 256 --mem-fraction-static 0.8 --watchdog-timeout 3600 \
      --chunked-prefill-size 16384 --cuda-graph-max-bs-decode 16 --max-running-requests 128 --context-length 180000 \
      --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --enable-dp-lm-head --swa-full-tokens-ratio 0.02 \
      --enable-metrics --enable-metrics-for-all-schedulers --incremental-streaming-output --stream-interval 1 \
      --host 0.0.0.0 --port $PORT $*"
echo "started $NAME (MEGA=$MEGA TRANSPORT=$TRANSPORT); logs: docker logs -f $NAME"
