#!/bin/bash
# ci_serve.sh: CI image (vLLM main + Dynamo, toklin/vllm-sm120:e10ecb6bd-v41) as an aggregated DP4/EP4 server,
# CI decode-worker args, only --moe-backend changes. Patched vLLM files (SM120 MegaMoE backend) and FlashInfer's
# 6000D green-context fix are bind-mounted on both sides; MegaMoE additionally gets NVSHMEM 3.7.0 + nvshmem4py
# and the recipe's NVSHMEM/CuTeDSL env. Env: GPUS PORT NAME MOE CGS(optional) MAXSEQ(optional)
R=/raid/kimiz_mm; C=$R/ci_mm; SP=/usr/local/lib/python3.12/dist-packages
GPUS=${GPUS:-0,1,2,3}; PORT=${PORT:-18910}; NAME=${NAME:-kimiz_ci_mm}; MOE=${MOE:-flashinfer_moe_ep_sm120_mxfp4}
MNT=""
PATCHED=${PATCHED:-patched}
for f in $(cd $C/$PATCHED && find vllm -name "*.py"); do MNT="$MNT -v $C/$PATCHED/$f:$SP/$f:ro"; done
GC=moe_ep/kernel_src/sm120/split_cutedsl_megakernel/src/moe_sm120_mxfp4mxfp8_split/runtime/green_context.py
MNT="$MNT -v $R/fi_4632/flashinfer/$GC:$SP/flashinfer/$GC:ro"
EXTRA=""
PRE="true"
if [ "$MOE" = flashinfer_moe_ep_sm120_mxfp4 ]; then
  EXTRA="-e NVSHMEM_HEAP_KIND=VIDMEM -e NVSHMEM_REMOTE_TRANSPORT=none -e NVSHMEM_SYMMETRIC_SIZE=8G -e NCCL_P2P_DISABLE=1 -e CUTE_DSL_ARCH=sm_120a"
  PRE="pip install -q --no-deps $R/wheels/nvidia_nvshmem_cu13-3.7.0-*.whl $R/wheels/nvshmem4py_cu13-0.3.1-cp312-*.whl > /tmp/pip.log 2>&1"
fi
if [ "$MOE" = kz_sm120_megamoe ] || [ "$MOE" = kz_sm120_tp ]; then
  EXTRA="-e KZ_SM120_MM_SO=${KZ_SO:-/raid/kimiz_mm/sm120mm/build/sm120mm.so}"
fi
MB=""
[ -n "${MBT:-}" ] && MB="--max-num-batched-tokens $MBT"
PARARGS="-tp 1 --data-parallel-size 4 --enable-expert-parallel"
[ "${PAR:-}" = tp8 ] && PARARGS="-tp 8"
CG=""
[ -n "${CGS:-}" ] && CG="--cudagraph-capture-sizes $CGS"
MS=""
[ -n "${MAXSEQ:-}" ] && MS="--max-num-seqs $MAXSEQ"
mkdir -p $C/cache
docker container rm --force "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" -u root --gpus "\"device=$GPUS\"" --network host --ipc host --init --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 $MNT $EXTRA -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 \
  -v $R:$R -v $C/cache:/root/.cache -v /raid/toklin-sm120/model:/model:ro \
  --entrypoint bash toklin/vllm-sm120:e10ecb6bd-v41 -c "set -e; $PRE; python3 -c 'import nvshmem, nvshmem.core; print(\"nvshmem\", nvshmem.__file__)' || true
    exec vllm serve /model --served-model-name dsv4flash --host 0.0.0.0 --port $PORT --trust-remote-code \
      --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 --max-model-len ${MML:-32768} \
      --gpu-memory-utilization 0.85 --enable-prefix-caching --moe-backend $MOE \
      $PARARGS $MB $CG $MS"
echo started $NAME
