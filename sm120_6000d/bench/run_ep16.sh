#!/bin/bash
# One host's half of a two-host run of tests/test_mega_moe_sm120_fp4.py (EP16 = 2 hosts x 8 GPUs): start it on both
# hosts at the same time, node rank 0 on the master. Same-host pairs go over P2P, the others over the NICs (NCCL GIN).
#   on host A:  MASTER=<A's address> run_ep16.sh 0 verify mxfp4 small socket
#   on host B:  MASTER=<A's address> run_ep16.sh 1 verify mxfp4 small socket
# Environment: MASTER (address of node rank 0, reachable from both), ROOT (this repository on the host, default: the
# checkout of this script), IMAGE, PORT, IFNAME (interface of NCCL's bootstrap sockets), HCA (NCCL_IB_HCA, e.g.
# "=mlx5_0,mlx5_2" on a host where only some NICs reach the other host; default: every NIC), NPROC (GPUs per host).
set -e
NODE_RANK=$1; shift
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
docker run --rm -u root --gpus all --ipc host --network host --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband \
  -e TILE=${TILE:-0} -e NCCL_SOCKET_IFNAME=${IFNAME:-bond0} ${HCA:+-e NCCL_IB_HCA=$HCA} \
  -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e DG_SM120_MEGA_MOE_BUILD_DIR=/w/build \
  -v "$ROOT":/w -w /w --entrypoint bash ${IMAGE:-toklin/vllm-sm120:e10ecb6bd-p3} -c \
  "timeout ${TIMEOUT:-1800} torchrun --nnodes 2 --node-rank $NODE_RANK --nproc-per-node ${NPROC:-8} \
   --master-addr ${MASTER:?MASTER} --master-port ${PORT:-29651} tests/test_mega_moe_sm120_fp4.py $*"
