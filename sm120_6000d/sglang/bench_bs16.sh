#!/bin/bash
# Decode SSE bench at BS16 per rank against a server started by launch.sh (same host).
#   CLIENT_DIR=/path/nvidia_decode_client_20261111 DATA=/path/public_pg19_all.jsonl.gz bash bench_bs16.sh
# CLIENT_DIR: the extracted decode client package (run.sh, decode_sse_bench.py, tokenizer/, wheelhouse/); DATA: the
# pg19 JSONL. Neither is in this repo. Environment: PORT (default 30041), BS (per rank, default 16), NAME (client
# container, default dsv41_bench). The log and results land in CLIENT_DIR (bench_bs$BS.log, results/).
set -e
CLIENT_DIR=$(cd "${CLIENT_DIR:?set CLIENT_DIR to the decode client package}" && pwd)
DATA=$(cd "$(dirname "${DATA:?set DATA to public_pg19_all.jsonl.gz}")" && pwd)/$(basename "$DATA")
PORT=${PORT:-30041}
BS=${BS:-16}
NAME=${NAME:-dsv41_bench}

curl -sf -m5 localhost:$PORT/health > /dev/null || { echo "no server on port $PORT"; exit 1; }
# the client needs every rank's sglang:realtime_tokens_total{mode="decode"}; a rank exports it after its first decode
for r in 0 1 2 3 4 5 6 7; do
  curl -sf -m120 localhost:$PORT/generate -H "Content-Type: application/json" \
    -d "{\"text\":\"hi\",\"sampling_params\":{\"max_new_tokens\":8,\"temperature\":0},\"routed_dp_rank\":$r}" > /dev/null
done
ranks=$(curl -s localhost:$PORT/metrics | grep '^sglang:realtime_tokens_total' | grep -c 'mode="decode"')
echo "ranks exporting decode tokens: $ranks"

# DP8 on one host, mode 4, mean input 100K (+-1%), 1100 output tokens, 1 s steady window
cat > "$CLIENT_DIR/config_dsv41_dp8.json" <<EOF
{"base_url": "http://127.0.0.1:$PORT", "model": "dsv41", "dp_size": 8, "metrics_urls": [], "metric_labels": [],
 "tokenizer": "tokenizer/", "data_file": "/data/$(basename "$DATA")", "batch_sizes": [$BS],
 "input_length_k": 100, "output_tokens": 1100, "measure_seconds": 1, "catchup_window_seconds": 1,
 "poll_interval": 0.2}
EOF
docker rm -f "$NAME" > /dev/null 2>&1 || true
# CPU-only client container; the SGLang image only supplies a Python the package's offline wheels accept
docker run --rm --name "$NAME" -u root --network host \
  -v "$CLIENT_DIR":/client -v "$(dirname "$DATA")":/data:ro --entrypoint bash lmsysorg/sglang:v0.5.22-cu130 \
  -c "cd /client && bash run.sh --config config_dsv41_dp8.json $BS" 2>&1 | tee "$CLIENT_DIR/bench_bs$BS.log" \
  | grep -E "BS/rank=|ERROR|SSE gap|Output TPS|Speculative lifetime|Formal input cache hit"
