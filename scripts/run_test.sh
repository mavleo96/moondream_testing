#!/usr/bin/env bash
# Run one Moondream batch test end to end: upload JSONL, poll until done, download results.
#
# Usage: scripts/run_test.sh <input.jsonl> <name>
#   e.g. scripts/run_test.sh data/batches/full_reasoning_1024px.jsonl full_reasoning_1024px
#
# Results land in results/<name>_<model>.jsonl; batch ids are logged to results/batches.log.
set -euo pipefail

INPUT="${1:?usage: run_test.sh <input.jsonl> <name>}"
NAME="${2:?usage: run_test.sh <input.jsonl> <name>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="https://api.moondream.ai/v1/batch"
MODEL="${MOONDREAM_MODEL:-moondream3.1-9B-A2B}"
CHUNK_SIZE=$((50 * 1024 * 1024))
POLL_SECONDS=60

set -a
source "$ROOT/.env"
set +a
: "${MOONDREAM_API_KEY:?MOONDREAM_API_KEY not set (check .env)}"
auth=(-H "X-Moondream-Auth: $MOONDREAM_API_KEY")

[[ -f "$INPUT" ]] || { echo "no such file: $INPUT" >&2; exit 1; }
mkdir -p "$ROOT/results"
MODEL_TAG="${MODEL//[@\/]/_}"
OUT_NAME="${NAME}_${MODEL_TAG}"

# 1. create multipart upload
init=$(curl -sf -X POST "$API?action=mpu-create" "${auth[@]}")
file_id=$(jq -r '.fileId' <<<"$init")
upload_id=$(jq -r '.uploadId' <<<"$init")
echo "[$NAME] upload created: fileId=$file_id"

# 2. upload in 50MB parts
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
split -b "$CHUNK_SIZE" "$INPUT" "$tmp/chunk_"

part_num=1
for chunk in "$tmp"/chunk_*; do
    resp=$(curl -sf -X PUT \
        "$API/$file_id?action=mpu-uploadpart&uploadId=$upload_id&partNumber=$part_num" \
        "${auth[@]}" -H "Content-Type: application/octet-stream" \
        --data-binary "@$chunk")
    etag=$(jq -r '.etag' <<<"$resp")
    jq -n --argjson n "$part_num" --arg e "$etag" '{partNumber: $n, etag: $e}' >>"$tmp/parts.jsonl"
    echo "[$NAME] uploaded part $part_num"
    part_num=$((part_num + 1))
done
parts=$(jq -s . "$tmp/parts.jsonl")

# 3. complete upload -> starts the batch
batch=$(curl -sf -X POST "$API/$file_id?action=mpu-complete&uploadId=$upload_id" \
    "${auth[@]}" -H "Content-Type: application/json" \
    -d "$(jq -n --argjson p "$parts" --arg m "$MODEL" '{parts: $p, model: $m}')")
batch_id=$(jq -r '.id' <<<"$batch")
echo "$(date -u +%FT%TZ) $NAME model=$MODEL $batch_id" >>"$ROOT/results/batches.log"
echo "[$NAME] batch started: $batch_id"

# 4. poll until done (tolerate transient poll failures)
while :; do
    if ! status=$(curl -sf "$API/$batch_id" "${auth[@]}"); then
        echo "[$NAME] $(date +%T) poll failed, retrying"
        sleep "$POLL_SECONDS"
        continue
    fi
    state=$(jq -r '.status' <<<"$status")
    echo "[$NAME] $(date +%T) $state $(jq -r '"\(.progress.completed // 0)/\(.progress.total // "?")"' <<<"$status")"
    case "$state" in
        completed) break ;;
        failed|cancelled) echo "[$NAME] batch $state" >&2; jq . <<<"$status" >&2; exit 1 ;;
    esac
    sleep "$POLL_SECONDS"
done

# 5. download result files (concatenated, in index order)
out="$ROOT/results/$OUT_NAME.jsonl"
: >"$out"
while IFS= read -r url; do
    curl -sf "$url" >"$tmp/part"
    cat "$tmp/part" >>"$out"
    if [[ -s "$tmp/part" && -n $(tail -c1 "$tmp/part") ]]; then
        echo >>"$out"  # part missing trailing newline would merge lines
    fi
done < <(jq -r '.outputs | sort_by(.index) | .[].url' <<<"$status")
echo "[$NAME] saved $(wc -l <"$out" | tr -d ' ') results to results/$OUT_NAME.jsonl"
