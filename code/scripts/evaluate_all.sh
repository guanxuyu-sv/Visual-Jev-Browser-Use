#!/usr/bin/env bash
# Everything that has to run after the four arms finish training, in order.
#
#   L1  fixed-observation diagnostic on the held-out websites, per arm
#   L2  the closed-loop pilot on the generated tasks, all arms in one process
#
# Each arm is evaluated with its own adapter and its own arm setting, so the
# observation and the output mechanism at evaluation match what it was trained
# for. Nothing here touches the Mind2Web test_* shards.
set -euo pipefail

REPO=${VJB_REPO:-$(pwd)}
DATA=${VJB_ROOT}
MODEL=$DATA/models/Qwen3-VL-4B-Instruct
PY=${VJB_PYTHON:-python3}
export PYTHONPATH=$DATA/pylibs_peft:$DATA/pylibs_browser
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BU_CDP_URL=http://127.0.0.1:9333

CHECKPOINT=${CHECKPOINT:-final}
LIMIT=${LIMIT:-300}

cd "$REPO"

echo "=== L1: held-out Mind2Web websites (limit $LIMIT) ==="
for arm in A B C D; do
  gpu=$(( $(printf '%d' "'$arm") - 65 ))
  echo "--- arm $arm on GPU $gpu ---"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/eval_offline.py \
    --model "$MODEL" \
    --adapter "$DATA/runs/arm$arm/$CHECKPOINT" \
    --data "$DATA/work/m2w/validation.jsonl" \
    --arm "$arm" --limit "$LIMIT" \
    --out "$DATA/reports/offline_$arm.json" \
    > "$DATA/logs/offline_$arm.log" 2>&1 &
done
wait
echo "L1 done"

echo "=== L2: closed-loop pilot, trained adapters ==="
# One process, one GPU: the arms share a backend and must not race for the browser.
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_local.py \
  --model "$MODEL" \
  --arms A B C D --max-steps 10 \
  --adapters "$DATA/runs/armA/$CHECKPOINT,$DATA/runs/armB/$CHECKPOINT,$DATA/runs/armC/$CHECKPOINT,$DATA/runs/armD/$CHECKPOINT" \
  --out "$DATA/reports/pilot_trained_ABCD.json" \
  2>&1 | tee "$DATA/logs/pilot_trained.log"

echo "all evaluations finished"
