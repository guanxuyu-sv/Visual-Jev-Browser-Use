#!/usr/bin/env bash
# Train the four arms, one per GPU, on the same corpus with the same budget.
# Only the modality (--no-image) and the output format differ between them.
set -euo pipefail

DATA=${VJB_ROOT}
REPO=${VJB_REPO:-$(pwd)}
PY=${VJB_PYTHON:-python3}
SUFFIX=${SUFFIX:-}          # set to archive existing runs under runs/arm<X><SUFFIX>
STEPS=${STEPS:-2000}
SEED=${SEED:-0}
TAG=${TAG:-}                # set to write runs/arm<X><TAG>, e.g. _s1 for a second seed

cd "$REPO"

if [ -n "$SUFFIX" ]; then
  for a in A B C D; do
    [ -d "$DATA/runs/arm$a" ] && mv "$DATA/runs/arm$a" "$DATA/runs/arm$a$SUFFIX"
    [ -f "$DATA/reports/offline_$a.json" ] && mv "$DATA/reports/offline_$a.json" "$DATA/reports/offline_$a$SUFFIX.json"
  done
  echo "archived previous runs with suffix $SUFFIX"
fi

COMMON="--model $DATA/models/Qwen3-VL-4B-Instruct --train $DATA/work/m2w/train.jsonl \
  --steps $STEPS --accumulate 4 --lr 1e-4 --rank 16 --alpha 32 --seed $SEED --log-every 200"

launch() {
  local arm=$1 gpu=$2; shift 2
  setsid nohup env \
    PYTHONPATH=$DATA/pylibs_peft:$DATA/pylibs_browser \
    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" vjb/train/sft.py $COMMON --arm "$arm" --out "$DATA/runs/arm$arm$TAG" "$@" \
    > "$DATA/logs/train_$arm$TAG.log" 2>&1 < /dev/null &
  echo "  arm $arm -> GPU $gpu -> runs/arm$arm$TAG"
}

launch A 0 --no-image --output-format compact
launch B 1 --output-format compact --image-dropout 0.15
launch C 2 --no-image --output-format branch
launch D 3 --output-format branch --image-dropout 0.15
echo "launched"
