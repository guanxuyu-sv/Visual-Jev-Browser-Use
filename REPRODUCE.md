# Reproducing the experiments

The experiment code lives in `code/`. The scored outputs in `data/` regenerate
every table in the README without a GPU. Producing new predictions needs the
backbone weights, the corpora, a GPU, and a headless Chrome.

## What is not here

Neither corpus is redistributed. Multimodal-Mind2Web is obtained from its own
distribution under its own terms; MiniWoB++ is vendored at a pinned commit by the
build step rather than copied into this repository.

The trained adapters are not in git either, but they are published:
[`guanxuyu/visual-jev-browser-4b`](https://huggingface.co/guanxuyu/visual-jev-browser-4b)
holds all four arms and the terminal-trained stage, so the corpus build and the
training runs can be skipped if you only want to reproduce the evaluation. The
backbone comes from `Qwen/Qwen3-VL-4B-Instruct`.

The executor and the benchmark are fetched at pinned commits by step 1 rather
than vendored, because both carry their own licences.

## 1. Fetch the executor and the benchmark

```bash
bash code/scripts/fetch_third_party.sh
```

This clones [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
and [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus) at the
commits in `PINNED_VERSIONS.md` and lays the task pages out where the runners
expect them. To keep them elsewhere, pass a destination and export
`VJB_THIRD_PARTY` to the same path.

Nothing in `code/` reaches for them until a closed-loop script runs; the offline
diagnostic and the table regeneration do not need them.

## Environment

Every run in `data/` was produced on one machine: Python 3.12, a single NVIDIA
RTX 5090 (32 GB) per job, CUDA 13.0, torch 2.13.0, transformers 5.17.0,
peft 0.21.0, pillow 12.3.0.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r code/requirements.txt
```

One variable decides where everything lives; nothing else is hard-coded:

```bash
export VJB_ROOT=/somewhere/with/space     # models, data, work, runs, reports
export VJB_MODEL=Qwen/Qwen3-VL-4B-Instruct
export VJB_MODEL_8B=Qwen/Qwen3-VL-8B-Instruct    # only for the scale check
```

`$VJB_ROOT` is laid out as `models/`, `data/` (corpora as downloaded), `work/`
(derived records), `runs/` (adapters), `reports/` (scored results), `logs/`.

## The browser

The executor drives a real Chrome over CDP. Headless works, with flags that are
part of the pinned environment rather than a convenience:

```bash
google-chrome --headless=new --remote-debugging-port=9333 \
  --user-data-dir=$VJB_ROOT/chrome-profile \
  --no-sandbox --disable-dev-shm-usage --window-size=1120,780 \
  --disable-background-timer-throttling \
  --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding about:blank
export BU_CDP_URL=http://127.0.0.1:9333
```

Two of these matter for correctness, not speed. Without the throttling flags a
background tab's timers stall and the observation never settles. And the agent's
tab presents no compositor surface, so `Page.captureScreenshot` with its default
`fromSurface=true` waits for a frame that never arrives — it timed out on 5 of 5
attempts in isolation, while `fromSurface=False` returned the same image in 28 ms.
`configure_cdp` in `code/vjb/browser/loop.py` applies that and retries; under load
40–64% of captures still need a retry.

## 2. Acceptance checks

Run these before anything expensive. They check the properties the method
depends on, and each one exists because its absence cost hours.

```bash
python code/scripts/check_backend.py --model $VJB_MODEL --out $VJB_ROOT/reports/backend_checks.json
python code/scripts/check_alignment.py --model $VJB_MODEL
```

`check_backend.py` verifies that candidate symbols are one token each in the
exact readout context, that image tokens actually reach the model and grow with
resolution, that KV reuse and a full recompute produce identical token ids, that
branch answers stay in their candidate set and are invariant to branch order, and
that running the branches leaves the shared prefix usable for generation.

`check_alignment.py` compares the training loss against the standard formulation
(full-sequence logits, labels shifted, everything but the answer masked). It must
report a difference of 0 before any training run.

## 3. Build the corpus

```bash
python code/scripts/build_mind2web.py --data $VJB_ROOT/data/mm-mind2web/data \
    --out $VJB_ROOT/work/m2w --val-websites 8
```

7362 of 7775 steps convert; 413 (5.3%) are dropped for a missing gold candidate,
which is a candidate-recall number to report rather than a step to skip. Websites
are held out whole, eight of them, so no template straddles the split.

Each step is then reduced to one viewport at training time. Mind2Web ships
whole-page screenshots — 1280×4977 and 1297×8965 are ordinary, worth 2600 to
11480 visual tokens — while the live executor produces a 1120×780 viewport at 1:1.
Scaling a whole page into a token budget instead gives a 39%-scale rendering in
which the small text the screenshot exists to convey is gone. 3327 of 6454 steps
survive the reduction; the rest have a gold element taller than a viewport or
fewer than four candidates in view.

## 4. Train the four arms

```bash
VJB_ROOT=$VJB_ROOT bash code/scripts/train_all.sh
```

All four train on the same steps with the same candidate lists and the same
budget. Only `--no-image` and `--output-format` differ. 2000 steps, accumulate 4,
lr 1e-4, LoRA r=16 alpha=32 on the language tower with the vision tower frozen:
33.0M trainable of 4.47B (0.74%), 9.5 GiB peak, roughly 1–2 hours per arm.

## 5. Evaluate

Adapters can come straight from the Hub instead of a local training run —
`--adapter guanxuyu/visual-jev-browser-4b` for arm D, and the subfolder paths in
the model card for the others.

```bash
# L1: fixed observations on the held-out websites
python code/scripts/eval_offline.py --model $VJB_MODEL \
    --adapter $VJB_ROOT/runs/armD/final --arm D \
    --data $VJB_ROOT/work/m2w/validation.jsonl --limit 400 \
    --out $VJB_ROOT/reports/offline_D.json

# L2: closed loop on MiniWoB++, scored by the environment
python code/scripts/run_miniwob.py --model $VJB_MODEL \
    --arms A B C D --adapters "$VJB_ROOT/runs/armA/final,...,$VJB_ROOT/runs/armD/final" \
    --seeds 0 1 2 3 --out $VJB_ROOT/reports/miniwob.json
```

Success on MiniWoB is `WOB_DONE_GLOBAL && WOB_RAW_REWARD_GLOBAL > 0`, read from
the page. The model's DONE is recorded but never decides the outcome. Two
deviations from stock MiniWoB are disclosed rather than hidden: the episode clock
is raised from its 10 s default, which assumes a policy acting in milliseconds,
and the raw reward is used rather than the displayed one, which is scaled by how
much of the clock is left and so mixes success with speed.

56 of 130 templates are actionable by this executor (43.1%). The subset was frozen
before any model was evaluated and each exclusion's reason is recorded in
`code/vjb/data/miniwob_subset.json`: 25 expose no actionable DOM at all, 22
expose a single element, which is not a choice.

## 6. Regenerate the tables

No GPU, no browser, no model — this works on a fresh clone with nothing else
installed.

```bash
python code/vjb/eval/report.py data/miniwob_done_CD.json
```

## Reading `data/`

| file | what it is |
| --- | --- |
| `offline_{A..D}.json` | L1, held-out websites, the arms the README reports |
| `offline_*_done.json` | L1 after terminal-action training |
| `offline_*_leaky.json` | **void** — the legal-operation list contained only the gold operation |
| `miniwob_{AB,CD}.json` | L2, 224 tasks per arm; success valid, **latency void** (see below) |
| `miniwob_done_*.json` | L2 after terminal-action training, 216 tasks on held-out families |
| `pilot_*.json` | the generated local task set, before and after training |
| `bench_stages.json` | latency decomposition by prefix length and branch count |
| `backend_checks.json` | the five acceptance checks |

The `_leaky` files are kept so the reason they are void can be checked rather
than taken on trust. The first MiniWoB round's latency is void for a different
reason: median step count equalled the budget in every arm, because the model had
never been taught to stop, so those times measure the budget and not the task.

## Defects found in this pipeline

Five, each costing hours, each now guarded:

| defect | how it showed | guard |
| --- | --- | --- |
| legal-operation list contained only the gold operation | TYPE_TEXT joint accuracy of exactly 100%, predicted count equal to gold | all operations offered in both training and evaluation |
| training screenshots plain, inference screenshots numbered | 0.99 confidence on the wrong element | the same overlay is drawn in both |
| no terminator in the training targets | a field filled with "Zurich, Zurich, Zurich, …" | terminator appended to targets |
| terminator in targets but not inputs | "Bloggs" generated as "gs" | `check_alignment.py` |
| stale re-observations not counted as no progress | 15 wasted model calls per finished task | a stale streak ends the run |
