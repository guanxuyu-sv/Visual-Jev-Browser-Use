# Visual Jev — Browser Use

**Where vision helps a browser agent, and where it breaks.** One Qwen3-VL reads a
screenshot and a DOM element table once, answers each decision from that shared
prefill, and — only when the chosen operation types text — continues the same KV
cache to write the field value. This repository holds the code, the scored
outputs, and the measurements behind each claim.

This extends [Visual Jev](https://github.com/guanxuyu-sv/Visual-Jev)
([paper](https://arxiv.org/abs/2609.25845)) from answering many questions about
one image to acting on a web page, where the questions are *which operation* and
*which element*, and the answers have to survive an executor.

Every number below is measured, with its interval attached. `docs/status.md` is
the standing account of what each hypothesis came to, including the one that did
not work out and the five defects found along the way.

## What the experiments say

### Vision decides *which element*, not *what action*

| arm | observation | output | joint | CLICK | operation |
| --- | --- | --- | --- | --- | --- |
| A | DOM | compact autoregressive line | 54.7% | 43.4% | 94.0% |
| B | screenshot + DOM | compact autoregressive line | 88.9% | 88.4% | 95.7% |
| C | DOM | branch readout + conditional text | 56.8% | 43.9% | 95.7% |
| D | screenshot + DOM | branch readout + conditional text | **91.0%** | **90.2%** | 96.6% |

Held-out Mind2Web websites, n=234, never seen in training.

Operation accuracy is 94–97% in **every** arm. The screenshot does not move it
and has no room to. The entire gain is target selection — CLICK joint accuracy
goes from 43% to 88–90%. **Deciding what to do needs only the DOM text. Deciding
which element to do it to needs the picture.**

| comparison | difference | 95% CI | wins/losses |
| --- | --- | --- | --- |
| add screenshot, autoregressive output | **+34.2%** | [+23.1, +47.7] | 85 / 5 |
| add screenshot, branch output | **+34.2%** | [+21.8, +47.6] | 81 / 1 |
| change output mechanism, DOM only | +2.1% | [−1.2, +7.2] | 19 / 14 |
| change output mechanism, with screenshot | +2.1% | [−0.8, +4.8] | 7 / 2 |

Cluster bootstrap over held-out websites.

### The vision effect is causal, not correlational

Paired pages: byte-identical DOM with the SVGs stripped, differing only in which
card carries which picture, so the correct target moves for a reason nothing but
the image can express.

| arm | pairs fully correct | exactly one of two | neither |
| --- | --- | --- | --- |
| DOM only (A, C) | **0 / 5** | 2–3 | 2–3 |
| screenshot (B, D) | **5 / 5** | 0 | 0 |

"Exactly one of two" is what a shortcut produces, and it is what the DOM-only
arms produce — picking the same element on both members of every pair.

### Vision is worth more on real pages than on benchmark pages

| benchmark | pages | screenshot gain |
| --- | --- | --- |
| Mind2Web held-out | real websites | **+34.2%** |
| MiniWoB++ | simplified, generated | **+5.6 to +7.1%** |

Same models, same training, same code. **Evaluating a visual web agent only on
MiniWoB-style pages understates what vision is worth**, by roughly five-fold here.

### The branch readout cuts decision latency without costing accuracy

| arm | decide | generate | success under a 3 s budget |
| --- | --- | --- | --- |
| A | 351 ms | — | 2.8% |
| B | 447 ms | — | 6.9% |
| C | **145 ms** | 197 ms | 10.6% |
| D | **243 ms** | 188 ms | **12.0%** |

Reading a restricted logit is 1.8–2.4× faster than generating an action line, and
accuracy does not fall (C ≥ A, D ≥ B). **End to end it does not show**: browser
observation is 1.2 s of a 1.5 s step, so a 200 ms saving is 14% of the step. We
report the model-side result and decline the end-to-end one.

### The negative result

The unified decision-and-generation mechanism — the thing this project set out to
show — **does not improve accuracy**. Four of five comparisons span zero. Why, as
far as the data says:

* A fair autoregressive baseline had **zero parse failures and zero out-of-set
  targets in 234 steps** with a median of 20 candidates. The errors a constrained
  readout structurally prevents did not occur, so preventing them bought nothing.
* The bottleneck is perception, not expression. Both mechanisms read the same
  prefix and see the same candidates; how the answer is spelled adds no information.
* Fan-out needs questions to share an observation, and a browser step has 2–3
  (52% have ≤2). The prior work's 8.9× came from N=32. Measured here: N=1 gives
  1.00×, N=2 gives 1.94×, N=4 gives 2.41×.

Its measurable benefit is field text: **+9.1%**, 95% CI [+2.1, +15.7].

### Two failure modes worth knowing about

**An out-of-distribution visual overlay overrides explicit text.** Numbered
badges of the kind now standard in Set-of-Mark prompting, drawn in a style the
model was not trained on, moved it from 0.93 on the right element to **0.99 on
the wrong one** — while the element table it was also given said `checked=true`
for the element it chose. The same checkpoint on the same page, given a plain
screenshot, answers correctly.

**Offline trajectory corpora contain no terminal action.** All 6454 Mind2Web
steps are CLICK, TYPE_TEXT or SELECT, because a recorded trajectory stores one
action per step and stops. The model saw DONE among its candidates and was never
once shown when to pick it, so **105 of 106 successful runs kept acting on a
finished page until the step budget ran out**. This is a property of the corpus
format, not of Mind2Web. Collecting real terminal observations from a resettable
environment fixes the symptom; doing it only at the moment of completion
introduces a false-DONE problem of its own, which `docs/status.md` documents.

## Layout

```
code/vjb/model/      prefill / decide / generate / release over one Qwen3-VL;
                     the four arms; verified single-token candidate symbols
code/vjb/browser/    element table with geometry, numbered screenshot,
                     agent loop with per-stage timing
code/vjb/data/       Mind2Web conversion, MiniWoB++ adapter and frozen subset,
                     generated paired-diagnostic pages
code/vjb/train/      viewport reduction, joint SFT
code/vjb/eval/       arm scoring, paired diagnostics, cluster bootstrap
code/scripts/        acceptance checks, benchmarks, training and evaluation
data/                every scored output behind the tables above
docs/                status report and findings
```

`REPRODUCE.md` has the environment and the steps. `data/` is enough to regenerate
the tables without a GPU.

## What this is not

No manuscript. One training seed where the plan calls for three, one backbone,
and no VisualWebArena — its site images are hosted where this network cannot
reach them, and the measurements behind that conclusion are in `docs/status.md`.
The paired diagnostic, which carries the strongest claim, has five pairs.

Upstream executor: [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
at `1231850a`. Benchmark: [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus)
at `33c3b4d`. Neither is redistributed here beyond the commit pins.
