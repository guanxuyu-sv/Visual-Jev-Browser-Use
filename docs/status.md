# Where the work stands

Date: 2026-09-26. Backbone `Qwen3-VL-4B-Instruct`, 33.0M LoRA parameters (0.74%),
vision tower frozen. One training seed. Everything below is measured on this box;
every number traces to a file in `reports/`.

## The four arms

| arm | observation | output |
| --- | --- | --- |
| A | DOM | one compact action line, autoregressive |
| B | screenshot + DOM | one compact action line, autoregressive |
| C | DOM | parallel decision branches + conditional text on reused KV |
| D | screenshot + DOM | parallel decision branches + conditional text on reused KV |

All four train on the same 3327 steps with the same candidate lists and the same
2000-step budget. Only the modality and the output mechanism differ.

## What the hypotheses came to

| | hypothesis | verdict | evidence |
| --- | --- | --- | --- |
| H1 | the screenshot helps visually demanding tasks | **supported** | paired diagnostic: DOM-only arms get 0/5 pairs, screenshot arms 5/5 |
| H2 | fusion helps on the ordinary task distribution | **supported** | +34.2% joint on held-out real web; +5.6 to +7.1% task success on MiniWoB, intervals exclude zero |
| H3 | branch decisions + conditional generation beat a compact AR line | **not supported** | four of five comparisons span zero; the one that does not is explained below |
| H4 | KV reuse cuts input-action latency | **micro only** | 2.0-3.0x in isolation, no closed-loop effect measured |
| H5 | joint training preserves text quality | **partly supported** | D-B text +9.1%, 95% CI [+2.1, +15.7] |
| H6 | speculative decoding | not attempted | — |

## L1: fixed observations, held-out Mind2Web websites (n=234)

Eight websites the training never saw.

| arm | operation | target | joint | CLICK | text exact | text normalised |
| --- | --- | --- | --- | --- | --- | --- |
| A | 94.0% | 58.5% | 54.7% | 43.4% | 59.1% | 70.5% |
| B | 95.7% | 92.3% | 88.9% | 88.4% | 61.4% | 77.3% |
| C | 95.7% | 59.4% | 56.8% | 43.9% | 68.2% | 81.8% |
| D | 96.6% | 94.4% | 91.0% | 90.2% | 72.7% | 86.4% |

Paired differences, cluster bootstrap over held-out websites:

| comparison | difference | 95% CI | wins/losses |
| --- | --- | --- | --- |
| B − A (add screenshot, AR) | **+34.2%** | [+23.1, +47.7] | 85 / 5 |
| D − C (add screenshot, branch) | **+34.2%** | [+21.8, +47.6] | 81 / 1 |
| C − A (change output, DOM) | +2.1% | [−1.2, +7.2] | 19 / 14 |
| D − B (change output, screenshot) | +2.1% | [−0.8, +4.8] | 7 / 2 |

**The single most informative decomposition in the project.** Operation accuracy
is 94–97% in every arm; the screenshot does not move it and cannot, there being
no room. The entire gain is in target selection: CLICK joint accuracy goes from
43% to 88–90%. Deciding *what* to do needs only the DOM text. Deciding *which
element* to do it to needs the picture.

## L2: MiniWoB++ closed loop, success decided by the environment

56 of 130 templates are actionable by this executor (43.1%); the subset was
frozen before any model was evaluated, and each exclusion's reason is recorded —
25 expose no actionable DOM at all, 22 expose a single element, which is not a
choice. Success is `WOB_DONE_GLOBAL && WOB_RAW_REWARD_GLOBAL > 0`, read from the
page; a run claiming DONE against a task that disagrees is a failure.

### Round 1 — 224 tasks per arm, before DONE supervision existed

| arm | success | 95% CI |
| --- | --- | --- |
| A | 38.4% | [27.0, 50.5] |
| B | 45.5% | [34.4, 57.1] |
| C | 43.8% | [32.7, 55.5] |
| D | 49.1% | [37.5, 61.4] |

| comparison | difference | 95% CI |
| --- | --- | --- |
| B − A | **+7.1%** | [+1.7, +13.5] |
| D − A | **+10.7%** | [+3.3, +19.1] |
| D − C | +5.4% | [−0.5, +12.1] |
| C − A | +5.4% | [−0.4, +11.8] |
| D − B | +3.6% | [−1.8, +9.7] |

Latency from this round is void: the median step count was 16 in every arm,
which is the budget, not the task. See "the termination problem" below.

### Round 2 — 216 tasks per arm, after DONE supervision, held-out families

| arm | success | 95% CI | false DONE | median steps | median ms (successes) |
| --- | --- | --- | --- | --- | --- |
| A | 28.7% | [16.3, 42.2] | 62 | 4 | 5270 |
| B | 34.3% | [21.0, 48.6] | 44 | 4 | 4719 |
| C | 34.7% | [21.0, 49.0] | 42 | 4 | 4594 |
| D | 36.1% | [22.2, 50.4] | 21 | 4 | 4503 |

| comparison | difference | 95% CI |
| --- | --- | --- |
| B − A | **+5.6%** | [+1.1, +10.9] |
| C − A | **+6.0%** | [+1.4, +12.0] |
| D − A | **+7.4%** | [+0.8, +15.3] |
| D − C | +1.4% | [−4.7, +7.8] |
| D − B | +1.9% | [−3.7, +8.7] |

**C − A excludes zero here and nowhere else, and it should not be read as support
for H3.** The DONE examples were collected only at the instant each task
completed, and MiniWoB empties `#area` at that moment, so what the model learned
is closer to "the page looks empty" than "the goal is met". False DONE went from
0 in every arm to 21–62, and accounts for 40% of A's failures against 15% of D's.
A constrained readout is more resistant to that particular misfire than free
generation, so C and D gain — on an error this project introduced. The fix is
negative examples from mid-trajectory states, and the expectation is that this
interval returns to spanning zero once they exist.

## The termination problem

The corpus contains **no DONE and no BLOCKED examples at all**: all 6454
Mind2Web steps are CLICK, TYPE_TEXT or SELECT, because an offline trajectory
records one action per step and simply stops. The model saw both symbols among
its candidates and was never once shown when to pick them, so it never did. In
round 1, **105 of 106 successful runs kept acting on a finished page until the
step budget ran out**.

This also corrected an earlier misreading. "BLOCKED predicted 0 times" looked
like good calibration; it was the model being unable to produce the token at all.

Mind2Web cannot supply the fix: it stores the observation *before* each action
and never the state after the last one, so a "task complete" observation would
have to be invented. MiniWoB can, because the environment says when a task is
done and the page is still there to observe. 93 examples were collected from 22
families, which are held out of the evaluation set — no template that taught the
model when to stop is one it is scored on.

Continued training (400 steps, 35% DONE share) fixed the symptom and introduced
the false-DONE problem above. Median steps fell from 16 to 4; **latency became
measurable for the first time**, at 4.5–5.3 s per successful task.

## Mechanism micro-benchmarks

One RTX 5090, prefix ≈1.8–2.0k tokens of which 840 are visual, median of 5 runs
after warm-up.

| generated tokens | KV reuse | full recompute | speedup |
| --- | --- | --- | --- |
| 1 | 55 ms | 165 ms | **3.0×** |
| 8 | 103 ms | 213 ms | **2.1×** |
| 32 | 103 ms | 213 ms | **2.1×** |

| branches | parallel | serial | speedup |
| --- | --- | --- | --- |
| 1 | 30 ms | 30 ms | 1.00× |
| 2 | 31 ms | 60 ms | 1.94× |
| 4 | 50 ms | 121 ms | 2.41× |

The saving is a fixed ≈110 ms — the re-prefill of the shared observation — so it
matters most for the short field values a browser actually types. Fan-out pays
from N=2 and not before, reproducing the old paper's shape on a browser workload.
The cache fork costs 2.6–3.9 ms and is not the bottleneck. Peak memory 9.8 GiB.

**None of this has been shown to reach the closed loop.** Browser I/O dominates,
and the one round with valid latency shows 4503 ms (D) against 5270 ms (A) — a
difference confounded by different success rates and step counts.

## Corpus and environment

| | |
| --- | --- |
| Mind2Web | 7362 of 7775 steps converted; 413 dropped (5.3%) for a missing gold candidate |
| viewport reduction | 3327 of 6454 usable (52%); the rest have a gold taller than a viewport or fewer than four candidates in view |
| splits | 65 training websites, 8 held out whole |
| MiniWoB | 56 of 130 templates actionable (43.1%), frozen before evaluation |
| DONE corpus | 93 examples, 174 episodes, 53.4% solve rate, 22 families held out of evaluation |
| paired visual set | 5 pairs, verified byte-identical DOM with the SVGs stripped |

**VisualWebArena is not reachable from this box.** Its site images are hosted on
`metis.lti.cs.cmu.edu` (403 from both the box and a laptop), `archive.org` (no
route from the box) and Google Drive (no route). An SSH reverse tunnel does work
— the box read a file from the laptop through it — but measured 33.7 KB/s
inbound and 48 KB/s outbound, which puts the 49.8 GiB reddit image alone at 12–18
days. WebArena-Verified's 812 task definitions download fine (0.2 MiB) and are
kept in `data/wa-verified`, but they address the same unreachable sites.

## What is still missing

| gap | cost | why it matters |
| --- | --- | --- |
| two more training seeds | ~5 h | the plan specifies three; a reviewer will ask |
| a second backbone | ~1 day | 8B is downloaded and unused; the previous paper used two |
| DONE negative examples | ~4 h | without them the C−A interval above is an artifact |
| a larger paired set | <1 day | the strongest evidence has the smallest n |
| speculative decoding (H6) | — | optional in the plan |
| calibration (NLL, Brier, reliability) | — | needed only if a probability claim is made |

## Findings the plan did not anticipate

The visual claim is the one the evidence supports, on two benchmarks with
different characteristics, with a causal test behind it and a mechanism located.
Three findings come with it:

1. **Visual value scales with page complexity.** +34.2% on real web pages against
   +5.6 to +7.1% on a simplified benchmark, same models, same training. Evaluating
   a visual web agent only on MiniWoB-style pages understates what vision is worth.
2. **An out-of-distribution visual overlay overrides explicit text.** With
   numbered badges the model had not been trained on, it put 0.99 on the wrong
   radio while the element table said `checked=true` for the one it chose; the
   same checkpoint on the same page put 0.93 on the right one from a plain
   screenshot. Numbered overlays are now standard practice, so this matters.
3. **Offline trajectory corpora contain no terminal action.** The consequence is
   agents that never stop, and it is a property of the corpus format rather than
   of Mind2Web.

The unified decision-and-generation mechanism has no support in accuracy. Its
measurable benefit is in field text (+9.1%, interval excludes zero) and in
isolated latency that has not been shown to survive the closed loop.

## Defects found in this pipeline, and the checks that now exist

Five, all mine, each costing hours:

| defect | effect | caught by |
| --- | --- | --- |
| legal-operation list contained only the gold operation | A/B operation accuracy meaningless; A/B vs C/D confounded | TYPE_TEXT joint accuracy of exactly 100% with predicted count equal to gold |
| training screenshots plain, inference screenshots numbered | D put 0.99 on the wrong element | trace inspection after an anomalous tier result |
| no terminator in the training targets | generation ran to the token cap and degenerated | a destination field filled with "Zurich, Zurich, Zurich, …" |
| terminator added to targets but not inputs | every target supervised one position early; answers lost their first token | "Bloggs" generated as "gs" |
| stale re-observations did not count as no progress | 15 wasted model calls per finished task | median step count equal to the budget in every arm |

`scripts/check_backend.py` (five properties), `scripts/check_alignment.py` (loss
alignment against the standard formulation) and the frozen-subset manifests now
run before anything expensive. The first three defects were each found two hours
into a retrain; the last two were found before or by a check that now exists.
