# Development pilot, untrained base model, arms A–D

Date: 2026-09-23. `Qwen3-VL-4B-Instruct`, **no adapter, no training**. 20 generated
local tasks (manifest `8a3802610e5ac856`), one attempt each, `max_steps=10`,
greedy throughout. Success is decided by the page's own JavaScript check; a run
that claims DONE without the page agreeing is a failure and is also counted as a
false DONE.

This is a pilot whose purpose was to exercise the loop, the executor and the
profiler end to end. It is 20 synthetic tasks, one seed, and an untrained model.
Nothing here is a result about the method.

## Success

| arm | observation | output | success | 95% CI | false DONE | median ms |
| --- | --- | --- | --- | --- | --- | --- |
| A | DOM | compact AR line | 65.0% | [45.5, 88.2] | 3 | 5173 |
| B | screenshot+DOM | compact AR line | **100.0%** | [100.0, 100.0] | 0 | 4249 |
| C | DOM | branch readout + KV text | 35.0% | [15.8, 57.9] | 6 | 354 |
| D | screenshot+DOM | branch readout + KV text | 50.0% | [23.5, 73.9] | 9 | 2160 |

Intervals are cluster bootstrap over template families. With 20 tasks they are
wide, and the B interval is degenerate because B did not miss anything.

| comparison | difference | 95% CI | wins/losses |
| --- | --- | --- | --- |
| B − A (add screenshot, AR output) | **+35.0%** | [+11.8, +56.0] | 7 / 0 |
| D − C (add screenshot, branch output) | +15.0% | [−25.0, +50.0] | 7 / 4 |
| C − A (change output, DOM) | −30.0% | [−52.9, −11.1] | 1 / 7 |
| D − B (change output, screenshot+DOM) | −50.0% | [−77.8, −26.1] | 0 / 10 |

## What this does and does not show

**The screenshot helps, holding the output mechanism fixed.** B − A is +35 points
with 7 wins and no losses, and the interval excludes zero. The paired diagnostic
says the same thing far more sharply:

| arm | pairs | both correct | exactly one | neither |
| --- | --- | --- | --- | --- |
| A | 5 | 0 | **3** | 2 |
| B | 5 | **5** | 0 | 0 |
| C | 5 | 0 | **2** | 3 |
| D | 5 | 4 | 1 | 0 |

Both members of a pair have byte-identical DOM and differ only in which card
carries which picture. "Exactly one correct" is the signature of a shortcut, and
that is what both DOM-only arms produce: in the logs A picks the same element on
both members of a pair every time — `Item Z-26` on pair 1, `Item K-98` on pair 2,
`Item L-29` on pair 3 — so it is right on whichever member happens to match. No
DOM-only arm gets a single pair fully right; the screenshot arms get 5/5 and 4/5.

**The output mechanism cannot be judged here at all.** C and D are worse than A
and B, badly so on the DOM-sufficient tier (0% for both), with 6 and 9 false
DONEs. That is what an *untrained* logit readout should look like: the base model
was never taught to answer these questions with a candidate symbol, and DONE is
one of five symbols competing on an uncalibrated distribution. D − C and D − B
therefore say nothing about the method; they say the readout needs the training
that has not happened yet. The A/B arms use the model's native generation, which
is the ability the base checkpoint actually has.

**Latency is not comparable yet either.** C's 354 ms median against A's 5173 ms is
mostly C failing early, not C being fast. Conditional times from arms with
different success rates are not a speed comparison.

## Environment behaviour worth recording

216 screenshot captures over the run, **87 needing a retry**, 0 ultimately
failing. The first-attempt failure rate of ~40% is a property of capturing from a
headless background tab, not of the page or the model; see
`configure_cdp` in `vjb/browser/loop.py`. Before that was handled, a single
capture cost a 20-second timeout and silently aborted the step it belonged to.

The screenshot depends on the Chrome flags, so those flags are part of the pinned
environment, not an incidental launch detail.

## Next

1. Train the four arms on the Mind2Web corpus (6454 train / 908 validation steps,
   65 train websites, 8 held out) and re-run this pilot. Only then do C and D mean
   anything.
2. The pilot tasks stay a development set and do not enter any final split.
