# Week 1 — infrastructure, and what the first measurements say

Date: 2026-09-23. Box: the box, 8× RTX 5090 (32 GB), driver 580.126.09, CUDA 13.0.
Backbone: `Qwen3-VL-4B-Instruct` (hf-mirror, no adapter yet). torch 2.13.0+cu130,
transformers 5.17.0. Upstream pinned at `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`.

Everything below is a measurement on the untrained base model. No task success
rate has been measured yet, and nothing here supports a quality claim.

## What now runs

| Plan item (§10) | State |
| --- | --- |
| snapshot / browser observation with bbox | Done, **no upstream change needed** — see below |
| model backend `observe/prefill/decide/generate/release` | Done, `vjb/model/backend.py`, 5/5 acceptance checks pass |
| `choose()` with operation + target branches, candidate mapping | Done, `vjb/model/policy.py` |
| `field_text()` as conditional continuation on reused KV | Done, numerically identical to recompute |
| `agent.py` cache lifecycle and staleness | Done, `vjb/browser/loop.py`; not yet exercised end-to-end |
| evaluator adapter | Not started |
| profiler | Done, `scripts/bench_stages.py` |

## Findings that changed the plan's assumptions

**Upstream already carries per-element geometry.** `snapshot.js` computes `rect`
for every action and strips it only from the `semantics` copy that feeds the
staleness marker — deliberately, so layout jitter does not invalidate a page. The
returned `actions` keep `rect`. Measured box coverage on a real observation:
**6/6 actionable elements, 1.00**. So the planned "add bbox" change is unnecessary,
and the marker stays geometry-insensitive, which we want.

**`DONE` and `BLOCKED` are not single tokens** in this tokenizer, while `WAIT` is.
Reading operations from the word's own logit would therefore have been invalid for
exactly the two operations that terminate a run. The policy maps every operation
to a verified single-token symbol instead. 24 symbols survived verification in the
exact readout context, which caps K at 24 per branch until the symbol set is
extended — over that, `CandidateOverflow` is raised rather than truncating.

**A recompute control must carry the image.** The first equivalence run failed
(`'Zurich'` vs `'Z'`) because the no-reuse path concatenated token ids without
`pixel_values`, so it was re-prefilling a context whose image placeholders were
never filled. That is not a control for KV reuse; it is a different context. Fixed
by keeping the processor output on the `Prefix`. After the fix the two paths agree
on **identical token ids**.

## Backend acceptance (`reports/backend_checks.json`)

| Check | Result |
| --- | --- |
| symbols | 24 verified single-token symbols; `WAIT` single-token, `DONE`/`BLOCKED` multi-token |
| image | image tokens 0 / 216 / 840 for no-image / 560×390 / 1120×780 |
| equivalence | KV reuse and full recompute produce identical token ids |
| branches | answers stay in the candidate set, normalise to 1, are invariant to branch order, and parallel == serial (max \|Δp\| = 3.7e-6) |
| isolation | running decision branches leaves generation from the shared prefix unchanged |

The image check is the one that answers "does the model actually receive the
screenshot": token counts are non-zero and grow with resolution, rather than the
prompt merely mentioning an image.

## Latency decomposition (`reports/bench_stages.json`)

Median over 5 runs after warm-up, one RTX 5090, prefix ≈1.8–2.0k tokens of which
840 are visual.

**KV reuse for the text branch is a real win, and it grows as output shortens:**

| generated tokens | reuse | recompute | speedup |
| --- | --- | --- | --- |
| 1 | 55 ms | 165 ms | **3.0×** |
| 8 | 103 ms | 213 ms | **2.1×** |
| 32 | 103 ms | 213 ms | **2.1×** |

The saving is a fixed ~110 ms — the re-prefill of the shared observation — so it
matters most for the short field values a browser actually types.

**Branch fan-out pays from N=2, and not before:**

| branches | parallel | serial | speedup |
| --- | --- | --- | --- |
| 1 | 30 ms | 30 ms | 1.00× |
| 2 | 31 ms | 60 ms | 1.94× |
| 3 | 44 ms | 90 ms | 2.06× |
| 4 | 50 ms | 121 ms | 2.41× |

This reproduces the old paper's shape — no gain at N=1 — on the browser workload,
where N is 2–4 in practice.

**The cache copy is not the bottleneck.** `fork_cache` costs 2.6–3.9 ms against a
144–155 ms prefill. An earlier reading that reuse gave no speedup (172 vs 166 ms)
was an un-warmed measurement and is withdrawn; with warm-up the same operation is
~3× faster than recompute.

Peak memory 9.8–9.9 GiB, leaving room for the training runs on the same card.

### Limitation in this benchmark

The 4000- and 8000-token configurations did not materialise: `shared_prefix`
truncates page text at 6000 characters, so every padded page converges to ≈1988
tokens. The long-context end of the reuse curve is therefore **unmeasured**, and
the numbers above only establish the 1.8–2.0k regime. Raising that cap is a
prerequisite for the H4 claim at realistic page sizes.

## Task set

`vjb/data/local_site.py` generates 20 deterministic tasks (manifest hash
`8a3802610e5ac856`): 10 visual-necessary in 5 pairs, 5 DOM-sufficient, 5
visual-helpful. Verified property of every pair: with the SVGs stripped the two
pages are **byte-identical**, the images differ, and the gold target differs. A
model that answers both members correctly has used the picture.

This is a development pilot, not the evaluation set — it stays out of any final
test split.

## Environment notes

- `huggingface.co` and `github.com` are unreachable from the box; `hf-mirror.com`,
  `ghproxy.net` and `modelscope.cn` work. Model pulls need `HF_HUB_DISABLE_XET=1`,
  since hf-mirror returns 401 on the Xet CAS endpoint.
- Closed-loop browsing works headless: Google Chrome 154 with
  `--headless=new --remote-debugging-port=9333`, and `browser-harness` pointed at
  it through `BU_CDP_URL`. No desktop session needed.
- The machine is shared with other jobs; GPU allocation is not exclusive, and a
  long run has to treat eviction as expected rather than exceptional.

## Next

1. Raise the page-text cap and re-measure reuse at 4k/8k prefixes.
2. First end-to-end closed loop on the 20 local tasks, arms A–D, base model, so
   the loop and the profiler are exercised before any training.
3. Mind2Web acquisition and the split manifest.
