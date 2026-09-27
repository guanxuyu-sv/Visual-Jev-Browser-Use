# Pinned versions

Both of these projects move. Upstream changing one line of `snapshot.js` changes
what the element table contains; MiniWoB changing one task's HTML changes what
counts as success. Everything in `data/` was produced against exactly these
commits, so a different result from a later `main` has somewhere to be traced to.

| dependency | commit | what it provides |
| --- | --- | --- |
| [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) | `1231850a0bf1a0c0341fe408ef1668dbbfdfac46` | the executor: one CDP session, the DOM snapshot, action execution and the staleness guards |
| [Farama-Foundation/miniwob-plusplus](https://github.com/Farama-Foundation/miniwob-plusplus) | `33c3b4d` | the closed-loop benchmark: 130 task templates, of which 56 are actionable by this executor |

Neither is vendored here. `REPRODUCE.md` has the checkout step.

## Python environment

The versions every run in `data/` was produced under:

| | |
| --- | --- |
| Python | 3.12 |
| torch | 2.13.0+cu130 |
| transformers | 5.17.0 |
| peft | 0.21.0 |
| pillow | 12.3.0 |
| browser-harness | 0.1.13 |
| CUDA | 13.0, driver 580.126.09 |
| GPU | one NVIDIA RTX 5090 (32 GB) per job |
| Chrome | 154.0.8037.57, headless, with the flags in `REPRODUCE.md` |

The Chrome flags are part of this list rather than a convenience: the screenshot
the model is scored on depends on them.
