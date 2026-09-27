"""Collect real DONE supervision from MiniWoB, on families held out of the test set.

The corpus has no DONE at all: Mind2Web stores one action per step and stops, so
every one of its 6454 examples is CLICK, TYPE_TEXT or SELECT. The model therefore
sees DONE among the candidates and has never once been shown when to pick it --
which is why 105 of 106 successful runs kept re-clicking a finished page until
the step budget ran out.

Mind2Web cannot fix this: it records the observation *before* each action and
never the state after the last one, so a "task is complete" observation would
have to be invented. MiniWoB can, because the environment says when the task is
done and the page is still there to observe.

Split discipline: families used here are removed from the evaluation set. A
template that trains DONE never appears in a reported number.

Each episode is driven by a scripted solver where one exists, or by the trained
policy; the DONE example is the observation taken at the moment the environment
reports success, and it is kept only if the environment really did report it.
"""

import argparse
import http.server
import json
import os
import random
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _third_party(name):
    """Where the external dependencies live.

    `VJB_THIRD_PARTY` overrides the default so they need not sit inside a
    checkout of this repository. `code/scripts/fetch_third_party.sh` puts them
    at the default and pins both commits.
    """
    root = os.environ.get("VJB_THIRD_PARTY")
    base = Path(root) if root else Path(__file__).resolve().parents[1] / "third_party"
    target = base / name
    if not target.exists():
        raise SystemExit(
            f"{name} not found at {target}.\n"
            "Run:  bash code/scripts/fetch_third_party.sh\n"
            "or point VJB_THIRD_PARTY at an existing checkout."
        )
    return target

sys.path.insert(0, str(_third_party("jev-ultrafast")))

from vjb.browser.observe import action_space  # noqa: E402
from vjb.data import miniwob  # noqa: E402


def serve(directory):
    handler = type(
        "H",
        (http.server.SimpleHTTPRequestHandler,),
        {
            "__init__": lambda s, *a, **k: http.server.SimpleHTTPRequestHandler.__init__(
                s, *a, directory=str(directory), **k
            ),
            "log_message": lambda *_: None,
        },
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def to_example(page, goal, *, operation, target_index, history, template, seed, text=None):
    """One training example in the same shape the Mind2Web converter emits."""
    elements, _, _ = action_space(page["actions"])
    return {
        "task_id": f"miniwob:{template}#{seed}",
        "website": f"miniwob:{template}",
        "domain": "miniwob",
        "subdomain": miniwob.family(template),
        "step_id": f"miniwob:{template}#{seed}:{len(history)}",
        "goal": goal,
        "operation": operation,
        "supervised_branches": ["operation"] + ([f"{operation.lower()}_target"] if target_index else []),
        "target_position": (target_index - 1) if target_index else 0,
        "candidates": [
            {
                "tag": e.get("role") or "?",
                "label": e.get("label") or "",
                "value": e.get("value") or "",
                "node": e["index"],
                "rect": e.get("rect"),
            }
            for e in elements
        ],
        "text": text,
        "previous_actions": list(history),
        "source": "miniwob",
    }


def collect(templates, base_url, policy, *, seeds, max_steps, budget_ms, out_dir):
    from vjb.browser.loop import Run

    examples, stats = [], {"episodes": 0, "solved": 0, "done_examples": 0}
    shots = Path(out_dir) / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    for template in templates:
        for seed in seeds:
            stats["episodes"] += 1
            run = None
            try:
                run = Run(
                    miniwob.task_url(base_url, template, seed),
                    "placeholder",
                    policy,
                    screenshots=True,
                    max_steps=max_steps,
                )
                ok, goal = miniwob.start_episode(run.browser, seed, budget_ms=budget_ms)
                if not ok:
                    continue
                run.goal = goal
                run.page = run.browser.observe(screenshot=True)
                for _ in run.run():
                    done, raw, _ = miniwob.read_reward(run.browser)
                    if done and isinstance(raw, (int, float)) and raw > 0:
                        # The environment says the task is finished. The page is
                        # still observable, so this is a real DONE observation
                        # rather than a constructed one.
                        page = run.browser.observe(screenshot=True)
                        example = to_example(
                            page, goal, operation="DONE", target_index=None,
                            history=[h["action"] for h in run.history],
                            template=template, seed=seed,
                        )
                        path = shots / f"{template}_{seed}_done.jpg"
                        if page.get("screenshot"):
                            import base64

                            path.write_bytes(base64.b64decode(page["screenshot"]))
                            example["screenshot"] = str(path)
                            examples.append(example)
                            stats["done_examples"] += 1
                        stats["solved"] += 1
                        break
            except Exception:
                pass
            finally:
                if run is not None:
                    run.close()
        print(f"  {template:30s} episodes={stats['episodes']} done={stats['done_examples']}", flush=True)
    return examples, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="policy used to solve episodes")
    parser.add_argument("--out", default="${VJB_ROOT}/work/done")
    parser.add_argument("--train-fraction", type=float, default=0.5,
                        help="fraction of families reserved for DONE collection")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14, 15])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--budget-ms", type=int, default=600000)
    parser.add_argument("--arm", default="D")
    args = parser.parse_args()

    html = _third_party("miniwob-html")
    os.environ.setdefault("BU_CDP_URL", "http://127.0.0.1:9333")

    from vjb.browser.loop import configure_cdp

    configure_cdp(timeout=20.0)

    # Split by family, not template: near-duplicate templates must not straddle it.
    templates = miniwob.supported(html / "miniwob")
    families = sorted({miniwob.family(t) for t in templates})
    random.Random(0).shuffle(families)
    cut = int(len(families) * args.train_fraction)
    collect_families = set(families[:cut])
    held_out = sorted(set(families) - collect_families)
    collect_templates = [t for t in templates if miniwob.family(t) in collect_families]

    split = {
        "collection_families": sorted(collect_families),
        "evaluation_families": held_out,
        "collection_templates": collect_templates,
        "evaluation_templates": [t for t in templates if miniwob.family(t) not in collect_families],
        "seeds_used_for_collection": args.seeds,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "split.json").write_text(json.dumps(split, indent=2))
    print(
        f"families {len(collect_families)} for collection, {len(held_out)} held out; "
        f"templates {len(collect_templates)} / {len(split['evaluation_templates'])}"
    )

    server, base_url = serve(html)
    from vjb.model.backend import Qwen3VLBackend
    from vjb.model.policy import Policy

    backend = Qwen3VLBackend(args.model, adapter=args.adapter)
    arms = {
        "A": dict(use_screenshot=False, parallel_branches=False),
        "B": dict(use_screenshot=True, parallel_branches=False),
        "C": dict(use_screenshot=False, parallel_branches=True),
        "D": dict(use_screenshot=True, parallel_branches=True),
    }
    policy = Policy(backend, arm=args.arm, reuse_kv=True, **arms[args.arm])

    started = time.perf_counter()
    examples, stats = collect(
        collect_templates, base_url, policy,
        seeds=args.seeds, max_steps=args.max_steps, budget_ms=args.budget_ms, out_dir=args.out,
    )
    path = Path(args.out) / "done.jsonl"
    with open(path, "w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    stats["elapsed_s"] = round(time.perf_counter() - started)
    (Path(args.out) / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n{stats}")
    print(f"wrote {path}")
    server.shutdown()


if __name__ == "__main__":
    main()
