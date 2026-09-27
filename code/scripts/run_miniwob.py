"""Closed loop over MiniWoB++, scored by the task's own reward.

Success is `WOB_DONE_GLOBAL && WOB_RAW_REWARD_GLOBAL > 0`, read from the page.
The model's DONE is recorded but never decides the outcome: a run that claims
DONE against a task that disagrees is a failure and is counted as a false DONE.

Two disclosed deviations from stock MiniWoB, both forced by evaluating a VLM:

  * the episode clock is raised from its 10 s default, which assumes a policy
    acting in milliseconds; wall-clock time is reported separately and is not
    folded into the score
  * the raw reward is used rather than the displayed one, because MiniWoB scales
    some rewards by how much of the clock is left, mixing success with speed

The task list is frozen before any model runs: templates the executor cannot act
in are excluded by action space alone and the coverage is reported.
"""

import argparse
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

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

from vjb.data import miniwob  # noqa: E402

ARMS = {
    "A": dict(use_screenshot=False, parallel_branches=False, reuse_kv=True, arm="A"),
    "B": dict(use_screenshot=True, parallel_branches=False, reuse_kv=True, arm="B"),
    "C": dict(use_screenshot=False, parallel_branches=True, reuse_kv=True, arm="C"),
    "D": dict(use_screenshot=True, parallel_branches=True, reuse_kv=True, arm="D"),
}


def serve(directory):
    handler = type(
        "Handler",
        (http.server.SimpleHTTPRequestHandler,),
        {
            "__init__": lambda self, *a, **kw: http.server.SimpleHTTPRequestHandler.__init__(
                self, *a, directory=str(directory), **kw
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


def run_task(task, base_url, policy, *, max_steps, budget_ms):
    from vjb.browser.loop import Run

    url = miniwob.task_url(base_url, task["template"], task["seed"])
    started = time.perf_counter()
    run = error = None
    try:
        run = Run(url, "placeholder", policy, screenshots=True, max_steps=max_steps)
        ok, goal = miniwob.start_episode(run.browser, task["seed"], budget_ms=budget_ms)
        if not ok:
            return {**_blank(task, policy.arm), "error": goal, "wall_ms": 0}
        # The goal only exists once the problem is generated, so the run is given
        # its instruction here rather than at construction.
        run.goal = goal
        run.page = run.browser.observe(screenshot=True)
        for _ in run.run():
            pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    done = raw = shown = None
    if run is not None:
        done, raw, shown = miniwob.read_reward(run.browser)
    success = bool(done) and isinstance(raw, (int, float)) and raw > 0
    result = run.result() if run and run.started_at else {"status": "error", "steps": [], "elapsed_ms": 0}
    row = {
        **_blank(task, policy.arm),
        "goal": getattr(run, "goal", ""),
        "success": success,
        "env_done": bool(done),
        "raw_reward": raw,
        "shown_reward": shown,
        "claimed_done": result.get("status") == "done",
        "status": result.get("status"),
        "steps": len(result.get("steps", [])),
        "wall_ms": round((time.perf_counter() - started) * 1000),
        "elapsed_ms": result.get("elapsed_ms", 0),
        "error": error,
        "trace": result.get("steps", []),
    }
    if run is not None:
        run.close()
    return row


def _blank(task, arm):
    return {
        "task_id": task["task_id"],
        "template": task["template"],
        "seed": task["seed"],
        "family": task["family"],
        "tier": task["tier"],
        "arm": arm,
        "success": False,
        "claimed_done": False,
        "steps": 0,
        "trace": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--html", default=None, help="vendored miniwob html dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", nargs="+", default=["D"])
    parser.add_argument("--adapters", default=None, help="one adapter per arm, comma separated")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--templates", type=int, default=None, help="cap templates (debug only)")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--budget-ms", type=int, default=600000)
    parser.add_argument("--cdp-timeout", type=float, default=20.0)
    parser.add_argument("--only", default=None,
                        help="JSON file with the template list to evaluate on")
    args = parser.parse_args()

    html = Path(args.html or _third_party("miniwob-html"))
    os.environ.setdefault("BU_CDP_URL", "http://127.0.0.1:9333")

    from vjb.browser.loop import CDP_STATS, configure_cdp

    cdp_config = configure_cdp(timeout=args.cdp_timeout)
    print(f"cdp: {cdp_config}")

    only = json.loads(Path(args.only).read_text()) if args.only else None
    manifest = miniwob.build_manifest(html / "miniwob", seeds=tuple(args.seeds),
                                      limit=args.templates, only=only)
    print(
        f"templates: {manifest['templates_supported']}/{manifest['templates_available']} supported "
        f"({manifest['coverage']:.1%}), {len(manifest['tasks'])} tasks over seeds {args.seeds}"
    )

    server, base_url = serve(html)
    print(f"serving {html} at {base_url}")

    from vjb.model.backend import Qwen3VLBackend
    from vjb.model.policy import Policy

    per_arm = {}
    if args.adapters:
        names = [a.strip() for a in args.adapters.split(",")]
        if len(names) != len(args.arms):
            raise SystemExit(f"--adapters has {len(names)} entries for {len(args.arms)} arms")
        per_arm = dict(zip(args.arms, names))

    backend = None
    rows = []
    for arm in args.arms:
        adapter = per_arm.get(arm)
        del backend
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        backend = Qwen3VLBackend(args.model, adapter=adapter)
        print(f"arm {arm}: {adapter or 'base'}", flush=True)
        policy = Policy(backend, **ARMS[arm])
        for index, task in enumerate(manifest["tasks"], 1):
            row = run_task(task, base_url, policy, max_steps=args.max_steps, budget_ms=args.budget_ms)
            rows.append(row)
            if index % 20 == 0 or not row["success"]:
                mark = "ok " if row["success"] else "MISS"
                print(
                    f"  [{arm}] {mark} {row['task_id']:34s} r={row.get('raw_reward')} "
                    f"steps={row['steps']:2d} {row['wall_ms']:6d}ms {row.get('error') or ''}",
                    flush=True,
                )
        # Checkpoint after each arm: a run this long must not be lost to a
        # formatting error at the end. It already was, once.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".partial.json").write_text(json.dumps({"rows": rows}, indent=2))
        done = [r for r in rows if r["arm"] == arm]
        print(
            f"  [{arm}] success {sum(r['success'] for r in done)}/{len(done)}  "
            f"false DONE {sum(r['claimed_done'] and not r['success'] for r in done)}  "
            f"errors {sum(bool(r.get('error')) for r in done)}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "benchmark": manifest["source"],
                "coverage": manifest["coverage"],
                "templates_supported": manifest["templates_supported"],
                "templates_available": manifest["templates_available"],
                "unsupported_static": manifest.get("unsupported_static", []),
                "exclusion_reasons": manifest.get("exclusion_reasons", {}),
                "seeds": args.seeds,
                "episode_budget_ms": args.budget_ms,
                "model": args.model,
                "adapters": per_arm,
                "cdp": cdp_config,
                "cdp_stats": dict(CDP_STATS),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nscreenshots: {CDP_STATS}")
    print(f"wrote {args.out}")
    server.shutdown()


if __name__ == "__main__":
    main()
