"""Closed-loop run over the generated local tasks, one arm at a time.

Success is decided by the page, not by the model: each task carries a JavaScript
expression evaluated in the live document after the run, and compared to the
value the task manifest fixed before any model was involved. A run that says DONE
without the page agreeing is a failure, and is counted as one.

    python scripts/run_local.py --model ... --arms A B C D --out runs/pilot.json
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "jev-ultrafast"))

from vjb.data.local_site import generate  # noqa: E402

ARMS = {
    "A": dict(use_screenshot=False, parallel_branches=False, reuse_kv=True, arm="A"),
    "B": dict(use_screenshot=True, parallel_branches=False, reuse_kv=True, arm="B"),
    "C": dict(use_screenshot=False, parallel_branches=True, reuse_kv=True, arm="C"),
    "D": dict(use_screenshot=True, parallel_branches=True, reuse_kv=True, arm="D"),
}


def serve(directory):
    """A local static server, so pages load over http rather than file://."""
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
    server.allow_reuse_address = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def verify(browser, success):
    """Ask the live page whether the task is done. The model never answers this."""
    if success["kind"] != "js":
        raise ValueError(f"unknown success kind {success['kind']}")
    try:
        value = browser.evaluate(f"String(({success['expression']}))")
    except Exception as exc:  # a navigated-away page is a failure, not a crash
        return False, f"verification failed: {type(exc).__name__}"
    return str(value) == str(success["equals"]), value


def run_task(task, base_url, policy, *, max_steps, record_dir=None):
    from vjb.browser.loop import Run

    url = f"{base_url}/pages/{task['page']}"
    started = time.perf_counter()
    run = Run(url, task["goal"], policy, record_dir=record_dir, screenshots=True, max_steps=max_steps)
    error = None
    try:
        for _ in run.run():
            pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    passed, observed = verify(run.browser, task["success"])
    result = run.result()
    run.close()
    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "pair_id": task["pair_id"],
        "family": task["family"],
        "goal": task["goal"],
        "gold_label": task["gold_label"],
        "arm": policy.arm,
        "success": passed,
        "observed": observed,
        "expected": task["success"]["equals"],
        "claimed_done": result["status"] == "done",
        "status": result["status"],
        "steps": len(result["steps"]),
        "wall_ms": round((time.perf_counter() - started) * 1000),
        "elapsed_ms": result["elapsed_ms"],
        "startup_ms": result["startup_ms"],
        "error": error,
        "trace": result["steps"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--site", default="${VJB_ROOT}/work/local_site")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", nargs="+", default=["D"])
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tiers", nargs="+", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--adapters", default=None,
                        help="comma-separated adapter per --arms entry; each arm is "
                             "evaluated with the adapter it was trained with")
    parser.add_argument("--records", default=None)
    parser.add_argument("--cdp-timeout", type=float, default=20.0)
    args = parser.parse_args()

    os.environ.setdefault("BU_CDP_URL", "http://127.0.0.1:9333")

    from vjb.browser.loop import CDP_STATS, configure_cdp
    cdp_config = configure_cdp(timeout=args.cdp_timeout)
    print(f"cdp: {cdp_config}")

    manifest_path = Path(args.site) / "tasks.json"
    if not manifest_path.exists():
        print(f"generating task site at {args.site}")
        generate(args.site)
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    if args.tiers:
        tasks = [t for t in tasks if t["tier"] in args.tiers]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} tasks, manifest hash {manifest['hash']}")

    server, base_url = serve(args.site)
    print(f"serving {args.site} at {base_url}")

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
        adapter = per_arm.get(arm, args.adapter)
        # Each arm carries its own adapter, so the backend is rebuilt per arm
        # rather than reusing weights another arm was trained into.
        del backend
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        backend = Qwen3VLBackend(args.model, adapter=adapter)
        print(f"arm {arm}: {args.model}" + (f" + {adapter}" if adapter else " (base)"), flush=True)
        policy = Policy(backend, **ARMS[arm])
        for task in tasks:
            record_dir = Path(args.records) / arm / task["task_id"] if args.records else None
            row = run_task(task, base_url, policy, max_steps=args.max_steps, record_dir=record_dir)
            rows.append(row)
            mark = "ok " if row["success"] else "MISS"
            print(
                f"  [{arm}] {mark} {row['task_id']:10s} {row['tier']:18s} "
                f"steps={row['steps']:2d} {row['wall_ms']:6d}ms status={row['status']:8s} "
                f"observed={str(row['observed'])[:28]!r}",
                flush=True,
            )
        done = [r for r in rows if r["arm"] == arm]
        print(
            f"  [{arm}] success {sum(r['success'] for r in done)}/{len(done)}  "
            f"false DONE {sum(r['claimed_done'] and not r['success'] for r in done)}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"manifest_hash": manifest["hash"], "model": args.model,
                    "adapter": args.adapter, "adapters": per_arm, "cdp": cdp_config,
                    "cdp_stats": dict(CDP_STATS), "rows": rows}, indent=2)
    )
    print(f"\nscreenshot calls={CDP_STATS['screenshot_calls']} retries={CDP_STATS['screenshot_retries']} failures={CDP_STATS['screenshot_failures']}")
    print(f"\nwrote {args.out}")
    server.shutdown()


if __name__ == "__main__":
    main()
