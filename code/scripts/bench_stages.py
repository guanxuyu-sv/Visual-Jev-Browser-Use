"""Where the milliseconds actually go, per prefix length and branch count.

This exists because the first equivalence run showed KV reuse costing as much as
the re-prefill it avoids (172 ms vs 166 ms at ~1.1k tokens). Before claiming any
efficiency result we need the decomposition: visual encode, prefill, cache fork,
branch forward, and decode, each measured separately, at the prefix lengths and
branch counts a browser step actually produces.

Writes one JSON row per configuration so the paper's figures come from the log.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjb.model import prompts  # noqa: E402
from vjb.model.backend import Qwen3VLBackend, fork_cache  # noqa: E402
from vjb.model.symbols import verify_pool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_backend import fake_page, synthetic_image  # noqa: E402


def padded_page(target_tokens, tokenizer):
    """A page whose text is padded to land near a target prefix length."""
    page, elements = fake_page()
    filler = "The page describes available flights and fare conditions. "
    page = dict(page)
    page["text"] = ""
    base = prompts.shared_prefix("Book a flight", page, elements, [], use_screenshot=True)
    base_tokens = len(tokenizer.encode(base, add_special_tokens=False))
    per = len(tokenizer.encode(filler, add_special_tokens=False))
    repeats = max(0, (target_tokens - base_tokens) // max(per, 1))
    page["text"] = (filler * repeats)[:24000]
    return page, elements


def timed(fn, repeats, warmup=1, device="cuda"):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "min_ms": min(samples),
        "n": len(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    parser.add_argument("--branches", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    backend = Qwen3VLBackend(args.model)
    pool = verify_pool(backend.tokenizer, "Answer:")
    rows = []

    for target in args.lengths:
        page, elements = padded_page(target, backend.tokenizer)
        prefix_text = prompts.shared_prefix("Book a flight from Zurich", page, elements, [], use_screenshot=True)
        image = synthetic_image()

        prefill_stats = timed(lambda: backend.prefill(prefix_text, image), args.repeats)
        prefix = backend.prefill(prefix_text, image)
        length = prefix.length

        options = [(s, f"option {i}") for i, s in enumerate(pool.symbols[:8])]
        suffix = prompts.operation_branch(options)
        branch = {
            "key": "operation",
            "suffix": suffix,
            "candidate_token_ids": [pool.token_id(s) for s, _ in options],
        }

        for count in args.branches:
            branches = [dict(branch, key=f"q{i}") for i in range(count)]
            fork_stats = timed(lambda: fork_cache(prefix.cache, count), args.repeats)
            parallel = timed(lambda: backend.decide(prefix, branches, parallel=True), args.repeats)
            serial = timed(lambda: backend.decide(prefix, branches, parallel=False), args.repeats)
            rows.append(
                {
                    "prefix_tokens": length,
                    "image_tokens": prefix.image_token_count,
                    "branches": count,
                    "prefill_ms": prefill_stats["p50_ms"],
                    "fork_ms": fork_stats["p50_ms"],
                    "decide_parallel_ms": parallel["p50_ms"],
                    "decide_serial_ms": serial["p50_ms"],
                    "parallel_speedup": serial["p50_ms"] / parallel["p50_ms"],
                }
            )
            print(json.dumps(rows[-1]), flush=True)

        # Generation: reuse vs recompute, at several output lengths.
        for max_new in (1, 8, 32):
            text_suffix = prompts.text_branch("TYPE_TEXT", "Departure city", None)
            reuse = timed(
                lambda: backend.generate(prefix, text_suffix, max_new_tokens=max_new, stop_strings=()),
                args.repeats,
            )
            recompute = timed(
                lambda: backend.generate(
                    prefix, text_suffix, max_new_tokens=max_new, stop_strings=(), reuse=False
                ),
                args.repeats,
            )
            rows.append(
                {
                    "prefix_tokens": length,
                    "image_tokens": prefix.image_token_count,
                    "generate_tokens": max_new,
                    "reuse_ms": reuse["p50_ms"],
                    "recompute_ms": recompute["p50_ms"],
                    "reuse_speedup": recompute["p50_ms"] / reuse["p50_ms"],
                }
            )
            print(json.dumps(rows[-1]), flush=True)

        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"prefix_tokens": length, "peak_gib": round(peak, 2)})
        print(json.dumps(rows[-1]), flush=True)
        backend.release(prefix)
        torch.cuda.reset_peak_memory_stats()

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
