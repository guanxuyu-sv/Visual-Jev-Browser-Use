"""Week-1 acceptance checks for the model backend. Each prints PASS/FAIL and why.

1. symbols     every candidate symbol is one token in the exact readout context
2. image       the model actually receives image tokens, and more of them at
               higher resolution (so "it saw the screenshot" is measured, not assumed)
3. equivalence KV reuse and a full recompute agree within tolerance
4. branches    a branch cannot answer outside its candidate set, and branch order
               does not change any branch's answer
5. isolation   running the decision branches leaves the shared prefix usable for
               generation, and generation is unaffected by how many branches ran

Run on the box:
    .venv/bin/python scripts/check_backend.py --model ${VJB_ROOT}/models/Qwen3-VL-4B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjb.model import prompts  # noqa: E402
from vjb.model.backend import Qwen3VLBackend  # noqa: E402
from vjb.model.symbols import verify_pool, verify_words  # noqa: E402

RESULTS = []


def record(name, passed, detail):
    RESULTS.append({"check": name, "pass": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def fake_page():
    """A small synthetic page, so the check needs no browser."""
    elements = [
        {"index": "1", "role": "textbox", "label": "From", "value": "", "operations": ["TYPE_TEXT"],
         "rect": {"x": 40, "y": 120, "w": 220, "h": 36}},
        {"index": "2", "role": "textbox", "label": "To", "value": "", "operations": ["TYPE_TEXT"],
         "rect": {"x": 300, "y": 120, "w": 220, "h": 36}},
        {"index": "3", "role": "button", "label": "Search", "operations": ["CLICK"],
         "rect": {"x": 560, "y": 120, "w": 120, "h": 36}},
    ]
    page = {
        "url": "https://example.test/flights",
        "title": "Flight search",
        "text": "Flight search\nFrom\nTo\nSearch",
        "actions": [],
    }
    return page, elements


def synthetic_image(width=1120, height=780):
    image = Image.new("RGB", (width, height), (245, 245, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 120, 260, 156], outline=(60, 60, 60), width=2)
    draw.text((48, 130), "From", fill=(20, 20, 20))
    draw.rectangle([300, 120, 520, 156], outline=(60, 60, 60), width=2)
    draw.text((308, 130), "To", fill=(20, 20, 20))
    draw.rectangle([560, 120, 680, 156], fill=(0, 110, 220))
    draw.text((588, 130), "Search", fill=(255, 255, 255))
    return image


def check_symbols(backend):
    context = "Answer:"
    pool = verify_pool(backend.tokenizer, context)
    bad = []
    base = backend.tokenizer.encode(context, add_special_tokens=False)
    for symbol, token_id in zip(pool.symbols, pool.token_ids):
        ids = backend.tokenizer.encode(context + symbol, add_special_tokens=False)
        if ids != base + [token_id]:
            bad.append(symbol)
    words, dropped = verify_words(backend.tokenizer, context, ["DONE", "BLOCKED", "WAIT"])
    record(
        "symbols",
        not bad and len(pool) >= 16,
        f"{len(pool)} verified single-token symbols, re-check mismatches={bad}, "
        f"control words single-token={sorted(words)}, multi-token={dropped}",
    )
    return pool


def check_image(backend):
    page, elements = fake_page()
    text_only = prompts.shared_prefix("Find a flight", page, elements, [], use_screenshot=False)
    with_image = prompts.shared_prefix("Find a flight", page, elements, [], use_screenshot=True)

    none_prefix = backend.prefill(text_only, None)
    small = backend.prefill(with_image, synthetic_image(560, 390))
    large = backend.prefill(with_image, synthetic_image(1120, 780))
    detail = (
        f"image tokens: none={none_prefix.image_token_count} "
        f"small={small.image_token_count} large={large.image_token_count}; "
        f"prefill tokens {none_prefix.length} / {small.length} / {large.length}"
    )
    passed = (
        none_prefix.image_token_count == 0
        and small.image_token_count > 0
        and large.image_token_count > small.image_token_count
    )
    record("image", passed, detail)
    for prefix in (none_prefix, small, large):
        backend.release(prefix)
    return passed


def check_equivalence(backend, tolerance=0.02):
    """Reuse vs recompute on the same logical context."""
    page, elements = fake_page()
    prefix_text = prompts.shared_prefix(
        "Book a flight from Zurich to Berlin", page, elements, [], use_screenshot=True
    )
    prefix = backend.prefill(prefix_text, synthetic_image())
    suffix = prompts.text_branch("TYPE_TEXT", "From", None)

    reused = backend.generate(prefix, suffix, max_new_tokens=24, reuse=True)
    recomputed = backend.generate(prefix, suffix, max_new_tokens=24, reuse=False)
    same_text = reused["text"] == recomputed["text"]
    same_tokens = reused["token_ids"] == recomputed["token_ids"]
    record(
        "equivalence",
        same_text,
        f"reuse={reused['text']!r} ({reused['tokens']} tok, reused {reused['reused_prefix_tokens']} prefix tokens, "
        f"{reused['total_ms']:.0f} ms) vs recompute={recomputed['text']!r} "
        f"({recomputed['total_ms']:.0f} ms); token ids identical={same_tokens}",
    )
    backend.release(prefix)
    return same_text


def check_branches(backend, pool):
    page, elements = fake_page()
    prefix_text = prompts.shared_prefix(
        "Type Zurich in the From field", page, elements, [], use_screenshot=True
    )
    prefix = backend.prefill(prefix_text, synthetic_image())

    operations = [("1", "CLICK — click an element"), ("2", "TYPE_TEXT — type into a field"),
                  ("3", "DONE — everything is satisfied")]
    targets = [("1", "[1] 'From' role=textbox"), ("2", "[2] 'To' role=textbox")]
    op_branch = {
        "key": "operation",
        "suffix": prompts.operation_branch(operations),
        "candidate_token_ids": [pool.token_id(s) for s, _ in operations],
    }
    target_branch = {
        "key": "type_text_target",
        "suffix": prompts.target_branch("TYPE_TEXT", targets),
        "candidate_token_ids": [pool.token_id(s) for s, _ in targets],
    }

    both = backend.decide(prefix, [op_branch, target_branch], parallel=True)
    reversed_order = backend.decide(prefix, [target_branch, op_branch], parallel=True)
    serial = backend.decide(prefix, [op_branch, target_branch], parallel=False)

    by_key = {a["key"]: a for a in both}
    rev = {a["key"]: a for a in reversed_order}
    ser = {a["key"]: a for a in serial}

    in_range = all(
        0 <= a["choice_position"] < len(b["candidate_token_ids"])
        for a, b in ((by_key["operation"], op_branch), (by_key["type_text_target"], target_branch))
    )
    normalised = all(abs(sum(a["probabilities"]) - 1) < 1e-4 for a in both)
    order_stable = all(by_key[k]["choice_position"] == rev[k]["choice_position"] for k in by_key)
    serial_agrees = all(by_key[k]["choice_position"] == ser[k]["choice_position"] for k in by_key)
    max_drift = max(
        abs(p - q)
        for k in by_key
        for p, q in zip(by_key[k]["probabilities"], ser[k]["probabilities"])
    )
    record(
        "branches",
        in_range and normalised and order_stable and serial_agrees,
        f"in-set={in_range} normalised={normalised} order-invariant={order_stable} "
        f"parallel==serial={serial_agrees} max|Δp|={max_drift:.2e}; "
        f"operation→{operations[by_key['operation']['choice_position']][1].split(' —')[0]} "
        f"target→{targets[by_key['type_text_target']['choice_position']][0]}; "
        f"parallel {by_key['operation']['batch_ms']:.0f} ms for 2 branches vs serial "
        f"{sum(a['batch_ms'] for a in serial):.0f} ms",
    )
    backend.release(prefix)
    return in_range and normalised and order_stable and serial_agrees


def check_isolation(backend, pool):
    """Branches must not disturb the prefix that generation then continues."""
    page, elements = fake_page()
    prefix_text = prompts.shared_prefix(
        "Book a flight from Zurich to Berlin", page, elements, [], use_screenshot=True
    )
    prefix = backend.prefill(prefix_text, synthetic_image())
    suffix = prompts.text_branch("TYPE_TEXT", "From", None)

    before = backend.generate(prefix, suffix, max_new_tokens=24, reuse=True)
    operations = [("1", "CLICK — click"), ("2", "TYPE_TEXT — type"), ("3", "DONE — done")]
    backend.decide(
        prefix,
        [{"key": "operation", "suffix": prompts.operation_branch(operations),
          "candidate_token_ids": [pool.token_id(s) for s, _ in operations]}],
        parallel=True,
    )
    after = backend.generate(prefix, suffix, max_new_tokens=24, reuse=True)
    record(
        "isolation",
        before["token_ids"] == after["token_ids"],
        f"generation before branches={before['text']!r}, after branches={after['text']!r}",
    )
    backend.release(prefix)
    return before["token_ids"] == after["token_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--attn", default=None, help="attn_implementation, e.g. sdpa or eager")
    args = parser.parse_args()

    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {torch.cuda.get_device_name(0)}")
    backend = Qwen3VLBackend(args.model, attn=args.attn)
    print(f"loaded {args.model}", flush=True)

    pool = check_symbols(backend)
    check_image(backend)
    check_equivalence(backend)
    check_branches(backend, pool)
    check_isolation(backend, pool)

    passed = sum(r["pass"] for r in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    if args.out:
        Path(args.out).write_text(json.dumps(RESULTS, indent=2))
        print(f"wrote {args.out}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
