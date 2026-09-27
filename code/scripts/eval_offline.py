"""L1: fixed-observation evaluation on held-out Mind2Web websites.

Same states for every arm, so operation, target and text quality can be compared
without the noise a closed loop adds. This is a diagnostic, not the paper's main
endpoint: offline step accuracy is not online task success, and the plan is
explicit that it must not be reported as if it were.

Text is scored three ways, because exact match punishes answers that are right:
`orlando` against a goal saying `Orlando` is a casing difference, not an error.
Both the strict and the normalised numbers are reported.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjb.model import prompts  # noqa: E402
from vjb.model.backend import Qwen3VLBackend  # noqa: E402
from vjb.model.policy import parse_action_line  # noqa: E402
from vjb.model.symbols import verify_pool  # noqa: E402
from vjb.train.dataset import OPERATIONS, StepSequences, load_jsonl  # noqa: E402


def normalise(text):
    """The one normalisation used for every arm, fixed before looking at results."""
    text = (text or "").strip().lower()
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def evaluate(backend, pool, examples, builder, *, arm, use_image, output_format, limit=None):
    rows = []
    for example in examples[:limit] if limit else examples:
        view, image = builder.viewport(example)
        if image is None:
            continue
        if not use_image:
            image = None
        elements = builder._elements(view)
        page = builder._page(view)
        history = [{"action": a} for a in view.get("previous_actions", [])]
        prefix_text = prompts.shared_prefix(
            view["goal"], page, elements, history, use_screenshot=image is not None
        )
        prefix = backend.prefill(prefix_text, image)
        gold_operation = view["operation"]
        gold_target = str(view["target_position"] + 1)
        keys = [str(i) for i in range(1, len(elements) + 1)]

        try:
            if output_format == "compact":
                # All operations on offer, matching training and the branch arms.
                listing = "\n".join(
                    f"{name} targets: " + ", ".join(keys)
                    for name in OPERATIONS
                    if name not in ("DONE", "BLOCKED")
                )
                suffix = (
                    "Legal operations: " + ", ".join(OPERATIONS) + "\n"
                    + listing + "\n" + prompts.compact_action_branch()
                )
                generated = backend.generate(prefix, suffix, max_new_tokens=48, stop_strings=("\n",))
                legal = {
                    name: {k: {"id": k} for k in keys}
                    for name in OPERATIONS
                    if name not in ("DONE", "BLOCKED")
                }
                parsed = parse_action_line(generated["text"], legal, ["DONE", "BLOCKED"])
                operation, target = parsed.get("operation"), parsed.get("target")
                text = parsed.get("text")
            else:
                operation_map, _ = pool.assign(OPERATIONS)
                options = list(operation_map.items())
                target_map, _ = pool.assign(keys)
                target_options = [
                    (s, f"[{k}] {elements[int(k) - 1]['label']!r} role={elements[int(k) - 1]['role']}")
                    for s, k in target_map.items()
                ]
                branches = [
                    {
                        "key": "operation",
                        "suffix": prompts.operation_branch(options),
                        "candidate_token_ids": [pool.token_id(s) for s, _ in options],
                    },
                    {
                        "key": "target",
                        "suffix": prompts.target_branch(gold_operation, target_options),
                        "candidate_token_ids": [pool.token_id(s) for s, _ in target_options],
                    },
                ]
                answers = {a["key"]: a for a in backend.decide(prefix, branches)}
                operation = list(operation_map.values())[answers["operation"]["choice_position"]]
                target = list(target_map.values())[answers["target"]["choice_position"]]
                text = None
                if gold_operation == "TYPE_TEXT":
                    gold_element = elements[view["target_position"]]
                    generated = backend.generate(
                        prefix,
                        prompts.text_branch("TYPE_TEXT", gold_element["label"], None),
                        max_new_tokens=32,
                        stop_strings=("\n",),
                    )
                    text = generated["text"]
        finally:
            backend.release(prefix)

        row = {
            "step_id": view.get("step_id"),
            "website": view.get("website"),
            "arm": arm,
            "gold_operation": gold_operation,
            "gold_target": gold_target,
            "operation": operation,
            "target": target,
            "operation_correct": operation == gold_operation,
            "target_correct": target == gold_target,
            "joint_correct": operation == gold_operation and target == gold_target,
            "candidates": len(elements),
        }
        if gold_operation == "TYPE_TEXT":
            gold_text = view.get("text") or ""
            row.update(
                gold_text=gold_text,
                text=text,
                text_exact=(text or "").strip() == gold_text.strip(),
                text_normalised=normalise(text) == normalise(gold_text),
            )
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} steps", flush=True)
    return rows


def summarise(rows):
    if not rows:
        return {}
    typed = [r for r in rows if r["gold_operation"] == "TYPE_TEXT"]
    return {
        "n": len(rows),
        "operation_accuracy": sum(r["operation_correct"] for r in rows) / len(rows),
        "target_accuracy": sum(r["target_correct"] for r in rows) / len(rows),
        "joint_accuracy": sum(r["joint_correct"] for r in rows) / len(rows),
        "by_operation": {
            op: {
                "n": sum(1 for r in rows if r["gold_operation"] == op),
                "joint": sum(r["joint_correct"] for r in rows if r["gold_operation"] == op)
                / max(1, sum(1 for r in rows if r["gold_operation"] == op)),
            }
            for op in sorted({r["gold_operation"] for r in rows})
        },
        "text": {
            "n": len(typed),
            "exact": sum(r.get("text_exact", False) for r in typed) / max(1, len(typed)),
            "normalised": sum(r.get("text_normalised", False) for r in typed) / max(1, len(typed)),
        },
        "predicted_operations": dict(Counter(r["operation"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visual-tokens", type=int, default=1024)
    args = parser.parse_args()

    use_image = args.arm in ("B", "D")
    output_format = "compact" if args.arm in ("A", "B") else "branch"

    backend = Qwen3VLBackend(args.model, adapter=args.adapter)
    pool = verify_pool(backend.tokenizer, "Answer:")
    builder = StepSequences(pool, visual_token_budget=args.visual_tokens, seed=0)
    examples = load_jsonl(args.data)
    print(f"arm {args.arm}: image={use_image} format={output_format}, {len(examples)} candidate steps")

    with torch.inference_mode():
        rows = evaluate(
            backend, pool, examples, builder,
            arm=args.arm, use_image=use_image, output_format=output_format, limit=args.limit,
        )
    report = summarise(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"arm": args.arm, "adapter": args.adapter,
                                          "summary": report, "rows": rows}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
