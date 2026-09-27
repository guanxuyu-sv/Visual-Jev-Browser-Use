"""Build the offline supervision corpus from the Multimodal-Mind2Web train shards.

Reads the parquet shards directly with pyarrow — `datasets` is not needed and
pulls a large dependency tree. Writes JSONL examples plus the screenshots, and a
manifest recording how many steps were dropped and why. The drop reasons matter:
`gold_candidate_missing` is a candidate-recall failure that the paper has to
report, not something to quietly skip.

Splits are held out by website, so no template from a held-out site appears in
training. The official test_* shards are not read at all here.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjb.data.mind2web import convert_step, save_screenshot, split_manifest  # noqa: E402


def iter_rows(shards, columns=None, limit=None):
    seen = 0
    for shard in shards:
        table = pq.read_table(shard, columns=columns)
        for batch in table.to_batches(max_chunksize=64):
            for row in batch.to_pylist():
                yield row
                seen += 1
                if limit and seen >= limit:
                    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="${VJB_ROOT}/data/mm-mind2web/data")
    parser.add_argument("--out", default="${VJB_ROOT}/work/m2w")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--val-websites", type=int, default=8, help="websites held out for validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-screenshots", action="store_true")
    args = parser.parse_args()

    shards = sorted(Path(args.data).glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"no train shards under {args.data}")
    print(f"{len(shards)} shards")

    out = Path(args.out)
    (out / "screenshots").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    examples, reasons = [], Counter()
    for row in iter_rows(shards, limit=args.limit):
        task = {
            "annotation_id": row.get("annotation_id"),
            "website": row.get("website"),
            "domain": row.get("domain"),
            "subdomain": row.get("subdomain"),
            "confirmed_task": row.get("confirmed_task"),
        }
        example, reason = convert_step(row, task, rng=rng, limit=args.candidates)
        if example is None:
            reasons[reason] += 1
            continue
        if not args.no_screenshots:
            path = save_screenshot(row, out / "screenshots")
            if path is None:
                reasons["no_screenshot"] += 1
                continue
            example["screenshot"] = path
        examples.append(example)
        reasons["kept"] += 1
        if reasons["kept"] % 2000 == 0:
            print(f"  kept {reasons['kept']}", flush=True)

    # Hold out whole websites, so a template never straddles the split.
    websites = sorted({e["website"] for e in examples if e["website"]})
    rng.shuffle(websites)
    held_out = set(websites[: args.val_websites])
    train = [e for e in examples if e["website"] not in held_out]
    validation = [e for e in examples if e["website"] in held_out]

    for name, rows in (("train", train), ("validation", validation)):
        path = out / f"{name}.jsonl"
        with open(path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(rows)} examples)")

    manifest = {
        "shards": [s.name for s in shards],
        "drop_reasons": dict(reasons),
        "kept": len(examples),
        "candidate_limit": args.candidates,
        "seed": args.seed,
        "validation_websites": sorted(held_out),
        "train": split_manifest(train),
        "validation": split_manifest(validation),
    }
    manifest["train"]["operations"] = dict(manifest["train"]["operations"])
    manifest["train"]["candidate_sizes"] = dict(manifest["train"]["candidate_sizes"])
    manifest["validation"]["operations"] = dict(manifest["validation"]["operations"])
    manifest["validation"]["candidate_sizes"] = dict(manifest["validation"]["candidate_sizes"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(reasons.values())
    print(f"\nkept {reasons['kept']}/{total} steps")
    for reason, count in reasons.most_common():
        if reason != "kept":
            print(f"  dropped {reason}: {count} ({count / total:.1%})")
    print(f"train {len(train)}, validation {len(validation)} across {len(websites)} websites")


if __name__ == "__main__":
    main()
