"""Generate the project page from the result files, so it cannot drift from them.

Every number on the page is computed here from `data/`, the same files the README
tables come from. The template holds markup and `__TOKENS__` where numbers go. A
page that hard-codes its own figures is a page that disagrees with the repository
the first time an experiment is rerun.

    python site/build_site.py --data data --template site/template.html --out docs/index.html
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(data):
    d = Path(data)
    offline = {a: json.loads((d / f"offline_{a}.json").read_text()) for a in "ABCD"}
    miniwob = json.loads((d / "miniwob_AB.json").read_text())["rows"] + \
              json.loads((d / "miniwob_CD.json").read_text())["rows"]
    done = json.loads((d / "miniwob_done_AB.json").read_text())["rows"] + \
           json.loads((d / "miniwob_done_CD.json").read_text())["rows"]
    bench = json.loads((d / "bench_stages.json").read_text())
    pilot = json.loads((d / "pilot_trained_ABCD.json").read_text())["rows"]
    return offline, miniwob, done, bench, pilot


def by_arm(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    return grouped


def paired(index_a, index_b, field, cluster, draws=5000, seed=0):
    """Cluster bootstrap over `cluster`, the unit that repeats."""
    shared = sorted(set(index_a) & set(index_b))
    groups = defaultdict(list)
    for key in shared:
        groups[index_a[key][cluster]].append(key)
    names = list(groups)

    def difference(keys):
        if not keys:
            return 0.0
        return sum(float(index_b[k][field]) - float(index_a[k][field]) for k in keys) / len(keys)

    rng = random.Random(seed)
    samples = sorted(
        difference([k for _ in names for k in groups[rng.choice(names)]]) for _ in range(draws)
    )
    return (
        difference(shared),
        samples[int(0.025 * len(samples))],
        samples[int(0.975 * len(samples)) - 1],
        sum(index_b[k][field] and not index_a[k][field] for k in shared),
        sum(index_a[k][field] and not index_b[k][field] for k in shared),
    )


def build(data, template, out):
    offline, miniwob, done, bench, pilot = load(data)
    tokens = {}

    # L1, held-out websites
    for arm in "ABCD":
        summary = offline[arm]["summary"]
        tokens[f"L1_{arm}_JOINT"] = f"{summary['joint_accuracy']:.1%}"
        tokens[f"L1_{arm}_TARGET"] = f"{summary['target_accuracy']:.1%}"
        tokens[f"L1_{arm}_OP"] = f"{summary['operation_accuracy']:.1%}"
        tokens[f"L1_{arm}_CLICK"] = f"{summary['by_operation']['CLICK']['joint']:.1%}"
    tokens["L1_N"] = str(offline["A"]["summary"]["n"])

    index = {a: {r["step_id"]: r for r in offline[a]["rows"]} for a in "ABCD"}
    for left, right, name in (("A", "B", "BA"), ("C", "D", "DC"), ("A", "C", "CA"), ("B", "D", "DB")):
        point, low, high, wins, losses = paired(
            index[left], index[right], "joint_correct", "website"
        )
        tokens[f"L1_{name}"] = f"{point:+.1%}"
        tokens[f"L1_{name}_CI"] = f"[{low:+.1%}, {high:+.1%}]"
        tokens[f"L1_{name}_WL"] = f"{wins}/{losses}"

    # L2, MiniWoB
    grouped = by_arm(miniwob)
    for arm in "ABCD":
        rows = grouped[arm]
        tokens[f"L2_{arm}"] = f"{sum(r['success'] for r in rows) / len(rows):.1%}"
    tokens["L2_N"] = str(len(grouped["A"]))
    index2 = {a: {r["task_id"]: r for r in grouped[a]} for a in "ABCD"}
    for left, right, name in (("A", "B", "BA"), ("A", "D", "DA")):
        point, low, high, wins, losses = paired(index2[left], index2[right], "success", "family")
        tokens[f"L2_{name}"] = f"{point:+.1%}"
        tokens[f"L2_{name}_CI"] = f"[{low:+.1%}, {high:+.1%}]"

    # latency, from the round that can support it
    groupedd = by_arm(done)
    for arm in "ABCD":
        stages = defaultdict(list)
        for row in groupedd[arm]:
            for step in row.get("trace", []):
                for key, value in (step.get("stages_ms") or {}).items():
                    stages[key].append(value)
        tokens[f"MS_{arm}_DECIDE"] = f"{statistics.median(stages['decide']):.0f}"
        tokens[f"MS_{arm}_OBSERVE"] = (
            f"{statistics.median(stages['observe_after']):.0f}" if stages["observe_after"] else "—"
        )
        budget = sum(1 for r in groupedd[arm] if r["success"] and r["wall_ms"] <= 3000)
        tokens[f"B3_{arm}"] = f"{budget / len(groupedd[arm]):.1%}"

    # mechanism micro-benchmarks
    reuse = [r for r in bench if "reuse_speedup" in r]
    fan = [r for r in bench if "parallel_speedup" in r]
    for tokens_out in (1, 8, 32):
        match = [r for r in reuse if r.get("generate_tokens") == tokens_out]
        tokens[f"KV_{tokens_out}"] = f"{statistics.median(r['reuse_speedup'] for r in match):.1f}×"
    for n in (1, 2, 4):
        match = [r for r in fan if r.get("branches") == n]
        tokens[f"FAN_{n}"] = f"{statistics.median(r['parallel_speedup'] for r in match):.2f}×"

    # the paired visual diagnostic, from the generated task set
    pairs = defaultdict(lambda: defaultdict(list))
    for row in pilot:
        if row["tier"] == "visual_necessary" and row.get("pair_id"):
            pairs[row["arm"]][row["pair_id"]].append(row)
    for arm in "ABCD":
        complete = [v for v in pairs[arm].values() if len(v) == 2]
        both = sum(all(r["success"] for r in v) for v in complete)
        tokens[f"PAIR_{arm}"] = f"{both}/{len(complete)}"

    html = Path(template).read_text()
    missing = []
    for key, value in tokens.items():
        marker = f"__{key}__"
        if marker not in html:
            missing.append(key)
        html = html.replace(marker, value)
    leftover = [w for w in html.split("__") if w.isupper() and "_" in w]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)
    print(f"wrote {out} from {len(tokens)} computed values")
    if missing:
        print(f"  computed but unused: {', '.join(sorted(missing))}")
    if leftover:
        print(f"  WARNING unfilled tokens remain: {', '.join(sorted(set(leftover)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--template", default="site/template.html")
    parser.add_argument("--out", default="docs/index.html")
    args = parser.parse_args()
    build(args.data, args.template, args.out)
