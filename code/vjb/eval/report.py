"""Turn raw run logs into the numbers the paper reports, with honest intervals.

Three things this is careful about, because the plan requires them:

  * success comes from the page's own verification, never from the model saying DONE;
    a claimed DONE that the page contradicts is counted as a false DONE
  * intervals are cluster bootstrap over template families, not over episodes,
    because paired pages and shared templates are not independent samples
  * the paired visual score is `both members of a pair correct`. Getting one of
    two right is what a DOM/position shortcut produces, so it is not half a point
"""

import json
import random
import statistics
from collections import defaultdict


def load(path):
    data = json.loads(open(path).read())
    return data["rows"], data


def by_arm(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    return grouped


def success_rate(rows):
    return sum(r["success"] for r in rows) / len(rows) if rows else 0.0


def false_done(rows):
    """Runs that claimed DONE while the page disagreed."""
    return sum(r["claimed_done"] and not r["success"] for r in rows)


def paired_score(rows):
    """For visual-necessary pairs: how many pairs are fully correct, and how many
    show the one-of-two pattern that a shortcut produces."""
    pairs = defaultdict(list)
    for row in rows:
        if row["tier"] == "visual_necessary" and row["pair_id"]:
            pairs[row["pair_id"]].append(row)
    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    both = sum(all(r["success"] for r in v) for v in complete.values())
    one = sum(sum(r["success"] for r in v) == 1 for v in complete.values())
    neither = sum(not any(r["success"] for r in v) for v in complete.values())
    return {
        "pairs": len(complete),
        "both_correct": both,
        "exactly_one": one,
        "neither": neither,
        "paired_rate": both / len(complete) if complete else 0.0,
    }


def cluster_bootstrap(rows, statistic, *, key="family", draws=5000, seed=0):
    """95% CI by resampling whole template families, which is the unit that repeats."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row[key]].append(row)
    names = list(clusters)
    if not names:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        picked = [clusters[rng.choice(names)] for _ in names]
        flat = [row for group in picked for row in group]
        samples.append(statistic(flat))
    samples.sort()
    low = samples[int(0.025 * len(samples))]
    high = samples[int(0.975 * len(samples)) - 1]
    return (low, high)


def paired_difference(rows_a, rows_b, *, key="family", draws=5000, seed=0):
    """D - C style comparison on the same tasks, resampled by family together."""
    index_a = {r["task_id"]: r for r in rows_a}
    index_b = {r["task_id"]: r for r in rows_b}
    shared = sorted(set(index_a) & set(index_b))
    if not shared:
        return None
    clusters = defaultdict(list)
    for task_id in shared:
        clusters[index_a[task_id]["family"]].append(task_id)
    names = list(clusters)

    def difference(task_ids):
        if not task_ids:
            return 0.0
        return statistics.mean(
            float(index_b[t]["success"]) - float(index_a[t]["success"]) for t in task_ids
        )

    point = difference(shared)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        picked = [t for _ in names for t in clusters[rng.choice(names)]]
        samples.append(difference(picked))
    samples.sort()
    return {
        "n_tasks": len(shared),
        "difference": point,
        "ci95": (samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]),
        "wins": sum(index_b[t]["success"] and not index_a[t]["success"] for t in shared),
        "losses": sum(index_a[t]["success"] and not index_b[t]["success"] for t in shared),
    }


def stage_totals(rows):
    """Median per-step milliseconds by stage, over every step of every run."""
    stages = defaultdict(list)
    for row in rows:
        for step in row.get("trace", []):
            for name, value in (step.get("stages_ms") or {}).items():
                stages[name].append(value)
            prefill = step.get("prefill") or {}
            if "prefill_ms" in prefill:
                stages["_prefill"].append(prefill["prefill_ms"])
            if "processor_ms" in prefill:
                stages["_processor"].append(prefill["processor_ms"])
            generation = step.get("generation") or {}
            if "total_ms" in generation:
                stages["_generate_total"].append(generation["total_ms"])
    return {name: statistics.median(values) for name, values in stages.items() if values}


def summarise(rows):
    grouped = by_arm(rows)
    report = {"arms": {}}
    for arm in sorted(grouped):
        arm_rows = grouped[arm]
        tiers = defaultdict(list)
        for row in arm_rows:
            tiers[row["tier"]].append(row)
        report["arms"][arm] = {
            "n": len(arm_rows),
            "success_rate": success_rate(arm_rows),
            "success_ci95": cluster_bootstrap(arm_rows, success_rate),
            "false_done": false_done(arm_rows),
            "errors": sum(bool(r.get("error")) for r in arm_rows),
            "median_wall_ms": statistics.median([r["wall_ms"] for r in arm_rows]),
            "median_steps": statistics.median([r["steps"] for r in arm_rows]),
            "by_tier": {
                tier: {"n": len(v), "success_rate": success_rate(v)} for tier, v in sorted(tiers.items())
            },
            "paired_visual": paired_score(arm_rows),
            "stages_ms": stage_totals(arm_rows),
        }
    report["comparisons"] = {}
    for left, right in (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")):
        if left in grouped and right in grouped:
            result = paired_difference(grouped[left], grouped[right])
            if result:
                report["comparisons"][f"{right}-{left}"] = result
    return report


def render(report):
    lines = []
    lines.append(f"{'arm':>4} {'n':>4} {'success':>9} {'95% CI':>18} {'falseDONE':>10} {'med ms':>8} {'steps':>6}")
    for arm, data in report["arms"].items():
        low, high = data["success_ci95"]
        lines.append(
            f"{arm:>4} {data['n']:>4} {data['success_rate']:>8.1%} "
            f"  [{low:>5.1%},{high:>6.1%}] {data['false_done']:>10} "
            f"{data['median_wall_ms']:>8.0f} {data['median_steps']:>6.0f}"
        )
    lines.append("")
    lines.append("by tier (success rate)")
    tiers = sorted({t for d in report["arms"].values() for t in d["by_tier"]})
    lines.append(f"{'arm':>4} " + " ".join(f"{t:>19}" for t in tiers))
    for arm, data in report["arms"].items():
        cells = []
        for tier in tiers:
            entry = data["by_tier"].get(tier)
            cells.append(f"{entry['success_rate']:>18.1%} " if entry else f"{'-':>19}")
        lines.append(f"{arm:>4} " + " ".join(cells))
    lines.append("")
    lines.append("paired visual-necessary (both members of a pair correct)")
    for arm, data in report["arms"].items():
        p = data["paired_visual"]
        lines.append(
            f"{arm:>4}  pairs={p['pairs']}  both={p['both_correct']}  "
            f"exactly_one={p['exactly_one']}  neither={p['neither']}  rate={p['paired_rate']:.1%}"
        )
    if report["comparisons"]:
        lines.append("")
        lines.append("paired differences (cluster bootstrap over families)")
        for name, data in report["comparisons"].items():
            low, high = data["ci95"]
            lines.append(
                f"  {name:>5}: {data['difference']:+.1%}  95% CI [{low:+.1%}, {high:+.1%}]  "
                f"wins={data['wins']} losses={data['losses']} n={data['n_tasks']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    rows, _ = load(sys.argv[1])
    report = summarise(rows)
    print(render(report))
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(json.dumps(report, indent=2))
        print(f"\nwrote {sys.argv[2]}")
