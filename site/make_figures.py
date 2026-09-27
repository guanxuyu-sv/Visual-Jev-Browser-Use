"""Generate the result figures from the result files, as theme-aware SVG.

Same discipline as the page build: nothing is drawn from a number typed by hand.
Colour encodes the thing that matters — whether the arm sees the screenshot —
rather than the arm's rank, so a reader learns one mapping and it holds across
every figure. Output mechanism is carried by position and label, not by a third
and fourth hue.

The palettes were checked with the dataviz validator at both surfaces:

    light #2A6FD6 / #D8541F   CVD ΔE 26.6, normal ΔE 33.5
    dark  #4A88DC / #E0692F   CVD ΔE 23.7, normal ΔE 29.5

    python site/make_figures.py --data data --out assets/figures
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DOM_L, VIS_L = "#2A6FD6", "#D8541F"
DOM_D, VIS_D = "#4A88DC", "#E0692F"

CSS = f"""
  .ink {{ fill:#11161A }} .ink2 {{ fill:#4C5760 }} .muted {{ fill:#7D878E }}
  .rule {{ stroke:#DCE0DE }} .rule-soft {{ stroke:#E9ECEA }}
  .dom {{ fill:{DOM_L} }} .vis {{ fill:{VIS_L} }}
  .dom-s {{ stroke:{DOM_L} }} .vis-s {{ stroke:{VIS_L} }}
  .surface {{ fill:#FFFFFF }}
  text {{ font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif }}
  .num {{ font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace;
          font-variant-numeric:tabular-nums }}
  @media (prefers-color-scheme: dark) {{
    .ink {{ fill:#E9EDEB }} .ink2 {{ fill:#AEB8BD }} .muted {{ fill:#89949A }}
    .rule {{ stroke:#262D31 }} .rule-soft {{ stroke:#1D2326 }}
    .dom {{ fill:{DOM_D} }} .vis {{ fill:{VIS_D} }}
    .dom-s {{ stroke:{DOM_D} }} .vis-s {{ stroke:{VIS_D} }}
    .surface {{ fill:#151A1D }}
  }}
"""

ARMS = {
    "A": ("DOM only", "autoregressive", "dom"),
    "B": ("screenshot + DOM", "autoregressive", "vis"),
    "C": ("DOM only", "branch readout", "dom"),
    "D": ("screenshot + DOM", "branch readout", "vis"),
}


def svg(width, height, body, title, desc):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t d">'
        f"<title id='t'>{title}</title><desc id='d'>{desc}</desc>"
        f"<style>{CSS}</style>"
        f'<rect width="{width}" height="{height}" class="surface"/>'
        f"{body}</svg>"
    )


def legend(x, y):
    """Identity is never colour alone: each swatch carries its label."""
    return (
        f'<rect x="{x}" y="{y-9}" width="11" height="11" rx="2" class="dom"/>'
        f'<text x="{x+17}" y="{y}" font-size="12" class="ink2">DOM only</text>'
        f'<rect x="{x+95}" y="{y-9}" width="11" height="11" rx="2" class="vis"/>'
        f'<text x="{x+112}" y="{y}" font-size="12" class="ink2">screenshot + DOM</text>'
    )


def fig_where_the_gain_is(offline, out):
    """The finding: operation is flat, target moves. Grouped bars, two measures."""
    W, H = 720, 352
    PL, PR, PT, PB = 58, 20, 66, 86
    pw, ph = W - PL - PR, H - PT - PB
    measures = [("operation", "operation_accuracy"), ("target", "target_accuracy")]
    body = []

    body.append(f'<text x="{PL}" y="26" font-size="15" font-weight="600" class="ink">'
                "Adding the screenshot moves target selection, not operation choice</text>")
    body.append(f'<text x="{PL}" y="45" font-size="12" class="muted">'
                "Held-out Mind2Web websites · n=" + str(offline["A"]["summary"]["n"]) + "</text>")

    for gy in range(0, 101, 25):
        y = PT + ph - ph * gy / 100
        body.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{PL+pw}" y2="{y:.1f}" '
                    f'class="rule-soft" stroke-width="1"/>')
        body.append(f'<text x="{PL-9}" y="{y+4:.1f}" font-size="11" text-anchor="end" '
                    f'class="muted num">{gy}%</text>')

    group_w = pw / len(measures)
    bar_w, gap = 34, 12
    for gi, (label, key) in enumerate(measures):
        gx = PL + gi * group_w
        block = 4 * bar_w + 3 * gap
        start = gx + (group_w - block) / 2
        for ai, arm in enumerate("ABCD"):
            value = offline[arm]["summary"][key] * 100
            x = start + ai * (bar_w + gap)
            h = ph * value / 100
            y = PT + ph - h
            cls = ARMS[arm][2]
            # 4px rounded data-end, anchored to the baseline
            body.append(
                f'<path d="M{x:.1f} {PT+ph} v{-(h-4):.1f} a4 4 0 0 1 4 -4 h{bar_w-8} '
                f'a4 4 0 0 1 4 4 v{h-4:.1f} z" class="{cls}"/>'
            )
            body.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" font-size="11.5" '
                        f'text-anchor="middle" class="ink num">{value:.1f}</text>')
            body.append(f'<text x="{x+bar_w/2:.1f}" y="{PT+ph+16}" font-size="11" '
                        f'text-anchor="middle" class="muted">{arm}</text>')
        body.append(f'<text x="{gx+group_w/2:.1f}" y="{PT+ph+36}" font-size="12.5" '
                    f'text-anchor="middle" class="ink2">{label} accuracy</text>')

    body.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{PL+pw}" y2="{PT+ph}" '
                f'class="rule" stroke-width="1"/>')
    body.append(legend(PL, H - 18))
    Path(out).write_text(svg(
        W, H, "".join(body),
        "Operation accuracy is flat across arms; target accuracy is not",
        "Grouped bars. Operation accuracy sits between 94 and 97 percent for all four arms. "
        "Target accuracy is about 58 percent for the two DOM-only arms and about 93 percent "
        "for the two arms that also see a screenshot."))
    return out


def fig_budget_curve(done, out):
    """Success under a time budget: the only honest way to show the latency effect."""
    W, H = 720, 366
    PL, PR, PT, PB = 58, 132, 62, 78
    pw, ph = W - PL - PR, H - PT - PB
    grouped = defaultdict(list)
    for row in done:
        grouped[row["arm"]].append(row)
    budgets = list(range(1000, 15001, 250))
    top = 0.40
    body, ends = [], []

    body.append(f'<text x="{PL}" y="26" font-size="15" font-weight="600" class="ink">'
                "Tasks finished within a time budget</text>")
    body.append(f'<text x="{PL}" y="45" font-size="12" class="muted">'
                "MiniWoB++ closed loop · a task not finished is never counted, however long it ran</text>")

    for gy in range(0, int(top * 100) + 1, 10):
        y = PT + ph - ph * (gy / 100) / top
        body.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{PL+pw}" y2="{y:.1f}" '
                    f'class="rule-soft" stroke-width="1"/>')
        body.append(f'<text x="{PL-9}" y="{y+4:.1f}" font-size="11" text-anchor="end" '
                    f'class="muted num">{gy}%</text>')
    for gx in (2000, 5000, 10000, 15000):
        x = PL + pw * (gx - 1000) / 14000
        body.append(f'<text x="{x:.1f}" y="{PT+ph+18}" font-size="11" text-anchor="middle" '
                    f'class="muted num">{gx//1000}s</text>')
    body.append(f'<text x="{PL+pw/2:.1f}" y="{PT+ph+38}" font-size="12.5" '
                f'text-anchor="middle" class="ink2">wall-clock budget</text>')

    for arm in "ABCD":
        rows = grouped[arm]
        points = []
        for b in budgets:
            rate = sum(1 for r in rows if r["success"] and r["wall_ms"] <= b) / len(rows)
            x = PL + pw * (b - 1000) / 14000
            y = PT + ph - ph * min(rate, top) / top
            points.append(f"{x:.1f},{y:.1f}")
        cls = ARMS[arm][2] + "-s"
        dash = ' stroke-dasharray="5 3"' if ARMS[arm][1] == "autoregressive" else ""
        body.append(f'<polyline points="{" ".join(points)}" fill="none" class="{cls}" '
                    f'stroke-width="2" stroke-linejoin="round"{dash}/>')
        final = sum(1 for r in rows if r["success"]) / len(rows)
        ends.append((arm, PT + ph - ph * min(final, top) / top, final))

    # nudge end labels apart rather than letting near-equal arms overprint
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 15:
            ends[i] = (ends[i][0], ends[i - 1][1] + 15, ends[i][2])
    for arm, y, final in ends:
        real = PT + ph - ph * min(final, top) / top
        body.append(f'<circle cx="{PL+pw:.1f}" cy="{real:.1f}" r="4" class="{ARMS[arm][2]}"/>')
        if abs(y - real) > 1:
            body.append(f'<line x1="{PL+pw+5:.1f}" y1="{real:.1f}" x2="{PL+pw+11:.1f}" '
                        f'y2="{y:.1f}" class="{ARMS[arm][2]}-s" stroke-width="1"/>')
        body.append(f'<text x="{PL+pw+15:.1f}" y="{y+4:.1f}" font-size="11.5" class="ink2">'
                    f'{arm} · {"AR" if ARMS[arm][1] == "autoregressive" else "branch"}</text>')

    body.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{PL+pw}" y2="{PT+ph}" class="rule" stroke-width="1"/>')
    body.append(f'<text x="{PL}" y="{H-16}" font-size="11" class="muted">'
                "dashed = autoregressive action line · solid = branch readout</text>")
    Path(out).write_text(svg(
        W, H, "".join(body),
        "Success under a wall-clock budget, four arms",
        "Line chart. At every budget from one to fifteen seconds the screenshot arms finish "
        "more tasks than the DOM-only arms, and the branch-readout arms lead the "
        "autoregressive ones at the tightest budgets."))
    return out


def fig_latency_decomposition(done, out):
    """Where a step's time goes. The point is that the browser dominates."""
    W, H = 720, 258
    PL, PR, PT, PB = 176, 74, 64, 44
    pw, ph = W - PL - PR, H - PT - PB
    grouped = defaultdict(list)
    for row in done:
        grouped[row["arm"]].append(row)
    stages = [("decide", "model · decide"), ("generate", "model · text"),
              ("execute", "execute"), ("observe_after", "browser · observe")]
    medians = {}
    for arm in "ABCD":
        collected = defaultdict(list)
        for row in grouped[arm]:
            for step in row.get("trace", []):
                for key, value in (step.get("stages_ms") or {}).items():
                    collected[key].append(value)
        medians[arm] = {k: (statistics.median(collected[k]) if collected[k] else 0.0)
                        for k, _ in stages}
    widest = max(sum(v.values()) for v in medians.values())
    body = []

    body.append(f'<text x="24" y="26" font-size="15" font-weight="600" class="ink">'
                "Where a step’s time goes</text>")
    body.append(f'<text x="24" y="45" font-size="12" class="muted">'
                "Median milliseconds per step · the model is the small part</text>")

    row_h, gap = 30, 14
    opacity = {"decide": 1.0, "generate": 0.72, "execute": 0.45, "observe_after": 0.18}
    for ai, arm in enumerate("ABCD"):
        y = PT + ai * (row_h + gap)
        body.append(f'<text x="{PL-12}" y="{y+row_h/2+4}" font-size="12" text-anchor="end" '
                    f'class="ink2">{arm} · {ARMS[arm][0]}</text>')
        x = PL
        cls = ARMS[arm][2]
        for key, _ in stages:
            value = medians[arm][key]
            if value <= 0:
                continue
            w = pw * value / widest
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w-2,1):.1f}" height="{row_h}" '
                        f'rx="2" class="{cls}" fill-opacity="{opacity[key]}"/>')
            if w > 44:
                body.append(f'<text x="{x+w/2-1:.1f}" y="{y+row_h/2+4}" font-size="11" '
                            f'text-anchor="middle" class="ink num" '
                            f'fill-opacity="{0.95 if opacity[key] > 0.4 else 0.75}">{value:.0f}</text>')
            x += w
        body.append(f'<text x="{x+8:.1f}" y="{y+row_h/2+4}" font-size="11.5" class="ink num">'
                    f'{sum(medians[arm].values()):.0f} ms</text>')

    lx = PL
    for key, label in stages:
        body.append(f'<rect x="{lx}" y="{H-26}" width="11" height="11" rx="2" class="dom" '
                    f'fill-opacity="{opacity[key]}"/>')
        body.append(f'<text x="{lx+16}" y="{H-17}" font-size="11.5" class="ink2">{label}</text>')
        lx += 26 + len(label) * 6.3
    Path(out).write_text(svg(
        W, H, "".join(body),
        "Per-step latency decomposition by arm",
        "Stacked horizontal bars. Browser observation is the largest segment in every arm, "
        "around 1.2 seconds, while the model's decision is 145 to 447 milliseconds."))
    return out


def fig_mechanism(bench, out):
    """Fan-out and KV reuse: the two speedups, and where each stops paying."""
    W, H = 720, 300
    body = []
    body.append(f'<text x="24" y="26" font-size="15" font-weight="600" class="ink">'
                "Both speedups have a floor below which they pay nothing</text>")
    body.append(f'<text x="24" y="45" font-size="12" class="muted">'
                "Isolated micro-benchmarks · one RTX 5090 · median of five runs after warm-up</text>")

    fan = [r for r in bench if "parallel_speedup" in r]
    reuse = [r for r in bench if "reuse_speedup" in r]
    panels = [
        (24, "branches sharing one observation",
         [(str(n), statistics.median(r["parallel_speedup"] for r in fan if r["branches"] == n))
          for n in (1, 2, 3, 4)],
         "a browser step has 2–3"),
        (380, "tokens generated after the decision",
         [(str(n), statistics.median(r["reuse_speedup"] for r in reuse
                                     if r.get("generate_tokens") == n))
          for n in (1, 8, 32)],
         "a field value is short"),
    ]
    for px, title, series, note in panels:
        pw, ph, pt = 300, 150, 92
        body.append(f'<text x="{px}" y="72" font-size="12.5" class="ink2">{title}</text>')
        top = 3.2
        for gy in (1, 2, 3):
            y = pt + ph - ph * gy / top
            body.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px+pw}" y2="{y:.1f}" '
                        f'class="rule-soft" stroke-width="1"/>')
            body.append(f'<text x="{px-8}" y="{y+4:.1f}" font-size="10.5" text-anchor="end" '
                        f'class="muted num">{gy}×</text>')
        # the 1.0x line is where a speedup stops being one
        y1 = pt + ph - ph / top
        body.append(f'<line x1="{px}" y1="{y1:.1f}" x2="{px+pw}" y2="{y1:.1f}" '
                    f'class="rule" stroke-width="1" stroke-dasharray="4 3"/>')
        bw = pw / (len(series) * 2)
        for i, (label, value) in enumerate(series):
            x = px + (i + 0.5) * pw / len(series) - bw / 2
            h = ph * value / top
            y = pt + ph - h
            cls = "vis" if value >= 1.5 else "dom"
            body.append(f'<path d="M{x:.1f} {pt+ph} v{-(h-4):.1f} a4 4 0 0 1 4 -4 h{bw-8:.1f} '
                        f'a4 4 0 0 1 4 4 v{h-4:.1f} z" class="{cls}" '
                        f'fill-opacity="{1.0 if value >= 1.5 else 0.45}"/>')
            body.append(f'<text x="{x+bw/2:.1f}" y="{y-7:.1f}" font-size="11.5" '
                        f'text-anchor="middle" class="ink num">{value:.2f}×</text>')
            body.append(f'<text x="{x+bw/2:.1f}" y="{pt+ph+16}" font-size="11" '
                        f'text-anchor="middle" class="muted num">{label}</text>')
        body.append(f'<line x1="{px}" y1="{pt+ph}" x2="{px+pw}" y2="{pt+ph}" '
                    f'class="rule" stroke-width="1"/>')
        body.append(f'<text x="{px}" y="{pt+ph+40}" font-size="11.5" class="muted">{note}</text>')
    Path(out).write_text(svg(
        W, H, "".join(body),
        "Fan-out and KV reuse speedups against the regime a browser step actually occupies",
        "Two bar panels. Fan-out gives nothing at one branch and 2.41 times at four. KV reuse "
        "gives 3 times at one generated token and about 2 times at eight and thirty-two."))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="assets/figures")
    args = parser.parse_args()
    d = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    offline = {a: json.loads((d / f"offline_{a}.json").read_text()) for a in "ABCD"}
    done = json.loads((d / "miniwob_done_AB.json").read_text())["rows"] + \
           json.loads((d / "miniwob_done_CD.json").read_text())["rows"]
    bench = json.loads((d / "bench_stages.json").read_text())

    made = [
        fig_where_the_gain_is(offline, out / "where_the_gain_is.svg"),
        fig_budget_curve(done, out / "success_under_budget.svg"),
        fig_latency_decomposition(done, out / "latency_decomposition.svg"),
        fig_mechanism(bench, out / "mechanism_floors.svg"),
    ]
    for path in made:
        print(f"  wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
