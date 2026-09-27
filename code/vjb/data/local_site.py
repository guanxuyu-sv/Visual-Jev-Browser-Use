"""Deterministic local pages: a DOM-sufficient tier, a visual-helpful tier, and
paired visual-necessary items where only the image decides the answer.

The paired construction is the point (plan 7.3). Two pages share the DOM exactly
-- same labels, same roles, same option text -- and differ only in which card
carries which picture. If a model answers both correctly it must have used the
image; if it answers one correctly and its pair wrong, it is following a DOM or
position shortcut. Card order, file names and element numbering are randomised
per page from the item seed, so nothing outside the picture identifies the target.

Everything is generated, so the pages can be rebuilt byte-identically from the
manifest and served from disk with no network.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

SHAPES = ("circle", "square", "triangle", "star", "hexagon")
COLORS = {
    "red": "#d92b2b",
    "blue": "#1f5fd0",
    "green": "#1d8a43",
    "yellow": "#e0a800",
    "purple": "#7b3fbf",
}

STYLE = """
body{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#fbfbfd;color:#16161a}
h1{font-size:20px;margin:0 0 4px}
p.sub{margin:0 0 20px;color:#5a5a68;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(4,150px);gap:16px}
.card{border:1px solid #d8d8e0;border-radius:8px;padding:10px;background:#fff;text-align:center}
.card svg{display:block;margin:0 auto 8px}
.card .name{font-size:13px;margin-bottom:8px;color:#333}
button{font:inherit;padding:6px 12px;border:1px solid #b9b9c6;border-radius:6px;background:#f2f2f6;cursor:pointer}
form{display:grid;gap:12px;max-width:420px}
label{display:block;font-size:13px;margin-bottom:4px;color:#44444f}
input,select{font:inherit;padding:6px 8px;border:1px solid #b9b9c6;border-radius:6px;width:100%;box-sizing:border-box}
.row{display:flex;gap:12px}.row>div{flex:1}
.bar{display:flex;gap:8px;margin-bottom:18px}
"""


def _svg(shape, color, size=64):
    fill = COLORS[color]
    half = size / 2
    body = {
        "circle": f'<circle cx="{half}" cy="{half}" r="{half - 4}" fill="{fill}"/>',
        "square": f'<rect x="6" y="6" width="{size - 12}" height="{size - 12}" rx="4" fill="{fill}"/>',
        "triangle": f'<polygon points="{half},6 {size - 6},{size - 8} 6,{size - 8}" fill="{fill}"/>',
        "star": (
            f'<polygon points="{half},4 {half + 12},{half - 6} {size - 6},{half - 4} '
            f'{half + 8},{half + 8} {half + 14},{size - 6} {half},{half + 14} '
            f'{half - 14},{size - 6} {half - 8},{half + 8} 6,{half - 4} {half - 12},{half - 6}" fill="{fill}"/>'
        ),
        "hexagon": (
            f'<polygon points="{half},5 {size - 7},{half / 2 + 6} {size - 7},{size - half / 2 - 6} '
            f'{half},{size - 5} 7,{size - half / 2 - 6} 7,{half / 2 + 6}" fill="{fill}"/>'
        ),
    }[shape]
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{body}</svg>'


@dataclass
class Task:
    task_id: str
    family: str          # template family, for split isolation
    tier: str            # dom_sufficient | visual_helpful | visual_necessary
    pair_id: str | None  # paired items share this; their answers differ
    page: str            # html file name
    goal: str
    success: dict        # how the environment verifies success, checked in the DOM
    gold_label: str      # the element label a correct run must act on


def _page(title, subtitle, body, success_js=""):
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>{STYLE}</style></head><body>
<h1>{title}</h1><p class="sub">{subtitle}</p>
{body}
<script>
window.__vjb = {{done:false, picked:null, fields:{{}} }};
{success_js}
</script></body></html>"""


def build_visual_necessary(seed, colour_shape_pool, count=4):
    """Two pages with identical DOM and swapped pictures.

    Card names are neutral ("Item K-3"): the DOM cannot say which card is the red
    circle. Only the SVG does.
    """
    rng = random.Random(seed)
    picks = rng.sample(colour_shape_pool, count)
    names = [f"Item {chr(65 + rng.randrange(26))}-{rng.randrange(10, 99)}" for _ in range(count)]
    while len(set(names)) < count:  # names must not collide, or the goal is ambiguous
        names = [f"Item {chr(65 + rng.randrange(26))}-{rng.randrange(10, 99)}" for _ in range(count)]

    target_color, target_shape = picks[0]
    goal = f"Add the {target_color} {target_shape} to the cart."

    # One card order for the whole pair. Both pages then have byte-identical DOM --
    # same names, same order, same button text -- and differ only in which card
    # carries which picture, so the answer moves for a reason nothing but the
    # image can express.
    order = list(range(count))
    rng.shuffle(order)

    pages = []
    for variant, rotation in enumerate((0, 1)):
        assignment = picks[rotation:] + picks[:rotation]
        cards = []
        gold_label = None
        for slot in order:
            color, shape = assignment[slot]
            name = names[slot]
            label = f"Add {name}"
            if (color, shape) == (target_color, target_shape):
                gold_label = label
            cards.append(
                f'<div class="card">{_svg(shape, color)}'
                f'<div class="name">{name}</div>'
                f'<button onclick="window.__vjb.picked=\'{name}\'">{label}</button></div>'
            )
        body = '<div class="grid">' + "".join(cards) + "</div>"
        pages.append((body, gold_label, goal, names[rotation if rotation else 0]))
    return pages, names, picks


def build_dom_sufficient(seed):
    """A form whose labels fully determine the answer. The check is for regressions:
    adding a screenshot must not make these worse."""
    rng = random.Random(seed)
    city = rng.choice(["Zurich", "Berlin", "Lisbon", "Oslo", "Porto", "Dublin"])
    cabin = rng.choice(["Economy", "Business"])
    body = f"""
<form>
  <div class="row">
    <div><label for="from">Departure city</label><input id="from" name="from" placeholder="City"></div>
    <div><label for="to">Destination city</label><input id="to" name="to" placeholder="City"></div>
  </div>
  <div><label for="cabin">Cabin class</label>
    <select id="cabin"><option>Economy</option><option>Business</option><option>First</option></select></div>
  <button type="button" id="search" onclick="window.__vjb.done=true">Search flights</button>
</form>"""
    goal = f"Set the destination city to {city} and the cabin class to {cabin}, then search."
    return body, goal, city, cabin


def build_visual_helpful(seed):
    """Repeated controls with identical accessible names, told apart by position/state."""
    rng = random.Random(seed)
    rows = ["Standard delivery", "Express delivery", "Pickup in store"]
    which = rng.randrange(len(rows))
    cells = []
    for index, row in enumerate(rows):
        checked = " checked" if index == 0 else ""
        cells.append(
            f'<div class="card" style="text-align:left;width:320px">'
            f'<div class="name"><b>{row}</b></div>'
            f'<label><input type="radio" name="ship" value="{index}"{checked}> Select</label></div>'
        )
    body = '<div style="display:grid;gap:10px">' + "".join(cells) + "</div>"
    goal = f"Choose {rows[which]} as the shipping option."
    return body, goal, rows[which], which


def generate(out_dir, *, seeds=range(1, 6)):
    """Write the pages and a manifest. Returns the task list."""
    out = Path(out_dir)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    pool = [(c, s) for c in COLORS for s in SHAPES]
    tasks = []

    for seed in seeds:
        family = f"grid{seed}"
        pages, names, _ = build_visual_necessary(f"vn-{seed}", pool)
        pair_id = f"pair-{seed}"
        for variant, (body, gold_label, goal, _) in enumerate(pages):
            name = f"visual_necessary_{seed}_{variant}.html"
            (out / "pages" / name).write_text(
                _page("Shop", "Pick the item that matches the description.", body)
            )
            tasks.append(
                Task(
                    task_id=f"vn-{seed}-{variant}",
                    family=family,
                    tier="visual_necessary",
                    pair_id=pair_id,
                    page=name,
                    goal=goal,
                    success={"kind": "js", "expression": "window.__vjb.picked", "equals": gold_label[4:]},
                    gold_label=gold_label,
                )
            )

        body, goal, city, cabin = build_dom_sufficient(f"ds-{seed}")
        name = f"dom_sufficient_{seed}.html"
        (out / "pages" / name).write_text(_page("Flights", "Complete the search.", body))
        tasks.append(
            Task(
                task_id=f"ds-{seed}",
                family=f"form{seed}",
                tier="dom_sufficient",
                pair_id=None,
                page=name,
                goal=goal,
                success={
                    "kind": "js",
                    "expression": (
                        "[document.getElementById('to').value,"
                        "document.getElementById('cabin').value,window.__vjb.done].join('|')"
                    ),
                    "equals": f"{city}|{cabin}|true",
                },
                gold_label="Search flights",
            )
        )

        body, goal, row, which = build_visual_helpful(f"vh-{seed}")
        name = f"visual_helpful_{seed}.html"
        (out / "pages" / name).write_text(_page("Checkout", "Choose how it ships.", body))
        tasks.append(
            Task(
                task_id=f"vh-{seed}",
                family=f"ship{seed}",
                tier="visual_helpful",
                pair_id=None,
                page=name,
                goal=goal,
                success={
                    "kind": "js",
                    "expression": "document.querySelector('input[name=ship]:checked').value",
                    "equals": str(which),
                },
                gold_label=row,
            )
        )

    manifest = {
        "tasks": [asdict(t) for t in tasks],
        "tiers": sorted({t.tier for t in tasks}),
        "families": sorted({t.family for t in tasks}),
        "hash": hashlib.sha256(
            json.dumps([asdict(t) for t in tasks], sort_keys=True).encode()
        ).hexdigest()[:16],
    }
    (out / "tasks.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "work/local_site"
    result = generate(target)
    print(f"{len(result['tasks'])} tasks, hash {result['hash']}, tiers {result['tiers']}")
