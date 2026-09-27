"""Element table and numbered screenshot, built from one upstream observation.

Upstream already returns per-action geometry: `snapshot.js` puts `rect` on every
action and strips it only from the `semantics` copy that feeds the staleness
marker, so that layout jitter does not invalidate a page. We keep `rect` on the
elements and leave the marker alone.

Numbering is drawn on an offline copy of the JPEG. The live page is never
modified, so the DOM the executor acts on is the DOM that was observed.
"""

import base64
import io

from PIL import Image, ImageDraw, ImageFont

OPERATIONS = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}


def action_space(actions):
    """Upstream's grouping, with geometry kept.

    Returns (elements, targets, controls) exactly as upstream does, so the rest of
    the agent loop is unchanged, except each element also carries `rect`.
    """
    elements, indices, targets, controls = [], {}, {}, {}
    for action in actions:
        kind = action["kind"]
        if kind not in OPERATIONS:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if action.get("rect"):
                element["rect"] = action["rect"]
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = OPERATIONS[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(screenshot, elements, *, scale=1.0, badge_height=15):
    """Draw each element's index beside its box, on a copy of the screenshot.

    Badges are placed just outside the element's top-left corner where there is
    room, so they cover page chrome rather than the label the model needs to read.
    Returns a PIL image; `screenshot` may be base64 (as upstream returns) or bytes.
    """
    if isinstance(screenshot, str):
        screenshot = base64.b64decode(screenshot)
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(badge_height - 3)

    taken = []
    for element in elements:
        rect = element.get("rect")
        if not rect:
            continue
        x, y = rect["x"] * scale, rect["y"] * scale
        w, h = rect["w"] * scale, rect["h"] * scale
        draw.rectangle([x, y, x + w, y + h], outline=(255, 64, 0, 180), width=1)

        label = str(element["index"])
        text_width = draw.textlength(label, font=font) + 6
        # Place the badge INSIDE the element wherever it fits.
        #
        # Putting it above the box, as this did, lands it on whatever sits above
        # the element -- and for a control whose own label is empty, that is
        # exactly the heading that gives the control its meaning. Measured on the
        # shipping-options page, where three radios share the accessible name
        # "Select" and the row heading is the only thing telling them apart:
        # above-the-box numbering moved arm D from 0.93 on the right target to
        # 0.98 on the wrong one, because each number sat under the previous row's
        # heading and bound itself to that instead. Inside the box, the number can
        # only ever be read as belonging to the element it marks.
        if h >= badge_height and w >= text_width:
            spot = (x, y)
        elif y >= badge_height:  # too small to hold a badge: sit just outside, to the LEFT
            spot = (max(0, x - text_width - 2), y)
        else:
            spot = (x, y + h)
        spot = (min(spot[0], image.width - text_width), max(0, min(spot[1], image.height - badge_height)))
        for _ in range(8):  # nudge while it would sit on an earlier badge
            box = (spot[0], spot[1], spot[0] + text_width, spot[1] + badge_height)
            if not any(_overlaps(box, other) for other in taken):
                break
            spot = (spot[0] + text_width + 1, spot[1])
        box = (spot[0], spot[1], spot[0] + text_width, spot[1] + badge_height)
        taken.append(box)
        draw.rectangle(box, fill=(255, 64, 0, 235))
        draw.text((spot[0] + 3, spot[1] + 1), label, fill=(255, 255, 255, 255), font=font)
    return image


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def plain(screenshot, scale=1.0):
    """The same screenshot without numbering, for the numbering ablation."""
    if isinstance(screenshot, str):
        screenshot = base64.b64decode(screenshot)
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def coverage(elements):
    """How many observed elements could actually be drawn. Reported, never assumed."""
    total = len(elements)
    boxed = sum(1 for e in elements if e.get("rect"))
    return {"elements": total, "with_box": boxed, "box_coverage": boxed / total if total else 0.0}
