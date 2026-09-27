"""Multimodal-Mind2Web -> the state/action supervision this model trains on.

Two rules from the plan govern this file, and both are about not inventing labels:

  * a trajectory records the operation that was taken and nothing else. If the
    step was a CLICK, only `operation` and `click_target` are supervised; the
    other target branches are masked. Writing a counterfactual TYPE_TEXT target
    because a text field happened to be on the page would be a fabricated label.
  * the candidate symbol for the gold target is assigned after the candidate
    order is permuted, so the label follows the shuffle instead of pinning the
    answer to a slot.

Splits are by website and by domain, which is how the official release already
separates them; the train portion alone is used here, and the held-out portions
are never read during training or tuning.
"""

import base64
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

OPERATION_MAP = {"CLICK": "CLICK", "TYPE": "TYPE_TEXT", "SELECT": "SELECT"}


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]


def _loads(value, default=None):
    """The release stores operation and candidates as JSON strings inside the row."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _candidate_list(raw):
    parsed = []
    for entry in raw or []:
        item = _loads(entry)
        if isinstance(item, dict):
            item["attributes"] = _loads(item.get("attributes"), {}) or {}
            parsed.append(item)
    return parsed


def candidate_elements(step, limit=24):
    """The positive and negative candidates the release already stores per step.

    Mind2Web ships a ranked candidate list per step (`pos_candidates` and
    `neg_candidates`). We keep the positives and fill up to `limit` with the
    highest-ranked negatives, which is the candidate set the model will see.
    Returns (candidates, gold_index) or (None, None) when the gold candidate is
    absent, which is a recall failure and must be reported, not patched.
    """
    positives = _candidate_list(step.get("pos_candidates"))
    negatives = _candidate_list(step.get("neg_candidates"))
    if not positives:
        return None, None
    gold = positives[0]
    gold_node = gold["attributes"].get("backend_node_id")
    pool = [gold] + [c for c in negatives if c["attributes"].get("backend_node_id") != gold_node]
    pool = pool[:limit]
    return pool, 0


def describe(candidate):
    """One line per candidate, from the stored attributes only."""
    attributes = candidate.get("attributes") or {}
    label = (
        attributes.get("label")
        or attributes.get("aria_label")
        or attributes.get("alt")
        or attributes.get("title")
        or attributes.get("value")
        or attributes.get("placeholder")
        or (candidate.get("text") or "").strip()
    )
    box = None
    rect = attributes.get("bounding_box_rect")
    if rect and rect != "-1,-1,-1,-1":
        try:
            x, y, w, h = (float(v) for v in str(rect).split(","))
            box = {"x": x, "y": y, "w": w, "h": h}
        except ValueError:
            box = None
    return {
        "tag": candidate.get("tag") or attributes.get("role") or "?",
        "label": (label or "").strip()[:120],
        "value": (attributes.get("value") or "").strip()[:80],
        "node": attributes.get("backend_node_id"),
        "rect": box,
    }


def _history(step, limit=10):
    actions = step.get("action_reprs") or []
    try:
        index = int(step.get("target_action_index"))
    except (TypeError, ValueError):
        return []
    return list(actions[max(0, index - limit) : index])


def convert_step(step, task, *, rng, limit=24):
    """One training example, or None when the step cannot be supervised honestly."""
    action = _loads(step.get("operation"), {}) or {}
    op = OPERATION_MAP.get(action.get("op"))
    if op is None:
        return None, "unsupported_operation"

    candidates, gold_index = candidate_elements(step, limit=limit)
    if candidates is None:
        return None, "gold_candidate_missing"

    order = list(range(len(candidates)))
    rng.shuffle(order)
    shuffled = [candidates[i] for i in order]
    gold_position = order.index(gold_index)

    example = {
        "task_id": task.get("annotation_id"),
        "website": task.get("website"),
        "domain": task.get("domain"),
        "subdomain": task.get("subdomain"),
        "step_id": step.get("action_uid"),
        "goal": task.get("confirmed_task"),
        "operation": op,
        # Only the branch that was actually executed carries a label. The rest
        # are masked, because this trajectory says nothing about them.
        "supervised_branches": ["operation", f"{op.lower()}_target"],
        "target_position": gold_position,
        "candidates": [describe(c) for c in shuffled],
        "text": action.get("value") if op == "TYPE_TEXT" else None,
        "screenshot_key": step.get("action_uid"),
        # The release stores the whole trajectory's action strings plus this
        # step's index into it, so history is everything strictly before it.
        "previous_actions": _history(step),
    }
    if op == "TYPE_TEXT" and not (example["text"] or "").strip():
        return None, "type_without_value"
    return example, None


def convert(rows, *, seed=0, limit=24):
    """Convert an iterable of Multimodal-Mind2Web rows. Returns (examples, reasons)."""
    rng = random.Random(seed)
    examples, reasons = [], Counter()
    for row in rows:
        task = {
            "annotation_id": row.get("annotation_id"),
            "website": row.get("website"),
            "domain": row.get("domain"),
            "subdomain": row.get("subdomain"),
            "confirmed_task": row.get("confirmed_task"),
        }
        example, reason = convert_step(row, task, rng=rng, limit=limit)
        if example is None:
            reasons[reason] += 1
            continue
        examples.append(example)
        reasons["kept"] += 1
    return examples, reasons


def save_screenshot(row, out_dir):
    """Write the step screenshot, if the row carries one. Returns the path or None."""
    image = row.get("screenshot")
    if image is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{row.get('action_uid') or _hash(row)}.jpg"
    path = out_dir / name
    if isinstance(image, dict) and "bytes" in image:
        path.write_bytes(image["bytes"])
    elif isinstance(image, (bytes, bytearray)):
        path.write_bytes(bytes(image))
    elif isinstance(image, str):
        path.write_bytes(base64.b64decode(image))
    else:  # a PIL image from datasets' Image feature
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        path.write_bytes(buffer.getvalue())
    return str(path)


def split_manifest(examples):
    """Group by website, then domain. Splitting anywhere finer leaks templates."""
    by_website = Counter(e["website"] for e in examples)
    by_domain = Counter(e["domain"] for e in examples)
    return {
        "examples": len(examples),
        "websites": len(by_website),
        "domains": len(by_domain),
        "operations": Counter(e["operation"] for e in examples),
        "candidate_sizes": Counter(len(e["candidates"]) for e in examples),
        "per_domain": dict(by_domain.most_common()),
    }
