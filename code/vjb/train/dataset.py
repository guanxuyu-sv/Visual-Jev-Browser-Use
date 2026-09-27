"""Turn converted Mind2Web steps into the exact sequences the model is read from.

A training sequence is built the same way inference builds one: the shared
observation prefix, then one branch suffix, then the single answer token. The
loss is taken on that answer token only, so what is trained is precisely what is
read at inference time.

One example yields several sequences:

  * `operation` — always
  * `<op>_target` — only for the operation that was actually executed
  * the text continuation — only when the operation was TYPE_TEXT

The target branches for operations the trajectory did not take are not emitted.
Mind2Web records one action per step and says nothing about what a different
operation would have targeted; emitting those would be inventing labels.
"""

import io
import json
import random
from pathlib import Path

from PIL import Image

from ..browser.observe import annotate
from ..model import prompts
from ..model.symbols import CandidateOverflow

OPERATIONS = ["CLICK", "TYPE_TEXT", "SELECT", "DONE", "BLOCKED"]


def _encode(image):
    """annotate() takes encoded bytes, as the browser hands them over."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


class StepSequences:
    """Builds the per-branch sequences for one converted step."""

    def __init__(self, pool, *, image_dropout=0.0, seed=0, visual_token_budget=1024,
                 viewport_size=(1120, 780), band_margin=120, patch=16, merge=2,
                 min_candidates=4, numbered=True):
        self.pool = pool
        self.image_dropout = image_dropout
        self.seed = seed
        self.rng = random.Random(seed)
        self.visual_token_budget = visual_token_budget
        self.viewport_size = viewport_size
        self.band_margin = band_margin
        self.patch = patch
        self.merge = merge
        self.min_candidates = min_candidates
        # Training images must carry the same overlay inference draws. They did
        # not, and arm D had never seen a numbered screenshot before the closed
        # loop handed it one: on the shipping page it then put 0.99 on the wrong
        # radio, where the same checkpoint on the same page put 0.93 on the right
        # one from a plain screenshot.
        self.numbered = numbered
        self.stats = {"viewport_ok": 0, "viewport_fallback": 0}

    def _page(self, example):
        return {
            "url": example.get("url", ""),
            "title": example.get("website", ""),
            "text": "",  # the release gives cleaned HTML, not rendered text
        }

    def _elements(self, example):
        elements = []
        for index, candidate in enumerate(example["candidates"], start=1):
            element = {
                "index": str(index),
                "role": candidate.get("tag") or "?",
                "label": candidate.get("label") or "",
                "operations": [],
            }
            if candidate.get("value"):
                element["value"] = candidate["value"]
            if candidate.get("rect"):
                element["rect"] = candidate["rect"]
            elements.append(element)
        return elements

    def window(self, example, page_height):
        """The crop window and surviving candidates, from geometry alone.

        Separated from `viewport` so a corpus can be filtered without decoding
        6454 JPEGs -- several of them 1297x8965 -- which costs more than the
        training step it precedes. PIL reports `size` without decoding, so the
        page height is all this needs.
        """
        boxes = [c.get("rect") for c in example["candidates"]]
        if not boxes:
            return None
        gold_box = boxes[example["target_position"]]
        if not gold_box:
            return None
        _, height = self.viewport_size
        rng = random.Random(f"{self.seed}-{example.get('step_id')}")
        span = max(0, height - gold_box["h"])
        jitter = rng.randint(0, int(span)) if span > 1 else 0
        top = gold_box["y"] - jitter
        top = max(0, min(top, max(0, page_height - height)))
        bottom = min(page_height, top + height)

        keep = [
            i
            for i, box in enumerate(boxes)
            if box and box["y"] >= top - 1 and box["y"] + box["h"] <= bottom + 1
        ]
        if example["target_position"] not in keep or len(keep) < self.min_candidates:
            return None
        return {"top": int(top), "bottom": int(bottom), "keep": keep}

    def feasible(self, example):
        """True when this step reduces to a viewport. Does not decode the image."""
        path = example.get("screenshot")
        if not path:
            return False
        try:
            with Image.open(path) as image:
                height = image.height
        except (OSError, ValueError):
            return False
        return self.window(example, height) is not None

    def viewport(self, example):
        """Restrict an offline step to one viewport, the way the live agent sees a page.

        Mind2Web candidates span the whole page, but the executor's element table
        only ever contains what is currently in view, and the screenshot it hands
        the model is a 1120x780 viewport at 1:1 scale. Training on a whole page
        squeezed to fit a token budget would show the model a 39%-scale rendering
        in which the small text it is supposed to read is gone.

        So a window is chosen that contains the gold element, jittered so the gold
        is not reliably centred, and the candidate list is filtered to that window
        and renumbered. Choosing the window around the gold mirrors the recorded
        trajectory, where the person had scrolled the target into view before
        acting; the gold is not marked within the window, and jitter plus the
        surviving negatives are what stop position from becoming the answer.

        Returns (example, image) with the crop applied, or (example, None) when
        the window would leave too few candidates to be a real choice.
        """
        path = example.get("screenshot")
        if not path or not Path(path).exists():
            return example, None
        image = Image.open(path)
        window = self.window(example, image.height)
        if window is None:
            return example, None
        top, bottom, keep = window["top"], window["bottom"], window["keep"]
        # The page was rendered at its own width; cropping narrower than that
        # would cut off the right-hand column and with it, often, the gold.
        left, right = 0, image.width
        image = image.convert("RGB")

        shifted = []
        for index in keep:
            candidate = dict(example["candidates"][index])
            box = candidate["rect"]
            candidate["rect"] = {"x": box["x"] - left, "y": box["y"] - top, "w": box["w"], "h": box["h"]}
            shifted.append(candidate)
        cropped = dict(example)
        cropped["candidates"] = shifted
        cropped["target_position"] = keep.index(example["target_position"])
        cropped["viewport_crop"] = {"top": int(top), "kept": len(keep), "of": len(example["candidates"])}
        view = image.crop((left, int(top), right, int(bottom)))
        if self.numbered:
            view = annotate(_encode(view), self._elements(cropped))
        return cropped, self._budget(view)

    def build_compact(self, example, *, with_image=True):
        """Arms A and B: one line, `OPERATION target [text]`, trained autoregressively.

        Same observation, same candidate list, same legality; only the way the
        answer is expressed differs from `build`. Keeping both on one corpus and
        one step budget is what makes A/B and C/D comparable at all.
        """
        elements = self._elements(example)
        operation = example["operation"]
        use_image = with_image and bool(example.get("screenshot"))
        if use_image and self.image_dropout and self.rng.random() < self.image_dropout:
            use_image = False

        prefix = prompts.shared_prefix(
            example["goal"],
            self._page(example),
            elements,
            [{"action": a} for a in example.get("previous_actions", [])],
            use_screenshot=use_image,
        )
        keys = [str(i) for i in range(1, len(elements) + 1)]
        # Every operation is on offer, and every candidate is a target for each of
        # them -- the same choice the branch arms face. Listing only the executed
        # operation would hand the answer over: the arm would only have to avoid
        # saying DONE, and its operation accuracy would measure nothing.
        listing = "\n".join(
            f"{name} targets: " + ", ".join(keys) for name in OPERATIONS if name not in ("DONE", "BLOCKED")
        )
        suffix = (
            "Legal operations: " + ", ".join(OPERATIONS) + "\n"
            + listing + "\n" + prompts.compact_action_branch()
        )
        gold = f"{operation} {example['target_position'] + 1}"
        if operation == "TYPE_TEXT" and example.get("text"):
            gold += f" {example['text']}"
        return [
            {
                "kind": "compact",
                "prefix": prefix,
                "suffix": suffix,
                "answer_text": gold,
                "use_image": use_image,
            }
        ], None

    def build(self, example, *, with_image=True):
        """Returns a list of {prefix, suffix, answer_token_or_text, kind, weight}."""
        elements = self._elements(example)
        operation = example["operation"]
        keys = [str(i) for i in range(1, len(elements) + 1)]

        use_image = with_image and bool(example.get("screenshot"))
        if use_image and self.image_dropout and self.rng.random() < self.image_dropout:
            use_image = False

        prefix = prompts.shared_prefix(
            example["goal"],
            self._page(example),
            elements,
            [{"action": a} for a in example.get("previous_actions", [])],
            use_screenshot=use_image,
        )

        sequences = []

        # operation branch
        operation_map, by_name = self.pool.assign(OPERATIONS)
        options = [(symbol, name) for symbol, name in operation_map.items()]
        sequences.append(
            {
                "kind": "operation",
                "prefix": prefix,
                "suffix": prompts.operation_branch(options),
                "answer_symbol": by_name[operation],
                "use_image": use_image,
            }
        )

        # target branch, only for the executed operation
        try:
            target_map, by_key = self.pool.assign(keys)
        except CandidateOverflow:
            return sequences, "candidate_overflow"
        gold_key = str(example["target_position"] + 1)
        target_options = [
            (symbol, f"[{key}] {elements[int(key) - 1]['label']!r} role={elements[int(key) - 1]['role']}")
            for symbol, key in target_map.items()
        ]
        sequences.append(
            {
                "kind": f"{operation.lower()}_target",
                "prefix": prefix,
                "suffix": prompts.target_branch(operation, target_options),
                "answer_symbol": by_key[gold_key],
                "use_image": use_image,
            }
        )

        # text continuation, only for TYPE_TEXT
        if operation == "TYPE_TEXT" and example.get("text"):
            gold_element = elements[example["target_position"]]
            sequences.append(
                {
                    "kind": "text",
                    "prefix": prefix,
                    "suffix": prompts.text_branch("TYPE_TEXT", gold_element["label"], None),
                    "answer_text": example["text"],
                    "use_image": use_image,
                }
            )
        return sequences, None

    def image_for(self, example):
        """The observation image, cropped to a viewport-like band and size-budgeted.

        Mind2Web stores whole-page screenshots: 1280x4977 and 1280x8965 are normal,
        which the processor turns into 2600-11480 visual tokens. Two things are
        wrong with feeding that directly. It costs more than the rest of the
        sequence combined, and at inference the browser hands us a 1120x780
        viewport worth about 840 tokens -- so training on whole pages and testing
        on viewports is a mismatch in the observation itself.

        The crop is the vertical band spanned by **all** candidates, not the gold
        one. The element table already lists exactly those candidates, so the band
        discloses nothing the text does not, while the gold's position within it
        stays unmarked. The band is then padded out to at least a viewport height
        and scaled down until it fits the visual-token budget.
        """
        path = example.get("screenshot")
        if not path or not Path(path).exists():
            return None
        image = Image.open(path).convert("RGB")
        boxes = [c["rect"] for c in example["candidates"] if c.get("rect")]
        if boxes:
            top = max(0, min(b["y"] for b in boxes) - self.band_margin)
            bottom = min(image.height, max(b["y"] + b["h"] for b in boxes) + self.band_margin)
            height = max(bottom - top, self.viewport_size[1])
            # Keep the band inside the page after padding it out.
            top = max(0, min(top, image.height - height))
            bottom = min(image.height, top + height)
            if bottom - top >= 32:
                image = image.crop((0, int(top), image.width, int(bottom)))
        return self._budget(image)

    def _budget(self, image):
        """Scale until the processor would emit at most `visual_token_budget` tokens."""
        if not self.visual_token_budget:
            return image
        # Qwen packs (patch*merge)^2 pixels per visual token.
        per_token = (self.patch * self.merge) ** 2
        tokens = (image.width * image.height) / per_token
        if tokens <= self.visual_token_budget:
            return image
        scale = (self.visual_token_budget / tokens) ** 0.5
        width = max(self.patch * self.merge, int(image.width * scale))
        height = max(self.patch * self.merge, int(image.height * scale))
        return image.resize((width, height), Image.LANCZOS)
