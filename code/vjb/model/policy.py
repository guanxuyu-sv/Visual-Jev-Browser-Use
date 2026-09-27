"""The four experiment arms behind one interface the agent loop can call.

    A  DOM            compact autoregressive action line
    B  screenshot+DOM compact autoregressive action line
    C  DOM            parallel decision branches + conditional text on reused KV
    D  screenshot+DOM parallel decision branches + conditional text on reused KV

Every arm shares this module's candidate construction, legality rules and history
budget, so a difference between arms is a difference of modality or of output
mechanism, never of prompt scaffolding.
"""

import re
import time

from ..browser.observe import action_space, annotate, plain
from . import prompts
from .symbols import CandidateOverflow, verify_pool, verify_words

CONTROL_WORDS = ("DONE", "BLOCKED", "WAIT", "SCROLL_DOWN", "SCROLL_UP")


class Policy:
    def __init__(
        self,
        backend,
        *,
        use_screenshot=True,
        parallel_branches=True,
        reuse_kv=True,
        numbered=True,
        with_boxes=False,
        image_scale=1.0,
        arm="D",
    ):
        self.backend = backend
        self.use_screenshot = use_screenshot
        self.parallel_branches = parallel_branches
        self.reuse_kv = reuse_kv
        self.numbered = numbered
        self.with_boxes = with_boxes
        self.image_scale = image_scale
        self.arm = arm
        self.answer_context = "Answer:"
        self.pool = verify_pool(backend.tokenizer, self.answer_context)
        self.operation_ids, dropped = verify_words(backend.tokenizer, self.answer_context, CONTROL_WORDS)
        self.dropped_operation_words = dropped

    # ------------------------------------------------------------- candidates

    def _operation_options(self, targets, controls):
        """Legal operations for this observation, each mapped to a verified symbol.

        Operations are read as symbols, not as words, so that a long word like
        SCROLL_DOWN cannot be excluded merely by being several tokens.
        """
        names = [op for op in ("CLICK", "TYPE_TEXT", "SELECT") if targets.get(op)]
        names += [name for name in controls]
        names += ["DONE", "BLOCKED"]
        descriptions = {
            "CLICK": "Click an element, button, menu option, suggestion, or calendar day.",
            "TYPE_TEXT": "Enter or replace text in an editable field.",
            "SELECT": "Choose an observed dropdown value.",
            "DONE": "Every requirement is visibly satisfied.",
            "BLOCKED": "No supported operation can make progress.",
        }
        for name, action in controls.items():
            descriptions.setdefault(name, action.get("label", name))
        mapping, _ = self.pool.assign(names)
        options = [(symbol, f"{name} — {descriptions.get(name, name)}") for symbol, name in mapping.items()]
        return options, mapping

    def _target_options(self, candidates, elements):
        """Targets for one operation. Raises CandidateOverflow rather than truncating."""
        keys = list(candidates)
        mapping, _ = self.pool.assign(keys)
        by_index = {e["index"]: e for e in elements}
        options = []
        for symbol, key in mapping.items():
            action = candidates[key]
            element = by_index.get(key.split(":")[0], {})
            label = action.get("label", "")
            current = action.get("current_value", action.get("value", ""))
            detail = f"[{key}] {label!r}"
            if current:
                detail += f" current={current!r}"
            if element.get("role"):
                detail += f" role={element['role']}"
            options.append((symbol, detail))
        return options, mapping

    # ------------------------------------------------------------------ image

    def _image(self, page, elements):
        if not self.use_screenshot or not page.get("screenshot"):
            return None
        if self.numbered:
            return annotate(page["screenshot"], elements, scale=self.image_scale)
        return plain(page["screenshot"], scale=self.image_scale)

    # ----------------------------------------------------------------- choose

    def choose(self, page, goal, history):
        """Upstream's `choose` contract: pick an operation and, if it takes one, a target."""
        started = time.perf_counter()
        elements, targets, controls = action_space(page["actions"])
        image = self._image(page, elements)
        prefix_text = prompts.shared_prefix(
            goal, page, elements, history, use_screenshot=image is not None, with_boxes=self.with_boxes
        )
        prefix = self.backend.prefill(prefix_text, image)
        try:
            if self.arm in ("A", "B"):
                decision = self._choose_autoregressive(prefix, elements, targets, controls)
            else:
                decision = self._choose_branches(prefix, elements, targets, controls)
        except Exception:
            self.backend.release(prefix)
            raise
        if self.arm in ("C", "D"):
            # The text branch must continue this exact cache, so it is released by
            # the caller through `release(decision)` once the step is finished.
            decision["_prefix"] = prefix
        else:
            self.backend.release(prefix)
        decision["latency_ms"] = round((time.perf_counter() - started) * 1000)
        decision["prefill"] = prefix.timings
        decision["image_tokens"] = prefix.image_token_count
        decision["arm"] = self.arm
        return decision

    def _choose_branches(self, prefix, elements, targets, controls):
        options, operation_map = self._operation_options(targets, controls)
        branches = [
            {
                "key": "operation",
                "suffix": prompts.operation_branch(options, self.answer_context),
                "candidate_token_ids": [self.pool.token_id(s) for s, _ in options],
                "_symbols": [s for s, _ in options],
                "_map": operation_map,
            }
        ]
        overflow = {}
        for operation, candidates in targets.items():
            try:
                target_options, target_map = self._target_options(candidates, elements)
            except CandidateOverflow as exc:
                overflow[operation] = str(exc)
                continue
            branches.append(
                {
                    "key": operation.lower() + "_target",
                    "suffix": prompts.target_branch(operation, target_options, self.answer_context),
                    "candidate_token_ids": [self.pool.token_id(s) for s, _ in target_options],
                    "_symbols": [s for s, _ in target_options],
                    "_map": target_map,
                    "_candidates": candidates,
                }
            )
        answers = self.backend.decide(prefix, branches, parallel=self.parallel_branches)
        by_key = {a["key"]: a for a in answers}
        index = {b["key"]: b for b in branches}

        operation_answer = by_key["operation"]
        branch = index["operation"]
        operation = branch["_map"][branch["_symbols"][operation_answer["choice_position"]]]
        operation_probabilities = {
            branch["_map"][s]: p for s, p in zip(branch["_symbols"], operation_answer["probabilities"])
        }

        target = target_probabilities = None
        choice = operation
        if operation in targets:
            key = operation.lower() + "_target"
            if key not in by_key:
                return {
                    "choice": "BLOCKED",
                    "operation": "BLOCKED",
                    "target": None,
                    "confidence": operation_answer["confidence"],
                    "probabilities": {},
                    "operation_probabilities": operation_probabilities,
                    "target_probabilities": {},
                    "candidate_overflow": overflow,
                    "branches": answers,
                }
            answer, tbranch = by_key[key], index[key]
            target = tbranch["_map"][tbranch["_symbols"][answer["choice_position"]]]
            target_probabilities = {
                tbranch["_map"][s]: p for s, p in zip(tbranch["_symbols"], answer["probabilities"])
            }
            choice = tbranch["_candidates"][target]["id"]
            probabilities = {
                a["id"]: target_probabilities[k] for k, a in tbranch["_candidates"].items()
            }
        elif operation in controls:
            choice = controls[operation]["id"]
            probabilities = {choice: operation_probabilities[operation]}
        else:
            probabilities = {choice: operation_probabilities[operation]}
        return {
            "choice": choice,
            "operation": operation,
            "target": target,
            "confidence": operation_answer["confidence"],
            "probabilities": probabilities,
            "operation_probabilities": operation_probabilities,
            "target_probabilities": target_probabilities or {},
            "candidate_overflow": overflow,
            "branches": answers,
        }

    def _choose_autoregressive(self, prefix, elements, targets, controls):
        """Groups A and B: one compact line, parsed and checked against the legal set.

        The baseline gets the same candidate list and the same legality check; it
        writes `CLICK 4` instead of having a logit read, which is the only difference.
        """
        legal = {op: candidates for op, candidates in targets.items()}
        listing = []
        for operation, candidates in legal.items():
            listing.append(f"{operation} targets: " + ", ".join(candidates))
        control_names = list(controls) + ["DONE", "BLOCKED"]
        suffix = (
            "Legal operations: "
            + ", ".join(list(legal) + control_names)
            + "\n"
            + "\n".join(listing)
            + "\n"
            + prompts.compact_action_branch()
        )
        result = self.backend.generate(prefix, suffix, max_new_tokens=48, stop_strings=("\n",), reuse=True)
        parsed = parse_action_line(result["text"], legal, control_names)
        parsed["generation"] = result
        return parsed

    # ------------------------------------------------------------------- text

    def field_text(self, decision, action, *, max_new_tokens=48):
        """Conditional generation for TYPE_TEXT, continuing the decision's own prefix."""
        prefix = decision.get("_prefix")
        if prefix is None:
            raise ValueError("No prefix retained; text generation must follow choose() in arms C/D")
        suffix = prompts.text_branch(
            decision["operation"], action.get("label", ""), action.get("value") or None
        )
        result = self.backend.generate(
            prefix, suffix, max_new_tokens=max_new_tokens, stop_strings=("\n",), reuse=self.reuse_kv
        )
        value = result["text"].strip().strip('"')
        if value == "<MISSING>" or not value:
            return None, result
        return value, result

    def release(self, decision):
        prefix = decision.pop("_prefix", None)
        if prefix is not None:
            self.backend.release(prefix)


def parse_action_line(line, legal, control_names):
    """Parse `TYPE_TEXT 2 Zurich`. Anything not in the legal set becomes BLOCKED."""
    line = (line or "").strip()
    match = re.match(r"^([A-Z_]+)\s*(\S+)?\s*(.*)$", line)
    if not match:
        return _blocked(line)
    operation, target, text = match.group(1), match.group(2), match.group(3)
    if operation in control_names and operation not in legal:
        return {
            "choice": operation,
            "operation": operation,
            "target": None,
            "confidence": None,
            "probabilities": {},
            "operation_probabilities": {},
            "target_probabilities": {},
            "raw_line": line,
        }
    if operation not in legal:
        return _blocked(line)
    candidates = legal[operation]
    if target not in candidates:
        return _blocked(line, reason=f"target {target!r} not offered")
    return {
        "choice": candidates[target]["id"],
        "operation": operation,
        "target": target,
        "text": text.strip() or None,
        "confidence": None,
        "probabilities": {},
        "operation_probabilities": {},
        "target_probabilities": {},
        "raw_line": line,
    }


def _blocked(line, reason="unparsable action line"):
    return {
        "choice": "BLOCKED",
        "operation": "BLOCKED",
        "target": None,
        "confidence": None,
        "probabilities": {},
        "operation_probabilities": {},
        "target_probabilities": {},
        "raw_line": line,
        "parse_error": reason,
    }
