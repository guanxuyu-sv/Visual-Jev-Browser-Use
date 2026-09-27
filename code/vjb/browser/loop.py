"""The agent loop, with upstream's executor and our policy.

Adapted from `jev_ultrafast.agent.Agent` (upstream 1231850a). The ordering
semantics that make the loop safe are preserved deliberately:

  * the decision is consumed before any mutation, so a retry cannot double-click
  * freshness is checked before text generation and again immediately before input
  * execution is logged before the post-action observation, so a stale observation
    cannot erase an action that did happen
  * a mutation is never retried automatically

What changed: `choose` and `field_text` come from a local Policy over one model,
and every stage is timed separately so the latency decomposition the paper needs
can be rebuilt from the log rather than estimated.
"""

import base64
import time
from pathlib import Path

import jev_ultrafast.browser as upstream_browser
from jev_ultrafast.browser import Browser, StalePage

from ..model.policy import Policy  # noqa: F401  (re-exported for callers)
from .observe import action_space

MAX_STEPS = 60


CDP_STATS = {"screenshot_calls": 0, "screenshot_retries": 0, "screenshot_failures": 0}


class ScreenshotUnavailable(RuntimeError):
    """The capture did not come back. The step is an environment failure, not a
    silent downgrade to a screenshot-free observation."""


def configure_cdp(timeout=20.0, headless_screenshots=True, screenshot_timeout=3.0, screenshot_attempts=4):
    """Adapt the upstream CDP calls to a headless background tab.

    Two changes, both narrow:

    `timeout` — browser_harness binds its 5 s default as a function default
    argument, so the module constant cannot be raised after import.

    `headless_screenshots` — upstream calls `Page.captureScreenshot` with the
    default `fromSurface=true`, which waits for a frame from the browser
    compositor. The agent's tab is created with `background=True`, and a
    background tab in `--headless=new` never presents a surface, so that call
    hangs until the timeout **every time**, whether or not anything changed. It
    is not a slow screenshot; it is a screenshot that never arrives. Measured on
    this box: default times out at 5/5 attempts, `fromSurface=False` returns the
    same 9932-byte image in 28 ms. Capturing from the renderer instead is the
    only change; the pixels are the page's own.
    """
    original = getattr(upstream_browser, "_vjb_original_cdp", None) or upstream_browser.cdp
    upstream_browser._vjb_original_cdp = original
    CDP_STATS.update(screenshot_retries=0, screenshot_failures=0, screenshot_calls=0)

    def cdp(method, session_id=None, _response_timeout=timeout, **params):
        if not (headless_screenshots and method == "Page.captureScreenshot"):
            return original(method, session_id=session_id, _response_timeout=_response_timeout, **params)
        params.setdefault("fromSurface", False)
        CDP_STATS["screenshot_calls"] += 1
        # The capture occasionally still does not come back. A single long wait is
        # the worst way to spend that: short attempts recover in well under the
        # time one timeout costs, and a genuine failure is then reported rather
        # than silently turning an arm with a screenshot into one without.
        last = None
        for attempt in range(screenshot_attempts):
            try:
                return original(
                    method, session_id=session_id, _response_timeout=screenshot_timeout, **params
                )
            except Exception as exc:
                last = exc
                CDP_STATS["screenshot_retries"] += 1
        CDP_STATS["screenshot_failures"] += 1
        raise ScreenshotUnavailable(
            f"captureScreenshot failed {screenshot_attempts}x at {screenshot_timeout}s"
        ) from last

    upstream_browser.cdp = cdp
    return {
        "timeout": timeout,
        "from_surface": not headless_screenshots,
        "screenshot_timeout": screenshot_timeout,
        "screenshot_attempts": screenshot_attempts,
    }


def set_cdp_timeout(seconds):
    """Backwards-compatible alias."""
    return configure_cdp(timeout=seconds)


class Stopwatch:
    """Per-stage timings for one step. Every number the paper reports comes from here."""

    def __init__(self):
        self.stages = {}

    def time(self, name):
        return _Stage(self, name)

    def add(self, name, milliseconds):
        self.stages[name] = self.stages.get(name, 0.0) + milliseconds


class _Stage:
    def __init__(self, watch, name):
        self.watch, self.name = watch, name

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.watch.add(self.name, (time.perf_counter() - self.started) * 1000)
        return False


class Run:
    """One task attempt. `run()` drives it to done/blocked or the step budget."""

    def __init__(self, url, goal, policy, *, record_dir=None, screenshots=True, max_steps=MAX_STEPS,
                 max_stale=3):
        self.policy = policy
        self.goal = goal.strip() if isinstance(goal, str) else "\n".join(goal).strip()
        if not self.goal:
            raise ValueError("Supply a task")
        self.max_steps = max_steps
        self.max_stale = max_stale
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        self.history = []
        self.steps = []
        self.status = "ready"
        # A stale step makes no progress: the page moved under the decision and we
        # only re-observed. Repeating that forever is how a run burned its whole
        # budget re-clicking a button on a page MiniWoB had already emptied --
        # 105 of 106 successful runs did exactly this. Upstream's repeat guard
        # cannot see it, because it tests `page_changed is False` and a stale step
        # never sets that field at all.
        self.stale_streak = 0
        self.started_at = None

        watch = Stopwatch()
        with watch.time("browser_start"):
            self.browser = Browser(url)
        try:
            with watch.time("observe"):
                self.page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.startup = watch.stages
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            if self.page.get("screenshot"):
                (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(self.page["screenshot"]))

    # -------------------------------------------------------------- one step

    def step(self):
        if self.status in {"done", "blocked"}:
            raise ValueError("This run has stopped")
        if self.started_at is None:
            self.started_at = time.perf_counter()
        watch = Stopwatch()

        with watch.time("freshness"):
            fresh = self.browser.fresh(self.page)
        if not fresh:
            with watch.time("observe"):
                self.page = self.browser.observe(screenshot=self.screenshots)

        with watch.time("decide"):
            decision = self.policy.choose(self.page, self.goal, self.history)

        try:
            return self._act(decision, watch)
        finally:
            self.policy.release(decision)

    def _stalled(self, decision, watch, *, text=None, generation=None):
        """Record a no-progress step, and stop once they only repeat."""
        self.stale_streak += 1
        if self.stale_streak >= self.max_stale:
            self.status = "blocked"
            return self._record(
                decision, text, generation, watch, stale=True,
                error=f"no progress after {self.stale_streak} stale observations",
            )
        return self._record(decision, text, generation, watch, stale=True)

    def _act(self, decision, watch):
        page = self.page
        selected = decision["choice"]
        if selected in {"DONE", "BLOCKED"}:
            with watch.time("freshness"):
                fresh = self.browser.fresh(page)
            if not fresh:
                # The page moved under the decision; re-observe and choose again.
                self.page = self.browser.observe(screenshot=self.screenshots)
                return self._stalled(decision, watch)
            self.status = "done" if selected == "DONE" else "blocked"
            return self._record(decision, None, None, watch)

        action = next((a for a in page["actions"] if a["id"] == selected), None)
        if action is None:
            self.status = "blocked"
            return self._record(decision, None, None, watch, error=f"choice {selected!r} is not an observed action")
        if len(self.history) >= self.max_steps:
            self.status = "blocked"
            return self._record(decision, None, None, watch, error="step budget reached")

        text = generation = None
        if action["kind"] == "fill":
            with watch.time("freshness"):
                fresh = self.browser.fresh(page)
            if not fresh:
                self.page = self.browser.observe(screenshot=self.screenshots)
                return self._stalled(decision, watch)
            with watch.time("generate"):
                if decision.get("text") is not None:  # arms A/B produced it in the action line
                    text, generation = decision["text"], decision.get("generation")
                else:
                    text, generation = self.policy.field_text(decision, action)
            if text is None:
                self.status = "blocked"
                return self._record(decision, None, generation, watch, error="no field value available")

        try:
            with watch.time("execute"):
                self.browser.act(action, page, text=text)
        except StalePage:
            self.page = self.browser.observe(screenshot=self.screenshots)
            return self._stalled(decision, watch, text=text, generation=generation)

        self.stale_streak = 0  # the action executed; this step made progress
        entry = self._record(decision, text, generation, watch, action=action)
        # Time this even when it raises. A post-action observation that times out
        # still cost its wall clock, and leaving it unrecorded hides the stage
        # that dominates a step.
        try:
            with watch.time("observe_after"):
                self.page = self.browser.observe(screenshot=self.screenshots)
        except Exception as exc:
            entry["observe_error"] = f"{type(exc).__name__}: {exc}"
            entry["stages_ms"] = dict(watch.stages)
            self.status = "blocked"
            return entry
        entry["stages_ms"] = dict(watch.stages)
        entry["page_changed"] = self.page["fingerprint"] != page["fingerprint"]
        entry["url_after"] = self.page["url"]
        self.history.append(
            {
                "step": len(self.history) + 1,
                "action": action["label"],
                "kind": action["kind"],
                "text": text,
                "page_changed": entry["page_changed"],
            }
        )
        if self.record_dir and self.page.get("screenshot"):
            elapsed = round((time.perf_counter() - self.started_at) * 1000)
            (self.record_dir / f"{elapsed:06d}.jpg").write_bytes(base64.b64decode(self.page["screenshot"]))

        recent = self.steps[-3:]
        if len(recent) == 3 and all(s.get("page_changed") is False and s.get("kind") != "wait" for s in recent):
            self.status = "blocked"
        return entry

    def _record(self, decision, text, generation, watch, *, action=None, stale=False, error=None):
        entry = {
            "index": len(self.steps) + 1,
            "operation": decision.get("operation"),
            "target": decision.get("target"),
            "choice": decision.get("choice"),
            "kind": action["kind"] if action else None,
            "label": action["label"] if action else None,
            "text": text,
            "confidence": decision.get("confidence"),
            "arm": decision.get("arm"),
            "stale": stale,
            "error": error,
            "stages_ms": dict(watch.stages),
            "decision_ms": decision.get("latency_ms"),
            "prefill": decision.get("prefill"),
            "image_tokens": decision.get("image_tokens"),
            "candidate_overflow": decision.get("candidate_overflow") or {},
            "branches": [
                {k: b[k] for k in ("key", "confidence", "suffix_tokens", "batch_ms", "batched_with") if k in b}
                for b in decision.get("branches", [])
            ],
            "generation": {
                k: generation[k]
                for k in ("tokens", "reused_prefix_tokens", "recomputed_tokens", "ttft_ms", "total_ms")
                if generation and k in generation
            }
            if generation
            else None,
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000),
        }
        self.steps.append(entry)
        return entry

    # ------------------------------------------------------------------- run

    def run(self):
        while self.status not in {"done", "blocked"} and len(self.steps) < self.max_steps * 2:
            yield self.step()

    def result(self):
        return {
            "goal": self.goal,
            "status": self.status,
            "steps": self.steps,
            "history": self.history,
            "startup_ms": self.startup,
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000) if self.started_at else 0,
            "final_url": self.page.get("url"),
            "elements_final": len(action_space(self.page["actions"])[0]),
        }

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
