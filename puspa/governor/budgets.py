"""Per-task thinking budgets (Kosmos Brain v2, Phase 2 "Fast", V2-14).

Every task gets a thinking budget set by its stakes, plus a
hard wall-clock cap. When the cap hits mid-task the turn
reports progress — what was done, what remains, what happens
next — and keeps going. It never goes silent and never stops
the task: the cap buys a report, not a halt.

How the budget is sized
------------------------
The depth the governor picked sets the base allowance, and
the task's stakes scale it: a high-stakes task gets more
thinking than a low-stakes one at the same depth.

- Reflex: 3 steps, 5 seconds. A replayed skill should be quick.
- Deliberate: 10 steps, 30 seconds. The default, one plan plus
  one self-critique.
- Deep: 25 steps, 120 seconds. Several candidate plans, a
  dry-run, a verifier — the most thinking, still capped.

Stakes scale both numbers by ``0.5 + stakes`` (stakes 0 gets
half the base, stakes 1 gets one and a half times), and the
hard ceilings bound the result: never more than 100 steps or
600 seconds for one task, whatever the stakes. The ceilings
are the "hard" in hard wall-clock cap — no task thinks
forever, however important it is.

How the cap behaves
-------------------
:class:`ThinkingBudget` counts thinking steps and watches a
monotonic clock. :meth:`ThinkingBudget.check` is called after
each step with what is still left to do: while the task is
inside its budget it returns None (silence is fine when the
work is on track); the moment a cap is hit it returns a
:class:`ProgressReport` — done, remaining, next step — and
opens a fresh slice of the same size, so the task continues
under a new deadline instead of stopping. Each cap hit is
audited in the budget's report log, the same spend-log shape
the K10 interruption budget keeps (:mod:`puspa.proactivity.budget`).

:func:`run_capped_task` is the call site the turn loop uses:
it runs the task's steps in order, checks the budget after
each one, and hands every progress report to ``on_progress``
(a channel send, a ledger note, a streamed message) as soon
as the cap hits. Every step still runs to completion; a cap
never raises and never drops a step.

The governor wires the budget in: :meth:`DepthGovernor.thinking_budget
<puspa.governor.governor.DepthGovernor.thinking_budget>` builds
one from the signal's stakes and the depth it just decided,
so a deep step with a small budget gets capped and reports.
The turn loop itself is not touched here — ``puspa/turns/loop.py``
is already over its size baseline, so the loop call site is a
follow-up that must land without growing that file.

Pure Python: no database, no network, no model call. The
clock is injectable (a ``time.monotonic``-shaped callable) so
tests can move time without waiting.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .governor import Depth

__all__ = [
    "MAX_THINKING_SECONDS",
    "MAX_THINKING_STEPS",
    "THINKING_SECONDS",
    "THINKING_STEPS",
    "CappedTaskResult",
    "ProgressReport",
    "ThinkingBudget",
    "run_capped_task",
]

#: Base thinking-step allowance per depth, before stakes scaling.
THINKING_STEPS: dict[Depth, int] = {
    Depth.REFLEX: 3,
    Depth.DELIBERATE: 10,
    Depth.DEEP: 25,
}

#: Base wall-clock allowance in seconds per depth, before stakes scaling.
THINKING_SECONDS: dict[Depth, float] = {
    Depth.REFLEX: 5.0,
    Depth.DELIBERATE: 30.0,
    Depth.DEEP: 120.0,
}

#: Hard ceilings: no task gets more thinking than this, whatever its stakes.
MAX_THINKING_STEPS = 100
MAX_THINKING_SECONDS = 600.0


def _check_stakes(stakes: float) -> float:
    """Validate a 0..1 stakes value; ValueError with the value otherwise."""
    if not isinstance(stakes, (int, float)) or isinstance(stakes, bool):
        raise ValueError(f"stakes must be a number within 0..1, got {stakes!r}")
    if not 0.0 <= float(stakes) <= 1.0:
        raise ValueError(f"stakes must be within 0..1, got {stakes}")
    return float(stakes)


def _scaled_steps(depth: Depth, stakes: float, step_scale: float) -> int:
    """Step allowance for a depth at given stakes, inside the hard ceiling."""
    base = THINKING_STEPS[depth]
    scaled = int(base * (0.5 + stakes) * step_scale)
    return max(1, min(MAX_THINKING_STEPS, scaled))


def _scaled_seconds(depth: Depth, stakes: float, time_scale: float) -> float:
    """Wall-clock allowance for a depth at given stakes, inside the hard ceiling."""
    base = THINKING_SECONDS[depth]
    scaled = base * (0.5 + stakes) * time_scale
    return max(1.0, min(MAX_THINKING_SECONDS, scaled))


@dataclass(frozen=True)
class ProgressReport:
    """One progress report, emitted when a thinking cap is hit.

    ``done`` is what the task finished, ``remaining`` is what
    is still to do, ``next_step`` is what the task does next.
    ``reason`` names the cap that fired: ``"step_budget"`` or
    ``"wall_clock"``. ``cap_number`` counts this task's cap
    hits, starting at 1. The step and time figures are the
    slice that just ended, so the report reads as a status
    update, not an error.
    """

    done: tuple[str, ...]
    remaining: tuple[str, ...]
    next_step: str | None
    reason: str
    steps_used: int
    steps_limit: int
    elapsed_seconds: float
    wall_clock_seconds: float
    cap_number: int

    def text(self) -> str:
        """The report as one plain message for the user."""
        done = ", ".join(self.done) if self.done else "nothing yet"
        remaining = ", ".join(self.remaining) if self.remaining else "nothing"
        nxt = self.next_step or "wrap up"
        return (
            f"Still working on it — done so far: {done}. "
            f"Still to do: {remaining}. Next: {nxt}."
        )

    def to_dict(self) -> dict:
        """JSON-safe dict of this report (ledger/audit shape)."""
        return {
            "done": list(self.done),
            "remaining": list(self.remaining),
            "next_step": self.next_step,
            "reason": self.reason,
            "steps_used": self.steps_used,
            "steps_limit": self.steps_limit,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "wall_clock_seconds": self.wall_clock_seconds,
            "cap_number": self.cap_number,
        }


@dataclass(frozen=True)
class CappedTaskResult:
    """What :func:`run_capped_task` returns when every step ran.

    ``results`` holds each step's return value in step order.
    ``reports`` holds every progress report the caps produced
    (empty when the task stayed inside its budget).
    ``completed`` is always True: a cap reports and continues,
    it never cuts the task short.
    """

    results: tuple[object, ...]
    reports: tuple[ProgressReport, ...]
    completed: bool = True

    def to_dict(self) -> dict:
        """JSON-safe dict of this result."""
        return {
            "results": list(self.results),
            "reports": [report.to_dict() for report in self.reports],
            "completed": self.completed,
        }


class ThinkingBudget:
    """One task's thinking budget: stakes-sized, wall-clock capped.

    ``stakes`` (0..1) scales the depth's base allowance;
    ``depth`` picks the base. ``max_steps`` and
    ``wall_clock_seconds`` override the scaled defaults when
    given (a deep step with a deliberately small budget, for
    example). ``step_scale`` and ``time_scale`` are extra
    multipliers the governor passes in. ``clock`` is a
    ``time.monotonic``-shaped callable, injectable for tests.
    """

    def __init__(
        self,
        *,
        stakes: float = 0.5,
        depth: Depth | str = Depth.DELIBERATE,
        max_steps: int | None = None,
        wall_clock_seconds: float | None = None,
        step_scale: float = 1.0,
        time_scale: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.stakes = _check_stakes(stakes)
        self.depth = depth if isinstance(depth, Depth) else Depth(depth)
        if step_scale <= 0:
            raise ValueError(f"step_scale must be positive, got {step_scale}")
        if time_scale <= 0:
            raise ValueError(f"time_scale must be positive, got {time_scale}")
        if max_steps is not None and max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if wall_clock_seconds is not None and wall_clock_seconds <= 0:
            raise ValueError(
                f"wall_clock_seconds must be positive, got {wall_clock_seconds}"
            )
        self.steps_limit = (
            max_steps
            if max_steps is not None
            else _scaled_steps(self.depth, self.stakes, step_scale)
        )
        self.wall_clock_seconds = (
            wall_clock_seconds
            if wall_clock_seconds is not None
            else _scaled_seconds(self.depth, self.stakes, time_scale)
        )
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._started_at = self._clock()
        self._deadline = self._started_at + self.wall_clock_seconds
        self.steps_used = 0
        self.cap_hits = 0
        self._done: list[str] = []
        self._reports: list[ProgressReport] = []

    # -- reads ---------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since the task started (all slices)."""
        return self._clock() - self._started_at

    @property
    def remaining_steps(self) -> int:
        """Thinking steps left in the current slice. Never negative."""
        return max(0, self.steps_limit - self.steps_used)

    @property
    def remaining_seconds(self) -> float:
        """Seconds left before the current slice's deadline. Never negative."""
        return max(0.0, self._deadline - self._clock())

    @property
    def exhausted(self) -> bool:
        """True when the current slice hit its step or wall-clock cap."""
        return self.steps_used >= self.steps_limit or self._clock() >= self._deadline

    @property
    def reports(self) -> list[ProgressReport]:
        """Every progress report this budget emitted, in order. A copy."""
        return list(self._reports)

    def summary(self) -> dict:
        """This budget in numbers, for the turn result and the ledger."""
        return {
            "depth": self.depth.value,
            "stakes": self.stakes,
            "steps_used": self.steps_used,
            "steps_limit": self.steps_limit,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "wall_clock_seconds": self.wall_clock_seconds,
            "remaining_seconds": round(self.remaining_seconds, 3),
            "cap_hits": self.cap_hits,
            "reports": len(self._reports),
        }

    # -- writes ----------------------------------------------------------

    def spend_step(self, label: str = "") -> int:
        """Spend one thinking step; returns the new ``steps_used`` count.

        Never raises: hitting the cap is a report, not a stop —
        :meth:`check` turns it into a progress report and opens
        a fresh slice. An empty label records the step without
        naming it in the report.
        """
        self.steps_used += 1
        if label:
            self._done.append(label)
        return self.steps_used

    def check(
        self,
        *,
        remaining: Sequence[str] = (),
        next_step: str | None = None,
    ) -> ProgressReport | None:
        """Report progress when a cap is hit; None while inside budget.

        When the step budget or the wall clock is exhausted, a
        :class:`ProgressReport` is built from the finished steps
        and ``remaining`` / ``next_step``, recorded in the report
        log, and a fresh slice of the same size opens — the step
        counter resets and the deadline moves forward — so the
        task continues instead of stopping. Returns None when
        neither cap is hit: silence is correct while the work
        is on track.
        """
        now = self._clock()
        if self.steps_used < self.steps_limit and now < self._deadline:
            return None
        reason = "step_budget" if self.steps_used >= self.steps_limit else "wall_clock"
        self.cap_hits += 1
        report = ProgressReport(
            done=tuple(self._done),
            remaining=tuple(remaining),
            next_step=next_step,
            reason=reason,
            steps_used=self.steps_used,
            steps_limit=self.steps_limit,
            elapsed_seconds=now - self._started_at,
            wall_clock_seconds=self.wall_clock_seconds,
            cap_number=self.cap_hits,
        )
        self._reports.append(report)
        self.steps_used = 0
        self._deadline = now + self.wall_clock_seconds
        return report


def run_capped_task(
    budget: ThinkingBudget,
    steps: Iterable[tuple[str, Callable[[], object]]],
    *,
    on_progress: Callable[[ProgressReport], None] | None = None,
) -> CappedTaskResult:
    """Run a task's steps under a thinking budget, reporting on every cap.

    Each step is a ``(label, callable)`` pair; the callable
    takes no arguments and its return value lands in
    ``results`` in step order. After each step the budget is
    checked against the steps still to run: on a cap hit the
    progress report goes to ``on_progress`` immediately (a
    channel send, a streamed message) and the task continues
    with the next step. A cap never raises and never drops a
    step — every step runs, and the task always completes.
    """
    planned = list(steps)
    results: list[object] = []
    for index, (label, step) in enumerate(planned):
        results.append(step())
        budget.spend_step(label)
        rest = [name for name, _ in planned[index + 1 :]]
        nxt = rest[0] if rest else None
        report = budget.check(remaining=rest, next_step=nxt)
        if report is not None and on_progress is not None:
            on_progress(report)
    return CappedTaskResult(
        results=tuple(results),
        reports=tuple(budget.reports),
        completed=True,
    )
