"""Tests for per-task thinking budgets (Kosmos Brain v2, Phase 2 "Fast", V2-14).

Pure and in-memory, no database: the budget is sized by the
task's stakes and the depth the governor picked, plus a hard
wall-clock cap. When a cap hits mid-task the turn reports
progress — what was done, what remains, what happens next —
and keeps going, never silent and never stopped. The
done-when proof is the capped-task test: a task under a tiny
budget still runs every step, and a progress report comes
out of it instead of silence or an exception.
"""

from __future__ import annotations

import pytest

from puspa.governor import (
    Depth,
    DepthGovernor,
    ProgressReport,
    StepSignal,
    ThinkingBudget,
    run_capped_task,
)


class FakeClock:
    """A monotonic-shaped clock the test moves by hand."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _steps(*labels: str):
    """A task of named steps, each returning its label."""
    return [(label, (lambda name=label: f"did {name}")) for label in labels]


# -- sizing: stakes scale the budget, depth sets the base -----------------


def test_budget_scales_with_stakes() -> None:
    """Same depth, higher stakes: more steps and more seconds."""
    low = ThinkingBudget(stakes=0.0, depth=Depth.DEEP)
    high = ThinkingBudget(stakes=1.0, depth=Depth.DEEP)
    assert high.steps_limit > low.steps_limit
    assert high.wall_clock_seconds > low.wall_clock_seconds


def test_reflex_budget_smaller_than_deep() -> None:
    """A reflex replay gets far less thinking than a deep step."""
    reflex = ThinkingBudget(stakes=0.5, depth=Depth.REFLEX)
    deep = ThinkingBudget(stakes=0.5, depth=Depth.DEEP)
    assert reflex.steps_limit < deep.steps_limit
    assert reflex.wall_clock_seconds < deep.wall_clock_seconds


def test_hard_ceiling_bounds_the_budget() -> None:
    """Even the highest stakes never pass the hard ceilings."""
    budget = ThinkingBudget(stakes=1.0, depth=Depth.DEEP, step_scale=100.0, time_scale=100.0)
    assert budget.steps_limit <= 100
    assert budget.wall_clock_seconds <= 600.0


def test_bad_stakes_and_limits_are_rejected() -> None:
    """Stakes outside 0..1 and non-positive limits raise ValueError."""
    with pytest.raises(ValueError):
        ThinkingBudget(stakes=1.5)
    with pytest.raises(ValueError):
        ThinkingBudget(stakes=-0.1)
    with pytest.raises(ValueError):
        ThinkingBudget(max_steps=0)
    with pytest.raises(ValueError):
        ThinkingBudget(wall_clock_seconds=0)


# -- the done-when proof: a capped task reports progress, then continues --


def test_capped_task_reports_progress_and_continues() -> None:
    """A deep task under a 2-step budget: every step runs, reports come out.

    This is the V2-14 done-when: the cap hits mid-task, a
    progress report names what was done, what remains and what
    is next, and the task still finishes — no silence, no
    exception, no dropped step.
    """
    budget = ThinkingBudget(stakes=0.9, depth=Depth.DEEP, max_steps=2)
    sent: list[ProgressReport] = []
    result = run_capped_task(
        budget,
        _steps("draft plan", "dry-run plan", "verify result", "act", "write ledger"),
        on_progress=sent.append,
    )
    assert result.completed is True
    assert result.results == (
        "did draft plan",
        "did dry-run plan",
        "did verify result",
        "did act",
        "did write ledger",
    )
    assert len(result.reports) == 2
    assert list(result.reports) == sent
    first, second = result.reports
    assert first.reason == "step_budget"
    assert first.done == ("draft plan", "dry-run plan")
    assert first.remaining == ("verify result", "act", "write ledger")
    assert first.next_step == "verify result"
    assert second.done[-1] == "act"
    assert second.remaining == ("write ledger",)
    assert second.next_step == "write ledger"
    assert budget.cap_hits == 2
    assert budget.summary()["cap_hits"] == 2


def test_progress_report_text_names_done_remaining_and_next() -> None:
    """The report reads as a plain status message, not an error."""
    budget = ThinkingBudget(max_steps=1)
    budget.spend_step("draft plan")
    report = budget.check(remaining=["verify result"], next_step="verify result")
    assert report is not None
    text = report.text()
    assert "draft plan" in text
    assert "verify result" in text
    assert "Still working" in text
    assert report.to_dict()["reason"] == "step_budget"
    assert report.to_dict()["cap_number"] == 1


def test_wall_clock_cap_reports_and_continues() -> None:
    """Time, not steps, can fire the cap: the report still comes out."""
    clock = FakeClock()
    budget = ThinkingBudget(
        stakes=0.9, depth=Depth.DEEP, wall_clock_seconds=10.0, clock=clock
    )
    sent: list[ProgressReport] = []

    def _slow_step() -> str:
        clock.advance(12.0)  # the step itself runs past the deadline
        return "did slow step"

    result = run_capped_task(
        budget,
        [("slow step", _slow_step), ("next step", lambda: "did next step")],
        on_progress=sent.append,
    )
    assert result.completed is True
    assert result.results == ("did slow step", "did next step")
    assert len(sent) == 1
    assert sent[0].reason == "wall_clock"
    assert sent[0].done == ("slow step",)
    assert sent[0].next_step == "next step"


def test_no_report_while_inside_budget() -> None:
    """Silence is correct while the work is on track: no report, no cap hit."""
    budget = ThinkingBudget(stakes=0.5, depth=Depth.DELIBERATE)
    result = run_capped_task(budget, _steps("one plan", "one critique"), on_progress=None)
    assert result.completed is True
    assert result.reports == ()
    assert budget.cap_hits == 0
    assert budget.check(remaining=[]) is None


# -- governor wiring: a decided deep step gets a stakes-sized budget ------


def test_governor_wires_budget_to_deep_decision() -> None:
    """decide() deep + thinking_budget(): the budget follows depth and stakes."""
    governor = DepthGovernor()
    signal = StepSignal(uncertainty=0.9, stakes=0.9)
    decision = governor.decide(signal)
    assert decision.depth is Depth.DEEP
    budget = governor.thinking_budget(signal, decision)
    assert budget.depth is Depth.DEEP
    assert budget.stakes == 0.9
    assert budget.steps_limit > ThinkingBudget(stakes=0.1, depth=Depth.DEEP).steps_limit


def test_governor_deep_step_with_small_budget_gets_capped_and_reports() -> None:
    """A deep step with a 1-step budget: capped, reported, still finished."""
    governor = DepthGovernor()
    signal = StepSignal(uncertainty=0.9, stakes=0.9)
    decision = governor.decide(signal)
    assert decision.depth is Depth.DEEP
    budget = governor.thinking_budget(signal, decision, max_steps=1)
    sent: list[ProgressReport] = []
    result = run_capped_task(
        budget,
        _steps("candidate plans", "dry-run", "verify"),
        on_progress=sent.append,
    )
    assert result.completed is True
    assert len(result.results) == 3
    assert len(sent) == 3
    assert sent[0].done == ("candidate plans",)
    assert sent[0].remaining == ("dry-run", "verify")
    assert sent[0].next_step == "dry-run"
    assert sent[2].done == ("candidate plans", "dry-run", "verify")
    assert sent[2].remaining == ()
    assert sent[2].next_step is None


def test_governor_budget_scales_follow_governor_thresholds() -> None:
    """The governor's own scales stretch the budget it hands out."""
    signal = StepSignal(uncertainty=0.9, stakes=0.9)
    decision = DepthGovernor().decide(signal)
    plain = DepthGovernor().thinking_budget(signal, decision)
    doubled = DepthGovernor(thinking_step_scale=2.0, thinking_time_scale=2.0)
    big = doubled.thinking_budget(signal, decision)
    assert big.steps_limit > plain.steps_limit
    assert big.wall_clock_seconds > plain.wall_clock_seconds
    with pytest.raises(ValueError):
        DepthGovernor(thinking_step_scale=0)
