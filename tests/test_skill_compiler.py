"""Tests for the skill compiler (Kosmos Brain v2, Phase 2 "Fast").

Four layers, matching the wiring:

- Pure generalisation (in-memory, deterministic, no database):
  differing arguments become parameters, invariant outcomes
  become postconditions, runs that share no first step cannot
  compile, and the common prefix is all that survives.
- Binding (in-memory): a compiled skill binds its parameters
  to concrete values, with defaults from the first run.
- Compile, replay and repair (in-memory store, plus the
  sandbox): a compiled skill replays in the local sandbox and
  passes; a changed world fails the postcondition; repair
  re-generalses from fresh runs or refreshes the stale
  postcondition with an acceptance check — and never silently
  drops the old version.
- Versioning and the governor hook: re-compilation bumps the
  version and keeps the history; two consecutive failed
  replays break a version; the governor generalises the
  streak's runs through the compiler when one is attached.
- Durable store: versions land as procedure rows with the
  version number on the row, and round-trip with parameters.
"""

from __future__ import annotations

import json
import sys

import pytest

from puspa.governor import (
    COMPILE_AFTER_SUCCESSES,
    CompiledSkill,
    Depth,
    DepthGovernor,
    InMemorySkillStore,
    ProceduralSkillStore,
    RegressionCase,
    SkillCompiler,
    SkillRun,
    StepSignal,
    generalise_runs,
)
from puspa.governor.compiler import (
    REPAIR_ALREADY_PASSING,
    REPAIR_FAILED,
    REPAIR_POSTCONDITION_REFRESHED,
    REPAIR_REGENERALISED,
)
from puspa.localexec import LocalExecutor, PermissionDecision, PermissionDenied
from puspa.memory import procedural as procedural_store

TASK_KEY = "check-price-beras"

PRICES = {"beras": 100, "gula": 200}

#: Sandbox stand-in for a price tool: reads the prices file
#: (argv[2]) and prints one JSON object for the product
#: (argv[1]). The world changes by rewriting the prices file.
PRICE_SCRIPT = (
    "import json, sys; "
    "prices = json.load(open(sys.argv[2])); "
    "print(json.dumps({'ok': True, 'price': prices[sys.argv[1]]}))"
)


class FakePrompt:
    """A prompt that answers from a queue and records every ask."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.asked = []

    def ask(self, command, cwd):
        self.asked.append((tuple(command), cwd))
        return self.decisions.pop(0)


def _steps(product: str) -> list[dict]:
    """The price-check steps for one product."""
    return [{"tool": "check_price", "arguments": {"product": product, "store": "tokopedia"}}]


def _run(product: str) -> SkillRun:
    """One successful deep run: check the price, find it in PRICES."""
    return SkillRun(steps=_steps(product), last_result={"ok": True, "price": PRICES[product]})


def _runs(*products: str) -> list[SkillRun]:
    return [_run(product) for product in products]


def _price_runner(step: dict) -> dict:
    """Fake tool registry: the price check reads PRICES."""
    assert step["tool"] == "check_price"
    return {"ok": True, "price": PRICES[step["arguments"]["product"]]}


def _compiler(store=None, **kwargs):
    return SkillCompiler(store if store is not None else InMemorySkillStore(), **kwargs)


def _compile(compiler, task_key=TASK_KEY, products=("beras", "gula", "beras")):
    """Compile from runs over ``products``; return the result."""
    return compiler.compile(task_key, _runs(*products), step_runner=_price_runner)


@pytest.fixture
def workspace(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.fixture
def prices_file(workspace):
    path = workspace / "prices.json"
    path.write_text(json.dumps(PRICES), encoding="utf-8")
    return path


@pytest.fixture
def sandbox(workspace, prices_file):
    """A skill compiler whose replay runs steps in the sandbox."""
    executor = LocalExecutor(workspace, prompt=FakePrompt([PermissionDecision.APPROVE_SESSION]))
    tool_commands = {
        "check_price": [sys.executable, "-c", PRICE_SCRIPT, "{product}", str(prices_file)]
    }
    return SkillCompiler(InMemorySkillStore(), executor=executor, tool_commands=tool_commands)


# ---------------------------------------------------------------------------
# Pure generalisation
# ---------------------------------------------------------------------------


def test_generalise_parameterises_differing_arguments():
    plan = generalise_runs(_runs("beras", "gula", "beras"))
    assert plan is not None
    assert plan.steps == [
        {
            "tool": "check_price",
            "arguments": {"product": {"$param": "product"}, "store": "tokopedia"},
        }
    ]
    assert plan.parameters == {"product": "beras"}
    # The price varies across runs, so it is not a postcondition;
    # only the invariant "ok" is.
    assert plan.expect == {"ok": True}


def test_generalise_keeps_literals_and_invariant_postconditions():
    plan = generalise_runs(_runs("beras", "beras", "beras"))
    assert plan is not None
    assert plan.steps == _steps("beras")
    assert plan.parameters == {}
    assert plan.expect == {"ok": True, "price": 100}


def test_generalise_truncates_at_a_diverging_tool():
    runs = _runs("beras", "beras", "beras")
    runs[2] = SkillRun(
        steps=[*_steps("beras"), {"tool": "notify_user", "arguments": {}}],
        last_result={"ok": True, "price": 100},
    )
    runs[1] = SkillRun(
        steps=[*_steps("beras"), {"tool": "write_note", "arguments": {}}],
        last_result={"ok": True, "price": 100},
    )
    plan = generalise_runs(runs)
    assert plan is not None
    # The second steps differ, so only the shared first step survives.
    assert plan.steps == _steps("beras")


def test_generalise_returns_none_when_first_steps_differ():
    runs = [
        SkillRun(steps=_steps("beras"), last_result={"ok": True, "price": 100}),
        SkillRun(
            steps=[{"tool": "search_store", "arguments": {"product": "beras"}}],
            last_result={"ok": True, "price": 100},
        ),
    ]
    assert generalise_runs(runs) is None


def test_generalise_returns_none_without_runs():
    assert generalise_runs([]) is None


def test_generalise_drops_an_argument_missing_from_some_run():
    runs = _runs("beras", "gula", "beras")
    runs[2] = SkillRun(
        steps=[{"tool": "check_price", "arguments": {"product": "beras"}}],
        last_result={"ok": True, "price": 100},
    )
    plan = generalise_runs(runs)
    assert plan is not None
    # "store" is not in every run, so it is not part of the procedure.
    assert plan.steps == [{"tool": "check_price", "arguments": {"product": {"$param": "product"}}}]


def test_generalise_ignores_runs_without_a_last_result():
    runs = [
        SkillRun(steps=_steps("beras")),
        SkillRun(steps=_steps("gula")),
    ]
    plan = generalise_runs(runs)
    assert plan is not None
    assert plan.parameters == {"product": "beras"}
    assert plan.expect == {}


def test_skill_run_rejects_steps_without_a_tool():
    with pytest.raises(ValueError):
        SkillRun(steps=[{"arguments": {"product": "beras"}}])


def test_skill_run_rejects_empty_steps():
    with pytest.raises(ValueError):
        SkillRun(steps=[])


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def test_bind_substitutes_parameters_with_defaults():
    compiler = _compiler()
    compiled = _compile(compiler)
    assert compiled.skill is not None
    bound = compiled.skill.bind()
    assert bound.steps == _steps("beras")
    assert bound.regression.steps == _steps("beras")
    assert bound.regression.expect == {"ok": True}


def test_bind_accepts_explicit_bindings():
    compiler = _compiler()
    compiled = _compile(compiler)
    assert compiled.skill is not None
    bound = compiled.skill.bind({"product": "gula"})
    assert bound.steps == _steps("gula")
    assert bound.parameters == {"product": "beras"}


def test_bind_rejects_an_unknown_parameter():
    compiler = _compiler()
    compiled = _compile(compiler)
    assert compiled.skill is not None
    with pytest.raises(ValueError):
        compiled.skill.bind({"seller": "toko"})


def test_bind_rejects_a_parameter_with_no_binding():
    skill = CompiledSkill(
        task_key=TASK_KEY,
        steps=[{"tool": "check_price", "arguments": {"product": {"$param": "product"}}}],
        regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
    )
    with pytest.raises(ValueError):
        skill.bind()


def test_compiled_skill_rejects_a_version_below_one():
    with pytest.raises(ValueError):
        CompiledSkill(
            task_key=TASK_KEY,
            steps=_steps("beras"),
            regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
            version=0,
        )


# ---------------------------------------------------------------------------
# Compile and replay with a step runner (in-memory store)
# ---------------------------------------------------------------------------


def test_compile_generalises_and_stores_v1():
    store = InMemorySkillStore()
    result = _compile(_compiler(store))
    assert result.skill is not None
    assert result.skill.task_key == TASK_KEY
    assert result.skill.version == 1
    assert result.skill.status == "trial"
    assert result.skill.parameters == {"product": "beras"}
    assert result.skill.steps[0]["arguments"]["product"] == {"$param": "product"}
    assert result.runs_used == 3
    assert store.find(TASK_KEY) is not None


def test_compile_fails_when_runs_share_no_first_step():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    runs = [
        SkillRun(steps=_steps("beras"), last_result={"ok": True, "price": 100}),
        SkillRun(
            steps=[{"tool": "search_store", "arguments": {"product": "beras"}}],
            last_result={"ok": True, "price": 100},
        ),
    ]
    result = compiler.compile(TASK_KEY, runs, step_runner=_price_runner)
    assert result.skill is None
    assert "no common first step" in result.detail
    assert store.find(TASK_KEY) is None


def test_compile_fails_when_the_replay_fails_and_stores_nothing():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    result = compiler.compile(
        TASK_KEY, _runs("beras", "beras", "beras"), step_runner=lambda step: {"price": 999}
    )
    assert result.skill is None
    assert "failed its replay" in result.detail
    assert store.find(TASK_KEY) is None


def test_replay_without_a_runner_or_sandbox_reports_not_configured():
    compiler = _compiler()
    skill = CompiledSkill(
        task_key=TASK_KEY,
        steps=_steps("beras"),
        regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
    )
    result = compiler.replay(skill)
    assert result.passed is False
    assert result.kind == "not_configured"
    assert "no step runner" in result.detail


def test_replay_reports_a_postcondition_failure_structured():
    compiler = _compiler()
    compiled = _compile(compiler, products=("beras", "beras", "beras"))
    assert compiled.skill is not None
    result = compiler.replay(compiled.skill, step_runner=lambda step: {"ok": True, "price": 999})
    assert result.passed is False
    assert result.kind == "postcondition_failed"
    assert result.failed_key == "price"
    assert result.expected == 100
    assert result.actual == 999
    assert result.last_result == {"ok": True, "price": 999}


def test_replay_reports_a_step_failure_when_the_result_is_not_a_dict():
    compiler = _compiler()
    compiled = _compile(compiler)
    assert compiled.skill is not None
    result = compiler.replay(compiled.skill, step_runner=lambda step: "not a dict")
    assert result.passed is False
    assert result.kind == "step_failed"
    assert result.failed_key is None


# ---------------------------------------------------------------------------
# Versioning and history (in-memory store)
# ---------------------------------------------------------------------------


def test_recompile_bumps_the_version_and_keeps_history():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    first = _compile(compiler, products=("beras", "gula", "beras"))
    assert first.skill is not None and first.skill.version == 1
    second = _compile(compiler, products=("gula", "gula", "gula"))
    assert second.skill is not None and second.skill.version == 2
    assert store.find(TASK_KEY).version == 2
    history = store.history(TASK_KEY)
    assert [item.version for item in history] == [1, 2]
    assert history[0].regression.expect == {"ok": True}
    assert history[1].regression.expect == {"ok": True, "price": 200}


def test_an_identical_resave_keeps_the_version():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    first = _compile(compiler)
    second = _compile(compiler)
    assert first.skill is not None and second.skill is not None
    assert second.skill.version == 1
    assert len(store.history(TASK_KEY)) == 1


def test_two_failed_replays_break_the_skill_but_keep_it_stored():
    store = InMemorySkillStore()
    compiled = _compile(_compiler(store))
    assert compiled.skill is not None
    store.mark_failed(compiled.skill)
    assert store.find(TASK_KEY) is not None
    store.mark_failed(compiled.skill)
    assert store.find(TASK_KEY) is None
    history = store.history(TASK_KEY)
    assert len(history) == 1
    assert history[0].status == "broken"
    assert history[0].version == 1


def test_a_repair_after_breaking_versions_from_history():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    compiled = _compile(compiler)
    assert compiled.skill is not None
    store.mark_failed(compiled.skill)
    store.mark_failed(compiled.skill)
    assert store.find(TASK_KEY) is None
    repaired = compiler.repair(
        compiled.skill, runs=_runs("gula", "gula", "gula"), step_runner=_price_runner
    )
    assert repaired.action == REPAIR_REGENERALISED
    assert repaired.skill is not None and repaired.skill.version == 2
    assert store.find(TASK_KEY).version == 2


# ---------------------------------------------------------------------------
# Repair (in-memory store, step runner)
# ---------------------------------------------------------------------------


def test_repair_regeneralises_from_fresh_runs():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    world = {"price": 100}

    def runner(step):
        return {"ok": True, "price": world["price"]}

    compiled = compiler.compile(TASK_KEY, _runs("beras", "beras", "beras"), step_runner=runner)
    assert compiled.skill is not None
    assert compiled.skill.regression.expect == {"ok": True, "price": 100}
    world["price"] = 120
    failed = compiler.replay(compiled.skill, step_runner=runner)
    assert failed.passed is False and failed.kind == "postcondition_failed"
    fresh = [
        SkillRun(steps=_steps("beras"), last_result={"ok": True, "price": 120}) for _ in range(3)
    ]
    repaired = compiler.repair(compiled.skill, runs=fresh, step_runner=runner)
    assert repaired.action == REPAIR_REGENERALISED
    assert repaired.skill is not None
    assert repaired.skill.version == 2
    assert repaired.skill.regression.expect == {"ok": True, "price": 120}
    assert repaired.changes == {"version": [1, 2]}
    # The old version is still there as history, not dropped.
    assert [item.version for item in store.history(TASK_KEY)] == [1, 2]
    assert compiler.replay(repaired.skill, step_runner=runner).passed is True


def test_repair_refreshes_a_stale_postcondition_with_acceptance():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    world = {"price": 100}

    def runner(step):
        return {"ok": True, "price": world["price"]}

    compiled = compiler.compile(TASK_KEY, _runs("beras", "beras", "beras"), step_runner=runner)
    assert compiled.skill is not None
    world["price"] = 120
    repaired = compiler.repair(
        compiled.skill, step_runner=runner, accept=lambda key, old, new: True
    )
    assert repaired.action == REPAIR_POSTCONDITION_REFRESHED
    assert repaired.skill is not None and repaired.skill.version == 2
    assert repaired.skill.regression.expect == {"ok": True, "price": 120}
    assert repaired.changes == {"postcondition": {"price": [100, 120]}}
    assert compiler.replay(repaired.skill, step_runner=runner).passed is True


def test_repair_refuses_when_the_acceptance_check_says_no():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    world = {"price": 100}

    def runner(step):
        return {"ok": True, "price": world["price"]}

    compiled = compiler.compile(TASK_KEY, _runs("beras", "beras", "beras"), step_runner=runner)
    assert compiled.skill is not None
    world["price"] = 120
    repaired = compiler.repair(
        compiled.skill, step_runner=runner, accept=lambda key, old, new: False
    )
    assert repaired.action == REPAIR_FAILED
    assert repaired.skill is None
    assert "refused" in repaired.detail
    # The old version stays stored — the repair did not drop it.
    assert store.find(TASK_KEY) is not None
    assert store.find(TASK_KEY).version == 1


def test_repair_fails_without_runs_or_an_acceptance_check():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    compiled = _compile(compiler, products=("beras", "beras", "beras"))
    assert compiled.skill is not None
    repaired = compiler.repair(compiled.skill, step_runner=_price_runner)
    assert repaired.action == REPAIR_FAILED
    assert "needs fresh successful runs" in repaired.detail
    assert store.find(TASK_KEY) is not None


def test_repair_fails_when_the_steps_themselves_break():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    compiled = _compile(compiler, products=("beras", "beras", "beras"))
    assert compiled.skill is not None
    repaired = compiler.repair(
        compiled.skill,
        step_runner=lambda step: "not a dict",
        accept=lambda key, old, new: True,
    )
    assert repaired.action == REPAIR_FAILED
    assert "before the postcondition" in repaired.detail
    assert store.find(TASK_KEY) is not None


def test_repair_reports_already_passing_when_the_world_heals():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    compiled = _compile(compiler, products=("beras", "beras", "beras"))
    assert compiled.skill is not None
    repaired = compiler.repair(
        compiled.skill, step_runner=_price_runner, accept=lambda key, old, new: True
    )
    assert repaired.action == REPAIR_ALREADY_PASSING
    assert repaired.skill is not None and repaired.skill.version == 1


def test_repair_fails_when_the_observed_value_is_not_stable():
    store = InMemorySkillStore()
    compiler = _compiler(store)
    prices = [120, 130]

    def runner(step):
        return {"ok": True, "price": prices.pop(0)}

    compiled = _compile(compiler, products=("beras", "beras", "beras"))
    assert compiled.skill is not None
    repaired = compiler.repair(
        compiled.skill, step_runner=runner, accept=lambda key, old, new: True
    )
    assert repaired.action == REPAIR_FAILED
    assert "changed between replays" in repaired.detail


# ---------------------------------------------------------------------------
# Sandbox replay
# ---------------------------------------------------------------------------


def test_compiled_skill_replays_in_the_sandbox_and_passes(sandbox):
    result = sandbox.compile(TASK_KEY, _runs("beras", "gula", "beras"))
    assert result.skill is not None
    assert result.skill.version == 1
    assert "in the sandbox" in result.detail
    replayed = sandbox.replay(result.skill)
    assert replayed.passed is True
    assert replayed.kind == "passed"
    assert replayed.steps_run == 1
    assert replayed.last_result == {"ok": True, "price": 100}
    assert len(replayed.executions) == 1
    assert replayed.executions[0].returncode == 0
    assert replayed.executions[0].timed_out is False


def test_sandbox_replay_with_a_different_binding(sandbox):
    result = sandbox.compile(TASK_KEY, _runs("beras", "gula", "beras"))
    assert result.skill is not None
    replayed = sandbox.replay(result.skill, bindings={"product": "gula"})
    assert replayed.passed is True
    assert replayed.last_result == {"ok": True, "price": 200}


def test_sandbox_replay_fails_on_a_changed_world(sandbox, prices_file):
    result = sandbox.compile(TASK_KEY, _runs("beras", "beras", "beras"))
    assert result.skill is not None
    assert result.skill.regression.expect == {"ok": True, "price": 100}
    prices_file.write_text(json.dumps({"beras": 120, "gula": 200}), encoding="utf-8")
    replayed = sandbox.replay(result.skill)
    assert replayed.passed is False
    assert replayed.kind == "postcondition_failed"
    assert replayed.failed_key == "price"
    assert replayed.expected == 100
    assert replayed.actual == 120


def test_sandbox_repair_regeneralises_in_the_sandbox(sandbox, prices_file):
    result = sandbox.compile(TASK_KEY, _runs("beras", "beras", "beras"))
    assert result.skill is not None
    prices_file.write_text(json.dumps({"beras": 120, "gula": 200}), encoding="utf-8")
    assert sandbox.replay(result.skill).passed is False
    fresh = [
        SkillRun(steps=_steps("beras"), last_result={"ok": True, "price": 120}) for _ in range(3)
    ]
    repaired = sandbox.repair(result.skill, runs=fresh)
    assert repaired.action == REPAIR_REGENERALISED
    assert repaired.skill is not None and repaired.skill.version == 2
    assert sandbox.replay(repaired.skill).passed is True


def test_sandbox_replay_fails_without_a_tool_command(workspace):
    executor = LocalExecutor(workspace, prompt=FakePrompt([PermissionDecision.APPROVE_SESSION]))
    compiler = SkillCompiler(InMemorySkillStore(), executor=executor, tool_commands={})
    skill = CompiledSkill(
        task_key=TASK_KEY,
        steps=_steps("beras"),
        regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
    )
    result = compiler.replay(skill)
    assert result.passed is False
    assert result.kind == "step_failed"
    assert "no sandbox command" in result.detail
    assert "check_price" in result.detail


def test_sandbox_replay_fails_on_non_json_stdout(workspace):
    executor = LocalExecutor(workspace, prompt=FakePrompt([PermissionDecision.APPROVE_SESSION]))
    compiler = SkillCompiler(
        InMemorySkillStore(),
        executor=executor,
        tool_commands={"check_price": [sys.executable, "-c", "print('hello')"]},
    )
    skill = CompiledSkill(
        task_key=TASK_KEY,
        steps=_steps("beras"),
        regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
    )
    result = compiler.replay(skill)
    assert result.passed is False
    assert result.kind == "step_failed"
    assert "did not print JSON" in result.detail


def test_sandbox_replay_fails_on_a_nonzero_exit(workspace):
    executor = LocalExecutor(workspace, prompt=FakePrompt([PermissionDecision.APPROVE_SESSION]))
    compiler = SkillCompiler(
        InMemorySkillStore(),
        executor=executor,
        tool_commands={"check_price": [sys.executable, "-c", "raise SystemExit(1)"]},
    )
    skill = CompiledSkill(
        task_key=TASK_KEY,
        steps=_steps("beras"),
        regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
    )
    result = compiler.replay(skill)
    assert result.passed is False
    assert result.kind == "step_failed"
    assert "exited 1" in result.detail


def test_sandbox_replay_denied_by_the_user_propagates(workspace, prices_file):
    executor = LocalExecutor(workspace, prompt=FakePrompt([PermissionDecision.DENY]))
    compiler = SkillCompiler(
        InMemorySkillStore(),
        executor=executor,
        tool_commands={
            "check_price": [sys.executable, "-c", PRICE_SCRIPT, "{product}", str(prices_file)]
        },
    )
    skill = CompiledSkill(
        task_key=TASK_KEY,
        steps=_steps("beras"),
        regression=RegressionCase.from_steps(_steps("beras"), {"ok": True}),
    )
    with pytest.raises(PermissionDenied):
        compiler.replay(skill)


def test_compiler_rejects_a_tool_command_that_is_not_a_list():
    with pytest.raises(ValueError):
        SkillCompiler(InMemorySkillStore(), tool_commands={"check_price": "echo hi"})


# ---------------------------------------------------------------------------
# Governor hook with a compiler attached
# ---------------------------------------------------------------------------


def test_governor_with_compiler_generalises_the_streak():
    store = InMemorySkillStore()
    compiler = SkillCompiler(store)
    governor = DepthGovernor(skill_store=store, compiler=compiler)
    compiled = None
    for product in ("beras", "gula", "beras"):
        compiled = governor.record_outcome(
            TASK_KEY,
            depth=Depth.DEEP,
            succeeded=True,
            steps=_steps(product),
            last_result={"ok": True, "price": PRICES[product]},
            step_runner=_price_runner,
        )
    assert compiled is not None
    assert compiled.version == 1
    assert compiled.parameters == {"product": "beras"}
    assert compiled.steps[0]["arguments"]["product"] == {"$param": "product"}
    assert governor.streak(TASK_KEY) == 0
    decision = governor.decide(StepSignal(uncertainty=0.1, stakes=0.1), task_key=TASK_KEY)
    assert decision.depth is Depth.REFLEX
    assert decision.skill_id == compiled.skill_id


def test_governor_compiler_path_needs_no_regression_argument():
    store = InMemorySkillStore()
    governor = DepthGovernor(skill_store=store, compiler=SkillCompiler(store))
    compiled = None
    for _ in range(COMPILE_AFTER_SUCCESSES):
        compiled = governor.record_outcome(
            TASK_KEY,
            depth=Depth.DEEP,
            succeeded=True,
            steps=_steps("beras"),
            last_result={"ok": True, "price": 100},
            step_runner=_price_runner,
        )
    assert compiled is not None
    assert compiled.version == 1


def test_governor_failure_drops_the_stashed_runs():
    store = InMemorySkillStore()
    governor = DepthGovernor(skill_store=store, compiler=SkillCompiler(store))
    for _ in range(2):
        assert (
            governor.record_outcome(
                TASK_KEY,
                depth=Depth.DEEP,
                succeeded=True,
                steps=_steps("gula"),
                last_result={"ok": True, "price": 200},
                step_runner=_price_runner,
            )
            is None
        )
    assert governor.record_outcome(TASK_KEY, depth=Depth.DEEP, succeeded=False) is None
    assert governor.streak(TASK_KEY) == 0
    compiled = None
    for _ in range(COMPILE_AFTER_SUCCESSES):
        compiled = governor.record_outcome(
            TASK_KEY,
            depth=Depth.DEEP,
            succeeded=True,
            steps=_steps("beras"),
            last_result={"ok": True, "price": 100},
            step_runner=_price_runner,
        )
    assert compiled is not None
    # The compiled skill comes from the post-failure runs only.
    assert compiled.parameters == {}


# ---------------------------------------------------------------------------
# Durable store: versions land as procedure rows
# ---------------------------------------------------------------------------


def test_compiled_skill_v1_lands_in_the_procedural_store(test_dsn, make_user):
    user = make_user()
    store = ProceduralSkillStore(user_id=user["user_id"], dsn=test_dsn)
    result = _compile(_compiler(store))
    assert result.skill is not None
    assert result.skill.version == 1
    rows = [
        row
        for row in procedural_store.list_procedures(user["user_id"], dsn=test_dsn)
        if row["service"] == f"compiled:{TASK_KEY}"
    ]
    assert len(rows) == 1
    assert rows[0]["version"] == 1
    assert rows[0]["status"] == "trial"
    assert rows[0]["step"][0]["arguments"]["product"] == {"$param": "product"}


def test_durable_repair_creates_a_v2_row_and_keeps_history(test_dsn, make_user):
    user = make_user()
    store = ProceduralSkillStore(user_id=user["user_id"], dsn=test_dsn)
    compiler = _compiler(store)
    world = {"price": 100}

    def runner(step):
        return {"ok": True, "price": world["price"]}

    compiled = compiler.compile(TASK_KEY, _runs("beras", "beras", "beras"), step_runner=runner)
    assert compiled.skill is not None
    world["price"] = 120
    fresh = [
        SkillRun(steps=_steps("beras"), last_result={"ok": True, "price": 120}) for _ in range(3)
    ]
    repaired = compiler.repair(compiled.skill, runs=fresh, step_runner=runner)
    assert repaired.action == REPAIR_REGENERALISED
    assert repaired.skill is not None and repaired.skill.version == 2
    history = store.history(TASK_KEY)
    assert [item.version for item in history] == [1, 2]
    found = store.find(TASK_KEY)
    assert found is not None and found.version == 2
    assert found.regression.expect == {"ok": True, "price": 120}
    assert found.run_regression(runner).passed is True


def test_durable_store_round_trips_parameters(test_dsn, make_user):
    user = make_user()
    store = ProceduralSkillStore(user_id=user["user_id"], dsn=test_dsn)
    result = _compile(_compiler(store))
    assert result.skill is not None
    found = store.find(TASK_KEY)
    assert found is not None
    assert found.parameters == {"product": "beras"}
    assert found.steps[0]["arguments"]["product"] == {"$param": "product"}
    bound = found.bind({"product": "gula"})
    assert bound.steps == _steps("gula")
    assert found.run_regression(_price_runner).passed is True


def test_broken_durable_skill_is_skipped_by_find(test_dsn, make_user):
    user = make_user()
    store = ProceduralSkillStore(user_id=user["user_id"], dsn=test_dsn)
    result = _compile(_compiler(store), products=("beras", "beras", "beras"))
    assert result.skill is not None
    store.mark_failed(result.skill)
    assert store.find(TASK_KEY) is not None
    store.mark_failed(result.skill)
    assert store.find(TASK_KEY) is None
    history = store.history(TASK_KEY)
    assert len(history) == 1
    assert history[0].status == "broken"


def test_insert_procedure_rejects_a_version_below_one(test_dsn, make_user):
    user = make_user()
    with pytest.raises(ValueError):
        procedural_store.insert_procedure(
            user["user_id"], f"compiled:{TASK_KEY}", _steps("beras"), version=0, dsn=test_dsn
        )
