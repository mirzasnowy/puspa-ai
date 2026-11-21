"""The skill compiler: generalise repeated deep runs into versioned skills.

Kosmos Brain v2, Phase 2 "Fast" (V2-13). The depth governor
(:mod:`puspa.governor.governor`) counts consecutive deep
successes and stores the exact steps that succeeded; this
module is the real compiler behind that hook: it generalises
the runs into a reusable procedural skill, versions it,
checks its postconditions on replay, and repairs it when a
postcondition fails instead of dropping it.

Generalisation
---------------
One successful run is a fact; three are a pattern. The
compiler aligns the runs' steps by position and keeps the
longest common prefix (same tool at the same position). Inside
the prefix, an argument that is identical in every run stays
literal; an argument that differs becomes a parameter (a
``{"$param": name}`` placeholder in the step, with the first
run's value as the default in
:attr:`CompiledSkill.parameters
<puspa.governor.skills.CompiledSkill.parameters>`). A
postcondition is an outcome key that is present with the same
value in every run's last result — only an invariant outcome
may be asserted on replay, so a value that varies across runs
is not a postcondition.

Replay
------
A compiled skill replays two ways. With a plain
``step_runner`` (one step dict in, one result dict out — the
same seam the governor's regression uses), the replay runs
the skill's regression case. With a sandbox executor
(:class:`puspa.localexec.LocalExecutor`) and a
``tool_commands`` map, each step runs as a real subprocess
inside the workspace — permission prompt first, stdout parsed
as JSON for the step result — and the postcondition is checked
against the last result. A skill is only stored after its
replay passes, so a stored skill always has a passing replay
behind it.

Repair
-------
A failed postcondition triggers repair, never a silent drop.
The compiler records the failure on the old version (two
consecutive failed replays break it, per the procedural
lifecycle) and then tries, in order:

1. Re-generalise from fresh successful runs: the deep runs
   that succeeded after the failure carry the corrected
   steps and outcomes, so they compile into the next version.
2. Refresh the postcondition: re-run the regression twice;
   when both runs agree on a new value for the failed key and
   the caller accepts that value, the postcondition is updated
   to the observed value in the next version.

When neither works the old version stays stored (marked
failed) and the repair reports why, so a human can look —
the compiler never deletes a skill.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .skills import (
    PARAM_MARKER,
    CompiledSkill,
    RegressionCase,
    SkillStore,
)

if TYPE_CHECKING:
    from puspa.localexec import ExecutionResult, Executor

__all__ = [
    "CompileResult",
    "GeneralisedPlan",
    "RepairResult",
    "ReplayResult",
    "SkillCompiler",
    "SkillRun",
    "generalise_runs",
]

#: What a replay failure means, as :attr:`ReplayResult.kind`.
REPLAY_PASSED = "passed"
REPLAY_NOT_CONFIGURED = "not_configured"
REPLAY_STEP_FAILED = "step_failed"
REPLAY_POSTCONDITION_FAILED = "postcondition_failed"

#: What a repair did, as :attr:`RepairResult.action`.
REPAIR_REGENERALISED = "regeneralised"
REPAIR_POSTCONDITION_REFRESHED = "postcondition_refreshed"
REPAIR_ALREADY_PASSING = "already_passing"
REPAIR_FAILED = "repair_failed"

#: ``{name}`` inside a sandbox command template: replaced with
#: the step argument of that name (see :func:`_render_argv`).
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class SkillRun:
    """One successful deep run, as the compiler generalises from it.

    ``steps`` are the run's steps (each a ``{"tool": ...,
    "arguments": {...}}`` dict); ``last_result`` is the observed
    result of the last step, which the postconditions are mined
    from. A run without a tool name on a step, or without steps
    at all, is a caller bug and raises ValueError.
    """

    steps: list[dict]
    last_result: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.steps, list)
            or not self.steps
            or any(not isinstance(step, dict) for step in self.steps)
        ):
            raise ValueError("a skill run needs a non-empty list of step dicts")
        for index, step in enumerate(self.steps):
            tool = step.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError(f"skill run step {index} needs a non-empty tool name")
        if not isinstance(self.last_result, dict):
            raise ValueError("a skill run's last result must be a dict")


@dataclass(frozen=True)
class GeneralisedPlan:
    """What :func:`generalise_runs` found across the runs.

    ``steps`` are the common-prefix steps with differing
    arguments parameterised; ``parameters`` maps each parameter
    name to its default value (the first run's value);
    ``expect`` is the mined postcondition (invariant outcome
    keys only).
    """

    steps: list[dict]
    parameters: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CompileResult:
    """The verdict of one :meth:`SkillCompiler.compile` call."""

    skill: CompiledSkill | None
    detail: str
    runs_used: int = 0


@dataclass(frozen=True)
class ReplayResult:
    """The verdict of one :meth:`SkillCompiler.replay` call.

    ``kind`` is ``"passed"``, ``"not_configured"`` (no runner
    and no sandbox), ``"step_failed"`` (a step did not run or
    did not return a usable result), or
    ``"postcondition_failed"`` (a step ran but the last result
    broke the postcondition — then ``failed_key`` names the key,
    with the observed value in ``actual`` and the expected value
    in ``expected``). ``executions`` holds the sandbox runs, one
    per replayed step, when the replay ran in the sandbox.
    """

    passed: bool
    steps_run: int
    detail: str
    last_result: dict | None = None
    executions: tuple[ExecutionResult, ...] = ()
    kind: str = REPLAY_PASSED
    failed_key: str | None = None
    expected: object = None
    actual: object = None


@dataclass(frozen=True)
class RepairResult:
    """The verdict of one :meth:`SkillCompiler.repair` call.

    ``action`` is ``"regeneralised"`` (fresh runs compiled into
    the next version), ``"postcondition_refreshed"`` (the stale
    postcondition updated to an observed, accepted value),
    ``"already_passing"`` (the skill passes now; nothing was
    changed), or ``"repair_failed"`` (the old version stays
    stored, marked failed — see ``detail`` for why, and what
    would unblock the repair). ``changes`` records what the
    repair changed, so the repair is auditable: a refreshed
    postcondition lists the key with ``[old, new]`` values.
    """

    skill: CompiledSkill | None
    action: str
    detail: str
    changes: dict = field(default_factory=dict)


def _common_prefix_length(runs: Sequence[SkillRun]) -> int:
    """How many leading steps all runs share (same tool per position)."""
    length = min(len(run.steps) for run in runs)
    for index in range(length):
        tools = [run.steps[index].get("tool") for run in runs]
        if any(tool != tools[0] for tool in tools[1:]):
            return index
    return length


def _param_name(path: list) -> str:
    """The parameter name for a value at ``path`` in a step.

    The dotted path from the step, with ``"arguments"`` segments
    skipped — the tool's arguments are the skill's named inputs,
    so the product argument is just ``"product"``, a nested
    ``filter.store`` stays ``"filter.store"``, and a list
    element is ``"items[0]"``.
    """
    parts: list[str] = []
    for segment in path:
        if segment == "arguments":
            continue
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        else:
            parts.append(segment if not parts else f".{segment}")
    return "".join(parts)


def _generalise_value(path: list, values: list, parameters: dict) -> object:
    """Generalise one value across the runs into a literal or a parameter.

    Equal in every run: the literal value. Dicts: merge key by
    key (keys missing from some run are dropped); lists of the
    same length: merge element by element. Anything else that
    differs becomes a ``{"$param": name}`` placeholder, with the
    first run's value recorded as the parameter default. A dict
    or list with nothing in common falls through to a whole
    parameter for the same reason.
    """
    first = values[0]
    if all(value == first for value in values[1:]):
        return copy.deepcopy(first)
    if all(isinstance(value, dict) for value in values):
        common = set(first.keys())
        for value in values[1:]:
            common &= set(value.keys())
        if common:
            return {
                key: _generalise_value([*path, key], [value[key] for value in values], parameters)
                for key in first
                if key in common
            }
    if all(isinstance(value, list) for value in values) and all(
        len(value) == len(first) for value in values
    ):
        return [
            _generalise_value([*path, index], [value[index] for value in values], parameters)
            for index in range(len(first))
        ]
    name = _param_name(path)
    parameters[name] = copy.deepcopy(first)
    return {PARAM_MARKER: name}


def generalise_runs(runs: Sequence[SkillRun]) -> GeneralisedPlan | None:
    """Generalise successful runs into one reusable plan.

    Returns None when the runs share no common first step —
    runs that start with different tools have no procedure in
    common and cannot compile. Otherwise returns the
    common-prefix steps, the parameters with their defaults,
    and the mined postcondition (outcome keys that are present
    with the same value in every run's last result). Inside a
    step, values are generalised deeply: an argument that is
    identical in every run stays literal, one that differs
    becomes a parameter, and a key that is missing from some
    run is dropped — it is not part of the shared procedure. A
    run with no last result contributes no postcondition keys.
    """
    runs = list(runs)
    if not runs:
        return None
    prefix = _common_prefix_length(runs)
    if prefix == 0:
        return None
    steps: list[dict] = []
    parameters: dict = {}
    for index in range(prefix):
        first = runs[0].steps[index]
        universal = set(first.keys())
        for run in runs[1:]:
            universal &= set(run.steps[index].keys())
        merged: dict = {}
        for key in first:
            if key not in universal:
                continue
            merged[key] = _generalise_value(
                [key], [run.steps[index][key] for run in runs], parameters
            )
        steps.append(merged)
    expect: dict = {}
    if all(run.last_result for run in runs):
        common_keys = set(runs[0].last_result.keys())
        for run in runs[1:]:
            common_keys &= set(run.last_result.keys())
        for key in common_keys:
            values = [run.last_result[key] for run in runs]
            if all(value == values[0] for value in values[1:]):
                expect[key] = copy.deepcopy(values[0])
    return GeneralisedPlan(steps=steps, parameters=parameters, expect=expect)


def _render_argv(template: Sequence[str], arguments: dict) -> list[str]:
    """Fill a sandbox command template from a step's arguments.

    Every ``{name}`` in a template element is replaced with the
    step argument of that name (scalars only: strings, numbers,
    booleans, None). Raises ValueError naming the argument when
    a placeholder has no argument behind it, or the argument is
    not a scalar — a command line cannot carry a dict or a list.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in arguments:
            raise ValueError(
                f"command template needs argument {name!r}, which the step does not supply"
            )
        value = arguments[name]
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        if isinstance(value, (str, int, float)):
            return str(value)
        raise ValueError(
            f"argument {name!r} cannot go in a command: "
            f"{type(value).__name__} values are not supported"
        )

    return [_PLACEHOLDER.sub(_replace, element) for element in template]


class SkillCompiler:
    """Compile repeated deep runs into versioned procedural skills.

    ``skill_store`` is where compiled skills live. With an
    ``executor`` (a :class:`puspa.localexec.Executor
    <puspa.localexec.executor.Executor>`) and a
    ``tool_commands`` map, compilation and replay verify the
    skill by running its steps as real subprocesses in the
    sandbox; without them, verification runs the regression
    through a caller-supplied ``step_runner`` instead. A
    ``tool_commands`` entry maps a tool name to an argv template
    whose ``{name}`` placeholders are filled from the step's
    arguments; the command must print one JSON object on stdout
    — that object is the step result the postcondition is
    checked against.
    """

    def __init__(
        self,
        skill_store: SkillStore,
        *,
        executor: Executor | None = None,
        tool_commands: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if executor is not None and not callable(getattr(executor, "execute", None)):
            raise ValueError("a sandbox executor needs an execute method")
        commands: dict[str, tuple[str, ...]] = {}
        for tool, argv in (tool_commands or {}).items():
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError("a tool command needs a non-empty tool name")
            if isinstance(argv, str) or not isinstance(argv, Sequence) or not argv:
                raise ValueError(f"tool command for {tool!r} needs a non-empty argument list")
            if any(not isinstance(part, str) for part in argv):
                raise ValueError(f"tool command for {tool!r} must be all strings")
            commands[tool] = tuple(argv)
        self._store = skill_store
        self._executor = executor
        self._tool_commands = commands

    # -- compilation ---------------------------------------------------

    def compile(
        self,
        task_key: str,
        runs: Sequence[SkillRun],
        *,
        step_runner: Callable[[dict], dict] | None = None,
        agent_id: str | None = None,
    ) -> CompileResult:
        """Generalise ``runs`` into a skill, verify it, store it.

        The version is one past the newest stored version for
        the task key (1 when there is none). The candidate is
        stored only when its replay passes — in the sandbox when
        this compiler has an executor, else through
        ``step_runner``. A failed replay, or runs with nothing
        in common, stores nothing and reports why.
        """
        if not task_key or not isinstance(task_key, str):
            raise ValueError("compiling a skill needs a non-empty task key")
        runs = list(runs)
        if not runs:
            return CompileResult(
                skill=None, detail="no successful runs to generalise from", runs_used=0
            )
        plan = generalise_runs(runs)
        if plan is None:
            return CompileResult(
                skill=None,
                detail="the runs share no common first step to generalise",
                runs_used=len(runs),
            )
        version = max((item.version for item in self._store.history(task_key)), default=0) + 1
        generalised = CompiledSkill(
            task_key=task_key,
            steps=plan.steps,
            regression=RegressionCase.from_steps(plan.steps, plan.expect),
            version=version,
            parameters=dict(plan.parameters),
        )
        # The stored regression replays concrete steps: bind the
        # generalised skill to its defaults first, then keep the
        # bound regression beside the generalised steps.
        candidate = CompiledSkill(
            task_key=task_key,
            steps=plan.steps,
            regression=generalised.bind().regression,
            version=version,
            parameters=dict(plan.parameters),
        )
        result = self.replay(candidate, step_runner=step_runner)
        if not result.passed:
            return CompileResult(
                skill=None,
                detail=f"compiled skill v{version} failed its replay: {result.detail}",
                runs_used=len(runs),
            )
        stored = self._store.save(candidate, agent_id=agent_id)
        parameter_note = (
            f" with parameters {sorted(stored.parameters)}" if stored.parameters else ""
        )
        replay_note = (
            "replay passed in the sandbox" if self._executor is not None else "replay passed"
        )
        return CompileResult(
            skill=stored,
            detail=(
                f"compiled {task_key!r} v{stored.version} from {len(runs)} runs"
                f"{parameter_note}; {replay_note}"
            ),
            runs_used=len(runs),
        )

    # -- replay --------------------------------------------------------

    def replay(
        self,
        skill: CompiledSkill,
        *,
        step_runner: Callable[[dict], dict] | None = None,
        bindings: dict | None = None,
    ) -> ReplayResult:
        """Replay ``skill`` and check its postcondition.

        ``bindings`` supplies parameter values; omitted ones
        fall back to the skill's defaults. In the sandbox (this
        compiler has an executor) the steps run as subprocesses
        and the executor's permission prompt applies — a denied
        command propagates as PermissionDenied, it is not a
        failed replay. Without an executor the regression runs
        through ``step_runner``; with neither, the replay cannot
        run and reports that.
        """
        try:
            bound = skill.bind(bindings)
        except ValueError as exc:
            return ReplayResult(
                passed=False,
                steps_run=0,
                detail=f"cannot bind skill {skill.task_key!r}: {exc}",
                kind=REPLAY_STEP_FAILED,
            )
        if self._executor is not None:
            return self._replay_in_sandbox(bound)
        if step_runner is None:
            return ReplayResult(
                passed=False,
                steps_run=0,
                detail="no step runner and no sandbox executor configured for replay",
                kind=REPLAY_NOT_CONFIGURED,
            )
        result = bound.run_regression(step_runner)
        kind = (
            REPLAY_PASSED
            if result.passed
            else (
                REPLAY_POSTCONDITION_FAILED if result.failed_key is not None else REPLAY_STEP_FAILED
            )
        )
        return ReplayResult(
            passed=result.passed,
            steps_run=result.steps_run,
            detail=result.detail,
            last_result=result.last_result,
            kind=kind,
            failed_key=result.failed_key,
            expected=result.expected,
            actual=result.actual,
        )

    def _replay_in_sandbox(self, bound: CompiledSkill) -> ReplayResult:
        """Replay bound steps as subprocesses; check the postcondition."""
        executor = self._executor
        if executor is None:
            return ReplayResult(
                passed=False,
                steps_run=0,
                detail="no sandbox executor configured for replay",
                kind=REPLAY_NOT_CONFIGURED,
            )
        executions: list[ExecutionResult] = []
        last_result: dict | None = None
        for index, step in enumerate(bound.steps):
            tool = step.get("tool")
            if not isinstance(tool, str) or not tool:
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"regression step {index} has no tool name",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            template = self._tool_commands.get(tool)
            if template is None:
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"step {index} tool {tool!r} has no sandbox command configured",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            arguments = step.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"step {index} tool {tool!r} arguments are not a dict",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            try:
                argv = _render_argv(template, arguments)
            except ValueError as exc:
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"step {index} tool {tool!r}: {exc}",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            result = executor.execute(argv)
            executions.append(result)
            if result.timed_out:
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"step {index} tool {tool!r} timed out in the sandbox",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            if result.returncode != 0:
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=(
                        f"step {index} tool {tool!r} exited {result.returncode} "
                        f"in the sandbox: {result.stderr.strip()[-300:]}"
                    ),
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            try:
                parsed = json.loads(result.stdout)
            except json.JSONDecodeError:
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"step {index} tool {tool!r} did not print JSON in the sandbox",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            if not isinstance(parsed, dict):
                return ReplayResult(
                    passed=False,
                    steps_run=index,
                    detail=f"step {index} tool {tool!r} printed JSON that is not an object",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_STEP_FAILED,
                )
            last_result = parsed
        if last_result is None:  # pragma: no cover - steps are non-empty by construction
            return ReplayResult(
                passed=False,
                steps_run=0,
                detail="the replay ran no steps",
                executions=tuple(executions),
                kind=REPLAY_STEP_FAILED,
            )
        for key, expected in bound.regression.expect.items():
            if key not in last_result:
                return ReplayResult(
                    passed=False,
                    steps_run=len(bound.steps),
                    detail=f"postcondition key {key!r} is missing from the last result",
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_POSTCONDITION_FAILED,
                    failed_key=key,
                    expected=expected,
                    actual=None,
                )
            if last_result[key] != expected:
                return ReplayResult(
                    passed=False,
                    steps_run=len(bound.steps),
                    detail=(
                        f"postcondition key {key!r} is {last_result[key]!r}, expected {expected!r}"
                    ),
                    last_result=last_result,
                    executions=tuple(executions),
                    kind=REPLAY_POSTCONDITION_FAILED,
                    failed_key=key,
                    expected=expected,
                    actual=last_result[key],
                )
        return ReplayResult(
            passed=True,
            steps_run=len(bound.steps),
            detail="all sandbox steps ran and the postcondition holds",
            last_result=last_result,
            executions=tuple(executions),
            kind=REPLAY_PASSED,
        )

    # -- repair ----------------------------------------------------------

    def repair(
        self,
        skill: CompiledSkill,
        *,
        runs: Sequence[SkillRun] | None = None,
        step_runner: Callable[[dict], dict] | None = None,
        accept: Callable[[str, object, object], bool] | None = None,
        agent_id: str | None = None,
    ) -> RepairResult:
        """Repair a skill whose postcondition failed on replay.

        The failure is recorded on the old version first, so the
        repair is never silent: two consecutive failed replays
        break the old version, and it stays stored as history.
        With fresh successful ``runs`` the skill is
        re-generalised into the next version. Otherwise, when
        ``accept`` is given, the postcondition is refreshed from
        two agreeing replays (see
        :meth:`_refresh_postcondition`). With neither, the repair
        cannot run and reports what it needs.
        """
        self._store.mark_failed(skill, agent_id=agent_id)
        fresh = list(runs or [])
        if fresh:
            compiled = self.compile(
                skill.task_key, fresh, step_runner=step_runner, agent_id=agent_id
            )
            if compiled.skill is not None:
                return RepairResult(
                    skill=compiled.skill,
                    action=REPAIR_REGENERALISED,
                    detail=(
                        f"repaired {skill.task_key!r} by re-generalisng "
                        f"{len(fresh)} fresh runs into v{compiled.skill.version}"
                    ),
                    changes={"version": [skill.version, compiled.skill.version]},
                )
            return RepairResult(
                skill=None,
                action=REPAIR_FAILED,
                detail=(
                    f"re-generalisng {len(fresh)} fresh runs did not produce "
                    f"a passing skill: {compiled.detail}"
                ),
            )
        if accept is not None:
            return self._refresh_postcondition(
                skill, step_runner=step_runner, accept=accept, agent_id=agent_id
            )
        return RepairResult(
            skill=None,
            action=REPAIR_FAILED,
            detail=(
                "repair needs fresh successful runs, or an acceptance check "
                "for the new postcondition values"
            ),
        )

    def _refresh_postcondition(
        self,
        skill: CompiledSkill,
        *,
        step_runner: Callable[[dict], dict] | None,
        accept: Callable[[str, object, object], bool],
        agent_id: str | None,
    ) -> RepairResult:
        """Refresh one stale postcondition from two agreeing replays.

        Replays the skill twice. When the first replay already
        passes, there is nothing to repair. When the failure is
        not a postcondition failure — a step itself broke — the
        postcondition cannot be refreshed, because there is no
        trustworthy observed value. When the two replays disagree
        on the observed value, the value is not stable enough to
        bless. Otherwise the caller accepts the new value, the
        postcondition is updated in the next version, and that
        version is verified by replay before it is stored.
        """
        first = self.replay(skill, step_runner=step_runner)
        if first.passed:
            return RepairResult(
                skill=skill,
                action=REPAIR_ALREADY_PASSING,
                detail=(
                    f"skill {skill.task_key!r} v{skill.version} passes its "
                    "postcondition now; nothing to repair"
                ),
            )
        if first.kind != REPLAY_POSTCONDITION_FAILED or first.failed_key is None:
            return RepairResult(
                skill=None,
                action=REPAIR_FAILED,
                detail=(
                    "cannot refresh the postcondition: the replay fails "
                    f"before the postcondition: {first.detail}"
                ),
            )
        key = first.failed_key
        if first.last_result is None or key not in first.last_result:
            return RepairResult(
                skill=None,
                action=REPAIR_FAILED,
                detail=(
                    f"postcondition key {key!r} is missing from the replay "
                    "result, so there is no observed value to refresh to"
                ),
            )
        second = self.replay(skill, step_runner=step_runner)
        if second.passed:
            return RepairResult(
                skill=skill,
                action=REPAIR_ALREADY_PASSING,
                detail="the postcondition passes on a second replay; the failure did not repeat",
            )
        if (
            second.kind != REPLAY_POSTCONDITION_FAILED
            or second.failed_key != key
            or second.actual != first.actual
        ):
            return RepairResult(
                skill=None,
                action=REPAIR_FAILED,
                detail=(
                    f"the observed value for postcondition key {key!r} changed "
                    "between replays, so no single new value is trustworthy"
                ),
            )
        old = first.expected
        new = first.actual
        if not accept(key, old, new):
            return RepairResult(
                skill=None,
                action=REPAIR_FAILED,
                detail=(
                    f"the acceptance check refused the new postcondition value "
                    f"for {key!r} ({old!r} -> {new!r})"
                ),
            )
        version = max((item.version for item in self._store.history(skill.task_key)), default=0) + 1
        if version <= skill.version:
            version = skill.version + 1
        expect = dict(skill.regression.expect)
        expect[key] = new
        refreshed = CompiledSkill(
            task_key=skill.task_key,
            steps=[dict(step) for step in skill.steps],
            regression=RegressionCase.from_steps(
                [dict(step) for step in skill.regression.steps], expect
            ),
            skill_id=skill.skill_id,
            status=skill.status,
            version=version,
            parameters=dict(skill.parameters),
        )
        check = self.replay(refreshed, step_runner=step_runner)
        if not check.passed:
            return RepairResult(
                skill=None,
                action=REPAIR_FAILED,
                detail=f"the refreshed skill v{version} still fails its replay: {check.detail}",
            )
        stored = self._store.save(refreshed, agent_id=agent_id)
        return RepairResult(
            skill=stored,
            action=REPAIR_POSTCONDITION_REFRESHED,
            detail=(f"postcondition {key!r} refreshed {old!r} -> {new!r} in v{stored.version}"),
            changes={"postcondition": {key: [old, new]}},
        )
