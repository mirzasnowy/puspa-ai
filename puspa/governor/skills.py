"""Compiled skills: what the depth governor promotes a repeated deep task into.

A compiled skill is a deep plan that succeeded repeatedly, stored
as a procedure so the next similar task runs as reflex instead of
deep ("deep once, reflex afterwards", Kosmos Brain v2, Phase 2).
This module is only the storage shape and the regression check —
the governor decides *when* to compile (see
:mod:`puspa.governor.governor`); the skill compiler that
generalises plans into reusable versioned skills lives in
:mod:`puspa.governor.compiler` (V2-13).

Two stores implement the same :class:`SkillStore` seam:

- :class:`InMemorySkillStore`: a dict, for unit tests and for
  callers with no user and no database.
- :class:`ProceduralSkillStore`: the durable store, an adapter
  over the memory procedural layer
  (:mod:`puspa.memory.procedural`, revision 7 section 3.3). A
  compiled skill is a trial procedure under the ``compiled:``
  service prefix, with its regression case kept in the
  procedure's ``required_checks`` as one ``{"kind":
  "regression", ...}`` entry, so the check travels with the
  skill and a later reader can re-run it. The procedural
  lifecycle still applies: trial until 3 successful runs by 2
  agents, broken after 2 consecutive failures, never used once
  broken.

A skill is only stored after its regression case passes (the
governor runs the regression on the candidate first and stores
it only on a pass), so a stored skill always has a passing
regression behind it. The regression itself replays the case's
steps through a caller-supplied ``step_runner`` — one step dict
in, one result dict out, the same seam the turn loop's tool
registry offers — and checks the last result against the
expected postcondition. A step result that is not a dict, or a
postcondition key that is missing or different, fails the
regression. A ``step_runner`` that raises is the caller's bug,
not a failed regression: the exception propagates, so a broken
runner is never mistaken for a broken skill.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from puspa.memory import procedural as procedural_store

__all__ = [
    "COMPILED_SERVICE_PREFIX",
    "PARAM_MARKER",
    "CompiledSkill",
    "InMemorySkillStore",
    "ProceduralSkillStore",
    "RegressionCase",
    "RegressionResult",
    "SkillStore",
]

#: Service prefix for governor-compiled procedures in the
#: procedural store. The task key follows, so one task key maps
#: to exactly one service.
COMPILED_SERVICE_PREFIX = "compiled:"

#: Marker inside a procedure's ``required_checks`` that holds a
#: compiled skill's regression case.
_REGRESSION_CHECK_KIND = "regression"


def _is_steps(steps: object) -> bool:
    """True when ``steps`` is a non-empty list of dicts."""
    return (
        isinstance(steps, list) and len(steps) > 0 and all(isinstance(step, dict) for step in steps)
    )


#: Marker for a generalised argument: ``{"$param": name}`` in a
#: step stands for the parameter ``name`` (see
#: :meth:`CompiledSkill.bind`).
PARAM_MARKER = "$param"


def _bind_value(value: object, bindings: dict, *, where: str) -> object:
    """Replace ``{"$param": name}`` placeholders with bound values.

    Walks dicts and lists; anything else passes through
    unchanged. Raises ValueError naming the parameter when a
    placeholder has no binding, and when a placeholder is not a
    ``{"$param": name}`` dict with a non-empty string name.
    """
    if isinstance(value, dict):
        if set(value.keys()) == {PARAM_MARKER}:
            name = value[PARAM_MARKER]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"a parameter placeholder at {where} needs a non-empty name")
            if name not in bindings:
                raise ValueError(f"parameter {name!r} at {where} has no binding")
            return bindings[name]
        return {key: _bind_value(item, bindings, where=where) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind_value(item, bindings, where=where) for item in value]
    return value


def _parameters_used(steps: list[dict]) -> set[str]:
    """The parameter names the generalised ``steps`` refer to."""

    def _collect(value: object, found: set[str]) -> None:
        if isinstance(value, dict):
            if set(value.keys()) == {PARAM_MARKER} and isinstance(value[PARAM_MARKER], str):
                found.add(value[PARAM_MARKER])
            else:
                for item in value.values():
                    _collect(item, found)
        elif isinstance(value, list):
            for item in value:
                _collect(item, found)

    used: set[str] = set()
    for step in steps:
        _collect(step, used)
    return used


@dataclass(frozen=True)
class RegressionCase:
    """The check a compiled skill must pass before it may be stored.

    ``steps`` are the replayed steps (usually the deep plan's own
    steps), each a ``{"tool": ..., "arguments": {...}}`` dict.
    ``expect`` is the postcondition: expected key/value pairs in
    the LAST step's result. An empty ``expect`` means only "the
    steps run to completion" is checked.
    """

    steps: list[dict]
    expect: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_steps(self.steps):
            raise ValueError("a regression case needs a non-empty list of step dicts")
        if not isinstance(self.expect, dict):
            raise ValueError("a regression case's expected postcondition must be a dict")

    @classmethod
    def from_steps(cls, steps: list[dict], expect: dict | None = None) -> RegressionCase:
        """Build a case that replays ``steps`` and checks ``expect``."""
        return cls(steps=list(steps), expect=dict(expect) if expect is not None else {})

    def to_dict(self) -> dict:
        """JSON-safe dict of this case (how it is stored)."""
        return {"steps": [dict(step) for step in self.steps], "expect": dict(self.expect)}

    @classmethod
    def from_dict(cls, data: dict) -> RegressionCase:
        """Rebuild a case from :meth:`to_dict` output; ValueError when malformed."""
        if not isinstance(data, dict):
            raise ValueError("a regression case must be a dict")
        return cls(steps=data.get("steps", []), expect=data.get("expect", {}))


@dataclass(frozen=True)
class RegressionResult:
    """The verdict of one regression run: passed or not, and why.

    When the run fails on a postcondition, ``failed_key`` names
    the key, with the observed value in ``actual`` and the
    expected value in ``expected`` — the compiler reads those
    three to decide whether the skill can be repaired, instead
    of parsing the human-readable ``detail``.
    """

    passed: bool
    steps_run: int
    detail: str
    last_result: dict | None = None
    failed_key: str | None = None
    expected: object = None
    actual: object = None

    def to_dict(self) -> dict:
        """JSON-safe dict of this verdict."""
        return {
            "passed": self.passed,
            "steps_run": self.steps_run,
            "detail": self.detail,
            "last_result": self.last_result,
            "failed_key": self.failed_key,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class CompiledSkill:
    """A deep plan compiled into a reflex skill.

    ``task_key`` names the task (one key, one skill). ``steps``
    are the plan's steps — after the V2-13 compiler, generalised:
    an argument that differed across the runs it was compiled
    from is a ``{"$param": name}`` placeholder, and
    ``parameters`` maps each such name to its default value (the
    first run's value). ``regression`` is the case the skill
    must pass. ``skill_id`` is set by the store on save (None
    while the skill is still a candidate). ``status`` is
    ``"candidate"`` until stored, then the procedural status
    (``"trial"`` at first, see the module docstring).
    ``version`` counts the skill's revisions for its task key:
    1 at first compile, +1 on every repair or re-compilation
    that changes it. Older versions stay stored as history;
    a re-save that changes nothing keeps the same version.
    """

    task_key: str
    steps: list[dict]
    regression: RegressionCase
    skill_id: str | None = None
    status: str = "candidate"
    version: int = 1
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_key or not isinstance(self.task_key, str):
            raise ValueError("a compiled skill needs a non-empty task key")
        if not _is_steps(self.steps):
            raise ValueError("a compiled skill needs a non-empty list of step dicts")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ValueError(f"a compiled skill version must be an integer, got {self.version!r}")
        if self.version < 1:
            raise ValueError(f"a compiled skill version must be at least 1, got {self.version}")
        if not isinstance(self.parameters, dict):
            raise ValueError("a compiled skill's parameters must be a dict")

    def run_regression(self, step_runner: Callable[[dict], dict]) -> RegressionResult:
        """Replay the regression case and check its postcondition.

        ``step_runner`` takes one step dict and returns one result
        dict. The verdict is ``passed`` only when every step ran
        and the last result carries every expected key with the
        expected value. A non-dict step result fails the run with
        the step named; a ``step_runner`` that raises propagates
        (see the module docstring).
        """
        last_result: dict | None = None
        for index, step in enumerate(self.regression.steps):
            result = step_runner(step)
            if not isinstance(result, dict):
                return RegressionResult(
                    passed=False,
                    steps_run=index,
                    detail=(
                        f"regression step {index} ({step.get('tool', '?')!r}) "
                        "returned a non-dict result, so the postcondition "
                        "cannot be checked"
                    ),
                    last_result=None,
                )
            last_result = result
        if last_result is None:  # pragma: no cover - guarded by __post_init__
            return RegressionResult(
                passed=False,
                steps_run=0,
                detail="regression case has no steps",
            )
        for key, expected in self.regression.expect.items():
            if key not in last_result:
                return RegressionResult(
                    passed=False,
                    steps_run=len(self.regression.steps),
                    detail=f"postcondition key {key!r} is missing from the last result",
                    last_result=last_result,
                    failed_key=key,
                    expected=expected,
                    actual=None,
                )
            if last_result[key] != expected:
                return RegressionResult(
                    passed=False,
                    steps_run=len(self.regression.steps),
                    detail=(
                        f"postcondition key {key!r} is {last_result[key]!r}, expected {expected!r}"
                    ),
                    last_result=last_result,
                    failed_key=key,
                    expected=expected,
                    actual=last_result[key],
                )
        return RegressionResult(
            passed=True,
            steps_run=len(self.regression.steps),
            detail="all regression steps ran and the postcondition holds",
            last_result=last_result,
        )

    def bind(self, bindings: dict | None = None) -> CompiledSkill:
        """Return this skill with its parameters replaced by values.

        ``bindings`` maps parameter names to concrete values;
        omitted names fall back to this skill's
        ``parameters`` defaults. A binding for an unknown
        parameter, or a parameter used in the steps with no
        binding at all, raises ValueError. The regression case
        is re-derived from the bound steps (same postcondition),
        so the bound skill's regression replays exactly the
        steps the bound skill would run.
        """
        merged = dict(self.parameters)
        for name, value in (bindings or {}).items():
            if name not in self.parameters:
                raise ValueError(f"unknown parameter {name!r} for skill {self.task_key!r}")
            merged[name] = value
        used = _parameters_used(self.steps)
        missing = sorted(used - set(merged))
        if missing:
            raise ValueError(f"skill {self.task_key!r} needs bindings for parameters {missing}")
        bound_steps = [
            {
                key: _bind_value(item, merged, where=f"step {index} {key!r}")
                for key, item in step.items()
            }
            for index, step in enumerate(self.steps)
        ]
        return CompiledSkill(
            task_key=self.task_key,
            steps=bound_steps,
            regression=RegressionCase.from_steps(bound_steps, self.regression.expect),
            skill_id=self.skill_id,
            status=self.status,
            version=self.version,
            parameters=dict(self.parameters),
        )

    def to_dict(self) -> dict:
        """JSON-safe dict of this skill."""
        return {
            "task_key": self.task_key,
            "skill_id": self.skill_id,
            "status": self.status,
            "version": self.version,
            "parameters": dict(self.parameters),
            "steps": [dict(step) for step in self.steps],
            "regression": self.regression.to_dict(),
        }


class SkillStore(Protocol):
    """Where compiled skills live: save one, find one by task key.

    A store keeps every version it is given: ``find`` returns
    the newest version that is not broken, ``history`` returns
    all of them oldest first. ``mark_failed`` records one failed
    replay of a stored skill; the default does nothing, and a
    store that tracks runs breaks the skill after 2 consecutive
    failures (the procedural lifecycle: a broken skill is never
    used again, but it is kept — repair replaces it with a new
    version, it is never silently deleted).
    """

    def save(self, skill: CompiledSkill, *, agent_id: str | None = None) -> CompiledSkill:
        """Store ``skill`` and return it with its id set."""
        ...

    def find(self, task_key: str) -> CompiledSkill | None:
        """The stored skill for ``task_key``, or None when there is none."""
        ...

    def history(self, task_key: str) -> list[CompiledSkill]:
        """Every stored version for ``task_key``, oldest first."""
        found = self.find(task_key)
        return [found] if found is not None else []

    def mark_failed(self, skill: CompiledSkill, *, agent_id: str | None = None) -> None:
        """Record one failed replay of a stored skill (default: no-op)."""


class InMemorySkillStore:
    """A dict-backed skill store (unit tests, no user, no database).

    Keeps every version per task key, newest first internally;
    ``find`` returns the newest version that is not broken.
    ``mark_failed`` counts consecutive failed replays per task
    key (reset when a new version is saved) and breaks the
    newest version after 2, mirroring the procedural lifecycle.
    """

    #: Consecutive failed replays after which a version breaks.
    BROKEN_AFTER_FAILURES = 2

    def __init__(self) -> None:
        self._skills: dict[str, list[CompiledSkill]] = {}
        self._failures: dict[str, int] = {}

    def save(self, skill: CompiledSkill, *, agent_id: str | None = None) -> CompiledSkill:
        """Store ``skill`` under its task key.

        A re-save that changes nothing (same steps, regression
        and parameters as the newest version) returns the
        newest version unchanged — it is another successful run
        of the same skill, not a new version. Anything else is
        stored as a new version: the skill's own version when
        it is ahead of the newest stored one, otherwise newest
        version + 1.
        """
        versions = self._skills.setdefault(skill.task_key, [])
        latest = versions[0] if versions else None
        if (
            latest is not None
            and latest.steps == [dict(step) for step in skill.steps]
            and latest.regression.to_dict() == skill.regression.to_dict()
            and latest.parameters == dict(skill.parameters)
        ):
            return latest
        version = skill.version
        if latest is not None and version <= latest.version:
            version = latest.version + 1
        stored = CompiledSkill(
            task_key=skill.task_key,
            steps=[dict(step) for step in skill.steps],
            regression=skill.regression,
            skill_id=f"skill:{skill.task_key}",
            status="trial",
            version=version,
            parameters=dict(skill.parameters),
        )
        versions.insert(0, stored)
        self._failures[skill.task_key] = 0
        return stored

    def find(self, task_key: str) -> CompiledSkill | None:
        """The newest non-broken stored skill for ``task_key``, or None."""
        for stored in self._skills.get(task_key, []):
            if stored.status != "broken":
                return stored
        return None

    def history(self, task_key: str) -> list[CompiledSkill]:
        """Every stored version for ``task_key``, oldest first."""
        return list(reversed(self._skills.get(task_key, [])))

    def mark_failed(self, skill: CompiledSkill, *, agent_id: str | None = None) -> None:
        """Count one failed replay; break the version after 2 in a row."""
        versions = self._skills.get(skill.task_key, [])
        target = next((item for item in versions if item.version == skill.version), None)
        if target is None:
            return
        count = self._failures.get(skill.task_key, 0) + 1
        self._failures[skill.task_key] = count
        if count < self.BROKEN_AFTER_FAILURES:
            return
        index = versions.index(target)
        versions[index] = CompiledSkill(
            task_key=target.task_key,
            steps=[dict(step) for step in target.steps],
            regression=target.regression,
            skill_id=target.skill_id,
            status="broken",
            version=target.version,
            parameters=dict(target.parameters),
        )


class ProceduralSkillStore:
    """A durable skill store over the memory procedural layer.

    ``user_id`` owns the procedures; ``dsn`` is passed through
    to the procedural functions (None means "read it from the
    environment", exactly as the procedural layer does — no
    database name is hardcoded here).
    """

    def __init__(self, *, user_id: str, dsn: str | None = None) -> None:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("a procedural skill store needs a non-empty user id")
        self._user_id = user_id
        self._dsn = dsn

    @staticmethod
    def _service(task_key: str) -> str:
        return f"{COMPILED_SERVICE_PREFIX}{task_key}"

    def save(self, skill: CompiledSkill, *, agent_id: str | None = None) -> CompiledSkill:
        """Store ``skill`` as a trial procedure; return it with its id set.

        The regression case travels in ``required_checks`` so a
        later reader can re-run it from the stored row alone,
        with the skill's parameters beside it. A re-save that
        changes nothing counts another successful run on the
        newest row (the procedural promotion path); anything
        else is inserted as a new row with the next version
        number, so a repair never overwrites the version it
        replaces. The returned skill carries the stored row's
        version and status.
        """
        check = _skill_check(skill)
        existing = self.find(skill.task_key)
        if (
            existing is not None
            and existing.steps == [dict(step) for step in skill.steps]
            and existing.regression.to_dict() == skill.regression.to_dict()
            and existing.parameters == dict(skill.parameters)
        ):
            row = procedural_store.record_successful_task(
                self._user_id,
                self._service(skill.task_key),
                [dict(step) for step in skill.steps],
                agent_id=agent_id,
                required_checks=[check],
                dsn=self._dsn,
            )
        else:
            history = self.history(skill.task_key)
            version = max([item.version for item in history], default=0) + 1
            version = max(version, skill.version)
            row = procedural_store.insert_procedure(
                self._user_id,
                self._service(skill.task_key),
                [dict(step) for step in skill.steps],
                version=version,
                agent_id=agent_id,
                required_checks=[check],
                dsn=self._dsn,
            )
        return CompiledSkill(
            task_key=skill.task_key,
            steps=[dict(step) for step in skill.steps],
            regression=skill.regression,
            skill_id=str(row["id"]),
            status=str(row["status"]),
            version=int(row["version"]),
            parameters=dict(skill.parameters),
        )

    def find(self, task_key: str) -> CompiledSkill | None:
        """The newest non-broken stored skill for ``task_key``, or None.

        A row whose ``required_checks`` hold no regression entry
        is not a governor-compiled skill and is skipped, so a
        procedure another path wrote under the same service prefix
        can never be mistaken for one. A broken row is skipped
        too: broken procedures are never used again, but the row
        stays, so a repair can still read it from the history.
        """
        for stored in reversed(self.history(task_key)):
            if stored.status != "broken":
                return stored
        return None

    def history(self, task_key: str) -> list[CompiledSkill]:
        """Every stored version for ``task_key``, oldest first."""
        service = self._service(task_key)
        found: list[CompiledSkill] = []
        for row in procedural_store.list_procedures(self._user_id, dsn=self._dsn):
            if row["service"] != service:
                continue
            stored = _skill_from_row(task_key, row)
            if stored is not None:
                found.append(stored)
        found.sort(key=lambda item: item.version)
        return found

    def mark_failed(self, skill: CompiledSkill, *, agent_id: str | None = None) -> None:
        """Record one failed replay on the stored procedure row.

        Follows the procedural lifecycle: two consecutive
        failures break the row. A skill id this store does not
        own is ignored — the skill lives in another store.
        """
        if skill.skill_id is None:
            return
        try:
            procedural_store.mark_run(
                skill.skill_id, self._user_id, success=False, agent_id=agent_id, dsn=self._dsn
            )
        except KeyError:
            return


def _skill_check(skill: CompiledSkill) -> dict:
    """The regression entry stored in a procedure's checks."""
    return {
        "kind": _REGRESSION_CHECK_KIND,
        "parameters": dict(skill.parameters),
        **skill.regression.to_dict(),
    }


def _skill_from_row(task_key: str, row: dict) -> CompiledSkill | None:
    """The compiled skill a procedure row holds, or None.

    Returns None when the row's checks hold no regression
    entry, or the stored steps are malformed — a malformed
    stored check is skipped, never trusted.
    """
    regression = _regression_from_checks(row.get("required_checks"))
    if regression is None:
        return None
    steps = row["step"]
    if not _is_steps(steps):
        return None
    return CompiledSkill(
        task_key=task_key,
        steps=[dict(step) for step in steps],
        regression=regression,
        skill_id=str(row["id"]),
        status=str(row["status"]),
        version=int(row["version"]),
        parameters=_parameters_from_checks(row.get("required_checks")),
    )


def _regression_from_checks(required_checks: object) -> RegressionCase | None:
    """The regression case inside a procedure's checks, or None.

    Returns None when the checks are not a list, hold no
    ``"regression"`` entry, or hold one that does not parse —
    a malformed stored check is skipped, never trusted.
    """
    if not isinstance(required_checks, list):
        return None
    for check in required_checks:
        if not isinstance(check, dict) or check.get("kind") != _REGRESSION_CHECK_KIND:
            continue
        try:
            return RegressionCase.from_dict(check)
        except ValueError:
            return None
    return None


def _parameters_from_checks(required_checks: object) -> dict:
    """The skill parameters inside a procedure's checks, or {}.

    A missing or malformed parameters entry means "no
    parameters" — the V2-12 rows were stored before parameters
    existed, and they read back as plain literal skills.
    """
    if not isinstance(required_checks, list):
        return {}
    for check in required_checks:
        if not isinstance(check, dict) or check.get("kind") != _REGRESSION_CHECK_KIND:
            continue
        parameters = check.get("parameters", {})
        return dict(parameters) if isinstance(parameters, dict) else {}
    return {}
