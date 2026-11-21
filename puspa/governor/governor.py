"""The depth governor (Kosmos Brain v2, Phase 2 "Fast").

One loop, many depths: the governor decides how much thinking
each step gets — reflex, deliberate, or deep — and nothing else.
It never executes a step, never calls a model, and never touches
the network or the database itself (the skill store it compiles
into is injected; the default is in-memory).

Where the inputs come from
---------------------------
- ``uncertainty``: the K20 metacognition confidence gap (0 =
  certain, 1 = guessing). K20 is a later phase, so until it
  lands the caller supplies its own estimate; the governor only
  reads the number.
- ``stakes``: what the step can cost if it goes wrong (0 =
  nothing, 1 = irreversible harm). K1 rates difficulty and
  stakes for the intent; the caller passes the rating in.
- ``has_evidence``: whether the world model holds anything
  usable for this step — the K6 ignorance map
  (:mod:`puspa.world.ignorance`): a belief with no usable value
  counts as no evidence.
- ``skill_match`` / ``skill_id``: whether a stored skill matches
  this step, and how confidently. The governor can fill these in
  from its own skill store when the caller passes ``task_key``.
- ``previous_failure``: whether the step already failed once at
  a lower depth.
- A failed verification reaches the governor through
  :meth:`DepthGovernor.escalate`, which the caller (the outcome
  watcher, the sentinel verifier, or the turn loop) calls with
  the failure. The governor never verifies anything itself.

The rules, in order (thresholds are constructor arguments —
initial values to tune from traces, per the design doc):

1. A previous failure goes deep: the step already failed once,
   so it gets maximum thinking.
2. Reflex: a stored skill matches with high confidence, the
   step is reversible, and stakes and uncertainty are both low.
   A matching skill counts as evidence for its own step.
3. Deep: uncertainty times stakes reaches the deep bar, or the
   ignorance map shows no evidence, or the step is irreversible
   at high stakes.
4. Deliberate otherwise: the default, one plan plus one
   self-critique plus pre-flight.

Escalation is exactly one retry, one level up: reflex retries
as deliberate, deliberate as deep, and a failed deep step has
nowhere higher to go — the governor returns None and the caller
must ask the user or fail the step. The failure context travels
on the escalated decision so the retry can use it. A retry that
already ran once is never retried again.

De-escalation is compilation: :meth:`DepthGovernor.record_outcome`
counts consecutive deep successes per task key, and when the
count reaches the compile bar it builds the skill candidate,
runs its regression case, and stores it only when the regression
passes. A stored skill is then matched by :meth:`decide` on the
next similar task, so the step runs as reflex — deep once,
reflex afterwards. With a skill compiler attached (V2-13,
:mod:`puspa.governor.compiler`), the governor generalises the
streak's runs into a versioned skill instead of storing the
exact steps: differing arguments become parameters, invariant
outcomes become postconditions, and the compiled skill replays
in the sandbox before it is stored. Without a compiler the hook
keeps its original behaviour — it compiles the exact steps that
succeeded, with their regression case attached.

What the governor does NOT decide: which model tier runs a
depth (the router owns that), whether the step may run at all
(the policy gateway and the K13 conflict gate own that), and
whether to act or ask (K2 owns that). The live turn loop does
not consult the governor yet: the loop cannot branch on depth
until the router maps depths to model tiers, so that wiring
lands with the tier mapping, not here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from .compiler import SkillCompiler, SkillRun
from .skills import CompiledSkill, InMemorySkillStore, RegressionCase, SkillStore

__all__ = [
    "COMPILE_AFTER_SUCCESSES",
    "DEEP_MIN_STAKES_IRREVERSIBLE",
    "DEEP_RISK",
    "REFLEX_MAX_STAKES",
    "REFLEX_MAX_UNCERTAINTY",
    "REFLEX_SKILL_MATCH",
    "Depth",
    "DepthDecision",
    "DepthGovernor",
    "StepSignal",
]

#: A stored skill must match at least this confidently to run as reflex.
REFLEX_SKILL_MATCH = 0.8
#: Reflex needs stakes at or below this.
REFLEX_MAX_STAKES = 0.3
#: Reflex needs uncertainty at or below this.
REFLEX_MAX_UNCERTAINTY = 0.3
#: Uncertainty times stakes at or above this goes deep.
DEEP_RISK = 0.5
#: An irreversible step at or above these stakes goes deep.
DEEP_MIN_STAKES_IRREVERSIBLE = 0.5
#: Consecutive deep successes after which a task compiles to a skill.
#: Matches the procedural promotion bar (3 runs, revision 7 section 3.3).
COMPILE_AFTER_SUCCESSES = 3


class Depth(StrEnum):
    """How much thinking one step gets."""

    REFLEX = "reflex"
    DELIBERATE = "deliberate"
    DEEP = "deep"

    def one_level_up(self) -> Depth | None:
        """The next depth up, or None when already at the top."""
        if self is Depth.REFLEX:
            return Depth.DELIBERATE
        if self is Depth.DELIBERATE:
            return Depth.DEEP
        return None


def _check_unit(name: str, value: float) -> float:
    """Validate a 0..1 value; ValueError with the name and value otherwise."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number within 0..1, got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within 0..1, got {value}")
    return float(value)


@dataclass(frozen=True)
class StepSignal:
    """What the governor reads to pick a depth for one step.

    ``uncertainty`` is the confidence gap (0 = certain, 1 =
    guessing). ``stakes`` is what the step can cost (0 =
    nothing, 1 = irreversible harm). ``reversible`` is whether
    the step can be undone. ``skill_match`` is the confidence a
    stored skill matches this step (None = no skill matches);
    ``skill_id`` names that skill and is required whenever a
    match is claimed. ``has_evidence`` is False when the
    ignorance map shows no usable evidence for this step.
    ``previous_failure`` is True when the step already failed
    once at a lower depth.
    """

    uncertainty: float = 0.5
    stakes: float = 0.5
    reversible: bool = True
    skill_match: float | None = None
    skill_id: str | None = None
    has_evidence: bool = True
    previous_failure: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "uncertainty", _check_unit("uncertainty", self.uncertainty))
        object.__setattr__(self, "stakes", _check_unit("stakes", self.stakes))
        if self.skill_match is not None:
            object.__setattr__(self, "skill_match", _check_unit("skill_match", self.skill_match))
        if self.skill_id is not None and self.skill_match is None:
            raise ValueError("a skill id without a match confidence names no match")
        if self.skill_id is not None and not self.skill_id.strip():
            raise ValueError("a skill id must not be blank")

    @property
    def risk(self) -> float:
        """Uncertainty times stakes: the number the deep bar reads."""
        return self.uncertainty * self.stakes


@dataclass(frozen=True)
class DepthDecision:
    """One depth pick, with the reason so the ledger can quote it."""

    depth: Depth
    reason: str
    risk: float
    skill_id: str | None = None
    failure_context: str | None = None
    retried: bool = False

    def to_dict(self) -> dict:
        """JSON-safe dict of this decision (ledger/audit shape)."""
        return {
            "depth": self.depth.value,
            "reason": self.reason,
            "risk": self.risk,
            "skill_id": self.skill_id,
            "failure_context": self.failure_context,
            "retried": self.retried,
        }


@dataclass
class DepthGovernor:
    """Picks a depth per step, escalates on failed verification, compiles repeats.

    ``skill_store`` is where compiled skills live (default: an
    in-memory store; pass :class:`ProceduralSkillStore
    <puspa.governor.skills.ProceduralSkillStore>` for the durable
    one). ``compiler`` is the V2-13 skill compiler
    (:class:`SkillCompiler <puspa.governor.compiler.SkillCompiler>`);
    when it is set, :meth:`record_outcome` generalises the
    streak's runs into a versioned skill through it, and the
    compiler must share this governor's skill store. The
    remaining arguments are the thresholds from the module
    docstring — initial values to tune from traces.
    """

    skill_store: SkillStore = field(default_factory=InMemorySkillStore)
    compiler: SkillCompiler | None = None
    compile_after: int = COMPILE_AFTER_SUCCESSES
    reflex_skill_match: float = REFLEX_SKILL_MATCH
    reflex_max_stakes: float = REFLEX_MAX_STAKES
    reflex_max_uncertainty: float = REFLEX_MAX_UNCERTAINTY
    deep_risk: float = DEEP_RISK
    deep_min_stakes_irreversible: float = DEEP_MIN_STAKES_IRREVERSIBLE

    def __post_init__(self) -> None:
        if self.compile_after < 1:
            raise ValueError(f"compile_after must be at least 1, got {self.compile_after}")
        self._deep_successes: dict[str, int] = {}
        self._runs: dict[str, list[SkillRun]] = {}

    # -- picking -------------------------------------------------------

    def decide(self, signal: StepSignal, *, task_key: str | None = None) -> DepthDecision:
        """Pick reflex, deliberate, or deep for one step.

        When ``task_key`` is given and the signal claims no skill
        match, the governor looks the key up in its skill store
        first: a stored skill for the exact key matches at full
        confidence, so a compiled task picks reflex on its own.
        An explicit ``skill_match`` on the signal always wins
        over the lookup.
        """
        match = signal.skill_match
        skill_id = signal.skill_id
        if task_key is not None and match is None:
            stored = self.skill_store.find(task_key)
            if stored is not None and stored.skill_id is not None:
                match = 1.0
                skill_id = stored.skill_id
        risk = signal.risk
        if signal.previous_failure:
            return DepthDecision(
                depth=Depth.DEEP,
                reason="previous attempt failed; escalate to deep",
                risk=risk,
            )
        if (
            match is not None
            and match >= self.reflex_skill_match
            and skill_id is not None
            and signal.reversible
            and signal.stakes <= self.reflex_max_stakes
            and signal.uncertainty <= self.reflex_max_uncertainty
        ):
            return DepthDecision(
                depth=Depth.REFLEX,
                reason=(
                    f"stored skill {skill_id!r} matches at {match:.2f}; step is "
                    f"reversible, stakes {signal.stakes:.2f} and uncertainty "
                    f"{signal.uncertainty:.2f} are low"
                ),
                risk=risk,
                skill_id=skill_id,
            )
        if risk >= self.deep_risk:
            return DepthDecision(
                depth=Depth.DEEP,
                reason=(
                    f"uncertainty {signal.uncertainty:.2f} x stakes "
                    f"{signal.stakes:.2f} = {risk:.2f} reaches the deep bar "
                    f"{self.deep_risk:.2f}"
                ),
                risk=risk,
            )
        if not signal.has_evidence:
            return DepthDecision(
                depth=Depth.DEEP,
                reason="no evidence for this step; deep mode gathers it",
                risk=risk,
            )
        if not signal.reversible and signal.stakes >= self.deep_min_stakes_irreversible:
            return DepthDecision(
                depth=Depth.DEEP,
                reason=(
                    f"step is irreversible at stakes {signal.stakes:.2f}; "
                    "deep mode plans and verifies before acting"
                ),
                risk=risk,
            )
        return DepthDecision(
            depth=Depth.DELIBERATE,
            reason="default depth: no skill match, risk below the deep bar",
            risk=risk,
        )

    def find_skill(self, task_key: str) -> CompiledSkill | None:
        """The stored compiled skill for ``task_key``, or None."""
        return self.skill_store.find(task_key)

    # -- escalation ----------------------------------------------------

    def escalate(self, decision: DepthDecision, *, failure: str) -> DepthDecision | None:
        """Retry one failed verification, one level up, carrying the failure.

        Returns the escalated decision (``retried`` True,
        ``failure_context`` set), or None when there is no higher
        level left (a failed deep step) or the one retry was
        already used — the caller must then ask the user or fail
        the step. A retry off reflex drops the skill: the replay
        failed verification, so the retry replans instead of
        replaying the same skill.
        """
        if not failure or not failure.strip():
            raise ValueError("escalation needs a non-empty failure description")
        if decision.retried:
            return None
        higher = decision.depth.one_level_up()
        if higher is None:
            return None
        return DepthDecision(
            depth=higher,
            reason=(
                f"verification failed at {decision.depth.value}: {failure}; "
                f"retry one level up at {higher.value}"
            ),
            risk=decision.risk,
            skill_id=None,
            failure_context=failure,
            retried=True,
        )

    # -- compilation ---------------------------------------------------

    def streak(self, task_key: str) -> int:
        """Consecutive deep successes counted for ``task_key`` so far."""
        return self._deep_successes.get(task_key, 0)

    def record_outcome(
        self,
        task_key: str,
        *,
        depth: Depth | str,
        succeeded: bool,
        steps: list[dict] | None = None,
        regression: RegressionCase | None = None,
        step_runner: Callable[[dict], dict] | None = None,
        agent_id: str | None = None,
        last_result: dict | None = None,
    ) -> CompiledSkill | None:
        """Record one step outcome; compile the task when its deep streak fills.

        Only a succeeded deep step counts. A failure resets the
        streak for the key (and drops the runs stashed for it);
        a deliberate or reflex success leaves it alone. When the
        streak reaches ``compile_after`` the governor compiles
        the task: with a compiler attached, it generalises the
        streak's runs — each call's ``steps`` plus its
        ``last_result`` (the observed result of the run's last
        step) — into a versioned skill and stores it only when
        its replay passes. Without a compiler it builds the
        skill candidate from ``steps`` and ``regression`` and
        runs the regression through ``step_runner`` — the skill
        is stored only when the regression passes, and only
        then is the streak reset and the stored skill returned.
        Anything missing (no steps, no regression case, no
        runner) or a failed regression returns None and stores
        nothing; the streak stays full so a later call with the
        missing pieces can still compile.
        """
        if not task_key or not isinstance(task_key, str):
            raise ValueError("recording an outcome needs a non-empty task key")
        run_depth = depth if isinstance(depth, Depth) else Depth(depth)
        if not succeeded:
            self._deep_successes[task_key] = 0
            self._runs.pop(task_key, None)
            return None
        if run_depth is not Depth.DEEP:
            return None
        count = self._deep_successes.get(task_key, 0) + 1
        self._deep_successes[task_key] = count
        if steps is not None:
            self._runs.setdefault(task_key, []).append(
                SkillRun(steps=list(steps), last_result=dict(last_result) if last_result else {})
            )
        if count < self.compile_after:
            return None
        if self.compiler is not None:
            return self._compile_with_compiler(task_key, step_runner=step_runner, agent_id=agent_id)
        if not steps or regression is None or step_runner is None:
            return None
        candidate = CompiledSkill(task_key=task_key, steps=list(steps), regression=regression)
        result = candidate.run_regression(step_runner)
        if not result.passed:
            return None
        stored = self.skill_store.save(candidate, agent_id=agent_id)
        self._deep_successes[task_key] = 0
        return stored

    def _compile_with_compiler(
        self,
        task_key: str,
        *,
        step_runner: Callable[[dict], dict] | None,
        agent_id: str | None,
    ) -> CompiledSkill | None:
        """Compile the streak's runs through the attached compiler.

        Returns the stored skill, or None when the streak holds
        no runs or the compiler could not produce a passing
        skill — the streak stays full either way, so a later
        success can still compile.
        """
        compiler = self.compiler
        if compiler is None:
            return None
        runs = self._runs.get(task_key, [])
        if not runs:
            return None
        result = compiler.compile(task_key, runs, step_runner=step_runner, agent_id=agent_id)
        if result.skill is None:
            return None
        self._deep_successes[task_key] = 0
        self._runs.pop(task_key, None)
        return result.skill
