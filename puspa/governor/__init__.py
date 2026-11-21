"""The depth governor: how much thinking each step gets.

Picks reflex, deliberate, or deep per step
(:class:`DepthGovernor`), escalates a failed verification one
level up exactly once, and compiles a repeatedly succeeding deep
task into a reflex skill that must pass its regression case
first — generalised into a versioned skill by the skill
compiler (:class:`SkillCompiler`), which replays the skill in
the sandbox and repairs it when a postcondition fails. See
:mod:`puspa.governor.governor` for the rules,
:mod:`puspa.governor.compiler` for the compiler, and
:mod:`puspa.governor.skills` for the compiled-skill shape and
stores.
"""

from __future__ import annotations

from .compiler import (
    CompileResult,
    GeneralisedPlan,
    RepairResult,
    ReplayResult,
    SkillCompiler,
    SkillRun,
    generalise_runs,
)
from .governor import (
    COMPILE_AFTER_SUCCESSES,
    DEEP_MIN_STAKES_IRREVERSIBLE,
    DEEP_RISK,
    REFLEX_MAX_STAKES,
    REFLEX_MAX_UNCERTAINTY,
    REFLEX_SKILL_MATCH,
    Depth,
    DepthDecision,
    DepthGovernor,
    StepSignal,
)
from .skills import (
    COMPILED_SERVICE_PREFIX,
    PARAM_MARKER,
    CompiledSkill,
    InMemorySkillStore,
    ProceduralSkillStore,
    RegressionCase,
    RegressionResult,
    SkillStore,
)

__all__ = [
    "COMPILED_SERVICE_PREFIX",
    "COMPILE_AFTER_SUCCESSES",
    "DEEP_MIN_STAKES_IRREVERSIBLE",
    "DEEP_RISK",
    "PARAM_MARKER",
    "REFLEX_MAX_STAKES",
    "REFLEX_MAX_UNCERTAINTY",
    "REFLEX_SKILL_MATCH",
    "CompileResult",
    "CompiledSkill",
    "Depth",
    "DepthDecision",
    "DepthGovernor",
    "GeneralisedPlan",
    "InMemorySkillStore",
    "ProceduralSkillStore",
    "RegressionCase",
    "RegressionResult",
    "RepairResult",
    "ReplayResult",
    "SkillCompiler",
    "SkillRun",
    "SkillStore",
    "StepSignal",
    "generalise_runs",
]
