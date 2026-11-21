"""Procedural - layer 3 of memory (definition 2.7 C55, 3.3).

How to do a task that once succeeded, stored as a procedure.
In the MVP procedures are written AUTOMATICALLY from tasks that
succeeded, with no procedure editor. The full Service Flow Graph
(canary, cross-agent reuse, cross-check) is built in Phase 3;
what is built here: the procedural write path + storage + the
basic lifecycle (trial -> verified -> broken), plus the
learning-loop hardening added in the Brain phase:

- Promotion: trial -> verified after 3 successful runs by at
  least 2 different agents (unchanged).
- Contradiction: a new step that conflicts with a stored
  procedure for the same service never silently overwrites it.
  It is stored as a separate trial candidate and reported as a
  proposal with `conflicts_with`; the user resolves it.

Note on demotion (integration decision, 18 Nov 2026): the Brain
phase first added a demotion rule (one failure on a verified
procedure drops it back to trial). That rule was removed again
during integration because it broke the revision 7 section 3.3
contract the canary relies on: `run_canary` in
`puspa/graphs/service.py` only runs on verified procedures, so a
demoted procedure could never record the second consecutive
failure that makes it broken, and the pre-existing canary tests
(`tests/test_service_canary.py`,
`tests/test_integration_gates.py`) failed. Revision 7 says
broken after 2 consecutive failed runs, with no demotion step,
so the lifecycle here follows revision 7 exactly. A demotion
rule would need a canary-guard change first, which is outside
this phase.

Patterns adapted (rewritten, not copied):
- Letta (Apache-2.0): sleep-time compute - a second agent
  tidies memory while idle; Puspa uses the same pattern in
  `consolidate.py`.
- Graphiti (Apache-2.0): procedures as parameterised templates
  without personal data; customer numbers never enter procedures.
- Voyager (MIT): learned skills stored as tested executable
  code with an embedding key; Puspa stores learned-skill drafts
  as trial procedures under the `learned_draft:` service prefix,
  inactive until the user approves them (see
  `puspa/memory/learning_loop.py`).

Lifecycle numbers here (first numbers, revision 7 section 3.3):
- `trial`: born from a successful task that passed the Checker.
- `verified`: after 3 successful runs, by at least 2 agents.
- `broken`: 2 consecutive failed runs; broken procedures are
  never used.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .schema import resolve_dsn

REQUIRED_VERIFIED_RUNS = 3
REQUIRED_VERIFIED_AGENTS = 2
BROKEN_AFTER_CONSECUTIVE_FAILURES = 2
STALE_AFTER_DAYS = 90
LEARNED_DRAFT_PREFIX = "learned_draft:"
LEARNED_PREFIX = "learned:"


def _connect(dsn=None):
    return psycopg.connect(resolve_dsn(dsn), row_factory=dict_row)


def _row_to_procedure(row):
    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "service": row["service"],
        "version": row["versi"],
        "step": row["step"],
        "output_schema": row["output_schema"],
        "required_checks": row["required_checks"],
        "status": row["status"],
        "successful_runs": row["successful_runs"],
        "failed_runs": row["failed_runs"],
        "consecutive_failed_runs": row["consecutive_failed_runs"],
        "agent_ids": row["agent_ids"],
        "source_event_id": str(row["source_event_id"]) if row["source_event_id"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _step_hash(step):
    canon = json.dumps(step, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def compute_run_update(proc, success, agent_id=None):
    """Pure lifecycle step: apply one run to a procedure dict.

    Returns a new dict with updated counts and status. No
    database access, so the promotion / broken rules are
    testable in memory. Revision 7 section 3.3, exactly: a
    verified procedure survives a single failure (it keeps its
    status and its successful runs, with one consecutive
    failure counted); two consecutive failures mean broken.
    There is no demotion step (see the module note: demotion
    was removed during Brain-phase integration because it broke
    the canary contract). The returned dict keeps a `demoted`
    key, always False, so callers reading the old shape do not
    break.
    """
    agent_ids = list(proc.get("agent_ids") or [])
    if agent_id and str(agent_id) not in agent_ids:
        agent_ids.append(str(agent_id))
    successful = proc["successful_runs"] + (1 if success else 0)
    failed = proc["failed_runs"] + (0 if success else 1)
    consecutive_failures = 0 if success else proc["consecutive_failed_runs"] + 1
    status = proc["status"]
    if consecutive_failures >= BROKEN_AFTER_CONSECUTIVE_FAILURES:
        status = "broken"
    elif (
        status == "trial"
        and successful >= REQUIRED_VERIFIED_RUNS
        and len(agent_ids) >= REQUIRED_VERIFIED_AGENTS
    ):
        status = "verified"
    updated = dict(proc)
    updated.update(
        {
            "successful_runs": successful,
            "failed_runs": failed,
            "consecutive_failed_runs": consecutive_failures,
            "agent_ids": agent_ids,
            "status": status,
            "demoted": False,
        }
    )
    return updated


def is_stale(proc, now=None, older_than_days=STALE_AFTER_DAYS):
    """True when a verified procedure was untouched for too long."""
    if proc.get("status") != "verified":
        return False
    updated_at = proc.get("updated_at")
    if updated_at is None:
        return False
    now = now or datetime.now(UTC)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return (now - updated_at) >= timedelta(days=older_than_days)


def apply_decay(proc, now=None, older_than_days=STALE_AFTER_DAYS):
    """Pure decay step: a stale verified procedure returns to trial.

    Returns a new dict; untouched procedures are returned as-is
    with `decayed` False. Decayed procedures must earn
    verification again from zero successful runs.
    """
    if not is_stale(proc, now=now, older_than_days=older_than_days):
        updated = dict(proc)
        updated["decayed"] = False
        return updated
    updated = dict(proc)
    updated.update({"status": "trial", "successful_runs": 0, "decayed": True})
    return updated


def steps_conflict(old_step, new_step):
    """True when two steps for the same service disagree.

    A conflict is a shared task with a different method: at
    least one identical step element (same task) and at least
    one difference (different method). Identical steps do not
    conflict; wholly unrelated steps do not conflict either.
    """
    old_set = {json.dumps(s, sort_keys=True, ensure_ascii=False) for s in (old_step or [])}
    new_set = {json.dumps(s, sort_keys=True, ensure_ascii=False) for s in (new_step or [])}
    if not old_set or not new_set:
        return False
    return bool(old_set & new_set) and old_set != new_set


def detect_contradiction(existing_procedures, service, step_new):
    """Find the stored procedure that `step_new` contradicts.

    Returns the conflicting procedure dict, or None. A conflict
    never overwrites: the caller must store the new step as a
    separate candidate (see `propose_conflicting_procedure`).
    """
    for proc in existing_procedures or []:
        if proc.get("service") == service and steps_conflict(proc.get("step"), step_new):
            return proc
    return None


def insert_procedure(
    user_id,
    service,
    step,
    *,
    version=1,
    agent_id=None,
    output_schema=None,
    required_checks=None,
    source_event_id=None,
    dsn=None,
):
    """Insert a NEW procedure row for a service, even when one exists.

    Unlike :func:`record_successful_task`, this never folds the
    write into an existing row: it always creates a fresh trial
    procedure with ``version`` and 1 successful run. The skill
    compiler uses it to store a repaired or re-generalised skill
    as a new version; the older rows stay as history and are
    never silently overwritten.
    """
    if not service or not isinstance(step, list) or not step:
        raise ValueError("service and non-empty steps are required")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"procedure version must be an integer at least 1, got {version!r}")
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            agent_ids = [str(agent_id)] if agent_id else []
            cur.execute(
                "INSERT INTO procedures (user_id, service, versi, step, "
                "output_schema, required_checks, status, successful_runs, "
                "agent_ids, source_event_id) VALUES (%s, %s, %s, %s, %s, %s, "
                "'trial', 1, %s, %s) RETURNING *",
                (
                    user_id,
                    service,
                    version,
                    Jsonb(step),
                    Jsonb(output_schema or {}),
                    Jsonb(required_checks or []),
                    Jsonb(agent_ids),
                    source_event_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _row_to_procedure(row)


def record_successful_task(
    user_id,
    service,
    step,
    agent_id=None,
    output_schema=None,
    required_checks=None,
    source_event_id=None,
    dsn=None,
):
    """Write a procedure from a successful task (automatic, no editor).

    If a procedure with the same service + steps already exists,
    record one more successful run (not a duplicate). A new
    procedure is born with status `trial` and 1 successful run.
    """
    if not service or not isinstance(step, list) or not step:
        raise ValueError("service and non-empty steps are required")
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM procedures WHERE user_id = %s AND service = %s "
                "ORDER BY created_at DESC",
                (user_id, service),
            )
            existing = None
            for row in cur.fetchall():
                if _step_hash(row["step"]) == _step_hash(step):
                    existing = row
                    break
            if existing is None:
                agent_ids = [str(agent_id)] if agent_id else []
                cur.execute(
                    "INSERT INTO procedures (user_id, service, step, "
                    "output_schema, required_checks, status, successful_runs, "
                    "agent_ids, source_event_id) VALUES (%s, %s, %s, %s, %s, "
                    "'trial', 1, %s, %s) RETURNING *",
                    (
                        user_id,
                        service,
                        Jsonb(step),
                        Jsonb(output_schema or {}),
                        Jsonb(required_checks or []),
                        Jsonb(agent_ids),
                        source_event_id,
                    ),
                )
                row = cur.fetchone()
            else:
                row = _mark_run_in_cursor(cur, existing, success=True, agent_id=agent_id)
        conn.commit()
    return _row_to_procedure(row)


def _mark_run_in_cursor(cur, row, success, agent_id):
    updated = compute_run_update(_row_to_procedure(row), success, agent_id=agent_id)
    cur.execute(
        "UPDATE procedures SET successful_runs = %s, failed_runs = %s, "
        "consecutive_failed_runs = %s, agent_ids = %s, status = %s, updated_at = now() "
        "WHERE id = %s RETURNING *",
        (
            updated["successful_runs"],
            updated["failed_runs"],
            updated["consecutive_failed_runs"],
            Jsonb(updated["agent_ids"]),
            updated["status"],
            row["id"],
        ),
    )
    return cur.fetchone()


def mark_run(procedure_id, user_id, success, agent_id=None, dsn=None):
    """Record one procedure run (success/failure); return the new procedure.

    Raise KeyError if the procedure is not found.
    """
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM procedures WHERE id = %s AND user_id = %s",
                (procedure_id, user_id),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"procedure not found: {procedure_id}")
            row = _mark_run_in_cursor(cur, row, success=success, agent_id=agent_id)
        conn.commit()
    return _row_to_procedure(row)


def get_procedure(procedure_id, user_id, dsn=None):
    """Fetch one procedure owned by the user; KeyError if missing."""
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM procedures WHERE id = %s AND user_id = %s",
            (procedure_id, user_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"procedure not found: {procedure_id}")
    return _row_to_procedure(row)


def list_procedures(user_id, status=None, dsn=None):
    """List the user's procedures, newest first."""
    with _connect(dsn) as conn, conn.cursor() as cur:
        if status is None:
            cur.execute(
                "SELECT * FROM procedures WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cur.execute(
                "SELECT * FROM procedures WHERE user_id = %s AND status = %s "
                "ORDER BY created_at DESC",
                (user_id, status),
            )
        return [_row_to_procedure(r) for r in cur.fetchall()]


def decay_stale_procedures(user_id, older_than_days=STALE_AFTER_DAYS, dsn=None, now=None):
    """Demote verified procedures untouched for `older_than_days`.

    Stale procedures decay to trial with zero successful runs, so
    they must earn verification again. Returns the list of
    decayed procedures. Broken and trial procedures are untouched.
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=older_than_days)
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM procedures WHERE user_id = %s AND status = 'verified' "
                "AND updated_at < %s",
                (user_id, cutoff),
            )
            rows = cur.fetchall()
            decayed = []
            for row in rows:
                cur.execute(
                    "UPDATE procedures SET status = 'trial', successful_runs = 0, "
                    "updated_at = now() WHERE id = %s RETURNING *",
                    (row["id"],),
                )
                decayed.append(_row_to_procedure(cur.fetchone()))
        conn.commit()
    return decayed


def propose_conflicting_procedure(
    user_id,
    service,
    step_new,
    agent_id=None,
    output_schema=None,
    required_checks=None,
    source_event_id=None,
    dsn=None,
):
    """Store a conflicting step as a separate trial candidate.

    A new fact that contradicts a stored procedure never
    overwrites it silently: the old procedure is untouched, the
    new step is stored as its own trial procedure, and the
    result reports `conflicts_with` (the old procedure id) and
    status "proposed". With no conflict the result reports
    status "recorded" and `conflicts_with` None.
    """
    if not service or not isinstance(step_new, list) or not step_new:
        raise ValueError("service and non-empty steps are required")
    existing = list_procedures(user_id, dsn=dsn)
    conflict = detect_contradiction(existing, service, step_new)
    proc = record_successful_task(
        user_id,
        service,
        step_new,
        agent_id=agent_id,
        output_schema=output_schema,
        required_checks=required_checks,
        source_event_id=source_event_id,
        dsn=dsn,
    )
    return {
        "procedure": proc,
        "conflicts_with": conflict["id"] if conflict else None,
        "status": "proposed" if conflict else "recorded",
    }


def resolve_contradiction(user_id, keep_id, drop_id, dsn=None):
    """Resolve a contradiction by user decision.

    The kept procedure is untouched; the dropped one is retired
    (status `broken`, so it is never reused) with its history
    kept. Returns {"kept": ..., "retired": ...}.
    """
    if keep_id == drop_id:
        raise ValueError("keep_id and drop_id must differ")
    kept = get_procedure(keep_id, user_id, dsn=dsn)
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE procedures SET status = 'broken', updated_at = now() "
                "WHERE id = %s AND user_id = %s RETURNING *",
                (drop_id, user_id),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"procedure not found: {drop_id}")
        conn.commit()
    return {"kept": kept, "retired": _row_to_procedure(row)}


def activate_draft_procedure(procedure_id, user_id, dsn=None):
    """Activate a learned-skill draft procedure after user approval.

    Drafts live under the `learned_draft:` service prefix and are
    never selected for execution under that prefix. Activation
    strips the prefix to `learned:`; the procedure keeps its
    trial status and must still earn verification by runs.
    Raise KeyError if the procedure is not found, ValueError if
    it is not a draft.
    """
    proc = get_procedure(procedure_id, user_id, dsn=dsn)
    if not proc["service"].startswith(LEARNED_DRAFT_PREFIX):
        raise ValueError(f"procedure is not a learned draft: {procedure_id}")
    new_service = LEARNED_PREFIX + proc["service"][len(LEARNED_DRAFT_PREFIX) :]
    with _connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE procedures SET service = %s, updated_at = now() "
                "WHERE id = %s AND user_id = %s RETURNING *",
                (new_service, procedure_id, user_id),
            )
            row = cur.fetchone()
        conn.commit()
    return _row_to_procedure(row)
