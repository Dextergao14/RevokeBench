"""Solver-side acceptance tests every scenario must pass before release."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .logic import Lit, Rule, RuleBase, contradictions, defeat_edges, lit, solve
from .generator import Scenario


@dataclass
class Report:
    ok: bool
    failures: List[str]
    stats: Dict[str, object]


def _rulebase_at(sc: Scenario, session: int, allow: str) -> RuleBase:
    rb = RuleBase(signature=dict(sc.signature),
                  universe={k: list(v) for k, v in sc.universe.items()})
    for r in sc.base_rules:
        rb.add(r)
    rb.add(Rule("feasible", lit("task_feasible"), (Lit(allow, ("?X",)),), 1))
    for ev in sc.events:
        if ev.session <= session:
            ev.apply(rb)
    return rb


def verify(sc: Scenario, allow: str) -> Report:
    fails: List[str] = []

    # (1) the closure is well defined and consistent at every timestep
    for s in range(0, sc.n_sessions + 1):
        sol = solve(_rulebase_at(sc, s, allow))
        if not sol.converged:
            fails.append(f"derivation did not converge at session {s}")
        bad = contradictions(sol.closure)
        if bad:
            fails.append(f"closure inconsistent at session {s}: "
                         + ", ".join(str(b) for b in bad))

    # (2) no must-violate probe, and the option set is never degenerate
    for p in sc.probes:
        if not p.options:
            fails.append(f"{p.probe_id}: empty option set")
        if not p.compliant:
            fails.append(f"{p.probe_id}: must-violate (no compliant option)")
        if not p.licensed:
            # the completion gate needs an option that is positively permitted,
            # otherwise "pick the entity nobody ever mentioned" passes for free
            fails.append(f"{p.probe_id}: no licensed option (task not completable)")
        if p.tests == "CANARY" and p.violating:
            fails.append(f"{p.probe_id}: canary probe has violating options")

    # (3) alternative-derivation guard.  ``task_feasible <= allow(?X)`` has one
    #     derivation per permitted entity, so it must be re-derived through a
    #     surviving path whenever any positive permission survives -- this is
    #     the over-deletion trap that naive incremental maintenance falls into.
    for p in sc.probes:
        rb_ = _rulebase_at(sc, p.session, allow)
        for i_, f in enumerate(getattr(p, "facts", [])):
            rb_.add(Rule(f"pf{i_}", lit(f), (), 1))
        sol = solve(rb_)
        positive = {l for l in sol.closure if l.pred == allow and not l.neg}
        feasible = lit("task_feasible") in sol.closure
        if bool(positive) != feasible:
            fails.append(f"{p.probe_id}: over-deletion at session {p.session} "
                         f"(permitted={len(positive)}, feasible={feasible})")

    # (4) discriminative: at least one probe where an option that used to be
    #     compliant has become a violation
    traps = [p for p in sc.probes if p.stale_trap]
    if not traps:
        fails.append("no stale-memory trap: scenario cannot separate systems")

    # (5) every probe must be decidable without ambiguity, and must actually
    #     appear in the rendered transcript
    for p in sc.probes:
        if set(p.compliant) & set(p.violating):
            fails.append(f"{p.probe_id}: option both compliant and violating")
        if p.turn_index < 0 or p.session > sc.n_sessions:
            fails.append(f"{p.probe_id}: probe never rendered into the transcript")

    # (6) self-containedness.  An option can only *be* a violation because of
    #     something the transcript said, so every violating option must have
    #     been named in an earlier turn.  Options that are merely compliant may
    #     be unmentioned -- that is exactly the permissive default under test.
    for p in sc.probes:
        seen = " ".join(t.text for t in sc.turns[:max(p.turn_index, 0)])
        for e in p.violating:
            if sc.entity_names[e] not in seen:
                fails.append(f"{p.probe_id}: violating option "
                             f"{sc.entity_names[e]} is unreachable from the "
                             f"transcript")

    # (7) priority is recoverable from the transcript.  For every defeat edge
    #     at a probe session the winner must be the more recent rule, or be
    #     written by an event marked as explicit override language, or the
    #     loser must be marked as explicitly advisory.  Otherwise the agent is
    #     being graded on a resolution it had no way of reading.
    order = {e.eid: i for i, e in enumerate(sc.events)}
    tags = {e.eid: set(e.tags) for e in sc.events}

    def rank(eid):
        for t in tags.get(eid, ()):
            if t.startswith("rank:"):
                return int(t.split(":")[1])
        return 1

    hier = sc.meta.get("hierarchy_session", 10 ** 9)
    for p in sc.probes:
        rb = _rulebase_at(sc, p.session, allow)
        sol = solve(rb)
        app = [n for n in sol.nodes if all(b in sol.closure for b in n.body)]
        for a, b in defeat_edges(app):
            ra = rb.rules[next(n.rid for n in app if n.nid == a)]
            rl = rb.rules[next(n.rid for n in app if n.nid == b)]
            if not ra.origin or not rl.origin:
                continue
            newer = order.get(ra.origin, -1) > order.get(rl.origin, -1)
            # a declared speaker hierarchy is the fourth admissible device:
            # a higher-ranked ruling beats a lower-ranked one regardless of order
            outranks = rank(ra.origin) > rank(rl.origin) and hier <= p.session
            if not (newer or outranks or "override" in tags.get(ra.origin, ())
                    or "advisory" in tags.get(rl.origin, ())):
                fails.append(f"{p.probe_id}: {ra.rid} defeats {rl.rid} but "
                             f"nothing in the transcript says so")
                break

    stats = {
        "sessions": sc.n_sessions,
        "turns": len(sc.turns),
        "events": len(sc.events),
        "probes": len(sc.probes),
        "trap_probes": len(traps),
        "flip_probes": sum(1 for p in sc.probes if p.flip),
        "by_event": {k: sum(1 for e in sc.events if e.kind == k)
                     for k in ("ADD", "SUPERSEDE", "CONDITION",
                               "CONFLICT", "SUPPORT", "RETRACT")},
    }
    return Report(not fails, fails, stats)
