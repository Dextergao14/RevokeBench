"""Scenario -> benchmark item, and the blind view handed to agents."""
from __future__ import annotations

from typing import Dict, List

from .domains.base import Domain
from .generator import Scenario
from .logic import solve
from .verify import _rulebase_at

from .events import event_from_json
from .logic import Lit, Rule, RuleBase, lit as _lit

SECRET_FIELDS = ("compliant", "violating", "blamed", "stale_trap", "flip",
                 "deleted_rules", "note")


def scenario_to_item(sc: Scenario, dom: Domain) -> Dict:
    sessions: List[Dict] = []
    for s in range(1, sc.n_sessions + 1):
        ts = [t for t in sc.turns if t.session == s]
        if ts:
            sessions.append({"index": s, "turns": [
                {"kind": t.kind, "text": t.text,
                 **({"speaker": t.speaker} if t.speaker else {}),
                 **({"probe_id": t.probe_id} if t.probe_id else {})} for t in ts]})

    timeline = []
    for p in sc.probes:
        rb_ = _rulebase_at(sc, p.session, dom.allow)
        for i_, f in enumerate(p.facts):
            rb_.add(Rule(f"pf{i_}", _lit(f), (), 1))
        sol = solve(rb_)
        timeline.append({
            "probe_id": p.probe_id, "session": p.session,
            "closure": sorted(str(l) for l in sol.closure),
            "extension": sorted({n.rid for n in sol.extension}),
            "deleted": sorted({n.rid for n in sol.deleted}),
        })

    act = dom.act_tool()
    return {
        "id": sc.sid,
        "domain": sc.domain,
        "sort": dom.sort,
        "allow": dom.allow,
        "goal": dom.goal,
        "contexts": sorted(k for k, v in sc.signature.items()
                           if v == () and k not in (dom.goal, "task_feasible")),
        # the full sort signature, so a grader rebuilds R_t without having to
        # know which predicates a tier uses
        "signature": {k: list(v) for k, v in sc.signature.items()},
        "universe": sc.universe,
        "base_rules": [{"rid": r.rid, "head": str(r.head),
                        "body": [str(b) for b in r.body], "prio": r.prio}
                       for r in sc.base_rules],
        "domain_title": dom.title,
        "regime": sc.regime,
        "density": sc.density,
        "tier": sc.meta.get("tier", "easy"),
        "meta": dict(sc.meta),
        "seed": sc.seed,
        "motifs": sc.motifs,
        "n_sessions": sc.n_sessions,
        "sessions": sessions,
        "tools": [{"name": t.name, "params": t.params, "kind": t.kind,
                   "doc": t.doc} for t in dom.tools],
        "act_tool": act.name,
        "act_param": list(act.params)[0],
        "entity_names": sc.entity_names,
        "probes": [{
            "probe_id": p.probe_id, "session": p.session,
            "tests": p.tests, "motif": p.motif, "motif_name": p.motif_name,
            "options": p.options, "option_names": p.option_names,
            "compliant": p.compliant,
            "compliant_names": [sc.entity_names[e] for e in p.compliant],
            "licensed": p.licensed,
            "licensed_names": [sc.entity_names[e] for e in p.licensed],
            "violating": p.violating,
            "violating_names": [sc.entity_names[e] for e in p.violating],
            "blamed": p.blamed, "stale_trap": p.stale_trap,
            "flip": p.flip, "deleted_rules": p.deleted_rules, "note": p.note,
            "params": dict(p.params), "facts": list(p.facts),
        } for p in sc.probes],
        "events": [e.to_json() for e in sc.events],
        "timeline": timeline,
    }


def add_difficulty(item: Dict) -> Dict:
    """Attach per-probe difficulty features and weights, in place."""
    from .difficulty import features
    feat = {f["probe_id"]: f for f in features(item)}
    for p in item["probes"]:
        f = feat.get(p["probe_id"], {})
        p["difficulty"] = {k: f[k] for k in
                           ("recall_span", "n_flips", "is_trap", "nudged",
                            "n_violating", "n_licensed", "weight", "span_tier")
                           if k in f}
    return item


def blind(item: Dict) -> Dict:
    """The view an agent is allowed to see: no closure, no compliant set."""
    # the blind view carries only what an agent legitimately sees: the
    # transcript, the tool surface, and the option sets.  No rule vocabulary,
    # no universe, no base facts -- those would let a parser reconstruct
    # constraints the conversation is supposed to be the sole source of.
    out = {k: v for k, v in item.items()
           if k not in ("timeline", "events", "motifs", "base_rules", "allow",
                        "sort", "goal", "contexts", "universe", "seed", "meta",
                        "signature", "probe_pool")}
    # turn kinds (update / noise / filler) would let a scaffold filter the
    # transcript by construction; only the probe marker is legitimately visible
    out["sessions"] = [{"index": s["index"], "turns": [
        {"text": t["text"], **({"probe_id": t["probe_id"]} if t.get("probe_id") else {})}
        for t in s["turns"]]} for s in item["sessions"]]
    # difficulty is ground truth: it names how far back the governing statement
    # is and how many options are violations, so it never reaches an agent
    out["probes"] = [{"probe_id": p["probe_id"], "session": p["session"],
                      "options": p["options"], "option_names": p["option_names"],
                      **({"params": p["params"]} if p.get("params") else {})}
                     for p in item["probes"]]
    return out


# --------------------------------------------------------------------------
# grader-side reconstruction: rebuild the persistent layer from a full item
# --------------------------------------------------------------------------

_GRP_RE = None


def rulebase_at(item: Dict, session: int, allow: str, facts=()) -> RuleBase:
    """Rebuild R_t from a full (non-blind) item, for independent re-grading."""
    if "signature" in item:
        sig = {k: tuple(v) for k, v in item["signature"].items()}
    else:                                          # easy-tier items predate it
        sig = {allow: (item["sort"],), item["goal"]: (),
               "grp": (item["sort"], "group"), "task_feasible": ()}
        for c in item["contexts"]:
            sig[c] = ()
    rb = RuleBase(signature=sig, universe={k: list(v)
                                           for k, v in item["universe"].items()})
    for r in item["base_rules"]:
        rb.add(Rule(r["rid"], _lit(r["head"]),
                    tuple(_lit(b) for b in r["body"]), r["prio"]))
    rb.add(Rule("feasible", _lit("task_feasible"), (Lit(allow, ("?X",)),), 1))
    for d in item["events"]:
        e = event_from_json(d)
        if e.session <= session:
            e.apply(rb)
    for i, f in enumerate(facts):                 # probe-scoped facts (hard tier)
        rb.add(Rule(f"pf{i}", _lit(f), (), 1))
    return rb
