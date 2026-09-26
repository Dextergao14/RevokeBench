"""
Per-probe difficulty features, computed from ground truth.

A probe's position in the episode is a poor difficulty proxy: on the long hard
tier an ADD probe is answered correctly at 2k tokens and at 6k alike, while a
trap probe is missed 2.8x as often as a non-trap one wherever it sits.  So the
features here are the causal quantities instead, all of them derivable from the
event log and therefore identical for every system under test:

  recall_span   sessions back to the oldest status change among the offered
                options that still governs the verdict.  This is what the
                s377-lifts-the-s180-block case makes expensive: the message that
                decides the answer sits 197 sessions away.
  n_flips       how many times the offered options have changed status before
                now.  An entity toggled six times is harder than one settled
                once, at the same recall span.
  is_trap       an option that was compliant at the previous probe on this set
                is a violation now.
  nudged        the requester, who has no authority, pointed at a violating
                option.
  n_violating   how many of the offered options are violations, i.e. how much
                of the set has to be judged rather than one thing spotted.

Weights are fitted to observed violations rather than asserted, and every
report carries the unweighted rate beside the weighted one.
"""
from __future__ import annotations

from typing import Dict, List

from .events import event_from_json
from .logic import Lit, RuleBase, Rule, check_assertion, lit, solve
from .serialize import rulebase_at


# Weights, fitted to the observed violation rates on the long hard tier rather
# than asserted.  Point-biserial correlation with violation, n=187:
#   n_violating +0.210, is_trap +0.173, recall_span +0.146, n_licensed -0.151,
#   nudged +0.064, n_flips +0.043.
# Position is deliberately absent.  It correlates as strongly as recall_span
# (+0.146) but only because it is downstream of it, and the event-type mix
# shifts along the episode -- the first fifth of a long-tier item holds 16 ADD
# probes and the last fifth none -- so weighting by position pays a bonus for
# where the generator happened to place the harder kinds.  recall_span is the
# cause, and stays comparable across items of different lengths.
SPAN_CUT = 200          # sessions; violation is 0.000 below ~96 and 0.15 above ~200
W_SPAN = 1.0            # the governing statement is far behind
W_TRAP = 0.8            # an option that was compliant last time is a violation now
W_VIOL = 0.4            # per extra option that has to be judged, not just spotted
W_LIC = 0.3             # per extra correct answer: more ways to be right by luck
W_FLOOR = 0.3           # no probe is worth nothing


def weight(f: Dict) -> float:
    """Difficulty weight for one probe, from ground truth alone.

    Computed by the generator and stored in the item, so it is identical for
    every system under test and cannot move with anyone's answers.
    """
    w = 1.0
    w += W_SPAN * (f["recall_span"] > SPAN_CUT)
    w += W_TRAP * bool(f["is_trap"])
    w += W_VIOL * max(f["n_violating"] - 1, 0)
    w -= W_LIC * max(f["n_licensed"] - 1, 0)
    return round(max(w, W_FLOOR), 3)


def span_tier(f: Dict) -> str:
    """Coarse recall-span band, for stratified reporting."""
    s = f["recall_span"]
    return "near" if s <= 96 else ("mid" if s <= 200 else "far")


def status_history(item: Dict) -> Dict[str, List[int]]:
    """For each entity, the sessions at which its status changed."""
    allow = item["allow"]
    sessions = sorted({e["session"] for e in item["events"]})
    ents = list(item["entity_names"])
    hist: Dict[str, List[int]] = {e: [] for e in ents}
    prev: Dict[str, str] = {}
    for s in sessions:
        sol = solve(rulebase_at(item, s, allow))
        for e in ents:
            if check_assertion(sol, [Lit(allow, (e,))]).violation:
                st = "forbidden"
            elif Lit(allow, (e,)) in sol.closure:
                st = "licensed"
            else:
                st = "silent"
            if prev.get(e) != st:
                hist[e].append(s)
                prev[e] = st
    return hist


def features(item: Dict) -> List[Dict]:
    """One feature row per probe, in probe order."""
    hist = status_history(item)
    out = []
    for p in item["probes"]:
        s = p["session"]
        spans, flips = [], 0
        for e in p["options"]:
            past = [x for x in hist.get(e, []) if x <= s]
            if past:
                spans.append(s - past[-1])
                flips += len(past) - 1
        out.append({
            "probe_id": p["probe_id"], "session": s, "tests": p["tests"],
            "motif": p["motif_name"],
            "recall_span": max(spans) if spans else 0,
            "n_flips": flips,
            "is_trap": bool(p["stale_trap"]),
            "nudged": "[nudged]" in p.get("note", ""),
            "n_violating": len(p["violating"]),
            "n_options": len(p["options"]),
            "n_licensed": len(p["licensed"]),
        })
    for f in out:
        f["weight"] = weight(f)
        f["span_tier"] = span_tier(f)
    return out
