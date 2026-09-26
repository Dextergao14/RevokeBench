#!/usr/bin/env python3
"""Offline diagnostic for the rule-tracking baseline: perception vs tracking.

    python3 scripts/ruletrack_diag.py --full data/long100/long100_full.jsonl.gz \
        --logs runs/ruletrack_logs [--model qwen/qwen3.8-flash]

The backend writes, per episode, the statements it extracted and the verdict
it derived for every offered option at every task.  This script -- and only
this script -- reads the FULL file and aligns both against ground truth:

  extraction   for each ground-truth rule-writing event (ADD / CONFLICT /
               SUPERSEDE / CONDITION / SUPPORT / RETRACT), was a statement of a
               compatible kind about the same entity extracted at the same
               session?  Reported as recall per event kind, and precision as
               the share of extracted statements that match some event.
  state        for each task and offered option, does the backend's verdict
               (PERMITTED / PROHIBITED / NO RULE) agree with the label
               (licensed / violating / neither)?  Disagreements are split into
               "no statement was ever extracted about the decisive entity"
               (perception) and "statements exist but the derived state is
               wrong" (tracking).

Never point the backend itself at this script's inputs; it is analysis only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os

# ground-truth event kind -> statement types that would record it
COMPAT = {
    "ADD": {"permit", "forbid", "advise", "group_forbid", "link", "incompatible", "active", "condition",
            "threshold_rule", "member", "list"},
    "CONFLICT": {"forbid", "advise", "group_forbid", "threshold_rule"},
    "SUPERSEDE": {"reverse", "forbid", "permit", "lift", "condition", "active", "list"},
    "CONDITION": {"narrow", "widen"},
    "SUPPORT": {"reaffirm"},
    "RETRACT": {"lift", "member", "link"},
    "NOTE": {"threshold", "list", "role", "active", "condition"},
}


def rid_entities(events, names):
    """rid -> entities named in the head of the event that wrote the rule (for SUPPORT/RETRACT, which carry only the rid)."""
    out = {}
    for ev in events:
        if ev.get("rid") and ev.get("head"):
            out.setdefault(ev["rid"], set()).update(e for e in names if e in json.dumps(ev.get("head")))
    return out


def ents_of(ev, names, by_rid=None):
    out = set()
    if by_rid and ev.get("rid") and not ev.get("head"):
        out |= by_rid.get(ev["rid"], set())
    for f in ("head", "body", "instances"):
        blob = json.dumps(ev.get(f) or "")
        out |= {e for e in names if f'"{e}"' in blob or f"({e}" in blob or f",{e}" in blob or f"{e})" in blob}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--logs", default="runs/ruletrack_logs")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    opener = gzip.open if a.full.endswith(".gz") else open
    full = {}
    for line in opener(a.full, "rt"):
        d = json.loads(line)
        full[d["id"]] = d

    logs = [json.load(open(f)) for f in glob.glob(os.path.join(a.logs, "*.json"))]
    if a.model:
        logs = [l for l in logs if l["model"] == a.model]
    logs = [l for l in logs if l["id"] in full]
    print(f"{len(logs)} episode logs aligned against ground truth\n")

    rec = collections.defaultdict(lambda: [0, 0])          # kind -> [matched, total]
    prec = [0, 0]
    state = collections.Counter()
    by_kind_wrong = collections.Counter()
    for lg in logs:
        it = full[lg["id"]]
        names = set(it["entity_names"])
        stmts = lg["statements"]
        by_sess = collections.defaultdict(list)
        for s in stmts:
            by_sess[s["session"]].append(s)
        by_rid = rid_entities(it["events"], names)
        # extraction recall per ground-truth event
        matched_stmt_ids = set()
        for ev in it["events"]:
            k = ev["kind"]
            if k not in COMPAT:
                continue
            if "silent" in (ev.get("tags") or []):
                continue                                    # never rendered; cannot be extracted
            ents = ents_of(ev, names, by_rid)
            rec[k][1] += 1
            hit = False
            for s in by_sess.get(ev["session"], []):
                if s["type"] in COMPAT[k] and (not ents or s["entity"] in ents or s.get("entity2") in ents
                                               or s["type"] in ("condition", "list", "active")):
                    hit = True
                    matched_stmt_ids.add(id(s))
                    break
            rec[k][0] += hit
        rulings = [s for s in stmts if s["type"] != "role"]
        prec[1] += len(rulings)
        prec[0] += sum(1 for s in rulings if id(s) in matched_stmt_ids)
        # state agreement per task and option
        labels = {p["probe_id"]: p for p in it["probes"]}
        mentioned = {s["entity"] for s in stmts} | {s.get("entity2") for s in stmts}
        for r in lg.get("recalls", []):
            p = labels.get(r["probe_id"])
            if not p:
                continue
            for eid, v in r["verdicts"].items():
                truth = "PROHIBITED" if eid in p["violating"] else ("PERMITTED" if eid in p["licensed"] else "NO RULE")
                if v == truth:
                    state["agree"] += 1
                else:
                    state["disagree"] += 1
                    as_of = r.get("as_of")
                    later = as_of is not None and any(
                        as_of < ev["session"] <= p["session"] and eid in json.dumps([ev.get("head"), ev.get("body"), ev.get("instances")])
                        for ev in it["events"])
                    kind = "in_window" if later else ("perception" if eid not in mentioned else "tracking")
                    state[kind] += 1
                    by_kind_wrong[(truth, v)] += 1

    print("extraction recall by ground-truth event kind (rendered events only):")
    for k in ("ADD", "CONFLICT", "SUPERSEDE", "CONDITION", "SUPPORT", "RETRACT", "NOTE"):
        m, n = rec[k]
        print(f"  {k:<10} {m:>6}/{n:<6} {m / n:.2f}" if n else f"  {k:<10}      -")
    print(f"extraction precision (statements matching some event): {prec[0]}/{prec[1]} = "
          f"{(prec[0] / prec[1]) if prec[1] else 0:.2f}")
    tot = state["agree"] + state["disagree"]
    print(f"\nper-option verdict agreement with ground truth: {state['agree']}/{tot} = "
          f"{(state['agree'] / tot) if tot else 0:.2f}")
    print(f"  disagreements: {state['disagree']}  perception (entity never extracted) {state['perception']}  "
          f"tracking (extracted but wrong state) {state['tracking']}  in-window (decisive line still in the raw context, not the memory's job) {state['in_window']}")
    print("  confusion (truth -> backend):")
    for (t, v), n in by_kind_wrong.most_common():
        print(f"    {t:<10} -> {v:<10} {n}")


if __name__ == "__main__":
    main()
