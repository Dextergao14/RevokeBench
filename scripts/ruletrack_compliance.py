#!/usr/bin/env python3
"""Decompose a rule-tracking run: did the agent follow the memory, and was the memory right?

    python3 scripts/ruletrack_compliance.py --full data/long100/long100_full.jsonl.gz \
        --runs runs/ruletrack --logs runs/ruletrack_logs [--exclude <episode id>]

For every acted task, the memory's verdict on the option the agent chose (PERMITTED /
PROHIBITED / NO RULE, as placed in the prompt) is crossed with the graded outcome
(completion / violation / abstention).  Three shares follow:

  followed        the agent chose an option the memory labelled PERMITTED
  memory-wrong    ... and that option was a violation (an extraction or tracking error)
  disobeyed       the agent chose an option the memory labelled PROHIBITED (a violation by
                  construction if the memory was right)
  memory-silent   the agent chose an option the memory had no rule on

Reads the FULL file for grading only; the backend never sees it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.grade import grade_item                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--runs", required=True, help="run directory (final .jsonl and/or .part files)")
    ap.add_argument("--logs", required=True, help="the backend's diagnostic log directory")
    ap.add_argument("--exclude", action="append", default=[], help="episode id to leave out (e.g. the development episode)")
    a = ap.parse_args()
    opener = gzip.open if a.full.endswith(".gz") else open
    full = {}
    for line in opener(a.full, "rt"):
        d = json.loads(line)
        full[d["id"]] = d
    verdicts = {}
    for f in glob.glob(os.path.join(a.logs, "*.json")):
        for r in json.load(open(f)).get("recalls", []):
            verdicts[r["probe_id"]] = r["verdicts"]
    steps = collections.defaultdict(list)
    for f in glob.glob(os.path.join(a.runs, "*.jsonl")) + glob.glob(os.path.join(a.runs, "*.jsonl.part")):
        if f.endswith(".part") and os.path.exists(f.split("__REVOKE")[0] + ".jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("id") in full and r["id"] not in a.exclude:
                steps[r["id"]] += r["steps"]
    c = collections.Counter()
    n = 0
    for iid, st in steps.items():
        g = grade_item(full[iid], {"id": iid, "steps": st})
        for p in g["probes"]:
            if not p["chosen"]:
                c["no_action"] += 1
                continue
            v = verdicts.get(p["probe_id"])
            if not v:
                c["no_verdict"] += 1
                continue
            n += 1
            label = v.get(p["chosen"][0], "?")
            outcome = "VIOL" if p["violation"] else ("DONE" if p["completed"] else "ABST")
            c[(label, outcome)] += 1
            if not any(x == "PERMITTED" for x in v.values()):
                c["no_permitted_offered"] += 1
    viol = sum(v for (lab, out), v in ((k, v) for k, v in c.items() if isinstance(k, tuple)) if out == "VIOL")
    fol = sum(v for (lab, out), v in ((k, v) for k, v in c.items() if isinstance(k, tuple)) if lab == "PERMITTED")
    print(f"{len(steps)} episodes, {n} acted tasks, {c['no_action']} tasks with no action, {c['no_verdict']} without a recorded verdict")
    print(f"followed the memory (chose a PERMITTED option): {fol}/{n} = {fol / n:.0%}; "
          f"memory offered no PERMITTED option on {c['no_permitted_offered']} tasks")
    print("\nmemory label of the chosen option -> outcome")
    for lab in ("PERMITTED", "NO RULE", "PROHIBITED", "?"):
        row = {out: c[(lab, out)] for out in ("DONE", "VIOL", "ABST") if c[(lab, out)]}
        if row:
            print(f"  {lab:<11} {row}")
    if viol:
        print(f"\nviolations decomposed ({viol} total):")
        print(f"  memory said PERMITTED, was wrong : {c[('PERMITTED', 'VIOL')]:>4}  ({c[('PERMITTED', 'VIOL')] / viol:.0%})")
        print(f"  memory had no rule (missed it)   : {c[('NO RULE', 'VIOL')]:>4}  ({c[('NO RULE', 'VIOL')] / viol:.0%})")
        print(f"  agent chose a PROHIBITED option  : {c[('PROHIBITED', 'VIOL')]:>4}  ({c[('PROHIBITED', 'VIOL')] / viol:.0%})")


if __name__ == "__main__":
    main()
