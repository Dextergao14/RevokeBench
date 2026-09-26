#!/usr/bin/env python3
"""Ask a different set of questions of a released long-tier episode.

A full episode carries `probe_pool`: every task the generator emitted for it
(~180-270), each with its session, option set, ground-truth labels, difficulty
and the rendered task text.  The transcript, events and closure are the
episode; the released probes are one selection.  This script makes another:

    python3 scripts/reselect_probes.py --full data/long100/long100_full.jsonl.gz \
        --id REVOKE_long_companion_20264913 --k 12 --strategy traps --seed 3 \
        --out data/long100/reselect/

--full is either the released combined file (one episode per line; pick the
episode with --id) or a single-episode file such as those eval/build_long.py
writes to <out>/episodes/<id>_full.jsonl.gz (--id optional).

Strategies: `random` (uniform over the pool), `traps` (stale-memory traps
first), `far` (largest recall span first), `event:CONDITION` (one event kind),
`ids:p12,p57,...` (explicit pool ids).  Never-restricted ("crutch") tasks are
excluded unless --allow-crutch.  Output: <id>_<tag>_full.jsonl(.gz) and
_blind.jsonl(.gz); the grader works on them unchanged.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revoke.serialize import add_difficulty, blind          # noqa: E402


def load(path, iid=""):
    """One full item: the line whose id is `iid`, or the file's only item."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        if iid:
            for line in fh:
                if line.strip() and json.loads(line)["id"] == iid:
                    return json.loads(line)
            sys.exit(f"{iid}: not in {path}")
        first = fh.readline()
        if any(line.strip() for line in fh):
            sys.exit(f"{path} holds several episodes; choose one with --id <episode id>")
        return json.loads(first)


def dump(obj, path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as fh:
        fh.write(json.dumps(obj) + "\n")


def choose(pool, k, strategy, rng, allow_crutch):
    cand = [p for p in pool if allow_crutch or not p.get("crutch")]
    cand = [p for p in cand if "[echo]" not in p.get("note", "")]
    if strategy.startswith("ids:"):
        want = set(strategy[4:].split(","))
        return [p for p in pool if p["probe_id"].split("#")[1] in want]
    if strategy.startswith("event:"):
        cand = [p for p in cand if p["tests"] == strategy[6:]]
    rng.shuffle(cand)
    if strategy == "traps":
        cand.sort(key=lambda p: not p["stale_trap"])
    elif strategy == "far":
        cand.sort(key=lambda p: -p["difficulty"]["recall_span"])
    # never two tasks in one session
    out, seen = [], set()
    for p in cand:
        if p["session"] in seen:
            continue
        out.append(p)
        seen.add(p["session"])
        if len(out) == k:
            break
    return sorted(out, key=lambda p: p["session"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="full-view JSONL (plain or .gz): combined or single-episode")
    ap.add_argument("--id", default="", help="episode id; required when --full holds several episodes")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--allow-crutch", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    item = load(a.full, a.id)
    if "probe_pool" not in item:
        sys.exit("this item carries no probe_pool (built before the release form)")
    rng = random.Random(a.seed)
    picks = choose(item["probe_pool"], a.k, a.strategy, rng, a.allow_crutch)
    if len(picks) < a.k:
        print(f"only {len(picks)} tasks satisfy the strategy", file=sys.stderr)

    new = copy.deepcopy(item)
    # strip the released task turns, then place the chosen ones at the end of
    # their sessions (rule turns precede tasks within a session, as generated)
    for s in new["sessions"]:
        s["turns"] = [t for t in s["turns"] if not t.get("probe_id")]
    by_session = {s["index"]: s for s in new["sessions"]}
    probes = []
    for i, p in enumerate(picks):
        pid = f"{item['id']}#p{i}"
        q = {k: v for k, v in p.items() if k not in ("task_text", "crutch", "difficulty")}
        q["probe_id"] = pid
        q["note"] = (p.get("note", "") + f" [pool:{p['probe_id'].split('#')[1]}]").strip()
        probes.append(q)
        by_session[p["session"]]["turns"].append({"kind": "probe", "text": p["task_text"], "probe_id": pid})
    new["probes"] = probes
    new["timeline"] = [t for t in item["timeline"] if False]        # rebuilt by the grader from events
    new["meta"] = dict(item["meta"])
    new["meta"]["reselected"] = {"strategy": a.strategy, "seed": a.seed, "k": a.k,
                                 "from": [p["probe_id"] for p in picks]}
    add_difficulty(new)
    tag = a.tag or f"{a.strategy.replace(':', '-')}{a.k}s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    gz = ".gz" if a.full.endswith(".gz") else ""
    dump(new, os.path.join(a.out, f"{item['id']}_{tag}_full.jsonl{gz}"))
    dump(blind(new), os.path.join(a.out, f"{item['id']}_{tag}_blind.jsonl{gz}"))
    traps = sum(1 for p in probes if p["stale_trap"])
    print(f"{item['id']}: {len(probes)} tasks ({a.strategy}), traps {traps}, sessions "
          f"{[p['session'] for p in probes]} -> {a.out}/{item['id']}_{tag}_{{full,blind}}.jsonl{gz}")


if __name__ == "__main__":
    main()
