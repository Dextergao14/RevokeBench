#!/usr/bin/env python3
"""
Build long-context episodes (300k-1M tokens) from schema-v2 corpora.

Release form (N episodes over worlds x length bands, in parallel); the default bands are those of
the released set (paper Table 5), and the release's seed base is 20260912 (episode i is seeded
20260912 + 1000 * (i + 1); rejected seeds add 1, 2, ...), so this regenerates data/long100:
    python3 eval/build_long.py --n 100 --jobs 8 --seed 20260912 --out data/long100_regen
Use a fresh --out: episodes whose seed is already in <out>/manifest.jsonl are skipped, so
pointing it at data/long100 builds nothing.  Output is one pair of files per episode,
<out>/episodes/<id>_{full,blind}.jsonl.gz; the released long100_*.jsonl.gz concatenate them.
One released episode alone: take its manifest row and pass
--n 1 --worlds <world> --bands <target_tokens>:<cycles>:<probes> --seed <seed - rejections - 1000>,
e.g. REVOKE_long_companion_20264913 is --worlds companion --bands 300000:6:10 --seed 20263912.
Byte identity: the decompressed JSON is identical to the released line, and each released
long100_*.jsonl.gz is a concatenation of per-episode gzip members; a rebuilt .gz differs from its
released member only in the 4-byte gzip header MTIME (bytes 4-7, the build time), so manifest
`sha256_blind` (sha256 of the released blind member) matches once that field is copied over.

Single-episode form (one episode per plan entry, for development):
    python3 eval/build_long.py --plan clinical_ward=300000 --probes 10 --out data/dev

A band is target_tokens:cycles:probes.  `cycles` grows with the target so that
rule density stays roughly constant along the length ladder; `probes` is how
many of the ~180-270 generated tasks a released episode carries.

Per episode: build_hard_scenario -> verify -> render_long -> select probes ->
verify the rendered subset -> item (full + blind).  Generation is rejection
sampling as elsewhere: a seed that fails any acceptance check is skipped.

Probe selection.  From ground truth alone: 20% from the first third of the
episode, 30% from the middle, 50% from the last third, preferring stale-memory
traps (to ~70%), higher difficulty weight and a spread of event types, never
two in one session, never an echo probe, and never a task whose licensed
options were all approved long ago and never restricted since.  The unselected
task turns are removed from the transcript.  Every generated task is kept in
the full item as `probe_pool`, with its rendered task text, so a different set
of questions can be asked of the same episode without regenerating it; the
blind view carries only the released probes.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revoke.domains.corpus import CORPORA, load_corpus          # noqa: E402
from revoke.hard import build_hard_scenario                      # noqa: E402
from revoke.render_long import render_long                        # noqa: E402
from revoke.serialize import add_difficulty, blind, scenario_to_item  # noqa: E402
from revoke.verify import verify                                  # noqa: E402

BANDS = ((0.0, 1 / 3, 0.2), (1 / 3, 2 / 3, 0.3), (2 / 3, 1.01, 0.5))
WORLDS = ["clinical_ward", "devops_release", "design_system", "companion", "game_narrative", "meetings"]


def crutch_probes(item, old: int = 100) -> set:
    """Probes whose every licensed option was approved more than `old` sessions
    earlier and never restricted since -- answerable by spotting what was never
    touched, without tracking any state change.  Computed from ground truth."""
    from revoke.serialize import rulebase_at
    from revoke.logic import Lit, check_assertion, solve
    allow = item["allow"]
    ents = list(item["entity_names"])
    forbidden = {e: [] for e in ents}
    licensed_from = {}
    for s_ in sorted({e["session"] for e in item["events"]}):
        sol = solve(rulebase_at(item, s_, allow))
        for e in ents:
            if check_assertion(sol, [Lit(allow, (e,))]).violation:
                forbidden[e].append(s_)
            elif Lit(allow, (e,)) in sol.closure:
                licensed_from.setdefault(e, s_)
    out = set()
    for p in item["probes"]:
        lic = p["licensed"]
        if lic and all(not any(f < p["session"] for f in forbidden[e])
                       and p["session"] - licensed_from.get(e, p["session"]) > old for e in lic):
            out.add(p["probe_id"])
    return out


def select_probes(item, k: int, rng: random.Random, excluded: set, crutch: set = frozenset()) -> list:
    n = item["n_sessions"]
    used_sessions, types, chosen = set(), collections.Counter(), []
    quotas = [round(k * q) for _, _, q in BANDS]
    quotas[-1] += k - sum(quotas)
    for (lo, hi, _), quota in zip(BANDS, quotas):
        cand = [p for p in item["probes"]
                if lo * n <= p["session"] < hi * n and "[echo]" not in p["note"]
                and p["probe_id"] not in excluded]
        for _ in range(quota):
            cand = [p for p in cand if p["session"] not in used_sessions]
            if not cand:
                break

            def score(p):
                d = p["difficulty"]
                trap = bool(p["stale_trap"])
                s = 1.2 * trap + d["weight"] + 0.5 * (d["span_tier"] == "far")
                s -= 0.7 * types[p["tests"]]
                # keep some non-trap probes so trap / non-trap can be compared
                # within the tier; ~70% traps is the target
                if trap and types["_trap"] >= round(0.7 * k):
                    s -= 2.5
                if p["tests"] == "CANARY" and types["CANARY"]:
                    s -= 3
                # the licensed answer should depend on the history, not on
                # what was never mentioned
                if p["probe_id"] in crutch:
                    s -= 2.0
                return s + rng.random() * 0.05
            best = max(cand, key=score)
            chosen.append(best)
            used_sessions.add(best["session"])
            types[best["tests"]] += 1
            types["_trap"] += bool(best["stale_trap"])
    return chosen


def subset(sc, keep_ids: list):
    """Scenario with only the kept probes and their task turns, renumbered."""
    order = {pid: i for i, pid in enumerate(keep_ids)}
    new_id = {pid: f"{sc.sid}#p{i}" for pid, i in order.items()}
    probes = sorted((p for p in sc.probes if p.probe_id in order), key=lambda p: order[p.probe_id])
    turns = []
    for t in sc.turns:
        if t.kind == "probe":
            if t.probe_id not in order:
                continue
            t = replace(t, probe_id=new_id[t.probe_id])
        turns.append(t)
    pos = {t.probe_id: i for i, t in enumerate(turns) if t.probe_id}
    out = []
    for p in probes:
        q = replace(p, probe_id=new_id[p.probe_id])
        q.turn_index = pos[q.probe_id]
        out.append(q)
    return replace(sc, turns=turns, probes=out)


def build_one(corpus_path: str, target: int, k: int, seed: int, cycles: int, tries: int = 20):
    dom = load_corpus(corpus_path)
    corpus = CORPORA[dom.key]
    rng = random.Random(seed)
    log = []
    for attempt in range(tries):
        s = seed + attempt
        sid = f"REVOKE_long_{dom.key}_{s}"
        sc = build_hard_scenario(dom, sid, s, cycles=cycles)
        rep = verify(sc, dom.allow)
        if not rep.ok:
            log.append(f"seed {s}: rejected ({rep.failures[0][:70]})")
            continue
        rendered = render_long(sc, corpus, seed=s, target_tokens=target)
        full = add_difficulty(scenario_to_item(rendered, dom))
        crutch = crutch_probes(full)
        # the whole generated task pool, with rendered task text, so other
        # questions can be asked of this episode later
        pool = []
        for p in full["probes"]:
            turn = next(t for s_ in full["sessions"] for t in s_["turns"] if t.get("probe_id") == p["probe_id"])
            q = dict(p)
            q["task_text"] = turn["text"]
            q["crutch"] = p["probe_id"] in crutch
            pool.append(q)
        excluded: set = set()
        for _ in range(6):
            picks = select_probes(full, k, rng, excluded, crutch)
            if len(picks) < k:
                break
            sub = subset(rendered, [p["probe_id"] for p in sorted(picks, key=lambda p: p["session"])])
            rep2 = verify(sub, dom.allow)
            if rep2.ok:
                item = add_difficulty(scenario_to_item(sub, dom))
                item["setting"] = corpus["setting"]
                item["you_prefix"] = corpus["surface"]["you_prefix"]
                item["corpus"] = os.path.basename(corpus_path)
                item["tier"] = "long"
                item["probe_pool"] = pool
                item["meta"]["crutch_probes"] = sum(1 for p in picks if p["probe_id"] in crutch)
                item["meta"]["crutch_pool"] = f"{len(crutch)}/{len(full['probes'])}"
                item["meta"]["target_tokens"] = target
                item["meta"]["rejections"] = attempt
                return item, log
            bad = {f.split(":")[0] for f in rep2.failures if "#p" in f.split(":")[0]}
            back = {}
            for old, new in zip(sorted(picks, key=lambda p: p["session"]), sub.probes):
                back[new.probe_id] = old["probe_id"]
            hit = {back[b] for b in bad if b in back}
            if not hit:
                log.append(f"seed {s}: subset rejected ({rep2.failures[0][:70]})")
                break
            excluded |= hit
        log.append(f"seed {s}: could not assemble a clean {k}-probe subset")
    raise RuntimeError(f"{corpus_path}: no acceptable episode in {tries} seeds\n  " + "\n  ".join(log))


def summary(item) -> str:
    m = item["meta"]
    tiers = collections.Counter(p["difficulty"]["span_tier"] for p in item["probes"])
    n = len(item["probes"])
    return (f"{item['id']}: {m['tokens'] // 1000}k tok ({m['format']}), {item['n_sessions']} sessions, "
            f"{n} probes, traps {sum(1 for p in item['probes'] if p['stale_trap'])}, span {dict(tiers)}, "
            f"mean w {sum(p['difficulty']['weight'] for p in item['probes']) / n:.2f}, "
            f"pad dup {m['pad_dup_rate']:.1%}, crutch {m['crutch_probes']}/{n} (pool {m['crutch_pool']}), "
            f"rejections {m.get('rejections', 0)}")


def _job(args):
    corpus_path, target, k, seed, cycles, out_dir = args
    item, log = build_one(corpus_path, target, k, seed, cycles)
    base = os.path.join(out_dir, "episodes", item["id"])
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with gzip.open(base + "_full.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(item) + "\n")
    with gzip.open(base + "_blind.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(blind(item)) + "\n")
    m = item["meta"]
    row = {"id": item["id"], "world": item["domain"], "layout": m["format"], "target_tokens": target,
           "tokens": m["tokens"], "sessions": item["n_sessions"], "cycles": cycles,
           "probes": len(item["probes"]), "traps": sum(1 for p in item["probes"] if p["stale_trap"]),
           "far": sum(1 for p in item["probes"] if p["difficulty"]["span_tier"] == "far"),
           "mean_weight": round(sum(p["difficulty"]["weight"] for p in item["probes"]) / len(item["probes"]), 3),
           "crutch": m["crutch_probes"], "pool": len(item["probe_pool"]), "seed": item["seed"],
           "rejections": m.get("rejections", 0), "pad_dup": m["pad_dup_rate"],
           "sha256_blind": hashlib.sha256(open(base + "_blind.jsonl.gz", "rb").read()).hexdigest()[:16]}
    return row, summary(item), log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="", help="single-episode form: key=target_tokens,...")
    ap.add_argument("--n", type=int, default=0, help="release form: number of episodes over worlds x bands")
    ap.add_argument("--bands", default="300000:6:10,440000:7:11,580000:8:12,720000:9:13,830000:10:15",
                    help="target_tokens:cycles:probes, comma-separated")
    ap.add_argument("--worlds", default=",".join(WORLDS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--corpora", default="revoke/domains/corpora")
    ap.add_argument("--out", default="data/long")
    ap.add_argument("--probes", type=int, default=10, help="single-episode form only")
    ap.add_argument("--cycles", type=int, default=6, help="single-episode form only")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.plan:                                              # ---- single-episode form
        items = []
        for i, kv in enumerate(a.plan.split(",")):
            key, target = kv.split("=")[0], int(kv.split("=")[1])
            print(f"[{key}] target {target // 1000}k tokens", flush=True)
            item, log = build_one(os.path.join(a.corpora, f"{key}.json"), target, a.probes,
                                  a.seed + 100 * i, a.cycles)
            for line in log:
                print("  " + line, flush=True)
            print("  ok  " + summary(item), flush=True)
            items.append(item)
            with open(os.path.join(a.out, f"{key}_full.jsonl"), "w") as fh:
                fh.write(json.dumps(item) + "\n")
            with open(os.path.join(a.out, f"{key}_blind.jsonl"), "w") as fh:
                fh.write(json.dumps(blind(item)) + "\n")
        with open(os.path.join(a.out, "long_full.jsonl"), "w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        with open(os.path.join(a.out, "long_blind.jsonl"), "w") as fh:
            for it in items:
                fh.write(json.dumps(blind(it)) + "\n")
        print(f"\nwrote {len(items)} episodes to {a.out}/")
        return

    # ---- release form: N episodes over worlds x bands, round-robin, in parallel
    worlds = a.worlds.split(",")
    bands = [tuple(int(x) for x in b.split(":")) for b in a.bands.split(",")]
    cells = [(w, b) for b in bands for w in worlds]
    jobs = []
    for i in range(a.n):
        w, (target, cycles, k) = cells[i % len(cells)]
        jobs.append((os.path.join(a.corpora, f"{w}.json"), target, k, a.seed + 1000 * (i + 1), cycles, a.out))
    print(f"{a.n} episodes over {len(worlds)} worlds x {len(bands)} bands, {a.jobs} parallel jobs", flush=True)
    manifest = os.path.join(a.out, "manifest.jsonl")
    done_seeds = set()
    if os.path.exists(manifest):
        for line in open(manifest):
            done_seeds.add(json.loads(line)["seed"])
    jobs = [j for j in jobs if not any(j[3] <= s < j[3] + 20 for s in done_seeds)]
    print(f"{len(jobs)} to build ({a.n - len(jobs)} already in manifest)", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex, open(manifest, "a") as mf:
        futs = {ex.submit(_job, j): j for j in jobs}
        for n_done, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                row, line, log = f.result()
            except Exception as e:                               # noqa: BLE001
                print(f"  FAILED {os.path.basename(j[0])} @{j[1] // 1000}k: {str(e).splitlines()[0][:120]}", flush=True)
                continue
            rows.append(row)
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print(f"  [{n_done}/{len(jobs)}] {line}", flush=True)
    print(f"\n{len(rows)} episodes built this run; manifest at {manifest}")


if __name__ == "__main__":
    main()
