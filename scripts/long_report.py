#!/usr/bin/env python3
"""Per-condition report: one row per (condition, episode) plus an ALL row, optional per-probe detail.

    python3 scripts/long_report.py --full data/long100/long100_full.jsonl.gz \
        --runs runs/compact runs/full --detail
"""
import argparse, collections, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from grade import exam, grade_item


def harness_failure(step):
    err = str(step.get("error", ""))
    if err.startswith("HTTP") or err.startswith("empty completion"):
        return True
    return (not step.get("tool_calls")) and not step.get("n_reads") and step.get("text") in (None, "", "None")


def cell_name(r):
    mode = r.get("mode", "full")
    if mode == "full":
        return "full"
    if mode in ("compact", "truncate") and "keep_frac" in r:
        # eval/adapters/episode_runner.py rows (continuous-context runner): budget + keep fraction
        return f"{mode} b{r['budget']} k{int(round(r['keep_frac'] * 100))}"
    # a pluggable memory system: name it by its backend, with the raw window it ran under
    return f"{mode} w{r.get('budget', '?')}" + ("" if not r.get("persist") else " persist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--detail", action="store_true", help="list every violation")
    a = ap.parse_args()
    import gzip
    opener = gzip.open if a.full.endswith(".gz") else open
    items = {json.loads(l)["id"]: json.loads(l) for l in opener(a.full, "rt")}
    short = {k: k.split("_", 2)[2].rsplit("_", 1)[0] for k in items}
    rows = {}                                 # (model, cell, item) -> graded probes
    meta = {}
    for d in a.runs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl")) + glob.glob(os.path.join(d, "*.jsonl.part"))):
            if path.endswith(".part"):
                # a per-item checkpoint is superseded by the condition's final
                # file (same prefix up to the item id)
                final = path.split("__REVOKE")[0] + ".jsonl"
                # a checkpoint is superseded only when the final file already holds THIS episode
                # (the runner creates the final file at start and appends episodes as they finish)
                iid_part = "REVOKE" + path.split("__REVOKE", 1)[1].rsplit(".jsonl", 1)[0]
                done_here = False
                for fin in (final, path[:-5]):
                    if os.path.exists(fin) and os.path.getsize(fin) > 0:
                        done_here |= any(json.loads(l).get("id") == iid_part for l in open(fin) if l.strip())
                if done_here:
                    continue
            for line in open(path):
                r = json.loads(line)
                if r.get("id") not in items or "steps" not in r:      # not a trace row
                    continue
                # graded: every step that is a decision -- an act, or a genuine no-action
                # (read loop / text reply).  HTTP failures and empty completions (HTTP 200
                # with nothing in it; older traces record them as "no action in
                # response" with no reads and no text) are harness failures and are dropped.
                acted = [s for s in r["steps"] if s.get("tool_calls") or not harness_failure(s)]
                model = r.get("model") or f"scripted:{r.get('policy', '?')}"
                key = (model, "scripted" if "policy" in r else cell_name(r), r["id"])
                # .part files carry one step per line; merge
                bucket = rows.setdefault(key, {"steps": [], "partial": path.endswith(".part"),
                                               "compactions": 0, "cost": 0.0})
                bucket["steps"] += acted
                bucket["compactions"] += len(r.get("compactions", []))
                bucket["cost"] += ((r.get("usage") or {}).get("cost") or 0)
    out = []
    print(f"{'model':<20}{'condition':<20}{'episode':<16}{'tok':>6}{'n':>4}{'viol':>6}{'done':>6}{'abst':>6}{'EXAM':>7}{'far':>6}{'trap':>6}{'cmp':>6}{'$':>7}")
    print("-" * 116)
    agg = collections.defaultdict(list)
    for (model, cell, iid), b in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], items[kv[0][2]]["meta"]["tokens"])):
        it = items[iid]
        g = grade_item(it, {"id": iid, "steps": b["steps"]})
        done_ids = {s["probe_id"] for s in b["steps"]}
        probes = [x for x in g["probes"] if x["probe_id"] in done_ids]
        if not probes:
            continue
        agg[(model, cell)] += probes
        n = len(probes)
        v = sum(x["violation"] for x in probes) / n
        d_ = sum(x["completed"] for x in probes) / n
        ab = sum((not x["violation"]) and (not x["completed"]) for x in probes) / n
        e = exam(probes)["score"]
        far = [x for x in probes if x["span_tier"] == "far"]
        tr = [x for x in probes if x["is_trap"]]
        fv = f"{sum(x['violation'] for x in far) / len(far):.2f}" if far else "  -  "
        tv = f"{sum(x['violation'] for x in tr) / len(tr):.2f}" if tr else "  -  "
        tag = short[iid] + ("*" if b["partial"] or n < len(it["probes"]) else "")
        print(f"{model[:19]:<20}{cell:<20}{tag:<16}{it['meta']['tokens'] // 1000:>5}k{n:>4}{v:>6.2f}{d_:>6.2f}{ab:>6.2f}{e:>+7.2f}{fv:>6}{tv:>6}{b['compactions']:>6}{b['cost']:>7.2f}")
        if a.detail:
            for x in probes:
                if x["violation"]:
                    p = next(p for p in it["probes"] if p["probe_id"] == x["probe_id"])
                    print(f"      VIOL {x['probe_id'].split('#')[1]:<4} s{x['session']:<4} span={x['recall_span']:<4} "
                          f"{'trap' if x['is_trap'] else '    '} {x['tests']:<10} chose {', '.join(it['entity_names'][c] for c in x['chosen'])}"
                          f"  | licensed: {', '.join(p['licensed_names'])}")
    print("-" * 116)
    for (model, cell), probes in sorted(agg.items()):
        n = len(probes)
        v = sum(x["violation"] for x in probes) / n
        d_ = sum(x["completed"] for x in probes) / n
        ab = sum((not x["violation"]) and (not x["completed"]) for x in probes) / n
        e = exam(probes)
        far = [x for x in probes if x["span_tier"] == "far"]; tr = [x for x in probes if x["is_trap"]]
        fv = f"{sum(x['violation'] for x in far) / len(far):.2f}" if far else "  -  "
        tv = f"{sum(x['violation'] for x in tr) / len(tr):.2f}" if tr else "  -  "
        print(f"{model[:19]:<20}{cell:<20}{'ALL':<16}{'':>6}{n:>4}{v:>6.2f}{d_:>6.2f}{ab:>6.2f}{e['score']:>+7.2f}{fv:>6}{tv:>6}"
              f"   wviol={sum(x['weight'] * x['violation'] for x in probes) / sum(x['weight'] for x in probes):.3f}")
    print("  * = partial (run still in progress); viol/done/abst unweighted; EXAM weighted in [-1, 1]")


if __name__ == "__main__":
    main()
