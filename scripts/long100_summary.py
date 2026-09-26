#!/usr/bin/env python3
"""Summarise a release build from its manifest.

    python3 scripts/long100_summary.py data/long100/manifest.jsonl
"""
import collections, json, statistics, sys

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "data/long100/manifest.jsonl")]
print(f"{len(rows)} episodes, {sum(r['probes'] for r in rows)} released probes, "
      f"{sum(r['pool'] for r in rows)} tasks in the pools, "
      f"{sum(r['tokens'] for r in rows) / 1e6:.1f}M tokens in total\n")
print(f"{'band':>6} {'n':>3} {'tokens (min-max)':>18} {'sessions':>9} {'probes':>7} {'traps':>6} {'far':>5} {'crutch':>7} {'rej':>5} {'pad dup':>8}")
for tgt, grp in sorted(collections.defaultdict(list, {t: [r for r in rows if r["target_tokens"] == t]
                                                          for t in {r["target_tokens"] for r in rows}}).items()):
    print(f"{tgt // 1000:>5}k {len(grp):>3} {min(r['tokens'] for r in grp) // 1000:>7}k-{max(r['tokens'] for r in grp) // 1000:<6}k "
          f"{statistics.mean(r['sessions'] for r in grp):>9.0f} {sum(r['probes'] for r in grp):>7} "
          f"{sum(r['traps'] for r in grp) / sum(r['probes'] for r in grp):>6.2f} "
          f"{sum(r['far'] for r in grp) / sum(r['probes'] for r in grp):>5.2f} "
          f"{sum(r['crutch'] for r in grp):>7} {statistics.mean(r['rejections'] for r in grp):>5.1f} "
          f"{statistics.mean(r['pad_dup'] for r in grp):>8.1%}")
print()
for w in sorted({r["world"] for r in rows}):
    grp = [r for r in rows if r["world"] == w]
    print(f"  {w:<16} {len(grp):>3} episodes  layout={grp[0]['layout']:<9} probes={sum(r['probes'] for r in grp):>4}  "
          f"traps={sum(r['traps'] for r in grp) / sum(r['probes'] for r in grp):.2f}  far={sum(r['far'] for r in grp) / sum(r['probes'] for r in grp):.2f}  "
          f"mean w={statistics.mean(r['mean_weight'] for r in grp):.2f}")
