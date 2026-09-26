#!/usr/bin/env python3
"""Offline acceptance test for a memory backend: no network, no cost.

    python3 scripts/smoke_memory.py --memory memp [--blind data/long100/long100_blind.jsonl.gz] [--limit 6]

Runs one episode through eval/adapters/memory_runner.py with a FAKE backbone:
every act call picks the first offered option; every other call (the memory
system's own LLM use) returns a short canned paragraph.  Embeddings are the
hashed local embedder.  Checks, and exits non-zero on failure:

  1. the backend loads through eval.memory.base.load_backend
  2. the episode runs to the end; every task produced an action
  3. the backend was exercised: observe() saw sessions, recall() returned text
     for at least one task after the window first overflowed
  4. no ground truth reached the backend: the blind item has no `licensed`,
     `violating`, `blamed`, `stale_trap` or `events`, and the backend never
     opened a *_full* file
  5. the backend's own LLM calls went through the shared LLM object
     (llm.usage["memory_calls"] > 0 for a backend that declares uses_llm)
  6. begin_episode(persist=True) keeps state across two episodes when the
     backend declares persist_across_episodes

Prints the backend's stats() and the per-task recall sizes.
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eval", "adapters"))
os.environ.setdefault("OPENROUTER_API_KEY", "smoke-test-key")
os.environ["REVOKE_EMBED_HASH"] = "1"

from eval.memory import base                                        # noqa: E402
import memory_runner as MR                                          # noqa: E402

CANNED = ("Rules noted so far: the standing choice for this task is whatever was most recently approved "
          "by someone with authority; two options were prohibited earlier and one prohibition was later "
          "withdrawn. Nothing else changes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--blind", default=os.path.join(HERE, "data", "long100", "long100_blind.jsonl.gz"))
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--window", type=int, default=800)
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    fails = []

    import gzip
    opener = gzip.open if a.blind.endswith(".gz") else open
    with opener(a.blind, "rt") as fh:
        items = [json.loads(next(fh))]          # the first episode is enough for a smoke test
    item = items[0]
    for k in ("licensed", "violating", "blamed", "stale_trap"):
        assert all(k not in p for p in item["probes"]), f"blind item carries {k}"
    assert "events" not in item and "timeline" not in item

    opened = []
    real_open = builtins.open

    def spy_open(f, *args, **kw):
        if isinstance(f, (str, bytes, os.PathLike)) and "_full" in str(f):
            opened.append(str(f))
        return real_open(f, *args, **kw)
    builtins.open = spy_open

    calls = {"act": 0, "mem": 0}

    def fake_chat(self, messages, max_tokens=2000, tools=None, tool_choice="auto", memory_call=True, tries=3, **kw):
        self.usage["calls"] += 1
        if tools:
            calls["act"] += 1
            txt = messages[-1]["content"]
            opt = txt.split("options open to you right now are: ")[1].split(".")[0].split(", ")[0] \
                if "options open to you right now are: " in txt else "unknown"
            return {"choices": [{"message": {"content": "picked", "tool_calls": [{"function": {
                "name": item["act_tool"], "arguments": json.dumps({item["act_param"]: opt})}}]}}], "usage": {}}
        calls["mem"] += 1
        self.usage["memory_calls"] += 1
        return {"choices": [{"message": {"content": CANNED}}], "usage": {}}
    base.LLM.chat = fake_chat

    try:
        Backend = base.load_backend(a.memory)
    except Exception as e:                                           # noqa: BLE001
        print(f"FAIL load_backend({a.memory}): {type(e).__name__}: {e}")
        sys.exit(1)
    llm = base.LLM("fake/backbone", "k")
    backend = Backend(llm, json.loads(a.cfg))
    observed = {"n": 0}
    orig_observe = backend.observe

    def counting_observe(idx, text):
        observed["n"] += 1
        return orig_observe(idx, text)
    backend.observe = counting_observe

    row = MR.run_episode(item, llm, backend, a.window, 2000, 400, "", False, 0, a.limit)
    acted = sum(1 for s in row["steps"] if s.get("tool_calls"))
    n = len(row["steps"])
    print(f"{a.memory}: tasks={n} acted={acted} observe()={observed['n']} act_calls={calls['act']} "
          f"memory_llm_calls={calls['mem']} recall_chars={[s['mem_chars'] for s in row['steps']]}")
    print(f"  stats: {row['memory_stats']}")
    if acted != n or n == 0:
        fails.append(f"only {acted}/{n} tasks acted")
    if observed["n"] == 0:
        fails.append("observe() never called (window never overflowed?)")
    if getattr(backend, "recalls", True) and not any(s["mem_chars"] > 0 for s in row["steps"]):
        fails.append("recall() returned nothing for every task")
    if getattr(backend, "uses_llm", True) and calls["mem"] == 0:
        fails.append("backend declares uses_llm but made no calls through the shared LLM")
    if opened:
        fails.append(f"backend opened a ground-truth file: {opened[:2]}")

    if getattr(backend, "persist_across_episodes", False) and len(items) > 1:
        before = json.dumps(backend.stats(), sort_keys=True)
        item2 = items[1]
        backend.begin_episode(item2, persist=True)
        after = json.dumps(backend.stats(), sort_keys=True)
        if before != after and before != "{}":
            fails.append(f"persist=True reset the memory state: {before} -> {after}")
        backend.begin_episode(item2, persist=False)

    builtins.open = real_open
    if fails:
        print("FAIL: " + "; ".join(fails))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
