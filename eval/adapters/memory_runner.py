#!/usr/bin/env python3
"""
Run a memory system through REVOKE episodes.

    python3 eval/adapters/memory_runner.py --blind data/long100/long100_blind.jsonl.gz \
        --model z-ai/glm-5.3-flash --memory compact --window 8000 --out runs/mem \
        [--items id1,id2] [--persist] [--workers 4] [--cfg '{"notes_cap": 4000}']

One continuous pass per episode.  The most recent --window tokens of transcript
stay in the prompt verbatim; every session that scrolls out is handed to the
memory system (`observe`).  At a task the prompt is: system + setting +
memory.recall(task) + raw window + the task; the backbone must call the act tool
once.  The memory system is then told what the agent did, and nothing else.

--memory names a backend: built-in `none` / `compact` / `full` (window
unbounded, no memory), or eval/memory/<name>.py.  --persist keeps one backend
instance across the episodes of a world, in manifest order, so consolidation
across episodes can be measured; the trace records the episode's position in
that sequence.  Traces are checkpointed per probe and graded by
scripts/long_report.py exactly like the other runners.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eval", "adapters"))

from episode_runner import SYSTEM, act_from, resolves, tool_result_messages   # noqa: E402
from eval.memory.base import LLM, TOK, approx_tokens, load_backend  # noqa: E402

MAX_TOOL_TURNS = 4          # read calls the agent may make before it must act
JSON_TAIL = ("\n\n---\nAct on the last message. Reply with ONLY a JSON object, no prose:\n"
             '{{"{param}": "<the option name exactly as offered>"}}')


def run_episode(item, llm: LLM, backend, window: int, recall_budget: int, max_tokens: int,
                sink_path: str, persist: bool, seq_index: int, limit: int = 0):
    tools = [{"type": "function", "function": {
        "name": t["name"], "description": t["doc"],
        "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in t["params"]},
                       "required": list(t["params"])}}} for t in item["tools"]]
    setting = item.get("setting", "")
    sys_msg = SYSTEM.format(domain=item["domain"], act=item["act_tool"],
                            setting=(" " + setting.strip()) if setting else "")
    you = item.get("you_prefix", "you")
    backend.begin_episode(item, persist=persist)
    blocks: list[str] = []                       # raw window, oldest first
    steps = []
    sink = open(sink_path, "a") if sink_path else None
    t0 = time.time()

    def ctx_tokens():
        return sum(approx_tokens(b) for b in blocks)

    def shed():
        if window <= 0:
            return
        while blocks and ctx_tokens() > window:
            oldest = blocks.pop(0)
            m = re.match(r"\[Session (\d+)", oldest)      # also matches "[Session N continued]"
            backend.observe(int(m.group(1)) if m else -1, oldest)

    n_probes = 0
    consecutive_empty = 0
    for s in item["sessions"]:
        if limit and n_probes >= limit:
            break
        lines = [f"[Session {s['index']}]"]
        for t in s["turns"]:
            pid = t.get("probe_id")
            if not pid:
                lines.append(t["text"])
                continue
            if limit and n_probes >= limit:
                continue
            n_probes += 1
            # everything before the task in this session enters the window first
            blocks.append("\n".join(lines))
            lines = [f"[Session {s['index']} continued]"]
            shed()
            mem = backend.recall(t["text"], recall_budget)
            msgs = [{"role": "system", "content": sys_msg}]
            if mem:
                msgs.append({"role": "system", "content": mem})
            msgs.append({"role": "user", "content": "\n".join(blocks) + "\n" + t["text"] +
                         f"\n\n---\nAct on the last message: call `{item['act_tool']}` exactly once with one of the offered options."})
            val, text, n_reads, via = "", "", 0, "tool"
            base_msgs = list(msgs)
            for turn in range(MAX_TOOL_TURNS):
                resp = llm.chat(msgs, max_tokens=max_tokens * (3 if turn and not n_reads else 1),
                                tools=tools, memory_call=False)
                if "error" in resp:
                    break
                val, text = act_from(resp, item)
                if val and resolves(item, val):
                    break
                if val:                              # unusable argument: try JSON below
                    val = ""
                msg = ((resp.get("choices") or [{}])[0].get("message") or {})
                if msg.get("tool_calls"):            # read tools: answer them and continue
                    msgs = msgs + tool_result_messages(msg, item)
                    n_reads += len(msg["tool_calls"])
            if not val:
                # Same fallback the per-probe runner gives every model: some
                # backbones do not produce a usable tool call at all (one writes
                # its whole chain of thought into the argument slot), so ask for
                # the choice as JSON without tools.  Without this the two
                # conditions would offer the same model different chances.
                # rebuild from the ORIGINAL prompt: after a read turn, msgs ends
                # with a tool message, not the task
                jmsgs = base_msgs[:-1] + [{"role": "user", "content":
                                           base_msgs[-1]["content"].rsplit("\n\n---\n", 1)[0]
                                           + JSON_TAIL.format(param=item["act_param"])}]
                resp = llm.chat(jmsgs, max_tokens=max_tokens, memory_call=False)
                if "error" not in resp:
                    v2, t2 = act_from(resp, item)
                    if v2:
                        val, text, via = v2, t2, "json"
            step = {"probe_id": pid, "ctx_tokens": ctx_tokens(), "mem_chars": len(mem),
                    "seq_index": seq_index, "n_reads": n_reads, "via": via}
            if val:
                step["tool_calls"] = [{"name": item["act_tool"], "arguments": {item["act_param"]: val}}]
                step["text"] = text[:300]
                backend.record_action(pid, t["text"], val, text)
                lines.append(f"{you}: {(text or val)[:120]}")
            else:
                step["tool_calls"] = []
                step["error"] = resp.get("error") or "no action in response"
            if str(step.get("error", "")).startswith("empty completion"):
                # the provider is dropping generations: wait before the next task rather
                # than sweeping the rest of the episode into the same outage
                outage = min(120 * 2 ** consecutive_empty, 1800)
                consecutive_empty += 1
                print(f"    empty completion at {pid}; pausing {outage}s", flush=True)
                time.sleep(outage)
            else:
                consecutive_empty = 0
            steps.append(step)
            if sink:
                sink.write(json.dumps({"id": item["id"], "model": llm.model, "mode": backend.name,
                                       "budget": window, "steps": [step]}) + "\n")
                sink.flush()
        blocks.append("\n".join(lines))
        shed()
    backend.end_episode()
    if sink:
        sink.close()
    return {"id": item["id"], "model": llm.model, "mode": backend.name, "budget": window,
            "recall_budget": recall_budget, "persist": persist, "seq_index": seq_index,
            "steps": steps, "usage": dict(llm.usage), "memory_stats": backend.stats(),
            "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--memory", required=True, help="none | compact | full | <eval/memory module>")
    ap.add_argument("--window", type=int, default=8000, help="raw transcript kept verbatim, tokens (0 = unbounded)")
    ap.add_argument("--recall-budget", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--cfg", default="{}", help="JSON passed to the backend")
    ap.add_argument("--items", default="")
    ap.add_argument("--persist", action="store_true", help="one backend instance per world, episodes in manifest order")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks of each episode")
    ap.add_argument("--manifest", default="", help="for --persist ordering (default: <blind dir>/manifest.jsonl)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY"))
    a = ap.parse_args()
    if "full" in os.path.basename(a.blind) and not a.blind.endswith("_blind.jsonl.gz"):
        sys.exit("refusing: that file name suggests it carries ground truth")
    opener = gzip.open if a.blind.endswith(".gz") else open
    items = [json.loads(l) for l in opener(a.blind, "rt")]
    if any("licensed" in p for i in items for p in i["probes"]):
        sys.exit("refusing: dataset carries ground truth")
    if a.items:
        keep = set(a.items.split(","))
        items = [i for i in items if i["id"] in keep]
    os.makedirs(a.out, exist_ok=True)
    cfg = json.loads(a.cfg)
    mem_name = a.memory
    window = 0 if mem_name == "full" else a.window
    if mem_name == "full":
        mem_name = "none"
    Backend = load_backend(mem_name)
    tag = f"{a.model.replace('/', '__').replace(':', '_')}__{a.memory}__w{a.window}{'__persist' if a.persist else ''}"
    final = os.path.join(a.out, tag + ".jsonl")

    def job(group):
        """A group is one world's episodes (persist) or a single episode."""
        llm = LLM(a.model, a.key)
        backend = Backend(llm, cfg)
        rows = []
        for k, it in enumerate(group):
            sink = os.path.join(a.out, f"{tag}__{it['id']}.jsonl.part")
            rows.append(run_episode(it, llm, backend, window, a.recall_budget, a.max_tokens, sink, a.persist, k, a.limit))
        return rows

    # resume: skip episodes already complete in the final file.  A long run is
    # hours of API calls, so an interruption must not mean starting over.
    done_ids = set()
    if os.path.exists(final):
        for line in open(final):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("steps"):
                done_ids.add(r["id"])
    if done_ids:
        before = len(items)
        items = [i for i in items if i["id"] not in done_ids]
        print(f"resume: {before - len(items)} episodes already in {final}, {len(items)} to go", flush=True)
    if not items:
        print("nothing to do")
        return

    if a.persist:
        order = {}
        man = a.manifest or os.path.join(os.path.dirname(a.blind), "manifest.jsonl")
        if os.path.exists(man):
            for i, l in enumerate(open(man)):
                order[json.loads(l)["id"]] = i
        by_world = collections.defaultdict(list)
        for it in sorted(items, key=lambda x: order.get(x["id"], 10 ** 6)):
            by_world[it["domain"]].append(it)
        groups = list(by_world.values())
    else:
        groups = [[it] for it in items]
    print(f"{len(items)} episodes in {len(groups)} groups; model {a.model}; memory {a.memory}; window {a.window}; "
          f"persist {a.persist}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(final, "a") as fh:
        futs = {ex.submit(job, g): g for g in groups}
        for f in as_completed(futs):
            try:
                rows = f.result()
            except Exception as e:                                       # noqa: BLE001
                print(f"  FAILED {[i['id'] for i in futs[f]]}: {e}", flush=True)
                continue
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done += 1
                acted = sum(1 for s in r["steps"] if s.get("tool_calls"))
                print(f"  [{done}/{len(items)}] {r['id']:<40} acted={acted}/{len(r['steps'])} "
                      f"${r['usage']['cost']:.2f} mem={r['memory_stats']} {r['seconds']}s", flush=True)
    print(f"-> {final}")


if __name__ == "__main__":
    main()
