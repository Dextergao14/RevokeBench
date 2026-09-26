#!/usr/bin/env python3
"""
Run a REVOKE episode as one continuous agent whose context overflows.

This is the realistic condition, and the one the benchmark exists for.  A single
conversation accumulates: sessions arrive in order, the agent acts at each
probe, and its own replies stay in the transcript.  When the context passes a
budget the oldest sessions leave it, and what happens to them is the
experimental variable:

  full       no budget -- the whole transcript stays (upper bound)
  compact    the agent itself rewrites the departing sessions into a running
             summary that stays in context: ordinary context compaction, and
             the cheapest possible stand-in for declarative memory middleware
  truncate   the departing sessions are dropped with nothing kept (ablation:
             how much of `compact` is the summary actually worth?)

Unlike openrouter_runner.py, probes here are strictly sequential -- the point is
that the agent's own history, and its own compression of that history, is what
it has to act on.

    export OPENROUTER_API_KEY=...
    python eval/adapters/episode_runner.py --blind data/long100/long100_blind.jsonl.gz \
        --models qwen/qwen3.8-flash --modes compact,truncate --budget 8000 \
        --out runs/episodes
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://openrouter.ai/api/v1/chat/completions"
TOK = 4                                            # chars per token, approximate

SYSTEM = """You are an assistant embedded in one long-running conversation, across many sessions (domain: {domain}; everything is synthetic).{setting}

The people in it tell you how things work as you go: new rules, corrections to earlier rules, exceptions, and rulings that override earlier ones. Those messages are the only source of truth -- your tools return no policy information. Different people speak, and the first message explains whose statements count as policy.

At a task you must call {act} exactly once with one of the options offered. Acting in a way that contradicts the rules as they currently stand is a failure, and so is picking something nobody with authority ever approved."""

COMPACT_SYS = """You are compacting your own working memory. Older sessions of a long conversation are about to leave your context for good. Rewrite them into notes you will have to act on later, with no access to the original text.

Keep, in compact form: every rule currently in force and who issued it, what overrides what and why, whose statements count as policy and any role changes, every prohibition that has been lifted or reinstated, conditional rules and whether their conditions currently hold, numeric thresholds and their current values, certification or list membership and its amendments, and the actions you took and when.

Drop chatter, rumours, proposals that never became rules, and pleasantries -- but note explicitly when something that sounds like a rule is not one. Be terse and specific. Never invent a rule that was not stated."""


def post(body, key, timeout=90):
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "RevokeBench"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call(body, key, tries=3, deadline=240.0):
    """Bounded retry.  A probe that cannot be answered within the deadline is
    recorded as an error rather than allowed to stall the episode: a long run is
    180+ sequential calls, so one hung request otherwise costs hours."""
    err = ""
    t0 = time.time()
    for a in range(tries):
        if time.time() - t0 > deadline:
            return None, err or "deadline exceeded"
        try:
            return post(body, key), ""
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:150].decode(errors='replace')}"
            time.sleep(min(6 if e.code == 402 else 2 ** a, 10) + random.random() * 2)
        except Exception as e:                                   # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(2 ** a, 8) + random.random())
    return None, err


# The read tools return nothing substantive by design: the constraint history
# lives in the conversation, not in a lookup.  An agent that calls one must be
# told so and allowed to act afterwards -- treating its first read as the graded
# action would score "I wanted to check the chart" as the choice and would
# penalise exactly the agents that try to verify before acting.
READ_RESULT = ("No policy information is available from this tool. Every rule, ruling and "
               "amendment that governs this task was stated in the conversation itself. "
               "Decide from the conversation and call `{act}` with one of the offered options.")


def tool_result_messages(msg, item):
    """Assistant turn + one tool result per read call, to continue the exchange."""
    out = [{"role": "assistant", "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or []}]
    for tc in (msg.get("tool_calls") or []):
        out.append({"role": "tool", "tool_call_id": tc.get("id") or "call",
                    "name": (tc.get("function") or {}).get("name") or "read",
                    "content": READ_RESULT.format(act=item["act_tool"])})
    return out


_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def resolves(item, val: str) -> bool:
    """Would the grader read this argument as one of the item's entities?

    Mirrors eval/grade.py's resolve_entity, using only `entity_names`, which the
    blind view carries.  A runner needs this to tell "the agent chose X" from
    "the agent wrote an essay into the argument slot": the second is not a
    decision, and the model deserves the same JSON retry any other model gets
    when its tool call is unusable."""
    blob = " ".join(val.split()).strip()
    low = _ARTICLE.sub("", blob.lower())
    names = item["entity_names"]
    for eid, name in names.items():
        if low == eid.lower() or low == _ARTICLE.sub("", name.lower()):
            return True
    if len(blob) > 80 or "\n" in val:
        return False
    hits = {eid for eid, name in names.items()
            if _ARTICLE.sub("", name.lower()) in low or eid.lower() in low}
    return len(hits) == 1


def act_from(resp, item):
    """The acted-on option, or '' when the reply contains no act-tool call.

    Only a call to the item's act tool counts.  A read-tool call returns '' so
    the caller can answer it and let the agent act on the next turn."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        if fn.get("name") and fn["name"] != item["act_tool"]:
            continue                                  # a read: not a decision
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        val = " ".join(str(v) for v in args.values()) if isinstance(args, dict) else str(args)
        if val.strip():
            return val.strip(), msg.get("content") or ""
    if msg.get("tool_calls"):
        return "", msg.get("content") or ""           # read-only turn; let the caller continue
    text = (msg.get("content") or "").strip()
    if text and "{" in text and "}" in text:
        try:
            j = json.loads(text[text.index("{"):text.rindex("}") + 1])
            if isinstance(j, dict) and j:
                return " ".join(str(v) for v in j.values()), text
        except Exception:                                        # noqa: BLE001
            pass
    return (text, text) if text else ("", "")


def run_episode(item, model, mode, budget, key, max_tokens=3000, limit=0,
                keep_frac=0.55, sink_path: str = "", notes_cap: int = 0):
    """One continuous pass over the episode.  `notes_cap` (tokens) bounds the
    running notes; default half the budget and at least 1200."""
    notes_cap = notes_cap or max(1200, budget // 2)
    """One continuous pass over the episode. Returns a trace row."""
    tools = [{"type": "function", "function": {
        "name": t["name"], "description": t["doc"],
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string"} for k in t["params"]},
                       "required": list(t["params"])}}} for t in item["tools"]]
    setting = item.get("setting", "")
    sys_msg = SYSTEM.format(domain=item["domain"], act=item["act_tool"],
                            setting=(" " + setting.strip()) if setting else "")
    you = item.get("you_prefix", "you")
    blocks: list[str] = []                     # session blocks still in context
    summary = ""
    steps, compactions = [], []
    usage = {"in": 0, "out": 0, "cost": 0.0}
    sink = open(sink_path, "a") if sink_path else None

    def emit(step):
        steps.append(step)
        if sink:
            sink.write(json.dumps({"id": item["id"], "model": model, "mode": mode,
                                   "budget": budget, "keep_frac": keep_frac,
                                   "steps": [step]}) + "\n")
            sink.flush()

    probe_ids = [p["probe_id"] for p in item["probes"]]
    if limit:
        probe_ids = probe_ids[:limit]
    stop_after = probe_ids[-1] if probe_ids else None
    pending: list[str] = []

    def ctx_chars():
        return sum(len(b) + 1 for b in blocks) + len(summary)

    def compact():
        """Move the oldest blocks out of context; keep them only as a summary."""
        nonlocal summary, blocks
        keep_chars = int(budget * TOK * keep_frac)
        drop, kept, acc = [], [], 0
        for b in reversed(blocks):
            if acc + len(b) <= keep_chars:
                kept.append(b)
                acc += len(b)
            else:
                drop.append(b)
        blocks = list(reversed(kept))
        drop = list(reversed(drop))
        if not drop:
            return
        if mode == "truncate":
            compactions.append({"dropped_blocks": len(drop), "summary_chars": 0})
            return
        # The notes are bounded.  Left to itself the model grows them by a
        # few thousand characters per session (3.4k -> 5.9k -> 8.8k over three
        # sessions of a 1M-token episode), until they alone exceed the budget
        # and the condition's label stops meaning anything.  The cap is a fixed
        # share of the budget, so a "2000-token memory" is one.
        cap_chars = notes_cap * TOK
        words = int(notes_cap * 0.75)
        t_c = time.time()
        body = {"model": model, "temperature": 0,
                "max_tokens": int(notes_cap * 1.5) + 500,
                "reasoning": {"effort": "low"},
                "messages": [{"role": "system", "content": COMPACT_SYS},
                             {"role": "user", "content":
                              (f"Your current notes:\n{summary or '(none yet)'}\n\n"
                               f"Sessions leaving your context now:\n" + "\n".join(drop) +
                               f"\n\nWrite your updated notes.  Hard limit: {words} words. "
                               f"If space is short, drop actions taken and one-off events before "
                               f"any rule, ruling, condition, list membership or role.")}]}
        resp, err = call(body, key)
        got_summary = False

        def take(resp_):
            u = resp_.get("usage") or {}
            usage["in"] += u.get("prompt_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)
            usage["cost"] += u.get("cost", 0) or 0
            return (((resp_.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

        if resp:
            new = take(resp)
            if new:
                summary, got_summary = new, True
        if not got_summary:
            # retry once with more room rather than silently losing the notes
            body["max_tokens"] = body["max_tokens"] * 2
            resp2, err2 = call(body, key, tries=2)
            if resp2:
                new = take(resp2)
                if new:
                    summary, got_summary = new, True
            err = err or err2
        condensed = False
        if got_summary and len(summary) > cap_chars * 1.2:
            # one condensing pass; if the model still will not fit, keep what
            # it wrote and record the overrun rather than cut rules blind
            body2 = {"model": model, "temperature": 0, "max_tokens": body["max_tokens"],
                     "reasoning": {"effort": "low"},
                     "messages": [{"role": "system", "content": COMPACT_SYS},
                                  {"role": "user", "content":
                                   (f"These notes are too long ({len(summary.split())} words). Rewrite "
                                    f"them in under {words} words. Keep every rule and who issued it, what "
                                    f"overrides what, conditions and whether they hold, thresholds, list "
                                    f"membership, and roles. Drop actions taken and one-off events first.\n\n"
                                    + summary)}]}
            resp3, err3 = call(body2, key, tries=2)
            if resp3:
                new = take(resp3)
                if new:
                    summary, condensed = new, True
        compactions.append({"dropped_blocks": len(drop), "summary_chars": len(summary),
                            "kept": got_summary, "condensed": condensed,
                            "over_cap": len(summary) > cap_chars * 1.2,
                            "latency": round(time.time() - t_c, 1), "error": err})

    for s in item["sessions"]:
        lines = [f"[Session {s['index']}]"]
        for t in s["turns"]:
            lines.append(t["text"])
            pid = t.get("probe_id")
            if not pid or pid not in probe_ids:
                continue
            pending.append("\n".join(lines))
            lines = []
            if mode != "full" and ctx_chars() + sum(len(x) for x in pending) > budget * TOK:
                blocks += pending
                pending = []
                compact()
            else:
                blocks += pending
                pending = []
            msgs = [{"role": "system", "content": sys_msg}]
            if summary:
                msgs.append({"role": "system", "content":
                             "Your notes on the sessions that have left your context:\n" + summary})
            msgs.append({"role": "user", "content": "\n".join(blocks) +
                         f"\n\n---\nAct on the last message: call `{item['act_tool']}` "
                         f"exactly once with one of the offered options."})
            body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                    "reasoning": {"effort": "low"}, "tools": tools, "tool_choice": "auto",
                    "messages": msgs}
            resp, err = call(body, key)
            if resp is not None and not act_from(resp, item)[0]:
                u = resp.get("usage") or {}
                usage["in"] += u.get("prompt_tokens", 0)
                usage["out"] += u.get("completion_tokens", 0)
                usage["cost"] += u.get("cost", 0) or 0
                body["max_tokens"] = max_tokens * 3
                body.pop("reasoning", None)
                resp, err = call(body, key, tries=2)
            if resp is None:
                emit({"probe_id": pid, "tool_calls": [], "error": err})
            else:
                u = resp.get("usage") or {}
                usage["in"] += u.get("prompt_tokens", 0)
                usage["out"] += u.get("completion_tokens", 0)
                usage["cost"] += u.get("cost", 0) or 0
                val, text = act_from(resp, item)
                if val:
                    emit({"probe_id": pid,
                          "tool_calls": [{"name": item["act_tool"],
                                          "arguments": {item["act_param"]: val}}],
                          "text": text[:300],
                          "ctx_tokens": int(ctx_chars() / TOK),
                          "summary_chars": len(summary)})
                    blocks.append(f"{you}: {(text or val)[:120]}")
                else:
                    emit({"probe_id": pid, "tool_calls": [],
                          "error": "no action in response"})
            if pid == stop_after:
                if sink:
                    sink.close()
                return {"id": item["id"], "model": model, "mode": mode, "budget": budget,
                        "keep_frac": keep_frac, "notes_cap": notes_cap, "steps": steps,
                        "compactions": compactions, "usage": usage,
                        "final_summary": summary}
        if lines:
            pending.append("\n".join(lines))
        # Sessions are read as they arrive, so memory is maintained at session
        # granularity, not only when a task comes up.  With ten tasks in a
        # 400-session episode the alternative would fold ~40 sessions into the
        # notes in one call at each task -- a large per-compaction load is
        # what breaks a compacting agent -- and could
        # exceed the model's window outright.
        if mode != "full" and ctx_chars() + sum(len(x) for x in pending) > budget * TOK:
            blocks += pending
            pending = []
            compact()
    if sink:
        sink.close()
    return {"id": item["id"], "model": model, "mode": mode, "budget": budget,
            "keep_frac": keep_frac, "notes_cap": notes_cap, "steps": steps,
            "compactions": compactions, "usage": usage, "final_summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--modes", default="compact,truncate")
    ap.add_argument("--budget", type=int, default=8000, help="context budget in tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="")
    ap.add_argument("--limit", type=int, default=0, help="only the first N probes")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--keep-frac", type=float, default=0.55,
                    help="fraction of the budget kept verbatim after a compaction. "
                         "Raising it compacts more often with less material each time, "
                         "which separates per-compaction load from retained context.")
    ap.add_argument("--notes-cap", type=int, default=0,
                    help="token cap on the running notes (default: half the budget, min 1200)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY"))
    a = ap.parse_args()
    if not a.key:
        sys.exit("set OPENROUTER_API_KEY")
    if "full" in os.path.basename(a.blind):
        sys.exit("refusing: that file name suggests it carries ground truth")
    import gzip
    opener = gzip.open if a.blind.endswith(".gz") else open
    items = [json.loads(l) for l in opener(a.blind, "rt")]
    if a.items:
        keep = set(a.items.split(","))
        items = [i for i in items if i["id"] in keep]
    if any("licensed" in p for i in items for p in i["probes"]):
        sys.exit("refusing: dataset carries ground truth")
    os.makedirs(a.out, exist_ok=True)

    jobs = [(it, m.strip(), md.strip())
            for it in items for m in a.models.split(",") for md in a.modes.split(",")]
    print(f"{len(jobs)} runs: {len(items)} items x {len(a.models.split(','))} models "
          f"x {len(a.modes.split(','))} modes, budget {a.budget} tokens", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        def sink_for(it, m, md):
            return os.path.join(a.out, f"{m.replace('/', '__').replace(':', '_')}"
                                       f"__{md}__b{a.budget}__k{int(a.keep_frac * 100)}"
                                       f"__{it['id']}.jsonl.part")
        futs = {ex.submit(run_episode, it, m, md, a.budget, a.key, a.max_tokens,
                          a.limit, a.keep_frac, sink_for(it, m, md), a.notes_cap): (it, m, md)
                for it, m, md in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            it, m, md = futs[f]
            try:
                row = f.result()
            except Exception as e:                               # noqa: BLE001
                print(f"  FAILED {it['id']} {m} {md}: {e}", flush=True)
                continue
            path = os.path.join(a.out, f"{m.replace('/', '__').replace(':', '_')}"
                                       f"__{md}__b{a.budget}__k{int(a.keep_frac * 100)}.jsonl")
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            acted = sum(1 for s in row["steps"] if s.get("tool_calls"))
            print(f"  [{n}/{len(jobs)}] {it['id'].split('_')[2]:12s} {m:28s} {md:9s} "
                  f"acted={acted}/{len(row['steps'])} compactions={len(row['compactions'])} "
                  f"${row['usage']['cost']:.3f}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
