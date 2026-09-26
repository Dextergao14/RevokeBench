"""
One contract for every memory system under test.

The runner (eval/adapters/memory_runner.py) replays an episode session by
session.  A raw window of the most recent `window` tokens stays in the prompt
verbatim; every session that leaves the window is handed to the memory system
through `observe`.  At a task the runner asks the memory system for whatever it
wants placed in the prompt (`recall`), the backbone acts, and the memory system
is told what the agent did (`record_action`) -- and nothing else.  REVOKE gives
no correctness feedback during an episode: the trace is graded afterwards, so an
experiential system can reflect on its own trajectory but never on a reward.

Every system uses the same backbone for its own LLM calls (`LLM.chat`), so a
memory system is compared with the raw model on equal terms, and the same
local embedder (`LLM.embed`) where it needs one.  Token accounting flows through
the LLM object, so a run's cost includes the memory system's own calls.

Cross-episode consolidation: when the runner is started with --persist, the
same backend instance sees the episodes of one world in sequence and
`begin_episode(..., persist=True)` must keep what it has.  This is the setting
under which "more experience, more violations" is measured.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence

API = "https://openrouter.ai/api/v1/chat/completions"
TOK = 4                                              # chars per token, approximate


def approx_tokens(text: str) -> int:
    return (len(text) + 1) // TOK


EMPTY_TRIES = 5                       # attempts at a completion that comes back empty
EMPTY_BACKOFF = (2, 8, 30, 60, 120)   # seconds between them


def _empty_completion(resp) -> bool:
    """HTTP 200 whose choice carries neither text nor a tool call.  A completion cut
    by the output budget (finish_reason == "length") is the model's own doing and is
    returned as is; anything else empty is treated as a dropped generation."""
    ch = (resp.get("choices") or [{}])
    if not ch:
        return True
    msg = ch[0].get("message") or {}
    if msg.get("tool_calls") or (msg.get("content") or "").strip():
        return False
    return (ch[0].get("finish_reason") or "") != "length"


class LLM:
    """The backbone under test, shared by the agent and its memory system."""

    def __init__(self, model: str, key: Optional[str] = None, temperature: float = 0.0,
                 reasoning: Optional[str] = "low", timeout: int = 120):
        self.model = model
        self.key = key or os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise SystemExit("set OPENROUTER_API_KEY")
        self.temperature = temperature
        self.reasoning = reasoning
        self.timeout = timeout
        self.usage = {"in": 0, "out": 0, "cost": 0.0, "calls": 0, "memory_calls": 0}
        self._embedder = None

    # -- chat ---------------------------------------------------------------
    def chat(self, messages: List[Dict], max_tokens: int = 2000, tools=None,
             tool_choice: str = "auto", memory_call: bool = True, tries: int = 3,
             reasoning="default") -> Dict:
        """`reasoning` overrides the instance default for this call: a dict is sent as
        OpenRouter's `reasoning` object verbatim, None/False sends none, "default"
        keeps the instance setting.  A memory system whose call must return plain
        structured output (not a thought stream) passes {"enabled": False}."""
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                "messages": messages}
        if reasoning == "default":
            # a memory system's own call must come back as text, not as a thought
            # stream: reasoning models otherwise spend the whole completion budget
            # thinking and return nothing (measured: 6,500 reasoning tokens, empty
            # notes).  The agent's act calls keep the instance setting.
            if memory_call:
                body["reasoning"] = {"enabled": False}
            elif self.reasoning:
                body["reasoning"] = {"effort": self.reasoning}
        elif isinstance(reasoning, dict):
            body["reasoning"] = reasoning
        elif isinstance(reasoning, str) and reasoning:
            body["reasoning"] = {"effort": reasoning}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        err = ""
        for a in range(max(tries, EMPTY_TRIES)):
            if a >= tries and err != "empty completion":
                break                                            # extra attempts are for empties only
            try:
                req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                    "X-Title": "RevokeBench"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                u = resp.get("usage") or {}
                self.usage["in"] += u.get("prompt_tokens", 0)
                self.usage["out"] += u.get("completion_tokens", 0)
                self.usage["cost"] += u.get("cost", 0) or 0
                self.usage["calls"] += 1
                self.usage["memory_calls"] += int(memory_call)
                if _empty_completion(resp):
                    # HTTP 200 with nothing in it: the provider dropped the generation
                    # (some providers do this for hours at a time).  Not a decision and
                    # not a rule extraction -- retry with a longer back-off, and if it
                    # persists report it as an error so the step is excluded, not graded
                    # as an abstention and not silently folded into the notes.
                    self.usage["empty"] = self.usage.get("empty", 0) + 1
                    err = "empty completion"
                    time.sleep(EMPTY_BACKOFF[min(a, len(EMPTY_BACKOFF) - 1)])
                    continue
                return resp
            except urllib.error.HTTPError as e:                          # noqa: PERF203
                raw = e.read()[:400].decode(errors="replace")
                err = f"HTTP {e.code}: {raw[:160]}"
                if e.code in (400, 422) and "reasoning" in body and "reasoning" in raw.lower():
                    body.pop("reasoning", None)              # provider rejects the field: retry without it
                    continue
                time.sleep(min(6 if e.code == 402 else 2 ** a, 10))
            except Exception as e:                                       # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(min(2 ** a, 8))
        return {"error": err}

    def text(self, messages: List[Dict], max_tokens: int = 2000, **kw) -> str:
        """Plain completion for a memory system's own reasoning; '' on failure."""
        resp = self.chat(messages, max_tokens=max_tokens, **kw)
        if "error" in resp:
            return ""
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        return (msg.get("content") or "").strip()

    # -- embeddings --------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Local embeddings: sentence-transformers when installed, otherwise a
        deterministic hashed bag-of-words (no network, no cost)."""
        if self._embedder is None:
            if os.environ.get("REVOKE_EMBED_HASH"):
                self._embedder = "hash"
            if self._embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer        # type: ignore
                    self._embedder = SentenceTransformer(os.environ.get("REVOKE_EMBEDDER", "all-MiniLM-L6-v2"))
                except Exception:                                            # noqa: BLE001
                    self._embedder = "hash"
        if self._embedder == "hash":
            return [_hash_embed(t) for t in texts]
        return [list(map(float, v)) for v in self._embedder.encode(list(texts), normalize_embeddings=True)]


_WORD = re.compile(r"[a-z0-9][a-z0-9\-']+")


def _hash_embed(text: str, dim: int = 512) -> List[float]:
    v = [0.0] * dim
    toks = _WORD.findall(text.lower())
    for i, t in enumerate(toks):
        for g in (t, toks[i - 1] + "_" + t if i else None):
            if not g:
                continue
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class MemoryBackend:
    """Base class.  Subclasses override observe / recall (and, for experiential
    systems, record_action / end_episode).  Everything is plain Python; a
    backend that wraps an external library should keep the library's own state
    behind these six calls."""

    name = "base"
    persist_across_episodes = True      # can keep memory between episodes of a world
    uses_llm = True                     # makes its own calls through self.llm
    recalls = True                      # returns text at a task

    def __init__(self, llm: LLM, cfg: Optional[Dict] = None):
        self.llm = llm
        self.cfg = cfg or {}
        self.item = None

    # lifecycle
    def begin_episode(self, item: Dict, persist: bool = False) -> None:
        """`item` is the BLIND view (transcript, tools, options; no labels).
        Reset memory unless persist is set and the backend supports it."""
        self.item = item

    def observe(self, session_index: int, text: str) -> None:
        """A session block has left the raw window for good."""

    def recall(self, task_text: str, budget_tokens: int) -> str:
        """Text to place in the prompt at a task, at most about budget_tokens."""
        return ""

    def record_action(self, probe_id: str, task_text: str, action: str, rationale: str) -> None:
        """What the agent did.  No correctness signal is ever passed here."""

    def end_episode(self) -> None:
        """Between-episode consolidation for experiential systems."""

    def stats(self) -> Dict:
        return {}

    # helper
    def clip(self, text: str, budget_tokens: int) -> str:
        lim = budget_tokens * TOK
        return text if len(text) <= lim else text[-lim:]


class NoMemory(MemoryBackend):
    """Sessions that leave the window are gone (the `truncate` ablation)."""
    name = "none"
    uses_llm = False
    recalls = False
    persist_across_episodes = False


class CompactNotes(MemoryBackend):
    """The agent's own running notes, bounded (the `compact` condition of the
    pilot, re-expressed as a backend so every system runs through one harness)."""
    name = "compact"

    SYS = ("You are compacting your own working memory. Older sessions of a long conversation are "
           "about to leave your context for good. Rewrite them into notes you will have to act on "
           "later, with no access to the original text.\n\nKeep, in compact form: every rule currently "
           "in force and who issued it, what overrides what and why, whose statements count as policy "
           "and any role changes, every prohibition that has been lifted or reinstated, conditional rules "
           "and whether their conditions currently hold, numeric thresholds and their current values, "
           "certification or list membership and its amendments, and the actions you took and when.\n\n"
           "Drop chatter, rumours, proposals that never became rules, and pleasantries -- but note "
           "explicitly when something that sounds like a rule is not one. Be terse and specific. Never "
           "invent a rule that was not stated.")

    def __init__(self, llm, cfg=None):
        super().__init__(llm, cfg)
        self.cap = int(self.cfg.get("notes_cap", 4000))
        self.notes = ""
        self.pending: List[str] = []
        self.n_compactions = 0
        self.over_cap = 0
        self.fold_failed = 0

    def begin_episode(self, item, persist=False):
        super().begin_episode(item, persist)
        if not persist:
            self.notes = ""
        self.pending = []

    def observe(self, session_index, text):
        self.pending.append(text)
        # fold as soon as roughly a quarter of the cap has accumulated, so each
        # compaction carries a small load (small, frequent compactions hold up better)
        if sum(approx_tokens(p) for p in self.pending) >= max(300, self.cap // 4):
            self._fold()

    def _fold(self):
        if not self.pending:
            return
        words = int(self.cap * 0.75)
        prompt = (f"Your current notes:\n{self.notes or '(none yet)'}\n\nSessions leaving your context now:\n"
                  + "\n".join(self.pending) + f"\n\nWrite your updated notes.  Hard limit: {words} words. If space is "
                  "short, drop actions taken and one-off events before any rule, ruling, condition, list membership or role.")
        new = self.llm.text([{"role": "system", "content": self.SYS}, {"role": "user", "content": prompt}],
                            max_tokens=int(self.cap * 1.5) + 500)
        if not new:
            # the rewrite failed (empty completion or HTTP error after retries): keep the
            # pending sessions for the next fold instead of dropping them on the floor
            self.fold_failed += 1
            return
        if new:
            if len(new) > self.cap * TOK * 1.2:
                cond = self.llm.text([{"role": "system", "content": self.SYS}, {"role": "user", "content":
                                      f"These notes are too long ({len(new.split())} words). Rewrite them in under {words} words. "
                                      "Keep every rule and who issued it, what overrides what, conditions and whether they hold, "
                                      "thresholds, list membership, and roles. Drop actions taken and one-off events first.\n\n" + new}],
                                     max_tokens=int(self.cap * 1.5) + 500)
                new = cond or new
                self.over_cap += int(len(new) > self.cap * TOK * 1.2)
            self.notes = new
        self.pending = []
        self.n_compactions += 1

    def recall(self, task_text, budget_tokens):
        self._fold()
        return ("Your notes on the sessions that have left your context:\n" + self.notes) if self.notes else ""

    def record_action(self, probe_id, task_text, action, rationale):
        self.pending.append(f"you: {(rationale or action)[:160]}")

    def stats(self):
        return {"compactions": self.n_compactions, "over_cap": self.over_cap, "notes_chars": len(self.notes),
                "fold_failed": self.fold_failed}


REGISTRY: Dict[str, type] = {"none": NoMemory, "compact": CompactNotes}


def load_backend(name: str):
    """Built-in names, or eval/memory/<name>.py exposing a `Backend` class."""
    if name in REGISTRY:
        return REGISTRY[name]
    import importlib
    mod = importlib.import_module(f"eval.memory.{name}")
    return getattr(mod, "Backend")
