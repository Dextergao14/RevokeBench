"""
MEMP -- procedural memory for agents, as a REVOKE memory backend.

Method
------
Fang et al. 2025, "Memp: Exploring Agent Procedural Memory" (arXiv:2508.06433).
Source: the EvolveLab provider from MemEvolve / Flash-Searcher
(``EvolveLab/providers/memp_memory_provider.py`` plus ``base_memory.py`` and
``memory_types.py``), vendored byte-identical under
``eval/memory/vendor/memp_evolvelab/`` (see PROVENANCE.txt there).  This module
imports ``MempMemoryProvider`` from that copy and subclasses it; no prompt,
record format, retrieval rule or write/adjust logic is rewritten.

MEMP stores one *procedural record* per solved task: the task text (``question``),
a <100-word high-level ``procedural_script`` distilled from the trajectory, a
<150-word ``concrete_steps_summary`` of the same trajectory, both produced by the
LLM.  Records are retrieved by cosine similarity between the embedding of the
new task text and the stored task texts (top-1 in the original).  A trajectory
marked as failed is not stored; instead the record(s) that were provided for
that task are rewritten by the LLM to absorb the lesson (``_adjust_memory``).

Mapping onto the six contract calls
-----------------------------------
begin_episode(item, persist)
    persist=False -> ``initialize()``: empty record list, empty embedding cache,
    empty provided-ids cache.  persist=True -> all three are kept (the
    cross-episode consolidation the paper measures).  The rolling buffer of
    scrolled-out sessions and the per-episode repeat tracker are always reset.
observe(session_index, text)
    Appended to a rolling buffer bounded by ``buffer_tokens``.  MEMP has no
    per-observation write; its memory is written from trajectories, so no LLM
    call is made here (cost: none).
recall(task_text, budget_tokens)
    ``provide_memory(MemoryRequest(query=task_text, context=<buffer>, status=BEGIN))``
    -> the original's formatted "[High-Level Script] / [Concrete Steps (Example)]"
    block(s), under a one-line header, clipped to the budget with ``self.clip``.
    The provider caches which record ids it served for this exact query (its
    own ``_last_provided_cache``), as upstream.
record_action(probe_id, task_text, action, rationale)
    Builds ``TrajectoryData(query=task_text, trajectory=[observation steps = the
    buffered scrolled-out sessions, a thought step = the agent's rationale, an
    action step = "Called <act_tool>(<act_param>=<action>)"], result="<act_tool>(...)")``
    and calls ``take_in_memory``.  ``result`` is the agent's own output, exactly
    what Flash-Searcher passes as the final answer; it carries no correctness.
    Default path: ``_add_new_memory`` -> two LLM calls (script, concrete
    summary) -> one new record + one embedding of the task text.
end_episode()
    MEMP has no between-episode step.  The buffer is dropped; records stay.
stats()
    Cumulative counters only (stable across ``begin_episode(persist=True)``).

When a distillation call comes back empty
-----------------------------------------
``LLM.text`` returns '' on any API failure, and a reasoning-enabled backbone can
spend the whole ``max_tokens`` budget on reasoning before emitting any content.
Upstream then stores, and later serves to the agent as a procedure to follow:

* as the ``procedural_script`` -- ``_call_llm``'s literal placeholder
  "(LLM returned empty response)" (or "(LLM call error: ...)"), which
  ``_extract_tag`` passes through unchanged for want of a ``<script>`` block and
  ``_generate_script_for_trace`` accepts as a non-empty script;
* as the ``concrete_steps_summary`` -- the full reconstructed trajectory, which
  here is the rolling session buffer: thousands of characters of raw transcript,
  of which ``recall`` (clipping from the tail) would serve a tail-slice labelled
  "[Concrete Steps (Example)]".

REVOKE stores neither.  ``_generate_script_for_trace`` and
``_summarize_trajectory_with_llm`` are wrapped: a reply that is empty or one of
upstream's placeholder strings is replaced by a short deterministic stub built
only from the task text (already the embedded retrieval key) and the action
taken -- no session text, no placeholder -- and counted in ``stats()``
(``script_stubs`` / ``summary_stubs``), so a run can be audited for how many
records are stubs.  A script the backbone cut off before ``</script>`` keeps its
partial text, with the dangling tag stripped.  The record is still written: what
the agent did on that task is real, only its distillation is missing.  Upstream's
adjust path needs no guard -- a placeholder fails ``json.loads`` there, the record
is left as it was and the attempt is counted in ``adjust_failed``.

Unknown outcome
---------------
Upstream ``take_in_memory`` branches on ``metadata["is_correct"]``: "false" ->
``_adjust_memory`` on the records served for that query; anything else -> add
a new record.  REVOKE never grades during an episode, so the failure branch
cannot be triggered by ground truth.  Consequences, as implemented:

* Every trajectory takes the add-new-memory path.  The store therefore grows
  by one record per task and never contains a "revised_after_failure" record
  unless ``adjust_on_repeat`` is set (below).  Wrong actions are distilled into
  scripts exactly like right ones -- this is the condition the benchmark wants
  to measure, not a bug.
* Upstream's concrete-summary prompt prints ``Correctness: Correct/Wrong`` from
  ``metadata.get("is_correct", False)``, i.e. it would label every REVOKE
  trajectory "Wrong".  ``_summarize_trajectory_with_llm`` is overridden with the
  same prompt except that line reads ``Correctness: Unknown (...)`` and the
  closing-sentence instruction no longer asks for a success/failure reason.
  This is the only prompt text changed anywhere.
* ``adjust_on_repeat`` (default False, ADAPTATION): when the same task stem
  (task text minus speaker, digits and the option list) recurs in the episode
  and the agent now picks a different option than last time, the new
  trajectory is handed to ``take_in_memory`` with ``is_correct="false"``, so
  upstream's own failure branch runs ``_adjust_memory`` on the record(s) that
  ``recall`` served for this task, and no new record is added.  This is a
  proxy, not ground truth: in REVOKE the licensed option legitimately changes
  as rules are issued and revoked, so a changed action is weak evidence of a
  mistake.  Off by default; report both settings if used.

Fairness / isolation
--------------------
* All LLM calls go through the upstream seam ``config["model"]``: a callable
  that flattens the provider's message parts and calls ``self.llm.text`` on
  the backbone under test (counted as memory calls).  ``_call_llm`` and
  ``_adjust_memory`` are untouched; ``_generate_script_for_trace`` runs upstream's
  own prompt and extraction through ``super()`` and only substitutes a stub for a
  reply upstream would have stored as a placeholder (above).
* ``_embed_texts`` is overridden to ``self.llm.embed`` (local).  ``initialize``
  is overridden so upstream's ``load_embedding_model`` (a Hugging Face
  download) never runs; the vendored module's ``sentence_transformers`` import
  is satisfied by a stub during import, and ``sklearn.metrics.pairwise
  .cosine_similarity`` by a NumPy equivalent only if scikit-learn is absent.
  Nothing is left in ``sys.modules`` afterwards.
* JSON persistence: ``_save_memories_to_json`` is a no-op unless
  ``store_json`` is set, in which case it writes under ``store_path`` (a fresh
  temp dir by default).  Nothing is ever loaded from disk.
* The backend sees only the blind item (``act_tool``, ``act_param``) and the
  texts the runner hands it.

Configuration (self.cfg)
------------------------
top_k            int   1       records returned per recall (upstream hard-codes 1)
buffer_tokens    int   3000    rolling buffer of scrolled-out sessions that
                               forms the observation steps of each trajectory
max_tokens       int   1000    completion cap for each of MEMP's own LLM calls
adjust_on_repeat bool  False   the adaptation described above
stub_chars       int   300     max chars per field of the stub stored when a
                               distillation call comes back empty
store_json       bool  False   mirror the record store to disk as upstream does
store_path       str   None    directory for that mirror; default a temp dir
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import tempfile
import types
from typing import Dict, List, Optional, Tuple

import numpy as np

from eval.memory.base import LLM, MemoryBackend, approx_tokens

log = logging.getLogger("revoke.memory.memp")


# --------------------------------------------------------------------------- #
# Import the vendored provider.  Its module header does
#   from sentence_transformers import SentenceTransformer
#   from sklearn.metrics.pairwise import cosine_similarity
# neither of which we use once embeddings route through self.llm.embed.
# --------------------------------------------------------------------------- #
def _np_cosine_similarity(X, Y=None):
    """Drop-in for sklearn.metrics.pairwise.cosine_similarity (row-normalised dot)."""
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Xn @ Yn.T


def _import_vendored():
    stubs: Dict[str, types.ModuleType] = {}
    # always stub sentence_transformers: importing the real package loads torch
    # for a class this adapter never instantiates
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = type("SentenceTransformer", (), {})          # never constructed
    stubs["sentence_transformers"] = st
    try:
        importlib.import_module("sklearn.metrics.pairwise")
    except Exception:                                                     # noqa: BLE001
        sk = types.ModuleType("sklearn"); sk.__path__ = []                # type: ignore[attr-defined]
        skm = types.ModuleType("sklearn.metrics"); skm.__path__ = []      # type: ignore[attr-defined]
        skp = types.ModuleType("sklearn.metrics.pairwise")
        skp.cosine_similarity = _np_cosine_similarity
        stubs.update({"sklearn": sk, "sklearn.metrics": skm, "sklearn.metrics.pairwise": skp})
    missing = object()
    saved = {k: sys.modules.get(k, missing) for k in stubs}
    sys.modules.update(stubs)
    try:
        prov = importlib.import_module("eval.memory.vendor.memp_evolvelab.providers.memp_memory_provider")
        types_mod = importlib.import_module("eval.memory.vendor.memp_evolvelab.memory_types")
    finally:
        for k, old in saved.items():
            if old is missing:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old
    return prov, types_mod


_P, _T = _import_vendored()
MempMemoryProvider = _P.MempMemoryProvider
MemoryRequest, MemoryStatus, TrajectoryData = _T.MemoryRequest, _T.MemoryStatus, _T.TrajectoryData


def _task_key(text: str) -> str:
    """Task stem used by `adjust_on_repeat`: drop the speaker, the offered
    option list and all digits, so the same request recurring with a different
    option set or traffic figure counts as a repeat."""
    t = text.strip()
    m = re.match(r"^([A-Z][\w .'\-]{0,40}):\s", t)
    if m:
        t = t[m.end():]
    t = re.split(r"\bThe only options open to you\b", t, maxsplit=1)[0]
    t = re.sub(r"\d+", "", t).lower()
    return re.sub(r"\s+", " ", t).strip()


# Every string upstream substitutes for a reply it could not use: `_call_llm`
# returns "(LLM returned empty response)" for an empty completion and
# "(LLM call error: ...)" / "(LLM call failed: ...)" for a failed one, and
# `_generate_script_for_trace` wraps those in "(Script generation failed)" /
# "(Script generation error)".  Upstream stores whichever it got as the record's
# procedural_script; see "When a distillation call comes back empty" above.
_PLACEHOLDER = re.compile(
    r"^\(\s*(?:LLM (?:returned empty response|call error|call failed)"
    r"|Script generation (?:failed|error)"
    r"|No (?:trace available to generate script|script available))", re.I)


def _clean_script(text: str) -> str:
    """The stored script, with a stray `<script>`/`</script>` removed: on a reply
    the backbone cut off before the closing tag, upstream's `_extract_tag` finds no
    complete block and returns the raw reply, opening tag included."""
    return re.sub(r"</?script>", "", text or "").strip()


def _unusable(text: str) -> bool:
    """True for a distillation that carries no content: empty, or one of upstream's
    placeholder strings, which must never reach the store or the agent."""
    return not text or bool(_PLACEHOLDER.match(text))


class _Provider(MempMemoryProvider):
    """The upstream provider with four seams redirected: LLM (via the upstream
    `model` config hook), embeddings, initialisation and JSON persistence; plus
    the unknown-outcome summary prompt and a top_k override."""

    def __init__(self, backend: "Backend"):
        self._b = backend
        super().__init__(config={"model": backend._model, "store_path": backend.store_path,
                                 "records_file": "procedural_records.json"})

    # -- seams ---------------------------------------------------------------
    def initialize(self) -> bool:
        # upstream: load_embedding_model() (Hugging Face download) + read JSON store.
        self.embedding_model = None
        self.embedding_dim = None
        self.memories = []
        self.embeddings_cache = np.empty((0, 0), dtype=np.float32)
        self._last_provided_cache = {}
        return True

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        return np.asarray(self._b.llm.embed(list(texts)), dtype=np.float32)

    def _save_memories_to_json(self):
        if not self._b.store_json:
            return
        os.makedirs(self.store_path, exist_ok=True)
        emb = [] if self.embeddings_cache is None or self.embeddings_cache.size == 0 else self.embeddings_cache.tolist()
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump({"memories": self.memories, "embeddings": emb}, f, indent=4, ensure_ascii=False)

    def _search(self, qvec, top_k):
        # upstream provide_memory() hard-codes top_k = 1; honour cfg["top_k"] when larger
        return super()._search(qvec, max(int(top_k), self._b.top_k))

    # -- nothing usable came back ---------------------------------------------
    def _one_line(self, x) -> str:
        return " ".join(str(x or "").split())[:self._b.stub_chars] or "(unrecorded)"

    def _stub_script(self, trajectory_data) -> str:
        """What is stored as the script when the distillation call produced nothing.
        Deterministic, bounded, built only from the task text (already the embedded
        retrieval key) and the action taken -- never upstream's placeholder string,
        which the agent would later be served as a procedure to follow."""
        return ("(no script was distilled for this task: the distillation call returned no text)\n"
                f"- Task: {self._one_line(trajectory_data.query)}\n"
                f"- Action taken: {self._one_line(trajectory_data.result)}")

    def _stub_summary(self, trajectory_data) -> str:
        """What is stored as the concrete-steps summary when the summarisation call
        produced nothing.  Upstream falls back to the full reconstructed trajectory,
        which here is the rolling session buffer -- thousands of characters of raw
        transcript; with recall() clipping from the tail, one such record would be
        served to the agent as a tail-slice of transcript labelled as an example."""
        return (f"Step 0: Task: {self._one_line(trajectory_data.query)}\n"
                f"Step 1: Acted: {self._one_line(trajectory_data.result)}\n"
                "Conclusion: no summary was produced for this trajectory; the grounds for the "
                "action were not recorded.")

    def _generate_script_for_trace(self, trajectory_data) -> str:
        """Upstream's prompt and extraction, with its placeholder returns caught:
        anything `_unusable` is replaced by `_stub_script` and counted."""
        script = _clean_script(super()._generate_script_for_trace(trajectory_data))
        if _unusable(script):
            log.warning("MEMP script distillation returned no usable text; storing a stub script")
            self._b.n_script_stubs += 1
            return self._stub_script(trajectory_data)
        return script

    # -- unknown outcome -------------------------------------------------------
    def _summarize_trajectory_with_llm(self, trajectory_data) -> str:
        """Upstream prompt verbatim except the Correctness line (upstream would
        print 'Wrong' whenever is_correct is absent) and the closing-sentence
        instruction, which no longer asks for a success/failure reason.  The
        no-text fallback is `_stub_summary`, not upstream's raw trajectory."""
        current_trajectory = self._reconstruct_trajectory_string(trajectory_data)
        if self.model is None:
            return self._stub_summary(trajectory_data)
        system_prompt = (
f"""Generate an ultra-concise summary of the agent's actions.

Question: {trajectory_data.query}
Final Result: {trajectory_data.result}
Correctness: Unknown (this benchmark gives no feedback during the episode)
Trajectory: {current_trajectory}

IMPORTANT: Provide a "bare bones" step-by-step summary. Focus *only* on the single most important action, tool call, or observation for each step. Omit all filler words.

Format:
Step 0: [Key action/thought (e.g., "Initial plan")]
Step 1: [Key action/tool (e.g., "Called search(X)")]
Step 2: [Key observation/result (e.g., "Found Y")]
...

Rules:
* Use actual step indices (0, 1, 2, ...).
* **Each step description MUST be a single short phrase or sentence (max 10 words).**
* After the steps, add a **single sentence conclusion** stating what was decided and on what basis (the outcome's correctness is unknown).
* Keep the **total length under 150 words**.
"""
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": system_prompt}]}]
        try:
            resp = self.model(messages)
            summary = getattr(resp, "content", str(resp)).strip()
        except Exception as e:                                            # noqa: BLE001
            log.warning("MEMP summarisation failed: %s", e)
            summary = ""
        if _unusable(summary):        # API failure, or a reasoning backbone that spent
            self._b.n_summary_stubs += 1              # max_tokens before emitting text
            return self._stub_summary(trajectory_data)
        return summary


class Backend(MemoryBackend):
    name = "memp"
    persist_across_episodes = True
    uses_llm = True
    recalls = True

    def __init__(self, llm: LLM, cfg: Optional[Dict] = None):
        super().__init__(llm, cfg)
        self.top_k = max(1, int(self.cfg.get("top_k", 1)))
        self.buffer_tokens = int(self.cfg.get("buffer_tokens", 3000))
        self.max_tokens = int(self.cfg.get("max_tokens", 1000))
        self.adjust_on_repeat = bool(self.cfg.get("adjust_on_repeat", False))
        self.stub_chars = int(self.cfg.get("stub_chars", 300))
        self.store_json = bool(self.cfg.get("store_json", False))
        self.store_path = self.cfg.get("store_path") or tempfile.mkdtemp(prefix="revoke_memp_")
        self.buffer: List[Tuple[int, str]] = []          # scrolled-out sessions, oldest first
        self._seen: Dict[str, str] = {}                  # task stem -> last action (adjust_on_repeat)
        self.n_added = self.n_adjusted = self.n_adjust_failed = 0
        self.n_llm_calls = self.n_llm_empty = 0
        self.n_script_stubs = self.n_summary_stubs = 0
        self.n_recalls = self.n_recall_hits = 0
        self.mem = _Provider(self)
        self.mem.initialize()

    # -- the upstream `model` hook: every MEMP LLM call lands here ----------------
    def _model(self, messages: List[Dict]) -> str:
        flat = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
            flat.append({"role": m.get("role", "user"), "content": c or ""})
        self.n_llm_calls += 1
        out = self.llm.text(flat, max_tokens=self.max_tokens)
        if not out:
            self.n_llm_empty += 1
        return out

    # -- lifecycle ------------------------------------------------------------------
    def begin_episode(self, item, persist=False):
        super().begin_episode(item, persist)
        self.buffer = []
        self._seen = {}
        if not persist:
            self.mem.initialize()

    def observe(self, session_index, text):
        self.buffer.append((session_index, text))
        while len(self.buffer) > 1 and sum(approx_tokens(t) for _, t in self.buffer) > self.buffer_tokens:
            self.buffer.pop(0)

    def recall(self, task_text, budget_tokens):
        self.n_recalls += 1
        ctx = "\n".join(t for _, t in self.buffer)
        resp = self.mem.provide_memory(MemoryRequest(query=task_text, context=ctx, status=MemoryStatus.BEGIN))
        if not resp.memories:
            return ""
        self.n_recall_hits += 1
        body = "\n\n".join(str(m.content) for m in resp.memories)
        return self.clip("Procedural memory retrieved from your earlier tasks in this conversation "
                         "(a distilled script and a concrete example each):\n" + body, budget_tokens)

    def record_action(self, probe_id, task_text, action, rationale):
        tool = (self.item or {}).get("act_tool", "act")
        param = (self.item or {}).get("act_param", "option")
        steps = [{"type": "observation", "content": t} for _, t in self.buffer]
        rat = (rationale or "").strip()
        if rat and rat != action:
            steps.append({"type": "thought", "content": rat})
        steps.append({"type": "action", "content": f"Called {tool}({param}={action})"})
        meta = {"probe_id": probe_id, "session_indices": [i for i, _ in self.buffer]}

        key = _task_key(task_text)
        prev = self._seen.get(key)
        self._seen[key] = action
        served = self.mem._last_provided_cache.get(task_text.strip())
        if self.adjust_on_repeat and prev is not None and prev != action and served:
            meta["is_correct"] = "false"     # ADAPTATION: proxy signal, see module docstring
        traj = TrajectoryData(query=task_text, trajectory=steps, result=f"{tool}({param}={action})", metadata=meta)

        n_before = len(self.mem.memories)
        rev_before = self._revisions()
        self.mem.take_in_memory(traj)
        self.n_added += len(self.mem.memories) - n_before
        if meta.get("is_correct") == "false":
            got = self._revisions() - rev_before
            self.n_adjusted += got
            self.n_adjust_failed += max(0, len(served) - got)

    def _revisions(self) -> int:
        return sum(int((r.get("meta") or {}).get("revision_count", 0)) for r in self.mem.memories)

    def end_episode(self):
        self.buffer = []

    def stats(self):
        return {"memories": len(self.mem.memories), "added": self.n_added, "adjusted": self.n_adjusted,
                "adjust_failed": self.n_adjust_failed, "llm_calls": self.n_llm_calls, "llm_empty": self.n_llm_empty,
                "script_stubs": self.n_script_stubs,       # records whose script is a stub, not a distillation
                "summary_stubs": self.n_summary_stubs,     # ... and whose concrete-steps summary is
                "recalls": self.n_recalls, "recall_hits": self.n_recall_hits}
