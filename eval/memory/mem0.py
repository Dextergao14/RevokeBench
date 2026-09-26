"""
mem0 -- declarative memory middleware -- as a REVOKE memory backend.

Method.  Chhikara, Khant, Aryan, Singh & Yadav, "Mem0: Building Production-Ready AI Agents
with Scalable Long-Term Memory" (arXiv:2504.19413, 2025).  Source: the official repository
github.com/mem0ai/mem0 (PyPI package ``mem0ai``).  This adapter imports and wraps the installed
release 2.1.0 -- ``mem0/memory/main.py`` (Memory.add / search / reset, _create_memory /
_update_memory / _delete_memory), ``mem0/configs/prompts.py``, ``mem0/llms/base.py``,
``mem0/embeddings/base.py``, ``mem0/vector_stores/qdrant.py``, ``mem0/utils/factory.py``,
``mem0/memory/{storage,telemetry,notices}.py``.  It was written against upstream commit
a39a802 (2026-09-18).  What runs is the installed package, and requirements-mem0.txt records the
sha256 of each file above so a reader can tell whether the library moved under the adapter.

What mem0 is.  A store of short natural-language "memories" (facts) in a vector database,
scoped by user_id.  Writing is LLM-driven: a conversation turn goes to the backbone with
mem0's extraction prompt and the facts that come back are embedded and stored.  Reading is
vector search over the facts.  Two write pipelines exist in the library's history; both are
available here (cfg ``pipeline``):

  * ``additive`` (default) -- what ``pip install mem0ai==2.1.0`` runs, imported unchanged.
    Memory.add(infer=True) retrieves the 10 nearest existing memories, makes ONE LLM call with
    ADDITIVE_EXTRACTION_PROMPT + generate_additive_extraction_prompt (existing memories, the
    last 10 turns saved for this user, the new turn), parses {"memory": [{"text", ...}]},
    embeds the texts, drops exact (MD5) duplicates, inserts them and records history.  It is
    ADD-only: it never rewrites or deletes an existing memory (the LLM's "linked_memory_ids"
    are accepted but do not change what is stored or retrieved).  Hybrid BM25 and entity
    boosts are optional extras (fastembed, spaCy) that a default install lacks, so retrieval
    is cosine search over the facts -- exactly what a default ``pip install mem0ai`` does.
  * ``reconcile`` -- the two-phase pipeline of the Mem0 paper: extract facts, then let the
    LLM decide ADD / UPDATE / DELETE / NONE for each against the nearest existing memories.
    mem0 removed it from Memory.add in release 2.0.0; ``_Mem0._reconcile_add`` below is a
    port of the synchronous ``_add_to_vector_store`` (infer branch) of mem0ai 1.0.11, the last
    release that shipped it, running on the prompts and helpers 2.1.0 still ships
    (USER_MEMORY_EXTRACTION_PROMPT via get_fact_retrieval_messages, DEFAULT_UPDATE_MEMORY_PROMPT
    via get_update_memory_messages, ensure_json_instruction, normalize_facts) and on 2.1.0's
    own _create_memory / _update_memory / _delete_memory.  Two LLM calls per session, or one
    when extraction returns no facts (``if new_retrieved_facts`` guards the second call).
    Dropped from the port: the graph-store branch (off in the paper's OSS default) and the
    NONE-branch refresh of agent_id/run_id (this adapter scopes by user_id only).  The vector
    store's ``limit=`` keyword became ``top_k=`` between the two releases; nothing else changed.
    CAVEAT: 1.0.11 is neither installed nor vendored here, so the port cannot be diffed against
    its source in this checkout; the deviations listed above are the ones the author recorded
    while porting, and the empty-extraction short-circuit is the one visible in the port itself.
    See mem0.md and requirements-mem0.txt.

Cost (this is what decides whether a pipeline is affordable at the long tier).  ``additive`` is
about 2.7x more expensive per session than ``reconcile`` despite making half the calls, because
ADDITIVE_EXTRACTION_PROMPT is a 33,653-char (~8.4k tok) static system prompt re-sent on EVERY
observe.  Measured on data/long100 episode 1 (433 sessions, ~3.3k chars/session), one observe:

  additive   1 call,  40,193 chars ~= 10.0k tok   (system 33,653 | last-k 3,121 | new session
                                                   3,174 | existing memories ~130)
  reconcile  2 calls, 14,928 chars ~=  3.7k tok   (extract 7,344 | update 7,584)

i.e. ~4.4M vs ~1.6M input tok of memory-write cost per episode, ~218M vs ~81M for a 50-episode
condition.  Note what does NOT drive this: ``generate_additive_extraction_prompt`` re-sends the
last ``last_k_messages`` rows of mem0's messages table, and observe() hands mem0 a whole session
block as one message, so "last 10 messages" does mean "the last 10 sessions" -- but mem0
truncates each past message to PAST_MESSAGE_TRUNCATION_LIMIT = 300 chars, so that section costs
~3.1k chars, not ten transcripts.  cfg ``last_k_messages`` bounds it anyway (last_k=2 saves
~2.5k chars/session, ~6%); the static prompt is the cost and no knob bounds it.  additive's
per-session cost is near-flat in store size, reconcile's second call grows with it.  Method and
per-pipeline totals: mem0.md.

Configuration deltas from library defaults are listed under "Configuration" below and justified
in mem0.md; the two that change results are ``top_k`` (12 here, 20 upstream) and
``action_task_chars`` (0 here -- see record_action).

Contract mapping.
  begin_episode(item, persist)  user_id := item["domain"] (the world).  persist=False ->
                                 Memory.reset(): fresh collection, fresh history and
                                 last-messages tables.  persist=True -> nothing; memories of
                                 the world's earlier episodes stay and keep being reconciled.
  observe(session_index, text)  Memory.add(text, user_id, metadata={"session": i, "kind":
                                 "session"}) with inference on -> the pipeline above.
  recall(task_text, budget)     Memory.search(task_text, filters={"user_id"}, top_k=cfg.top_k,
                                 threshold=cfg.threshold) -> one line per memory: "[session N]"
                                 from its metadata, "[you]" for the agent's own actions,
                                 "[earlier session]" when the runner could not parse a session
                                 index (it passes -1 for a post-probe continuation block), in
                                 mem0's score order (cfg.order="session" for chronological),
                                 as many lines as fit the budget, then self.clip.
  record_action(...)            Memory.add([{"role": "assistant", "name": "you", "content":
                                 "you did: chose <action> -- <rationale>"}],
                                 metadata={"kind": "action", "probe_id"}, infer=False): mem0's
                                 raw-memory path, stored verbatim and tagged role=assistant /
                                 actor_id=you, no LLM call.  cfg.action_infer=True sends it
                                 through the extraction pipeline instead.  The probe's task text
                                 is NOT spliced in by default (cfg.action_task_chars=0): REVOKE
                                 probes within a world are near-identical in wording, so an action
                                 memory carrying the task text is by construction the nearest
                                 neighbour of every later probe and crowds the extracted session
                                 rules out of top_k (measured in development: 47% of recalled lines
                                 were "[you]" at 240 chars, 24% at 0).  The task text is verbatim
                                 in the prompt at the moment of recall, so nothing is lost.
  end_episode()                 nothing: mem0 has no between-episode consolidation step.
  stats()                       counters, the memory system's own LLM-call tally, and the
                                 live point count of the collection.

Unknown outcome.  mem0 never branches on task success or failure; record_action carries
no correctness signal and none is inferred.  Nothing in either pipeline changes.

Fairness and isolation.  mem0's LLM interface (LLMBase.generate_response) and embedder
interface (EmbeddingBase.embed / embed_batch) are implemented by thin classes that delegate
to the shared REVOKE ``LLM.text`` and ``LLM.embed``, so every extraction / reconciliation
call runs on the backbone under test and is counted, and no embedding leaves the process.
They are injected by a Memory subclass whose __init__ assigns the components instead of
calling mem0's provider factories (the pydantic LlmConfig / EmbedderConfig validators
whitelist provider names, so a factory registration cannot carry a live object).  The vector
store is mem0's own Qdrant wrapper on ``QdrantClient(":memory:")``; history and last-messages
live in SQLite ``:memory:``.  Telemetry (PostHog) is forced off before import, which also
disables mem0's remote "notices" fetch; spaCy's auto-download and fastembed's model download
are pre-empted (see _offline and the bm25 flag).  The backend sees only the blind item and the
texts the runner hands it; it reads no dataset file.  The only filesystem write is mem0's own
$MEM0_DIR/config.json (a generated installation id), written by setup_config() at import time
and redirected to the system temp dir below.

Configuration (self.cfg) and defaults:
  pipeline="additive"            "additive" | "reconcile" (see above)
  top_k=12                       memories retrieved per task.  Memory.search's own default is 20;
                                 12 matches the other retrieval backends (MemOS also uses 12) so
                                 they are compared at equal retrieval depth.  12 short facts render
                                 to ~1-2k chars, well under a 4,000-token recall budget, so mem0 is
                                 retrieval-bound, not budget-bound: sweep 12/20/40 before a long run.
  threshold=0.1                  mem0's search default: minimum cosine score of a hit.  The threshold
                                 gates the raw cosine before hybrid combination; with the hashed
                                 fallback embedder 96.5% of candidates on long100 score >= 0.1, so
                                 mem0's default excludes almost nothing and needs no adjustment.
  max_tokens=2000                mem0's BaseLlmConfig default for its own completions
  last_k_messages=10             upstream's limit on the message rows the additive prompt re-sends
                                 (one row = one whole session here, but mem0 truncates each to 300
                                 chars, so lowering it saves only ~6% -- see Cost above)
  order="score"                  "score" (mem0's ranking) | "session" (chronological).  mem0's own
                                 created_at is the wall-clock time of the write, not the in-world
                                 session time, so chronology is recoverable only from the session
                                 index in the metadata -- which is what order="session" sorts on.
  action_infer=False             route record_action through the extraction pipeline
  action_task_chars=0            how much of the task text an action memory keeps (0 = none; 240
                                 reproduces the earlier behaviour, kept for the ablation)
  max_recall_errors=3            consecutive Memory.search failures tolerated before recall raises
                                 rather than silently returning "" for the rest of the episode
  custom_instructions=None       additive pipeline: mem0's custom_instructions
  custom_fact_extraction_prompt=None, custom_update_memory_prompt=None   reconcile pipeline
  bm25=False                     enable mem0's fastembed BM25 hybrid (needs the Qdrant/bm25
                                 model already cached; it is downloaded otherwise)
  collection="revoke"            Qdrant collection name
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import warnings
from copy import deepcopy
from typing import Dict, List, Optional

# mem0 reads these at import time: telemetry (PostHog, plus a remote notice-config fetch)
# must be off, and its home directory must not be the user's ~/.mem0 -- importing
# mem0.memory.main runs setup_config(), which mkdir -p's MEM0_DIR and writes a config.json
# holding a generated installation id.  Set unconditionally: an inherited MEM0_DIR would put
# that write back in the operator's home, and nothing here needs the operator's copy.
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["MEM0_DIR"] = os.path.join(tempfile.gettempdir(), "revoke_mem0")
warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")

from qdrant_client import QdrantClient                                            # noqa: E402

import mem0.memory.telemetry as _telemetry                                        # noqa: E402
import mem0.utils.spacy_models as _spacy                                          # noqa: E402
from mem0.configs.base import MemoryConfig                                        # noqa: E402
from mem0.configs.embeddings.base import BaseEmbedderConfig                       # noqa: E402
from mem0.configs.llms.base import BaseLlmConfig                                  # noqa: E402
from mem0.configs.prompts import get_update_memory_messages                       # noqa: E402
from mem0.embeddings.base import EmbeddingBase                                    # noqa: E402
from mem0.llms.base import LLMBase                                                # noqa: E402
from mem0.memory.main import Memory                                               # noqa: E402
from mem0.memory.storage import SQLiteManager                                     # noqa: E402
from mem0.memory.utils import (ensure_json_instruction, extract_json,             # noqa: E402
                               get_fact_retrieval_messages, normalize_facts,
                               parse_messages, remove_code_blocks)
from mem0.vector_stores.qdrant import Qdrant                                      # noqa: E402

from eval.memory.base import TOK, MemoryBackend                                   # noqa: E402

_telemetry.MEM0_TELEMETRY = False        # also covers a process that imported mem0 before us
logging.getLogger("mem0").setLevel(logging.ERROR)     # "Resetting all memories" etc. per episode


def _offline() -> None:
    """mem0 downloads spaCy's en_core_web_sm the first time entity extraction or BM25
    lemmatisation runs if spaCy is installed without the model.  REVOKE allows no network at
    run time, so unless the model is already present mark the loaders as failed -- mem0's own
    degraded path (no entity boosts, unlemmatised text), which is also what a default install
    without spaCy does."""
    try:
        import spacy                                                              # type: ignore
        have_model = bool(spacy.util.is_package("en_core_web_sm"))
    except Exception:                                                             # noqa: BLE001
        have_model = False
    if not have_model:
        _spacy._load_failed_full = True
        _spacy._load_failed_lemma = True


def _json_ok(text: str) -> bool:
    """Would mem0's parser (remove_code_blocks, then extract_json) accept this reply?"""
    cleaned = remove_code_blocks(text)
    for cand in (cleaned, extract_json(cleaned)):
        try:
            json.loads(cand, strict=False)
            return True
        except Exception:                                                         # noqa: BLE001
            pass
    return False



def _wrap_bare_list(text: str) -> str:
    """Deviation (recorded): some backbones answer mem0's additive extraction prompt with the
    bare list of memories instead of the {"memory": [...]} object the prompt asks for.
    Upstream's parser rejects the bare list outright.  Wrapping it keeps the memories the
    model did produce; nothing else about the reply is altered.  Counted as parsed, so it
    does not hide in `unparsable`."""
    import json as _json
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
        t = t.strip()
    if t.startswith("["):
        try:
            v = _json.loads(t)
        except _json.JSONDecodeError:
            return text
        if isinstance(v, list):
            return _json.dumps({"memory": v})
    return text

class RevokeLLM(LLMBase):
    """mem0's LLM interface over the REVOKE backbone under test.  Every call mem0 makes for
    itself goes through LLM.text, so it runs on the same model as the agent and is counted
    as a memory call.  response_format is advisory: mem0's prompts already demand JSON and
    its parser tolerates prose around it; we only count replies it could not parse."""

    def __init__(self, llm, max_tokens: int):
        super().__init__(BaseLlmConfig(model=llm.model, temperature=llm.temperature, max_tokens=max_tokens))
        self._llm = llm
        self.calls = 0
        self.empty = 0
        self.unparsable = 0

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        self.calls += 1
        out = self._llm.text(messages, max_tokens=self.config.max_tokens)
        if out and response_format:
            out = _wrap_bare_list(out)
        if not out:
            self.empty += 1
        elif response_format and not _json_ok(out):
            self.unparsable += 1
        if tools:                                   # Memory.add never passes tools; interface parity only
            return {"content": out, "tool_calls": []}
        return out


class RevokeEmbedder(EmbeddingBase):
    """mem0's embedder interface over the shared local embedder (LLM.embed)."""

    def __init__(self, llm, dims: int):
        super().__init__(BaseEmbedderConfig(model="revoke-local", embedding_dims=dims))
        self._llm = llm

    def embed(self, text, memory_action=None):
        return self._llm.embed([text])[0]

    def embed_batch(self, texts, memory_action="add"):
        texts = list(texts)
        return self._llm.embed(texts) if texts else []


class _Mem0(Memory):
    """mem0's Memory with its LLM, embedder and vector store injected.  __init__ assigns the
    components instead of building them through the provider factories and skips telemetry;
    add / search / reset / history are the library's code.  _add_to_vector_store switches the
    inference branch to the 1.0.11 reconcile pipeline when asked; infer=False is untouched."""

    pipeline = "additive"
    custom_fact_extraction_prompt: Optional[str] = None       # 1.0.11 MemoryConfig fields that
    custom_update_memory_prompt: Optional[str] = None         # 2.1.0's MemoryConfig no longer has

    def __init__(self, config: MemoryConfig, llm: RevokeLLM, embedder: RevokeEmbedder, store: Qdrant,
                 last_k_messages: int = 10):
        self.config = config
        self.embedding_model = embedder
        self.vector_store = store
        self.llm = llm
        self.last_k_messages = int(last_k_messages)
        self.db = SQLiteManager(config.history_db_path)
        self._bind_last_k()
        self.collection_name = config.vector_store.config.collection_name
        self.api_version = config.version
        self.custom_instructions = config.custom_instructions
        self.reranker = None
        self._entity_store = None

    def _bind_last_k(self) -> None:
        """The additive prompt re-sends ``self.db.get_last_messages(scope, limit=10)``, and one row
        is one whole session here (observe hands mem0 a session block as a single message).  Cap it
        by wrapping the history manager's method instead of patching mem0's source; the upstream
        default (10) is a no-op wrapper.  Worth ~6% of the write cost, not more: mem0 truncates
        each past message to 300 chars, and the 33.6k-char static prompt is the real cost (see the
        module docstring).  Memory.reset() builds a fresh SQLiteManager, so this re-applies there."""
        raw = self.db.get_last_messages
        if getattr(raw, "_revoke_bounded", False):
            return
        cap = self.last_k_messages

        def bounded(session_scope, limit=10, _raw=raw, _cap=cap):
            return _raw(session_scope, limit=min(int(limit), _cap))

        bounded._revoke_bounded = True
        self.db.get_last_messages = bounded

    def reset(self):
        super().reset()
        self._bind_last_k()

    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        if infer and self.pipeline == "reconcile":
            return self._reconcile_add(messages, metadata, filters)
        return super()._add_to_vector_store(messages, metadata, filters, infer, prompt)

    # -- mem0ai 1.0.11  Memory._add_to_vector_store, infer=True branch (ported) ------------------
    def _reconcile_add(self, messages, metadata, filters):
        parsed_messages = parse_messages(messages)
        if self.custom_fact_extraction_prompt:
            system_prompt = self.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            is_agent_memory = self._should_use_agent_memory_extraction(messages, metadata)
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)
        system_prompt, user_prompt = ensure_json_instruction(system_prompt, user_prompt)

        response = self.llm.generate_response(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"})
        try:
            cleaned = remove_code_blocks(response)
            if not cleaned.strip():
                new_retrieved_facts = []
            else:
                try:
                    new_retrieved_facts = json.loads(cleaned, strict=False)["facts"]
                except json.JSONDecodeError:
                    new_retrieved_facts = json.loads(extract_json(response), strict=False)["facts"]
                new_retrieved_facts = normalize_facts(new_retrieved_facts)
        except Exception:                                                         # noqa: BLE001
            new_retrieved_facts = []

        retrieved_old_memory: List[Dict] = []
        new_message_embeddings: Dict[str, List[float]] = {}
        search_filters = {k: filters[k] for k in ("user_id", "agent_id", "run_id") if filters.get(k)}
        for new_mem in new_retrieved_facts:
            emb = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = emb
            for mem in self.vector_store.search(query=new_mem, vectors=emb, top_k=5, filters=search_filters):
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})
        unique: Dict = {}
        for item in retrieved_old_memory:
            unique[item["id"]] = item
        retrieved_old_memory = list(unique.values())
        temp_uuid_mapping: Dict[str, str] = {}           # integer ids: guards against UUID hallucination
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            item["id"] = str(idx)

        new_memories_with_actions: Dict = {}
        if new_retrieved_facts:
            prompt = get_update_memory_messages(retrieved_old_memory, new_retrieved_facts,
                                                self.custom_update_memory_prompt)
            response = self.llm.generate_response(messages=[{"role": "user", "content": prompt}],
                                                  response_format={"type": "json_object"})
            try:
                if response and response.strip():
                    try:
                        new_memories_with_actions = json.loads(remove_code_blocks(response), strict=False)
                    except json.JSONDecodeError:
                        new_memories_with_actions = json.loads(extract_json(response), strict=False)
            except Exception:                                                     # noqa: BLE001
                new_memories_with_actions = {}
        if not isinstance(new_memories_with_actions, dict):
            new_memories_with_actions = {}

        returned = []
        for resp in new_memories_with_actions.get("memory", []) or []:
            try:
                action_text = resp.get("text")
                if not action_text:
                    continue
                event = resp.get("event")
                if event == "ADD":
                    if action_text not in new_message_embeddings:
                        new_message_embeddings[action_text] = self.embedding_model.embed(action_text, "add")
                    mid = self._create_memory(data=action_text, existing_embeddings=new_message_embeddings,
                                              metadata=deepcopy(metadata))
                    returned.append({"id": mid, "memory": action_text, "event": event})
                elif event == "UPDATE":
                    if action_text not in new_message_embeddings:
                        new_message_embeddings[action_text] = self.embedding_model.embed(action_text, "update")
                    mid = temp_uuid_mapping[resp.get("id")]
                    self._update_memory(memory_id=mid, data=action_text, existing_embeddings=new_message_embeddings,
                                        metadata=deepcopy(metadata))
                    returned.append({"id": mid, "memory": action_text, "event": event,
                                     "previous_memory": resp.get("old_memory")})
                elif event == "DELETE":
                    mid = temp_uuid_mapping[resp.get("id")]
                    self._delete_memory(memory_id=mid)
                    returned.append({"id": mid, "memory": action_text, "event": event})
                # "NONE": no change.  (1.0.11 also refreshed agent_id/run_id here; user_id scope only.)
            except Exception:                                                     # noqa: BLE001
                continue                                   # 1.0.11 logs and skips a malformed action
        return returned


class Backend(MemoryBackend):
    name = "mem0"
    persist_across_episodes = True
    uses_llm = True
    recalls = True

    HEADER = ("Memories mem0 retrieved for this task, most relevant first ([session N] = the session the "
              "memory was extracted from; [you] = an action you took earlier):\n")

    def __init__(self, llm, cfg=None):
        super().__init__(llm, cfg)
        c = self.cfg
        self.pipeline = str(c.get("pipeline", "additive"))
        if self.pipeline not in ("additive", "reconcile"):
            raise ValueError(f"mem0 cfg.pipeline must be 'additive' or 'reconcile', got {self.pipeline!r}")
        self.top_k = int(c.get("top_k", 12))          # mem0's Memory.search default is 20; see docstring
        self.threshold = float(c.get("threshold", 0.1))
        # mem0 validates these inside Memory.search and raises there, where recall's except would
        # turn a cfg typo into an empty recall for every probe of the episode.  Fail at startup.
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"mem0 cfg.threshold must be in [0.0, 1.0], got {self.threshold!r}")
        if self.top_k < 1:
            raise ValueError(f"mem0 cfg.top_k must be >= 1, got {self.top_k!r}")
        self.order = str(c.get("order", "score"))
        self.action_infer = bool(c.get("action_infer", False))
        self.action_task_chars = int(c.get("action_task_chars", 0))
        self.max_recall_errors = int(c.get("max_recall_errors", 3))
        self.last_k_messages = max(1, int(c.get("last_k_messages", 10)))
        _offline()

        dims = len(llm.embed(["dimension probe"])[0])
        collection = str(c.get("collection", "revoke"))
        self._client = QdrantClient(":memory:")
        store = Qdrant(collection_name=collection, embedding_model_dims=dims, client=self._client)
        if not c.get("bm25"):
            # mem0's own "tried, unavailable" sentinel: cosine-only search, and fastembed is never
            # imported (it would download the Qdrant/bm25 model on first use).
            store._bm25_encoder = False
        config = MemoryConfig(
            vector_store={"provider": "qdrant", "config": {
                "collection_name": collection, "embedding_model_dims": dims,
                "client": self._client, "path": os.environ["MEM0_DIR"]}},
            history_db_path=":memory:",
            custom_instructions=c.get("custom_instructions"))
        self.mem_llm = RevokeLLM(llm, int(c.get("max_tokens", 2000)))
        self.mem = _Mem0(config, self.mem_llm, RevokeEmbedder(llm, dims), store, self.last_k_messages)
        self.mem.pipeline = self.pipeline
        self.mem.custom_fact_extraction_prompt = c.get("custom_fact_extraction_prompt")
        self.mem.custom_update_memory_prompt = c.get("custom_update_memory_prompt")
        self.user_id = "world"
        self.n = self._zero()
        self.last_error = ""
        self.recall_error_streak = 0

    @staticmethod
    def _zero() -> Dict[str, int]:
        return {"observed": 0, "memories": 0, "updates": 0, "deletes": 0, "extraction_empty": 0,
                "actions": 0, "recalls": 0, "hits": 0, "errors": 0}

    # -- lifecycle ------------------------------------------------------------------------
    def begin_episode(self, item, persist=False):
        super().begin_episode(item, persist)
        world = str(item.get("domain") or item.get("id") or "world")
        self.user_id = re.sub(r"\s+", "_", world).strip() or "world"     # mem0 rejects whitespace in ids
        if not persist:
            self.mem.reset()
            self.n = self._zero()
            self.mem_llm.calls = self.mem_llm.empty = self.mem_llm.unparsable = 0
            self.last_error = ""
            self.recall_error_streak = 0

    def _add(self, messages, metadata: Dict, infer: bool) -> List[Dict]:
        try:
            out = self.mem.add(messages, user_id=self.user_id, metadata=metadata, infer=infer)
            return out.get("results") or []
        except Exception as e:                                                    # noqa: BLE001
            self.n["errors"] += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            return []

    def observe(self, session_index, text):
        self.n["observed"] += 1
        results = self._add(text, {"session": session_index, "kind": "session"}, infer=True)
        if not results:
            self.n["extraction_empty"] += 1
        for r in results:
            ev = r.get("event")
            if ev == "ADD":
                self.n["memories"] += 1
            elif ev == "UPDATE":
                self.n["updates"] += 1
            elif ev == "DELETE":
                self.n["deletes"] += 1

    # -- read ------------------------------------------------------------------------------
    @staticmethod
    def _session(row: Dict):
        """The session a memory came from, or None when the runner could not name one.  It passes
        session_index=-1 for a block it cannot parse an index out of (its post-probe
        '[Session N continued]' blocks), and -1 is not a position in the episode: treat it as
        unknown so order="session" sorts it with the other unknowns instead of before session 0."""
        s = (row.get("metadata") or {}).get("session")
        if not isinstance(s, int) or isinstance(s, bool) or s < 0:
            return None
        return s

    @classmethod
    def _tag(cls, row: Dict) -> str:
        meta = row.get("metadata") or {}
        if meta.get("kind") == "action" or row.get("role") == "assistant":
            return "[you]"
        s = cls._session(row)
        return f"[session {s}]" if s is not None else "[earlier session]"

    def recall(self, task_text, budget_tokens):
        self.n["recalls"] += 1
        query = " ".join(task_text.split())
        if not query:
            return ""
        try:
            rows = self.mem.search(query, filters={"user_id": self.user_id}, top_k=self.top_k,
                                   threshold=self.threshold).get("results") or []
        except Exception as e:                                                    # noqa: BLE001
            self.n["errors"] += 1
            self.recall_error_streak += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            # A transient store hiccup degrades one probe; a systematically broken store (a bad
            # cfg mem0 rejects, a dead collection) would otherwise produce a complete, plausible,
            # entirely memory-free episode.  Fail the episode instead of grading that.
            if self.recall_error_streak >= self.max_recall_errors:
                raise RuntimeError(f"mem0 recall failed {self.recall_error_streak} times in a row; "
                                   f"last error: {self.last_error}") from e
            return ""
        self.recall_error_streak = 0
        if not rows:
            return ""
        self.n["hits"] += len(rows)
        if self.order == "session":
            rows.sort(key=lambda r: (self._session(r) is None, self._session(r) or 0))
        lim = budget_tokens * TOK
        lines: List[str] = []
        used = len(self.HEADER)
        for r in rows:
            line = f"- {self._tag(r)} " + " ".join(str(r.get("memory") or "").split())
            if lines and used + len(line) + 1 > lim:
                break
            lines.append(line)
            used += len(line) + 1
        return self.clip(self.HEADER + "\n".join(lines), budget_tokens)

    # -- the agent's own actions -------------------------------------------------------------
    def record_action(self, probe_id, task_text, action, rationale):
        self.n["actions"] += 1
        # Default action_task_chars=0: the probe text is deliberately NOT stored.  REVOKE probes
        # within a world are near-identical in wording, so an action memory that carries the task
        # text is the nearest neighbour of every later probe and evicts the extracted session rules
        # from top_k -- an adapter artifact that would bias the backbone toward repeating its own
        # earlier (possibly since-revoked) choice.  240 stays available for that ablation.
        asked = " ".join(task_text.split())[: self.action_task_chars] if self.action_task_chars else ""
        why = " ".join((rationale or "").split())[:200]
        text = f"you did: chose {action}" + (f" -- {why}" if why else "") + (f" (when asked: {asked})" if asked else "")
        self._add([{"role": "assistant", "name": "you", "content": text}],
                  {"kind": "action", "probe_id": probe_id}, infer=self.action_infer)

    def end_episode(self):
        """mem0 has no between-episode consolidation."""

    def stats(self):
        d: Dict = dict(self.n)
        d["pipeline"] = self.pipeline
        d["llm_calls"] = self.mem_llm.calls
        d["llm_empty"] = self.mem_llm.empty
        d["llm_unparsable"] = self.mem_llm.unparsable
        d["last_k_messages"] = self.mem.last_k_messages
        # A backbone outage looks exactly like "the model extracted nothing" in `memories` and
        # `extraction_empty`.  This is the flag that separates them: an episode with degraded=True
        # (or any llm_empty) must be discarded and re-run, not graded.
        d["degraded"] = bool(self.mem_llm.calls) and \
            (self.mem_llm.empty + self.mem_llm.unparsable) / self.mem_llm.calls > 0.2
        try:
            d["points"] = int(self._client.count(collection_name=self.mem.collection_name, exact=True).count)
        except Exception:                                                         # noqa: BLE001
            d["points"] = -1
        if self.last_error:
            d["last_error"] = self.last_error
        return d
