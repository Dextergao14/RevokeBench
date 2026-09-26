"""
Dynamic Cheatsheet, cumulative variant (DC-Cu), as a REVOKE memory backend.

Method
    Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou.  "Dynamic Cheatsheet: Test-Time
    Learning with Adaptive Memory."  arXiv:2504.07952 (2025).
Source read for this adapter (github suzgunmirac/dynamic-cheatsheet, commit
5cfe3c3 "dynamic cheatsheet v2.0"):
    dynamic_cheatsheet/language_model.py
        LanguageModel.advanced_generate, branch "DynamicCheatsheet_Cumulative"
    dynamic_cheatsheet/utils/extractor.py      extract_cheatsheet (vendored below, with the
                                               closing-tag guard noted under "Truncated curator replies")
    prompts/curator_prompt_for_dc_cumulative.txt   (CURATOR_PROMPT below, adapted; see "Prompt")
    prompts/generator_prompt.txt               (how the cheatsheet is framed for the solver)
    run_benchmark.py                           (the cheatsheet starts as the literal "(empty)")
Also read: EvolveLab/providers/dynamic_cheatsheet_provider.py (MemEvolve).  That
is the retrieval-synthesis flavour (one nearest trajectory -> a 200-word
per-query cheatsheet); it is not DC-Cu and nothing of it is used here.

The original DC-Cu loop, per query k
    answer_k   = LLM(generator_prompt[cheatsheet, question_k])
    cheatsheet = extract_cheatsheet(LLM(curator_prompt[cheatsheet, question_k, answer_k]),
                                    old_cheatsheet=cheatsheet)
One evolving text.  The curator rewrites the WHOLE cheatsheet from (previous
cheatsheet, current input, model answer); an output with no <cheatsheet> block
leaves the cheatsheet unchanged; nothing is ever retrieved or embedded.  Length
is bounded by the prompt's "circa 2000-2500 words" and the curator call's
max_tokens (2 * 2048).  There is no ground-truth signal in the original either:
the curator is asked to "assess the correctness of the provided solution" itself.

Mapping onto the six contract calls
    begin_episode(item, persist)
        persist=False: cheatsheet <- "(empty)", counters reset.
        persist=True : cheatsheet kept -- the cheatsheet IS what DC persists across queries.
    observe(idx, text)
        The session block joins the current input batch.  Once the batch holds
        `sessions_per_update` distinct sessions the curator runs once, exactly as
        the original's step 2:  [[QUESTION]] <- the batched session texts (the
        CURRENT INPUT), [[MODEL_ANSWER]] <- the actions recorded since the last
        update, [[PREVIOUS_CHEATSHEET]] <- the cheatsheet; the extracted
        <cheatsheet> replaces it (unchanged when the block is missing or
        malformed, or the call fails).  This is where the method learns in the rule-evolution setting.
    recall(task, budget)
        The current cheatsheet, framed the way generator_prompt.txt frames it
        ("CHEATSHEET: ... ''' ... '''"), clipped to the budget with self.clip.
        If sessions are pending, the curator runs first (cfg flush_at_recall), so
        the cheatsheet reflects everything that has left the window -- the
        invariant the original keeps by curating after every answer.
    record_action(probe_id, task, action, rationale)
        Appended to the next curator batch as the MODEL ACTIONS (the original's
        MODEL ANSWER TO THE CURRENT INPUT), labelled with the session the task was
        in (from the blind item's probe list) and the act tool call.
    end_episode()
        When persisting, whatever is still pending is curated in one final call
        so the cheatsheet carried into the next episode includes the tail of this
        one.  Without persist the final cheatsheet has no reader; the call is skipped.
    stats()
        curator_calls, curator_no_tag (outputs lacking a <cheatsheet> block, which
        leave the cheatsheet unchanged), curator_failures (empty LLM response),
        curator_malformed (a block opened but never closed, or closed empty -- see
        "Truncated curator replies"), over_cap (cheatsheets hard-clipped to
        max_chars), cheatsheet_chars, episodes.

Truncated curator replies (the one behavioural fix to the vendored extractor)
    The original extract_cheatsheet takes everything after `<cheatsheet>` and
    before `</cheatsheet>`, by splitting on each in turn.  When the curator's
    max_tokens cuts the reply off after the opening tag, there is no closing tag
    to split on, so the split returns the partial text and the whole accumulated
    cheatsheet is replaced by that fragment: one truncated call can wipe the
    sheet.  Here a well-formed closing tag is required before anything is
    replaced -- an opened-but-unclosed block, and a block that closes empty, both
    keep the previous cheatsheet (the extractor's own fallback for a reply with
    no block at all) and are counted as curator_malformed.  Nothing else about
    the extractor changes; a well-formed reply is handled exactly as upstream.

Unknown outcome
    DC never receives labels; the curator judges the model's answer on its own.
    Here that self-judgement stays ("assess the correctness of the model's
    actions ... no verdict on them is available"), and the usage counter's
    "successfully used" became "used to act on a task" because success is
    unknowable.  Consequence: a wrong action can be curated into the cheatsheet
    as a worked example exactly as a wrong solution could in the original.

Length bound
    cfg max_chars (default 12000 chars ~ 2000-2400 words, the original's "circa
    2000-2500 words") is written into the prompt and the curator call's
    max_tokens is max_chars/4 + 1000 (~ the original's 4096).  A well-formed
    cheatsheet that still comes back longer is clipped to max_chars and counted
    in over_cap.  A reply that max_tokens cut off mid-block is not a short
    cheatsheet but a broken one, and is discarded instead (curator_malformed).

Prompt
    CURATOR_PROMPT is prompts/curator_prompt_for_dc_cumulative.txt with these and
    only these changes (the source's typos "REFRENCE", "knolwedge", "actioanble",
    "Refence" are left as they are):
      * "solutions / code snippets / Python / algorithms / problems / questions"
        -> "rules, rulings, decision patterns / strategies / inputs / tasks";
        the two code-specific bullets ("encourage Python usage", the bipartite-
        graph example) and "(e.g., specific Python libraries or solution hacks)"
        are dropped; section 1 of the structure and template is renamed from
        "SOLUTIONS, IMPLEMENTATION PATTERNS, AND CODE SNIPPETS" to
        "RULES, RULINGS, AND DECISION PATTERNS".
      * "assess the correctness of the provided solution" -> "... of the model's
        actions (no verdict on them is available; judge them against the input)".
      * usage counter: "successfully used in problem-solving" -> "used to act on a task".
      * reference tags Q1/Q14 -> S1/S14 (sessions instead of questions).
      * "circa 2000-2500 words" -> computed from max_chars.
      * the CURRENT INPUT and MODEL ANSWER slots each gain one parenthetical line
        saying what they contain (a span of sessions leaving the context; the
        model's task / tool call / reasoning), and the latter header reads
        "MODEL ACTIONS DURING THE CURRENT INPUT".
    Everything else -- structure, responsibilities, principles, memory_item
    format, the "previous content not directly included will be lost" warning,
    the template -- is the original text.

What was re-implemented rather than imported
    LanguageModel (language_model.py) constructs its own provider client
    (UnifiedLLMClient over OpenAI/Anthropic/Gemini SDKs, keys from config.env)
    and needs tiktoken/sklearn/numpy; that violates the fairness rule, so the
    ~30-line DC-Cu branch of advanced_generate is re-implemented here on
    self.llm.text, line for line.  extract_cheatsheet is copied from the source
    with one change, the closing-tag requirement described above.
    Dropped: the generator prompt and its code-execution loop (the REVOKE runner
    owns the acting prompt and the agent has only the act tool), the
    max_num_rounds refinement loop (always 1 in the paper's runs), and the
    "PREVIOUS ANSWERS" appendix that only exists for rounds > 1.

Configuration (self.cfg)
    sessions_per_update  int   2        distinct sessions per curator call (cost knob)
    max_chars            int   12000    cheatsheet length bound, characters
    curator_max_tokens   int   None     curator completion cap; default max_chars//4 + 1000
    flush_at_recall      bool  True     curate pending sessions before answering a task
    rationale_chars      int   600      how much of the agent's stated reasoning enters a batch
    initial_cheatsheet   str   "(empty)"  the original's --initialize_cheatsheet_path
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from eval.memory.base import TOK, MemoryBackend

_SESSION_HDR = re.compile(r"^\[Session (\d+)")

# prompts/curator_prompt_for_dc_cumulative.txt, adapted as listed in the module docstring.
CURATOR_PROMPT = """# CHEATSHEET REFRENCE CURATOR

#### 1. Purpose and Goals
As the Cheatsheet Curator, you are tasked with creating a continuously evolving reference designed to help act on a wide variety of tasks that arise in one long-running conversation. The cheatsheet's purpose is to consolidate verified rules and rulings, reusable strategies, and critical insights into a single, well-structured resource.

- The cheatsheet should include quick, accurate, reliable, and practical guidance for the range of tasks the conversation poses.
- After seeing each input, you should improve the content of the cheatsheet, synthesizing lessons, insights, tricks, and errors learned from past inputs and adapting to new challenges.

---

#### 2. Core Responsibilities
As the Cheatsheet Curator, you should:
   - Curate and preserve knolwedge: Select and document only the most relevant, most useful, and most actionable rules, rulings, and strategies, while preserving old content of the cheatsheet.
   - Maintain accuracy: Ensure that all entries in the cheatsheet are accurate, clear, and well-contextualized.
   - Refine and update content: Continuously update and improve the content of the cheatsheet by incorporating new insights and rulings, removing repetitions or trivial information, and adding effective strategies.
   - Ensure practicality and comprehensiveness: Provide critical and informative examples, as well as actionable guidelines.

Before updating the cheatsheet, however, you should first assess the correctness of the model's actions (no verdict on them is available; judge them against the input) and strategically incorporate rules, insights, and decision patterns into the new cheatsheet. Always aim to preserve and keep correct, useful, and illustrative rules and strategies for future cheatsheets.

---

#### 3. Principles and Best Practices
1. Accuracy and Relevance:
   - Only include rules, rulings, and strategies that have been stated in the inputs or proven effective.
   - Clearly state any assumptions, limitations, or dependencies.

2. Iterative Refinement:
   - Continuously improve the cheatsheet by synthesizing both old and new inputs, refining explanations, and removing redundancies.
   - Rather than deleting old content and writing new content each time, consider ways to maintain table content and synthesize information from multiple inputs.
   - After each new input, document any reusable rules, strategies, edge cases, or decision techniques.

3. Clarity and Usability:
   - Write concise, actioanble, well-structured entries.
   - Focus on key insights or strategies that make actions correct and effective.

4. Reusability:
   - Provide clear rules, decision procedures, and meta strategies that are easily adaptable to different contexts.
   - Avoid trivial content; focus on non-obvious, critical details and approaches.
   - Make sure to add as many examples as you can in the cheatsheet.
   - Any useful, efficient, generalizable, and illustrative lessons from the previous inputs should be included in the cheatsheet.

---

#### 4. Cheatsheet Structure
The cheatsheet can be divided into the following sections:

1. Rules, Rulings, and Decision Patterns:
   - Document the rules and rulings in force and reusable decision templates.
   - Include descriptions, annotated examples, and potential pitfalls, albeit succinctly.

2. [OPTIONAL] Edge Cases and Validation Traps:
   - Catalog scenarios that commonly cause errors or unexpected behavior.
   - Provide checks, validations, or alternative approaches to handle them.

3. General Meta-Reasoning Strategies:
   - Describe high-level problem-solving frameworks and heuristics.
   - Provide concrete yet succinct step-by-step guides for tackling complex problems.

4. Implement a Usage Counter
   - Each entry must include a usage count: Increase the count every time a strategy is used to act on a task.
   - Use the count to prioritize frequently used entries over rarely applied ones.

---

#### 5. Formatting Guidelines
Use the following structure for each memory item:

```
<memory_item>
<description>
[Briefly describe the context, purpose, and key aspects of the entry.] (Refence: S1, S2, S6, etc.)
</description>
<example>
[Provide a well-documented rule or ruling, worked-out decision, or efficient strategy.]
</example>
</memory_item>
** Count:  [Number of times this strategy has been used to act on a task.]


<memory_item>
[...]
</memory_item>

[...]

<memory_item>
[...]
</memory_item>

```

- Tagging: Use references like `(S14)` or `(S22)` to link entries to their originating sessions.
- Grouping: Organize entries into logical sections and subsections.
- Prioritizing: incorporate effective rules, tricks, and strategies into the cheatsheet.
- Diversity: Have as many useful and relevant memory items as possible to guide the model to tackle future tasks.

N.B. Keep in mind that once the cheatsheet is updated, any previous content not directly included will be lost and cannot be retrieved. Therefore, make sure to explicitly copy any (or all) relevant information from the previous cheatsheet to the new cheatsheet!!!

---

#### 6. Cheatsheet Template
Use the following format for creating and updating the cheatsheet:

NEW CHEATSHEET:
```
<cheatsheet>

Version: [Version Number]

RULES, RULINGS, AND DECISION PATTERNS
<memory_item>
[...]
</memory_item>

<memory_item>
[...]
</memory_item>

GENERAL META-REASONING STRATEGIES
<memory_item>
[...]
</memory_item>

</cheatsheet>
```

N.B. Make sure that all information related to the cheatsheet is wrapped inside the <cheatsheet> block. The cheatsheet can be as long as circa [[WORDS]] words.

-----
-----

## PREVIOUS CHEATSHEET

[[PREVIOUS_CHEATSHEET]]

-----
-----

## CURRENT INPUT

(The sessions below come from one long-running conversation in which the model is embedded, and have just left the model's context. Each line is a message prefixed by its speaker; lines prefixed `[[YOU]]:` are the model's own messages.)

[[QUESTION]]

-----
-----

## MODEL ACTIONS DURING THE CURRENT INPUT

(Each entry is a task the model was given, the tool call it made in response, and its stated reasoning. The session in which an action was taken may still be in the model's context and appear in a later input.)

[[MODEL_ANSWER]]"""

# generator_prompt.txt frames the cheatsheet as  CHEATSHEET:\n'''\n[[CHEATSHEET]]\n'''  and tells the
# solver to "search for and identify any applicable patterns, strategies, or examples within the cheatsheet".
RECALL_HEAD = ("CHEATSHEET: your own continuously curated reference, built from the sessions that have left "
               "your context. Search it for applicable rules, patterns, strategies, or examples before acting.\n'''\n")
RECALL_TAIL = "\n'''"


def extract_cheatsheet_status(response: str, old_cheatsheet: str):
    """dynamic_cheatsheet/utils/extractor.py's extract_cheatsheet, with the
    closing tag required (see "Truncated curator replies" in the module docstring).

    The original is
        txt = response.split("<cheatsheet>")[1].strip()
        txt = txt.split("</cheatsheet>")[0].strip()
    which on a reply the curator's max_tokens cut off after `<cheatsheet>` has no
    `</cheatsheet>` to split on, so it returns the partial text and the whole
    accumulated cheatsheet is replaced by a fragment.  Here an opening tag with
    no closing tag after it, and a well-formed but empty block, both leave the
    cheatsheet as it was -- the extractor's own fallback for a reply with no
    block at all.

    Returns (cheatsheet, status), status in {"ok", "no_tag", "malformed"}.
    """
    response = (response or "").strip()
    # <cheatsheet> (content) </cheatsheet>
    if "<cheatsheet>" not in response:
        return old_cheatsheet, "no_tag"
    tail = response.split("<cheatsheet>", 1)[1]
    if "</cheatsheet>" not in tail:
        return old_cheatsheet, "malformed"                  # truncated before the closing tag
    txt = tail.split("</cheatsheet>", 1)[0].strip()
    if not txt:
        return old_cheatsheet, "malformed"                  # empty block
    return txt, "ok"


def extract_cheatsheet(response: str, old_cheatsheet: str) -> str:
    """The text inside the first well-formed <cheatsheet> block, or the old
    cheatsheet when the reply has none (see extract_cheatsheet_status)."""
    return extract_cheatsheet_status(response, old_cheatsheet)[0]


class Backend(MemoryBackend):
    name = "dynamic_cheatsheet"
    persist_across_episodes = True
    uses_llm = True
    recalls = True

    def __init__(self, llm, cfg=None):
        super().__init__(llm, cfg)
        self.sessions_per_update = max(1, int(self.cfg.get("sessions_per_update", 2)))
        self.max_chars = int(self.cfg.get("max_chars", 12000))
        self.curator_max_tokens = int(self.cfg.get("curator_max_tokens") or (self.max_chars // TOK + 1000))
        self.flush_at_recall = bool(self.cfg.get("flush_at_recall", True))
        self.rationale_chars = int(self.cfg.get("rationale_chars", 600))
        self.initial = str(self.cfg.get("initial_cheatsheet", "(empty)"))
        # ~5-6 characters per word: 12000 chars -> "2000-2400 words", the original's "2000-2500"
        self.template = CURATOR_PROMPT.replace("[[WORDS]]", f"{self.max_chars // 6}-{self.max_chars // 5}")
        self.cheatsheet = self.initial
        self.pending_sessions: List[str] = []
        self.pending_ids: Set[int] = set()
        self.pending_actions: List[str] = []
        self.probe_session: Dict[str, int] = {}
        self.you = "you"
        self.persist = False
        self.n_calls = self.n_no_tag = self.n_fail = self.n_malformed = self.over_cap = self.n_episodes = 0

    # -- lifecycle ------------------------------------------------------------
    def begin_episode(self, item, persist=False):
        super().begin_episode(item, persist)
        self.persist = persist
        if not persist:
            self.cheatsheet = self.initial
            self.n_calls = self.n_no_tag = self.n_fail = self.n_malformed = self.over_cap = self.n_episodes = 0
        self.pending_sessions, self.pending_ids, self.pending_actions = [], set(), []
        # blind fields only: which session each task sits in, so an action can be labelled
        self.probe_session = {p["probe_id"]: p["session"] for p in item.get("probes", [])
                              if "probe_id" in p and "session" in p}
        self.you = item.get("you_prefix", "you")

    def observe(self, session_index, text):
        self.pending_sessions.append(text)
        m = _SESSION_HDR.match(text)
        # a "[Session N continued]" block (the runner splits a session at a task) belongs to session N
        self.pending_ids.add(int(m.group(1)) if m else -len(self.pending_sessions))
        if len(self.pending_ids) >= self.sessions_per_update:
            self._curate()

    def recall(self, task_text, budget_tokens):
        if self.flush_at_recall and self.pending_sessions:
            self._curate()
        body = self.clip(self.cheatsheet, max(50, budget_tokens - (len(RECALL_HEAD) + len(RECALL_TAIL)) // TOK - 1))
        return RECALL_HEAD + body + RECALL_TAIL

    def record_action(self, probe_id, task_text, action, rationale):
        item = self.item or {}
        sess = self.probe_session.get(probe_id)
        entry = (f"[Session {sess}] " if sess is not None else "") + f"Task: {task_text.strip()}\n" \
            f"Model action: {item.get('act_tool', 'act')}({item.get('act_param', 'option')}={action})"
        if (rationale or "").strip():
            entry += f"\nModel's stated reasoning: {rationale.strip()[:self.rationale_chars]}"
        self.pending_actions.append(entry)

    def end_episode(self):
        if self.persist:
            self._curate()
        self.n_episodes += 1

    def stats(self):
        return {"curator_calls": self.n_calls, "curator_no_tag": self.n_no_tag, "curator_failures": self.n_fail,
                "curator_malformed": self.n_malformed, "over_cap": self.over_cap,
                "cheatsheet_chars": len(self.cheatsheet), "episodes": self.n_episodes}

    # -- the curator step (advanced_generate, DynamicCheatsheet_Cumulative, STEP 2) ---------
    def _curate(self):
        if not self.pending_sessions and not self.pending_actions:
            return
        current_input = "\n\n".join(self.pending_sessions) or "(no new sessions since the last update)"
        model_actions = "\n\n".join(self.pending_actions) or "(the model took no actions during this input)"
        prompt = (self.template.replace("[[YOU]]", self.you)
                  .replace("[[QUESTION]]", current_input)
                  .replace("[[MODEL_ANSWER]]", model_actions)
                  .replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet))
        out = self.llm.text([{"role": "user", "content": prompt}], max_tokens=self.curator_max_tokens)
        self.n_calls += 1
        if not out:
            self.n_fail += 1                      # LLM error: cheatsheet unchanged, as in the original
        else:
            new, status = extract_cheatsheet_status(out, self.cheatsheet)
            if status == "no_tag":
                self.n_no_tag += 1                # no block: the extractor keeps the old cheatsheet
            elif status == "malformed":
                self.n_malformed += 1             # truncated or empty block: ditto, rather than replacing
            elif len(new) > self.max_chars:
                new = new[: self.max_chars]
                self.over_cap += 1
            self.cheatsheet = new
        self.pending_sessions, self.pending_ids, self.pending_actions = [], set(), []
