"""Rule-tracking baseline: the paper's two-layer representation used as a memory.

The backbone does one thing -- it reads each session that leaves the raw
window and writes down the *rulings* in it as structured statements (who said
it, about which entity, what kind of change, under what condition).  Everything
else is done by the engine in revoke.logic: statements are compiled into a
persistent rule base with integer priorities, and at every task the closure is
recomputed and read off for the offered options.  The backbone never reasons
about "what is in force now"; the solver does.

What the backbone may see is exactly what every other memory system sees: the
blind item (transcript text, tool surface, option sets, the display names of
the entities in play).  Nothing here imports the grader-side reconstruction or
reads events, labels or closures; the engine module is the only shared code.

Priority.  prio = RANK_STEP * rank + seq (RANK_STEP = 10**6), where rank is the
speaker's rank at the session the ruling was made (resolved from the authority
notice and later role changes) and seq is a global counter.  Rank dominates
recency; within a rank the newer ruling wins.  A speaker may change only rulings
and facts made at or below their own rank; a change to a higher-ranked ruling is
dropped and counted.  Advisory objections are the rank-0 tier (prio = seq) and
never defeat a ranked ruling until escalated.  Standing constructions that the
text itself declares to override individual rulings -- a certification regime,
an alias linkage -- sit one level above any speaker.  Reversals are never
mutations: a reversal adds the complementary rule at the reverser's rank and the
engine settles it by priority.  Facts (a condition holding, an item being active,
group membership, the regime flag) are single rule ids rewritten in place, so on
and off never conflict, and they carry the rank of whoever set them.

Every failure fails safe: a malformed reply, an unresolvable name, an unranked
or outranked speaker or a dangling reference stores nothing and is counted in
stats().
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from revoke.logic import Lit, Rule, RuleBase, check_assertion, solve

from .base import LLM, MemoryBackend

_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_SESS = re.compile(r"\s*\[Session\s+(\d+)")
RANK_STEP = 10 ** 6          # seq never approaches this in an episode, so rank and recency never mix
FACT_PRIO = RANK_STEP * 10   # facts: above every rule; on/off rewrite one id, so they never conflict
TASK_FACT_PRIO = RANK_STEP * 20

TYPES = ("permit", "forbid", "advise", "narrow", "widen", "lift", "reverse", "reaffirm",
         "group_forbid", "member", "link", "incompatible", "active", "condition",
         "threshold", "threshold_rule", "list", "role")

SYS = """You read one block of a long record -- rounds, minutes, a chat channel, a review -- and write down every RULING in it as structured data. A ruling is a statement that decides something about what may be used, given, deployed or suggested: an approval, a prohibition, a condition on a prohibition, a withdrawal or reversal of an earlier decision, a restatement at a higher level, a rule about a group, a linkage between two things, an incompatibility, a numeric threshold, a certified or approved list, or a change in who holds which role.

Report the speaker exactly as written at the start of the line. Do NOT judge whether the speaker has authority; that is decided elsewhere. DO judge whether the sentence has the form of a decision: questions, proposals, rumours, praise, and someone recalling what they believe the rule is are NOT rulings and must not be reported. A line that reads like a recorded decision IS reported even if you doubt the speaker.

EVERY statement is a JSON object with "speaker" (REQUIRED: the name before the colon at the start of the line, copied exactly; a statement without it is discarded), "type", and the fields of its type. Example: {"speaker": "Rosa Ekholm", "type": "forbid", "entity": "Bastion Gate", "condition": "the quarterly release freeze is on"}. A "condition" may be one name or a list of names that must all hold at once.

Statement types and their fields:
  permit         entity                         X is approved, is the standing choice, is cleared, or is back in use
  forbid         entity, condition?             X is not to be used, optionally only while <condition> holds
  advise         entity, condition?             an objection to X that explicitly stops short of a prohibition and leaves the existing approval standing
  narrow         entity, condition              a blanket prohibition on X now applies only while <condition>
  widen          entity                         a conditional prohibition on X now applies at all times
  lift           entity?, refers_to_session?    a prohibition on X is withdrawn with no position either way; or a speaker withdraws their own earlier decision by its session number
  reverse        entity?, refers_to_session?    the decision in force on X is turned around (a ban becomes a permission, or a permission a ban); or a speaker reverses their own earlier decision by its session number
  reaffirm       entity, polarity               X's existing permission or prohibition is restated at a higher level without changing its wording, or an earlier advisory objection to X is made binding; polarity is "permit" or "forbid"
  group_forbid   group, condition?, members?    nothing in <group> may be used, optionally only while <condition>; list the members if the line names them
  member         entity, group, in              X joins (in=true) or leaves / is carved out of (in=false) <group>
  link           entity, follows, on            X is governed as part of Y, so decisions on Y apply to X (on=true); that linkage is broken (on=false)
  incompatible   entity, with                   X and Y may not both be in use
  active         entity, on                     Y is now in use, placed, live, on the chart or on the screen (on=true), or no longer (on=false); report this even though it approves or forbids nothing
  condition      name, on                       a named state of the world now holds (on=true) or has lapsed (on=false)
  threshold      name, value                    a numeric threshold is set or moved to <value>
  threshold_rule entity, name, side             X may not be used above (side="above") or below (side="below") <threshold name>
  list           op, members?, entity?, condition?   op in set / add / remove / regime_on / regime_off: a certified or approved list is declared (set, with its members), amended (add / remove one entity), or the regime under which anything off the list is barred is switched on or off. If the list is binding only while a named condition holds (a gate, a window, a protocol), give that condition on the set statement and report the gate's changes as condition statements
  role           person, rank                   a person holds or moves to a rank: 3 = sets policy and outranks everyone; 2 = outranks rank 1; 1 = may decide but is outranked; 0 = no authority, discussion only. Report EVERY person named in an authority notice with their rank, and every later change of role; the "speaker" of a role statement is whoever wrote the notice or announced the change.

Use the entity's name exactly as it appears in the text. Use the same condition name each time the same condition is meant; a list of condition names already in use is given. Output ONLY a JSON object of the form {"statements": [ ... ]} with no prose. If the block contains no ruling, output {"statements": []}."""

_FALSE = {"false", "no", "off", "0", "f", "n", "none", ""}
_TRUE = {"true", "yes", "on", "1", "t", "y"}
_FORBID_WORDS = ("forbid", "prohib", "ban", "hold", "block", "neg", "not", "no", "against", "deny", "bar")


def _norm(s) -> str:
    return _ARTICLE.sub("", " ".join(str(s).lower().split()))


def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _norm(s)).strip("_")[:40] or "x"


def _speaker_key(s) -> str:
    return _norm(str(s or "").strip().rstrip(":"))


def _flag(v, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        return False if t in _FALSE else (True if t in _TRUE else default)
    return bool(v)


def _int(v) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    m = _NUM.search(str(v).lower().replace("session", ""))
    try:
        return int(float(m.group(0))) if m else None
    except (ValueError, OverflowError):
        return None


def _members(d: Dict) -> List[str]:
    m = d.get("members")
    if isinstance(m, str):
        m = re.split(r"[,;]| and ", m)
    if not isinstance(m, list):
        return []
    return [x for x in m if isinstance(x, str) and x.strip()]


@dataclass
class Stmt:
    """One accepted statement, kept for the recall justification and the diagnostic log."""
    session: int
    speaker: str
    rank: int
    type: str
    entity: str = ""
    entity2: str = ""
    condition: str = ""
    rid: str = ""
    quote: str = ""
    extra: Dict = field(default_factory=dict)


class Backend(MemoryBackend):
    name = "ruletrack"
    persist_across_episodes = False
    uses_llm = True
    recalls = True

    # ------------------------------------------------------------------ setup
    def __init__(self, llm: LLM, cfg: Optional[Dict] = None):
        super().__init__(llm, cfg)
        self.log_dir = str(self.cfg.get("log_dir") or "runs/ruletrack_logs")
        self.regime_rank = int(self.cfg.get("regime_rank", 4))      # standing constructions sit above any speaker
        self.entity_hint = _flag(self.cfg.get("entity_hint"), True)  # list the entities in play in the extraction prompt
        self._reset()

    def _reset(self) -> None:
        self.rb = RuleBase(signature={"ok": ("e",), "in_group": ("e", "g"), "active": ("e",),
                                      "unlisted": ("e",), "regime": ()},
                           universe={"e": [], "g": []})
        self.seq = 0
        self.stmts: List[Stmt] = []
        self.roles: List[Tuple[int, str, int]] = []        # (session, speaker key, rank)
        self.role_names: Dict[str, str] = {}
        self.conds: Dict[str, str] = {}                     # slug -> display name
        self.groups: Dict[str, str] = {}
        self.listed: set = set()
        self.list_declared = False
        self.list_rank = 0
        self.regime_rid = ""
        self.regime_gate: Tuple[str, ...] = ()              # condition slugs gating a condition-bound list
        self.threshold: Dict[str, Tuple[float, int]] = {}   # slug -> (value, rank)
        self.thr_names: Dict[str, str] = {}
        self.fact_rank: Dict[str, int] = {}
        self.names: Dict[str, str] = {}                     # eid -> display
        self.by_norm: Dict[str, str] = {}
        self.probe_by_text: Dict[str, Dict] = {}
        self.dropped: List[Dict] = []
        self.recalls: List[Dict] = []
        self.last_session = 0
        self.st = {"sessions_observed": 0, "statements": 0, "bad_json": 0, "malformed_stmt": 0,
                   "unknown_type": 0, "unresolved_entity": 0, "dropped_rank0": 0, "unknown_speaker": 0,
                   "outranked": 0, "dangling_ref": 0, "bad_role": 0, "apply_error": 0,
                   "speaker_inferred": 0, "nonconverged": 0, "recalls": 0}
        self._block_lines: List[Tuple[str, str]] = []       # (speaker prefix, line) of the block being observed

    def begin_episode(self, item: Dict, persist: bool = False) -> None:
        super().begin_episode(item, persist)
        self._reset()
        self.names = {str(k): str(v) for k, v in (item.get("entity_names") or {}).items()}
        self.by_norm = {_norm(v): k for k, v in self.names.items()}
        self.rb.universe["e"] = sorted(self.names)
        probes = {p.get("probe_id"): p for p in item.get("probes", []) if isinstance(p, dict)}
        for s in item.get("sessions", []):
            for t in s.get("turns", []):
                if t.get("probe_id") in probes:
                    self.probe_by_text[t["text"]] = probes[t["probe_id"]]

    # ------------------------------------------------------------- resolution
    def _entity(self, s) -> str:
        if s is None or isinstance(s, bool) or not isinstance(s, (str, int, float)) or not str(s).strip():
            return ""
        s = str(s)
        low = _norm(s)
        if low in self.by_norm:
            return self.by_norm[low]
        if s in self.names:
            return s
        hits = {eid for n, eid in self.by_norm.items() if n in low or (low in n and len(low) >= 0.6 * len(n))}
        return hits.pop() if len(hits) == 1 else ""

    def _rank(self, person, session: int) -> Optional[int]:
        key = _speaker_key(person)
        if not key:
            return None
        best: Optional[Tuple[int, int]] = None                 # (session, order) of the governing role entry
        rank = None
        for i, (s, who, rk) in enumerate(self.roles):
            if who == key and s <= session and (best is None or (s, i) >= best):
                best, rank = (s, i), rk
        return rank

    def _cond(self, name) -> str:
        slug = "cond_" + _slug(name)
        if slug not in self.conds:
            # coreference: "the design review gate is on" and "design review gate" name one
            # condition; map a new name onto an existing one that contains it or that it contains
            n = _norm(name)
            for old, disp in self.conds.items():
                o = _norm(disp)
                if len(o) >= 8 and len(n) >= 8 and (o in n or n in o):
                    return old
            self.conds[slug] = str(name)
            self.rb.signature[slug] = ()
        return slug

    def _group(self, name) -> str:
        slug = "g_" + _slug(name)
        if slug not in self.groups:
            self.groups[slug] = str(name)
            self.rb.universe["g"].append(slug)
        return slug

    def _prio(self, rank: int) -> int:
        self.seq += 1
        return RANK_STEP * rank + self.seq

    def _rid(self, kind: str) -> str:
        return f"{kind}_{self.seq}"

    def _rules_about(self, eid: str, polarity: Optional[bool] = None) -> List[Rule]:
        out = []
        for r in self.rb.rules.values():
            if not r.alive or r.head.pred != "ok" or r.head.args != (eid,) or r.rid.startswith("link_"):
                continue
            if (eid,) in r.retracted:
                continue
            if polarity is not None and r.head.neg == polarity:      # polarity True means a permission
                continue
            out.append(r)
        return out

    def _target(self, eid: str, rank: int, polarity: Optional[bool] = None,
                want_body: Optional[bool] = None) -> Tuple[Optional[Rule], bool]:
        """The rule in force about `eid` (highest priority) that the speaker may change.
        Returns (rule, outranked): outranked is True when a target exists only above the speaker's rank."""
        cands = self._rules_about(eid, polarity)
        if want_body is not None:
            pref = [r for r in cands if bool(r.body) == want_body]
            cands = pref or cands
        if not cands:
            return None, False
        within = [r for r in cands if r.prio // RANK_STEP <= rank]
        if not within:
            return None, True
        return max(within, key=lambda r: r.prio), False

    def _referential(self, speaker: str, ref: Optional[int], eid: str, rank: int) -> Tuple[Optional[Rule], bool]:
        """A speaker's own rule written at session `ref`; exactly one must match."""
        if ref is None:
            return None, False
        key = _speaker_key(speaker)
        cands = [r for r in self.rb.rules.values() if r.alive and r.session == ref and r.head.pred == "ok"
                 and not r.rid.startswith("link_") and r.origin == key]
        if eid:
            cands = [r for r in cands if r.head.args == (eid,)]
        if len(cands) != 1:
            return None, False
        if cands[0].prio // RANK_STEP > rank:
            return None, True
        return cands[0], False

    # ---------------------------------------------------------------- observe
    def observe(self, session_index: int, text: str) -> None:
        self.st["sessions_observed"] += 1
        if session_index is None or session_index < 0:
            m = _SESS.match(text or "")
            session_index = int(m.group(1)) if m else self.last_session
        self.last_session = max(self.last_session, session_index)
        known_conds = ", ".join(self.conds.values()) or "(none yet)"
        roles_now = {}
        for s, who, rk in self.roles:
            if s <= session_index:
                roles_now[self.role_names.get(who, who)] = rk
        role_txt = "; ".join(f"{p}: rank {r}" for p, r in sorted(roles_now.items())) or "(none recorded yet)"
        hint = f"Entities in play -- {', '.join(sorted(self.names.values()))}\n" if self.entity_hint else ""
        user = (f"Roles recorded so far -- {role_txt}\n{hint}"
                f"Condition names already in use -- {known_conds}\n\n"
                f"BLOCK (session {session_index}):\n{text}")
        # extraction must come back as JSON, not as a thought stream: reasoning models
        # otherwise spend the whole completion budget thinking and return nothing
        self._block_lines = []
        for line in (text or "").split("\n"):
            m = re.match(r"\s*([A-Z][^:\n]{1,60}?):\s+(.*)$", line)
            if m and not m.group(1).startswith("[Session"):
                self._block_lines.append((m.group(1).strip(), m.group(2)))
        reply = self.llm.text([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                              max_tokens=1800, reasoning={"enabled": False})
        stmts = self._parse(reply)
        if stmts is None:
            self.st["bad_json"] += 1
            return
        # a person's FIRST role entry (the authority notice) is applied before the block's rulings,
        # so a ruling and its speaker's rank may arrive in one block in any order; later role
        # CHANGES take effect from their position in the block
        known = {w for _, w, _ in self.roles}
        for d in stmts:
            if d.get("type") == "role" and _speaker_key(d.get("person")) not in known:
                self._guard(self._apply_role, d, session_index)
        for d in stmts:
            if d.get("type") == "role":
                if not d.get("_applied"):
                    self._guard(self._apply_role, d, session_index)
            else:
                self._guard(self._apply, d, session_index)

    def _guard(self, fn, d: Dict, session: int) -> None:
        try:
            fn(d, session)
        except Exception as e:                                    # noqa: BLE001 -- never lose an episode to one statement
            self.st["apply_error"] += 1
            self.dropped.append({**d, "session": session, "why": f"apply_error:{type(e).__name__}"})

    def _parse(self, reply) -> Optional[List[Dict]]:
        if not reply or not isinstance(reply, str):
            return None
        s = _FENCE.sub("", reply.strip())
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j < 0:
            return None
        try:
            d = json.loads(s[i:j + 1])
        except (json.JSONDecodeError, RecursionError):
            return None
        out = d.get("statements") if isinstance(d, dict) else d
        if not isinstance(out, list):
            return None
        good = [x for x in out if isinstance(x, dict)]
        self.st["malformed_stmt"] += len(out) - len(good)
        return good

    def _apply_role(self, d: Dict, session: int) -> None:
        d["_applied"] = True
        who = str(d.get("person") or "").strip().rstrip(":").strip()
        rk = _int(d.get("rank"))
        if not who or rk is None or not 0 <= rk <= 3:
            self.st["bad_role"] += 1
            self.dropped.append({**d, "session": session, "why": "bad_role"})
            return
        key = _speaker_key(who)
        # once an authority notice exists, only a policy-setter may change roles
        if self.roles and session > self.roles[0][0]:
            spk_rank = self._rank(d.get("speaker"), session)
            if spk_rank != 3:
                self.st["outranked"] += 1
                self.dropped.append({**d, "session": session, "why": "role_change_not_by_rank3"})
                return
        self.roles.append((session, key, rk))
        self.role_names[key] = who
        self.stmts.append(Stmt(session, who, rk, "role", extra={"rank": rk}, quote=str(d.get("quote") or "")[:200]))
        self.st["statements"] += 1

    def _infer_speaker(self, d: Dict) -> str:
        """A statement with no speaker: the one ranked person whose line in this block names
        the statement's entity (or group / condition / list keyword).  Deterministic, from the
        blind text only; ambiguity means no inference."""
        keys = [str(d.get(k) or "") for k in ("entity", "follows", "with", "group", "name", "person")]
        keys = [k for k in keys if len(k) >= 3]
        if not keys and str(d.get("type")) == "list":
            keys = ["list", "certif", "approved"]
        cands = set()
        for who, line in self._block_lines:
            low = line.lower()
            if any(_norm(k) in _norm(low) for k in keys) and self._rank(who, self.last_session) is not None:
                cands.add(who)
        return cands.pop() if len(cands) == 1 else ""

    def _drop(self, why: str, d: Dict, session: int) -> None:
        self.st[why] += 1
        self.dropped.append({**d, "session": session, "why": why})

    def _apply(self, d: Dict, session: int) -> None:
        t = str(d.get("type") or "")
        if t not in TYPES:
            return self._drop("unknown_type", d, session)
        speaker = str(d.get("speaker") or "").strip()
        if not speaker:
            speaker = self._infer_speaker(d)
            if speaker:
                self.st["speaker_inferred"] += 1
                d["speaker"] = speaker
        rank = self._rank(speaker, session)
        if rank is None:
            return self._drop("unknown_speaker", d, session)
        if rank == 0:
            return self._drop("dropped_rank0", d, session)
        quote = str(d.get("quote") or "")[:200]
        raw_c = d.get("condition")
        conds = [str(c).strip() for c in (raw_c if isinstance(raw_c, list) else [raw_c])
                 if isinstance(c, (str, int, float)) and str(c).strip()]
        cond = " AND ".join(conds)
        body: Tuple[Lit, ...] = tuple(Lit(self._cond(c)) for c in conds)
        e = self._entity(d.get("entity")) if d.get("entity") is not None else ""
        needs_entity = t in ("permit", "forbid", "advise", "narrow", "widen", "reaffirm", "member", "link",
                             "incompatible", "active", "threshold_rule")
        if needs_entity and not e:
            return self._drop("unresolved_entity", d, session)
        st = Stmt(session, speaker, rank, t, e, condition=cond, quote=quote)
        key = _speaker_key(speaker)

        if t == "permit":
            prio = self._prio(rank)
            st.rid = self._rid("permit")
            self.rb.add(Rule(st.rid, Lit("ok", (e,)), (), prio, origin=key, session=session))
        elif t == "forbid":
            prio = self._prio(rank)
            st.rid = self._rid("forbid")
            self.rb.add(Rule(st.rid, Lit("ok", (e,), neg=True), body, prio, origin=key, session=session))
        elif t == "advise":
            self.seq += 1                                          # rank-0 tier: below every ranked ruling
            st.rid = self._rid("advise")
            self.rb.add(Rule(st.rid, Lit("ok", (e,), neg=True), body, self.seq, origin=key, session=session))
        elif t == "narrow":
            if not cond:
                return self._drop("dangling_ref", d, session)
            tgt, out = self._target(e, rank, polarity=False, want_body=False)
            if tgt is None:
                return self._drop("outranked" if out else "dangling_ref", d, session)
            self.seq += 1
            tgt.body = body
            st.rid = tgt.rid
        elif t == "widen":
            tgt, out = self._target(e, rank, polarity=False, want_body=True)
            if tgt is None or not tgt.body:
                return self._drop("outranked" if out else "dangling_ref", d, session)
            self.seq += 1
            tgt.body = ()
            st.rid = tgt.rid
        elif t == "reaffirm":
            pol_s = str(d.get("polarity") or "permit").lower()
            pol = not any(pol_s.startswith(k) for k in _FORBID_WORDS)
            cands = self._rules_about(e, polarity=pol)
            if not cands:
                return self._drop("dangling_ref", d, session)
            tgt = max(cands, key=lambda r: r.prio)
            prio = self._prio(rank)
            tgt.prio = max(tgt.prio, prio)                         # raised to the reaffirming rank; never lowered
            st.rid, st.extra = tgt.rid, {"polarity": "permit" if pol else "forbid"}
        elif t in ("lift", "reverse"):
            ref = _int(d.get("refers_to_session"))
            tgt, out = (None, False)
            if e:
                tgt, out = self._target(e, rank, polarity=(False if t == "lift" else None))
            if tgt is None and not out and ref is not None:
                tgt, out = self._referential(speaker, ref, e, rank)
            if tgt is None:
                return self._drop("outranked" if out else "dangling_ref", d, session)
            self.seq += 1
            st.rid = tgt.rid
            st.entity = e or (tgt.head.args[0] if tgt.head.is_ground and tgt.head.args[0] in self.names else "")
            if t == "lift":
                if tgt.head.is_ground or not st.entity:
                    tgt.alive = False                              # withdraw the ruling whole
                else:
                    tgt.retracted = frozenset(tgt.retracted) | {(st.entity,)}
            else:
                # a reversal is a new ruling at the reverser's rank, never a rewrite
                rid = f"reverse_{self.seq}"
                self.rb.add(Rule(rid, tgt.head.complement(), tgt.body, RANK_STEP * rank + self.seq,
                                 origin=key, session=session))
                st.rid, st.extra = rid, {"reverses": tgt.rid}
        elif t == "group_forbid":
            g = self._group(d.get("group") or "group")
            prio = self._prio(rank)
            st.rid, st.entity2 = self._rid("groupban"), g
            self.rb.add(Rule(st.rid, Lit("ok", ("?X",), neg=True), (Lit("in_group", ("?X", g)),) + body,
                             prio, origin=key, session=session))
            for m in _members(d):
                me = self._entity(m)
                if me:
                    self._fact(f"mem_{me}_{g}", Lit("in_group", (me, g)), session, rank)
                else:
                    self.st["unresolved_entity"] += 1
        elif t == "member":
            g = self._group(d.get("group") or "group")
            st.entity2 = g
            rid = f"mem_{e}_{g}"
            if _flag(d.get("in"), True):
                if not self._fact(rid, Lit("in_group", (e, g)), session, rank):
                    return self._drop("outranked", d, session)
            elif rid in self.rb.rules:
                if self.fact_rank.get(rid, 0) > rank:
                    return self._drop("outranked", d, session)
                self.rb.rules[rid].alive = False
                self.seq += 1
        elif t == "link":
            y = self._entity(d.get("follows"))
            if not y or y == e:
                return self._drop("unresolved_entity", d, session)
            st.entity2 = y
            rp, rn = f"link_{e}_{y}_p", f"link_{e}_{y}_n"
            if _flag(d.get("on"), True):
                prio = self._prio(self.regime_rank)               # the text declares the linkage overriding
                self.rb.add(Rule(rp, Lit("ok", (e,)), (Lit("ok", (y,)),), prio, origin=key, session=session))
                self.rb.add(Rule(rn, Lit("ok", (e,), neg=True), (Lit("ok", (y,), neg=True),), prio,
                                 origin=key, session=session))
                st.rid = rp
            else:
                for rid in (rp, rn):
                    if rid in self.rb.rules:
                        self.rb.rules[rid].alive = False
                self.seq += 1
        elif t == "incompatible":
            y = self._entity(d.get("with"))
            if not y or y == e:
                return self._drop("unresolved_entity", d, session)
            st.entity2 = y
            prio = self._prio(rank)                                # one ground rule per direction, at the speaker's rank
            self.rb.add(Rule(f"pair_{e}_{y}_{self.seq}", Lit("ok", (e,), neg=True), (Lit("active", (y,)),), prio,
                             origin=key, session=session))
            self.rb.add(Rule(f"pair_{y}_{e}_{self.seq}", Lit("ok", (y,), neg=True), (Lit("active", (e,)),), prio,
                             origin=key, session=session))
            st.rid = f"pair_{e}_{y}_{self.seq}"
        elif t == "active":
            if not self._fact(f"active_{e}", Lit("active", (e,), neg=not _flag(d.get("on"), True)), session, rank):
                return self._drop("outranked", d, session)
        elif t == "condition":
            slug = self._cond(d.get("name") or "condition")
            st.condition = self.conds[slug]
            if not self._fact(f"fact_{slug}", Lit(slug, (), neg=not _flag(d.get("on"), True)), session, rank):
                return self._drop("outranked", d, session)
        elif t == "threshold":
            m = _NUM.search(str(d.get("value", "")))
            if not m:
                return self._drop("dangling_ref", d, session)
            slug = "thr_" + _slug(d.get("name") or "threshold")
            if self.threshold.get(slug, (0.0, 0))[1] > rank:
                return self._drop("outranked", d, session)
            self.threshold[slug] = (float(m.group(0)), rank)
            self.thr_names[slug] = str(d.get("name") or "threshold")
            for side in ("above", "below"):
                self.rb.signature.setdefault(f"{slug}__{side}", ())
            st.condition, st.extra = self.thr_names[slug], {"value": float(m.group(0))}
        elif t == "threshold_rule":
            slug = "thr_" + _slug(d.get("name") or "threshold")
            side = "below" if str(d.get("side") or "above").lower().startswith("below") else "above"
            self.thr_names.setdefault(slug, str(d.get("name") or "threshold"))
            for sd in ("above", "below"):
                self.rb.signature.setdefault(f"{slug}__{sd}", ())
            prio = self._prio(rank)
            st.rid, st.extra = self._rid("thr"), {"side": side, "threshold": slug}
            self.rb.add(Rule(st.rid, Lit("ok", (e,), neg=True), (Lit(f"{slug}__{side}"),), prio,
                             origin=key, session=session))
        elif t == "list":
            op = str(d.get("op") or "").lower()
            if self.list_rank > rank and op in ("set", "add", "remove"):
                return self._drop("outranked", d, session)
            if conds and op in ("set", "regime_on", "regime_off", "add", "remove"):
                # the list binds only while a named condition holds: bind that condition as the
                # gate, and (re)build the regime rule with the gate in its body
                self.regime_gate = tuple(self._cond(c) for c in conds)
                self.regime_rid = "regime_rule"
                self.seq += 1
                prio = RANK_STEP * self.regime_rank + self.seq
                self.rb.add(Rule(self.regime_rid, Lit("ok", ("?X",), neg=True),
                                 tuple(Lit(g) for g in self.regime_gate) + (Lit("unlisted", ("?X",)),),
                                 prio, origin=key, session=session))
            if op == "set":
                resolved = [self._entity(m) for m in _members(d)]
                self.st["unresolved_entity"] += resolved.count("")
                resolved = [x for x in resolved if x]
                if not resolved:
                    return self._drop("dangling_ref", d, session)
                self.listed, self.list_declared, self.list_rank = set(resolved), True, rank
            elif op == "add" and e:
                self.listed.add(e)
                self.list_rank = max(self.list_rank, rank)
            elif op == "remove" and e:
                self.listed.discard(e)
                self.list_rank = max(self.list_rank, rank)
            elif op in ("regime_on", "regime_off") and self.regime_gate:
                # a gated regime statement that names its condition is the RULE ("only listed
                # items while the gate is on"), not an assertion that the gate holds now; the
                # gate is switched by condition statements.  Without a named condition it is
                # the switch itself.
                if not conds:
                    for slug in self.regime_gate:
                        if not self._fact(f"fact_{slug}", Lit(slug, (), neg=(op == "regime_off")), session, rank):
                            return self._drop("outranked", d, session)
            elif op in ("regime_on", "regime_off"):
                if not self.regime_rid:
                    self.regime_rid = "regime_rule"
                    prio = self._prio(self.regime_rank)
                    self.rb.add(Rule(self.regime_rid, Lit("ok", ("?X",), neg=True),
                                     (Lit("regime"), Lit("unlisted", ("?X",))), prio, origin=key, session=session))
                if not self._fact("fact_regime", Lit("regime", (), neg=(op == "regime_off")), session, rank):
                    return self._drop("outranked", d, session)
            else:
                return self._drop("dangling_ref", d, session)
            st.extra = {"op": op, "n_listed": len(self.listed)}
            st.rid = self.regime_rid
        self.stmts.append(st)
        self.st["statements"] += 1

    def _fact(self, rid: str, head: Lit, session: int, rank: int) -> bool:
        """A fact is one rule id whose head is rewritten in place, so on/off never conflict.
        A fact set by a higher rank cannot be overwritten by a lower one."""
        if self.fact_rank.get(rid, 0) > rank:
            return False
        self.seq += 1
        prio = FACT_PRIO + self.seq
        if rid in self.rb.rules:
            r = self.rb.rules[rid]
            r.head, r.prio, r.alive, r.session, r.retracted = head, prio, True, session, frozenset()
        else:
            self.rb.add(Rule(rid, head, (), prio, session=session))
        self.fact_rank[rid] = rank
        return True

    # ----------------------------------------------------------------- recall
    def _solve_now(self, probe: Optional[Dict]) -> Tuple[object, List[str]]:
        rb = self.rb.copy()
        notes: List[str] = []
        if self.regime_rid and self.list_declared:
            for eid in self.names:
                if eid not in self.listed:
                    rb.add(Rule(f"unl_{eid}", Lit("unlisted", (eid,)), (), TASK_FACT_PRIO))
        params = probe.get("params") if isinstance(probe, dict) else None
        if self.threshold and isinstance(params, dict):
            nums = {}
            for k, v in params.items():
                m = _NUM.search(str(v))
                if m:
                    nums[str(k)] = float(m.group(0))
            for slug, (lim, _) in self.threshold.items():
                val = next((x for k, x in nums.items() if _slug(k) and _slug(k) in slug), None)
                if val is None and nums:
                    val = list(nums.values())[-1]
                if val is None:
                    continue
                rb.add(Rule(f"fact_{slug}__above", Lit(f"{slug}__above", (), neg=not (val > lim)), (), TASK_FACT_PRIO))
                rb.add(Rule(f"fact_{slug}__below", Lit(f"{slug}__below", (), neg=not (val < lim)), (), TASK_FACT_PRIO))
                notes.append(f"{self.thr_names.get(slug, slug)}: task value {val:g} vs threshold {lim:g}")
        return solve(rb), notes

    def recall(self, task_text: str, budget_tokens: int) -> str:
        self.st["recalls"] += 1
        probe = self.probe_by_text.get(task_text)
        options = [o for o in (probe.get("options") or []) if isinstance(o, str)] if isinstance(probe, dict) else []
        if not options:
            options = [eid for eid, nm in self.names.items() if nm in (task_text or "")]
        sol, notes = self._solve_now(probe)
        rulings = [s for s in self.stmts if s.type != "role"]
        if not sol.converged:
            self.st["nonconverged"] += 1
            head = "Tracked rules could not be resolved to a single consistent state here. Statements on record:"
            body = [f"- s{s.session} {s.speaker} (r{s.rank}) {s.type} {self.names.get(s.entity, s.entity)}"
                    for s in rulings[-40:]]
            return self.clip("\n".join([head] + body), budget_tokens)
        lines = [f"Tracked rules as of session {self.last_session} ({len(rulings)} rulings kept; "
                 f"{self.st['dropped_rank0']} statements ignored as coming from people without authority):"]
        verdicts = {}
        for eid in options:
            nm = self.names.get(eid, eid)
            if check_assertion(sol, [Lit("ok", (eid,))]).violation:
                v = "PROHIBITED"
            elif Lit("ok", (eid,)) in sol.closure:
                v = "PERMITTED"
            else:
                v = "NO RULE"
            verdicts[eid] = v
            hist = [s for s in rulings if s.entity == eid][-4:]
            why = "; ".join(self._describe(s) for s in hist) or "never the subject of a ruling"
            if v == "PROHIBITED" and not hist:
                why = self._indirect_why(eid)
            lines.append(f"- {nm}: {v} -- {why}")
        lines += [f"  ({n})" for n in notes]
        if self.regime_rid:
            on = all(Lit(g) in sol.closure for g in self.regime_gate) if self.regime_gate else Lit("regime") in sol.closure
            lines.append(f"  (certification regime is {'ON' if on else 'off'}; {len(self.listed)} items on the list)")
        if isinstance(probe, dict):
            self.recalls.append({"probe_id": probe.get("probe_id"), "session": probe.get("session"),
                                 "as_of": self.last_session, "verdicts": verdicts,
                                 "n_nodes": len(sol.nodes), "n_deleted": len(sol.deleted)})
        return self.clip("\n".join(lines), budget_tokens)

    def _describe(self, s: Stmt) -> str:
        nm2 = self.names.get(s.entity2, self.groups.get(s.entity2, s.entity2)) if s.entity2 else ""
        what = {"permit": "permitted", "forbid": "prohibited", "advise": "advised against (approval still governs)",
                "narrow": "prohibition narrowed to a condition", "widen": "prohibition made unconditional",
                "lift": "prohibition withdrawn", "reverse": "decision reversed",
                "reaffirm": f"{s.extra.get('polarity', 'permission')} reaffirmed at a higher level",
                "member": f"membership in {nm2} changed", "link": f"follows {nm2}",
                "incompatible": f"incompatible with {nm2}", "active": "activity changed",
                "threshold_rule": f"prohibited {s.extra.get('side', '')} a threshold"}.get(s.type, s.type)
        cond = f' while "{s.condition}"' if s.condition and s.type in ("forbid", "advise", "narrow") else ""
        return f"{what}{cond} s{s.session} ({s.speaker}, r{s.rank})"

    def _indirect_why(self, eid: str) -> str:
        cands = []
        if self.regime_rid and self.list_declared and eid not in self.listed:
            cands.append("not on the certified list while the regime is on")
        for s in self.stmts:
            if s.type == "group_forbid":
                cands.append(f"group rule on {self.groups.get(s.entity2, s.entity2)} s{s.session}")
            if s.type == "link" and s.entity == eid:
                cands.append(f"follows {self.names.get(s.entity2, s.entity2)}")
            if s.type == "incompatible" and eid in (s.entity, s.entity2):
                cands.append("incompatible with an item now in use")
        return "; ".join(dict.fromkeys(cands)) or "derived from standing rules"

    # ------------------------------------------------------------ bookkeeping
    def record_action(self, probe_id: str, task_text: str, action: str, rationale: str) -> None:
        pass

    def end_episode(self) -> None:
        if not self.item:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            iid = str(self.item.get("id", "unknown")).replace("/", "_")
            path = os.path.join(self.log_dir, f"{iid}__{str(self.llm.model).replace('/', '__')}.json")
            payload = json.dumps({"id": iid, "model": self.llm.model, "stats": self.stats(),
                                  "roles": [(s, self.role_names.get(w, w), r) for s, w, r in self.roles],
                                  "statements": [s.__dict__ for s in self.stmts],
                                  "dropped": self.dropped, "recalls": self.recalls}, default=str)
            with open(path, "w") as fh:
                fh.write(payload)
        except (OSError, TypeError, ValueError, AttributeError):
            pass

    def stats(self) -> Dict:
        return dict(self.st, rules_live=sum(1 for r in self.rb.rules.values() if r.alive))
