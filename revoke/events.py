"""
The six state-evolution events of REVOKE, as operators on the persistent layer.

    ADD         insert a new defeasible rule
    SUPERSEDE   revise-and-replace an existing rule's head/body
    CONDITION   rewrite a rule body (unconditional <-> conditional)
    CONFLICT    insert a rule with a contradictory head; resolved by priority
    SUPPORT     insert nothing -- raise the priority of an existing rule
    RETRACT     withdraw part of a rule's ground instances

Only SUPERSEDE and RETRACT are destructive; everything else is append-only,
which is what makes staleness attribution well defined.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .logic import Lit, Rule, RuleBase, lit

KINDS = ("ADD", "SUPERSEDE", "CONDITION", "CONFLICT", "SUPPORT", "RETRACT", "NOTE")


@dataclass
class Event:
    eid: str
    kind: str
    session: int = 0
    # payload -----------------------------------------------------------
    rid: str = ""                                  # rule created / targeted
    head: Optional[Lit] = None
    body: Tuple[Lit, ...] = ()
    prio: int = 1
    delta: int = 0                                 # SUPPORT priority bump
    instances: Tuple[Tuple[str, ...], ...] = ()    # RETRACT head arg tuples
    # bookkeeping -------------------------------------------------------
    speaker: str = "system"
    text: str = ""                                 # natural-language rendering
    tags: Tuple[str, ...] = ()
    motif: str = ""

    def apply(self, rb: RuleBase) -> None:
        k = self.kind
        if k in ("ADD", "CONFLICT"):
            rb.add(Rule(self.rid, self.head, self.body, self.prio,
                        origin=self.eid, session=self.session))
        elif k == "SUPERSEDE":
            old = rb.rules[self.rid]
            rb.rules[self.rid] = replace(
                old, head=self.head or old.head,
                body=self.body if self.head or self.body else old.body,
                prio=self.prio or old.prio, origin=self.eid, alive=True,
                retracted=frozenset())
        elif k == "CONDITION":
            old = rb.rules[self.rid]
            rb.rules[self.rid] = replace(old, body=self.body, origin=self.eid)
        elif k == "SUPPORT":
            old = rb.rules[self.rid]
            rb.rules[self.rid] = replace(old, prio=old.prio + self.delta,
                                         origin=self.eid)
        elif k == "RETRACT":
            old = rb.rules[self.rid]
            keep = frozenset(old.retracted) | frozenset(self.instances)
            rb.rules[self.rid] = replace(old, retracted=keep, origin=self.eid)
        elif k == "NOTE":
            pass                                   # numeric state / role change: no rule
        else:
            raise ValueError(f"unknown event kind {k!r}")

    def to_json(self) -> Dict:
        return {
            "eid": self.eid, "kind": self.kind, "session": self.session,
            "rid": self.rid,
            "head": str(self.head) if self.head else None,
            "body": [str(b) for b in self.body],
            "prio": self.prio, "delta": self.delta,
            "instances": [list(i) for i in self.instances],
            "speaker": self.speaker, "text": self.text,
            "tags": list(self.tags), "motif": self.motif,
        }


def event_from_json(d: Dict) -> Event:
    return Event(
        eid=d["eid"], kind=d["kind"], session=d["session"], rid=d["rid"],
        head=lit(d["head"]) if d.get("head") else None,
        body=tuple(lit(b) for b in d.get("body", ())),
        prio=d.get("prio", 1), delta=d.get("delta", 0),
        instances=tuple(tuple(i) for i in d.get("instances", ())),
        speaker=d.get("speaker", ""), text=d.get("text", ""),
        tags=tuple(d.get("tags", ())), motif=d.get("motif", ""))


def replay(base: RuleBase, events: Sequence[Event], upto: int) -> RuleBase:
    """Persistent layer after applying every event with ``session <= upto``."""
    rb = base.copy()
    for e in events:
        if e.session <= upto:
            e.apply(rb)
    return rb


def timeline(base: RuleBase, events: Sequence[Event]) -> List[Tuple[int, RuleBase]]:
    """One rule base per session at which at least one event fires."""
    out, rb = [], base.copy()
    for s in sorted({e.session for e in events}):
        for e in events:
            if e.session == s:
                e.apply(rb)
        out.append((s, rb.copy()))
    return out
