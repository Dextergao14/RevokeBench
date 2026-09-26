"""Common schema shared by every REVOKE seed domain.

A domain contributes vocabulary, entities, natural language and a mock tool
surface.  It contributes *no* logic: all constraint evolution comes from the
domain-agnostic motif library, so every domain is graded by exactly the same
engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Tool:
    name: str
    params: Dict[str, str]                 # param -> sort ("" = free string)
    kind: str                              # "read" | "act"
    asserts: Tuple[str, ...] = ()          # literal templates, {arg} filled
    achieves: Tuple[str, ...] = ()         # goal literals reached on success
    doc: str = ""


@dataclass
class Domain:
    key: str
    title: str
    sort: str                              # primary sort name
    allow: str                             # permission predicate, arity 1
    goal: str                              # 0-ary completion predicate
    entities: Dict[str, str]               # const id -> display name
    contexts: Dict[str, str]               # 0-ary predicate -> display clause
    groups: Dict[str, str]                 # group const -> display name
    tools: List[Tool]
    persona: str = ""                      # who speaks the update turns
    # natural-language templates, keyed by event kind; each is a list of
    # phrasings taking {e} (entity), {c} (context clause), {g} (group)
    nl: Dict[str, List[str]] = field(default_factory=dict)
    task_nl: List[str] = field(default_factory=list)
    filler_nl: List[str] = field(default_factory=list)

    def signature(self) -> Dict[str, Tuple[str, ...]]:
        sig = {self.allow: (self.sort,), self.goal: (),
               "grp": (self.sort, "group"),
               # hard tier: pairwise incompatibility and an "in effect" state,
               # so a prohibition can be derived rather than stated
               "pair": (self.sort, self.sort), "active": (self.sort,),
               # hard tier: certification, so an unqualified choice is a
               # violation rather than merely an incomplete task
               "uncert": (self.sort,), "needs_cert": ()}
        for c in self.contexts:
            sig[c] = ()
        return sig

    def universe(self, ents: List[str]) -> Dict[str, List[str]]:
        return {self.sort: list(ents), "group": sorted(self.groups)}

    def act_tool(self) -> Tool:
        return next(t for t in self.tools if t.kind == "act")
