"""
Load a schema-v2 scenario corpus (docs/corpus_schema.md) into a Domain and
register it with every hard-tier table.

A corpus is vocabulary and surface only.  Loading one adds a new domain key to
HARD / TERSE / PEOPLE / NUMERIC / CERT / DERIVED / CONJ / HNL exactly as the
meetings domain does at import, so `revoke.hard.build_hard_scenario` runs on it
unchanged and every item is graded by the same engine.

    from revoke.domains.corpus import load_corpus
    dom = load_corpus("revoke/domains/corpora/clinical_ward.json")

The loader refuses a corpus that does not pass scripts/validate_corpus.py: a
missing template would otherwise surface as a KeyError deep inside a motif.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

from .base import Domain, Tool
from . import hard_ext

CORPORA: Dict[str, Dict] = {}          # key -> raw corpus, for the renderer


def _validate(path: str, c: Dict) -> None:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(here, "scripts"))
    from validate_corpus import V                      # noqa: E402
    fails = V(c).run()
    if fails:
        raise ValueError(f"{path}: corpus fails validation:\n  " + "\n  ".join(fails[:12]))


def load_corpus(path: str, validate: bool = True) -> Domain:
    c = json.load(open(path))
    if validate:
        _validate(path, c)
    key = c["key"]
    E = c["entities"]
    pre = E["prefix"]
    # first 48 are the base pool, the rest the hard-tier extension, matching
    # how the seed domains are split
    ents = {f"{pre}{i:02d}": n for i, n in enumerate(E["names"][:48], 1)}
    extra = {f"{pre}{i:02d}": n for i, n in enumerate(E["names"][48:96], 49)}
    contexts = {f"k_{i}": cl for i, cl in enumerate(E["conditions"])}
    groups = {f"g_{i}": g for i, g in enumerate(E["groups"])}

    tools = [Tool(r["name"], {r["param"]: ""}, "read", doc=r["doc"]) for r in c["tools"]["read"]]
    a = c["tools"]["act"]
    tools.append(Tool(a["name"], {c["sort"]: c["sort"]}, "act",
                      asserts=(f"{c['allow']}({{{c['sort']}}})",), achieves=(c["goal"],),
                      doc=a["doc"]))

    dom = Domain(key=key, title=c["title"], sort=c["sort"], allow=c["allow"], goal=c["goal"],
                 persona=c["persona"], entities=ents, contexts=contexts, groups=groups,
                 tools=tools, nl=dict(c["nl"]), task_nl=list(c["task_nl"]),
                 filler_nl=list(c["filler_nl"]))

    # ---- hard-tier tables ---------------------------------------------------
    start = {p: int(r) for p, r in c["people"]["start"].items()}
    hard_ext.HARD[key] = {
        "speakers": {r: [p for p, k in start.items() if k == r] for r in (3, 2, 1, 0)},
        "hierarchy": c.get("hierarchy") or None,
        "noise": c["noise"],
        "suggest": c.get("suggest"),
        "extra": extra,
    }
    hard_ext.TERSE[key] = c["terse"]
    hard_ext.PEOPLE[key] = {"roles": {int(k): v for k, v in c["people"]["roles"].items()},
                            "start": start, "descr": dict(c["people"]["descr"])}
    hard_ext.NUMERIC[key] = c["numeric"]
    hard_ext.CERT[key] = c["cert"]
    hard_ext.DERIVED[key] = c["derived"]
    hard_ext.CONJ[key] = c["conj"]
    from ..motifs_hard import HNL                      # noqa: E402  (import cycle otherwise)
    HNL[key] = c["hnl"]
    CORPORA[key] = c
    return dom
