#!/usr/bin/env python3
"""Validate a scenario corpus against docs/corpus_schema.md.

    python3 scripts/validate_corpus.py revoke/domains/corpora/<key>.json

Exit 0 and print "OK" when every check passes; otherwise print one line per
failure and exit 1.  Everything the schema document says is enforced here, so a
corpus that passes can be loaded by revoke.domains.corpus and rendered by
revoke.render_long without further inspection.
"""
from __future__ import annotations

import json
import re
import string
import sys
from typing import Dict, List

HIER = {"top", "r3", "p3", "r2", "p2", "r1", "p1", "p0"}
NUMSLOTS = {"n_small", "n_few", "n_few2", "n_weeks", "n_days", "n_min", "n_min_small",
            "n_people", "n_tickets", "n_k", "n_big", "pct", "n_pct_small", "hour", "d", "mon", "tk"}

# key -> (min count, allowed placeholders, required placeholders)
NL = {
    "ADD": (4, {"e"}, {"e"}), "ADD_BAN": (4, {"e"}, {"e"}),
    "CONFLICT": (4, {"e", "c"}, {"e", "c"}), "SUPERSEDE": (4, {"e"}, {"e"}),
    "SUPPORT": (4, {"e"}, {"e"}), "CONDITION": (4, {"e", "c"}, {"e", "c"}),
    "CONDITION_WIDEN": (4, {"e"}, {"e"}), "RETRACT": (4, {"e", "g"}, {"e", "g"}),
    "GROUP_BAN": (4, {"g"}, {"g"}), "CONFLICT_WEAK": (4, {"e", "c"}, {"e", "c"}),
    "SUPPORT_BAN": (4, {"e"}, {"e"}), "CTX_ON": (4, {"c"}, {"c"}), "CTX_OFF": (4, {"c"}, {"c"}),
}
HNL = {
    "ALIAS": (1, {"x", "y"}, {"x", "y"}), "ALIAS_BREAK": (1, {"x", "y"}, {"x", "y"}),
    "GROUP_BAN_COND": (1, {"g", "c", "roster"}, {"g", "c", "roster"}),
    "JOIN": (2, {"e", "g"}, {"e", "g"}), "LEAVE": (2, {"e", "g"}, {"e", "g"}),
    "LIFT": (2, {"e"}, {"e"}), "REVERSE": (2, {"e"}, {"e"}),
    "LEAD_DEFAULT": (2, {"e"}, {"e"}), "SEC_BAN": (2, {"e"}, {"e"}),
}
TERSE = {"ADD": (3, {"e"}, {"e"}), "ADD_BAN": (3, {"e"}, {"e"}), "CONFLICT": (2, {"e", "c"}, {"e", "c"})}
NOISE_KEYS = ("proposal", "hearsay", "question", "other_team", "stale_echo", "praise", "rescind", "near_miss")

# words that assert a specific level of authority.  The engine chooses who speaks
# a rule template, often a rank-1 or rank-2 person, so a template that names the
# top body would claim an authority the grader does not grant the speaker.
AUTHORITY = ["council", "board", "committee", "directors?", "leadership", "consultants?",
             "registrars?", "coordinator", "policy", "signed off", "sign-off", "minuted at"]
# templates whose speaker rank is fixed by the engine and matches their wording
RANK_FIXED = {("hnl", "SEC_BAN"), ("hnl", "LIFT"), ("hnl", "LEAD_DEFAULT"), ("derived", "PAIR")}

REGISTER = re.compile(
    r"\b(approved?|approval|prohibit(ed|ion)?|bann?ed|ban\b|supersed\w*|reaffirm\w*|reinstat\w*|"
    r"contraindicat\w*|deprecat\w*|forbid(den)?|may not|must not|off the list|on the list|"
    r"no longer allowed|cleared for)\b", re.I)


def placeholders(t: str) -> set:
    out = set()
    for _, f, _, _ in string.Formatter().parse(t):
        if f is not None:
            out.add(f)
    return out


class V:
    def __init__(self, c: Dict):
        self.c = c
        self.fails: List[str] = []
        self.ents: List[str] = []
        self.auth_re = re.compile(r"(?!x)x")

    def fail(self, msg: str) -> None:
        self.fails.append(msg)

    def need(self, d, key, typ, where):
        if key not in d:
            self.fail(f"{where}: missing '{key}'")
            return None
        if typ and not isinstance(d[key], typ):
            self.fail(f"{where}.{key}: expected {typ.__name__}")
            return None
        return d[key]

    def neutral(self, lst, where):
        """Rank-variable rule templates may not assert a specific authority."""
        for i, t in enumerate(lst if isinstance(lst, list) else [lst]):
            if not isinstance(t, str):
                continue
            m = self.auth_re.search(t)
            if m:
                self.fail(f"{where}[{i}]: asserts authority ('{m.group(0)}') but the engine may "
                          f"have a rank-1 or rank-2 speaker say it -- make it rank-neutral")

    def templates(self, lst, where, n_min, allowed, required=frozenset(), no_entity=False,
                  no_register=False):
        if not isinstance(lst, list):
            self.fail(f"{where}: expected a list")
            return
        if len(lst) < n_min:
            self.fail(f"{where}: {len(lst)} templates, need ≥{n_min}")
        for i, t in enumerate(lst):
            if not isinstance(t, str) or not t.strip():
                self.fail(f"{where}[{i}]: empty or not a string")
                continue
            try:
                ph = placeholders(t)
            except ValueError as e:
                self.fail(f"{where}[{i}]: bad format string ({e})")
                continue
            extra = ph - set(allowed)
            if extra:
                self.fail(f"{where}[{i}]: unknown placeholder(s) {sorted(extra)}")
            missing = set(required) - ph
            if missing:
                self.fail(f"{where}[{i}]: must use {sorted(missing)}")
            if no_entity:
                low = t.lower()
                hit = [e for e in self.ents if e.lower() in low]
                if hit:
                    self.fail(f"{where}[{i}]: names a governed entity {hit[:2]}")
            if no_register:
                m = REGISTER.search(t)
                if m:
                    self.fail(f"{where}[{i}]: rule register word '{m.group(0)}'")

    def run(self) -> List[str]:
        c = self.c
        for k in ("key", "title", "setting", "sort", "allow", "goal", "persona"):
            v = self.need(c, k, str, "top")
            if v is not None and not v.strip():
                self.fail(f"top.{k}: empty")
        if isinstance(c.get("key"), str) and not re.fullmatch(r"[a-z_]+", c["key"]):
            self.fail("top.key: must match [a-z_]+")
        for k in ("sort", "allow", "goal"):
            if isinstance(c.get(k), str) and not re.fullmatch(r"[a-z_]+", c[k]):
                self.fail(f"top.{k}: must be snake_case")

        # ---- entities
        E = self.need(c, "entities", dict, "top") or {}
        pre = self.need(E, "prefix", str, "entities")
        if pre is not None and not re.fullmatch(r"[a-z]{2}", pre):
            self.fail("entities.prefix: two lowercase letters")
        names = self.need(E, "names", list, "entities") or []
        if len(names) != 96:
            self.fail(f"entities.names: {len(names)} names, need exactly 96")
        low = [str(n).lower().strip() for n in names]
        if len(set(low)) != len(low):
            dup = sorted({n for n in low if low.count(n) > 1})
            self.fail(f"entities.names: duplicates {dup[:5]}")
        for i, a in enumerate(low):
            for j, b in enumerate(low):
                if i != j and a and a in b:
                    self.fail(f"entities.names: '{names[i]}' is a substring of '{names[j]}'")
                    break
        self.ents = [str(n) for n in names]
        conds = self.need(E, "conditions", list, "entities") or []
        if len(conds) != 6:
            self.fail(f"entities.conditions: {len(conds)}, need 6")
        grps = self.need(E, "groups", list, "entities") or []
        if len(grps) != 6:
            self.fail(f"entities.groups: {len(grps)}, need 6")

        # ---- tools
        T = self.need(c, "tools", dict, "top") or {}
        rd = self.need(T, "read", list, "tools") or []
        if len(rd) != 2:
            self.fail(f"tools.read: {len(rd)} tools, need 2")
        for i, t in enumerate(rd):
            for k in ("name", "param", "doc"):
                self.need(t, k, str, f"tools.read[{i}]")
        act = self.need(T, "act", dict, "tools") or {}
        for k in ("name", "param", "doc"):
            self.need(act, k, str, "tools.act")
        if act.get("param") != c.get("sort"):
            self.fail(f"tools.act.param ('{act.get('param')}') must equal sort ('{c.get('sort')}')")

        # ---- nl / hnl / terse
        for table, spec in (("nl", NL), ("hnl", HNL), ("terse", TERSE)):
            D = self.need(c, table, dict, "top") or {}
            for k, (n, allowed, req) in spec.items():
                if k not in D:
                    self.fail(f"{table}: missing '{k}'")
                    continue
                self.templates(D[k], f"{table}.{k}", n, allowed, req)
        self.templates(c.get("task_nl", []), "task_nl", 4, set(), no_entity=True)
        self.templates(c.get("filler_nl", []), "filler_nl", 8, set(), no_entity=True, no_register=True)

        # ---- people / hierarchy
        P = self.need(c, "people", dict, "top") or {}
        roles = self.need(P, "roles", dict, "people") or {}
        for r in ("3", "2", "1"):
            if r not in roles:
                self.fail(f"people.roles: missing rank '{r}'")
        start = self.need(P, "start", dict, "people") or {}
        by_rank = {r: [p for p, k in start.items() if k == r] for r in (0, 1, 2, 3)}
        for p, k in start.items():
            if k not in (0, 1, 2, 3):
                self.fail(f"people.start['{p}']: rank must be 0-3")
        for r, n in ((3, 2), (2, 1), (1, 2), (0, 4)):
            if len(by_rank[r]) < n:
                self.fail(f"people.start: need ≥{n} people at rank {r}, have {len(by_rank[r])}")
        descr = self.need(P, "descr", dict, "people") or {}
        for p in by_rank[0]:
            if p not in descr:
                self.fail(f"people.descr: missing description for rank-0 person '{p}'")
        terms = P.get("authority_terms")
        if not (isinstance(terms, list) and len(terms) >= 3 and all(isinstance(x, str) and x.strip() for x in terms)):
            self.fail("people.authority_terms: ≥3 words/phrases that assert rank-3 or rank-2 authority")
            terms = []
        pats = AUTHORITY + [re.escape(x.strip()) for x in terms]
        self.auth_re = re.compile(r"(?<![A-Za-z])(" + "|".join(pats) + r")(?![A-Za-z])", re.I)
        for table in ("nl", "hnl", "terse"):
            for k, v in (c.get(table) or {}).items():
                if (table, k) not in RANK_FIXED:
                    self.neutral(v, f"{table}.{k}")
        M_ = c.get("numeric") or {}
        self.neutral(M_.get("set", []), "numeric.set")
        self.neutral(M_.get("ban", []), "numeric.ban")
        D_ = c.get("derived") or {}
        for k in ("ON", "OFF"):
            if k in D_:
                self.neutral([D_[k]], f"derived.{k}")
        if "conj" in c:
            self.neutral([c["conj"]], "conj")
        h = c.get("hierarchy")
        if h is not None:
            if not isinstance(h, str):
                self.fail("hierarchy: string or null")
            else:
                self.templates([h], "hierarchy", 1, HIER, {"r3", "p3", "r2", "p2", "r1", "p1", "p0"})

        # ---- noise / suggest
        N = self.need(c, "noise", dict, "top") or {}
        for k in NOISE_KEYS:
            if k not in N:
                self.fail(f"noise: missing '{k}'")
                continue
            self.templates(N[k], f"noise.{k}", 6, {"e"}, {"e"})
        if "suggest" in c:
            self.templates(c["suggest"], "suggest", 4, {"e"}, {"e"})

        # ---- numeric
        M = self.need(c, "numeric", dict, "top") or {}
        for k in ("ctx", "param", "limit"):
            self.need(M, k, str, "numeric")
        if "unit" not in M or not isinstance(M["unit"], str):
            self.fail("numeric.unit: string (may be empty)")
        if M.get("dir") not in ("above", "below"):
            self.fail("numeric.dir: 'above' or 'below'")
        if isinstance(M.get("ctx"), str) and not re.fullmatch(r"[a-z_]+", M["ctx"]):
            self.fail("numeric.ctx: snake_case")
        lo, hi, vals = M.get("lo"), M.get("hi"), M.get("values")
        if not (isinstance(lo, int) and isinstance(hi, int) and lo < hi):
            self.fail("numeric.lo/hi: ints with lo < hi")
        if not (isinstance(vals, list) and len(vals) == 4 and len(set(vals)) == 4
                and all(isinstance(v, int) for v in vals)):
            self.fail("numeric.values: 4 distinct ints")
        elif isinstance(lo, int) and isinstance(hi, int):
            bad = [v for v in vals if not (lo < v < hi)]
            if bad:
                self.fail(f"numeric.values: {bad} not strictly inside ({lo}, {hi})")
        self.templates(M.get("set", []), "numeric.set", 2, {"limit", "v", "u"}, {"limit", "v"})
        self.templates(M.get("probe", []), "numeric.probe", 2, {"v", "u"}, {"v"})
        self.templates(M.get("ban", []), "numeric.ban", 2, {"e", "limit"}, {"e", "limit"})

        # ---- cert / derived / conj
        C = self.need(c, "cert", dict, "top") or {}
        for k, req in (("rule", set()), ("roster", {"roster"}), ("on", set()), ("off", set()),
                       ("grant", {"e"}), ("revoke", {"e"})):
            if k in C:
                self.templates([C[k]], f"cert.{k}", 1, req | {"e", "roster"} if k in ("roster", "grant", "revoke") else req, req)
            else:
                self.fail(f"cert: missing '{k}'")
        if isinstance(C.get("rule"), str) and "violation" not in C["rule"].lower() and "breach" not in C["rule"].lower():
            self.fail("cert.rule: must say off-list use is a violation/breach")
        Dv = self.need(c, "derived", dict, "top") or {}
        for k, req in (("PAIR", {"x", "y"}), ("ON", {"y"}), ("OFF", {"y"})):
            if k in Dv:
                self.templates([Dv[k]], f"derived.{k}", 1, {"x", "y"}, req)
            else:
                self.fail(f"derived: missing '{k}'")
        if "conj" in c:
            self.templates([c["conj"]], "conj", 1, {"e", "c1", "c2"}, {"e", "c1", "c2"})
        else:
            self.fail("top: missing 'conj'")

        # ---- surface
        S = self.need(c, "surface", dict, "top") or {}
        if S.get("layout") not in ("sectioned", "flat"):
            self.fail("surface.layout: 'sectioned' or 'flat'")
        self.templates(S.get("session_titles", []), "surface.session_titles", 6, {"n", "date"}, {"n"},
                       no_entity=True)
        self.templates(S.get("header_lines", []), "surface.header_lines", 6, {"n", "people", "date"},
                       no_entity=True)
        sec = self.need(S, "sections", dict, "surface") or {}
        for k in ("discussion", "decisions", "actions"):
            lst = sec.get(k)
            if not isinstance(lst, list):
                self.fail(f"surface.sections.{k}: list")
            elif S.get("layout") == "sectioned" and len(lst) < 3:
                self.fail(f"surface.sections.{k}: ≥3 headers for a sectioned layout")
        nf = S.get("notice_frame")
        if not (isinstance(nf, list) and len(nf) == 2 and all(isinstance(x, str) for x in nf)):
            self.fail("surface.notice_frame: [pre, post]")
        self.templates(S.get("artifacts", []), "surface.artifacts", 6, set(), no_entity=True)
        self.templates(S.get("action_items", []), "surface.action_items", 24, set(),
                       no_entity=True, no_register=True)
        blocks = S.get("blocks", [])
        self.templates(blocks, "surface.blocks", 40, set(), no_entity=True, no_register=True)
        for i, b in enumerate(blocks if isinstance(blocks, list) else []):
            if isinstance(b, str) and not (3 <= b.count("\n") + 1 <= 6):
                self.fail(f"surface.blocks[{i}]: {b.count(chr(10)) + 1} lines, want 3-6")
        self.need(S, "you_prefix", str, "surface")

        # ---- pad
        Pd = self.need(c, "pad", dict, "top") or {}
        fn = Pd.get("first_names", [])
        ln = Pd.get("last_names", [])
        if not (isinstance(fn, list) and len(fn) >= 30):
            self.fail("pad.first_names: ≥30")
        if not (isinstance(ln, list) and len(ln) >= 20):
            self.fail("pad.last_names: ≥20")
        clash = sorted(set(map(str, fn)) & set(start))
        if clash:
            self.fail(f"pad.first_names: {clash} are also people (speakers)")
        slots = Pd.get("slots", {})
        if not (isinstance(slots, dict) and len(slots) >= 6):
            self.fail("pad.slots: ≥6 named vocabularies")
        for k, v in (slots.items() if isinstance(slots, dict) else []):
            if not re.fullmatch(r"[a-z_]+", k) or k in NUMSLOTS or k in ("p", "p2"):
                self.fail(f"pad.slots.{k}: bad slot name (snake_case, not a reserved slot)")
            if not (isinstance(v, list) and len(v) >= 8):
                self.fail(f"pad.slots.{k}: ≥8 entries")
            else:
                self.templates(v, f"pad.slots.{k}", 8, set(), no_entity=True)
        allowed = set(slots) | NUMSLOTS | {"p", "p2"}
        fams = Pd.get("families", {})
        if not (isinstance(fams, dict) and len(fams) >= 5):
            self.fail("pad.families: ≥5 families")
        for k, v in (fams.items() if isinstance(fams, dict) else []):
            self.templates(v, f"pad.families.{k}", 8, allowed, no_entity=True, no_register=True)
        self.templates(Pd.get("side", []), "pad.side", 10, allowed, no_entity=True, no_register=True)
        self.templates(Pd.get("actions", []), "pad.actions", 6, allowed, {"p"}, no_entity=True,
                       no_register=True)
        return self.fails


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        c = json.load(open(sys.argv[1]))
    except Exception as e:                                       # noqa: BLE001
        print(f"cannot parse JSON: {e}")
        sys.exit(1)
    fails = V(c).run()
    if fails:
        for f in fails[:80]:
            print(f)
        if len(fails) > 80:
            print(f"... {len(fails) - 80} more")
        print(f"\n{len(fails)} failure(s)")
        sys.exit(1)
    n_pad = sum(len(v) for v in c["pad"]["families"].values())
    print(f"OK  {c['key']}: 96 entities, {n_pad} pad templates over {len(c['pad']['slots'])} slots, "
          f"{len(c['surface']['blocks'])} blocks, layout={c['surface']['layout']}")


if __name__ == "__main__":
    main()
