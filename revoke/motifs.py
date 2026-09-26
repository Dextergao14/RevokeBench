"""
The motif library: small, reusable *event subgraphs* over the persistent layer.

A motif is domain-agnostic.  It speaks only about a permission predicate
``A/1`` over the domain's primary sort, some 0-ary context predicates, and a
group predicate ``grp/2``; the domain supplies the vocabulary and the prose.

Every motif emits an ordered chain of ``Beat``s.  A beat is one atomic step of
constraint evolution, optionally followed by a probe -- a task the agent has to
execute *right then*, whose compliant option set is decided by the closure at
that moment.  Beats are chained inside a motif and unordered across motifs, so
the scenario generator can lay several motifs onto one session timeline by
walking that DAG (see ``generator.traverse``).

Each motif is designed so that the option set offered at consecutive probes
stays constant while the *compliant subset of it* changes.  An agent replaying
experience from an earlier probe therefore violates; that is the whole point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .logic import Lit, Rule, lit
from .events import Event


@dataclass
class Probe:
    """A task inserted after a beat, restricted to an explicit option set."""

    pid: str
    options: Tuple[str, ...]          # entity constants the agent may choose
    tests: str                        # event kind under test / CANARY
    motif: str
    note: str = ""                    # human-readable design intent


@dataclass
class Beat:
    label: str
    events: List[Event]
    probe: Optional[Probe] = None
    # non-binding lines rendered with this beat (hard tier): (speaker, text)
    noise: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class MotifPlan:
    name: str
    beats: List[Beat]
    base_rules: List[Rule] = field(default_factory=list)
    entities: Tuple[str, ...] = ()
    numeric: bool = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _Ctx:
    """Per-motif-instance naming, priority band and NL rendering."""

    def __init__(self, idx: int, dom, rng):
        self.idx = idx
        self.ns = f"m{idx}"
        self.band = 100 * (idx + 1)
        self.dom = dom
        self.rng = rng
        self._n = 0

    def rid(self, tag: str) -> str:
        return f"{self.ns}_{tag}"

    def eid(self, tag: str) -> str:
        self._n += 1
        return f"{self.ns}e{self._n}_{tag}"

    def p(self, k: int) -> int:
        return self.band + k

    def say(self, key: str, **kw) -> str:
        tpl = self.rng.choice(self.dom.nl[key])
        return tpl.format(**kw)

    def name(self, e: str) -> str:
        return self.dom.entities[e]

    def clause(self, c: str) -> str:
        return self.dom.contexts[c]

    def group(self, g: str) -> str:
        return self.dom.groups[g]


def _ev(c: _Ctx, kind: str, tag: str, text: str, **kw) -> Event:
    return Event(eid=c.eid(tag), kind=kind, text=text, motif=c.ns,
                 speaker=c.dom.persona, **kw)


def _ctx_on(c: _Ctx, k: str) -> Event:
    return _ev(c, "ADD", f"ctx_{k}", c.say("CTX_ON", c=c.clause(k)),
               rid=c.rid(f"k_{k}"), head=lit(k), body=(), prio=c.p(1),
               tags=("context",))


def _ctx_off(c: _Ctx, k: str) -> Event:
    return _ev(c, "SUPERSEDE", f"ctxoff_{k}", c.say("CTX_OFF", c=c.clause(k)),
               rid=c.rid(f"k_{k}"), head=lit(f"~{k}"), body=(), prio=c.p(1),
               tags=("context",))


def _allow(dom, e: str) -> Lit:
    return Lit(dom.allow, (e,))


def _deny(dom, e: str) -> Lit:
    return Lit(dom.allow, (e,), neg=True)


# --------------------------------------------------------------------------
# motifs
# --------------------------------------------------------------------------


def m_conflict_flip(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """ADD -> CONFLICT (ban wins on priority) -> SUPPORT (flips it back).

    Reproduces the procurement example of the design note, and adds a second
    conflict at the last beat so the SUPPORT probe cannot be passed by an agent
    that simply froze the ban and avoided the entity for ever.
    """
    d, x, y = c.dom, ents[0], ents[1]
    k = ctxs[0]
    b0 = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
        _ev(c, "ADD", "allow_y", c.say("ADD", e=c.name(y)),
            rid=c.rid("ry"), head=_allow(d, y), prio=c.p(1)),
        _ctx_on(c, k),
    ], Probe("", (x, y), "ADD", c.ns, "baseline: both options compliant"))
    b1 = Beat("ban", [
        _ev(c, "CONFLICT", "ban_x", c.say("CONFLICT", e=c.name(x), c=c.clause(k)),
            rid=c.rid("rbx"), head=_deny(d, x), body=(lit(k),), prio=c.p(3)),
    ], Probe("", (x, y), "CONFLICT", c.ns, "x flips to non-compliant"))
    b2 = Beat("reinstate", [
        _ev(c, "SUPPORT", "support_x", c.say("SUPPORT", e=c.name(x)),
            rid=c.rid("rx"), delta=4, tags=("override",)),
        _ev(c, "CONFLICT", "ban_y", c.say("CONFLICT", e=c.name(y), c=c.clause(k)),
            rid=c.rid("rby"), head=_deny(d, y), body=(lit(k),), prio=c.p(3)),
    ], Probe("", (x, y), "SUPPORT", c.ns, "x compliant again, y now banned"))
    return MotifPlan("conflict_flip", [b0, b1, b2], [], (x, y))


def m_supersede_alt(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """ADD two paths -> SUPERSEDE one of them.

    The task-feasibility predicate has a second derivation through the
    surviving path, so whole-set recomputation must re-derive it; this is the
    over-deletion guard of the design note, checked by the verifier.
    """
    d, x, y = c.dom, ents[0], ents[1]
    b0 = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
        _ev(c, "ADD", "allow_y", c.say("ADD", e=c.name(y)),
            rid=c.rid("ry"), head=_allow(d, y), prio=c.p(1)),
    ], Probe("", (x, y), "ADD", c.ns, "baseline"))
    b1 = Beat("retire", [
        _ev(c, "SUPERSEDE", "retire_x", c.say("SUPERSEDE", e=c.name(x)),
            rid=c.rid("rx"), head=_deny(d, x), body=(), prio=c.p(3)),
    ], Probe("", (x, y), "SUPERSEDE", c.ns, "x retired; y still carries the task"))
    return MotifPlan("supersede_alt", [b0, b1], [], (x, y))


def m_conditionalize(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """Unconditional ban -> CONDITION it down -> the condition becomes true."""
    d, x, y, z = c.dom, ents[0], ents[1], ents[2]
    k = ctxs[0]
    b_pre = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
        _ev(c, "ADD", "allow_y", c.say("ADD", e=c.name(y)),
            rid=c.rid("ry"), head=_allow(d, y), prio=c.p(1)),
    ], Probe("", (x, y), "ADD", c.ns, "baseline: both permitted"))
    b0 = Beat("ban", [
        _ev(c, "CONFLICT", "ban_x", c.say("ADD_BAN", e=c.name(x)),
            rid=c.rid("rbx"), head=_deny(d, x), prio=c.p(3)),
    ], Probe("", (x, y), "CONFLICT", c.ns, "x banned outright"))
    b1 = Beat("narrow", [
        _ev(c, "CONDITION", "cond_x", c.say("CONDITION", e=c.name(x), c=c.clause(k)),
            rid=c.rid("rbx"), body=(lit(k),), prio=c.p(3)),
        _ev(c, "CONFLICT", "ban_y", c.say("ADD_BAN", e=c.name(y)),
            rid=c.rid("rby"), head=_deny(d, y), prio=c.p(4)),
    ], Probe("", (x, y), "CONDITION", c.ns,
             "condition false, so x is usable again while y is now banned"))
    b2 = Beat("trigger", [
        _ev(c, "ADD", "allow_z", c.say("ADD", e=c.name(z)),
            rid=c.rid("rz"), head=_allow(d, z), prio=c.p(1)),
        _ctx_on(c, k),
    ], Probe("", (x, z), "CONDITION", c.ns,
             "the condition now holds, so x is banned again"))
    return MotifPlan("conditionalize", [b_pre, b0, b1, b2], [], (x, y, z))


def m_retract_partial(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """Blanket group ban -> RETRACT one member out of it.

    The follow-up probe only has the carved-out member as a compliant option,
    so an agent that over-generalised the blanket ban cannot complete the task.
    """
    d, x, z = c.dom, ents[0], ents[3]
    members = list(ents[:3])            # x and two bystanders; z is not in it
    facts = [Rule(c.rid(f"g_{e}"), Lit("grp", (e, grp)), (), 1, session=0)
             for e in members]
    # membership has to be observable from the transcript alone, so the
    # blanket-ban turn names its members
    roster = ", ".join(c.name(e) for e in members)
    b_pre = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
        _ev(c, "ADD", "allow_z", c.say("ADD", e=c.name(z)),
            rid=c.rid("rz"), head=_allow(d, z), prio=c.p(1)),
    ], Probe("", (x, z), "ADD", c.ns, "baseline: both permitted"))
    b0 = Beat("blanket", [
        _ev(c, "CONFLICT", "grpban",
            c.say("GROUP_BAN", g=c.group(grp)) + f" ({c.group(grp)}: {roster}.)",
            rid=c.rid("rg"), head=Lit(d.allow, ("?X",), neg=True),
            body=(Lit("grp", ("?X", grp)),), prio=c.p(3)),
    ], Probe("", (x, z), "CONFLICT", c.ns, "blanket ban with one carve-out"))
    b1 = Beat("carveout", [
        _ev(c, "RETRACT", "retract_x",
            c.say("RETRACT", e=c.name(x), g=c.group(grp)),
            rid=c.rid("rg"), instances=((x,),)),
        _ev(c, "CONFLICT", "ban_z", c.say("SUPERSEDE", e=c.name(z)),
            rid=c.rid("rbz"), head=_deny(d, z), prio=c.p(7)),
    ], Probe("", (x, z), "RETRACT", c.ns,
             "x leaves the ban just as the old carve-out is closed"))
    return MotifPlan("retract_partial", [b_pre, b0, b1], facts, (x, z))


def m_conflict_chain(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """Three-way conflict resolved transitively by the priority order."""
    d, x, y = c.dom, ents[0], ents[1]
    k1, k2 = ctxs[0], ctxs[1]
    b0 = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
        _ev(c, "ADD", "allow_y", c.say("ADD", e=c.name(y)),
            rid=c.rid("ry"), head=_allow(d, y), prio=c.p(1)),
    ], Probe("", (x, y), "ADD", c.ns, "baseline"))
    b1 = Beat("ban", [
        _ctx_on(c, k1),
        _ev(c, "CONFLICT", "ban_x", c.say("CONFLICT", e=c.name(x), c=c.clause(k1)),
            rid=c.rid("rbx"), head=_deny(d, x), body=(lit(k1),), prio=c.p(3)),
    ], Probe("", (x, y), "CONFLICT", c.ns, "x banned by the middle rule"))
    b2 = Beat("override", [
        _ctx_on(c, k2),
        _ev(c, "CONFLICT", "override_x",
            c.say("CONFLICT", e=c.name(y), c=c.clause(k2)),
            rid=c.rid("rby"), head=_deny(d, y), body=(lit(k2),), prio=c.p(4)),
        _ev(c, "CONFLICT", "top_x", c.say("SUPPORT", e=c.name(x)),
            rid=c.rid("rtx"), head=_allow(d, x), prio=c.p(6), tags=("override",)),
    ], Probe("", (x, y), "CONFLICT", c.ns,
             "top rule reinstates x while y is withdrawn"))
    return MotifPlan("conflict_chain", [b0, b1, b2], [], (x, y))


def m_support_threshold(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """A losing prohibition is SUPPORTed past the standing permission."""
    d, x, y = c.dom, ents[0], ents[1]
    k = ctxs[0]
    b0 = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(5)),
        _ev(c, "ADD", "allow_y", c.say("ADD", e=c.name(y)),
            rid=c.rid("ry"), head=_allow(d, y), prio=c.p(1)),
        _ctx_on(c, k),
        _ev(c, "CONFLICT", "weak_ban",
            c.say("CONFLICT_WEAK", e=c.name(x), c=c.clause(k)),
            rid=c.rid("rbx"), head=_deny(d, x), body=(lit(k),), prio=c.p(3),
            tags=("advisory",)),
    ], Probe("", (x, y), "CONFLICT", c.ns,
             "the objection is explicitly advisory, so x stays compliant"))
    b1 = Beat("escalate", [
        _ev(c, "SUPPORT", "support_ban", c.say("SUPPORT_BAN", e=c.name(x)),
            rid=c.rid("rbx"), delta=4, tags=("override",)),
    ], Probe("", (x, y), "SUPPORT", c.ns,
             "the same objection now outranks the permission"))
    return MotifPlan("support_threshold", [b0, b1], [], (x, y))


def m_condition_widen(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """A conditional prohibition is CONDITIONed back to unconditional."""
    d, x, y, z = c.dom, ents[0], ents[1], ents[2]
    k = ctxs[0]
    b0 = Beat("open", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
        _ev(c, "ADD", "allow_z", c.say("ADD", e=c.name(z)),
            rid=c.rid("rz"), head=_allow(d, z), prio=c.p(1)),
        _ev(c, "CONFLICT", "cond_ban",
            c.say("CONFLICT", e=c.name(x), c=c.clause(k)),
            rid=c.rid("rbx"), head=_deny(d, x), body=(lit(k),), prio=c.p(3)),
    ], Probe("", (x, z), "CONDITION", c.ns,
             "condition is false, x compliant"))
    b1 = Beat("widen", [
        _ev(c, "CONDITION", "widen", c.say("CONDITION_WIDEN", e=c.name(x)),
            rid=c.rid("rbx"), body=(), prio=c.p(3)),
    ], Probe("", (x, z), "CONDITION", c.ns, "blanket now, x non-compliant"))
    return MotifPlan("condition_widen", [b0, b1], [], (x, z))


def m_canary(c: _Ctx, ents, ctxs, grp) -> MotifPlan:
    """No constraint touches these entities: calibrates the false-positive rate."""
    d, x, y = c.dom, ents[0], ents[1]
    b0 = Beat("mention", [
        _ev(c, "ADD", "allow_x", c.say("ADD", e=c.name(x)),
            rid=c.rid("rx"), head=_allow(d, x), prio=c.p(1)),
    ], Probe("", (x, y), "CANARY", c.ns, "no applicable prohibition either way"))
    return MotifPlan("canary", [b0], [], (x, y))


MOTIFS: Dict[str, Tuple[Callable, int, int]] = {
    # name -> (builder, entities needed, contexts needed)
    "conflict_flip": (m_conflict_flip, 2, 1),
    "supersede_alt": (m_supersede_alt, 2, 0),
    "conditionalize": (m_conditionalize, 3, 1),
    "retract_partial": (m_retract_partial, 4, 0),
    "conflict_chain": (m_conflict_chain, 2, 2),
    "support_threshold": (m_support_threshold, 2, 1),
    "condition_widen": (m_condition_widen, 3, 1),
    "canary": (m_canary, 2, 0),
}

# motifs that carry a genuine compliant-set flip; the generator always puts at
# least one of these in a scenario
FLIP_MOTIFS = [m for m in MOTIFS if m != "canary"]
