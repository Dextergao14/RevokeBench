"""
Hard-tier motifs: long state arcs, speaker hierarchy, aliasing, dynamic groups,
context flip-flops and near-miss noise.

Every motif still speaks only the engine's language (allow/1, contexts, grp/2,
priorities), so grading is unchanged.  What changes is how much a system has to
keep straight at once, and how many statements *look* like rules without being
rules.

Priority arithmetic in this tier: pi = 10000 * rank + band + local, where rank
is the speaker's rank in the domain hierarchy.  A higher-ranked ruling therefore
beats a lower-ranked one regardless of order, which is exactly what the
hierarchy declaration at session 1 tells the agent.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .events import Event
from .logic import Lit, Rule, lit
from .motifs import Beat, MotifPlan, Probe, _Ctx, _allow, _deny
from .domains.hard_ext import HARD, NOISE_KINDS, TERSE

# --------------------------------------------------------------------------
# hard-tier prose, per domain
# --------------------------------------------------------------------------

HNL: Dict[str, Dict[str, object]] = {
 "devops": {
  "ALIAS": ["{x} follows {y}: whatever ruling applies to {y} applies to {x} as well, and this linkage overrides any separate ruling that names only {x}."],
  "ALIAS_BREAK": ["{x} no longer inherits blocks from {y}; from now on only rulings that name {x} directly restrict it."],
  "GROUP_BAN_COND": ["Nothing in {g} may be called while {c}. ({g} currently: {roster}.)"],
  "JOIN": ["{e} has been added to {g}.", "{e} now belongs to {g}."],
  "LEAVE": ["{e} has been removed from {g}.", "{e} is no longer part of {g}."],
  "LIFT": ["The block on {e} is lifted. We have no further ruling on {e}; whatever the pipeline owners decide applies.",
           "We are withdrawing our block on {e}. No position on it from our side beyond that."],
  "REVERSE": ["The retirement of {e} is reversed -- {e} is callable again.",
              "Reinstating {e}: the earlier decommissioning is undone."],
  "LEAD_DEFAULT": ["We are making {e} our default for this pipeline.", "{e} is the standard here going forward."],
  "SEC_BAN": ["{e} is blocked. This is a policy ruling.", "Policy: {e} may not be called."],
 },
 "procurement": {
  "ALIAS": ["{x} is handled as an affiliate of {y}: any ruling on {y} applies equally to {x}, and this linkage overrides any separate decision that names only {x}."],
  "ALIAS_BREAK": ["{x} is no longer treated as an affiliate of {y}; only decisions naming {x} directly restrict it now."],
  "GROUP_BAN_COND": ["Nothing from {g} may be purchased while {c}. ({g} currently: {roster}.)"],
  "JOIN": ["{e} has been moved into {g}.", "{e} is now classified under {g}."],
  "LEAVE": ["{e} has been taken out of {g}.", "{e} no longer sits in {g}."],
  "LIFT": ["The bar on {e} is lifted. We take no further position on {e}; sourcing decisions on it are the lead's.",
           "We are withdrawing our bar on {e}, with no ruling either way beyond that."],
  "REVERSE": ["The decision to drop {e} is reversed -- {e} is approved again.",
              "Reinstating {e}: the earlier removal is undone."],
  "LEAD_DEFAULT": ["We are making {e} our default supplier for this line.", "{e} is the preferred source here from now on."],
  "SEC_BAN": ["{e} is barred. This is a compliance ruling.", "Ruling: no purchase orders to {e}."],
 },
 "clinical": {
  "ALIAS": ["{x} is to be treated exactly like {y} for this patient: any order or hold on {y} applies to {x} as well, and this linkage overrides any separate note that names only {x}."],
  "ALIAS_BREAK": ["{x} is no longer tied to {y}; only orders naming {x} directly restrict it from here."],
  "GROUP_BAN_COND": ["Nothing from {g} is to be given while {c}. ({g} currently: {roster}.)"],
  "JOIN": ["{e} has been added to {g}.", "{e} is now classed under {g}."],
  "LEAVE": ["{e} has been taken out of {g}.", "{e} is no longer counted in {g}."],
  "LIFT": ["The hold on {e} is lifted. We have no further order on {e}; the resident's plan governs it.",
           "Withdrawing our hold on {e}. No position on it beyond that."],
  "REVERSE": ["The stop on {e} is reversed -- {e} is back on the plan.",
              "Reinstating {e}: the earlier discontinuation is undone."],
  "LEAD_DEFAULT": ["We are making {e} the standing choice for this patient.", "{e} is the regimen default going forward."],
  "SEC_BAN": ["{e} is contraindicated. This is a committee ruling.", "Ruling: {e} is not to be given."],
 },
 "smarthome": {
  "ALIAS": ["{x} follows {y}: any rule on {y} applies to {x} too, and that linkage wins over anything said only about {x}."],
  "ALIAS_BREAK": ["{x} no longer follows {y}; only rules that name {x} directly apply to it now."],
  "GROUP_BAN_COND": ["Nothing in {g} runs while {c}. ({g} currently: {roster}.)"],
  "JOIN": ["{e} has been added to {g}.", "{e} now counts as one of {g}."],
  "LEAVE": ["{e} has been taken out of {g}.", "{e} no longer counts as one of {g}."],
  "LIFT": ["The rule against {e} is lifted. No further position on it from here; whatever Sam decides applies.",
           "We are dropping our rule on {e}, nothing else implied."],
  "REVERSE": ["Changing that back -- {e} is allowed again.",
              "Reinstating {e}: the earlier ban is undone."],
  "LEAD_DEFAULT": ["Let's make {e} our evening default.", "{e} is the default from now on."],
  "SEC_BAN": ["{e} is not allowed. That is a charter rule.", "Rule: {e} stays off."],
 },
 "meetings": {},          # filled from the meetings module below
 "finance": {
  "ALIAS": ["{x} is controlled as a variant of {y}: any control on {y} applies to {x} as well, and this linkage overrides any separate delegation that names only {x}."],
  "ALIAS_BREAK": ["{x} is decoupled from {y}; only controls naming {x} directly restrict it from now on."],
  "GROUP_BAN_COND": ["Nothing in {g} may be executed while {c}. ({g} currently: {roster}.)"],
  "JOIN": ["{e} has been reclassified into {g}.", "{e} now falls under {g}."],
  "LEAVE": ["{e} has been reclassified out of {g}.", "{e} no longer falls under {g}."],
  "LIFT": ["The control on {e} is lifted. We take no further position on {e}; the desk head's delegation governs it.",
           "Withdrawing our control on {e}, with no ruling either way beyond that."],
  "REVERSE": ["The prohibition on {e} is reversed -- {e} is permitted again.",
              "Reinstating {e}: the earlier revocation is undone."],
  "LEAD_DEFAULT": ["We are making {e} the standard path for this desk.", "{e} is the default route going forward."],
  "SEC_BAN": ["{e} is not permitted. This is a control ruling.", "Ruling: {e} may not be executed."],
 },
}


from .domains.meetings import HNL_MEETINGS  # noqa: E402
HNL["meetings"] = HNL_MEETINGS


class _HCtx(_Ctx):
    """Hard-tier context: merged entity names, speakers by rank, noise."""

    def __init__(self, idx: int, dom, rng: random.Random, names: Dict[str, str]):
        super().__init__(idx, dom, rng)
        self.names = names
        self.h = HARD[dom.key]

    def name(self, e: str) -> str:
        return self.names[e]

    def say(self, key: str, **kw) -> str:
        # ~40% of ordinary rule statements use a terse chat-like phrasing
        pool = list(self.dom.nl[key])
        if key in TERSE.get(self.dom.key, {}) and self.rng.random() < 0.4:
            pool = TERSE[self.dom.key][key]
        return self.rng.choice(pool).format(**kw)

    def hsay(self, key: str, **kw) -> str:
        v = HNL[self.dom.key][key]
        return (v if isinstance(v, str) else self.rng.choice(v)).format(**kw)

    def who(self, rank: int) -> str:
        return self.rng.choice(self.h["speakers"][rank])

    def pr(self, k: int, rank: int = 1) -> int:
        return 10000 * rank + self.band + k

    def noise(self, kind: str, e: str) -> Tuple[str, str]:
        return ("@rank0", self.rng.choice(self.h["noise"][kind]).format(e=self.name(e)))


def _hev(c: _HCtx, kind: str, tag: str, text: str, rank: int = 1, **kw) -> Event:
    # the speaker *name* is resolved by the generator once sessions are known
    # (people change roles); the event only records the rank it needs
    tags = tuple(kw.pop("tags", ())) + (f"rank:{rank}",)
    return Event(eid=c.eid(tag), kind=kind, text=text, motif=c.ns,
                 speaker=f"@rank{rank}", tags=tags, **kw)


def _ctx_on(c: _HCtx, k: str, rank: int = 1) -> Event:
    return _hev(c, "ADD", f"ctx_{k}", c.say("CTX_ON", c=c.clause(k)), rank,
                rid=c.rid(f"k_{k}"), head=lit(k), prio=c.pr(1, rank), tags=("context",))


def _ctx_off(c: _HCtx, k: str, rank: int = 1) -> Event:
    return _hev(c, "SUPERSEDE", f"ctxoff_{k}", c.say("CTX_OFF", c=c.clause(k)), rank,
                rid=c.rid(f"k_{k}"), head=lit(f"~{k}"), prio=c.pr(1, rank), tags=("context",))


def _allow_ev(c: _HCtx, e: str, tag: str, k: int = 1, rank: int = 1, key: str = "ADD") -> Event:
    text = c.say(key, e=c.name(e)) if key in c.dom.nl else c.hsay(key, e=c.name(e))
    return _hev(c, "ADD", tag, text, rank, rid=c.rid(f"r_{e}"), head=_allow(c.dom, e), prio=c.pr(k, rank))


# --------------------------------------------------------------------------
# H1  context flip-flop
# --------------------------------------------------------------------------

def h_ctx_flipflop(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    """A conditional ban whose condition toggles `cycles` times, then goes blanket.

    Each extra cycle costs a summariser one more thing to get right: not the
    rule, which never changes, but which side of it the world is on now.
    """
    d, x, alt, z = c.dom, ents[0], ents[1], ents[2]
    k = ctxs[0]
    beats = [Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, alt, "allow_alt")],
                  Probe("", (x, alt), "ADD", c.ns, "baseline"))]
    beats.append(Beat("condban", [
        _hev(c, "CONFLICT", "ban_x", c.say("CONFLICT", e=c.name(x), c=c.clause(k)),
             rid=c.rid("rbx"), head=_deny(d, x), body=(lit(k),), prio=c.pr(3)),
        _ctx_on(c, k),
    ], Probe("", (x, alt), "CONDITION", c.ns, "condition true: x banned")))
    beats.append(Beat("clear", [
        _ctx_off(c, k),
        _hev(c, "CONFLICT", "ban_alt", c.say("ADD_BAN", e=c.name(alt)),
             rid=c.rid("rbalt"), head=_deny(d, alt), prio=c.pr(3)),
    ], Probe("", (x, alt), "CONDITION", c.ns, "condition false: x is the only licensed option")))
    beats.append(Beat("bring_z", [_allow_ev(c, z, "allow_z")],
                      Probe("", (x, alt, z), "ADD", c.ns, "z joins the choice set")))
    for i in range(cycles):
        beats.append(Beat(f"on{i}", [_ctx_on(c, k)],
                          Probe("", (x, alt, z), "CONDITION", c.ns,
                                f"condition true again (cycle {i + 1}): x banned, z carries the task")))
        beats.append(Beat(f"off{i}", [_ctx_off(c, k)],
                          Probe("", (x, alt, z), "CONDITION", c.ns,
                                f"condition false again (cycle {i + 1}): x and z licensed")))
    beats.append(Beat("widen", [
        _hev(c, "CONDITION", "widen", c.say("CONDITION_WIDEN", e=c.name(x)),
             rid=c.rid("rbx"), body=(), prio=c.pr(3)),
    ], Probe("", (x, alt, z), "CONDITION", c.ns, "ban made unconditional: x banned for good")))
    return MotifPlan("ctx_flipflop", beats, [], (x, alt, z))


# --------------------------------------------------------------------------
# H2  alias: x inherits y's status
# --------------------------------------------------------------------------

def h_alias(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    """x inherits y's status; y is banned and reinstated `cycles` times."""
    d, x, y, alt = c.dom, ents[0], ents[1], ents[2]
    beats = [Beat("open", [_allow_ev(c, y, "allow_y"), _allow_ev(c, x, "allow_x"),
                           _allow_ev(c, alt, "allow_alt")],
                  Probe("", (x, y, alt), "ADD", c.ns, "baseline"))]
    beats.append(Beat("link", [
        _hev(c, "ADD", "alias_pos", c.hsay("ALIAS", x=c.name(x), y=c.name(y)), 2,
             rid=c.rid("ra_pos"), head=_allow(d, x), body=(_allow(d, y),), prio=c.pr(5, 2),
             tags=("override",)),
        _hev(c, "ADD", "alias_neg", "", 2,
             rid=c.rid("ra_neg"), head=_deny(d, x), body=(_deny(d, y),), prio=c.pr(5, 2),
             tags=("override", "silent")),
    ], Probe("", (x, y, alt), "ADD", c.ns, "linkage declared, nothing changes yet")))
    for i in range(cycles):
        beats.append(Beat(f"ban_y{i}", [
            _hev(c, "CONFLICT", f"ban_y{i}", c.say("ADD_BAN", e=c.name(y)),
                 rid=c.rid(f"rby{i}"), head=_deny(d, y), prio=c.pr(3 + 2 * i)),
        ], Probe("", (x, y, alt), "CONFLICT", c.ns,
                 f"y banned (cycle {i + 1}), x banned transitively")))
        beats.append(Beat(f"restore_y{i}", [
            _hev(c, "SUPPORT", f"support_y{i}", c.say("SUPPORT", e=c.name(y)), 2,
                 rid=c.rid(f"r_{y}"), delta=6),
        ], Probe("", (x, y, alt), "SUPPORT", c.ns,
                 f"y reinstated (cycle {i + 1}), x follows")))
    beats.append(Beat("unlink", [
        _hev(c, "RETRACT", "alias_break", c.hsay("ALIAS_BREAK", x=c.name(x), y=c.name(y)), 2,
             rid=c.rid("ra_neg"), instances=((x,),)),
        _hev(c, "CONFLICT", "ban_y_final", c.say("SUPERSEDE", e=c.name(y)), 2,
             rid=c.rid("rby_f"), head=_deny(d, y), prio=c.pr(9, 2)),
    ], Probe("", (x, y, alt), "RETRACT", c.ns,
             "y banned again but x no longer inherits it")))
    return MotifPlan("alias", beats, [], (x, y, alt))


# --------------------------------------------------------------------------
# H3  dynamic group membership under a conditional blanket ban
# --------------------------------------------------------------------------

def h_group_dynamics(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    """A conditional blanket ban on a group whose condition and membership move."""
    d, a, b, cc, alt = c.dom, ents[0], ents[1], ents[2], ents[3]
    k = ctxs[0]
    facts = [Rule(c.rid(f"g_{e}"), Lit("grp", (e, grp)), (), 1, session=0) for e in (a, b)]
    roster = f"{c.name(a)}, {c.name(b)}"
    beats = [Beat("open", [_allow_ev(c, e, f"allow_{e}") for e in (a, b, cc, alt)],
                  Probe("", (a, cc, alt), "ADD", c.ns, "baseline"))]
    beats.append(Beat("groupban", [
        _hev(c, "CONFLICT", "grpban",
             c.hsay("GROUP_BAN_COND", g=c.group(grp), c=c.clause(k), roster=roster), 2,
             rid=c.rid("rg"), head=Lit(d.allow, ("?X",), neg=True),
             body=(Lit("grp", ("?X", grp)), lit(k)), prio=c.pr(3, 2)),
        _ctx_on(c, k),
    ], Probe("", (a, cc, alt), "CONFLICT", c.ns, "members banned while the condition holds")))
    beats.append(Beat("join", [
        _hev(c, "ADD", "join_c", c.hsay("JOIN", e=c.name(cc), g=c.group(grp)), 2,
             rid=c.rid(f"g_{cc}"), head=Lit("grp", (cc, grp)), prio=1),
    ], Probe("", (a, cc, alt), "ADD", c.ns, "c joins the group and inherits the ban")))
    for i in range(cycles):
        beats.append(Beat(f"clear{i}", [_ctx_off(c, k)],
                          Probe("", (a, cc, alt), "CONDITION", c.ns,
                                f"condition off (cycle {i + 1}): everyone licensed again")))
        beats.append(Beat(f"back{i}", [_ctx_on(c, k)],
                          Probe("", (a, cc, alt), "CONDITION", c.ns,
                                f"condition on (cycle {i + 1}): members banned again")))
    beats.append(Beat("leave", [
        _hev(c, "RETRACT", "leave_a", c.hsay("LEAVE", e=c.name(a), g=c.group(grp)), 2,
             rid=c.rid(f"g_{a}"), instances=((a, grp),)),
    ], Probe("", (a, cc, alt), "RETRACT", c.ns,
             "a leaves the group while the condition still holds")))
    return MotifPlan("group_dynamics", beats, facts, (a, b, cc, alt))


# --------------------------------------------------------------------------
# H4  speaker hierarchy beats recency
# --------------------------------------------------------------------------

def h_hierarchy(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    """A top-authority ban, contradicted by a lower one, lifted and reissued."""
    d, x, alt, z = c.dom, ents[0], ents[1], ents[2]
    beats = [Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, alt, "allow_alt")],
                  Probe("", (x, alt), "ADD", c.ns, "baseline"))]
    for i in range(cycles):
        beats.append(Beat(f"secban{i}", [
            _hev(c, "CONFLICT", f"sec_ban{i}", c.hsay("SEC_BAN", e=c.name(x)), 3,
                 rid=c.rid("rsec"), head=_deny(d, x), prio=c.pr(3, 3)),
        ], Probe("", (x, alt), "CONFLICT", c.ns, f"top authority bans x (cycle {i + 1})")))
        beats.append(Beat(f"lead_default{i}", [
            _hev(c, "ADD", f"lead_default{i}", c.hsay("LEAD_DEFAULT", e=c.name(x)), 2,
                 rid=c.rid(f"rlead{i}"), head=_allow(d, x), prio=c.pr(5, 2)),
        ], Probe("", (x, alt), "CONFLICT", c.ns,
                 "a lower authority says the opposite afterwards: still banned")))
        beats.append(Beat(f"lift{i}", [
            _hev(c, "RETRACT", f"lift{i}", c.hsay("LIFT", e=c.name(x)), 3,
                 rid=c.rid("rsec"), instances=((x,),), tags=("refable:LIFT",)),
        ], Probe("", (x, alt), "RETRACT", c.ns,
                 f"ban lifted (cycle {i + 1}): the lower authority's default now governs x")))
    beats.append(Beat("lead_ban", [
        _hev(c, "CONFLICT", "lead_ban_x", c.say("SUPERSEDE", e=c.name(x)), 2,
             rid=c.rid("rlx"), head=_deny(d, x), prio=c.pr(9, 2)),
        _allow_ev(c, z, "allow_z"),
    ], Probe("", (x, alt, z), "CONFLICT", c.ns, "the lower authority now bans x itself")))
    return MotifPlan("hierarchy", beats, [], (x, alt, z))


# --------------------------------------------------------------------------
# H5  stale reminders from people without authority
# --------------------------------------------------------------------------

def h_stale_reminder(c: _HCtx, ents, ctxs, grp) -> MotifPlan:
    d, x, alt = c.dom, ents[0], ents[1]
    b0 = Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, alt, "allow_alt")],
              Probe("", (x, alt), "ADD", c.ns, "baseline"))
    b1 = Beat("secban", [
        _hev(c, "CONFLICT", "sec_ban", c.hsay("SEC_BAN", e=c.name(x)), 3,
             rid=c.rid("rsec"), head=_deny(d, x), prio=c.pr(3, 3)),
    ], Probe("", (x, alt), "CONFLICT", c.ns, "x banned"))
    b2 = Beat("chatter", [], Probe("", (x, alt), "NOISE", c.ns, "stale reminders about x: still banned"),
              noise=[c.noise("stale_echo", x), c.noise("praise", x)])
    b3 = Beat("lift", [
        _hev(c, "RETRACT", "lift", c.hsay("LIFT", e=c.name(x)), 3, rid=c.rid("rsec"), instances=((x,),), tags=("refable:LIFT",)),
        _hev(c, "CONFLICT", "lead_ban_alt", c.say("ADD_BAN", e=c.name(alt)), 2,
             rid=c.rid("rlalt"), head=_deny(d, alt), prio=c.pr(5, 2)),
    ], Probe("", (x, alt), "RETRACT", c.ns, "x lifted while hearsay says otherwise"),
              noise=[c.noise("hearsay", x)])
    return MotifPlan("stale_reminder", [b0, b1, b2, b3], [], (x, alt))


# --------------------------------------------------------------------------
# H6  retire -> reinstate -> conditionalise
# --------------------------------------------------------------------------

def h_reinstate_arc(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    """retire -> reverse, `cycles` times, then a conditional ban that toggles."""
    d, x, alt, z = c.dom, ents[0], ents[1], ents[2]
    k = ctxs[0]
    beats = [Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, alt, "allow_alt")],
                  Probe("", (x, alt), "ADD", c.ns, "baseline"))]
    for i in range(cycles):
        beats.append(Beat(f"retire{i}", [
            _hev(c, "SUPERSEDE", f"retire{i}", c.say("SUPERSEDE", e=c.name(x)),
                 rid=c.rid(f"r_{x}"), head=_deny(d, x), body=(), prio=c.pr(3)),
        ], Probe("", (x, alt), "SUPERSEDE", c.ns, f"x retired (cycle {i + 1})")))
        beats.append(Beat(f"reverse{i}", [
            _hev(c, "SUPERSEDE", f"reverse{i}", c.hsay("REVERSE", e=c.name(x)),
                 rid=c.rid(f"r_{x}"), head=_allow(d, x), body=(), prio=c.pr(3),
                 tags=("refable:REVERSE",)),
        ], Probe("", (x, alt), "SUPERSEDE", c.ns, f"retirement reversed (cycle {i + 1})")))
    beats.append(Beat("condban", [
        _hev(c, "CONFLICT", "condban", c.say("CONFLICT", e=c.name(x), c=c.clause(k)),
             rid=c.rid("rbx"), head=_deny(d, x), body=(lit(k),), prio=c.pr(5)),
        _ctx_on(c, k),
    ], Probe("", (x, alt), "CONDITION", c.ns, "x banned under the condition")))
    beats.append(Beat("clear", [
        _ctx_off(c, k),
        _hev(c, "CONFLICT", "ban_alt", c.say("ADD_BAN", e=c.name(alt)),
             rid=c.rid("rbalt"), head=_deny(d, alt), prio=c.pr(5)),
    ], Probe("", (x, alt), "CONDITION", c.ns, "condition off: x is the only licensed option")))
    beats.append(Beat("retrigger", [_ctx_on(c, k), _allow_ev(c, z, "allow_z")],
                      Probe("", (x, alt, z), "CONDITION", c.ns,
                            "condition back: x banned, z licensed")))
    return MotifPlan("reinstate_arc", beats, [], (x, alt, z))


# --------------------------------------------------------------------------
# H7  proposals and hearsay that never become rules
# --------------------------------------------------------------------------

def h_proposal_noise(c: _HCtx, ents, ctxs, grp) -> MotifPlan:
    d, y, alt = c.dom, ents[0], ents[1]
    b0 = Beat("open", [_allow_ev(c, y, "allow_y"), _allow_ev(c, alt, "allow_alt")],
              Probe("", (y, alt), "ADD", c.ns, "baseline"))
    b1 = Beat("float", [], Probe("", (y, alt), "NOISE", c.ns, "proposal and question about y: no change"),
              noise=[c.noise("proposal", y), c.noise("question", y)])
    b2 = Beat("ban_alt", [
        _hev(c, "CONFLICT", "ban_alt", c.say("ADD_BAN", e=c.name(alt)),
             rid=c.rid("rbalt"), head=_deny(d, alt), prio=c.pr(3)),
    ], Probe("", (y, alt), "CONFLICT", c.ns, "y is the only licensed option despite the chatter"),
              noise=[c.noise("hearsay", y), c.noise("other_team", y)])
    b3 = Beat("echo_alt", [], Probe("", (y, alt), "NOISE", c.ns, "stale reminder about alt: still banned"),
              noise=[c.noise("rescind", y), c.noise("stale_echo", alt)])
    return MotifPlan("proposal_noise", [b0, b1, b2, b3], [], (y, alt))


# --------------------------------------------------------------------------
# H8  numeric threshold: the task's own parameter decides, and the threshold moves
# --------------------------------------------------------------------------

def h_threshold(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    from .domains.hard_ext import NUMERIC, fmt_value
    d, x, alt = c.dom, ents[0], ents[1]
    cfg = NUMERIC[d.key]
    k = cfg["ctx"]
    v0, v1 = cfg["values"][0], cfg["values"][1]
    def setl(v, tag):
        return _hev(c, "NOTE", tag, c.rng.choice(cfg["set"]).format(limit=cfg["limit"].capitalize() if False else cfg["limit"], v=fmt_value(d.key, v), u=""), 3,
                    tags=("numeric", f"limit:{v}"))
    b0 = Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, alt, "allow_alt"), setl(v0, "limit0")],
              Probe("", (x, alt), "ADD", c.ns, "baseline, threshold declared"))
    b1 = Beat("rule", [
        _hev(c, "CONFLICT", "thr_ban", c.rng.choice(cfg["ban"]).format(e=c.name(x), limit=cfg["limit"]), 2,
             rid=c.rid("rthr"), head=_deny(d, x), body=(lit(k),), prio=c.pr(3, 2), tags=("numeric",)),
    ], Probe("", (x, alt), "CONDITION", c.ns, "threshold rule live; verdict depends on the task's parameter", ))
    b2 = Beat("probe_side", [], Probe("", (x, alt), "CONDITION", c.ns, "same rule, parameter on the other side"))
    beats = [b0, b1, b2]
    vals = cfg["values"]
    for i in range(cycles):
        beats.append(Beat("move", [setl(vals[(i + 1) % len(vals)], f"limit{i + 1}")],
                          Probe("", (x, alt), "CONDITION", c.ns,
                                f"threshold moved (cycle {i + 1}): the same parameter falls on the other side")))
        beats.append(Beat("probe_side", [],
                          Probe("", (x, alt), "CONDITION", c.ns,
                                f"same threshold, parameter on the other side (cycle {i + 1})")))
    beats.append(Beat("probe_again", [
        _hev(c, "CONFLICT", "ban_alt", c.say("ADD_BAN", e=c.name(alt)), 2,
             rid=c.rid("rbalt"), head=_deny(d, alt), prio=c.pr(5, 2)),
    ], Probe("", (x, alt), "CONDITION", c.ns, "alt banned outright; x depends on the parameter")))
    plan = MotifPlan("threshold", beats, [], (x, alt))
    plan.numeric = True
    return plan


# --------------------------------------------------------------------------
# H9  derived prohibition: the ban on x follows from a fact about y
# --------------------------------------------------------------------------

def h_derived(c: _HCtx, ents, ctxs, grp) -> MotifPlan:
    from .domains.hard_ext import DERIVED
    d, x, y, alt = c.dom, ents[0], ents[1], ents[2]
    D = DERIVED[d.key]
    facts = [Rule(c.rid("pxy"), Lit("pair", (x, y)), (), 1, session=0),
             Rule(c.rid("pyx"), Lit("pair", (y, x)), (), 1, session=0)]
    b0 = Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, y, "allow_y"), _allow_ev(c, alt, "allow_alt")],
              Probe("", (x, y, alt), "ADD", c.ns, "baseline"))
    b1 = Beat("pairrule", [
        _hev(c, "ADD", "pairrule", D["PAIR"].format(x=c.name(x), y=c.name(y)), 3,
             rid=c.rid("rpair"), head=Lit(d.allow, ("?A",), neg=True),
             body=(Lit("pair", ("?A", "?B")), Lit("active", ("?B",))), prio=c.pr(5, 3),
             tags=("derived",)),
    ], Probe("", (x, y, alt), "ADD", c.ns, "rule stated but nothing is active yet"))
    b2 = Beat("activate", [
        _hev(c, "ADD", "active_y", D["ON"].format(y=c.name(y)), 2,
             rid=c.rid("acty"), head=Lit("active", (y,)), prio=c.pr(1, 2), tags=("derived",)),
    ], Probe("", (x, y, alt), "CONFLICT", c.ns, "x is now banned two steps away: pair(x,y) and active(y)"))
    b3 = Beat("deactivate", [
        _hev(c, "SUPERSEDE", "inactive_y", D["OFF"].format(y=c.name(y)), 2,
             rid=c.rid("acty"), head=Lit("active", (y,), neg=True), body=(), prio=c.pr(1, 2), tags=("derived",)),
        _hev(c, "CONFLICT", "ban_alt", c.say("ADD_BAN", e=c.name(alt)), 2,
             rid=c.rid("rbalt"), head=_deny(d, alt), prio=c.pr(7, 2)),
    ], Probe("", (x, y, alt), "SUPERSEDE", c.ns, "the derived ban lapses as the alternative is banned"))
    b4 = Beat("reactivate", [
        _hev(c, "SUPERSEDE", "active_y2", D["ON"].format(y=c.name(y)), 2,
             rid=c.rid("acty"), head=Lit("active", (y,)), body=(), prio=c.pr(1, 2), tags=("derived",)),
    ], Probe("", (x, y, alt), "CONFLICT", c.ns, "derived ban is back; only y itself is licensed"))
    return MotifPlan("derived", [b0, b1, b2, b3, b4], facts, (x, y, alt))


# --------------------------------------------------------------------------
# H10  conjunctive condition: banned only when both conditions hold
# --------------------------------------------------------------------------

def h_conjunctive(c: _HCtx, ents, ctxs, grp, cycles: int = 1) -> MotifPlan:
    from .domains.hard_ext import CONJ
    d, x, alt, z = c.dom, ents[0], ents[1], ents[2]
    k1, k2 = ctxs[0], ctxs[1]
    beats = [Beat("open", [_allow_ev(c, x, "allow_x"), _allow_ev(c, alt, "allow_alt"),
                           _ctx_on(c, k1)],
                  Probe("", (x, alt), "ADD", c.ns, "baseline, first condition already true"))]
    beats.append(Beat("conjrule", [
        _hev(c, "CONFLICT", "conj",
             CONJ[d.key].format(e=c.name(x), c1=c.clause(k1), c2=c.clause(k2)), 3,
             rid=c.rid("rconj"), head=_deny(d, x), body=(lit(k1), lit(k2)), prio=c.pr(3, 3),
             tags=("conjunctive",)),
    ], Probe("", (x, alt), "CONDITION", c.ns, "only one condition holds, so x is still fine")))
    beats.append(Beat("second", [_ctx_on(c, k2), _allow_ev(c, z, "allow_z")],
                      Probe("", (x, alt, z), "CONDITION", c.ns,
                            "both conditions now hold: x banned")))
    beats.append(Beat("ban_alt", [
        _hev(c, "CONFLICT", "ban_alt", c.say("ADD_BAN", e=c.name(alt)), 2,
             rid=c.rid("rbalt"), head=_deny(d, alt), prio=c.pr(5, 2)),
    ], Probe("", (x, alt, z), "CONDITION", c.ns, "alt banned outright; only z is licensed")))
    for i in range(cycles):
        beats.append(Beat(f"drop1_{i}", [_ctx_off(c, k1)],
                          Probe("", (x, alt, z), "CONDITION", c.ns,
                                f"first condition lapses (cycle {i + 1}): x is fine again")))
        beats.append(Beat(f"back1_{i}", [_ctx_on(c, k1)],
                          Probe("", (x, alt, z), "CONDITION", c.ns,
                                f"both hold again (cycle {i + 1}): x banned")))
        beats.append(Beat(f"drop2_{i}", [_ctx_off(c, k2)],
                          Probe("", (x, alt, z), "CONDITION", c.ns,
                                f"second condition lapses (cycle {i + 1}): x is fine again")))
        beats.append(Beat(f"back2_{i}", [_ctx_on(c, k2)],
                          Probe("", (x, alt, z), "CONDITION", c.ns,
                                f"both hold again (cycle {i + 1}): x banned")))
    return MotifPlan("conjunctive", beats, [], (x, alt, z))


HARD_MOTIFS = {
    # name -> (builder, entities, contexts, groups)
    "ctx_flipflop": (h_ctx_flipflop, 3, 1, 0),
    "alias": (h_alias, 3, 0, 0),
    "group_dynamics": (h_group_dynamics, 4, 1, 1),
    "hierarchy": (h_hierarchy, 3, 0, 0),
    "stale_reminder": (h_stale_reminder, 2, 0, 0),
    "reinstate_arc": (h_reinstate_arc, 3, 1, 0),
    "proposal_noise": (h_proposal_noise, 2, 0, 0),
    "threshold": (h_threshold, 2, 0, 0),
    "derived": (h_derived, 3, 0, 0),
    "conjunctive": (h_conjunctive, 3, 2, 0),
}
