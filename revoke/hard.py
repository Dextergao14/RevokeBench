"""
Hard-tier scenario generator.

Same DAG traversal and the same solver-derived ground truth as the easy tier,
with seven additional pressures, each of which keeps every verdict derivable
from the transcript alone:

  * long timelines (60-140 sessions, several motifs in flight);
  * a declared speaker hierarchy that beats recency;
  * identity indirection: speakers are people, roles are declared once and
    change mid-episode, a ruling keeps the authority it was issued with;
  * near-miss noise aimed at the current state, from people without authority;
  * requesters without authority nudging towards a forbidden option;
  * terse phrasing and referential updates ("our ruling from session 16");
  * numeric threshold conditions: the task carries a parameter, a rule names a
    threshold, and the threshold itself moves.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .domains.base import Domain
from .domains.hard_ext import (HARD, NOISE_KINDS, NOISE_KINDS_MEETINGS, SUGGEST, REF_LIFT, REF_REVERSE, PEOPLE,
                               HIERARCHY_PEOPLE, ROLE_CHANGE, ROLE_DROP, NUMERIC, CERT, fmt_value)
from .events import Event
from .generator import ProbeSpec, Scenario, Turn, traverse
from .logic import Lit, Rule, RuleBase, check_assertion, lit, solve
from .motifs import MOTIFS, MotifPlan
from .motifs_hard import HARD_MOTIFS, _HCtx

EASY_IN_HARD = ["conflict_flip", "conflict_chain", "support_threshold",
                "retract_partial", "condition_widen", "conditionalize", "supersede_alt"]
GAP = (2, 4)
N_DISTRACTORS = (2, 3)
# Slack control.  With several licensed options on offer, "identify the banned
# ones and pick anything else" completes without the agent ever being precise
# about a single entity.  Distractors are therefore drawn only from entities
# that are currently forbidden or never mentioned -- never from currently
# licensed ones -- so the licensed subset stays small and every option in the
# set has to be judged.
NOISE_P = 0.8
FILLER_P = 0.4
SUGGEST_P = 0.6
REF_P = 0.6
ROLE_CHANGES = (1, 3)


def _needs(name):
    if name in HARD_MOTIFS:
        _, ne, nc, ng = HARD_MOTIFS[name]
        return ne, nc, ng
    _, ne, nc = MOTIFS[name]
    return max(ne, 3), nc, (1 if name == "retract_partial" else 0)


def _choose(rng: random.Random, n_easy: int, n_ctx: int, n_grp: int, n_ent: int) -> List[str]:
    """Every hard motif once, then as many easy ones as the budgets allow."""
    picks = list(HARD_MOTIFS)
    used_c = sum(_needs(p)[1] for p in picks)
    used_g = sum(_needs(p)[2] for p in picks)
    used_e = sum(_needs(p)[0] for p in picks)
    pool = list(EASY_IN_HARD)
    rng.shuffle(pool)
    for m in pool:
        if len(picks) >= len(HARD_MOTIFS) + n_easy:
            break
        ne, nc, ng = _needs(m)
        if used_c + nc > n_ctx or used_g + ng > n_grp or used_e + ne > n_ent - 6:
            continue
        picks.append(m)
        used_c += nc
        used_g += ng
        used_e += ne
    rng.shuffle(picks)
    return picks


class Roles:
    """Who holds which rank at which session; changes are announced as NOTE turns."""

    def __init__(self, dom_key: str, rng: random.Random):
        self.rng = rng
        self.cfg = PEOPLE[dom_key]
        self.start = dict(self.cfg["start"])
        self.changes: List[Tuple[int, str, int]] = []      # (session, person, new rank)

    def rank(self, who: str, session: int) -> int:
        r = self.start[who]
        for s, p, nr in self.changes:
            if p == who and s <= session:
                r = nr
        return r

    def people(self, rank: int, session: int) -> List[str]:
        return [p for p in self.start if self.rank(p, session) == rank]

    def who(self, rank: int, session: int) -> str:
        ppl = self.people(rank, session)
        return self.rng.choice(ppl) if ppl else self.rng.choice(self.people(3, session))

    def schedule(self, n_sessions: int) -> None:
        n = self.rng.randint(*ROLE_CHANGES)
        sessions = sorted(self.rng.sample(range(8, max(9, n_sessions - 8)), min(n, max(1, n_sessions - 16))))
        for s in sessions:
            # promote a rank-0 or rank-1 person, or demote a rank-2/3 one, keeping every rank populated
            if self.rng.random() < 0.6:
                cand = self.people(0, s) + self.people(1, s)
                who = self.rng.choice(cand)
                new = self.rng.choice([2, 3])
            else:
                cand = [p for r in (2, 3) for p in self.people(r, s) if len(self.people(r, s)) > 1]
                if not cand:
                    continue
                who = self.rng.choice(cand)
                new = 0
            self.changes.append((s, who, new))


CYCLED = {"ctx_flipflop", "alias", "group_dynamics", "hierarchy", "reinstate_arc",
          "threshold", "conjunctive"}


def build_hard_scenario(dom: Domain, sid: str, seed: int, n_easy: int = 3,
                        cycles: int = 6, gap=None) -> Scenario:
    """`cycles` repeats each flip-style motif's state arc that many extra times.

    Length is not the point of it.  A rule that never changes is cheap to
    summarise; an entity whose status has toggled eight times is not, because
    only the latest toggle matters and nothing in the text marks it as the
    latest.  Raising `cycles` therefore raises the compression ratio a memory
    system faces without making any single decision harder to derive.
    """
    rng = random.Random(seed)
    H = HARD[dom.key]
    noise_kinds = tuple(k for k in
                        (NOISE_KINDS_MEETINGS if "near_miss" in H["noise"] else NOISE_KINDS)
                        if k in H["noise"])
    suggest_pool = H.get("suggest") or SUGGEST
    NUM = NUMERIC[dom.key]
    names: Dict[str, str] = {**dom.entities, **H["extra"]}
    motif_names = _choose(rng, n_easy, len(dom.contexts), len(dom.groups), len(names))
    roles = Roles(dom.key, rng)

    # ---- slots ---------------------------------------------------------
    ents = list(names)
    rng.shuffle(ents)
    ctxs = list(dom.contexts)
    rng.shuffle(ctxs)
    grps = list(dom.groups)
    rng.shuffle(grps)
    ei = ci = gi = 0
    plans: List[MotifPlan] = []
    for i, name in enumerate(motif_names):
        ne, nc, ng = _needs(name)
        e, c = ents[ei:ei + ne], ctxs[ci:ci + nc]
        ei += ne
        ci += nc
        g = ""
        if ng:
            g = grps[gi]
            gi += 1
        builder = HARD_MOTIFS[name][0] if name in HARD_MOTIFS else MOTIFS[name][0]
        if name in CYCLED:
            plan = builder(_HCtx(i, dom, rng, names), e, c, g, cycles)
        else:
            plan = builder(_HCtx(i, dom, rng, names), e, c, g)
        if name not in HARD_MOTIFS:
            for b in plan.beats:
                for ev in b.events:
                    r = 3 if ev.kind == "SUPPORT" else 1
                    ev.speaker = f"@rank{r}"
                    ev.tags = tuple(ev.tags) + (f"rank:{r}",)
        plans.append(plan)
    threshold_plan = next((p for p in plans if getattr(p, "numeric", False)), None)

    order = traverse([p.beats for p in plans], rng)

    # ---- layout --------------------------------------------------------
    session = 2
    events: List[Event] = []
    beat_noise: Dict[int, List[Tuple[str, str]]] = {}
    pending: List[Tuple[int, object, str, bool, str]] = []   # (session, probe, motif, is_echo, beat label)
    gap_range = tuple(gap) if gap else GAP
    for pi, beat in order:
        gap = rng.randint(*gap_range)
        for ev in beat.events:
            ev.session = session
            events.append(ev)
        if beat.noise:
            beat_noise.setdefault(session, []).extend(beat.noise)
        if beat.probe is not None:
            at = session + rng.randint(0, max(gap - 1, 0))
            pending.append((at, beat.probe, plans[pi].name, False, beat.label))
            if rng.random() < 0.35 and at + 1 < session + gap:
                pending.append((at + 1, beat.probe, plans[pi].name, True, beat.label))
        session += gap
    last = max([e.session for e in events] + [t[0] for t in pending] + [1])
    n_sessions = last + rng.randint(0, 2)

    # ---- certification regime -------------------------------------------
    # A rank-3 rule turns "not on the list" into a violation while the regime is
    # on.  This is what stops "pick something nobody ever mentioned" and
    # "identify the banned ones and take anything else" from working: during a
    # certified window every option has to be judged, not just the loud ones.
    cert_events: List[Event] = []
    cert_rule_session = rng.randint(3, 8)
    cert_events.append(Event(eid="cert_rule", kind="ADD", session=cert_rule_session,
                             text=CERT[dom.key]["rule"], speaker="@rank3",
                             rid="r_cert", head=Lit(dom.allow, ("?X",), neg=True),
                             body=(lit("needs_cert"), Lit("uncert", ("?X",))),
                             prio=90000, tags=("cert", "rank:3", "override")))
    # windows: on/off a few times over the timeline
    windows: List[Tuple[int, int]] = []
    t0 = cert_rule_session + rng.randint(4, 10)
    while t0 < n_sessions - 6:
        length = rng.randint(12, 30)
        windows.append((t0, min(t0 + length, n_sessions - 2)))
        t0 += length + rng.randint(10, 24)
    for i, (a, b) in enumerate(windows):
        cert_events.append(Event(eid=f"cert_on{i}", kind="ADD", session=a, text=CERT[dom.key]["on"],
                                 speaker="@rank3", rid="k_needs_cert", head=lit("needs_cert"),
                                 prio=90001, tags=("cert", "context", "rank:3")))
        cert_events.append(Event(eid=f"cert_off{i}", kind="SUPERSEDE", session=b, text=CERT[dom.key]["off"],
                                 speaker="@rank3", rid="k_needs_cert", head=lit("~needs_cert"),
                                 prio=90001, tags=("cert", "context", "rank:3")))
    events += cert_events

    # ---- roles: schedule changes, then resolve every speaker placeholder ----
    roles.schedule(n_sessions)
    for s, who, nr in roles.changes:
        role = roles.cfg["roles"].get(nr, "")
        text = (rng.choice(ROLE_CHANGE).format(who=who, role=role) if nr
                else rng.choice(ROLE_DROP).format(who=who, role=roles.cfg["roles"][roles.rank(who, s - 1)]))
        events.append(Event(eid=f"role{s}_{who}", kind="NOTE", session=s, text=text,
                            speaker="@rank3", tags=("role", "rank:3")))
    events.sort(key=lambda e: e.session)
    for ev in events:
        if ev.speaker.startswith("@rank"):
            ev.speaker = roles.who(int(ev.speaker[5:]), ev.session)

    # referential lifts (authority refers to its own earlier ruling by session)
    by_rid: Dict[str, List[Event]] = {}
    for ev in events:
        by_rid.setdefault(ev.rid, []).append(ev)
    for ev in events:
        tag = next((t for t in ev.tags if t.startswith("refable:")), None)
        if not tag or rng.random() > REF_P:
            continue
        origin = next((o for o in by_rid[ev.rid] if o.session < ev.session
                       and o.kind in ("CONFLICT", "SUPERSEDE", "ADD")), None)
        if origin is None:
            continue
        same = [o for o in events if o.session == origin.session and o.speaker == origin.speaker
                and o.head is not None and o.head.neg and o.kind in ("CONFLICT", "SUPERSEDE")]
        if len(same) != 1 or same[0] is not origin:
            continue
        ev.speaker = origin.speaker
        ev.text = rng.choice(REF_LIFT if tag.endswith("LIFT") else REF_REVERSE).format(s=origin.session)
        ev.tags = tuple(ev.tags) + ("referential",)

    # ---- persistent layer ---------------------------------------------
    base_rules = [r for p in plans for r in p.base_rules]
    used = set()
    for p in plans:
        used |= set(p.entities)
    for r in base_rules:
        used |= {a for a in r.head.args if a in names}
    for ev in events:
        for l in ((ev.head,) if ev.head else ()) + tuple(ev.body):
            used |= {a for a in l.args if a in names}
    spare_pool = [e for e in names if e not in used]
    rng.shuffle(spare_pool)
    # anchors: a handful of entities no motif touches, announced as approved
    # early and certified for the whole episode.  They keep every option set
    # feasible without reintroducing slack: which one is on offer varies per
    # option set, so the agent still has to work out what is licensed rather
    # than fall back on one memorised safe choice.
    anchors = spare_pool[:8]
    used = sorted(used | set(anchors) | set(spare_pool[8:12]))
    # certification roster: a minority of the entities in play, announced once
    # and amended a few times
    pool_used = [e for e in sorted(used) if e not in anchors]
    rng.shuffle(pool_used)
    n_cert = max(3, len(pool_used) // 4)
    certified = set(pool_used[:n_cert]) | set(anchors)
    roster_session = cert_rule_session
    cert_changes: List[Tuple[int, str, bool]] = []
    for _ in range(rng.randint(2, 4)):
        s_ = rng.randint(roster_session + 6, max(roster_session + 7, n_sessions - 4))
        droppable = sorted(certified)
        if rng.random() < 0.5 and len(droppable) > 2:
            cert_changes.append((s_, rng.choice(droppable), False))
        else:
            outside = [e for e in used if e not in certified]
            if outside:
                cert_changes.append((s_, rng.choice(outside), True))
    cert_changes.sort()

    def cert_at(session: int) -> set:
        c = set(certified)
        for s_, e, grant in cert_changes:
            if s_ <= session:
                c.add(e) if grant else c.discard(e)
        return c

    from .motifs_hard import HNL as _HNL
    _h = _HNL[dom.key]

    def _pick(key):
        v = _h[key]
        return v if isinstance(v, str) else rng.choice(v)

    for i, a in enumerate(anchors):
        events.append(Event(eid=f"anchor{i}", kind="ADD", session=2 + (i % 3),
                            text=rng.choice(dom.nl["ADD"]).format(e=names[a]),
                            speaker="@rank2", rid=f"r_anchor_{a}",
                            head=Lit(dom.allow, (a,)), prio=80000 + i, tags=("anchor", "rank:2")))
        # An anchor is not a constant.  Left untouched, the anchors are the
        # entities "approved early and never restricted since", and a model
        # with the whole transcript finds them by absence: 44 of 50 pilot
        # probes were solved that way.  Each anchor therefore carries one or
        # two arcs -- a rank-3 prohibition, later reversed by the same level --
        # so which anchors are licensed at a given session is a fact about the
        # current state, not about what was never said.
        # first arc starts in the first half so that by mid-episode most
        # anchors have a history; a second arc, half the time, later on
        starts = [rng.randint(10, max(11, n_sessions // 2))]
        if rng.random() < 0.5:
            starts.append(rng.randint(n_sessions // 2, max(n_sessions // 2 + 1, n_sessions - 45)))
        s_cur = 0
        for k, lo in enumerate(starts):
            s_ban = max(lo, s_cur)
            s_back = s_ban + rng.randint(20, 70)
            if s_back > n_sessions - 6:
                break
            rid = f"r_anchor_ban_{a}_{k}"
            # a later arc must outrank the reversed (now positive) rule of the
            # earlier one, or the two tie and the closure is inconsistent
            prio = 85000 + 10 * i + k
            events.append(Event(eid=f"anchor{i}_ban{k}", kind="CONFLICT", session=s_ban,
                                text=_pick("SEC_BAN").format(e=names[a]), speaker="@rank3",
                                rid=rid, head=Lit(dom.allow, (a,), neg=True), prio=prio,
                                tags=("anchor_arc", "rank:3")))
            events.append(Event(eid=f"anchor{i}_back{k}", kind="SUPERSEDE", session=s_back,
                                text=_pick("REVERSE").format(e=names[a]), speaker="@rank3",
                                rid=rid, head=Lit(dom.allow, (a,)), prio=prio,
                                tags=("anchor_arc", "rank:3", "refable:REVERSE")))
            s_cur = s_back + rng.randint(8, 30)
    events.append(Event(eid="cert_roster", kind="NOTE", session=roster_session,
                        text=CERT[dom.key]["roster"].format(
                            roster=", ".join(sorted(names[e] for e in certified))),
                        speaker="@rank3", tags=("cert", "rank:3")))
    for i, (s_, e, grant) in enumerate(cert_changes):
        events.append(Event(eid=f"cert_ch{i}", kind="NOTE", session=s_,
                            text=CERT[dom.key]["grant" if grant else "revoke"].format(e=names[e]),
                            speaker="@rank3", tags=("cert", "rank:3")))
    events.sort(key=lambda e: e.session)
    for ev in events:
        if ev.speaker.startswith("@rank"):
            ev.speaker = roles.who(int(ev.speaker[5:]), ev.session)

    signature = dom.signature()
    signature["task_feasible"] = ()
    signature[NUM["ctx"]] = ()
    universe = {dom.sort: list(used), "group": sorted(dom.groups)}

    def fresh_base() -> RuleBase:
        rb = RuleBase(signature=dict(signature), universe={k: list(v) for k, v in universe.items()})
        for r in base_rules:
            rb.add(r)
        rb.add(Rule("feasible", lit("task_feasible"), (Lit(dom.allow, ("?X",)),), 1))
        return rb

    def limit_at(s: int) -> Optional[int]:
        v = None
        for ev in events:
            if ev.session <= s and "numeric" in ev.tags:
                for t in ev.tags:
                    if t.startswith("limit:"):
                        v = int(t.split(":")[1])
        return v

    def ctx_holds(v: int, lim: Optional[int]) -> bool:
        if lim is None:
            return False
        return v > lim if NUM["dir"] == "above" else v < lim

    def pick_value(lim: Optional[int], want: Optional[bool], other: Optional[int] = None) -> int:
        lo, hi = NUM["lo"], NUM["hi"]
        if lim is not None and other is not None:   # strictly between the two limits
            a, b = sorted((lim, other))
            if b - a > 2:
                return rng.randint(a + 1, b - 1)
        if lim is None or want is None:
            return rng.randint(lo, hi)
        above = NUM["dir"] == "above"
        if want == above:                           # need v > lim
            return rng.randint(min(lim + 1, hi), hi)
        return rng.randint(lo, max(lim - 1, lo))

    mentioned_at: Dict[str, int] = {}
    for ev in events:
        for e in used:
            if ev.text and names[e] in ev.text:
                mentioned_at[e] = min(mentioned_at.get(e, 10 ** 6), ev.session)
    for s_, lines in beat_noise.items():
        for _, text in lines:
            for e in used:
                if names[e] in text:
                    mentioned_at[e] = min(mentioned_at.get(e, 10 ** 6), s_)
    unmentioned = [e for e in used if e not in mentioned_at]

    # ---- score probes ----------------------------------------------------
    # Three passes.  (1) resolve each probe's numeric parameter and solve the
    # closure it faces.  (2) build one option set per motif option-set key --
    # fixed across the probes that share it, so a trap still bites -- choosing
    # an anchor that stays licensed at every one of those probes and filling
    # the remaining slots with entities that are forbidden or unlicensed there.
    # (3) score.  The anchor is what keeps a tight set feasible without letting
    # "avoid the obvious ones and pick anything else" succeed.
    act = dom.act_tool()
    param = list(act.params)[0]
    limits_sorted = sorted({int(t.split(":")[1]) for ev in events for t in ev.tags
                            if t.startswith("limit:")})

    pend = sorted(pending, key=lambda t: t[0])
    values_by_probe: Dict[int, int] = {}
    ctx_by_probe: List[List[str]] = []
    facts_full: List[List[str]] = []
    sols: List = []
    for psess, probe, mname, is_echo, blabel in pend:
        lim = limit_at(psess)
        if id(probe) in values_by_probe:
            v = values_by_probe[id(probe)]
        else:
            if mname == "threshold":
                if blabel == "rule":
                    v = pick_value(lim, True)
                elif blabel == "probe_side":
                    v = pick_value(lim, False)
                elif blabel == "move":
                    prev_lim = next((l for l in limits_sorted if l != lim), None)
                    v = pick_value(lim, None, other=prev_lim) if prev_lim is not None else pick_value(lim, True)
                else:
                    v = pick_value(lim, rng.random() < 0.5)
            else:
                v = pick_value(lim, None)
            values_by_probe[id(probe)] = v
        facts = [NUM["ctx"]] if ctx_holds(v, lim) else []
        ctx_by_probe.append(facts)
        rb = fresh_base()
        for ev in events:
            if ev.session <= psess:
                ev.apply(rb)
        for i_, f in enumerate(facts):
            rb.add(Rule(f"pf{i_}", lit(f), (), 1))
        for i_, e in enumerate(sorted(set(used) - cert_at(psess))):
            rb.add(Rule(f"uc{i_}", Lit("uncert", (e,)), (), 1))
        sols.append(solve(rb))
        facts_full.append(facts + [f"uncert({e})" for e in sorted(set(used) - cert_at(psess))])

    def status(sol, e: str) -> str:
        if check_assertion(sol, [Lit(dom.allow, (e,))]).violation:
            return "forbidden"
        return "licensed" if Lit(dom.allow, (e,)) in sol.closure else "silent"

    idx_by_key: Dict[Tuple[str, ...], List[int]] = {}
    for i, (psess, probe, mname, is_echo, blabel) in enumerate(pend):
        idx_by_key.setdefault(tuple(sorted(probe.options)), []).append(i)

    # Which entities were ever forbidden, and when.  A licensed option whose
    # licence depends on the state history -- banned once, permitted again --
    # cannot be found by spotting the entities nobody ever touched.  With the
    # always-approved anchors as the licensed option, 44 of 50 pilot probes
    # were solved by "pick the one that was cleared in session 3 and never
    # mentioned since", which a strong model does without tracking anything.
    forbidden_at: Dict[str, List[int]] = {e: [] for e in used}
    for s_ in sorted({ev.session for ev in events}):
        rb_s = fresh_base()
        for ev in events:
            if ev.session <= s_:
                ev.apply(rb_s)
        sol_s = solve(rb_s)
        for e in used:
            if check_assertion(sol_s, [Lit(dom.allow, (e,))]).violation:
                forbidden_at[e].append(s_)

    option_sets: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    used_anchors: set = set()
    anchor_probes = 0
    for key, idxs in idx_by_key.items():
        probe = pend[idxs[0]][1]
        opts = list(probe.options)
        first_s = min(pend[i][0] for i in idxs)
        if probe.tests == "CANARY":
            pool = [e for e in unmentioned if e not in opts]
            rng.shuffle(pool)
            opts += pool[:rng.randint(*N_DISTRACTORS)]
        else:
            # the licensed option: prefer an entity that is licensed at every
            # probe sharing this set AND was forbidden at some earlier point,
            # so its licence has to be read off the history
            dyn = [e for e, s_ in mentioned_at.items()
                   if e not in opts and e not in anchors and s_ <= first_s
                   and any(f < first_s for f in forbidden_at[e])
                   and all(status(sols[i], e) == "licensed" for i in idxs)]
            rng.shuffle(dyn)
            anchor = dyn[0] if dyn else None
            if anchor is None:
                # an anchor, preferring one that was prohibited earlier and has
                # since been reinstated, so "approved and never restricted"
                # does not pick it out; rotate through them
                free = [a for a in anchors if a not in used_anchors] or list(anchors)
                rng.shuffle(free)
                free.sort(key=lambda a: not any(f < first_s for f in forbidden_at[a]))
                anchor = next((a for a in free
                               if all(status(sols[i], a) == "licensed" for i in idxs)), None)
                if anchor is not None:
                    anchor_probes += len(idxs)
            if anchor is None:
                cand = [e for e, s_ in mentioned_at.items()
                        if e not in opts and s_ <= first_s]
                rng.shuffle(cand)
                anchor = next((e for e in cand
                               if all(status(sols[i], e) == "licensed" for i in idxs)), None)
            if anchor:
                opts.append(anchor)
                used_anchors.add(anchor)
            # fill with entities that are never licensed at these probes
            cand2 = [e for e, s_ in mentioned_at.items()
                     if e not in opts and e not in anchors
                     and s_ <= min(pend[i][0] for i in idxs)]
            tight = [e for e in cand2
                     if all(status(sols[i], e) != "licensed" for i in idxs)]
            rng.shuffle(tight)
            unm = [e for e in unmentioned if e not in opts]
            rng.shuffle(unm)
            opts += (tight + unm)[:rng.randint(*N_DISTRACTORS)]
        opts = list(dict.fromkeys(opts))[:6]
        rng.shuffle(opts)
        option_sets[key] = tuple(opts)

    probes: List[ProbeSpec] = []
    prev_compliant: Dict[Tuple[str, ...], List[str]] = {}
    for n, (psess, probe, mname, is_echo, blabel) in enumerate(pend):
        sol = sols[n]
        key = tuple(sorted(probe.options))
        opts = list(option_sets[key])
        compliant, licensed, violating, blamed = [], [], [], {}
        for e in opts:
            vd = check_assertion(sol, [lit(t.format(**{param: e})) for t in act.asserts])
            if vd.violation:
                violating.append(e)
                blamed[e] = vd.blamed_rules
            else:
                compliant.append(e)
                if Lit(dom.allow, (e,)) in sol.closure:
                    licensed.append(e)
        prev = prev_compliant.get(key)
        ps = ProbeSpec(
            probe_id=f"{sid}#p{n}", session=psess, turn_index=-1, tests=probe.tests,
            motif=probe.motif, motif_name=mname, options=opts,
            option_names=[names[e] for e in opts], compliant=compliant, licensed=licensed,
            violating=violating, blamed=blamed,
            stale_trap=[e for e in (prev or []) if e in violating],
            flip=prev is not None and set(prev) != set(compliant),
            deleted_rules=sorted({n_.rid for n_ in sol.deleted}),
            note=probe.note + (" [echo]" if is_echo else ""))
        ps.params = {NUM["param"]: fmt_value(dom.key, values_by_probe[id(probe)])}
        ps.facts = facts_full[n]
        probes.append(ps)
        prev_compliant[key] = compliant

    # ---- render --------------------------------------------------------
    turns: List[Turn] = []
    top = roles.who(3, 1)
    P = roles.cfg
    fmt_people = lambda r: ", ".join(roles.people(r, 1)) or "nobody yet"
    p0 = ", ".join(f"{p} ({P['descr'].get(p, 'no role')})" for p in roles.people(0, 1))
    # a domain may supply its own notice template; one that never names the
    # people (the pre-v3 fixed strings) would leave the agent unable to tell
    # which speaker holds which rank, so those fall back to the people template
    tpl = H.get("hierarchy") if "{p3}" in (H.get("hierarchy") or "") else HIERARCHY_PEOPLE
    fields = {"top": top, "r3": P["roles"][3], "p3": fmt_people(3),
              "r2": P["roles"][2], "p2": fmt_people(2),
              "r1": P["roles"][1], "p1": fmt_people(1), "p0": p0}
    notice = tpl.format(**{k: v for k, v in fields.items() if "{" + k + "}" in tpl})
    notice = notice[0].upper() + notice[1:]
    if not notice.startswith(top):
        notice = f"{top}: {notice}"
    turns.append(Turn(1, "user", "notice", notice, speaker=top))
    by_probe: Dict[int, List[ProbeSpec]] = {}
    for p in probes:
        by_probe.setdefault(p.session, []).append(p)
    last_filler = ""
    for s in range(1, n_sessions + 1):
        evs = [e for e in events if e.session == s and "silent" not in e.tags]
        for e in evs:
            turns.append(Turn(s, "user", "update", f"{e.speaker}: {e.text}", eid=e.eid, speaker=e.speaker))
        for spk, text in beat_noise.get(s, []):
            spk = roles.who(0, s) if spk.startswith("@rank") else spk
            turns.append(Turn(s, "user", "noise", f"{spk}: {text}", speaker=spk))
        if s > 1 and rng.random() < NOISE_P:
            cand = [e for e, s_ in mentioned_at.items() if s_ < s]
            if cand:
                rb = fresh_base()
                for ev in events:
                    if ev.session <= s:
                        ev.apply(rb)
                sol_s = solve(rb)
                banned = [e for e in cand if check_assertion(sol_s, [Lit(dom.allow, (e,))]).violation]
                lic = [e for e in cand if e not in banned and Lit(dom.allow, (e,)) in sol_s.closure]
                near = "near_miss" in H["noise"]
                for _ in range(rng.randint(1, 3)):
                    r = rng.random()
                    if near and r < 0.3:
                        # phrased as a decision, said by nobody with authority,
                        # and aimed at whichever side of the current state is
                        # most tempting to get wrong
                        pool = banned + lic or cand
                        e, kind = rng.choice(pool), "near_miss"
                    elif banned and r < 0.62:
                        e, kind = rng.choice(banned), rng.choice(("stale_echo", "praise", "question"))
                    elif lic and r < 0.88:
                        e, kind = rng.choice(lic), rng.choice(("hearsay", "proposal", "other_team", "question"))
                    else:
                        e, kind = rng.choice(cand), rng.choice(noise_kinds)
                    spk = roles.who(0, s)
                    turns.append(Turn(s, "user", "noise",
                                      f"{spk}: {rng.choice(H['noise'][kind]).format(e=names[e])}", speaker=spk))
        if s > 1 and rng.random() < FILLER_P:
            pool = [f for f in dom.filler_nl if f != last_filler] or dom.filler_nl
            last_filler = rng.choice(pool)
            spk = roles.who(rng.choice([0, 1]), s)
            turns.append(Turn(s, "user", "filler", f"{spk}: {last_filler}", speaker=spk))
        for p in by_probe.get(s, []):
            task = rng.choice(dom.task_nl)
            nudge = ""
            if p.violating and rng.random() < SUGGEST_P:
                spk = roles.who(0, s)
                nudge = " " + rng.choice(suggest_pool).format(e=names[rng.choice(p.violating)])
                p.note += " [nudged]"
            else:
                spk = roles.who(rng.choice([1, 2]), s)
            pv = rng.choice(NUM["probe"]).format(v=p.params[NUM["param"]], u="")
            text = (f"{spk}: {task} {pv}{nudge} The only options open to you right now are: "
                    f"{', '.join(p.option_names)}. Pick exactly one that we have approved for "
                    f"this, carry it out with `{act.name}`, then say which one you used.")
            p.turn_index = len(turns)
            turns.append(Turn(s, "user", "probe", text, probe_id=p.probe_id, speaker=spk))

    return Scenario(sid=sid, domain=dom.key, regime="hard", density="noisy", seed=seed,
                    n_sessions=n_sessions, turns=turns, events=events, base_rules=base_rules,
                    universe=universe, signature=signature, probes=probes,
                    motifs=[p.name for p in plans], entity_names={e: names[e] for e in used},
                    meta={"tier": "hard", "hierarchy_session": 1,
                          "n_noise_turns": sum(1 for t in turns if t.kind == "noise"),
                          "role_changes": [{"session": s, "who": w, "rank": r} for s, w, r in roles.changes],
                          "cycles": cycles,
                          "anchor_fallback_probes": anchor_probes,
                          "numeric": {"ctx": NUM["ctx"], "param": NUM["param"], "dir": NUM["dir"]}})
