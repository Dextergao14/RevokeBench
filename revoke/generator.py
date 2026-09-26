"""
Scenario generator: a randomised traversal over the motif beat-DAG.

Each motif instance contributes a *chain* of beats; chains are mutually
unordered, so the set of all beats forms a DAG whose linear extensions are the
admissible session timelines.  ``traverse`` walks that DAG, interleaving
chains, and the result is laid onto a session grid together with filler turns
and probe turns.

Ground truth is never taken from the motif's design intent.  After the
timeline exists, every probe is re-scored against the closure actually
obtained by ``logic.solve`` at that session, so interleaving that changes the
intended semantics is caught rather than silently mis-labelled.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .domains.base import Domain
from .events import Event
from .logic import Lit, Rule, RuleBase, Solution, check_assertion, lit, solve
from .motifs import MOTIFS, FLIP_MOTIFS, Beat, MotifPlan, Probe, _Ctx


# --------------------------------------------------------------------------


@dataclass
class Turn:
    session: int
    role: str                    # "user" | "assistant_expected"
    kind: str                    # "update" | "filler" | "probe"
    text: str
    eid: str = ""
    probe_id: str = ""
    speaker: str = ""


@dataclass
class ProbeSpec:
    probe_id: str
    session: int
    turn_index: int
    tests: str
    motif: str
    motif_name: str
    options: List[str]                        # entity ids
    option_names: List[str]
    compliant: List[str]
    licensed: List[str]                       # allow(e) in C_t -- completes the task
    violating: List[str]
    blamed: Dict[str, List[str]]              # entity -> rules that make it a violation
    stale_trap: List[str]                     # options compliant at the previous probe
    flip: bool                                # compliant set changed since last probe
    deleted_rules: List[str]                  # D_t rule ids at this session
    note: str = ""
    params: Dict[str, str] = field(default_factory=dict)   # the task's own parameters (hard tier)
    facts: List[str] = field(default_factory=list)         # probe-scoped context facts derived from them


@dataclass
class Scenario:
    sid: str
    domain: str
    regime: str
    density: str
    seed: int
    n_sessions: int
    turns: List[Turn]
    events: List[Event]
    base_rules: List[Rule]
    universe: Dict[str, List[str]]
    signature: Dict[str, Tuple[str, ...]]
    probes: List[ProbeSpec]
    motifs: List[str]
    entity_names: Dict[str, str]
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------


def traverse(chains: List[List[Beat]], rng: random.Random) -> List[Tuple[int, Beat]]:
    """Randomised topological linearisation of the beat DAG.

    Chains are internally ordered and mutually independent, so any interleaving
    is a valid linear extension.  Picking the next chain proportionally to how
    many beats it still has left keeps long motifs from clumping at the end.
    """
    remaining = [list(ch) for ch in chains]
    out: List[Tuple[int, Beat]] = []
    while any(remaining):
        weights = [len(ch) for ch in remaining]
        i = rng.choices(range(len(remaining)), weights=weights)[0]
        out.append((i, remaining[i].pop(0)))
    return out


def _pick_slots(dom: Domain, plan_specs, rng: random.Random):
    ents = list(dom.entities)
    rng.shuffle(ents)
    ctxs = list(dom.contexts)
    rng.shuffle(ctxs)
    grps = list(dom.groups)
    rng.shuffle(grps)
    ei = ci = gi = 0
    out = []
    for name in plan_specs:
        _, n_e, n_c = MOTIFS[name]
        e = ents[ei:ei + max(n_e, 3)]
        ei += max(n_e, 3)
        if len(e) < n_e:
            raise ValueError("entity pool exhausted")
        c = ctxs[ci:ci + n_c]
        ci += n_c
        g = ""
        if name in GROUP_MOTIFS:
            g = grps[gi]                 # never reused across motifs
            gi += 1
        out.append((e, c, g))
    return out


GROUP_MOTIFS = {"retract_partial"}
MAX_GROUPS = 3          # every seed domain defines exactly three groups


def _choose_motifs(rng: random.Random, n: int) -> List[str]:
    picks = [rng.choice(FLIP_MOTIFS)]
    pool = list(MOTIFS)
    while len(picks) < n:
        m = rng.choice(pool)
        # cap context consumption so motifs never have to share a context
        if sum(MOTIFS[p][2] for p in picks) + MOTIFS[m][2] > 6:
            m = rng.choice([p for p in MOTIFS if MOTIFS[p][2] == 0])
        # and cap group consumption so blanket bans never cross motifs
        if m in GROUP_MOTIFS and sum(p in GROUP_MOTIFS for p in picks) >= MAX_GROUPS:
            m = rng.choice([p for p in MOTIFS
                            if p not in GROUP_MOTIFS and MOTIFS[p][2] == 0])
        picks.append(m)
    rng.shuffle(picks)
    return picks


REGIMES = {"short": (10, 18, 3, 4), "long": (22, 40, 5, 7)}
DENSITY = {"sparse": (3, 5), "dense": (1, 3)}
N_DISTRACTORS = (1, 2)      # extra options per probe, on top of the motif's own


def build_scenario(dom: Domain, sid: str, seed: int,
                   regime: str = "short", density: str = "sparse") -> Scenario:
    rng = random.Random(seed)
    smin, smax, mmin, mmax = REGIMES[regime]
    gap_lo, gap_hi = DENSITY[density]
    n_motifs = rng.randint(mmin, mmax)
    names = _choose_motifs(rng, n_motifs)
    slots = _pick_slots(dom, names, rng)

    plans: List[MotifPlan] = []
    for i, (name, (e, c, g)) in enumerate(zip(names, slots)):
        builder = MOTIFS[name][0]
        plans.append(builder(_Ctx(i, dom, rng), e, c, g))

    order = traverse([p.beats for p in plans], rng)

    # ---- lay beats onto the session grid -------------------------------
    session = 1
    events: List[Event] = []
    pending_probes: List[Tuple[int, Probe, str]] = []   # (session, probe, motif_name)
    for pi, beat in order:
        for ev in beat.events:
            ev.session = session
            events.append(ev)
        gap = rng.randint(gap_lo, gap_hi)
        if beat.probe is not None:
            # keep the probe strictly inside this beat's window, so it is
            # scored against the state the beat established
            at = session + rng.randint(0, max(gap - 1, 0))
            pending_probes.append((at, beat.probe, plans[pi].name, False))
            # echo probes on the *same* option set measure adaptation lag:
            # how many further chances the agent needs before it stops
            # replaying the invalidated choice
            for k in range(1, rng.randint(1, 3)):
                if at + k < session + gap:
                    pending_probes.append((at + k, beat.probe, plans[pi].name, True))
        session += gap
    # the timeline is as long as its content; the regime steers how many
    # motifs and how wide the gaps are, it never truncates a probe away
    last = max([e.session for e in events] +
               [t[0] for t in pending_probes] + [1])
    n_sessions = max(last + rng.randint(0, 2), smin)

    # ---- persistent-layer scaffolding ---------------------------------
    base_rules = [r for p in plans for r in p.base_rules]
    used = sorted({e for p in plans for e in p.entities} |
                  {a for r in base_rules for a in r.head.args if a in dom.entities})
    for ev in events:
        for l in ((ev.head,) if ev.head else ()) + tuple(ev.body):
            for a in l.args:
                if a in dom.entities and a not in used:
                    used.append(a)
    # a few entities that no rule ever touches: they enter the universe (so
    # they can be offered as options) but are never named by any update turn
    spare_pool = [e for e in dom.entities if e not in used]
    rng.shuffle(spare_pool)
    spare = spare_pool[:4]
    used = sorted(set(used) | set(spare))
    signature = dom.signature()
    signature["task_feasible"] = ()
    universe = dom.universe(used)

    def fresh_base() -> RuleBase:
        rb = RuleBase(signature=dict(signature),
                      universe={k: list(v) for k, v in universe.items()})
        for r in base_rules:
            rb.add(r)
        # alternative-derivation guard: feasibility must survive rule deletion
        rb.add(Rule("feasible", lit("task_feasible"),
                    (Lit(dom.allow, ("?X",)),), 1))
        return rb

    # ---- score every probe against the real closure --------------------
    # distractor options widen every choice set, so that neither a fixed
    # positional prior nor "pick something nobody ever mentioned" survives:
    # unmentioned entities are never violations, but they are not licensed
    # either, so they cannot complete the task
    # split the universe by what the transcript actually names, so a canary's
    # distractors can be guaranteed unconstrained and a cross-motif distractor
    # is always traceable to something the agent was told
    mentioned = sorted({e for e in used
                        if any(dom.entities[e] in ev.text for ev in events)})
    unmentioned = [e for e in used if e not in mentioned]

    option_sets: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    probes: List[ProbeSpec] = []
    prev_compliant: Dict[Tuple[str, ...], List[str]] = {}
    act = dom.act_tool()
    param = list(act.params)[0]
    for n, (psess, probe, mname, is_echo) in enumerate(
            sorted(pending_probes, key=lambda t: t[0])):
        rb = fresh_base()
        for ev in events:
            if ev.session <= psess:
                ev.apply(rb)
        sol = solve(rb)

        key = tuple(sorted(probe.options))
        opts = list(probe.options)
        if key in option_sets:
            opts = list(option_sets[key])          # echoes reuse the same set
        else:
            # a canary must stay genuinely unconstrained, so its distractors
            # come only from entities no rule ever mentions
            # a cross-motif distractor is only admissible once the transcript
            # has already said something about it -- otherwise a violation
            # would not be derivable from what the agent was told
            seen_by_now = {e for e in mentioned
                           if any(dom.entities[e] in ev.text and ev.session <= psess
                                  for ev in events)}
            pool = ([e for e in unmentioned if e not in opts]
                    if probe.tests == "CANARY"
                    else [e for e in list(seen_by_now) + unmentioned
                          if e not in opts])
            pool = sorted(set(pool))
            rng.shuffle(pool)
            opts += pool[:rng.randint(*N_DISTRACTORS)]
            opts = list(dict.fromkeys(opts))
            rng.shuffle(opts)
            option_sets[key] = tuple(opts)

        compliant, licensed, violating, blamed = [], [], [], {}
        for e in opts:
            lits = [lit(t.format(**{param: e})) for t in act.asserts]
            v = check_assertion(sol, lits)
            if v.violation:
                violating.append(e)
                blamed[e] = v.blamed_rules
            else:
                compliant.append(e)
                if Lit(dom.allow, (e,)) in sol.closure:
                    licensed.append(e)
        prev = prev_compliant.get(key)
        probes.append(ProbeSpec(
            probe_id=f"{sid}#p{n}", session=psess, turn_index=-1,
            tests=probe.tests, motif=probe.motif, motif_name=mname,
            options=opts, option_names=[dom.entities[e] for e in opts],
            compliant=compliant, licensed=licensed, violating=violating,
            blamed=blamed,
            stale_trap=[e for e in (prev or []) if e in violating],
            flip=prev is not None and set(prev) != set(compliant),
            deleted_rules=sorted({n_.rid for n_ in sol.deleted}),
            note=probe.note + (" [echo]" if is_echo else "")))
        prev_compliant[key] = compliant

    # ---- render turns --------------------------------------------------
    turns: List[Turn] = []
    last_filler = ""
    by_probe = {p.session: [] for p in probes}
    for p in probes:
        by_probe[p.session].append(p)
    for s in range(1, n_sessions + 1):
        evs = [e for e in events if e.session == s]
        for e in evs:
            turns.append(Turn(s, "user", "update", e.text, eid=e.eid))
        if not evs and not by_probe.get(s) and rng.random() < 0.7:
            pool = [f for f in dom.filler_nl if f != last_filler] or dom.filler_nl
            last_filler = rng.choice(pool)
            turns.append(Turn(s, "user", "filler", last_filler))
        for p in by_probe.get(s, []):
            names_ = ", ".join(p.option_names)
            task = rng.choice(dom.task_nl)
            text = (f"{task} The only options open to you right now are: "
                    f"{names_}. Pick exactly one that we have approved for this, carry it out "
                    f"with `{act.name}`, then say which one you used.")
            p.turn_index = len(turns)
            turns.append(Turn(s, "user", "probe", text, probe_id=p.probe_id))

    return Scenario(sid=sid, domain=dom.key, regime=regime, density=density,
                    seed=seed, n_sessions=n_sessions, turns=turns,
                    events=events, base_rules=base_rules, universe=universe,
                    signature=signature, probes=probes,
                    motifs=[p.name for p in plans],
                    entity_names={e: dom.entities[e] for e in used})
