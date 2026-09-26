"""
REVOKE two-layer constraint engine.

Persistent layer : defeasible rules with explicit integer priorities.
                   Append-only except for SUPERSEDE / RETRACT.
Derivation layer : recomputed from scratch at every timestep --
                   grounding -> defeat graph -> grounded extension ->
                   stratified forward chaining -> current closure C_t.

Nothing in the derivation layer is ever stored, so grading never depends on
deletion order.  With strictly-ordered priorities the defeat graph is acyclic,
hence the grounded extension is unique and computed in polynomial time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Literals
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Lit:
    """A (possibly non-ground) literal.  `neg` is *classical* negation."""

    pred: str
    args: Tuple[str, ...] = ()
    neg: bool = False

    @property
    def is_ground(self) -> bool:
        return not any(a.startswith("?") for a in self.args)

    def complement(self) -> "Lit":
        return Lit(self.pred, self.args, not self.neg)

    def subst(self, theta: Dict[str, str]) -> "Lit":
        return Lit(self.pred, tuple(theta.get(a, a) for a in self.args), self.neg)

    def vars(self) -> Tuple[str, ...]:
        return tuple(a for a in self.args if a.startswith("?"))

    def __str__(self) -> str:
        core = f"{self.pred}({','.join(self.args)})" if self.args else self.pred
        return ("~" if self.neg else "") + core


def lit(spec: str) -> Lit:
    """Parse ``"~ok(A)"`` / ``"deploy_ok"`` / ``"blocked(?V,q3)"`` into a Lit."""
    spec = spec.strip()
    neg = spec.startswith("~")
    if neg:
        spec = spec[1:].strip()
    if "(" in spec:
        pred, rest = spec.split("(", 1)
        args = tuple(a.strip() for a in rest.rstrip(")").split(",") if a.strip())
    else:
        pred, args = spec, ()
    return Lit(pred.strip(), args, neg)


# --------------------------------------------------------------------------
# Persistent layer
# --------------------------------------------------------------------------


@dataclass
class Rule:
    """A defeasible rule ``head <= body`` carrying an explicit priority."""

    rid: str
    head: Lit
    body: Tuple[Lit, ...] = ()
    prio: int = 1
    origin: str = ""          # id of the event that last wrote this rule
    session: int = 0          # session at which the rule entered the base
    # ground argument tuples of *the head* whose instances are withdrawn
    retracted: FrozenSet[Tuple[str, ...]] = field(default_factory=frozenset)
    alive: bool = True

    def __str__(self) -> str:
        b = " & ".join(str(x) for x in self.body) or "T"
        return f"{self.rid}: {self.head} <= {b}  [pi={self.prio}]"


@dataclass
class RuleBase:
    """The persistent layer plus the sort signature needed for grounding."""

    rules: Dict[str, Rule] = field(default_factory=dict)
    # predicate -> tuple of sort names, one per argument position
    signature: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # sort name -> list of constants
    universe: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, rule: Rule) -> None:
        self.rules[rule.rid] = rule

    def live(self) -> List[Rule]:
        return [r for r in self.rules.values() if r.alive]

    def copy(self) -> "RuleBase":
        return RuleBase(
            rules={k: replace(v) for k, v in self.rules.items()},
            signature=dict(self.signature),
            universe={k: list(v) for k, v in self.universe.items()},
        )


# --------------------------------------------------------------------------
# Derivation layer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A ground rule instance -- a node of the derivation layer."""

    nid: str
    rid: str
    head: Lit
    body: Tuple[Lit, ...]
    prio: int


def _sorts_of(rb: RuleBase, l: Lit) -> Tuple[str, ...]:
    sig = rb.signature.get(l.pred)
    if sig is None:
        raise KeyError(f"predicate {l.pred!r} missing from domain signature")
    if len(sig) != len(l.args):
        raise ValueError(f"arity mismatch for {l}: signature {sig}")
    return sig


def ground(rb: RuleBase) -> List[Node]:
    """Instantiate every live rule over the sorted Herbrand universe."""
    nodes: List[Node] = []
    for r in rb.live():
        # collect variables together with the sort forced by their position
        vsort: Dict[str, str] = {}
        for l in (r.head,) + tuple(r.body):
            for pos, a in enumerate(l.args):
                if a.startswith("?"):
                    s = _sorts_of(rb, l)[pos]
                    if vsort.setdefault(a, s) != s:
                        raise ValueError(f"variable {a} used at two sorts in {r}")
        names = sorted(vsort)
        domains = [rb.universe.get(vsort[v], []) for v in names]
        for combo in product(*domains) if names else [()]:
            theta = dict(zip(names, combo))
            head = r.head.subst(theta)
            if head.args in r.retracted:
                continue                       # RETRACT withdraws this instance
            body = tuple(b.subst(theta) for b in r.body)
            key = ",".join(f"{k}={v}" for k, v in sorted(theta.items()))
            nodes.append(Node(f"{r.rid}#{key}" if key else r.rid,
                              r.rid, head, body, r.prio))
    return nodes


def defeat_edges(nodes: Sequence[Node]) -> List[Tuple[str, str]]:
    """n -> m whenever the heads are complementary and pi(n) > pi(m)."""
    by_head: Dict[Lit, List[Node]] = {}
    for n in nodes:
        by_head.setdefault(n.head, []).append(n)
    edges: List[Tuple[str, str]] = []
    for n in nodes:
        for m in by_head.get(n.head.complement(), ()):
            if n.prio > m.prio:
                edges.append((n.nid, m.nid))
    return edges


def grounded_extension(
    nodes: Sequence[Node], edges: Sequence[Tuple[str, str]]
) -> Tuple[List[Node], List[Node]]:
    """Iterated deletion to fixpoint: survivors E, deleted D.

    A node is IN once all of its attackers are OUT, and OUT once any attacker
    is IN.  Priorities are strict, so the attack relation is acyclic and every
    node is decided; any node left undecided (a tie cycle) is reported OUT and
    will be caught by the consistency verifier.
    """
    attackers: Dict[str, List[str]] = {n.nid: [] for n in nodes}
    for a, b in edges:
        if b in attackers:
            attackers[b].append(a)
    status: Dict[str, Optional[bool]] = {n.nid: None for n in nodes}
    changed = True
    while changed:
        changed = False
        for nid, atk in attackers.items():
            if status[nid] is not None:
                continue
            st = [status[a] for a in atk]
            if all(s is False for s in st):
                status[nid], changed = True, True
            elif any(s is True for s in st):
                status[nid], changed = False, True
    keep = [n for n in nodes if status[n.nid] is True]
    drop = [n for n in nodes if status[n.nid] is not True]
    return keep, drop


def forward_chain(nodes: Iterable[Node], seed: Iterable[Lit] = ()) -> FrozenSet[Lit]:
    """Naive stratified evaluation over the surviving instances."""
    derived = set(seed)
    pool = list(nodes)
    changed = True
    while changed:
        changed = False
        for n in pool:
            if n.head not in derived and all(b in derived for b in n.body):
                derived.add(n.head)
                changed = True
    return frozenset(derived)


@dataclass
class Solution:
    """The derivation layer at one timestep."""

    closure: FrozenSet[Lit]
    extension: List[Node]
    deleted: List[Node]
    nodes: List[Node]
    converged: bool = True

    @property
    def consistent(self) -> bool:
        return not contradictions(self.closure)

    def rules_supporting(self, l: Lit) -> List[str]:
        return sorted({n.rid for n in self.extension if n.head == l})


def contradictions(closure: Iterable[Lit]) -> List[Lit]:
    """Every literal in the set whose complement is also present."""
    s = set(closure)
    return sorted(l for l in s if l.complement() in s and not l.neg)


def solve(rb: RuleBase, max_rounds: int = 32) -> Solution:
    """Recompute the whole derivation layer from the persistent layer."""
    nodes = ground(rb)
    closure: FrozenSet[Lit] = frozenset()
    keep: List[Node] = list(nodes)
    drop: List[Node] = []
    history: List[FrozenSet[Lit]] = []
    for _ in range(max_rounds):
        applicable = [n for n in nodes if all(b in closure for b in n.body)]
        app_ids = {n.nid for n in applicable}
        keep_app, drop = grounded_extension(applicable, defeat_edges(applicable))
        # instances not yet applicable stay available for later rounds
        keep = keep_app + [n for n in nodes if n.nid not in app_ids]
        new = forward_chain(keep)
        if new == closure:
            return Solution(closure, keep_app, drop, nodes, converged=True)
        if new in history:                       # oscillation -> reject upstream
            return Solution(new, keep_app, drop, nodes, converged=False)
        history.append(closure)
        closure = new
    return Solution(closure, keep, drop, nodes, converged=False)


# --------------------------------------------------------------------------
# Action grading
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    violation: bool
    conflicts: List[Tuple[Lit, Lit]] = field(default_factory=list)
    blamed_rules: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:                  # truthy == compliant
        return not self.violation


def check_assertion(sol: Solution, asserted: Sequence[Lit]) -> Verdict:
    """Violation iff ``C_t U assert(a)`` derives a contradiction.

    The asserted literals are pushed through the *surviving* instances, so a
    violation may surface several derivation steps away from the action.
    """
    after = forward_chain(sol.extension, seed=set(sol.closure) | set(asserted))
    bad: List[Tuple[Lit, Lit]] = []
    blamed: List[str] = []
    for l in sorted(after):
        c = l.complement()
        if c in after and not l.neg:
            bad.append((l, c))
            blamed += sol.rules_supporting(l) + sol.rules_supporting(c)
    return Verdict(bool(bad), bad, sorted(set(blamed)))


def entails(sol: Solution, l: Lit) -> bool:
    return l in sol.closure
