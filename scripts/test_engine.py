#!/usr/bin/env python3
"""The two worked examples of the design note, as executable specification."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revoke.logic import (Rule, RuleBase, check_assertion, contradictions,
                          entails, lit, solve)

fails = []


def eq(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r}, want {want!r}")


# -- Example 1: CONFLICT resolved by priority, then flipped by SUPPORT -------
rb = RuleBase(signature={"ok": ("vendor",), "low_budget": ()},
              universe={"vendor": ["A", "B"]})
rb.add(Rule("f1", lit("low_budget"), (), 5))
rb.add(Rule("r1", lit("ok(A)"), (), 1))                       # ADD, session 2
rb.add(Rule("r2", lit("~ok(A)"), (lit("low_budget"),), 2))    # ADD, session 5

s = solve(rb)
eq("ex1 closure", sorted(map(str, s.closure)), ["low_budget", "~ok(A)"])
eq("ex1 deleted", [n.nid for n in s.deleted], ["r1"])
eq("ex1 order(A) violates", check_assertion(s, [lit("ok(A)")]).violation, True)
eq("ex1 order(B) passes", check_assertion(s, [lit("ok(B)")]).violation, False)

rb.rules["r1"].prio = 3                                        # SUPPORT, session 9
s = solve(rb)
eq("ex1 flipped closure", sorted(map(str, s.closure)), ["low_budget", "ok(A)"])
eq("ex1 flipped deleted", [n.nid for n in s.deleted], ["r2"])
eq("ex1 order(A) passes after SUPPORT",
   check_assertion(s, [lit("ok(A)")]).violation, False)

# -- Example 2: SUPERSEDE, alternative derivations, no over-deletion --------
rb = RuleBase(signature={"callable": ("ep",), "deploy_ok": ()},
              universe={"ep": ["v1", "v2"]})
rb.add(Rule("r1", lit("callable(v1)"), (), 1))
rb.add(Rule("r2", lit("callable(v2)"), (), 1))
rb.add(Rule("r3", lit("deploy_ok"), (lit("callable(?E)"),), 1))
s = solve(rb)
eq("ex2 before", entails(s, lit("deploy_ok")), True)

rb.rules["r1"] = Rule("r1", lit("~callable(v1)"), (), 2)       # SUPERSEDE
s = solve(rb)
eq("ex2 closure", sorted(map(str, s.closure)),
   ["callable(v2)", "deploy_ok", "~callable(v1)"])
eq("ex2 call v1 violates", check_assertion(s, [lit("callable(v1)")]).violation, True)
eq("ex2 call v2 passes", check_assertion(s, [lit("callable(v2)")]).violation, False)
eq("ex2 deploy_ok re-derived through the surviving path",
   entails(s, lit("deploy_ok")), True)

# -- instance-level defeat: a rule defeated in one context stays valid in another
rb = RuleBase(signature={"ok": ("v",), "grp": ("v", "g")},
              universe={"v": ["A", "B"], "g": ["gx"]})
rb.add(Rule("f", lit("grp(A,gx)"), (), 1))
rb.add(Rule("rA", lit("ok(A)"), (), 1))
rb.add(Rule("rB", lit("ok(B)"), (), 1))
rb.add(Rule("rg", lit("~ok(?X)"), (lit("grp(?X,gx)"),), 3))
s = solve(rb)
eq("instance-level defeat", sorted(map(str, s.closure)),
   ["grp(A,gx)", "ok(B)", "~ok(A)"])

# RETRACT withdraws only the named instance
rb.rules["rg"].retracted = frozenset({("A",)})
s = solve(rb)
eq("retract restores the instance", sorted(map(str, s.closure)),
   ["grp(A,gx)", "ok(A)", "ok(B)"])

# -- no closure may ever be self-contradictory ------------------------------
eq("closure consistent", contradictions(s.closure), [])

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print(f"engine spec: all checks passed")
