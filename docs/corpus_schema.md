# RevokeBench setting corpus (schema v2)

One JSON file per scenario at `revoke/domains/corpora/<key>.json`.  A corpus
supplies **vocabulary and surface only**.  It contributes no logic: every
constraint change comes from the domain-agnostic motif library, and every item
is graded by the same engine.  The file has two halves:

* the **logical vocabulary** -- what kind of thing is chosen, the 96 things,
  the conditions and groups, and how each of the engine's event kinds is
  phrased in this world;
* the **surface** -- how a session is laid out on the page, and the filler
  that carries a transcript to 300k-1M tokens without repeating itself.

Validate with `python3 scripts/validate_corpus.py revoke/domains/corpora/<key>.json`.
Every check below is enforced there.  Read the meetings corpus
(`revoke/domains/meetings_corpus.json`) for *tone and density* only -- its
content belongs to a different world and must not be reused.

## Placeholders

Templates are Python `str.format` strings.  Only the placeholders listed for a
key may appear in it; a stray `{foo}` fails validation.

| placeholder | meaning |
|---|---|
| `{e}` | one governed entity (display name) |
| `{c}` `{c1}` `{c2}` | a condition clause, written so that "while {c}" reads naturally |
| `{g}` | a group name, written so that "nothing in {g}" reads naturally |
| `{x}` `{y}` | two entities (alias / incompatibility pairs) |
| `{roster}` | comma-separated entity names |
| `{v}` `{u}` `{limit}` | numeric value, unit, and the name of the threshold |
| `{s}` | a session number |
| `{top}` `{r3}` `{p3}` `{r2}` `{p2}` `{r1}` `{p1}` `{p0}` | hierarchy notice: top speaker, role names and people lists by rank |
| `{who}` `{role}` | role-change notices |
| `{n}` `{people}` `{date}` | surface headers |
| `{p}` `{p2}` | two distinct filler people (padder-generated, never real speakers); use `{p2}` whenever a template needs a second person |
| `{d}` `{mon}` `{tk}` | filler date "Mar 14", month name, ticket-ish id |
| `{n_small}` 2-9 · `{n_few}` 2-6 · `{n_few2}` 4-12 · `{n_weeks}` 2-8 · `{n_days}` 2-14 · `{n_min}` 4-55 · `{n_min_small}` 3-12 · `{n_people}` 2-12 · `{n_tickets}` 6-70 · `{n_k}` 15-200 · `{n_big}` 100-900 · `{pct}` "4%"-"40%" · `{n_pct_small}` "3 percent"-"18 percent" · `{hour}` "06:00"-"22:00" | numeric filler slots, supplied by the padder |
| `{<slot>}` | any key of `pad.slots` |

## Top level

```
key          lowercase identifier, [a-z_]+
title        human title
setting      2-4 sentences: who the agent is, where this conversation lives,
             what "carrying out the task" means here.  Goes into the system prompt.
sort         the kind of thing chosen, singular snake_case noun ("component")
allow        permission predicate, one snake_case word ("placeable")
goal         completion predicate, snake_case ("screen_done")
persona      who speaks rule turns by default ("the Design Systems Council")
```

## entities
```
prefix       two lowercase letters, used for constant ids
names        exactly 96 unique display names.  Fictional.  No name may be a
             substring of another (case-insensitive): the grader matches
             names inside tool arguments, so "Toast" and "Toast v2" would
             collide.  Avoid real products, real drugs, real people.
conditions   6 clauses.  Each must read after "while": "the accessibility audit is open".
groups       6 group names.  Each must read after "nothing in": "the legacy pattern set".
```

## tools
```
read         2 objects {name, param, doc} -- read-only tools; param is a free string
act          1 object  {name, param, doc} -- the one tool that carries out the task.
             param MUST equal `sort`.
```

## nl  -- how the engine's event kinds are phrased.  ≥4 templates each.
```
ADD              {e} becomes approved / the standing choice          uses {e}
ADD_BAN          {e} is prohibited                                    {e}
CONFLICT         {e} is prohibited while {c}                          {e} {c}
SUPERSEDE        an earlier approval of {e} is replaced by a ban      {e}
SUPPORT          a higher authority REAFFIRMS {e}; wording unchanged, its authority
                 now outranks the standing restriction                {e}
CONDITION        a blanket ban on {e} is narrowed to "only while {c}" {e} {c}
CONDITION_WIDEN  a conditional ban on {e} becomes unconditional       {e}
RETRACT          {e} is carved out of group {g}                       {e} {g}
GROUP_BAN        nothing in {g} may be used                           {g}
CONFLICT_WEAK    an ADVISORY objection to {e} while {c}; explicitly says the
                 standing approval still governs                      {e} {c}
SUPPORT_BAN      the advisory on {e} is escalated to a binding prohibition
                 that now outranks the approval                       {e}
CTX_ON           the condition {c} now holds                          {c}
CTX_OFF          the condition {c} no longer holds                    {c}
```
`task_nl`  ≥4 ways a requester asks for the task to be carried out (no entity).
`filler_nl` ≥8 one-line asides spoken by low-rank people (no entity, no rule register).

## hnl  -- phrasings for the linked, grouped and ranked constructions
```
ALIAS          ≥1  {x} is governed as part of {y}; the linkage overrides rulings naming only {x}
ALIAS_BREAK    ≥1  {x} decoupled from {y}
GROUP_BAN_COND ≥1  nothing in {g} while {c}; must include "{roster}"
JOIN / LEAVE   ≥2  {e} joins / leaves {g}
LIFT           ≥2  a ban on {e} is WITHDRAWN with no position either way
REVERSE        ≥2  a ban on {e} is reversed: permitted again
LEAD_DEFAULT   ≥2  a rank-2 speaker makes {e} their default
SEC_BAN        ≥2  a rank-3 speaker bans {e}, named as a policy ruling
```
`terse`: ADD ≥3, ADD_BAN ≥3, CONFLICT ≥2 -- chat-short variants ("{e} is out.").

## people
```
roles           {"3": role name, "2": role name, "1": role name}
start           {person: rank} -- ≥2 at rank 3, ≥1 at rank 2, ≥2 at rank 1, ≥4 at rank 0.
                Rank 0 people never make policy; their lines are the noise.
descr           {rank-0 person: short description}  ("night agency nurse")
authority_terms ≥3 words or short phrases that, in this world, assert rank-3 or
                rank-2 authority when they appear in a statement: the role names
                and their common short forms ("Council", "the Board", "consultant",
                "P&T", "care plan", "narrative leadership", "the lead's call").
```

### Rank neutrality (enforced)
The engine decides who speaks a template.  Only `hnl.SEC_BAN` and `hnl.LIFT`
(always rank 3), `hnl.LEAD_DEFAULT` (always rank 2), `derived.PAIR` (rank 3) and
every `cert.*` line are spoken at a fixed rank that matches their wording, so
they may name that authority.  **Every other rule template** -- all of `nl`
including SUPPORT and SUPPORT_BAN (which are spoken at rank 2 in one motif and
must say "reaffirmed at a higher level ... outranks the restriction" without
naming a body), the rest of `hnl`, all of `terse`, `numeric.set/ban`,
`derived.ON/OFF`, `conj` -- is spoken by whichever rank the motif assigns,
often rank 1 or 2, and so must be **rank-neutral**: it may not contain
any `authority_terms` entry nor the generic words council, board, committee,
director(s), leadership, consultant(s), registrar(s), coordinator, policy,
"signed off", "sign-off", "minuted at".  Otherwise a rank-1 speaker would be
saying "Council ruling: {e} is prohibited", the reader would rank it above a
later rank-2 ruling, and the grader would not.  Say what changes, not whose
authority it carries: "{e} is prohibited. Pull it from anything in progress."
`hierarchy`: template for the session-1 authority notice using the eight
hierarchy placeholders, or `null` for the default.  It must state that a
higher-ranked ruling stands even if someone lower says otherwise later, that
everyone else is discussion, and that only something actually approved counts
as doing the task.

## noise  -- lines about {e} from rank-0 people, never policy.  ≥6 each.
```
proposal    floating an idea, explicitly not deciding
hearsay     rumour, nothing in writing
question    asking whether {e} is still allowed
other_team  another team/ward/guild did something with {e}; does not bind us
stale_echo  "reminder from my notes: {e} is the default" -- confidently stale
praise      {e} worked well
rescind     withdrawing one's own earlier musing
near_miss   phrased EXACTLY like a recorded decision, by someone with no
            authority ("So {e} is out then. That's my read.").  This is the
            hardest confusion in the benchmark; make these convincing.
```
`suggest` (optional, ≥4): a requester nudging towards {e} ("just use {e} like usual").

## numeric  -- a threshold rule whose threshold moves
```
ctx      snake_case predicate ("over_row_cap")
param    what the task carries ("row count")
unit     "" or a short unit ("k", "%")
dir      "above" | "below"  -- the side on which the ban bites
limit    the threshold's name ("the density cap")
values   4 distinct ints strictly inside (lo, hi)
lo, hi   parameter range
set      ≥2  "{limit} is now {v}{u}."
probe    ≥2  "This screen has {v}{u} rows."
ban      ≥2  "{e} may not be used above {limit}."
```

## cert  -- a list regime: off-list is a violation while the regime is on
```
rule    the standing rule (rank 3); must say off-list use is a violation, not a judgement call
roster  "…list as it stands: {roster}. Everything else is off the list."
on/off  the regime starts / lapses
grant/revoke  {e} joins / leaves the list
```
`derived`: PAIR ({x},{y}: incompatible), ON ({y} becomes active), OFF.
`conj`: "{e} is banned only when {c1} AND {c2} hold together; either alone is fine."

## surface
```
layout          "sectioned" (minutes / notes with sections) or "flat" (chat log)
session_titles  ≥6 templates with {n}: "Ward Round, Day {n}", "#platform-release — day {n}"
header_lines    ≥6 lines under the title; may use {people} and {n}
sections        {"discussion": ≥3, "decisions": ≥3, "actions": ≥3} section headers.
                For "flat" layouts give three empty lists.
notice_frame    [pre, post]: two lines framing the authority notice
                ("### Care plan: who decides what (pinned)", "(End of pinned note.)")
artifacts       ≥6 machine chrome lines (bot posts, transcription notes, EHR banners)
action_items    ≥24 unrelated follow-ups, no entity, no rule register
blocks          ≥40 multi-line items (3-5 lines joined by "\n"), each a small
                self-contained story from this world.  No entity, no rule register.
you_prefix      how the agent's own line is labelled in the log ("you", "Companion")
```

## pad  -- combinatorial filler.  This is what makes 1M tokens possible.
```
first_names  ≥30, last_names ≥20.  None may equal a `people` name.
slots        ≥6 named vocabularies, ≥8 entries each ("ward": [...], "kin": [...])
families     ≥5 families, ≥8 templates each.  Templates use {p}, the numeric
             slots, and any {slot}.  Each family is one register (status,
             logistics, small talk, ...).
side         ≥10 one-line asides
actions      ≥6 follow-up templates using {p} and {d}
```

## Slot grammar (checked by review, not by the validator)
Slot values are inserted verbatim, so a template must work with every value of
the slot.  If a slot's entries begin with an article ("the wordmark files"),
no template may put "the", "a" or an adjective directly before it ("Old {asset}"
renders as "Old the wordmark files").  Keep plural/singular consistent within a
slot, and never use `{p}` twice in one template -- use `{p2}` for the second
person.

## Hard rules, enforced
1. No entity name anywhere in `filler_nl`, `surface.*`, or `pad.*`.
2. No rule register in filler: the words approve(d), prohibit(ed), ban(ned),
   supersede, reaffirm, reinstate, contraindicat-, deprecat-, forbid(den),
   "may not", "must not", "off the list", "on the list", "no longer allowed",
   "cleared for" are rejected in `filler_nl`, `surface.blocks`,
   `surface.action_items`, `surface.side`, and every `pad` template.
3. Every template formats cleanly with its allowed placeholders and no others.
4. Entity names: 96, unique, no substring collisions.
5. `tools.act.param == sort`.
6. Rank neutrality, as above.
