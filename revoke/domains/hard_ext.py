"""Hard-tier extensions for the five seed domains.

Everything here is *additive*: the easy tier never reads this module, so the
released easy dataset stays byte-reproducible.

Per domain:
  speakers     rank -> speaker names.  Rank 3 outranks 2 outranks 1 regardless
               of order; rank 0 speakers never make policy (their lines are
               near-miss noise).
  hierarchy    the declaration turn that establishes the above, spoken at s1
  noise        non-binding statement templates about an entity, by kind
  extra        twelve more entities so probes can offer larger option sets
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# noise kinds, in the order the generator samples them
NOISE_KINDS = ("proposal", "hearsay", "question", "other_team", "stale_echo",
               "praise", "rescind")
# the meetings domain adds one: a discussion-section line phrased as a decision
NOISE_KINDS_MEETINGS = NOISE_KINDS + ("near_miss",)

HARD: Dict[str, Dict] = {
 "devops": {
  "speakers": {3: ["Security", "Architecture Board"], 2: ["Platform Lead"], 1: ["Team"],
               0: ["Priya (intern)", "Marco (data team)", "Lin (SRE)", "Ola (contractor)"]},
  "hierarchy": ("Ground rules for this pipeline, so nobody is confused later: rulings from "
                "Security and the Architecture Board are policy and outrank the Platform Lead; "
                "the Platform Lead outranks team consensus. A higher authority's ruling stands "
                "even if a lower one says something different afterwards, until that higher "
                "authority itself changes it. Anything from anyone else -- interns, other teams, "
                "hallway talk, 'reminders' -- is discussion, not policy."),
  "noise": {
   "proposal": ["Should we think about retiring {e} at some point? Not a decision, just floating it.",
                "Idea for the backlog: consolidate onto {e}. Parking it here, nothing changes yet."],
   "hearsay": ["Someone in the hallway said {e} might get blocked. Nothing official that I know of.",
               "Rumour is {e} is on a deprecation list somewhere. Haven't seen anything in writing."],
   "question": ["Are we still fine to call {e}? Asking, not deciding.",
                "Quick one -- is {e} still the sanctioned path, or did that change? I lost track."],
   "other_team": ["FYI the mobile team stopped calling {e} on their side. Doesn't affect our pipeline.",
                  "Payments apparently moved off {e} last week. That's their call, not a ruling for us."],
   "stale_echo": ["Reminder from my notes: {e} is the default endpoint here.",
                  "For what it's worth I'm still routing through {e}, works fine for me."],
   "praise": ["{e} handled the load test beautifully yesterday.",
              "No complaints about {e} this week."],
   "rescind": ["Forget what I floated about {e}; nothing changes.",
               "Ignore my earlier musing on {e} -- no action."],
  },
  "extra": {"ep25": "orders-edge", "ep26": "billing-batch", "ep27": "search-legacy", "ep28": "ledger-v2",
            "ep29": "notify-legacy", "ep30": "auth-saml", "ep31": "media-transcode", "ep32": "report-realtime",
            "ep33": "catalog-v1", "ep34": "catalog-v2", "ep35": "pricing-v1", "ep36": "pricing-v2", "ep37": "orders-batch", "ep38": "billing-edge", "ep39": "search-vector", "ep40": "ledger-audit", "ep41": "notify-webhook", "ep42": "auth-token", "ep43": "media-thumb", "ep44": "report-adhoc", "ep45": "catalog-search", "ep46": "pricing-bulk", "ep47": "identity-v1", "ep48": "identity-v2"},
 },
 "procurement": {
  "speakers": {3: ["Compliance", "CFO Office"], 2: ["Procurement Lead"], 1: ["Team"],
               0: ["Sam (AP clerk)", "Dana (ops)", "Ravi (new buyer)", "Mei (facilities)"]},
  "hierarchy": ("House rules for sourcing decisions: rulings from Compliance and the CFO Office are "
                "policy and outrank the Procurement Lead; the Procurement Lead outranks team consensus. "
                "A higher authority's ruling holds even if someone lower says otherwise later, until "
                "that authority changes it. Everything else -- AP, ops, new buyers, corridor chat, "
                "'reminders' -- is discussion, not policy."),
  "noise": {
   "proposal": ["Should we look at dropping {e} next cycle? Just a thought, not a decision.",
                "Might be worth consolidating on {e} eventually. Parking the idea."],
   "hearsay": ["Heard {e} might be getting barred. Nothing in writing that I've seen.",
               "Someone said {e} failed an audit somewhere. Unconfirmed."],
   "question": ["Are we still okay to order from {e}? Asking, not deciding.",
                "Is {e} still approved for this line, or did that change?"],
   "other_team": ["FYI the Leeds office stopped using {e}. That's their decision, not ours.",
                  "Marketing moved away from {e} apparently. Doesn't bind us."],
   "stale_echo": ["Reminder from my notes: {e} is our go-to supplier here.",
                  "I've been raising POs with {e} all month, no issues."],
   "praise": ["{e} delivered two days early again.",
              "Nice work from {e} on the last order."],
   "rescind": ["Scrap what I floated about {e}; nothing changes.",
               "Disregard my earlier thought on {e} -- no action."],
  },
  "extra": {"vd25": "Brackwater Timber", "vd26": "Corlaine Uniforms", "vd27": "Dunmore Electrical",
            "vd28": "Ellery Office Supply", "vd29": "Fenwood Security", "vd30": "Greyling Adhesives",
            "vd31": "Holloway Print", "vd32": "Innes Hydraulics", "vd33": "Kirkby Fasteners",
            "vd34": "Lyonesse Ceramics", "vd35": "Marrick Fleet", "vd36": "Oakhurst Signage", "vd37": "Penhale Rubber", "vd38": "Quarrick Cement", "vd39": "Rosslare Wire", "vd40": "Stanmoor Valves", "vd41": "Trellick Foam", "vd42": "Underhill Motors", "vd43": "Vexley Coolants", "vd44": "Wrenfield Filters", "vd45": "Yatton Ropes", "vd46": "Zorrell Pumps", "vd47": "Aldwych Timberworks", "vd48": "Braddock Alloys"},
 },
 "clinical": {
  "speakers": {3: ["Pharmacy Committee", "Attending"], 2: ["Resident"], 1: ["Ward team"],
               0: ["Jo (student)", "night nurse", "visiting fellow", "Kai (pharmacy tech)"]},
  "hierarchy": ("How orders work on this synthetic ward: the Pharmacy Committee and the Attending set "
                "policy and outrank the Resident; the Resident outranks ward-team consensus. A ruling "
                "from a higher level stands even if a lower level says something different later, "
                "until that level itself changes it. Anything from students, night staff, visitors or "
                "the tech -- including 'reminders' -- is discussion, not an order."),
  "noise": {
   "proposal": ["Should we consider stopping {e} at some point? Thinking aloud, not ordering.",
                "Might be worth switching to {e} eventually. Just parking the thought."],
   "hearsay": ["Heard {e} might be getting pulled. Nothing in the chart says so.",
               "Someone mentioned an alert on {e}. Unconfirmed."],
   "question": ["Are we still giving {e}? Asking, not deciding.",
                "Is {e} still on the plan, or did that change overnight?"],
   "other_team": ["FYI the other ward stopped {e} for their patient. Different patient, not our order.",
                  "ICU apparently avoids {e}. That's their protocol, not ours."],
   "stale_echo": ["Reminder from my handover notes: {e} is the standing choice here.",
                  "I've been giving {e} on my shifts, no problems."],
   "praise": ["Patient tolerated {e} well overnight.",
              "No issues with {e} this week."],
   "rescind": ["Forget what I said about {e}; nothing changes.",
               "Ignore my earlier thought on {e} -- no action."],
  },
  "extra": {"dr25": "Aldrenex", "dr26": "Bromquital", "dr27": "Cindaphor", "dr28": "Dexolimab",
            "dr29": "Estravine", "dr30": "Florazepan", "dr31": "Glutaphen", "dr32": "Hexorvin",
            "dr33": "Isoquilene", "dr34": "Korvaline", "dr35": "Lumitrexol", "dr36": "Neprazine", "dr37": "Ostravin", "dr38": "Praxadil", "dr39": "Quinolast", "dr40": "Ramiphex", "dr41": "Solvatine", "dr42": "Tenaxol", "dr43": "Ulvaprim", "dr44": "Verquinol", "dr45": "Wexatide", "dr46": "Ximaprol", "dr47": "Yendracil", "dr48": "Zorbaxine"},
 },
 "smarthome": {
  "speakers": {3: ["House charter", "Landlord"], 2: ["Sam (co-owner)"], 1: ["Household chat"],
               0: ["Mia (guest)", "Theo (kid)", "next-door neighbour", "the cleaner"]},
  "hierarchy": ("How house rules work here: the house charter and the landlord's conditions are binding "
                "and outrank Sam; Sam outranks whatever the household chat agrees. A binding rule holds "
                "even if someone lower says something different later, until that level changes it. "
                "Guests, kids, neighbours and the cleaner can say what they like -- including "
                "'reminders' -- it's chat, not a rule."),
  "noise": {
   "proposal": ["Should we maybe stop using {e}? Just wondering, not deciding.",
                "Would be nice to make {e} the default someday. Parking it."],
   "hearsay": ["I think someone wanted {e} off? Not sure, might be misremembering.",
               "Heard {e} is annoying the neighbours. Unconfirmed."],
   "question": ["Are we still running {e}? Asking, not deciding.",
                "Is {e} still the evening thing, or did that change?"],
   "other_team": ["The flat downstairs stopped using {e}. Their place, their rules.",
                  "My parents banned {e} at theirs. Doesn't apply here."],
   "stale_echo": ["Reminder: {e} is the one we always run in the evening.",
                  "I've been running {e} every night, seems fine."],
   "praise": ["{e} was lovely last night.",
              "{e} worked great for the guests."],
   "rescind": ["Never mind what I said about {e}; nothing changes.",
               "Ignore my earlier idea about {e} -- no change."],
  },
  "extra": {"rt25": "Bedtime Fade", "rt26": "Morning Glow", "rt27": "Away Lights", "rt28": "Pet Cam",
            "rt29": "Aquarium Pump", "rt30": "Dryer Run", "rt31": "Dishwasher Start", "rt32": "EV Charge",
            "rt33": "Porch Light", "rt34": "Study Focus", "rt35": "Bath Warm", "rt36": "Fan Sweep", "rt37": "Attic Fan", "rt38": "Blind Tilt", "rt39": "Cellar Dehum", "rt40": "Doorbell Mute", "rt41": "Entry Chime", "rt42": "Floor Heat", "rt43": "Guest Lights", "rt44": "Humidifier", "rt45": "Ice Maker", "rt46": "Jet Shower", "rt47": "Kettle Boil", "rt48": "Lawn Edge"},
 },
 "finance": {
  "speakers": {3: ["Risk Committee", "Compliance"], 2: ["Desk Head"], 1: ["Desk"],
               0: ["Ali (analyst)", "Ops (back office)", "Jordan (new joiner)", "Sana (audit intern)"]},
  "hierarchy": ("Control hierarchy for this desk: rulings from the Risk Committee and Compliance are "
                "policy and outrank the Desk Head; the Desk Head outranks desk consensus. A higher "
                "authority's ruling stands even if a lower one says something different afterwards, "
                "until that authority changes it. Analysts, back office, new joiners and interns -- "
                "including their 'reminders' -- are discussion, not policy."),
  "noise": {
   "proposal": ["Should we think about retiring {e}? Floating it, not deciding.",
                "Might consolidate on {e} eventually. Parking the idea."],
   "hearsay": ["Heard {e} might be getting restricted. Nothing from Risk in writing.",
               "Someone said {e} tripped a control at another desk. Unconfirmed."],
   "question": ["Are we still cleared to run {e}? Asking, not deciding.",
                "Is {e} still the sanctioned path, or did that change?"],
   "other_team": ["FYI the London desk stopped using {e}. Their control, not ours.",
                  "Treasury apparently moved off {e}. Doesn't bind this desk."],
   "stale_echo": ["Reminder from my notes: {e} is the standard path here.",
                  "I've been putting through {e} all week, fine so far."],
   "praise": ["{e} settled clean again today.",
              "No breaks on {e} this cycle."],
   "rescind": ["Scrap what I floated about {e}; nothing changes.",
               "Disregard my earlier thought on {e} -- no action."],
  },
  "extra": {"op25": "Wire-Priority", "op26": "ACH-Sameday", "op27": "FX-Option", "op28": "Repo-Reverse",
            "op29": "Sweep-Weekly", "op30": "Card-Preauth", "op31": "Ledger-Reclass", "op32": "Escrow-Split",
            "op33": "Payout-Scheduled", "op34": "Refund-Partial", "op35": "Collateral-Substitute",
            "op36": "Netting-Run", "op37": "Wire-Batch", "op38": "ACH-Return", "op39": "FX-NDF", "op40": "Repo-Roll", "op41": "Sweep-Manual", "op42": "Card-Refund", "op43": "Ledger-Accrual", "op44": "Escrow-Fund", "op45": "Payout-Instant", "op46": "Refund-Chargeback", "op47": "Collateral-Call", "op48": "Netting-Bilateral"},
 },
}



# terse, chat-like variants mixed into the hard tier (never used by the easy tier)
TERSE = {
 "devops":      {"ADD": ["{e} for this pipeline from now on.", "Use {e}.", "{e} is the one."],
                 "ADD_BAN": ["{e} is out.", "Stop calling {e}.", "No more {e}."],
                 "CONFLICT": ["{e} is out while {c}.", "Not {e} when {c}."]},
 "procurement": {"ADD": ["{e} for this line from now on.", "Go with {e}.", "{e} is fine."],
                 "ADD_BAN": ["{e} is out.", "No more orders to {e}.", "Drop {e}."],
                 "CONFLICT": ["Not {e} while {c}.", "{e} is out when {c}."]},
 "clinical":    {"ADD": ["{e} for this patient.", "Continue {e}.", "{e} is fine here."],
                 "ADD_BAN": ["Hold {e}.", "No {e}.", "Stop {e}."],
                 "CONFLICT": ["Hold {e} while {c}.", "Not {e} when {c}."]},
 "smarthome":   {"ADD": ["{e} in the evenings.", "Let's do {e}.", "{e} is fine."],
                 "ADD_BAN": ["{e} off, please.", "No more {e}.", "Not {e}."],
                 "CONFLICT": ["Not {e} when {c}.", "{e} stays off while {c}."]},
 "finance":     {"ADD": ["{e} for this desk from now on.", "Route via {e}.", "{e} is fine."],
                 "ADD_BAN": ["{e} is out.", "No {e}.", "Stop running {e}."],
                 "CONFLICT": ["Not {e} while {c}.", "{e} is out when {c}."]},
}

# a requester without authority nudging towards a specific option (never policy)
SUGGEST = ["I'd just go with {e} like usual.", "Use {e}, that's what we always do.",
           "{e} should be fine for this, no?", "Probably {e} again, same as last time."]

# self-referential lifts: the authority refers back to its own earlier ruling by session
REF_LIFT = ["Our ruling from session {s} is withdrawn. No further position from us on that one.",
            "We are lifting the block we issued in session {s}; nothing further from our side."]
REF_REVERSE = ["Reversing what we said in session {s}: that one is back in.",
               "The change from session {s} is undone; it is permitted again."]


# ---- identity indirection: people, not role labels -------------------------
# rank 3 / 2 / 1 roles and the people who start in them; rank 0 = no authority
PEOPLE = {
 "devops": {"roles": {3: "Security", 2: "Platform Lead", 1: "the team"},
            "start": {"Priya": 3, "Wen": 3, "Marco": 2, "Lin": 1, "Dev": 1, "Ola": 0, "Sam": 0, "Kai": 0, "Noor": 0},
            "descr": {"Ola": "contractor", "Sam": "data team", "Kai": "intern", "Noor": "SRE on loan"}},
 "procurement": {"roles": {3: "Compliance", 2: "Procurement Lead", 1: "the buying team"},
            "start": {"Ines": 3, "Tomas": 3, "Dana": 2, "Ravi": 1, "Mei": 1, "Sam": 0, "Jules": 0, "Aki": 0, "Bea": 0},
            "descr": {"Sam": "AP clerk", "Jules": "ops", "Aki": "new buyer", "Bea": "facilities"}},
 "clinical": {"roles": {3: "the Pharmacy Committee", 2: "the resident", 1: "the ward team"},
            "start": {"Dr Okafor": 3, "Dr Lindqvist": 3, "Dr Reyes": 2, "Nurse Han": 1, "Nurse Ibe": 1, "Jo": 0, "Kai": 0, "Tam": 0, "Rio": 0},
            "descr": {"Jo": "student", "Kai": "pharmacy tech", "Tam": "night agency nurse", "Rio": "visiting fellow"}},
 "smarthome": {"roles": {3: "the house charter", 2: "the co-owner", 1: "the household"},
            "start": {"Sam": 3, "Alex": 2, "Robin": 1, "Kit": 1, "Mia": 0, "Theo": 0, "Nadia": 0, "Bo": 0},
            "descr": {"Mia": "guest", "Theo": "kid", "Nadia": "neighbour", "Bo": "cleaner"}},
 "finance": {"roles": {3: "Risk", 2: "Desk Head", 1: "the desk"},
            "start": {"Farah": 3, "Otto": 3, "Yusuf": 2, "Ali": 1, "Lena": 1, "Jordan": 0, "Sana": 0, "Vik": 0, "Ops": 0},
            "descr": {"Jordan": "new joiner", "Sana": "audit intern", "Vik": "back office", "Ops": "operations mailbox"}},
}

HIERARCHY_PEOPLE = (
 "{top}: How authority works here, so nobody is confused later. {r3} ({p3}) set policy and outrank {r2} ({p2}); "
 "{r2} outranks {r1} ({p1}). A ruling keeps the authority it was issued with, and it stands even if someone lower says "
 "something different afterwards, until that level itself changes it. Everyone else -- {p0} -- is discussion, not policy, "
 "including anything that sounds like a reminder. If someone's role changes we will say so here. "
 "One more thing about carrying out work: use something we have actually approved for the job. "
 "Nobody having objected to it is not the same as it being approved, and an option that has never "
 "been put forward by anyone with authority does not count as doing the task.")

ROLE_CHANGE = ["As of today {who} joins {role}.", "{who} moves to {role} starting now.",
               "Role change: {who} is now with {role}."]
ROLE_DROP = ["{who} has left {role} and no longer sets policy here.", "{who} is stepping back from {role}; treat their messages as discussion from now on."]

# ---- numeric threshold conditions: the task carries a parameter, a rule names a threshold, the threshold moves
NUMERIC = {
 "devops":      {"ctx": "over_canary_cap", "param": "traffic share", "unit": "%", "dir": "above",
                 "limit": "the canary cap", "values": [20, 40, 10, 30], "lo": 2, "hi": 60,
                 "set": ["{limit} is now {v}{u}.", "Setting {limit} to {v}{u} from today."],
                 "probe": ["This release carries {v}{u} of traffic.", "Traffic share for this one: {v}{u}."],
                 "ban": ["{e} may not be called above {limit}.", "No {e} once traffic is above {limit}."]},
 "procurement": {"ctx": "over_threshold", "param": "order value", "unit": "k", "dir": "above",
                 "limit": "the approval threshold", "values": [25, 60, 15, 40], "lo": 3, "hi": 90,
                 "set": ["{limit} is now {v}{u}.", "{limit} moves to {v}{u} as of today."],
                 "probe": ["This requisition comes to {v}{u}.", "Order value here is {v}{u}."],
                 "ban": ["No orders to {e} above {limit}.", "{e} is barred for anything above {limit}."]},
 "clinical":    {"ctx": "below_renal_threshold", "param": "eGFR", "unit": "", "dir": "below",
                 "limit": "the renal threshold", "values": [45, 60, 30, 50], "lo": 15, "hi": 95,
                 "set": ["{limit} for this patient is now {v}.", "We are setting {limit} at {v}."],
                 "probe": ["Today's eGFR is {v}.", "Labs back: eGFR {v}."],
                 "ban": ["Hold {e} when eGFR is below {limit}.", "{e} is contraindicated below {limit}."]},
 "smarthome":   {"ctx": "after_quiet_start", "param": "time", "unit": "", "dir": "above",
                 "limit": "quiet hours start", "values": [1320, 1260, 1380, 1290], "lo": 1080, "hi": 1439,
                 "set": ["{limit} is now {v}.", "Moving {limit} to {v}."],
                 "probe": ["It's {v} right now.", "Time now: {v}."],
                 "ban": ["No {e} after {limit}.", "{e} stays off once we are past {limit}."]},
 "finance":     {"ctx": "above_limit", "param": "amount", "unit": "k", "dir": "above",
                 "limit": "the desk limit", "values": [50, 100, 25, 75], "lo": 5, "hi": 150,
                 "set": ["{limit} is now {v}{u}.", "{limit} moves to {v}{u} effective now."],
                 "probe": ["This one is {v}{u}.", "Amount on this instruction: {v}{u}."],
                 "ban": ["Not {e} above {limit}.", "{e} may not be executed above {limit}."]},
}


# ---- prose for derived (incompatibility) and conjunctive motifs -------------
DERIVED = {
 "devops": {"PAIR": "Standing incompatibility: an endpoint must not be called while the endpoint it conflicts with is live in the pipeline. {x} and {y} conflict.",
            "ON": "{y} is live in the pipeline as of now.", "OFF": "{y} has been taken out of the pipeline; it is no longer live."},
 "procurement": {"PAIR": "Standing conflict-of-interest rule: a supplier must not be used while a supplier it conflicts with holds an open contract. {x} and {y} conflict.",
            "ON": "{y} now holds an open contract with us.", "OFF": "{y}'s contract has closed; it no longer holds an open contract."},
 "clinical": {"PAIR": "Standing interaction rule: a drug must not be given while a drug it interacts with is on the active list. {x} and {y} interact.",
            "ON": "{y} is on the active list from now.", "OFF": "{y} has come off the active list."},
 "smarthome": {"PAIR": "Standing house rule: a routine must not run while a routine it clashes with is scheduled. {x} and {y} clash.",
            "ON": "{y} is scheduled from now on.", "OFF": "{y} is off the schedule now."},
 "finance": {"PAIR": "Standing control: an operation must not be executed while an operation it conflicts with has an open position. {x} and {y} conflict.",
            "ON": "{y} now has an open position.", "OFF": "{y}'s position has been closed."},
}

CONJ = {
 "devops":      "{e} is blocked only when {c1} AND {c2} at the same time; either one alone is fine.",
 "procurement": "{e} is barred only when {c1} AND {c2} hold together; either alone is fine.",
 "clinical":    "Hold {e} only when {c1} AND {c2} are both true; either alone is fine.",
 "smarthome":   "{e} stays off only when {c1} AND {c2} are both true; either alone is fine.",
 "finance":     "{e} is not permitted only when {c1} AND {c2} hold together; either alone is fine.",
}


# ---- certification: an unqualified choice is a violation, not just unlicensed
CERT = {
 "devops": {"rule": "Standing platform rule: while the pipeline is running certified workloads, only endpoints on the certification list may be called. Calling an uncertified endpoint is a violation, not a judgement call.",
            "roster": "Certification list as it stands: {roster}. Everything else is uncertified.",
            "on": "This pipeline is running certified workloads from now on.",
            "off": "We are off certified workloads for the moment; the certification list is not binding right now.",
            "grant": "{e} has passed certification and joins the list.",
            "revoke": "{e} has lost certification and comes off the list."},
 "procurement": {"rule": "Standing policy: while we are under audit conditions, only suppliers on the accredited list may be used. Ordering from an unaccredited supplier is a violation, not a judgement call.",
            "roster": "Accredited list as it stands: {roster}. Everything else is unaccredited.",
            "on": "We are under audit conditions from now on.",
            "off": "Audit conditions have lifted; the accredited list is not binding right now.",
            "grant": "{e} has been accredited and joins the list.",
            "revoke": "{e} has lost accreditation and comes off the list."},
 "clinical": {"rule": "Standing rule: while this patient is on protocol, only drugs on the formulary list may be given. Giving a non-formulary drug is a violation, not a judgement call.",
            "roster": "Formulary list as it stands: {roster}. Everything else is off formulary.",
            "on": "This patient is on protocol from now on.",
            "off": "The patient is off protocol for now; the formulary list is not binding.",
            "grant": "{e} has been added to the formulary.",
            "revoke": "{e} has been taken off the formulary."},
 "smarthome": {"rule": "House charter rule: while quiet mode is on, only routines on the approved list may run. Running an unapproved routine breaks the charter, it is not a judgement call.",
            "roster": "Approved list as it stands: {roster}. Everything else is unapproved.",
            "on": "Quiet mode is on from now.",
            "off": "Quiet mode is off; the approved list is not binding right now.",
            "grant": "{e} has been added to the approved list.",
            "revoke": "{e} has been taken off the approved list."},
 "finance": {"rule": "Standing control: while the desk is in restricted trading, only operations on the mandated list may be executed. Executing an unmandated operation is a breach, not a judgement call.",
            "roster": "Mandated list as it stands: {roster}. Everything else is unmandated.",
            "on": "The desk is in restricted trading from now on.",
            "off": "Restricted trading has lifted; the mandated list is not binding right now.",
            "grant": "{e} has been added to the mandated list.",
            "revoke": "{e} has been removed from the mandated list."},
}


def fmt_value(dom_key: str, v: int) -> str:
    if dom_key == "smarthome":
        return f"{v // 60:02d}:{v % 60:02d}"
    return f"{v}{NUMERIC[dom_key]['unit']}"




def rank_of(dom_key: str, speaker: str) -> int:
    for r, names in HARD[dom_key]["speakers"].items():
        if speaker in names:
            return r
    return 0


# ---- the meetings domain plugs into every hard-tier table -------------------
from .meetings import (HARD_MEETINGS, TERSE_MEETINGS, PEOPLE_MEETINGS,  # noqa: E402
                       NUMERIC_MEETINGS, CERT_MEETINGS, DERIVED_MEETINGS,
                       CONJ_MEETINGS, HNL_MEETINGS)

HARD["meetings"] = HARD_MEETINGS
TERSE["meetings"] = TERSE_MEETINGS
PEOPLE["meetings"] = PEOPLE_MEETINGS
NUMERIC["meetings"] = NUMERIC_MEETINGS
CERT["meetings"] = CERT_MEETINGS
DERIVED["meetings"] = DERIVED_MEETINGS
CONJ["meetings"] = CONJ_MEETINGS
