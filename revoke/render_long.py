"""
Render a scenario onto a long-form surface described by its corpus, padded to a
token target.

The engine, the events and the ground truth are untouched.  This decides how
the turns are laid out on the page and what carries the transcript to
300k-1M tokens.  Two layouts:

  sectioned   minutes / notes: a title, header lines, a Discussion section
              (chatter and other business, shuffled as whole items), a
              Decisions section (the rule events, in issue order), and an
              Actions section (unrelated follow-ups plus the graded task).
              The section header is a cue and the phrasing of a near-miss
              line in Discussion is a counter-cue; only the speaker settles it.
  flat        a chat log: rules, chatter and filler interleaved in one stream
              with a speaker on every line and nothing marking which is which.
              The task still comes last in its session.

Filler is drawn from the corpus's own slot templates (`CorpusPadder`), so each
scenario pads in its own register -- a companion chat does not talk about p95
latency.  Filler never names a governed entity and never uses the rule
register, so it cannot change the closure; scripts/validate_corpus.py enforces
both on the corpus and the padder rejects any line that slips through.
"""
from __future__ import annotations

import datetime as _dt
import random
from dataclasses import replace
from typing import Dict, List, Sequence

from .generator import Scenario, Turn

TOK = 4                                              # chars per token, approximate
MONTHS = ("January February March April May June July August September October "
          "November December").split()


def _tokens(turns: List[Turn]) -> int:
    return sum(len(t.text) + 1 for t in turns) // TOK


class CorpusPadder:
    """Draws filler lines from a corpus's slot templates, unique by construction."""

    def __init__(self, rng: random.Random, banned: Sequence[str], spec: Dict):
        self.rng = rng
        self.banned = [b.lower() for b in banned]
        self.people = [f"{f} {l}" for f in spec["first_names"] for l in spec["last_names"]]
        rng.shuffle(self.people)
        self.slots = spec["slots"]
        self.fams = spec["families"]
        self.side_t = spec["side"]
        self.act_t = spec["actions"]
        self.seen: set = set()
        # templates already used in the current session: the same sentence
        # shape twice on one page with different slot values ("... came in 32%
        # over" then "... came in 25% over") is what reads as generated, even
        # when the rendered strings differ
        self.used_tpl: set = set()

    def new_session(self) -> None:
        self.used_tpl = set()

    def person(self) -> str:
        return self.rng.choice(self.people)

    def _slots(self) -> Dict[str, str]:
        r = self.rng
        d = {k: r.choice(v) for k, v in self.slots.items()}
        p1, p2 = r.sample(self.people, 2)
        d.update({
            "p": p1, "p2": p2,
            "n_small": str(r.randint(2, 9)), "n_few": str(r.randint(2, 6)),
            "n_few2": str(r.randint(4, 12)), "n_weeks": str(r.randint(2, 8)),
            "n_days": str(r.randint(2, 14)), "n_min": str(r.randint(4, 55)),
            "n_min_small": str(r.randint(3, 12)), "n_people": str(r.randint(2, 12)),
            "n_tickets": str(r.randint(6, 70)), "n_k": str(r.choice((15, 20, 30, 45, 60, 80, 120, 200))),
            "n_big": str(r.randint(100, 900)),
            "pct": f"{r.randint(4, 40)}%", "n_pct_small": f"{r.randint(3, 18)} percent",
            "hour": f"{r.randint(6, 22):02d}:00",
            "mon": r.choice(MONTHS), "d": f"{r.choice(MONTHS)[:3]} {r.randint(1, 28)}",
            "tk": f"{r.choice(('OPS', 'REQ', 'INC', 'CHG', 'TSK'))}-{r.randint(100, 9999)}",
        })
        return d

    def _draw(self, pool: Sequence[str], tries: int = 14) -> str:
        s = ""
        fresh = [t for t in pool if t not in self.used_tpl] or list(pool)
        for _ in range(tries):
            tpl = self.rng.choice(fresh)
            s = tpl.format(**self._slots())
            # a slot value at the head of a sentence keeps its lowercase article
            # ("the plan picker is back"); a line always starts with a capital
            s = s[0].upper() + s[1:] if s else s
            if s in self.seen or any(b in s.lower() for b in self.banned):
                continue
            self.seen.add(s)
            self.used_tpl.add(tpl)
            return s
        return s

    def line(self, family: str = "") -> str:
        return self._draw(self.fams[family] if family else self.rng.choice(list(self.fams.values())))

    def side(self) -> str:
        return self._draw(self.side_t)

    def action(self) -> str:
        return self._draw(self.act_t)

    def block(self, lines: int = 0) -> List[str]:
        """A few lines on one subject.  Family is redrawn per line: taking every
        line of a block from one family exhausts its templates and is where
        visible repetition comes from."""
        fams = list(self.fams)
        main = self.rng.choice(fams)
        return [self.line(main if i == 0 or self.rng.random() < 0.5 else self.rng.choice(fams))
                for i in range(lines or self.rng.randint(2, 5))]


def render_long(sc: Scenario, corpus: Dict, seed: int = 0, target_tokens: int = 0) -> Scenario:
    """Return a copy of `sc` laid out on the corpus's surface, padded to target."""
    rng = random.Random(seed or sc.seed)
    S = corpus["surface"]
    flat = S["layout"] == "flat"
    pad = CorpusPadder(rng, banned=list(sc.entity_names.values()), spec=corpus["pad"])
    people = list(corpus["people"]["start"])
    by_session: Dict[int, List[Turn]] = {}
    for t in sc.turns:
        by_session.setdefault(t.session, []).append(t)

    base = _tokens(sc.turns)
    n_sessions = max(len(by_session), 1)
    need = max(target_tokens - base, 0)
    per_session_chars = (need * TOK) // n_sessions

    out: List[Turn] = []

    def add(session: int, kind: str, text: str, **kw) -> None:
        out.append(Turn(session, "user", kind, text, **kw))

    # hand-written blocks and follow-ups are drawn without replacement,
    # reshuffled only when the pool runs dry, so nothing repeats within a
    # stretch of the transcript shorter than the pool
    # A hand-written pool is small (40-60 items) and an episode has ~450
    # sessions, so unlimited cycling would show every item ten times verbatim.
    # Each pool is used at most MAX_CYCLES times; after that the combinatorial
    # padder, which never repeats, supplies the rest.
    MAX_CYCLES = 2
    pool: List[str] = []
    todo_pool: List[str] = []
    cycles = {"blocks": 0, "todo": 0}

    def story_block() -> List[str]:
        nonlocal pool
        if not pool:
            if cycles["blocks"] >= MAX_CYCLES:
                return pad.block(rng.randint(3, 5))
            pool = list(S["blocks"])
            rng.shuffle(pool)
            cycles["blocks"] += 1
        return pool.pop().split("\n")

    def follow_up() -> str:
        """A follow-up line: mostly combinatorial, sometimes a hand-written one."""
        nonlocal todo_pool
        if rng.random() < 0.7:
            return pad.action()
        if not todo_pool:
            if cycles["todo"] >= MAX_CYCLES:
                return pad.action()
            todo_pool = list(S["action_items"])
            rng.shuffle(todo_pool)
            cycles["todo"] += 1
        return todo_pool.pop()

    # a running calendar: sessions are one to three days apart
    day = _dt.date(2025, 1, 6)

    def pad_chunk(session: int) -> List[Turn]:
        r = rng.random()
        if r < 0.12:
            lines = story_block()
        elif r < 0.9:
            lines = pad.block()
        else:
            lines = [pad.side()]
        if flat:
            # a chat block is a mini-thread between one or two filler people
            who = [pad.person() for _ in range(rng.randint(1, 3))]
            lines = [f"{rng.choice(who)}: {x}" for x in lines]
        return [Turn(session, "user", "pad", x) for x in lines]

    sessions = sorted(by_session.items())
    notice = next((t for ts in by_session.values() for t in ts if t.kind == "notice"), None)
    if notice is not None:
        add(notice.session, "meta", S["notice_frame"][0])
        out.append(notice)
        add(notice.session, "meta", S["notice_frame"][1])

    for seq, (session, turns) in enumerate(sessions, 1):
        pad.new_session()
        day += _dt.timedelta(days=rng.randint(1, 3))
        date = f"{day.strftime('%a')} {day.day} {MONTHS[day.month - 1][:3]} {day.year}"
        present = rng.sample(people, min(len(people), rng.randint(3, 6)))
        add(session, "meta", rng.choice(S["session_titles"]).format(n=seq, date=date))
        first_people = True
        for line in rng.sample(S["header_lines"], rng.randint(1, 2)):
            # the full attendee list goes on one line; a second line that names
            # people (late joiners, apologies) gets a subset of it
            if "{people}" in line:
                # a line with both a count and a list is the attendee list and
                # gets everyone; any other people line gets a subset
                full = "{n}" in line or first_people
                who = present if full else rng.sample(present, max(1, len(present) // 2))
                first_people = False
                line = line.replace("{n}", str(len(who)), 1)
            else:
                who = present
            # each remaining {n} is its own draw, so a line never says
            # "scheduled for 7 minutes; ran 7 over"
            while "{n}" in line:
                v = rng.choice((25, 30, 45, 50, 60)) if "minute" in line and "ran" not in line.split("{n}")[0] \
                    else rng.randint(2, 12)
                line = line.replace("{n}", str(v), 1)
            add(session, "meta", line.format(people=", ".join(who), date=date))
        if rng.random() < 0.25:
            add(session, "meta", rng.choice(S["artifacts"]))

        rules = [t for t in turns if t.kind == "update"]
        chat = [t for t in turns if t.kind in ("noise", "filler")]
        probes = [t for t in turns if t.kind == "probe"]

        budget = per_session_chars
        chunks: List[List[Turn]] = []
        while budget > 0:
            ch = pad_chunk(session)
            chunks.append(ch)
            budget -= sum(len(t.text) for t in ch)

        if flat:
            # one stream: rule and chatter turns keep their order; filler
            # threads land between them at random
            stream: List[List[Turn]] = [[t] for t in turns if t.kind in ("update", "noise", "filler")]
            for ch in chunks:
                stream.insert(rng.randint(0, len(stream)), ch)
            for ch in stream:
                out += [replace(t, session=session) for t in ch]
            for t in probes:
                out.append(t)
                add(session, "pad", f"{pad.person()}: {pad.action()}")
            continue

        # -- sectioned
        chunks += [[t] for t in chat]
        if chunks:
            add(session, "meta", rng.choice(S["sections"]["discussion"]))
            rng.shuffle(chunks)
            for ch in chunks:
                out += [replace(t, session=session) for t in ch]
        if rules:
            add(session, "meta", rng.choice(S["sections"]["decisions"]))
            out += rules
        if probes or rng.random() < 0.5:
            add(session, "meta", rng.choice(S["sections"]["actions"]))
            for _ in range(rng.randint(1, 3)):
                add(session, "pad", follow_up())
            for t in probes:
                out.append(t)
                add(session, "pad", pad.action())

    new = replace(sc, turns=out)
    pos = {t.probe_id: i for i, t in enumerate(out) if t.probe_id}
    for p in new.probes:
        p.turn_index = pos.get(p.probe_id, -1)
    new.meta = dict(sc.meta)
    pads = [t.text for t in out if t.kind == "pad"]
    new.meta.update({"format": S["layout"], "target_tokens": target_tokens,
                     "tokens": _tokens(out), "pad_turns": len(pads),
                     "pad_dup_rate": round(1 - len(set(pads)) / max(len(pads), 1), 4),
                     "meta_turns": sum(1 for t in out if t.kind == "meta")})
    return new
