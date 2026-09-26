# RevokeBench

Code and data for **RevokeBench: Evaluating LLM Agent Behaviors and Memory Under Evolving
Constraints** (anonymous submission).

RevokeBench asks whether an agent acts on the rules *currently* in force after hundreds of
sessions in which those rules were approved, prohibited, conditioned, overridden, reinstated and
withdrawn. It contains 100 histories in six settings (344k–935k tokens each) with 1,202 graded tool-use
tasks, and a labelled pool of 21,996 tasks for alternative selections. Every label is derived by a
formal solver over a two-layer rule representation, so grading needs no LLM judge.

## Repository layout

| path | what it is | paper |
|---|---|---|
| `revoke/logic.py` | two-layer representation: persistent rule base, derivation layer (grounding, applicability, defeat graph, grounded extension, forward chaining, outer fixpoint), behavioural check | §2, App. A |
| `revoke/events.py` | the six update operations (ADD, SUPERSEDE, CONDITION, CONFLICT, SUPPORT, RETRACT) | §2, App. A |
| `revoke/motifs.py`, `revoke/motifs_hard.py`, `revoke/hard.py`, `revoke/generator.py` | the 16 rule-update patterns, scheduling and history-level structure (speaker hierarchy, certification regime, anchors, referential updates) | §3, App. C.2 |
| `revoke/render_long.py`, `revoke/domains/` | rendering, padding and the six setting corpora (`corpora/*.json`) | §3, App. C.1, C.3 |
| `revoke/verify.py` | the seven acceptance checks | App. C.4 |
| `revoke/difficulty.py`, `revoke/serialize.py` | difficulty features and weights; full and blind item formats | §4, App. D |
| `eval/build_long.py` | builds the release from seeds (generate, verify, render, select tasks, re-verify) | §3, App. C |
| `eval/grade.py` | deterministic grader: argument resolution, per-task verdict, exam score | §2, §4, App. D.1 |
| `eval/adapters/openrouter_runner.py` | full-context condition: one request per task with the whole history | §5 |
| `eval/adapters/memory_runner.py` | every bounded-memory condition: 8k-token raw window, memory backend, recall budget | §5 |
| `eval/memory/base.py` | backend interface, LLM wrapper, `none` and `compact` (compact-8k running notes) | §5.2 |
| `eval/memory/ruletrack.py` | Rule-Tracking Memory (RTM) | §5.4 |
| `eval/memory/{mem0,memos,memp,dynamic_cheatsheet}.py` | adapters for mem0, MemOS, Memp (vendored EvolveLab code in `eval/memory/vendor/`) and Dynamic Cheatsheet; one `.md` per adapter documents deviations from upstream | §5.3, App. F |
| `scripts/long_report.py` | per-condition tables (violation, completion, abstention, exam, far, trap, weighted violation) | Tables 1–2 |
| `scripts/ruletrack_diag.py`, `scripts/ruletrack_compliance.py` | RTM extraction recall, verdict agreement and violation decomposition | §5.4 |
| `scripts/reselect_probes.py` | new task selections from the pool (uniform, traps first, longest span first, one event kind, explicit ids) | App. C.3 |
| `scripts/validate_corpus.py` | corpus validator (filler and rank-neutrality invariants) | App. C.1 |
| `scripts/test_engine.py`, `scripts/smoke_memory.py` | engine tests; offline smoke test of a memory backend with a fake LLM | — |
| `data/long100/` | the released histories, manifest and Stage 2 split (see its README) | §3 |
| `docs/GRADING.md`, `docs/corpus_schema.md` | grading rules in detail; corpus schema for adding a setting | App. C.1, D |

## Setup

Python ≥ 3.10. The generator, solver, verifier, grader and runners use only the standard library.
Memory-framework backends pin their own packages in `eval/memory/requirements-*.txt`.
Model calls go through the OpenRouter chat-completions API:

```
export OPENROUTER_API_KEY=...
```

## Quick start

Check the engine and the release offline:

```
python3 scripts/test_engine.py
python3 scripts/smoke_memory.py --memory compact --limit 3      # fake LLM, no API calls
```

Run one backbone under all conditions on the Stage 2 episodes and print the tables:

```
scripts/run_conditions.sh qwen/qwen3.8-flash data/long100/splits/stage2_episodes.txt runs/qwen3.8
```

or run a single condition:

```
python3 eval/adapters/memory_runner.py --blind data/long100/long100_blind.jsonl.gz \
    --model qwen/qwen3.8-flash --memory compact --window 8000 --recall-budget 4000 --out runs/compact
python3 eval/adapters/openrouter_runner.py --blind data/long100/long100_blind.jsonl.gz \
    --models qwen/qwen3.8-flash --out runs/full
python3 scripts/long_report.py --full data/long100/long100_full.jsonl.gz --runs runs/compact runs/full
```

`--memory` takes `none`, `compact`, `ruletrack`, `mem0`, `memos`, `memp` or `dynamic_cheatsheet`.
The settings used in the paper (token budgets, completion budgets, top-k, embedder) are listed in
Appendix E and in `scripts/run_conditions.sh`.

## Evaluating your own agent or memory system

A system under test reads only `long100_blind.jsonl.gz`. It writes one JSON line per episode:

```
{"id": "<episode id>", "steps": [{"probe_id": "<task id>", "tool_calls": [{"name": "<act tool>", "arguments": {"<param>": "<option>"}}]}, ...]}
```

Grade it with `scripts/long_report.py --full data/long100/long100_full.jsonl.gz --runs <dir>`, or
with `eval/grade.py --dataset data/long100/long100_full.jsonl.gz --trace <file>`. Only calls to the
act tool are scored. A task with no act call, an unresolvable argument, or a choice that is neither
licensed nor prohibited is an abstention.

A new memory system plugs in as `eval/memory/<name>.py` exposing a `Backend` class with the
interface in `eval/memory/base.py` (`begin_episode`, `observe`, `recall`, `record_action`,
`end_episode`, `stats`). It then runs with `--memory <name>`.

## Regenerating the data

```
python3 eval/build_long.py --n 100 --jobs 8 --seed 20260912 --out data/long100_regen
```

The build is deterministic given the seeds in `data/long100/manifest.jsonl`. See
`data/long100/README.md` for reselecting tasks from the pool.
