# RevokeBench release: 100 episodes, 1,202 graded tasks

Each episode (*history*) is one continuous record of 428–671 sessions (412–637 rendered) in one of six
settings, 344k–935k tokens long, with 10–15 graded tasks. Summary statistics: `SUMMARY.txt`.

| file | contents | who may read it |
|---|---|---|
| `long100_blind.jsonl.gz` | one JSON object per episode: rendered sessions (turn text, no kind or speaker fields), tool surface, setting text, entity display names, and for each task its session, option set and numeric parameter | the system under test |
| `long100_full.jsonl.gz` | the same episodes with ground truth: event log, per-session closures, labelled tasks (licensed / compliant / violating options, blamed rules, stale-trap set, defeated instances, difficulty features) and `probe_pool`, every task the generator emitted (21,996 in total) | the grader only |
| `manifest.jsonl` | one row per episode: id, setting, layout, target and actual tokens, sessions, cycles, tasks, traps, far-span tasks, mean weight, pool size, generation seed, rejected seeds, filler repetition, sha256 prefix of the blind item | anyone |
| `splits/stage2_episodes.txt` | the nine episodes used for the Stage 2 memory-framework comparison on Qwen3.8-Flash | anyone |

Only blind data may reach an agent. The runners refuse a file whose name contains `full` or
whose tasks carry labels.

## Length bands (paper Table 5)

| target tokens | cycles | tasks per episode | episodes |
|---|---|---|---|
| 300k | 6 | 10 | 24 |
| 440k | 7 | 11 | 22 |
| 580k | 8 | 12 | 18 |
| 720k | 9 | 13 | 18 |
| 830k | 10 | 15 | 18 |

Tokens are estimated at four characters per token.

## Regenerating and reselecting

Generation is deterministic given a seed:

```
python3 eval/build_long.py --n 100 --jobs 8 --seed 20260912 --out data/long100_regen
```

rebuilds every episode (the default `--seed` is not the release's; see the docstring of
`eval/build_long.py` for rebuilding one episode). The `sha256_blind` column is the sha256 prefix of
that episode's gzip member inside `long100_blind.jsonl.gz` (the file is a concatenation of
per-episode members). A rebuilt `.gz` differs from its member only in the gzip header MTIME
(bytes 4-7); the decompressed JSON is byte-identical.

`probe_pool` holds every generated task with its labels and rendered task text, so a different
task set can be asked of the same history without regenerating it:

```
python3 scripts/reselect_probes.py --full data/long100/long100_full.jsonl.gz --id <episode id> \
    --k 12 --strategy traps --out data/reselect/
```

Labels of reselected tasks remain valid, because they depend only on the event log. Set-level
acceptance checks run on the rendered text, so a reselected set should be passed through
`revoke/verify.py` before it is used as a benchmark (paper Appendix C.3).
