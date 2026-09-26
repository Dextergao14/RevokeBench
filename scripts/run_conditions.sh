#!/usr/bin/env bash
# Run the evaluation conditions of the paper for one backbone on a set of episodes.
#
#   export OPENROUTER_API_KEY=...
#   scripts/run_conditions.sh qwen/qwen3.8-flash data/long100/splits/stage2_episodes.txt runs/qwen3.8
#
# Conditions: full context (per-task whole history), no memory, compact-8k, RTM, mem0, MemOS, Memp,
# Dynamic Cheatsheet.  Memory conditions share an 8,000-token raw window and a 4,000-token recall
# budget; every memory system uses the agent's backbone for its own calls.  The framework
# backends need their pinned packages (eval/memory/requirements-*.txt).
set -euo pipefail
MODEL=${1:?model id}; IDS_FILE=${2:?file with one episode id per line}; OUT=${3:?output dir}
BLIND=${BLIND:-data/long100/long100_blind.jsonl.gz}
WORKERS=${WORKERS:-10}
IDS=$(grep -v '^#' "$IDS_FILE" | grep -v '^$' | paste -s -d, -)
CONDITIONS=${CONDITIONS:-"full none compact ruletrack mem0 memos memp dynamic_cheatsheet"}
mkdir -p "$OUT"
for c in $CONDITIONS; do
  case "$c" in
    full)   python3 eval/adapters/openrouter_runner.py --blind "$BLIND" --models "$MODEL" --items "$IDS" \
              --workers "$WORKERS" --out "$OUT/full" ;;
    mem0)   CFG='{"max_tokens": 6000}' ;;
    dynamic_cheatsheet) CFG='{"curator_max_tokens": 8000}' ;;
    *)      CFG='{}' ;;
  esac
  [ "$c" = full ] && continue
  # no memory: the agent acts on the task turn alone (a 1-token window keeps no history block)
  WINDOW=8000; [ "$c" = none ] && WINDOW=1
  python3 eval/adapters/memory_runner.py --blind "$BLIND" --model "$MODEL" --memory "$c" \
    --window "$WINDOW" --recall-budget 4000 --cfg "$CFG" --items "$IDS" --workers "$WORKERS" --out "$OUT/$c"
done
python3 scripts/long_report.py --full data/long100/long100_full.jsonl.gz --runs "$OUT"/*
