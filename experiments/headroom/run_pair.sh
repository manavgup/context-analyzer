#!/usr/bin/env bash
#
# run_pair.sh — run ONE matched task pair for the headroom experiment (issue #81).
#
# Usage:
#   ./run_pair.sh tasks/<task>.yaml <pair-index> [--order random|PH|HP] [--model <model>] [--dry-run]
#
#   pair-index   Integer distinguishing repetitions of the same task (1, 2, ...).
#   --order      Which arm runs first. P = plain, H = headroom. Default: random
#                (recorded in the manifest either way — see METHODOLOGY.md,
#                "Ordering effects").
#   --model      Claude model to pin for BOTH arms (default: leave CLI default,
#                but pinning is strongly recommended — see METHODOLOGY.md).
#   --dry-run    Print every command instead of executing sessions. No API spend.
#
# What it does per arm:
#   1. Creates a clean workspace dir:  workspaces/<task-id>-pair<idx>/<arm>/
#   2. Runs the task's `setup:` block inside it (clone + pin + state injection).
#   3. Launches the session non-interactively:
#        plain arm:     claude -p "<prompt>" --output-format json ...
#        headroom arm:  headroom wrap claude -- -p "<prompt>" --output-format json ...
#      `claude -p` (print mode) is the mechanism: it runs one full agentic session
#      without a TTY and emits a JSON result envelope whose `session_id` field is
#      the same session id context-analyzer ingests from ~/.claude/projects.
#      NOTE: confirm the exact `headroom wrap` argv form against the pinned
#      headroom version before running; adjust HEADROOM_CMD below if it differs.
#   4. Runs the task's `success_check:` block in the workspace (exit 0 = success).
#   5. Appends one JSON line per arm to manifest.jsonl mapping arm -> session_id.
#
# After running all pairs:  python analyze.py --manifest manifest.jsonl --ingest
#
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
MANIFEST="$EXP_DIR/manifest.jsonl"

# Prefer the project venv python (has pyyaml); fall back to python3.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# --- args -------------------------------------------------------------------
TASK_FILE="${1:?usage: run_pair.sh tasks/<task>.yaml <pair-index> [--order random|PH|HP] [--model <m>] [--dry-run]}"
PAIR_INDEX="${2:?missing pair-index}"
shift 2

ORDER="random"
MODEL=""
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --order) ORDER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- parse task YAML --------------------------------------------------------
task_field() {  # task_field <key>
  "$PY" -c '
import sys, yaml
with open(sys.argv[1]) as f:
    task = yaml.safe_load(f)
print(task[sys.argv[2]], end="")
' "$TASK_FILE" "$1"
}

TASK_ID="$(task_field id)"
PROMPT="$(task_field prompt)"
SETUP="$(task_field setup)"
SUCCESS_CHECK="$(task_field success_check)"

# --- arm order --------------------------------------------------------------
if [ "$ORDER" = "random" ]; then
  if [ $((RANDOM % 2)) -eq 0 ]; then ORDER="PH"; else ORDER="HP"; fi
fi
case "$ORDER" in
  PH) ARMS=(plain headroom) ;;
  HP) ARMS=(headroom plain) ;;
  *) echo "invalid --order: $ORDER (use random|PH|HP)" >&2; exit 2 ;;
esac

# --- version pins (recorded per manifest row) -------------------------------
CLAUDE_VERSION="$(claude --version 2>/dev/null || echo unknown)"
HEADROOM_VERSION="$(headroom --version 2>/dev/null || echo unknown)"

MODEL_ARGS=()
[ -n "$MODEL" ] && MODEL_ARGS=(--model "$MODEL")

echo "task=$TASK_ID pair=$PAIR_INDEX order=$ORDER model=${MODEL:-<cli default>}"
echo "claude=$CLAUDE_VERSION headroom=$HEADROOM_VERSION"

run_arm() {  # run_arm <arm> <order_position>
  local arm="$1" position="$2"
  local ws="$EXP_DIR/workspaces/${TASK_ID}-pair${PAIR_INDEX}/${arm}"
  local result_json="$ws/.experiment-result.json"

  echo "--- arm=$arm (position $position) workspace=$ws"
  rm -rf "$ws"
  mkdir -p "$ws"

  # 1. Workspace setup (clone + pin + injected state). Fails the arm on error.
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] setup in %s:\n%s\n' "$ws" "$SETUP"
  else
    ( cd "$ws" && bash -euo pipefail -c "$SETUP" )
  fi

  # 2. Launch the session. --dangerously-skip-permissions is required for
  #    unattended print mode when the task edits files; the workspace is a
  #    throwaway clone, which is why that is acceptable here.
  local -a session_cmd
  # ${MODEL_ARGS[@]+...} guard: empty-array expansion is an error under
  # `set -u` on bash 3.2 (macOS default).
  if [ "$arm" = "plain" ]; then
    session_cmd=(claude -p "$PROMPT" --output-format json ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} --dangerously-skip-permissions)
  else
    # HEADROOM_CMD: verify against your pinned headroom version's docs.
    session_cmd=(headroom wrap claude -- -p "$PROMPT" --output-format json ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} --dangerously-skip-permissions)
  fi

  local started_at finished_at session_id success cost_usd
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] (cd $ws && ${session_cmd[*]} > $result_json)"
    session_id="dry-run"
    success=null
    cost_usd=null
  else
    ( cd "$ws" && "${session_cmd[@]}" > "$result_json" ) || \
      echo "WARNING: session command exited non-zero for arm=$arm (continuing; success_check decides)" >&2

    # 3. Session id from the print-mode JSON envelope. This is the id
    #    context-analyzer ingests (transcript in ~/.claude/projects/<proj>/<id>.jsonl).
    session_id="$("$PY" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1]))["session_id"], end="")
except Exception:
    print("unknown", end="")
' "$result_json")"

    # 3b. Authoritative model-correct cost from the same envelope
    #     (`total_cost_usd`). This is the cost source analyze.py reports;
    #     the DB's sessions.total_cost_usd is computed with fixed Opus rates
    #     and is only a warned-about fallback.
    cost_usd="$("$PY" -c '
import json, sys
try:
    v = json.load(open(sys.argv[1])).get("total_cost_usd")
    print("null" if v is None else json.dumps(v), end="")
except Exception:
    print("null", end="")
' "$result_json")"

    # 4. Objective success check, in the workspace, fail-fast semantics.
    if ( cd "$ws" && bash -euo pipefail -c "$SUCCESS_CHECK" ); then
      success=true
    else
      success=false
    fi
  fi
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "    session_id=$session_id success=$success cost_usd=$cost_usd"

  # 5. Manifest row: the arm <-> session_id mapping analyze.py joins on.
  #    cost_usd is the authoritative model-correct cost from the print-mode
  #    JSON envelope (null when unavailable, e.g. dry runs).
  "$PY" -c '
import json, sys
row = {
    "task_id": sys.argv[1],
    "pair": int(sys.argv[2]),
    "arm": sys.argv[3],
    "order_position": int(sys.argv[4]),
    "order": sys.argv[5],
    "session_id": sys.argv[6],
    "success": json.loads(sys.argv[7]),
    "cost_usd": json.loads(sys.argv[8]),
    "started_at": sys.argv[9],
    "finished_at": sys.argv[10],
    "model": sys.argv[11] or None,
    "claude_version": sys.argv[12],
    "headroom_version": sys.argv[13],
    "workspace": sys.argv[14],
}
with open(sys.argv[15], "a") as f:
    f.write(json.dumps(row) + "\n")
' "$TASK_ID" "$PAIR_INDEX" "$arm" "$position" "$ORDER" "$session_id" "$success" \
  "$cost_usd" "$started_at" "$finished_at" "$MODEL" "$CLAUDE_VERSION" "$HEADROOM_VERSION" \
  "$ws" "$MANIFEST"
}

run_arm "${ARMS[0]}" 1
run_arm "${ARMS[1]}" 2

echo "pair complete -> $MANIFEST"
