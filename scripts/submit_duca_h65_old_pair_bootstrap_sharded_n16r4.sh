#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_H65_BOOTSTRAP_SUBMIT][FAIL] $*" >&2; exit 1; }
: "${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
: "${DUCA_BOOTSTRAP_COMMIT:?DUCA_BOOTSTRAP_COMMIT is required}"
: "${DUCA_BOOTSTRAP_SHARD_ROOT:?DUCA_BOOTSTRAP_SHARD_ROOT is required}"
: "${DUCA_BOOTSTRAP_OUTPUT_ROOT:?DUCA_BOOTSTRAP_OUTPUT_ROOT is required}"
: "${DUCA_INPUT_IDENTITY_PATH:?DUCA_INPUT_IDENTITY_PATH is required}"

ROOT="$(cd -- "$DUCA_REPO_ROOT" && pwd -P)"
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$DUCA_BOOTSTRAP_COMMIT" ]] || fail "checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "checkout must be clean"
[[ -r "$DUCA_INPUT_IDENTITY_PATH" ]] || fail "input identity is not readable"
[[ ! -e "$DUCA_BOOTSTRAP_SHARD_ROOT" ]] || fail "shard root already exists"
[[ ! -e "$DUCA_BOOTSTRAP_OUTPUT_ROOT" ]] || fail "merge output root already exists"

array_job="$(sbatch --parsable --export=ALL scripts/run_duca_h65_old_pair_bootstrap_shard_n16r4.sbatch)"
array_job="${array_job%%;*}"
[[ "$array_job" =~ ^[0-9]+$ ]] || fail "array submission did not return a numeric job id"
merge_job="$(sbatch --parsable --export=ALL --dependency="afterok:$array_job" \
  scripts/run_duca_h65_old_pair_bootstrap_merge_n16r4.sbatch)"
merge_job="${merge_job%%;*}"
[[ "$merge_job" =~ ^[0-9]+$ ]] || fail "merge submission did not return a numeric job id"

printf 'DUCA_H65_BOOTSTRAP_ARRAY_JOB=%s\n' "$array_job"
printf 'DUCA_H65_BOOTSTRAP_MERGE_JOB=%s\n' "$merge_job"
printf 'DUCA_H65_BOOTSTRAP_DEPENDENCY=afterok:%s\n' "$array_job"
