#!/bin/bash
set -euo pipefail

ORIGINAL_ARGV=("$0" "$@")

usage() {
  cat <<'EOF_USAGE'
Usage: bash scripts/duca_unified_fullmatrix/submit_all.sh \
  --repo-root <remote clean checkout> \
  --revision <final commit sha> \
  --run-root <run output root> \
  --base <remote base path> \
  --account <slurm account> \
  --partition <gpu partition> \
  --qos <slurm qos> \
  --max-concurrent <train array concurrency>
EOF_USAGE
}

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
REVISION="${REVISION:-}"
RUN_ROOT="${RUN_ROOT:-}"
ACCOUNT="${ACCOUNT:-sczc063}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-normal}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --base)
      BASE="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
if [[ -z "$REVISION" ]]; then
  REVISION="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="${BASE}/experiments/duca_unified_fullmatrix_${REVISION:0:12}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrent must be a positive integer: $MAX_CONCURRENT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT" "${BASE}/slurm_logs"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; run this script on the N16R4 Slurm login node." >&2
  exit 2
fi

cd "$PROJECT_DIR"
ACTUAL_REVISION="$(git rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
  echo "checkout revision mismatch: expected $REVISION, got $ACTUAL_REVISION" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked working tree is not clean at $PROJECT_DIR" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

export PROJECT_DIR RUN_ROOT

SLURM_SHARED_ARGS=()
if [[ -n "$ACCOUNT" ]]; then
  SLURM_SHARED_ARGS+=("--account=$ACCOUNT")
fi
if [[ -n "$QOS" ]]; then
  SLURM_SHARED_ARGS+=("--qos=$QOS")
fi
SLURM_GPU_ARGS=("${SLURM_SHARED_ARGS[@]}")
if [[ -n "$PARTITION" ]]; then
  SLURM_GPU_ARGS+=("--partition=$PARTITION")
fi

preflight=$(sbatch --parsable "${SLURM_GPU_ARGS[@]}" scripts/duca_unified_fullmatrix/preflight.sbatch)
train=$(sbatch --parsable "${SLURM_GPU_ARGS[@]}" --dependency=afterok:$preflight --array=0-40%$MAX_CONCURRENT scripts/duca_unified_fullmatrix/train_eval_array.sbatch)
cost=$(sbatch --parsable "${SLURM_GPU_ARGS[@]}" --dependency=afterok:$train scripts/duca_unified_fullmatrix/cost_array.sbatch)
boot=$(sbatch --parsable "${SLURM_SHARED_ARGS[@]}" --dependency=afterok:$train scripts/duca_unified_fullmatrix/bootstrap_array.sbatch)
finalize=$(sbatch --parsable "${SLURM_SHARED_ARGS[@]}" --dependency=afterok:$train:$cost:$boot scripts/duca_unified_fullmatrix/finalize.sbatch)
audit=$(sbatch --parsable "${SLURM_SHARED_ARGS[@]}" --dependency=afterany:$train:$cost:$boot:$finalize scripts/duca_unified_fullmatrix/audit_afterany.sbatch)

printf -v SUBMISSION_ARGV '%q ' "${ORIGINAL_ARGV[@]}"
SUBMISSION_ARGV="${SUBMISSION_ARGV%% }"

python - "$RUN_ROOT" "$PROJECT_DIR" "$REVISION" "$PROJECT_DIR" "$SUBMISSION_ARGV" "$preflight" "$train" "$cost" "$boot" "$finalize" "$audit" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

run_root = pathlib.Path(sys.argv[1])
project_dir = pathlib.Path(sys.argv[2])
revision = sys.argv[3]
remote_repo = sys.argv[4]
submission_argv = sys.argv[5]
job_ids = sys.argv[6:12]

def atomic_write(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

def atomic_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    atomic_write(dst, src.read_bytes())

run_root.mkdir(parents=True, exist_ok=True)
atomic_copy(project_dir / "scripts/duca_unified_fullmatrix/matrix.tsv", run_root / "matrix.tsv")
atomic_copy(project_dir / "docs/experiments/DUCA_UNIFIED_FULLMATRIX_FREEZE.md", run_root / "scientific_freeze.md")

tracked = subprocess.check_output(
    ["git", "ls-files"],
    cwd=project_dir,
    text=True,
    encoding="utf-8",
).splitlines()
hash_lines = []
for rel_path in sorted(tracked):
    src = project_dir / rel_path
    if src.is_file():
        hash_lines.append(f"{hashlib.sha256(src.read_bytes()).hexdigest()}  {rel_path}")
atomic_write(run_root / "source_manifest_sha256.txt", ("\n".join(hash_lines) + "\n").encode("utf-8"))

payload = {
    "schema_version": "duca_unified_slurm_submission_v1",
    "matrix_id": "DUCA-UNIFIED-FULLMATRIX-v001-20260902",
    "preflight_job_id": job_ids[0],
    "train_eval_array_job_id": job_ids[1],
    "cost_array_job_id": job_ids[2],
    "bootstrap_array_job_id": job_ids[3],
    "finalizer_job_id": job_ids[4],
    "audit_afterany_job_id": job_ids[5],
    "final_commit": revision,
    "remote_repo": remote_repo,
    "run_root": str(run_root),
    "submission_argv": submission_argv,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
atomic_write(run_root / "submission_manifest.json", data)
atomic_write(run_root / "slurm_submission_manifest.json", data)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
