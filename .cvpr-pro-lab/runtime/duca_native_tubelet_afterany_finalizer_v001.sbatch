#!/usr/bin/env bash
#SBATCH --job-name=duca-nt3-finalize
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data/run01/sczc063/yuzibo/duca_native_tubelet_b3339112_20260829/slurm_logs/finalizer_%j.out

set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/duca_native_tubelet_b3339112_20260829
OUT="$ROOT/terminal_collection_1260184_1260185.txt"
TMP="$OUT.tmp.${SLURM_JOB_ID}"

{
  echo "collector_job=${SLURM_JOB_ID}"
  echo "collected_at=$(date --iso-8601=seconds)"
  echo "dependency_jobs=1260184,1260185"
  echo "job_accounting_begin"
  sacct -X -j 1260184,1260185 --format=JobIDRaw,State,ExitCode,Elapsed,Start,End -P -n
  echo "job_accounting_end"
  echo "terminal_checkpoints_begin"
  find "$ROOT" -type f -path '*/gpu1_id0/checkpoint/epoch_59.pth' -print | sort
  echo "terminal_checkpoints_end"
  echo "official_metrics_begin"
  find "$ROOT" -type f -name metrics_epoch59_ema.json -print | sort
  echo "official_metrics_end"
} > "$TMP"

checkpoint_count=$(find "$ROOT" -type f -path '*/gpu1_id0/checkpoint/epoch_59.pth' -print | wc -l)
metrics_count=$(find "$ROOT" -type f -name metrics_epoch59_ema.json -print | wc -l)
{
  echo "terminal_checkpoint_count=$checkpoint_count"
  echo "official_metrics_count=$metrics_count"
  if [[ "$checkpoint_count" -eq 2 && "$metrics_count" -eq 2 ]]; then
    echo "collection_status=EXPECTED_ARTIFACTS_PRESENT"
  else
    echo "collection_status=EXPECTED_ARTIFACTS_MISSING"
  fi
} >> "$TMP"

mv "$TMP" "$OUT"
