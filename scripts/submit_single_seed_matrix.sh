#!/bin/bash
# ==============================================================================
# Master Slurm Dispatcher for Unified Single-Seed DUCA / ZoomToken / H65-Pro Matrix
# ==============================================================================
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:-${BASE}/projects/duca_unified_single_seed_20260903}"
PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-sczc063}"
QOS="${QOS:-normal}"
SEED="${SEED:-3407}"
ROUTE="${ROUTE:-all}"
TIME_LIMIT="${TIME_LIMIT:-18:00:00}"

LOG_DIR="${BASE}/slurm_logs/single_seed_$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "DISPATCHING SINGLE-SEED MATRIX TRAINING (seed=${SEED}, route=${ROUTE})"
echo "Project Directory: ${PROJECT_DIR}"
echo "Log Directory:     ${LOG_DIR}"
echo "========================================================================"

# Declarative map of routes to their config paths
declare -A ROUTE_CONFIGS

ROUTE_CONFIGS["h65_pro"]="
configs/adatad/thumos/h65_pro/h65_pro_c0.py
configs/adatad/thumos/h65_pro/h65_pro_c1.py
configs/adatad/thumos/h65_pro/h65_pro_c2.py
configs/adatad/thumos/h65_pro/h65_pro_c3.py
configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py
configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py
configs/adatad/thumos/h65_pro/h65_pro_ref_mnv3fc384.py
configs/adatad/thumos/h65_pro/h65_pro_f01.py
configs/adatad/thumos/h65_pro/h65_pro_f02.py
configs/adatad/thumos/h65_pro/h65_pro_f03.py
configs/adatad/thumos/h65_pro/h65_pro_f04.py
configs/adatad/thumos/h65_pro/h65_pro_f05.py
configs/adatad/thumos/h65_pro/h65_pro_f06.py
configs/adatad/thumos/h65_pro/h65_pro_f07.py
configs/adatad/thumos/h65_pro/h65_pro_f08.py
configs/adatad/thumos/h65_pro/h65_pro_f09.py
configs/adatad/thumos/h65_pro/h65_pro_f10.py
configs/adatad/thumos/h65_pro/h65_pro_f11.py
configs/adatad/thumos/h65_pro/h65_pro_f12.py
configs/adatad/thumos/h65_pro/h65_pro_f13.py
configs/adatad/thumos/h65_pro/h65_pro_f14.py
configs/adatad/thumos/h65_pro/h65_pro_f15.py
configs/adatad/thumos/h65_pro/h65_pro_f16.py
"

ROUTE_CONFIGS["ctdp"]="
configs/adatad/thumos/duca_ctdp_geometry_g0.py
configs/adatad/thumos/duca_ctdp_geometry_g1.py
configs/adatad/thumos/duca_ctdp_geometry_g2.py
configs/adatad/thumos/duca_ctdp_geometry_g3.py
"

ROUTE_CONFIGS["duca_unified"]="
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_u0_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a00_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a01_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a10_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a11_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b00_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b10_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b11_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c01_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c11_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_d1_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_e01_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_e11_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_f11_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_g10_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_g11_seed3407.py
configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_h0_seed3407.py
"

ROUTE_CONFIGS["bafdr"]="
configs/adatad/thumos/bafdr_k16_full_seed4407.py
configs/adatad/thumos/bafdr_k16_nokd_seed4407.py
configs/adatad/thumos/bafdr_k16_late_seed4407.py
configs/adatad/thumos/bafdr_k16_g96_seed4407.py
configs/adatad/thumos/bafdr_k16_d160_seed4407.py
configs/adatad/thumos/bafdr_k16_u16_uniform_a0_seed4407.py
configs/adatad/thumos/bafdr_k16_u128_all48_a0_seed4407.py
"

ROUTE_CONFIGS["et_trc"]="
configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py
configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed4407.py
configs/adatad/thumos/continuous_roi_s2_v3_d160_seed4407.py
configs/adatad/thumos/continuous_roi_s2_v3_g96_seed4407.py
configs/adatad/thumos/continuous_roi_s2_v3_u128_a0_seed4407.py
"

ROUTE_CONFIGS["evidence_recovery"]="
configs/adatad/thumos/duca_evidence_recovery_full.py
configs/adatad/thumos/duca_evidence_recovery_no_coverage.py
configs/adatad/thumos/duca_evidence_recovery_no_merge.py
configs/adatad/thumos/duca_evidence_recovery_no_recovery.py
configs/adatad/thumos/duca_evidence_recovery_no_robust.py
configs/adatad/thumos/duca_evidence_recovery_no_time.py
configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py
"

ROUTES_TO_RUN=()
if [[ "$ROUTE" == "all" ]]; then
  ROUTES_TO_RUN=("h65_pro" "ctdp" "duca_unified" "bafdr" "et_trc" "evidence_recovery")
else
  ROUTES_TO_RUN=("$ROUTE")
fi

TOTAL_SUBMITTED=0
for r in "${ROUTES_TO_RUN[@]}"; do
  echo ">>> Processing Route: ${r}"
  CONFIG_LIST="${ROUTE_CONFIGS[$r]:-}"
  if [[ -z "$CONFIG_LIST" ]]; then
    echo "Warning: Route $r not found or empty."
    continue
  fi

  for cfg in $CONFIG_LIST; do
    if [[ -z "$cfg" || "$cfg" =~ ^# ]]; then
      continue
    fi
    job_tag="$(basename "$cfg" .py)"
    sbatch_script=$(mktemp /tmp/sbatch_${job_tag}_XXXXXX.sh)
    port=$((29000 + RANDOM % 10000))

    cat <<SBATCH_EOF > "$sbatch_script"
#!/bin/bash
#SBATCH --job-name=${job_tag}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${r}_${job_tag}_%j.out
#SBATCH --error=${LOG_DIR}/${r}_${job_tag}_%j.err

source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source ${BASE}/conda_envs/opentad/bin/activate
cd ${PROJECT_DIR}

echo "=== STARTING JOB ${job_tag} (Route: ${r}, Seed: ${SEED}) ==="
echo "Node: \$(hostname), CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-none}"

torchrun --nproc_per_node=1 --master_port=${port} tools/train.py ${cfg} --seed ${SEED}
echo "=== COMPLETED JOB ${job_tag} ==="
SBATCH_EOF

    chmod +x "$sbatch_script"
    job_id=$(sbatch --parsable "$sbatch_script")
    rm -f "$sbatch_script"
    echo "  [SUBMITTED] JobID: ${job_id} | Route: ${r} | Config: ${cfg}"
    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
  done
done

echo "========================================================================"
echo "SUMMARY: Successfully submitted ${TOTAL_SUBMITTED} jobs to Slurm queue."
echo "Check queue with: squeue -u ${USER}"
echo "========================================================================"
