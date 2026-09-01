#!/usr/bin/env bash
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
set -euo pipefail

REPO=/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_ca40c9c_20260723
echo '=== PYTHON ==='
python --version
echo '=== TEST FORMAL CHECK ==='
sed -n '1,105p' "${REPO}/tools/test.py"
sed -n '90,180p' "${REPO}/tools/test.py"
echo '=== GENERATED CONFIG HEAD/SELECTOR ==='
CFG=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_detached/official60/work/gpu1_id0/duca_boundary_burst_soft_g0_no_feedback_fixed384_official60.py
sed -n '1,100p' "${CFG}"
grep -n -E 'frame_selector|selected_opt|gate_suite|variant|work_dir|pretrain' "${CFG}" | head -100 || true
echo '=== AUDIT ==='
cat /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_detached/official60/work/gpu1_id0/duca_selected_axis_training_audit.json
echo '=== TEST HELP ==='
cd "${REPO}"
python tools/test.py --help | sed -n '1,180p'
