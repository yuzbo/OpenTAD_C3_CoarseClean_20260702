$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$sshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
    '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', 'C:\Users\skywalker\.ssh\id_rsa',
    '-p', '22',
    '-l', 'sczc063@BSCC-N16R4',
    'ssh.cn-zhongwei-1.paracloud.com',
    "tr -d '\r' | bash -s"
)

$remote = @'
set -eo pipefail
source /etc/profile
set -u

BASE=/data/run01/sczc063/yuzibo
BRANCH=codex/duca-boundary-burst-20260722
COMMIT=4f81299500000000000000000000000000000000
COMMIT=$(git ls-remote https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git "refs/heads/${BRANCH}" | awk '{print $1}')
[[ "$COMMIT" == 4f81299* ]] || { echo "unexpected remote commit: $COMMIT" >&2; exit 11; }
SNAP=${BASE}/projects/opentad_duca_coarse_${COMMIT:0:7}_20260723
RUN_ROOT=${BASE}/projects/c3_lowres_action_probe/duca_coarse_backends_${COMMIT:0:7}_20260723_0015
OFFICIAL=${BASE}/projects/external_official_action_segmentation_repos_20260702

if [[ ! -d "$SNAP/.git" ]]; then
  git clone --quiet --branch "$BRANCH" --single-branch \
    https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git "$SNAP"
fi
git -C "$SNAP" checkout --quiet --detach "$COMMIT"
[[ "$(git -C "$SNAP" rev-parse HEAD)" == "$COMMIT" ]]
[[ -z "$(git -C "$SNAP" status --porcelain --untracked-files=normal)" ]]

module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
cd "$SNAP"
bash -n scripts/run_duca_coarse_backend_ablation_gpu1.sh
python -m py_compile tools/bata/train_lowres_action_probe.py tests/test_duca_coarse_backend_ablation.py
python -m pytest tests/test_duca_coarse_backend_ablation.py tests/test_c3_coarse_classifier_model_matrix.py -q

mkdir -p "$RUN_ROOT/jobs" "$RUN_ROOT/logs" "$RUN_ROOT/outputs"
printf 'backend\tjob_id\tsbatch\tout_dir\n' > "$RUN_ROOT/jobs.tsv"
printf 'name\tcommit\n' > "$RUN_ROOT/official_sources.tsv"
for pair in \
  "MS-TCN2:${OFFICIAL}/MS-TCN2" \
  "ASFormer:${OFFICIAL}/ASFormer" \
  "FACT:${OFFICIAL}/CVPR2024-FACT" \
  "Video-Mamba-ASFormer:${OFFICIAL}/video-mamba-suite"; do
  name=${pair%%:*}
  path=${pair#*:}
  printf '%s\t%s\n' "$name" "$(git -C "$path" rev-parse HEAD)" >> "$RUN_ROOT/official_sources.tsv"
done

backends=(official_ms_tcn2 official_asformer official_fact official_video_mamba_asformer)
names=(dcb_mstcn2 dcb_asformer dcb_fact dcb_vmamba)
for i in "${!backends[@]}"; do
  backend=${backends[$i]}
  name=${names[$i]}
  sbatch_file="$RUN_ROOT/jobs/${backend}.sbatch"
  out_dir="$RUN_ROOT/outputs/${backend}"
  cat > "$sbatch_file" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${name}
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/${backend}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${backend}-%j.err

source /etc/profile
set -euo pipefail
export PROJECT_DIR='${SNAP}'
export BACKEND='${backend}'
export OUT_DIR='${out_dir}'
export C3_OFFICIAL_ACTION_SEG_REPOS='${OFFICIAL}'
bash '${SNAP}/scripts/run_duca_coarse_backend_ablation_gpu1.sh'
EOF
  bash -n "$sbatch_file"
  raw_job=$(sbatch --parsable "$sbatch_file")
  job_id=${raw_job%%;*}
  printf '%s\t%s\t%s\t%s\n' "$backend" "$job_id" "$sbatch_file" "$out_dir" >> "$RUN_ROOT/jobs.tsv"
done

cat > "$RUN_ROOT/deployment.env" <<EOF
BRANCH=${BRANCH}
COMMIT=${COMMIT}
SNAPSHOT=${SNAP}
RUN_ROOT=${RUN_ROOT}
OFFICIAL_REPOS=${OFFICIAL}
PROTOCOL=matched_64px_window768_seed3407_epochs20_final_only
EOF
sha256sum "$RUN_ROOT/jobs.tsv" "$RUN_ROOT/official_sources.tsv" "$RUN_ROOT/deployment.env"
cat "$RUN_ROOT/jobs.tsv"
cat "$RUN_ROOT/official_sources.tsv"
echo "SNAPSHOT=$SNAP"
echo "RUN_ROOT=$RUN_ROOT"
'@

($remote -replace "`r", '') | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "remote deployment failed with exit code $LASTEXITCODE"
}
