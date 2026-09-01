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
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
REPOS=$BASE/projects/external_official_action_segmentation_repos_20260702
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
source $BASE/conda_envs/opentad/bin/activate
printf 'python=%s\n' "$(command -v python)"
printf 'torch='
python -c 'import torch; print(torch.__version__)'
for path in \
  "$REPOS/MS-TCN2/model.py" \
  "$REPOS/ASFormer/model.py" \
  "$REPOS/CVPR2024-FACT/models/blocks.py" \
  "$REPOS/video-mamba-suite/video-mamba-suite/temporal-action-segmentation/model.py"; do
  if [[ -f "$path" ]]; then
    printf 'source_ok=%s\n' "$path"
  else
    printf 'source_missing=%s\n' "$path"
  fi
done
python - <<'PY'
try:
    import mamba_ssm
except Exception as exc:
    print('mamba_ssm_available=false', type(exc).__name__, str(exc))
else:
    print('mamba_ssm_available=true', getattr(mamba_ssm, '__version__', 'unknown'))
PY
'@

$remote | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH backend check failed with exit code $LASTEXITCODE"
}
