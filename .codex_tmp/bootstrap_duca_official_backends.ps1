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
ROOT=/data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702
mkdir -p "$ROOT"

clone_if_missing() {
  local name="$1"
  local url="$2"
  local required="$3"
  local target="$ROOT/$name"
  if [[ -f "$target/$required" ]]; then
    printf 'reuse=%s\n' "$target"
  elif [[ -e "$target" ]]; then
    printf 'incomplete_existing_target=%s\n' "$target" >&2
    exit 31
  else
    git clone --depth 1 --filter=blob:none "$url" "$target"
  fi
  printf '%s_commit=%s\n' "$name" "$(git -C "$target" rev-parse HEAD)"
}

clone_if_missing MS-TCN2 \
  https://ghfast.top/https://github.com/sj-li/MS-TCN2.git \
  model.py
clone_if_missing CVPR2024-FACT \
  https://ghfast.top/https://github.com/ZijiaLewisLu/CVPR2024-FACT.git \
  models/blocks.py
clone_if_missing video-mamba-suite \
  https://ghfast.top/https://github.com/OpenGVLab/video-mamba-suite.git \
  video-mamba-suite/temporal-action-segmentation/model.py

printf 'asformer_commit=%s\n' "$(git -C "$ROOT/ASFormer" rev-parse HEAD)"
'@

$remote | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "Official backend bootstrap failed with exit code $LASTEXITCODE"
}
