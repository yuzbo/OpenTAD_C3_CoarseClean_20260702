$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$args = @(
  '-o','BatchMode=yes','-o','ConnectTimeout=10','-o','IdentitiesOnly=yes',
  '-o','PubkeyAcceptedAlgorithms=+ssh-rsa','-o','HostkeyAlgorithms=+ssh-rsa',
  '-i','C:\Users\skywalker\.ssh\id_rsa','-p','22',
  '-l','sczc063@BSCC-N16R4','ssh.cn-zhongwei-1.paracloud.com',
  "tr -d '\r' | bash -s"
)
$script = @'
set -euo pipefail
TARGET=/data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702/video-mamba-suite
EXPECTED=/data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702/video-mamba-suite/video-mamba-suite/temporal-action-segmentation/model.py
[[ "$TARGET" == /data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702/video-mamba-suite ]]
[[ -d "$TARGET/.git" ]]
git -C "$TARGET" restore --source=HEAD --worktree -- .
[[ -f "$EXPECTED" ]]
printf 'video_mamba_commit=%s\n' "$(git -C "$TARGET" rev-parse HEAD)"
printf 'video_mamba_worktree_changes=%s\n' "$(git -C "$TARGET" status --porcelain | wc -l)"
'@
$script | & $ssh @args
if ($LASTEXITCODE -ne 0) { throw "Video-Mamba checkout restore failed: $LASTEXITCODE" }
