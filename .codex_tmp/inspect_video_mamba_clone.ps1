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
set -u
TARGET=/data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702/video-mamba-suite
printf 'target_exists=%s\n' "$(test -e "$TARGET" && echo true || echo false)"
printf 'git_exists=%s\n' "$(test -d "$TARGET/.git" && echo true || echo false)"
if [[ -d "$TARGET/.git" ]]; then
  ps -eo pid,etimes,cmd | grep -E '[g]it (clone|checkout|restore)|[g]it-remote-https' || true
  ls -l --time-style=long-iso "$TARGET/.git/index.lock" 2>/dev/null || true
  git -C "$TARGET" rev-parse HEAD 2>&1 || true
  git -C "$TARGET" status --short --branch --untracked-files=no 2>&1 | head -n 12 || true
  git -C "$TARGET" remote -v 2>&1 || true
fi
find "$TARGET" -maxdepth 4 -type f -name model.py -print 2>/dev/null | head -n 40
find "$TARGET" -maxdepth 3 -type d -print 2>/dev/null | head -n 80
'@
$script | & $ssh @args
if ($LASTEXITCODE -ne 0) { throw "inspection failed: $LASTEXITCODE" }
