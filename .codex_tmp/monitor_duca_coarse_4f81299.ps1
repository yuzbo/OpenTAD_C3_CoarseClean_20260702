$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$sshArgs = @(
  '-o','BatchMode=yes','-o','ConnectTimeout=10','-o','IdentitiesOnly=yes',
  '-o','PubkeyAcceptedAlgorithms=+ssh-rsa','-o','HostkeyAlgorithms=+ssh-rsa',
  '-i','C:\Users\skywalker\.ssh\id_rsa','-p','22',
  '-l','sczc063@BSCC-N16R4','ssh.cn-zhongwei-1.paracloud.com',
  "tr -d '\r' | bash -s"
)
$remote = @'
set -u
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_coarse_backends_4f81299_20260723_0015
date '+%F %T %z'
echo '===SQUEUE==='
squeue -j 1180502,1180503,1180504,1180505 -o '%i|%j|%T|%M|%R'
echo '===SACCT==='
sacct -j 1180502,1180503,1180504,1180505 --format=JobID,JobName%24,State,Elapsed,ExitCode,NodeList%12 -P
echo '===LOG_TAILS==='
find "$ROOT/logs" -type f -print 2>/dev/null | sort | while read -r file; do
  echo "---$file"
  tail -n 12 "$file"
done
echo '===ERROR_SCAN==='
grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite|ValueError|RuntimeError|(^|[^A-Z])FAIL([^A-Z]|$)' \
  "$ROOT/logs" "$ROOT/outputs" 2>/dev/null | tail -n 80 || true
echo '===SUMMARIES==='
find "$ROOT/outputs" -type f -name summary.json -print 2>/dev/null | sort
'@
($remote -replace "`r", '') | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) { throw "remote monitor failed with exit code $LASTEXITCODE" }
