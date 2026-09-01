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
set -u
date '+%F %T %z'
echo '===SQUEUE==='
squeue -j 1180490,1180491,1180492,1180493,1180494,1180495,1180496 -o '%i|%j|%T|%M|%R'
echo '===SACCT==='
sacct -j 1180490,1180491,1180492,1180493,1180494,1180495,1180496 --format=JobID,JobName%28,State,Elapsed,ExitCode -P

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343
echo '===TRAIN_TAILS==='
find "$ROOT" -type f -name train.out -print | sort | while read -r file; do
    echo "---$file"
    tail -n 4 "$file"
done

echo '===ERROR_SCAN==='
grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite loss|ValueError|(^|[^A-Z])FAIL([^A-Z]|$)' "$ROOT" --include='*.out' --include='*.err' 2>/dev/null | tail -n 80 || true

echo '===RESULT_FILES==='
find "$ROOT" -type f \( -name '*terminal*.json' -o -name '*summary*.json' -o -name '*decision*.json' -o -name '*receipt*.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -n 40
'@

($remote -replace "`r", '') | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH monitor failed with exit code $LASTEXITCODE"
}
