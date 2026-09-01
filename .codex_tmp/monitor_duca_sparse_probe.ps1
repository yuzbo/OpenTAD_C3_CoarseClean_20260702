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
echo '===QUEUE===' 
squeue -j 1180502,1180503,1180504,1180505 -o '%i|%j|%T|%M|%R'
sacct -j 1180502,1180503,1180504,1180505 --format=JobID,JobName%30,State,Elapsed,ExitCode -P

SNAP=/data/run01/sczc063/yuzibo/projects/opentad_duca_coarse_4f81299_20260723
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_coarse_backends_4f81299_20260723_0015
echo '===SPARSE_CODE_SCAN===' 
grep -R -n -E 'probe_stride|probe_interval|scan_stride|sparse_probe|hidden.linear|linear.*hidden|anchor_mask|anchor_distance' \
  "$SNAP/opentad" "$SNAP/configs" "$SNAP/tools" "$SNAP/scripts" "$SNAP/tests" 2>/dev/null | head -n 120 || true
echo '===JOB_COMMANDS===' 
find "$ROOT" -maxdepth 3 -type f \( -name '*.sbatch' -o -name 'manifest*.json' -o -name 'command*.txt' \) -print -exec sed -n '1,160p' {} \; 2>/dev/null | grep -E '(^---|backend|stride|interval|sparse|interpol|python|train_lowres)' | head -n 240 || true
echo '===PROGRESS===' 
find "$ROOT" -type f -name '*.out' -print | sort | while read -r file; do
  echo "---$file"
  grep -E 'Epoch|epoch|AUPRC|F1|ECE|summary|completed' "$file" | tail -n 5 || true
done
echo '===SUMMARIES===' 
find "$ROOT" -type f \( -name '*summary*.json' -o -name '*metrics*.json' -o -name '*map*.json' -o -name '*selection*.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort
echo '===ERROR_SCAN===' 
grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite|ValueError|(^|[^A-Z])FAIL([^A-Z]|$)' "$ROOT" --include='*.out' --include='*.err' 2>/dev/null | tail -n 80 || true
'@

($remote -replace "`r", '') | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH monitor failed with exit code $LASTEXITCODE"
}
