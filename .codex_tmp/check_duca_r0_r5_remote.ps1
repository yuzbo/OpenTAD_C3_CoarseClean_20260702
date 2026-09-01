$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$args = @(
  '-o', 'BatchMode=yes',
  '-o', 'ConnectTimeout=10',
  '-o', 'IdentitiesOnly=yes',
  '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
  '-o', 'HostkeyAlgorithms=+ssh-rsa',
  '-i', 'C:\Users\skywalker\.ssh\id_rsa',
  '-p', '22',
  '-l', 'sczc063@BSCC-N16R4',
  'ssh.cn-zhongwei-1.paracloud.com'
)
$remote = @'
set -u
echo '=== DATE ==='
date -Is
echo '=== SQUEUE DUCA ==='
squeue -u "$USER" -h -o '%i|%j|%T|%M|%R|%E' | grep -Ei 'duca|dbr|ind|r5|r0|temporal' || true
echo '=== SACCT ALL KNOWN ==='
sacct -X -j 1179795,1179796,1179797,1179798,1179799,1179825,1179826,1179827,1179861,1179862,1179863,1179864,1179865,1179956,1180111,1180112,1180113,1180114 --format=JobIDRaw,JobName%40,State,Elapsed,ExitCode -n -P || true
echo '=== CURRENT INDEPENDENT ROOT ==='
root=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_independent_8d85929_formal_20260722_1820
find "$root" -maxdepth 7 -type f \( -name 'completion.json' -o -name 'train.out' -o -name '*terminal*json' -o -name '*summary.json' -o -name 'jobs.tsv' -o -name 'submission_manifest.json' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS|%s|%p\n' 2>/dev/null | sort | tail -120 || true
echo '=== PROGRESS TAILS ==='
for f in $(find "$root" -maxdepth 7 -type f -name 'train.out' 2>/dev/null | sort); do echo "--- $f"; grep -E 'Epoch|epoch|Average-mAP|mAP at tIoU|Training Over|P0|winner|selected_count|max_hole' "$f" | tail -12 || tail -8 "$f"; done
echo '=== OLD R5 ROOTS ==='
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe -maxdepth 2 -type f \( -name 'r5_matrix.json' -o -name 'r5_submission_manifest.json' -o -name 'jobs.tsv' -o -name 'deployment_receipt.json' \) -path '*duca_boundary_e49ef69*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS|%s|%p\n' 2>/dev/null | sort | tail -80 || true
echo '=== ERROR SCAN CURRENT ==='
grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite loss|ValueError|\[FAIL\]' "$root" --include='*.out' --include='*.err' --include='train.out' 2>/dev/null | tail -80 || true
'@
& $ssh @args $remote
if ($LASTEXITCODE -ne 0) { throw "ssh failed with exit code $LASTEXITCODE" }
