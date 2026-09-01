$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$args = @(
  '-o','BatchMode=yes','-o','ConnectTimeout=10','-o','IdentitiesOnly=yes',
  '-o','PubkeyAcceptedAlgorithms=+ssh-rsa','-o','HostkeyAlgorithms=+ssh-rsa',
  '-i','C:\Users\skywalker\.ssh\id_rsa','-p','22',
  '-l','sczc063@BSCC-N16R4','ssh.cn-zhongwei-1.paracloud.com'
)
$remote = @'
set -u
base=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037
echo '=== R4 SBATCH ==='
for f in $(find "${base}_r4" -maxdepth 3 -type f -name '*.sbatch' 2>/dev/null | sort); do
  echo "--- $f"
  sed -n '1,260p' "$f"
done
echo '=== R3 FILES ==='
find "${base}_r0_r3" -maxdepth 4 -type f -printf '%s|%p\n' 2>/dev/null | sort -t'|' -nr | head -120 || true
echo '=== EXISTING FRONTEND ALIGNMENT TERMINAL ==='
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe -type f \
  \( -name 'frontend_decision.json' -o -name 'alignment.json' -o -name 'terminal_suite.json' -o -name 'aggregate.json' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS|%s|%p\n' 2>/dev/null | sort | tail -120 || true
'@
& $ssh @args $remote
if ($LASTEXITCODE -ne 0) { throw "ssh failed with exit code $LASTEXITCODE" }
