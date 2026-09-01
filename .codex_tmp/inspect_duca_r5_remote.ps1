$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$args = @(
  '-o','BatchMode=yes','-o','ConnectTimeout=10','-o','IdentitiesOnly=yes',
  '-o','PubkeyAcceptedAlgorithms=+ssh-rsa','-o','HostkeyAlgorithms=+ssh-rsa',
  '-i','C:\Users\skywalker\.ssh\id_rsa','-p','22',
  '-l','sczc063@BSCC-N16R4','ssh.cn-zhongwei-1.paracloud.com'
)
$remote = @'
set -eu
r=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r5
echo '=== ROOT ==='
find "$r" -maxdepth 3 -type f -printf '%p\n' | sort | sed -n '1,180p'
echo '=== MANIFEST CANDIDATES ==='
for f in "$r"/*.json "$r"/*/*.json; do [ -f "$f" ] || continue; echo "--- $f"; python - "$f" <<'PY'
import json,sys
p=sys.argv[1]
try:
 d=json.load(open(p))
except Exception as e:
 print('not-json',e); raise SystemExit
print(json.dumps(d,indent=2,sort_keys=True)[:16000])
PY
done
echo '=== JOB TSV ==='
cat "$r/jobs.tsv" || true
echo '=== FIRST TRAIN SBATCH ==='
f=$(find "$r/jobs" -maxdepth 1 -type f -name '*train*sbatch' | sort | head -1)
[ -n "$f" ] && { echo "$f"; sed -n '1,220p' "$f"; } || true
'@
& $ssh @args $remote
if ($LASTEXITCODE -ne 0) { throw "ssh failed with exit code $LASTEXITCODE" }
