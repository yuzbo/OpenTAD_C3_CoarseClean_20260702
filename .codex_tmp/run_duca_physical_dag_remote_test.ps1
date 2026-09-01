$ErrorActionPreference = 'Stop'

$sshExe = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scpExe = 'C:\Windows\System32\OpenSSH\scp.exe'
$identity = 'C:\Users\skywalker\.ssh\id_rsa'
$remoteHost = 'ssh.cn-zhongwei-1.paracloud.com'
$remoteUser = 'sczc063@BSCC-N16R4'
$remoteBase = '/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_b3222af_20260720'
$remoteRepo = '/data/run01/sczc063/yuzibo/projects/opentad_duca_physical_dag_draft_20260720_02'
$localRepo = 'E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.codex_tmp\OpenTAD_DUCA_ProtectedE2E_Final_20260720'

$sshArgs = @(
    '-o', 'BatchMode=yes'
    '-o', 'ConnectTimeout=10'
    '-o', 'IdentitiesOnly=yes'
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa'
    '-o', 'HostkeyAlgorithms=+ssh-rsa'
    '-i', $identity
    '-p', '22'
    '-l', $remoteUser
    $remoteHost
)

$prepareScript = @"
set -e
test -d '$remoteBase'
if ! test -d '$remoteRepo'; then
  cp -a '$remoteBase' '$remoteRepo'
fi
test -f '$remoteRepo/opentad/models/duca/structured_selection.py'
printf 'REMOTE_REPO=%s\n' '$remoteRepo'
printf 'BASE_SOURCE_PRESENT=true\n'
"@

$prepareScript | & $sshExe @sshArgs "tr -d '\r' | bash -l -s"
if ($LASTEXITCODE -ne 0) {
    throw "Remote preparation failed with exit code $LASTEXITCODE"
}

$scpArgs = @(
    '-o', 'BatchMode=yes'
    '-o', 'ConnectTimeout=10'
    '-o', 'IdentitiesOnly=yes'
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa'
    '-o', 'HostkeyAlgorithms=+ssh-rsa'
    '-o', "User=$remoteUser"
    '-i', $identity
    '-P', '22'
)

$sourceSelection = Join-Path $localRepo 'opentad\models\duca\structured_selection.py'
$sourceTest = Join-Path $localRepo 'tests\test_duca_protected_e2e_selection.py'
& $scpExe @scpArgs $sourceSelection "${remoteHost}:${remoteRepo}/opentad/models/duca/structured_selection.py"
if ($LASTEXITCODE -ne 0) {
    throw "Selection source upload failed with exit code $LASTEXITCODE"
}
& $scpExe @scpArgs $sourceTest "${remoteHost}:${remoteRepo}/tests/test_duca_protected_e2e_selection.py"
if ($LASTEXITCODE -ne 0) {
    throw "Selection test upload failed with exit code $LASTEXITCODE"
}

$testScript = @"
set -e
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
cd '$remoteRepo'
python -m py_compile opentad/models/duca/structured_selection.py tests/test_duca_protected_e2e_selection.py
python -m pytest tests/test_duca_protected_e2e_selection.py -q
"@

$testScript | & $sshExe @sshArgs "tr -d '\r' | bash -l -s"
if ($LASTEXITCODE -ne 0) {
    throw "Remote focused test failed with exit code $LASTEXITCODE"
}
