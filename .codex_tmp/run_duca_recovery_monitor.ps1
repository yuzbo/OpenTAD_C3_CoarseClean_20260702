$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scp = 'C:\Windows\System32\OpenSSH\scp.exe'
$key = 'C:\Users\skywalker\.ssh\id_rsa'
$hostName = 'ssh.cn-zhongwei-1.paracloud.com'
$login = 'sczc063@BSCC-N16R4'
$stageDir = '/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_recovery_submit_ca40c9c'
$scpArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
    '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-o', "User=$login",
    '-i', $key,
    '-P', '22'
)
$sshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
    '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', $key,
    '-p', '22',
    '-l', $login,
    $hostName
)

& $scp @scpArgs '.\.codex_tmp\monitor_duca_recovery.sh' "${hostName}:${stageDir}/"
if ($LASTEXITCODE -ne 0) { throw 'Failed to upload monitor script.' }

& $ssh @sshArgs "bash '$stageDir/monitor_duca_recovery.sh'"
if ($LASTEXITCODE -ne 0) { throw 'Remote monitor failed.' }
