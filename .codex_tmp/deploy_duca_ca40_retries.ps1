$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scp = 'C:\Windows\System32\OpenSSH\scp.exe'
$key = 'C:\Users\skywalker\.ssh\id_rsa'
$hostName = 'ssh.cn-zhongwei-1.paracloud.com'
$login = 'sczc063@BSCC-N16R4'
$stageDir = '/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_recovery_submit_ca40c9c'
$common = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
    '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-o', "User=$login",
    '-i', $key,
    '-P', '22'
)

$sshCommon = @(
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

& $ssh @sshCommon "mkdir -p '$stageDir'"
if ($LASTEXITCODE -ne 0) { throw 'Failed to create remote staging directory.' }

& $scp @common '.\.codex_tmp\recover_duca_r2_r3_ca40c9c.sbatch' '.\.codex_tmp\submit_duca_ca40_retries.sh' "${hostName}:${stageDir}/"
if ($LASTEXITCODE -ne 0) { throw 'Failed to upload recovery scripts.' }

& $ssh @sshCommon "chmod 700 '$stageDir/submit_duca_ca40_retries.sh' && bash '$stageDir/submit_duca_ca40_retries.sh'"
if ($LASTEXITCODE -ne 0) { throw 'Failed to submit recovery jobs.' }
