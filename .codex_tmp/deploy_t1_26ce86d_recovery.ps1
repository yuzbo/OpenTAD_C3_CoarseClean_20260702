$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scp = 'C:\Windows\System32\OpenSSH\scp.exe'
$key = 'C:\Users\skywalker\.ssh\id_rsa'
$hostName = 'ssh.cn-zhongwei-1.paracloud.com'
$login = 'sczc063@BSCC-N16R4'
$stageDir = '/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_recovery_submit_ca40c9c'
$scpArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa', '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-o', "User=$login", '-i', $key, '-P', '22'
)
$sshArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa', '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', $key, '-p', '22', '-l', $login, $hostName
)

foreach ($file in @('recover_t1_26ce86d.sbatch', 'submit_t1_26ce86d_recovery.sh')) {
    & $scp @scpArgs ".\.codex_tmp\$file" "${hostName}:${stageDir}/$file"
    if ($LASTEXITCODE -ne 0) { throw "Failed to upload $file." }
}

& $ssh @sshArgs "bash -n '$stageDir/recover_t1_26ce86d.sbatch' && bash -n '$stageDir/submit_t1_26ce86d_recovery.sh' && bash '$stageDir/submit_t1_26ce86d_recovery.sh'"
if ($LASTEXITCODE -ne 0) { throw 'T1 recovery submission failed.' }
