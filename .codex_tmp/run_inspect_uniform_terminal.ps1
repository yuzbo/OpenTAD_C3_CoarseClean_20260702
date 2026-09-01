$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$key = 'C:\Users\skywalker\.ssh\id_rsa'
$hostName = 'ssh.cn-zhongwei-1.paracloud.com'
$login = 'sczc063@BSCC-N16R4'
$sshArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa', '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', $key, '-p', '22', '-l', $login, $hostName, 'bash -s'
)

Get-Content -Raw '.\.codex_tmp\inspect_uniform_terminal.sh' | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) { throw 'Uniform terminal inspection failed.' }
