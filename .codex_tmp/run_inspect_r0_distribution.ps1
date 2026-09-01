$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$argsList = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa', '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', 'C:\Users\skywalker\.ssh\id_rsa', '-p', '22',
    '-l', 'sczc063@BSCC-N16R4', 'ssh.cn-zhongwei-1.paracloud.com', 'bash -s'
)
Get-Content -Raw '.\.codex_tmp\inspect_r0_distribution.sh' | & $ssh @argsList
if ($LASTEXITCODE -ne 0) { throw 'R0 distribution inspection failed.' }
