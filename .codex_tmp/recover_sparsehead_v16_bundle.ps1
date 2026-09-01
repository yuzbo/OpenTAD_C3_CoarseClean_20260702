$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scp = 'C:\Windows\System32\OpenSSH\scp.exe'
$key = 'C:\Users\skywalker\.ssh\id_rsa'
$hostName = 'ssh.cn-zhongwei-1.paracloud.com'
$login = 'sczc063@BSCC-N16R4'
$remoteScript = '/data/run01/sczc063/yuzibo/tmp/codex_sparsehead_v16_recovery/recover_sparsehead_v16_bundle.sh'
$remoteBundle = '/data/run01/sczc063/yuzibo/tmp/codex_sparsehead_v16_recovery/sparsehead_v16_54e7f9abeaabf710a505f0a0f595a4eb3bb47f98.bundle'
$localRoot = Join-Path $PSScriptRoot 'recovered'
$localBundle = Join-Path $localRoot 'sparsehead_v16_54e7f9abeaabf710a505f0a0f595a4eb3bb47f98.bundle'

$common = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=15',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
    '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', $key
)
$sshArgs = $common + @('-p', '22', '-l', $login, $hostName)
$scpArgs = $common + @('-o', "User=$login", '-P', '22')

New-Item -ItemType Directory -Force -Path $localRoot | Out-Null

& $scp @scpArgs (Join-Path $PSScriptRoot 'recover_sparsehead_v16_bundle.sh') "${hostName}:${remoteScript}"
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to stage the v16 recovery script.'
}

& $ssh @sshArgs "tr -d '\r' < '$remoteScript' | bash"
if ($LASTEXITCODE -ne 0) {
    throw 'Remote v16 identity verification or bundle creation failed.'
}

& $scp @scpArgs "${hostName}:${remoteBundle}" $localBundle
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to download the exact v16 Git bundle.'
}

& git bundle verify $localBundle
if ($LASTEXITCODE -ne 0) {
    throw 'Downloaded v16 Git bundle failed local verification.'
}

& git bundle list-heads $localBundle
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to list heads in the downloaded v16 Git bundle.'
}
