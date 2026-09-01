$ErrorActionPreference = 'Stop'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$sshArgs = @(
  '-o','BatchMode=yes','-o','ConnectTimeout=10','-o','IdentitiesOnly=yes',
  '-o','PubkeyAcceptedAlgorithms=+ssh-rsa','-o','HostkeyAlgorithms=+ssh-rsa',
  '-i','C:\Users\skywalker\.ssh\id_rsa','-p','22',
  '-l','sczc063@BSCC-N16R4','ssh.cn-zhongwei-1.paracloud.com',
  "tr -d '\r' | bash -s"
)
$remote = @'
set -u
R=/data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702
echo '===MS_TCN2_CLASSES==='
grep -n -E '^class |def forward' "$R/MS-TCN2/model.py" | head -n 80
sed -n '1,220p' "$R/MS-TCN2/model.py"
echo '===FACT_SURFACES==='
grep -n -E '^class |frame_(feature|clogit)|def forward|def _forward_one_video' "$R/CVPR2024-FACT/models/blocks.py" | head -n 160
sed -n '40,115p' "$R/CVPR2024-FACT/models/blocks.py"
echo '===VIDEO_MAMBA_CLASSES==='
V="$R/video-mamba-suite/video-mamba-suite/temporal-action-segmentation/model.py"
grep -n -E '^class |def forward' "$V" | head -n 140
sed -n '1,280p' "$V"
sed -n '268,375p' "$V"
'@
($remote -replace "`r", '') | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) { throw "remote source inspection failed with exit code $LASTEXITCODE" }
