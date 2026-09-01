$ErrorActionPreference = 'Stop'

$sshExe = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scpExe = 'C:\Windows\System32\OpenSSH\scp.exe'
$identity = 'C:\Users\skywalker\.ssh\id_rsa'
$remoteHost = 'ssh.cn-zhongwei-1.paracloud.com'
$remoteUser = 'sczc063@BSCC-N16R4'
$remoteBase = '/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_b3222af_20260720'
$remoteRepo = '/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_selector_draft_20260720_01'
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

$prepareScript = @"
set -e
test -d '$remoteBase'
if ! test -d '$remoteRepo'; then
  cp -a '$remoteBase' '$remoteRepo'
fi
printf 'REMOTE_REPO=%s\n' '$remoteRepo'
"@
$prepareScript | & $sshExe @sshArgs "tr -d '\r' | bash -l -s"
if ($LASTEXITCODE -ne 0) {
    throw "Remote preparation failed with exit code $LASTEXITCODE"
}

$files = @(
    @{ Local = 'opentad\models\duca\structured_selection.py'; Remote = 'opentad/models/duca/structured_selection.py' }
    @{ Local = 'opentad\models\duca\transition_only.py'; Remote = 'opentad/models/duca/transition_only.py' }
    @{ Local = 'opentad\models\duca\__init__.py'; Remote = 'opentad/models/duca/__init__.py' }
    @{ Local = 'opentad\models\selectors\duca_protected_e2e_frame_selector.py'; Remote = 'opentad/models/selectors/duca_protected_e2e_frame_selector.py' }
    @{ Local = 'opentad\models\selectors\__init__.py'; Remote = 'opentad/models/selectors/__init__.py' }
    @{ Local = 'opentad\models\detectors\actionformer.py'; Remote = 'opentad/models/detectors/actionformer.py' }
    @{ Local = 'opentad\models\dense_heads\anchor_free_head.py'; Remote = 'opentad/models/dense_heads/anchor_free_head.py' }
    @{ Local = 'opentad\datasets\duca_stateless.py'; Remote = 'opentad/datasets/duca_stateless.py' }
    @{ Local = 'opentad\datasets\__init__.py'; Remote = 'opentad/datasets/__init__.py' }
    @{ Local = 'opentad\datasets\transforms\end_to_end.py'; Remote = 'opentad/datasets/transforms/end_to_end.py' }
    @{ Local = 'tools\test.py'; Remote = 'tools/test.py' }
    @{ Local = 'tools\train.py'; Remote = 'tools/train.py' }
    @{ Local = 'tools\bata\aggregate_duca_protected_physical_official60.py'; Remote = 'tools/bata/aggregate_duca_protected_physical_official60.py' }
    @{ Local = 'tools\bata\aggregate_duca_protected_physical_p3.py'; Remote = 'tools/bata/aggregate_duca_protected_physical_p3.py' }
    @{ Local = 'tools\bata\authorize_duca_protected_physical_suite.py'; Remote = 'tools/bata/authorize_duca_protected_physical_suite.py' }
    @{ Local = 'tools\bata\duca_protected_physical_training.py'; Remote = 'tools/bata/duca_protected_physical_training.py' }
    @{ Local = 'tools\bata\duca_protected_physical_p3.py'; Remote = 'tools/bata/duca_protected_physical_p3.py' }
    @{ Local = 'tools\bata\finalize_duca_protected_physical_run.py'; Remote = 'tools/bata/finalize_duca_protected_physical_run.py' }
    @{ Local = 'tools\bata\freeze_duca_protected_physical_protocol.py'; Remote = 'tools/bata/freeze_duca_protected_physical_protocol.py' }
    @{ Local = 'tools\bata\run_duca_protected_physical_full_model_gate.py'; Remote = 'tools/bata/run_duca_protected_physical_full_model_gate.py' }
    @{ Local = 'tools\bata\run_duca_protected_physical_p3_shard.py'; Remote = 'tools/bata/run_duca_protected_physical_p3_shard.py' }
    @{ Local = 'configs\adatad\thumos\duca_protected_physical_fixed384_official60_base.py'; Remote = 'configs/adatad/thumos/duca_protected_physical_fixed384_official60_base.py' }
    @{ Local = 'configs\adatad\thumos\duca_protected_physical_exact_uniform_fixed384_official60.py'; Remote = 'configs/adatad/thumos/duca_protected_physical_exact_uniform_fixed384_official60.py' }
    @{ Local = 'configs\adatad\thumos\duca_protected_physical_transition_no_bridge_fixed384_official60.py'; Remote = 'configs/adatad/thumos/duca_protected_physical_transition_no_bridge_fixed384_official60.py' }
    @{ Local = 'configs\adatad\thumos\duca_protected_physical_e2e_fixed384_official60.py'; Remote = 'configs/adatad/thumos/duca_protected_physical_e2e_fixed384_official60.py' }
    @{ Local = 'configs\adatad\thumos\duca_protected_physical_e2e_rho001_fixed384_official60.py'; Remote = 'configs/adatad/thumos/duca_protected_physical_e2e_rho001_fixed384_official60.py' }
    @{ Local = 'configs\adatad\thumos\duca_protected_physical_p3_train_windows.py'; Remote = 'configs/adatad/thumos/duca_protected_physical_p3_train_windows.py' }
    @{ Local = 'tests\test_duca_protected_e2e_selection.py'; Remote = 'tests/test_duca_protected_e2e_selection.py' }
    @{ Local = 'tests\test_duca_protected_e2e_frame_selector.py'; Remote = 'tests/test_duca_protected_e2e_frame_selector.py' }
    @{ Local = 'tests\test_duca_protected_e2e_detector_contract.py'; Remote = 'tests/test_duca_protected_e2e_detector_contract.py' }
    @{ Local = 'tests\test_duca_protected_physical_official60_configs.py'; Remote = 'tests/test_duca_protected_physical_official60_configs.py' }
    @{ Local = 'tests\test_duca_protected_physical_protocol.py'; Remote = 'tests/test_duca_protected_physical_protocol.py' }
    @{ Local = 'tests\test_duca_protected_physical_p3.py'; Remote = 'tests/test_duca_protected_physical_p3.py' }
    @{ Local = 'tests\test_duca_protected_physical_evidence_chain.py'; Remote = 'tests/test_duca_protected_physical_evidence_chain.py' }
    @{ Local = 'tests\test_duca_protected_physical_official60_evidence.py'; Remote = 'tests/test_duca_protected_physical_official60_evidence.py' }
    @{ Local = 'scripts\complete_duca_protected_physical_gate_suite.sh'; Remote = 'scripts/complete_duca_protected_physical_gate_suite.sh' }
    @{ Local = 'scripts\duca_protected_physical_env.sh'; Remote = 'scripts/duca_protected_physical_env.sh' }
    @{ Local = 'scripts\freeze_duca_protected_physical_protocol.sh'; Remote = 'scripts/freeze_duca_protected_physical_protocol.sh' }
    @{ Local = 'scripts\run_duca_protected_physical_full_model_gate_gpu1.sh'; Remote = 'scripts/run_duca_protected_physical_full_model_gate_gpu1.sh' }
    @{ Local = 'scripts\run_duca_protected_physical_official60_variant_gpu1.sh'; Remote = 'scripts/run_duca_protected_physical_official60_variant_gpu1.sh' }
    @{ Local = 'scripts\run_duca_protected_physical_p3_shard_gpu1.sh'; Remote = 'scripts/run_duca_protected_physical_p3_shard_gpu1.sh' }
    @{ Local = 'scripts\submit_duca_protected_physical_gate_suite.sh'; Remote = 'scripts/submit_duca_protected_physical_gate_suite.sh' }
    @{ Local = 'scripts\submit_duca_protected_physical_official60_suite.sh'; Remote = 'scripts/submit_duca_protected_physical_official60_suite.sh' }
)
foreach ($pair in $files) {
    $source = Join-Path $localRepo $pair.Local
    $uploaded = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & $scpExe @scpArgs $source "${remoteHost}:${remoteRepo}/$($pair.Remote)"
        if ($LASTEXITCODE -eq 0) {
            $uploaded = $true
            break
        }
        Start-Sleep -Seconds (2 * $attempt)
    }
    if (-not $uploaded) {
        throw "Upload failed for $($pair.Local) after 3 attempts"
    }
}

$testScript = @"
set -e
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1
cd '$remoteRepo'
python -m py_compile \
  opentad/models/duca/structured_selection.py \
  opentad/models/duca/transition_only.py \
  opentad/models/selectors/duca_protected_e2e_frame_selector.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/datasets/duca_stateless.py \
  opentad/datasets/transforms/end_to_end.py \
  tools/test.py \
  tools/train.py \
  tools/bata/aggregate_duca_protected_physical_official60.py \
  tools/bata/aggregate_duca_protected_physical_p3.py \
  tools/bata/authorize_duca_protected_physical_suite.py \
  tools/bata/duca_protected_physical_training.py \
  tools/bata/duca_protected_physical_p3.py \
  tools/bata/finalize_duca_protected_physical_run.py \
  tools/bata/freeze_duca_protected_physical_protocol.py \
  tools/bata/run_duca_protected_physical_full_model_gate.py \
  tools/bata/run_duca_protected_physical_p3_shard.py \
  tests/test_duca_protected_e2e_selection.py \
  tests/test_duca_protected_e2e_frame_selector.py \
  tests/test_duca_protected_e2e_detector_contract.py \
  tests/test_duca_protected_physical_official60_configs.py \
  tests/test_duca_protected_physical_protocol.py \
  tests/test_duca_protected_physical_p3.py \
  tests/test_duca_protected_physical_evidence_chain.py \
  tests/test_duca_protected_physical_official60_evidence.py
bash -n \
  scripts/complete_duca_protected_physical_gate_suite.sh \
  scripts/duca_protected_physical_env.sh \
  scripts/freeze_duca_protected_physical_protocol.sh \
  scripts/run_duca_protected_physical_full_model_gate_gpu1.sh \
  scripts/run_duca_protected_physical_official60_variant_gpu1.sh \
  scripts/run_duca_protected_physical_p3_shard_gpu1.sh \
  scripts/submit_duca_protected_physical_gate_suite.sh \
  scripts/submit_duca_protected_physical_official60_suite.sh
python -m pytest \
  tests/test_duca_protected_e2e_selection.py \
  tests/test_duca_protected_e2e_frame_selector.py \
  tests/test_duca_protected_e2e_detector_contract.py \
  tests/test_duca_protected_physical_official60_configs.py \
  tests/test_duca_protected_physical_protocol.py \
  tests/test_duca_protected_physical_p3.py \
  tests/test_duca_protected_physical_evidence_chain.py \
  tests/test_duca_protected_physical_official60_evidence.py \
  -q
"@
$testScript | & $sshExe @sshArgs "tr -d '\r' | bash -l -s"
if ($LASTEXITCODE -ne 0) {
    throw "Remote focused test failed with exit code $LASTEXITCODE"
}
