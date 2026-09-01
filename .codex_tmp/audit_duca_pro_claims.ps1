$ErrorActionPreference = 'Stop'

$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$sshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
    '-o', 'HostkeyAlgorithms=+ssh-rsa',
    '-i', 'C:\Users\skywalker\.ssh\id_rsa',
    '-p', '22',
    '-l', 'sczc063@BSCC-N16R4',
    'ssh.cn-zhongwei-1.paracloud.com',
    "tr -d '\r' | bash -s"
)

$remote = @'
set -u
R=/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_9f97f2c_20260722
if [ ! -d "$R" ]; then
  R=/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_9f97f2c_20260723
fi
if [ ! -d "$R" ]; then
  R=/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_9f97f2c_formal_20260722_2343
fi
echo "R=$R"
if [ -d "$R/.git" ] || git -C "$R" rev-parse --git-dir >/dev/null 2>&1; then
  echo '===IDENTITY===' 
  git -C "$R" rev-parse HEAD
  git -C "$R" status --short --branch
  echo '===DIFF_A004_TO_9F97===' 
  git -C "$R" diff --name-status a00498e15d69294f78d0abeadfb47bc456db0b0e..9f97f2c7f081b10fbf1f63d0602a621c6b43a780 || true
else
  echo 'NO_GIT_SNAPSHOT'
fi

echo '===MODEL_CODE_FACTS===' 
for f in \
  tools/bata/profile_duca_full_stack_cost.py \
  tools/bata/aggregate_duca_budget_curve.py \
  opentad/models/duca/transition_only.py \
  opentad/models/duca/structured_selection.py \
  opentad/models/duca/hard_soft_alignment.py \
  tools/bata/duca_r5_paper_matrix.py \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/models/detectors/actionformer.py; do
  echo "---$f"
  if [ -f "$R/$f" ]; then
    grep -n -E 'R5_MAX_UNSELECTED_HOLES|384\|256|model_contract_sha256|selector_tree_sha256|required_mask|mandatory|complet|local_slope|selected_axis_index|detector_axis|physical_grid|true_time|MAX_UNSELECTED_HOLES|BUDGETS' "$R/$f" | head -n 140 || true
  else
    echo MISSING
  fi
done

echo '===TTDI_EXISTS===' 
find "$R/opentad" -type f \( -iname '*true*time*detector*' -o -iname '*physical*time*pyramid*' \) -print 2>/dev/null || true
echo '===TARGETED_SNIPPETS===' 
sed -n '738,865p' "$R/opentad/models/duca/transition_only.py"
sed -n '1058,1095p' "$R/opentad/models/duca/structured_selection.py"
grep -R -n -E '_add_protected_structured_transport_gradient_path|expected_position|local.*slope|left.*right.*gather' \
  "$R/opentad/models" 2>/dev/null | head -n 100 || true

COARSE=/data/run01/sczc063/yuzibo/projects/opentad_duca_coarse_4f81299_20260723
echo '===DIFF_9F97_TO_4F81299===' 
if git -C "$COARSE" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$COARSE" rev-parse HEAD
  git -C "$COARSE" diff --name-status 9f97f2c7f081b10fbf1f63d0602a621c6b43a780..HEAD
fi

echo '===TEST_FACTS===' 
find "$R/tests" -maxdepth 1 -type f -name 'test_duca*' -print0 | xargs -0 grep -n -E 'complet|whole.group|true.time.*adapter|timestamp.*rank|all.budget|source.equivalence|hard.swap.*spearman' 2>/dev/null | head -n 180 || true
'@

($remote -replace "`r", '') | & $ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "SSH audit failed with exit code $LASTEXITCODE"
}
