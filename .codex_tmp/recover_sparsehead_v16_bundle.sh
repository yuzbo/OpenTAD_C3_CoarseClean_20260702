#!/usr/bin/env bash
set -euo pipefail

runtime=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v16
bundle_dir=/data/run01/sczc063/yuzibo/tmp/codex_sparsehead_v16_recovery
bundle_path="$bundle_dir/sparsehead_v16_54e7f9abeaabf710a505f0a0f595a4eb3bb47f98.bundle"
expected_branch=codex/sparsehead-evidence-recovery-20260729-v16
expected_commit=54e7f9abeaabf710a505f0a0f595a4eb3bb47f98
expected_tree=f8490f9c25c2e0e6958c406e19c83cc3d5a40535

test -d "$runtime/.git"
actual_branch="$(git -C "$runtime" branch --show-current)"
actual_commit="$(git -C "$runtime" rev-parse HEAD)"
actual_tree="$(git -C "$runtime" rev-parse HEAD^{tree})"
test "$actual_branch" = "$expected_branch"
test "$actual_commit" = "$expected_commit"
test "$actual_tree" = "$expected_tree"
test -z "$(git -C "$runtime" status --porcelain=v1 --untracked-files=all)"

mkdir -p "$bundle_dir"
rm -f "$bundle_path"
git -C "$runtime" bundle create "$bundle_path" HEAD
git -C "$runtime" bundle verify "$bundle_path" >&2
sha256sum "$bundle_path"
wc -c "$bundle_path"
