#!/usr/bin/env bash
set -euo pipefail

# Push only the recent experiment branches that were verified as local-only.
# Historical worktree/snapshot branches are intentionally excluded.
REMOTE="${REMOTE:-origin}"
EXPECTED_URL="https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git"
BRANCHES=(
  "codex/duca-evidence-recovery-fullmatrix-20260901"
  "codex/zoomtoken-continuous-roi-s2-v3-fresh-3x3-v001"
)

remote_url="$(git remote get-url "${REMOTE}")"
if [[ "${remote_url}" != "${EXPECTED_URL}" ]]; then
  echo "[ERROR] Unexpected ${REMOTE} URL: ${remote_url}" >&2
  echo "        Expected: ${EXPECTED_URL}" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[ERROR] Working tree is dirty. Commit intended experiment changes first." >&2
  git status --short >&2
  exit 3
fi

git fetch --prune "${REMOTE}"

for branch in "${BRANCHES[@]}"; do
  if ! git show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "[ERROR] Missing local branch: ${branch}" >&2
    exit 4
  fi

  if ! git show-ref --verify --quiet "refs/remotes/${REMOTE}/${branch}"; then
    echo "[PUSH] Creating ${REMOTE}/${branch}"
    git push --set-upstream "${REMOTE}" "${branch}:${branch}"
    echo "[URL] https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/${branch}"
    continue
  fi

  read -r behind ahead < <(git rev-list --left-right --count "${REMOTE}/${branch}...${branch}")
  if (( behind > 0 && ahead > 0 )); then
    echo "[ERROR] Diverged branch: ${branch} (behind=${behind}, ahead=${ahead})" >&2
    exit 5
  elif (( ahead > 0 )); then
    echo "[PUSH] Updating ${REMOTE}/${branch} (ahead=${ahead})"
    git push "${REMOTE}" "${branch}:${branch}"
  else
    echo "[OK] ${branch} already synchronized"
  fi
  echo "[URL] https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/${branch}"
done

echo "[DONE] Verified experiment branches are synchronized with ${REMOTE}."
