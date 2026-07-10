#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PHYSTIME_DATA][FAIL] $*" >&2
  exit 1
}

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
DATA_ROOT="${PHYSTIME_THUMOS_ROOT:-${BASE}/datasets/phystime_thumos_i3d}"
FEATURE_DIR="${PHYSTIME_FEATURE_PATH:-${DATA_ROOT}/features/i3d_actionformer_stride4_thumos}"
ANNOTATION_SOURCE="${PHYSTIME_ANNOTATION_SOURCE:-${BASE}/thumos14/annotations}"
DOWNLOAD_DIR="${DATA_ROOT}/downloads"
EXTRACT_DIR="${DATA_ROOT}/extract_i3d"
ARCHIVE="${DOWNLOAD_DIR}/actionformer_i3d_thumos.zip"
GDOWN_ROOT="${BASE}/tools/gdown_lib"
GDRIVE_URL="${PHYSTIME_I3D_URL:-https://drive.google.com/file/d/1iemRUtCVshYD3o9WahUrTTOW08GyCjfi/view?usp=sharing}"
READY_JSON="${DATA_ROOT}/data_ready.json"
MIN_FEATURE_FILES="${PHYSTIME_MIN_FEATURE_FILES:-300}"

mkdir -p "${DOWNLOAD_DIR}" "${FEATURE_DIR}" "${DATA_ROOT}/annotations"
[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -f "${ANNOTATION_SOURCE}/thumos_14_anno.json" ]] || fail "THUMOS annotation source missing"
[[ -f "${ANNOTATION_SOURCE}/category_idx.txt" ]] || fail "THUMOS class map source missing"

feature_count="$(find "${FEATURE_DIR}" -maxdepth 1 -type f -name '*.npy' | wc -l)"
if (( feature_count < MIN_FEATURE_FILES )); then
  if ! PYTHONPATH="${GDOWN_ROOT}" "${PYTHON}" -c "import gdown" >/dev/null 2>&1; then
    mkdir -p "${GDOWN_ROOT}"
    "${PYTHON}" -m pip install --target "${GDOWN_ROOT}" --no-cache-dir gdown
  fi
  # gdown 6.x accepts Drive URLs directly and uses --continue for both fresh
  # downloads and resumable partial archives.
  PYTHONPATH="${GDOWN_ROOT}" "${PYTHON}" -m gdown --continue "${GDRIVE_URL}" -O "${ARCHIVE}"
  rm -rf "${EXTRACT_DIR}"
  mkdir -p "${EXTRACT_DIR}"
  "${PYTHON}" - "${ARCHIVE}" "${EXTRACT_DIR}" <<'PY'
import shutil
import sys

shutil.unpack_archive(sys.argv[1], sys.argv[2])
PY
  while IFS= read -r -d '' feature; do
    cp -f "${feature}" "${FEATURE_DIR}/$(basename "${feature}")"
  done < <(find "${EXTRACT_DIR}" -type f -name '*.npy' -print0)
fi

feature_count="$(find "${FEATURE_DIR}" -maxdepth 1 -type f -name '*.npy' | wc -l)"
(( feature_count >= MIN_FEATURE_FILES )) || fail "only ${feature_count} I3D feature files found"
cp -f "${ANNOTATION_SOURCE}/thumos_14_anno.json" "${DATA_ROOT}/annotations/thumos_14_anno.json"
cp -f "${ANNOTATION_SOURCE}/category_idx.txt" "${DATA_ROOT}/annotations/category_idx.txt"
cat "${ANNOTATION_SOURCE}"/missing_*_videos.txt 2>/dev/null | sort -u > "${FEATURE_DIR}/missing_files.txt" || true

"${PYTHON}" - "${READY_JSON}" "${FEATURE_DIR}" "${feature_count}" "${ARCHIVE}" <<'PY'
import json
import os
import sys

output, feature_dir, count, archive = sys.argv[1:]
payload = {
    "ready": True,
    "feature_dir": feature_dir,
    "feature_count": int(count),
    "archive": archive,
    "archive_bytes": os.path.getsize(archive) if os.path.exists(archive) else None,
    "support_contract": "original_feature_ownership_cells",
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY

echo "[PHYSTIME_DATA] PASS feature_count=${feature_count} ready=${READY_JSON}"
