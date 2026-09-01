#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=/data/run01/sczc063/yuzibo
PROJECT_ROOT="$USER_ROOT/projects"
V16_RUNTIME="$PROJECT_ROOT/opentad_sparsehead_approach_a_20260729_v16"
V16_RUN="$PROJECT_ROOT/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v16"
PYTHON_BIN="$USER_ROOT/conda_envs/opentad/bin/python"

echo "SECTION v16_identity"
git -C "$V16_RUNTIME" rev-parse --abbrev-ref HEAD
git -C "$V16_RUNTIME" rev-parse HEAD
git -C "$V16_RUNTIME" rev-parse 'HEAD^{tree}'
git -C "$V16_RUNTIME" status --porcelain

echo "SECTION v16_completion_paths"
find "$V16_RUN" -type f \
  \( -name 'DECODE_CROSS_COMPLETE.json' \
     -o -name 'DECODE_CROSS_EVIDENCE_SUITE_COMPLETE.json' \
     -o -name 'DECODE_CROSS_EVIDENCE_SUITE_VALIDATED.json' \
     -o -name 'capture_manifest.json' \
     -o -name 'decode_replay_capture.npz' \
     -o -name 'decoded_candidates.npz' \
     -o -name 'evaluation_metrics.json' \
     -o -name 'result_detection.json' \) \
  -print | sort | head -n 240

echo "SECTION v16_completion_summary"
"$PYTHON_BIN" - "$V16_RUN" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.rglob("DECODE_CROSS_COMPLETE.json")):
    payload = json.loads(path.read_text())
    print(json.dumps({
        "path": str(path),
        "status": payload.get("status"),
        "validation_pass": payload.get("validation_pass"),
        "keys": sorted(payload),
        "capture": payload.get("capture"),
        "artifacts": payload.get("artifacts"),
        "modes": payload.get("modes"),
    }, sort_keys=True))
PY

echo "SECTION sdpq_candidates"
find "$USER_ROOT" -maxdepth 9 -type f \
  \( -iname '*sdpq*.py' \
     -o -iname '*sdpq*.pth' \
     -o -iname '*sdpq*.pth.tar' \
     -o -iname '*sdpq*.pt' \) \
  -print 2>/dev/null | sort | head -n 240

echo "SECTION official_actionformer_candidates"
find "$USER_ROOT" -maxdepth 8 \
  \( -type d -name 'actionformer_release' \
     -o -type f -name 'thumos_i3d.yaml' \
     -o -type f -name 'thumos.tar.gz' \
     -o -type d -name 'i3d_features' \
     -o -type f -name 'epoch_034.pth.tar' \
     -o -type f -name 'eval_results.pkl' \) \
  -print 2>/dev/null | sort | head -n 320

echo "SECTION annotation_candidates"
find "$USER_ROOT" -maxdepth 9 -type f -name 'thumos14.json' \
  -print 2>/dev/null | sort | head -n 160
