#!/usr/bin/env bash
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1

python - <<'PY'
import hashlib
import json
import pickle
from pathlib import Path

path = Path(
    "/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/"
    "actionformer_official_repro_20260729_v4/eval_results_epoch034.pkl"
)
with path.open("rb") as handle:
    payload = pickle.load(handle)
video_ids = sorted(set(payload["video-id"]))
print(
    json.dumps(
        {
            "prediction_count": len(payload["video-id"]),
            "prediction_video_count": len(video_ids),
            "prediction_video_ids_sha256": hashlib.sha256(
                json.dumps(
                    video_ids,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "first_video": video_ids[0],
            "last_video": video_ids[-1],
        },
        sort_keys=True,
    )
)
PY
