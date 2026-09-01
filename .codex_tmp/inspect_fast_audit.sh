#!/usr/bin/env bash
set -u

AUDIT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107/arms/trainfree_slowfast_fast_fusion_r2q3/official60/work/gpu1_id0/duca_selected_axis_training_audit.json

/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python - "${AUDIT}" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
print("top_keys", sorted(data))
needles = (
    "selected", "budget", "hole", "gap", "repair", "transition", "boundary",
    "gradient", "update", "cost", "latency", "flop", "source", "probe",
)

def walk(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{prefix}.{key}" if prefix else key
            low = key.lower()
            if any(needle in low for needle in needles) and not isinstance(item, (dict, list)):
                print(here, repr(item))
            walk(item, here)
    elif isinstance(value, list) and len(value) <= 5:
        for index, item in enumerate(value):
            walk(item, f"{prefix}[{index}]")

walk(data)
PY
