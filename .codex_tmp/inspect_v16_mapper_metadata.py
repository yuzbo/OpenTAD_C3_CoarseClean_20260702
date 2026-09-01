#!/usr/bin/env python3
import glob
import json
import os
import sys

import numpy as np


def walk(value, path="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            lowered = key.lower()
            if any(
                token in lowered
                for token in (
                    "axis",
                    "domain",
                    "endpoint",
                    "configured",
                    "mapping",
                    "mapper",
                    "extent",
                    "coordinate",
                )
            ):
                rendered = repr(item)
                print(f"JSON {child}={rendered[:500]}")
            walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            walk(item, f"{path}[{index}]")


root = sys.argv[1]
for manifest in sorted(
    glob.glob(os.path.join(root, "*", "direct_work", "gpu1_id0", "decode_replay_manifest.json"))
):
    print(f"MANIFEST {manifest}")
    with open(manifest, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(f"TOP_KEYS {sorted(payload)}")
    walk(payload)
    npz_path = os.path.join(os.path.dirname(manifest), "decode_replay_inputs.npz")
    print(f"NPZ {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        for key in sorted(arrays.files):
            value = arrays[key]
            lowered = key.lower()
            if any(
                token in lowered
                for token in (
                    "axis",
                    "domain",
                    "endpoint",
                    "configured",
                    "mapping",
                    "mapper",
                    "extent",
                    "coordinate",
                    "point",
                    "position",
                    "count",
                )
            ):
                flat = value.reshape(-1)
                sample = flat[:5].tolist()
                tail = flat[-5:].tolist() if flat.size else []
                print(
                    f"ARRAY {key} shape={value.shape} dtype={value.dtype} "
                    f"head={sample} tail={tail}"
                )
