#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343
echo '=== R0 FILES ==='
find "$ROOT" -type f \( -iname '*r0*summary*.json' -o -iname '*bootstrap*.json' -o -iname '*prediction*.json' -o -iname '*map*.json' \) -printf '%s %p\n' 2>/dev/null | sort -n | tail -80

SUMMARY=$(find "$ROOT" -type f -name 'r0_summary.json' 2>/dev/null | head -1)
BOOT=$(find "$ROOT" -type f -name 'r0_bootstrap.json' 2>/dev/null | head -1)
echo "SUMMARY=$SUMMARY"
echo "BOOT=$BOOT"

/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python - "$SUMMARY" "$BOOT" <<'PY'
import json, os, statistics, sys

for label, path in zip(("summary", "bootstrap"), sys.argv[1:]):
    print(f"=== {label.upper()} STRUCTURE ===")
    if not path or not os.path.isfile(path):
        print("missing")
        continue
    data = json.load(open(path, encoding="utf-8"))
    print("path", path)
    print("top_keys", sorted(data) if isinstance(data, dict) else type(data).__name__)
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "rows" and isinstance(value, list):
                print(key, "families", [row.get("family") for row in value])
                continue
            if key == "sampled_average_mAP" and isinstance(value, dict):
                print(key, {name: len(samples) for name, samples in value.items()})
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                print(key, repr(value))
            elif isinstance(value, dict):
                print(key, "dict_keys", sorted(value)[:80])
                for k2, v2 in value.items():
                    if isinstance(v2, (str, int, float, bool)) or v2 is None:
                        print(f"  {k2}", repr(v2))
                    elif isinstance(v2, list):
                        print(f"  {k2}", "list", len(v2), "head", v2[:5])
            elif isinstance(value, list):
                print(key, "list", len(value), "head", value[:5])

boot_path = sys.argv[2]
if boot_path and os.path.isfile(boot_path):
    data = json.load(open(boot_path, encoding="utf-8"))
    sampled = data["sampled_average_mAP"]
    base = sampled[data["baseline_family"]]
    print("=== PAIRED DELTA DISTRIBUTIONS ===")
    for family in data["comparisons"]:
        delta = sorted((x - y) * 100.0 for x, y in zip(sampled[family], base))
        def q(p):
            pos = p * (len(delta) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(delta) - 1)
            frac = pos - lo
            return delta[lo] * (1 - frac) + delta[hi] * frac
        print(family, {
            "mean_pp": statistics.fmean(delta),
            "median_pp": statistics.median(delta),
            "q025_pp": q(.025),
            "q25_pp": q(.25),
            "q75_pp": q(.75),
            "q975_pp": q(.975),
            "positive_fraction": sum(x > 0 for x in delta) / len(delta),
            "above_0p2_fraction": sum(x > .2 for x in delta) / len(delta),
        })
PY
exit 0
