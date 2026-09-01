import glob
import json


pattern = (
    "/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/"
    "duca_coarse_backends_4f81299_20260723_0015/outputs/*/*/summary.json"
)
for path in sorted(glob.glob(pattern)):
    with open(path, "r") as handle:
        payload = json.load(handle)
    final = payload["final_val"]
    comparison = final["indirect_selection_quality"]["strategy_comparison"]
    print(
        "\t".join(
            str(value)
            for value in (
                payload["official_action_seg_backend"],
                final["average_precision"],
                final["roc_auc"],
                final["best_f1"],
                final["accuracy"],
                final["balanced_accuracy"],
                final["seconds"],
                comparison["best_boundary_support_strategy"],
                comparison["boundary_support_r1_by_strategy"]["delta_p_action"],
            )
        )
    )
