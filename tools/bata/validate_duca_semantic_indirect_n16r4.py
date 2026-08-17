"""Data-free validator using the real train parser and selector constructors."""
import os, sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

def validate(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("PRECHECK_ONLY") != "1" and not argv:
        raise SystemExit("PRECHECK_ONLY=1 required; future pilot/full argv must be declared")
    if any(x in " ".join(argv).lower() for x in ("download", "infer", "metric", "gpu", "slurm", "teacher", "cache", "raw_prediction")):
        raise SystemExit("fail-closed: undeclared execution is forbidden")
    p = Path(__file__).parents[2] / "configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py"
    spec = importlib.util.spec_from_file_location("duca_cfg", p); cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    required = {"dense_placeholder", "native_uniform_fixed_k", "actionness_only_fixed_k_control", "actionness_boundary_fixed_k", "actionness_boundary_dynamic_k_headline", "direct_selector_ablation"}
    if set(cfg.arms) != required or cfg.selector.get("selection_strategy") != "semantic_indirect": raise SystemExit("invalid six-arm semantic config")
    from tools.bata.duca_semantic_cycle2_contract import build_arm, validate_manifests, validate_deploy_entry
    validate_manifests(cfg.manifests)
    policies = {}
    for arm in sorted(required):
        validate_deploy_entry(cfg.arms[arm]["data_entry"])
        policies[arm] = build_arm(cfg, arm)
    if policies["actionness_only_fixed_k_control"]["policy"].get("semantic_acquisition") != "actionness_only": raise SystemExit("actionness-only policy mismatch")
    if policies["actionness_boundary_dynamic_k_headline"]["policy"].get("dynamic_k_min") == policies["actionness_boundary_dynamic_k_headline"]["policy"].get("dynamic_k_max"): raise SystemExit("dynamic arm is fixed-K")
    if policies["direct_selector_ablation"]["policy"].get("selection_strategy") != "frame_score_topk": raise SystemExit("direct ablation mismatch")
    return {"status": "precheck_only", "constructed_arms": sorted(required), "data_access": False, "execution": False}

if __name__ == "__main__":
    print(validate())
