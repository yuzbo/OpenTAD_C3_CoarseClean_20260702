"""Fail-closed static validator. PRE_RUN performs no data, model, GPU, or Slurm access."""
import os, sys, importlib.util, json
from pathlib import Path

def validate(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("PRECHECK_ONLY") != "1" and not argv:
        raise SystemExit("PRECHECK_ONLY=1 required; future pilot/full argv must be declared")
    if any(x in " ".join(argv).lower() for x in ("download", "infer", "train", "metric", "gpu", "slurm", "teacher", "cache", "raw_prediction")):
        raise SystemExit("fail-closed: undeclared execution is forbidden")
    p = Path(__file__).parents[2] / "configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py"
    spec = importlib.util.spec_from_file_location("duca_cfg", p); cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    required = {"shared_dense_reference", "native_uniform_fixed_k", "semantic_actionness_only_fixed_k_control", "semantic_actionness_boundary_fixed_k", "semantic_dynamic_outer_k_headline", "direct_selector_ablation"}
    if set(cfg.arms) != required or cfg.selector.get("selection_strategy") != "semantic_indirect": raise SystemExit("invalid six-arm semantic config")
    if cfg.recovery.get("retention", 0) < 3 or "final-EMA" not in cfg.recovery.get("final_selection", ""): raise SystemExit("invalid recovery contract")
    manifest = os.environ.get("DUCA_HOLD_MANIFEST") or cfg.PRE_RUN.get("hold_manifest")
    if not manifest:
        raise SystemExit("fail-closed: explicit disjoint HOLD manifest is required")
    path = Path(manifest)
    if not path.is_file():
        raise SystemExit(f"fail-closed: HOLD manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit, cal, hold = (set(payload.get(name, [])) for name in ("FIT", "CAL", "HOLD"))
    if not hold or fit & hold or cal & hold or fit & cal:
        raise SystemExit("fail-closed: FIT/CAL/HOLD manifest sets must be non-empty and pairwise disjoint")
    return {"status": "precheck_only", "data_access": False, "execution": False}

if __name__ == "__main__":
    print(validate())
