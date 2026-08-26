"""Fail-closed, read-only validator for the matched PJST-D1 OFF/ON Stage-2 configs.

Proves, without loading any model, data, checkpoint bytes, or GPU:

- the supplied Stage-1 epoch-29 checkpoint is a readable regular file whose
  streaming SHA-256 matches the supplied digest exactly (no fabricated path or
  all-zero digest can pass);
- both configs resolve and share the full selector / acquisition / data / model /
  loss / evaluator / optimizer / schedule / seed contract;
- the OFF/ON distinction is exactly ``work_dir`` and
  ``model.backbone.custom.pjst_derivative_only``;
- the learned H65 nonuniform selector is frozen (``policy_alpha`` pinned 1.0,
  detector-to-selector and auxiliary-selector adaptation routes pinned 0);
- the 60-epoch / 6000-successful-update contract, every-5-epoch resumable
  checkpoints, fixed final/final-EMA rule, fresh distinct output roots, and the
  required Stage-1 epoch-29 checkpoint binding (exact path/digest/epoch retained
  in both resolved configs).
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFF_CFG = ROOT / "configs/adatad/thumos/duca_pjst_d1_stage2_off.py"
ON_CFG = ROOT / "configs/adatad/thumos/duca_pjst_d1_stage2_on.py"

# The only two leaves that may differ between the matched arms.
ALLOWED_DIFF_PATHS = {
    ("work_dir",),
    ("model", "backbone", "custom", "pjst_derivative_only"),
}

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256_file(path):
    """Stream the file's SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff_paths(a, b, prefix=()):
    diffs = set()
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                diffs.add(prefix + (key,))
            else:
                diffs |= _diff_paths(a[key], b[key], prefix + (key,))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.add(prefix)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs |= _diff_paths(x, y, prefix + (i,))
    elif a != b:
        diffs.add(prefix)
    return diffs


def _resolve(path, checkpoint, sha256, epoch):
    """Resolve a config with the exact supplied Stage-1 binding (no fallback)."""
    os.environ["DUCA_STAGE1_CHECKPOINT"] = checkpoint
    os.environ["DUCA_STAGE1_CHECKPOINT_SHA256"] = sha256
    os.environ["DUCA_STAGE1_CHECKPOINT_EPOCH"] = str(epoch)
    return Config.fromfile(str(path)).to_dict()


def _schedule_value(schedule, key):
    entry = schedule.get(key, {})
    if not isinstance(entry, dict):
        return None
    return (entry.get("start"), entry.get("end"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True,
                        help="Stage-1 epoch-29 checkpoint path (required; no fabricated fallback)")
    parser.add_argument("--sha256", required=True,
                        help="Stage-1 checkpoint SHA-256 (required; exactly 64 hex chars)")
    parser.add_argument("--epoch", type=int, default=29)
    args = parser.parse_args()

    # 0. Fail closed before any config admission.
    if args.epoch != 29:
        raise SystemExit(f"epoch must be exactly 29, got {args.epoch}")
    checkpoint = Path(args.stage1)
    if not checkpoint.is_file():
        raise SystemExit(f"Stage-1 checkpoint is not a readable regular file: {args.stage1}")
    if not os.access(checkpoint, os.R_OK):
        raise SystemExit(f"Stage-1 checkpoint is not readable: {args.stage1}")
    if not SHA256_RE.match(args.sha256):
        raise SystemExit("Stage-1 sha256 must match ^[0-9a-fA-F]{64}$")
    actual_sha = _sha256_file(checkpoint)
    if actual_sha.lower() != args.sha256.lower():
        raise SystemExit(
            f"Stage-1 checkpoint sha256 mismatch: supplied={args.sha256} actual={actual_sha}"
        )

    off = _resolve(OFF_CFG, args.stage1, args.sha256, args.epoch)
    on = _resolve(ON_CFG, args.stage1, args.sha256, args.epoch)

    # 1. Sole distinction: only work_dir + the PJST flag may differ.
    diffs = _diff_paths(off, on)
    unexpected = diffs - ALLOWED_DIFF_PATHS
    if unexpected:
        raise SystemExit(f"OFF/ON differ outside the sole PJST flag: {sorted(str(p) for p in unexpected)}")
    if ("work_dir",) not in diffs or ("model", "backbone", "custom", "pjst_derivative_only") not in diffs:
        raise SystemExit("expected work_dir and pjst_derivative_only to differ between arms")
    if off.get("model", {}).get("backbone", {}).get("custom", {}).get("pjst_derivative_only") is not False:
        raise SystemExit("OFF arm must set pjst_derivative_only=False")
    if on.get("model", {}).get("backbone", {}).get("custom", {}).get("pjst_derivative_only") is not True:
        raise SystemExit("ON arm must set pjst_derivative_only=True")

    # 2. Shared scientific/training contract (identical in both arms).
    if (off.get("seed"), off.get("total_epochs"), off.get("max_updates")) != (3407, 60, 6000):
        raise SystemExit("training contract (seed/total_epochs/max_updates) failed")
    wf = off.get("workflow", {})
    if wf.get("end_epoch") != 60 or wf.get("expected_successful_optimizer_updates") != 6000:
        raise SystemExit("60-epoch / 6000-successful-update contract failed")

    # 3. Checkpoint policy: every 5 epochs, resumable, fixed final/final-EMA.
    if wf.get("checkpoint_interval") != 5:
        raise SystemExit("checkpoint_interval must be 5 (every-five-epoch resumable)")
    if wf.get("require_resumable_training_state") is not True:
        raise SystemExit("resumable training state must be required")
    if wf.get("primary_checkpoint_epoch") != 59 or wf.get("primary_checkpoint_state_key") != "state_dict_ema":
        raise SystemExit("fixed final/final-EMA rule failed")

    # 4. Stage-1 epoch-29 checkpoint binding retained exactly (no coercion/override).
    for name, cfg in (("OFF", off), ("ON", on)):
        mi = cfg.get("workflow", {}).get("model_initialization", {})
        if mi.get("state_key") != "state_dict_ema":
            raise SystemExit(f"{name} Stage-1 state_key must be state_dict_ema")
        if mi.get("expected_checkpoint_epoch") != 29:
            raise SystemExit(f"{name} Stage-1 expected_checkpoint_epoch must be 29")
        if mi.get("checkpoint_path") != args.stage1:
            raise SystemExit(f"{name} config did not retain the exact supplied checkpoint path")
        if mi.get("checkpoint_sha256") != args.sha256:
            raise SystemExit(f"{name} config did not retain the exact supplied checkpoint sha256")

    # 5. Selector freeze: learned H65 nonuniform, no adaptation routes.
    lws = off.get("model", {}).get("frame_selector", {}).get("loss_weight_schedule", {})
    if _schedule_value(lws, "policy_alpha") != (1.0, 1.0):
        raise SystemExit("policy_alpha must be pinned to 1.0 (learned H65, not uniform fallback)")
    for key in ("detector_gradient", "detector_contribution", "asformer_adapt"):
        if _schedule_value(lws, key) != (0.0, 0.0):
            raise SystemExit(f"selector freeze route {key} must be pinned to 0.0")

    # 6. Fresh, distinct output roots.
    off_root = off.get("work_dir")
    on_root = on.get("work_dir")
    if not off_root or not on_root or off_root == on_root:
        raise SystemExit("OFF/ON output roots must be distinct and non-empty")

    # 7. Stage-1 state identity: both arms admit no SingleClock but register the
    # zero architecture-identity scalar required by the frozen epoch-29 state
    # dict (backbone.model.backbone.blocks.0.relative_physical_time_scale).
    for name, cfg in (("OFF", off), ("ON", on)):
        model = cfg.get("model", {})
        if model.get("single_clock_admission") is not False:
            raise SystemExit(f"{name} arm must set single_clock_admission=False")
        bb = model.get("backbone", {}).get("backbone", {})
        if bb.get("relative_physical_time_residual") is not True:
            raise SystemExit(
                f"{name} arm must set relative_physical_time_residual=True "
                "(registers the zero block-0 identity scalar)"
            )

    print(
        "PASS PJST-D1 matched configs: "
        "sole_distinction=[work_dir, pjst_derivative_only] "
        "seed=3407 epochs=60 updates=6000 checkpoint_interval=5 "
        f"roots=[{off_root}, {on_root}] "
        f"stage1_epoch29_checkpoint={args.stage1}"
    )


if __name__ == "__main__":
    main()
