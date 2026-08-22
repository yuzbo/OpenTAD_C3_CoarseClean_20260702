"""Fail-closed validator for the H65 Cycle3 SingleClock contract."""
import argparse, hashlib, os
from pathlib import Path
import torch
from opentad.models.duca.structured_selection import exact_uniform_positions
try:
    from mmengine.config import Config
except ImportError:
    from mmcv import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/adatad/thumos/duca_h65_first_singleclock_cycle3.py"

def _get(o, k, d=None): return o.get(k, d) if hasattr(o, "get") else getattr(o, k, d)

def global_rank_clip_slices(batch, clips=24, clip_len=16, tubelet=2):
    pos = exact_uniform_positions(clips * clip_len * tubelet, clips * clip_len)
    return pos.repeat(batch, 1).reshape(batch, clips, -1).reshape(batch * clips, -1)

def main():
    p = argparse.ArgumentParser(description="Validate the H65 Cycle3 Stage1->Stage2 contract.")
    p.add_argument("--stage1", required=True, help="terminal Stage1 checkpoint path")
    p.add_argument("--sha256", required=True, help="lowercase SHA256 of --stage1")
    p.add_argument("--epoch", type=int, required=True, help="Stage1 checkpoint epoch; must be 29")
    p.add_argument("--chunk-dim", type=int, default=0)
    a = p.parse_args()
    path = Path(a.stage1)
    if not path.is_file():
        raise SystemExit(f"stage1 checkpoint does not exist: {path}")
    if a.epoch != 29:
        raise SystemExit("stage1 checkpoint epoch must be 29")
    if len(a.sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in a.sha256):
        raise SystemExit("stage1 checkpoint sha256 must be 64 hexadecimal characters")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != a.sha256.lower():
        raise SystemExit("stage1 checkpoint sha256 mismatch")
    # The inherited Stage2 config intentionally requires these names while it
    # is evaluated.  Establish the exact CLI contract before Config.fromfile,
    # so missing context cannot leak as a traceback ValueError.
    os.environ["DUCA_STAGE1_CHECKPOINT"] = str(path.resolve())
    os.environ["DUCA_STAGE1_CHECKPOINT_SHA256"] = actual_sha
    os.environ["DUCA_STAGE1_CHECKPOINT_EPOCH"] = "29"
    try:
        cfg = Config.fromfile(str(CONFIG))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"cycle3 config rejected: {exc}") from None
    if (_get(cfg,"seed"), _get(cfg,"total_epochs"), _get(cfg,"max_updates")) != (3407,60,6000): raise SystemExit("training contract failed")
    if _get(cfg,"single_clock_unit1",{}).get("k") != 384 or _get(_get(cfg,"model",{}),"single_clock_admission") is not True: raise SystemExit("admission/K failed")
    bb = _get(_get(_get(cfg,"model",{}),"backbone",{}),"backbone",{}); custom = _get(_get(_get(cfg,"model",{}),"backbone",{}),"custom",{})
    if (_get(bb,"total_frames"),_get(bb,"num_frames"),_get(bb,"tubelet_size")) != (768,16,2) or _get(custom,"global_rank_selection") is not True: raise SystemExit("geometry failed")
    if _get(_get(bb,"tubelet_packed_runtime_route",{}),"enabled") is not False or a.chunk_dim != 0: raise SystemExit("packed/chunk contract failed")
    meta=torch.load(path,map_location="cpu",weights_only=False)
    if not isinstance(meta,dict) or meta.get("epoch") != 29 or not isinstance(meta.get("state_dict_ema"),dict): raise SystemExit("stage1 EMA state failed")
    if global_rank_clip_slices(2).shape != (48,16): raise SystemExit("global helper failed")
    print("PASS H65 Cycle3 admission: K=384 updates=6000 epochs=60 seed=3407 packed=disabled chunk_dim=0")
if __name__ == "__main__": main()
