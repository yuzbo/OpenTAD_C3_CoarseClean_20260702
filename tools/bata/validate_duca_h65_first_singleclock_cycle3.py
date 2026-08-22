"""Fail-closed validator for the H65 Cycle3 SingleClock contract."""
import argparse, hashlib
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
    p = argparse.ArgumentParser(); p.add_argument("--stage1", required=True); p.add_argument("--sha256", required=True); p.add_argument("--epoch", type=int, required=True); p.add_argument("--chunk-dim", type=int, default=0)
    a = p.parse_args(); cfg = Config.fromfile(str(CONFIG))
    if (_get(cfg,"seed"), _get(cfg,"total_epochs"), _get(cfg,"max_updates")) != (3407,60,6000): raise SystemExit("training contract failed")
    if _get(cfg,"single_clock_unit1",{}).get("k") != 384 or _get(_get(cfg,"model",{}),"single_clock_admission") is not True: raise SystemExit("admission/K failed")
    bb = _get(_get(_get(cfg,"model",{}),"backbone",{}),"backbone",{}); custom = _get(_get(_get(cfg,"model",{}),"backbone",{}),"custom",{})
    if (_get(bb,"total_frames"),_get(bb,"num_frames"),_get(bb,"tubelet_size")) != (768,16,2) or _get(custom,"global_rank_selection") is not True: raise SystemExit("geometry failed")
    if _get(_get(bb,"tubelet_packed_runtime_route",{}),"enabled") is not False or a.chunk_dim != 0: raise SystemExit("packed/chunk contract failed")
    path=Path(a.stage1)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != a.sha256.lower() or a.epoch != 29: raise SystemExit("stage1 path/sha/epoch failed")
    meta=torch.load(path,map_location="cpu",weights_only=False)
    if not isinstance(meta,dict) or meta.get("epoch") != 29 or not isinstance(meta.get("state_dict_ema"),dict): raise SystemExit("stage1 EMA state failed")
    if global_rank_clip_slices(2).shape != (48,16): raise SystemExit("global helper failed")
    print("PASS H65 Cycle3 admission: K=384 updates=6000 epochs=60 seed=3407 packed=disabled chunk_dim=0")
if __name__ == "__main__": main()
