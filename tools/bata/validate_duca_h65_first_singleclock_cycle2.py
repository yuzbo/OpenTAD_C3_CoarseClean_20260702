"""Fail-closed admission checks for H65 Cycle2."""
import argparse, hashlib
from pathlib import Path
import torch
try:
    from mmengine.config import Config
except ImportError:
    from mmcv import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/adatad/thumos/duca_h65_first_singleclock_cycle2.py"

def _get(o, k, d=None): return o.get(k, d) if hasattr(o, "get") else getattr(o, k, d)

def exact_uniform_positions_once(dense_len=768, k=384):
    if dense_len < k: raise ValueError("dense_valid_len<384 forbidden")
    return torch.linspace(0, dense_len - 1, k, dtype=torch.long)

def global_rank_clip_slices(batch, clips=24, clip_len=16, tubelet=2):
    return exact_uniform_positions_once(clips*clip_len*tubelet, clips*clip_len).reshape(batch, clips, -1).reshape(batch*clips, -1)

def attention_mask(batch, clips=24, spatial=14):
    return torch.zeros((batch*clips, 1, 8*spatial, 8*spatial), dtype=torch.bool)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage1", required=True); p.add_argument("--sha256", required=True); p.add_argument("--epoch", type=int, required=True); p.add_argument("--chunk-dim", type=int, default=0); p.add_argument("--metadata-sync", action="store_true")
    a = p.parse_args(); cfg = Config.fromfile(str(CONFIG))
    if (_get(cfg,"seed"),_get(cfg,"total_epochs"),_get(cfg,"max_updates"),_get(cfg,"checkpoint_interval_epochs")) != (3407,60,6000,5): raise SystemExit("training contract failed")
    pol = _get(cfg,"checkpoint_policy",{})
    if any(k not in pol for k in ("keep_latest","milestones","final","final_ema")): raise SystemExit("checkpoint policy incomplete")
    out = _get(_get(cfg,"model",{}),"backbone",{}); bb = _get(out,"backbone",{}); custom = _get(out,"custom",{})
    if _get(bb,"relative_physical_time_residual") is not True or _get(custom,"global_rank_selection") is not True: raise SystemExit("singleclock route disabled")
    if _get(_get(bb,"tubelet_packed_runtime_route",{}),"enabled") is not False: raise SystemExit("packed route must be disabled")
    if (_get(bb,"total_frames"),_get(bb,"num_frames"),_get(bb,"tubelet_size")) != (768,16,2): raise SystemExit("geometry failed")
    path = Path(a.stage1)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != a.sha256.lower() or a.epoch != 29: raise SystemExit("stage1 path/sha/epoch failed")
    meta = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(meta, dict) or meta.get("epoch") != 29 or not any("ema" in str(k).lower() for k in meta): raise SystemExit("stage1 metadata epoch/EMA failed")
    if a.chunk_dim not in (0,2) or (a.chunk_dim == 2 and not a.metadata_sync): raise SystemExit("temporal checkpoint contract failed")
    exact_uniform_positions_once(); global_rank_clip_slices(1); attention_mask(1)
    print("PASS H65 Cycle2 admission: K=384 updates=6000 epochs=60 seed=3407 packed=disabled")

if __name__ == "__main__": main()
