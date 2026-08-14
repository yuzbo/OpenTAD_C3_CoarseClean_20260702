import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("duca_dynamic_physical", Path(__file__).parents[1] / "opentad/models/selectors/duca_dynamic_physical.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
dynamic_outer_k = _MOD.dynamic_outer_k
f1_uniform_positions = _MOD.f1_uniform_positions
f2_nonce_shuffle = _MOD.f2_nonce_shuffle
bounded_monotone_local_exact_k = _MOD.bounded_monotone_local_exact_k
attach_physical_timestamps = _MOD.attach_physical_timestamps


def test_outer_k_monotone_and_bounded():
    vals = [dynamic_outer_k(x, min_k=8, target_k=16, max_k=24) for x in (0, .2, .5, .8, 1)]
    assert vals == sorted(vals) and vals[0] == 8 and vals[-1] == 24


def test_f1_exact_uniform_and_f2_deterministic():
    p = f1_uniform_positions(10, 4)
    assert p == [0, 3, 6, 9]
    assert f2_nonce_shuffle(p, "n") == f2_nonce_shuffle(p, "n")


def test_bounded_transport_unique_order_and_mask():
    out = bounded_monotone_local_exact_k([0, 9, 1, 8, 2, 7], 3, local_radius=2, valid_mask=[1, 1, 1, 1, 0, 0])
    assert len(out) == 3 and out == sorted(set(out)) and max(out) < 4


def test_timestamp_metadata_stage():
    m = attach_physical_timestamps({}, [0, 2], fps=2.0)
    assert m["duca_physical_timestamps"] == [0.0, 1.0]
    assert m["duca_timestamp_stage"].startswith("before_")
