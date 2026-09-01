# Builder return — DUCA H65 SingleClock identity duplicate correction

- **Parent revision:** `e866a9ae52dd64b775854029d09ce72a6c86ad01`
- **Clean correction commit:** `c1a77e3f918c4c0bf653fe35231f4614570c6f5f`
- **Evidence class:** deterministic identity-accounting implementation; no efficacy evidence.

## MINIMAL_CHANGE_PLAN and implementation

The physical sample identity remains `(video_name, window_start_frame)`.  The
identity audit now keeps one record for an identical repeated tail-window
exposure, records total/unique/duplicate exposure counts, and fails closed when
the same physical key has different selected RGB, positions, mask, or valid
lengths.  The terminal ON/gate-zero comparison also requires the duplicate
accounting to match.

Modified files are limited to:

- `opentad/models/detectors/actionformer.py`
- `tools/bata/finalize_duca_h65_singleclock_terminal.py`
- `tests/test_duca_h65_cycle4_singleclock_contract.py`
- `tests/test_duca_h65_singleclock_finalizer.py`

`python -m py_compile` passed.  Local focused pytest collection was unavailable
because the Windows PyTorch `c10.dll` failed to initialize (`WinError 1114`), so
the same focused checks remain required in the N16R4 project environment.

No model computation, selector, checkpoint, selected RGB, physical position,
dataset, evaluator, configuration, launcher, scientific threshold, data access,
training, inference, metric, or efficacy claim changed.

- **next_owner:** independent Critic
- **next_action:** read-only review of the exact clean commit and focused N16R4 tests
- **dependency:** clean binding at `c1a77e3f...`
- **expected_return:** static pass or one concrete deterministic blocker
- **single_recovery:** none consumed by this evidence-only repair
