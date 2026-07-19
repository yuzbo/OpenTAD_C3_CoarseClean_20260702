---
type: experiment
node_id: exp:native-crop-s1-vertical-slice
title: "Native-Crop S1 development vertical slice"
stage: tested
status: cuda_gate_pass_crop_sufficiency_protocol_pending
outcome: implementation_graph_and_provenance_pass
tags: ["offline-tad", "native-crop", "source-coordinate", "vertical-slice"]
added: 2026-07-20
---

# Native-Crop S1 Development Vertical Slice

## Purpose

Validate the real research object before any crop-sufficiency or learned-policy
experiment: preserve the 768-point time axis, crop local source pixels before
resize, encode global/local views with one shared VideoMAE-S, fuse at 384
points, and preserve the AdaTAD-derived `[B,384,768]` feature contract.

## Implemented

- `NativeCropSourceViews`: decoded uint8 `global96` letterbox plus exact
  source-coordinate `center-local128`, with local interpolation and padding
  disabled.
- `NativeCropBackboneWrapper`: one shared VideoMAE instance, runtime 6x6/8x8
  position interpolation, 384-point fixed-mean fusion, deterministic 2x
  temporal interpolation.
- Development-only config with no `dataset.test`, no teacher/oracle, and
  generic train/test entrypoints fail-closed.
- Geometry census, exact checkpoint/gradient/full-model precheck, stage-wise
  cost schema, Slurm gate launcher, and focused tests.
- Frozen split/manifest/source hashes, a deterministic training-only annotation
  builder, exact 160/40/129/664 population checks, record-derived census
  validation, reference detector/NMS parity, and per-branch gradient checks.

## Evidence

- Clean formal Git commit:
  `0bf59be877eeb6879166893641c12bc4e60a2b53`.
- Clean remote replay: `173 passed` across Native-Crop, Spatial-Zoom S1,
  and required C3 focused suites.
- Formal Slurm CUDA gate Job `1174671`: `COMPLETED 0:0` on `g0059` in
  `00:01:40`, using one Slurm-visible GPU and process-local `cuda:0`.
- Gate run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/native_crop_s1_0bf59be_20260720_0225`.
- Full-model precheck internal/file SHA-256:
  `ba278a191905b492d78b07ec253857774a0311c363b70ceac8921159a855b0fc`
  /
  `b0cfe61261f39ef801be6b5800510d9feff54b0f0b73babfcc00d091a33bccde`.
- Formal census internal/file SHA-256:
  `9a688297fec2ac8537a44e46254df2e277ec69785cfa179aac658ad0b0dc39d8`
  /
  `99089f8260a7a8c7ec3610e521e638f5c18e297c11217597e5b7b139407b013b`.
- Real decode sample:
  - `global [1,3,768,96,96] uint8`;
  - `local [1,3,768,128,128] uint8`;
  - source `320x180`;
  - local box `[96,26,224,154]`;
  - padding `[0,0,0,0]`.
- Manifest-bound population audit:
  - fit: 160 videos;
  - gate: 40 videos, 129 windows;
  - complete development sliding population: 200 videos, 664 windows;
  - development-only annotation SHA:
    `0985d3711ab31f404ff0be5a1ba75420796a6807d486410337078b38090bf749`;
  - sealed-test files opened: 0.
- Geometry census:
  - 200 development videos, zero sealed-test files;
  - all source streams `320x180`;
  - 96/112/128 no-padding rates all 100%;
  - census SHA `73290dd5abbcac6e5a2da1945b8ebd5b44f2d62e5a570c549aee46679548a9f8`.

The formal CUDA gate additionally verified:

- expected commit equals actual commit; worktree is completely clean;
- all audited source files are tracked and byte-equal to their `HEAD` blobs;
- 200 census records are re-probed against current source files;
- checkpoint state/core contract is `163/161/22,482,048`, with no core
  missing, shape, or value mismatches;
- runtime position grids are exactly `6x6` and `8x8`;
- global/local features are `[1,384,384]`, fused detector input is
  `[1,384,768]`;
- detector loss reaches both branch features with `62,178` nonzero gradient
  elements each, plus nonzero backbone, projection, and head gradients;
- all present gradients are finite;
- official-test annotation records and video files opened are both zero;
- teacher, oracle, and paper-claim flags remain false.

The inherited 0.25 validation overlap silently omitted the final short-action
video `video_validation_0000054`. The isolated Native-Crop config now uses 0.5
overlap and covers all 200 development identities. Historical R0 is unchanged.

The first independent max audit returned `HOLD` and found two P0 defects:
`geometry_census_path` was attached to the wrong function signature and
synthetic uint8 views were left on CPU after the model moved to CUDA. Both are
fixed. Its P1 findings drove the frozen manifest/annotation, full census,
complete VideoMAE core, reference detector parity, branch-gradient, temporal
order, checkpointed-grid, and CLI tests. A second independent pass is pending.

The second pass found no P0 but returned `HOLD` on two evidence-integrity P1s:
the expected commit did not prove that audited working-tree files equalled its
Git blobs, and a self-hashed geometry census was not re-probed against the
current video files. It also found one P2 intersection-area error for sources
that are smaller on only one axis. The repair now requires a full expected
commit, a completely clean worktree, tracked and byte-equal `HEAD` sources,
and an in-gate re-probe of all 200 files (root containment, size, dimensions,
rotation, frame count, and frame rate). Intersection-area statistics and
forged/replaced-source negative tests were added. Remote focused verification
is `17 passed`. The same max auditor's third pass returned `DEPLOY` with
`P0/P1/P2/P3 = 0`.

## Boundaries

No crop accuracy, mAP, measured full-stack cost, oracle upper bound, learned
policy, official-test result, GO/KILL, or paper claim exists yet. The CUDA
gate proves implementation/provenance/gradient closure only. The geometry
census establishes feasibility of no-padding local128, not semantic
sufficiency.

## Next gate

Initiate the protocol discussion needed to freeze a development-only
crop-sufficiency experiment: candidate coverage, teacher split/cache,
matched-pixel/FLOP baselines, uncertainty, and GO/KILL margins. Do not
implement learned ROI before that protocol and sufficiency result.
