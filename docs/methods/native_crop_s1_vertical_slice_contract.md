# Native-Crop S1 Vertical-Slice Contract

## Research identity

This is a development-only infrastructure gate for dense-time spatial crop
experiments in offline TAD. It is not the historical Dense-R0 resolution
matrix, a learned ROI policy, an oracle experiment, or an official-test run.

The vertical slice asks one implementation question:

> Can decoded source frames be cropped before any full-frame spatial resize,
> encoded by one shared VideoMAE-S instance, fused at 384 temporal points, and
> delivered to the unchanged AdaTAD-derived ActionFormer projection/head as
> `[B,384,768]`?

Passing this contract proves only that the intended computation graph exists.
It does not prove crop sufficiency, accuracy, efficiency, novelty, or
publishability.

## Frozen vertical slice

1. Input population is the frozen THUMOS14 development fit/gate identity only.
   The gate accepts only the immutable 200-record `training`-only annotation
   with SHA-256
   `0985d3711ab31f404ff0be5a1ba75420796a6807d486410337078b38090bf749`.
   `dataset.test` is absent and the official-test root is rejected.
2. `LoadFrames` keeps the complete 768-point temporal window.
3. `NativeCropSourceViews` runs immediately after `DecordDecode`.
4. The global branch letterboxes the complete decoded frame to `96x96`.
5. The local branch takes a fixed `128x128` center crop in decoded source
   coordinates. It performs no interpolation. Padding is disabled.
6. Both views remain CPU `uint8` until the existing action data preprocessor.
   The full source video is never materialized as a float tensor.
7. One shared VideoMAE-S instance encodes both views. Runtime position
   interpolation targets are `6x6` and `8x8`.
8. Each branch produces `[B,384,384]`. Parameter-free fixed-mean fusion is used
   so this gate does not confound crop truth with fusion learning.
9. Audited deterministic 2x temporal interpolation produces `[B,384,768]`.
10. Existing ActionFormer projection, head, loss, and NMS remain unchanged.

The wrapper accepts exactly two input keys, `global` and `local`; extra
teacher, oracle, GT, cache, or test-evidence inputs fail closed.

## Development population

The frozen split contains 160 fit and 40 gate videos. The inherited 0.25
sliding overlap omitted `video_validation_0000054`, whose only 0.7-second
action occurs near the video end. Native-Crop development validation therefore
uses an explicit 0.5 overlap, yielding all 40 gate videos and 129 gate windows.
The complete development sliding population is 200 videos and 664 windows.
This correction is isolated to the new Native-Crop config; historical R0 is
unchanged.

## Geometry census

The 2026-07-20 development-only census inspected 200 fit/gate video streams and
zero sealed-test files. Census SHA-256:

`73290dd5abbcac6e5a2da1945b8ebd5b44f2d62e5a570c549aee46679548a9f8`

All available THUMOS development files decode at `320x180` with aspect ratio
`16:9`. This is the native resolution of the available experiment source, not
a claim about the original camera acquisition. Crops of 96, 112, and 128
source pixels all have 100% no-padding feasibility. Their frame-area fractions
are 16.00%, 21.78%, and 28.44%, respectively.

These observations authorize the no-padding `local128` implementation. They do
not by themselves establish that 128 pixels are sufficient for TAD.

## Required gate evidence

- exact source-pixel equality and reversible source-coordinate box;
- a required expected Git commit, a completely clean worktree, and byte-equal
  `HEAD` blobs for every audited executable/configuration source;
- no local interpolation, no local padding on the development population;
- uint8 structured inputs from one real decoded 768-frame window;
- exact fit/gate population and manifest/hash closure;
- record-derived geometry summary over exactly the frozen 160/40 identities;
- an in-gate re-probe of every census source, matching root containment, path,
  file size, width, height, rotation, frame count, and frame rate;
- exact pretrained VideoMAE core tensor loading: 163 checkpoint state tensors,
  161 backbone-core tensors, and 22,482,048 core parameters;
- exact 12-block VideoMAE-S architecture and normalized model/NMS parity with
  the reference AdaTAD-derived configuration;
- shared backbone instance with finite backward;
- nonzero detector-loss gradients at both global and local branch features;
- nonzero backbone-adapter, projection, and head gradients;
- exact `6x6`/`8x8` position interpolation and `[B,384,768]` projection input;
- an explicit opened-file inventory, zero official-test annotation records,
  and zero official-test video files;
- a full-stack cost schema that separates decode, crop, H2D, both backbone
  passes, fusion, detector, and NMS while keeping all paper cost claims off.

## Cost boundary

`global96 + local128` processes 25,600 view pixels per frame. At patch size 16,
the two branches contain 36 and 64 spatial tokens per tubelet. Shared weights
do not mean shared computation, and two separate attention calls do not have
the same cost as one dense view with the same total token count. Pixel/token
counts are budget descriptors only; latency, memory, and energy require a
trained-checkpoint full-stack profile on matched hardware.

## Next decision

Only after this vertical slice passes a clean commit-bound CUDA gate and an
independent P0/P1 audit may the project freeze a development-only crop
sufficiency experiment. That protocol still needs a separate decision on
candidate coverage, teacher split/cache, matched training distribution,
baselines, uncertainty, and GO/KILL margins. Learned ROI policy remains
forbidden until sufficiency is supported.
