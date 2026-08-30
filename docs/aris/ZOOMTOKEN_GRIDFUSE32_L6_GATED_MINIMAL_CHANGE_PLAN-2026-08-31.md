# ZoomToken GridFuse32-L6 gated minimal-change plan

## 1. Authority and immutable identity

- Pro decision: `PIVOT`, conversation `6a94842b-1370-83ea-a13c-2cc492170597`.
- Unique task: `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`.
- Frozen execution base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`.
- Candidate branch: `codex/zoomtoken-gridfuse32-l6-v001`.
- GitHub branch URL: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>.
- Exact clean/pushed candidate: `3f1e7961720ceb7c7fa4a6276b6767a42adff94c`.
- GitHub commit URL: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/3f1e7961720ceb7c7fa4a6276b6767a42adff94c>.
- GitHub repository URL: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.

This plan does not authorize a second mechanism, capacity sweep, rescue, or any GridFuse training before the real-shape G0 gate passes.

## 2. Frozen mechanism

The native R1 input remains a continuous, hole-free `8x8/K64` support in every one of the eight tubelets of each 16-frame VideoMAE clip. Blocks 0--5 remain unchanged dense clip-local Attention, MLP, and Adapter execution.

For blocks 6--11 only, each clip-local dense carrier `X` has shape `[B, 8, 8, 8, D]`, where the axes are batch, tubelet, spatial row, spatial column, and channel. Pairing is deterministic and parameter-free:

- even block index: horizontal pairs `(r, 2c)` and `(r, 2c+1)`;
- odd block index: vertical pairs `(2r, c)` and `(2r+1, c)`.

Each pair is mean-merged, producing exactly `8 * 32 = 256` tokens per clip. The ordinary block Attention and MLP execute on the complete merged sequence, so Q, K, V, and MLP all have length 256. Let `M` be the pair mean and `F(M)` the ordinary residual block output before the Adapter. The merged residual update is `Delta = F(M) - M`. The same `Delta` is broadcast to both native members of its pair:

`Y_i = X_i + Delta_pair(i)`.

Thus pair members retain distinct current native residuals; no token is dropped, no previous-clip state is used, and no learned completion is introduced. The native 512-token carrier is restored before the existing Adapter, which continues to execute on all 512 native positions.

## 3. Exact execution ledger

For one native clip:

| Blocks | Patch carrier | Attention Q/K/V | MLP | Adapter | Attention pairs |
|---|---:|---:|---:|---:|---:|
| 0--5 | 512 | 512/512/512 | 512 | 512 | `6 * 512^2` |
| 6--11 | 512 | 256/256/256 | 256 | 512 | `6 * 256^2` |

The expected totals per clip over 12 blocks are:

- Attention query tokens: `6*512 + 6*256 = 4608`;
- KV tokens: `4608`;
- MLP tokens: `4608`;
- Adapter tokens: `12*512 = 6144` when every block has an Adapter;
- Attention pairs: `6*512^2 + 6*256^2 = 1,966,080`.

There must be no hidden dense-512 Attention or MLP execution in blocks 6--11. Checkpoint recomputation must not be counted as an additional physical execution.

## 4. Minimal allowed diff

Only the Pro-authorized surfaces may change:

- `opentad/models/backbones/vit_adapter.py`;
- `configs/adatad/thumos/georoute_official_r1_gridfuse32_l6_prebackbone_seed42_v001.py`;
- `tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py`;
- `tools/bata/profile_zoomtoken_gridfuse32_l6_fullstack.py`;
- `scripts/run_zoomtoken_gridfuse32_l6_gated_n16r4.sh`;
- `tests/test_zoomtoken_gridfuse32_l6.py`;
- `tools/train.py` only if the unchanged route allowlist or successful-update hook cannot accept the new arm without it. Optimizer, update, schedule, EMA, recovery, and resume semantics must remain unchanged.

The GeoRoute wrapper remains in its already-audited R1 `full64` mode. GridFuse is an inner `VisionTransformerAdapter` execution policy and must not change R1 support selection or physical-token identity.

## 5. Focused verification before any GPU action

Focused tests must prove:

1. exact horizontal/vertical pair maps for all eight tubelets;
2. mean merge and broadcast-delta restoration preserve distinct pair-member residual identities;
3. blocks 0--5 match the existing full64 path;
4. blocks 6--11 execute only 256-token Attention Q/K/V and MLP, followed by a 512-token Adapter;
5. the ledger totals above are exact and checkpoint replay does not double count;
6. gradients pass through the pair mean, block update, native residual, and Adapter;
7. invalid support, clip shape, ordering, depth, or configuration fails before scientific execution;
8. no route uses GT, teacher, prediction cache, cross-clip state, router, top-k, or new trainable parameters.

## 6. Frozen gates

### G0: real-shape segment gate

Use R1 epoch-59 EMA blocks 6--11 with `B=1`, eight tubelets, 64 native tokens per tubelet, dense length 512, candidate length 256, embed dimension 384, six heads, FP16. Run 100 warmups and at least 500 synchronized timed iterations per arm in an alternating order.

Pass only if all hold:

- p50 speedup `>= 1.35x`;
- peak allocated-memory ratio `candidate/dense <= 1.05`;
- peak reserved-memory ratio `candidate/dense <= 1.05`.

p95 is reported but is not a gate. Any G0 failure yields `STOP_GRIDFUSE32_L6_BEFORE_TRAINING`; no tuning or rescue follows.

### G1: conditional single training

Only after G0 passes: one seed-42, 60-epoch, two-GPU run with global/local batch `2/1`. Against matched R1 final EMA `69.07 / 61.14 / 46.57`, require Avg `>=68.57`, mAP@0.6 `>=60.64`, mAP@0.7 `>=46.07`, short-action delta `>=-0.75 pp`, and start/end boundary ratio `<=1.05`.

### G2: conditional matched full stack

Only after G1 passes: canonical 211 videos / 792 loader items, same GPU and frozen order `R1,C,C,R1,C,R1,R1,C`, four full passes per arm, covering decode through Soft-NMS. Require p50 ratio and gross-energy ratio `<=0.95`, with allocated/reserved memory ratios `<=1.05`.

Every terminal outcome, including a valid negative or protocol failure, is preserved and returned in exactly one fresh exact-Project Pro discussion before any new experiment is considered. That request must include the current GitHub repository, branch, and exact commit URLs.
