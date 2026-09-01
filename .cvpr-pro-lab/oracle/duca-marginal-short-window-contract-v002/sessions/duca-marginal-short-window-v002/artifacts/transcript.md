# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a947e5a-0dcc-83ea-8d8d-abc55f1cb5e9

## Prompt

Read the attached authoritative prompt completely and answer it as one independent scientific adjudication. Preserve nonce DUCA-MARGINAL-SHORT-WINDOW-CONTRACT-v002-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_SHORT_WINDOW_CONTRACT-v002.md
Lines: 1-72
```md
 1 | # DUCA-Marginal-v1 short-window contract and implementation adjudication
 2 | 
 3 | **Nonce:** `DUCA-MARGINAL-SHORT-WINDOW-CONTRACT-v002-20260831`
 4 | 
 5 | You remain the independent scientific head of DUCA. Codex is implementing the already frozen `DUCA-Marginal-v1` counterfactual probe and has not launched any GPU job. Issue one direct scientific decision: **CONTINUE**, **REVISE**, or **STOP**. This turn is only to remove a real contradiction in the short-window budget contract and review the synchronized candidate implementation against that decision. Do not choose a different research route, add a new model component, start a hyperparameter search, or authorize detector training.
 6 | 
 7 | ## Authoritative synchronized code
 8 | 
 9 | - Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
10 | - Candidate branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-budget-v1-20260830
11 | - Exact candidate commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e45dda787a6880da4cbde0b6436ffd2a2b9df218
12 | - Runner: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/tools/bata/run_duca_marginal_frozen_h65_probe.py
13 | - Budget/head implementation: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/opentad/models/duca/dynamic_budget.py
14 | - Nested acquisition: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/opentad/models/duca/acquisition.py
15 | - Counterfactual targets: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/opentad/models/duca/counterfactual_utility.py
16 | - H65 selector integration: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/opentad/models/selectors/duca_online_frame_selector.py
17 | - Config: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py
18 | - Focused tests: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e45dda787a6880da4cbde0b6436ffd2a2b9df218/tests/test_duca_marginal_budget.py
19 | 
20 | This commit is an explicitly reviewable candidate, not accepted evidence. Static compilation and `git diff --check` pass. The local Windows Torch suite is skipped because of the documented `c10.dll` environment problem; Linux/N16R4 integration tests have not run. The current runner deliberately fails closed when `valid_count < 512`; this is the known incorrect placeholder that this adjudication must replace. No model training, inference result, mAP, or cost claim exists from this commit.
21 | 
22 | ## Frozen experiment that must otherwise remain unchanged
23 | 
24 | - Clean base: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
25 | - Frozen H65 terminal checkpoint: `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth`
26 | - SHA-256: `dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c`
27 | - State: `state_dict_ema`, epoch 59
28 | - Main tiers: requested `K in {256,384,512}`; one packet is 16 non-contiguous H65-ranked observations.
29 | - A: Fixed-H65-384; B: train-side Oracle-Reallocate-384; C: Learned-Reallocate-384.
30 | - 160/40 video-level training-side split, seed 3407. No official-test access in this probe.
31 | - H65 Scout, detector, loss, decoding, NMS, annotation, class map and evaluator remain frozen.
32 | - K384 must reproduce the sealed H65 behavior; utility labels are detached signed losses `L256-L384` and `L384-L512`.
33 | - No padding every arm to 512; different heavy budgets must cause genuinely different heavy input lengths.
34 | - Secondary mean-K320 arms remain unresolved and are not part of this question.
35 | 
36 | ## Newly verified real-data fact
37 | 
38 | Codex inspected the authoritative THUMOS14 annotation at `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json` using the frozen `feature_stride=4` sliding-window interpretation. The detector-training side contains 200 videos:
39 | 
40 | - 43 videos have fewer than 512 valid dense observations;
41 | - 62 videos have fewer than 768 valid dense observations;
42 | - the minimum is 67 valid observations.
43 | 
44 | The sliding-window dataset shifts the final window to full length only when the video itself is long enough. Therefore short windows are common and reachable, not an exotic corner case.
45 | 
46 | ## Contradiction that blocks faithful execution
47 | 
48 | Your previous frozen report simultaneously required:
49 | 
50 | 1. per-video total budget `384 * actual_window_count` and the focused assertion `sum(K_dynamic) == 384 * num_windows`;
51 | 2. short windows billed by actual valid length, with unused budget reallocated and not counted as padding;
52 | 3. 16-observation packets and real variable heavy tensor shapes;
53 | 4. K384 prediction parity with the existing H65 path for every window.
54 | 
55 | For a one-window video with only 67 valid observations, these cannot all be true at once. Its maximum unique actual cost is 67 (or 64 if whole 16-observation packets are mandatory), so it cannot meet an exact cost target of 384 within the same video. The existing H65 K384 path pads invalid slots and executes the nominal K384 tensor; shortening that heavy tensor changes the execution and may break the required prediction parity.
56 | 
57 | ## Required authoritative output
58 | 
59 | Freeze one coherent, executable rule set. State exact formulas and behavior, not general advice:
60 | 
61 | 1. Define `actual_cost(V,K)` for valid length `V` and requested tier `K`, including whether it is `min(V,K)`, rounded down/up to a 16-observation packet, or another precise rule. State how a partial final packet is executed and billed.
62 | 2. Define the exact per-video target budget. Choose whether it is nominal `384*N`, baseline actual cost `sum actual_cost(V_i,384)`, or another explicit quantity.
63 | 3. State the redistribution domain and fallback when a video's target is infeasible: within-video only, across-video, exclusion, fixed baseline, or another precise action. Preserve train/holdout isolation and no test leakage.
64 | 4. State exactly what K384 parity must mean for short windows: identical requested indices, identical valid prefix, identical predictions under the historical padded path, or a separately named admissible comparison.
65 | 5. State how to define `L256`, `L384`, and `L512` when two or more requested tiers collapse to the same valid prefix, and whether such windows train the utility head.
66 | 6. State how Oracle and Learned allocation metrics count the fraction assigned to K256/K512 when actual costs collapse.
67 | 7. Confirm whether the 160/40 split still includes all 200 training-side videos.
68 | 8. Review exact commit `e45dda787a6880da4cbde0b6436ffd2a2b9df218` and identify only code changes that are necessary to implement your frozen answer or correct a scientific/implementation error. Do not request generalized workflow machinery, provenance scaffolding, style refactors, or unrelated tests.
69 | 9. Give the minimal focused tests and the exact PRE_RUN success condition. Do not modify VideoMAE, ActionFormer/AdaTAD head, loss, NMS, split, annotation, class map, official evaluator, Scout or checkpoint.
70 | 10. Return one next action with owner and an absolute Beijing-time deadline. If the contract cannot be made coherent without changing the scientific question, say **STOP**.
71 | 
72 | The answer must be self-contained and must not rely on prior chat context. Treat this prompt and the linked exact GitHub commit as the complete material. Do not authorize GPU execution unless the short-window contract is fully frozen and the candidate's required corrections are explicit.
```

## Answer

# Independent scientific adjudication

**Nonce:** `DUCA-MARGINAL-SHORT-WINDOW-CONTRACT-v002-20260831`

## 1. Unique decision

# **REVISE**

The scientific question remains coherent and worth executing: test whether frozen H65 marginal evidence can reallocate **real, unique observations** within each training-side video while holding that video’s baseline actual-observation budget fixed.

The experiment must not stop or change route. However, commit `e45dda787a6880da4cbde0b6436ffd2a2b9df218` is **not PRE_RUN-admissible** because it still encodes the contradictory nominal target, rejects every window with fewer than 512 valid observations, requires fully valid exact-tier tensors, and cannot represent collapsed budget tiers. The correction is bounded and does not alter the Scout, detector, losses, data split, evaluator, checkpoint, or three requested tiers.

---

## 2. Frozen short-window budget contract

Let a video contain windows \(i=1,\ldots,N\). Let:

* \(V_i\) be the number of valid dense observations in window \(i\);
* \(K\in\{256,384,512\}\) be a requested tier;
* the packet size be \(P=16\).

### 2.1 Actual observation cost

The authoritative budget-accounting quantity is

$$
\boxed{c_i(K)=\min(V_i,K)}
$$

It counts unique valid observations made available to the heavy path. Invalid padding is never counted.

This deliberately distinguishes:

* **actual observation cost** \(c_i(K)\);
* **executed heavy slots** \(e_i(K)\);
* **padding slots** \(e_i(K)-c_i(K)\).

The probe may make claims about equal actual-observation budgets. It may **not** use this result alone to claim equal FLOPs, equal latency, or an end-to-end efficiency gain.

### 2.2 Collapsed requested tiers

A requested tier collapses to the baseline whenever it exposes the same valid prefix as K384:

$$
\boxed{\kappa_i(K)=
\begin{cases}
384,&c_i(K)=c_i(384),\\
K,&c_i(K)\neq c_i(384).
\end{cases}}
$$

Therefore:

| Valid length \(V_i\) | Distinct effective arms                |
| -------------------: | -------------------------------------- |
|        \(V_i\le256\) | K384 only; K256 and K512 both collapse |
|    \(256<V_i\le384\) | K256 and K384; K512 collapses          |
|      \(384<V_i<512\) | K256, K384, K512 all distinct          |
|        \(V_i\ge512\) | K256, K384, K512 all distinct          |

A collapsed K256 or K512 request is not separately forwarded through the detector. It aliases the K384 loss, prediction, valid positions, and execution record. It is canonicalized to effective tier K384 before allocation statistics are computed.

### 2.3 Heavy execution and partial packets

For the historical K384 baseline:

$$
\boxed{e_i(384)=384}
$$

for every window. This is the sole historical-parity exception. It preserves the existing K384 tensor, invalid-slot padding, masks, remapping, and detector execution.

For a **distinct nonbaseline** arm:

$$
\boxed{e_i(K)=16\left\lceil\frac{c_i(K)}{16}\right\rceil}
$$

Thus:

* every distinct K256 execution has 256 slots;
* a distinct K512 execution can have 400, 416, 432, 448, 464, 480, 496, or 512 slots, depending on \(V_i\);
* it is not padded through unused complete packets to 512.

For a partial final packet, the first \(c_i(K)\) slots contain the ordered valid H65-prefix observations. The remaining

$$
e_i(K)-c_i(K)\in\{0,\ldots,15\}
$$

slots are represented by `-1` positions, gathered as zeros, and marked invalid in the acquisition mask. They are executed only because VideoMAE requires complete 16-observation packets and are not billed as actual observations.

No distinct nonbaseline arm may contain an entirely empty packet after its final partially valid packet.

### 2.4 Exact per-video target

For each video independently, the target is the actual cost of its fixed K384 baseline:

$$
\boxed{T_v=\sum_{i=1}^{N}c_i(384)
      =\sum_{i=1}^{N}\min(V_i,384)}
$$

The previous assertion

$$
\sum_i c_i(K_i)=384N
$$

is deleted.

For a one-window video with \(V=67\):

$$
T_v=67.
$$

All requested tiers collapse to the historical K384 arm, so that video remains K384 and contributes no marginal-allocation choice.

### 2.5 Redistribution domain and fallback

Redistribution is **within one video only**.

For video \(v\), choose effective tiers \(a_i\) satisfying:

$$
\sum_i c_i(a_i)=T_v,
$$

and

$$
\sum_i \mathbf 1[a_i\neq384]\le
\left\lfloor0.5N\right\rfloor.
$$

There is:

* no cross-video transfer;
* no transfer between the 160-video fit partition and 40-video holdout partition;
* no video exclusion;
* no official-test access.

The all-K384 assignment is always feasible by construction. Therefore, “target infeasible” must never be an ordinary scientific outcome.

If no positive-utility nonbaseline assignment satisfies the exact target and changed-window bound, the allocation is the fixed all-K384 baseline. If the implementation cannot recover the baseline as an exact feasible solution, that is a PRE_RUN implementation failure, not an allocation result.

The deterministic optimization order is:

1. maximize predicted or oracle total utility;
2. prefer fewer effectively changed windows;
3. break remaining ties lexicographically by sorted window identity with tier preference `384`, then `256`, then `512`.

### 2.6 K384 parity on short windows

For every window, including \(V<384\), K384 parity means all of the following:

1. the complete 384-slot requested-index tensor, including trailing `-1` slots, is identical to the sealed H65 selector output;
2. its valid prefix is identical;
3. its K384 loss is obtained from the unchanged historical padded K384 training-loss path;
4. its prediction is obtained from the unchanged historical padded K384 inference path;
5. the serialized prediction is exactly equal under the same model, AMP, decoding, and post-processing settings.

A shortened K384 tensor is forbidden.

For \(V\ge384\), the explicit counterfactual reconstruction must additionally reproduce the historical K384 prediction exactly. For \(V<384\), the runner must use the historical path directly rather than attempting an all-valid explicit reconstruction.

This preserves the frozen H65 estimand while treating the extra invalid slots as unbilled padding.

---

## 3. Utility losses when tiers collapse

Let \(L_{256,i}\), \(L_{384,i}\), and \(L_{512,i}\) denote the frozen detector objective.

Define eligibility masks:

$$
m^-_i=\mathbf 1[c_i(256)<c_i(384)]
$$

and

$$
m^+_i=\mathbf 1[c_i(384)<c_i(512)].
$$

Then:

$$
L_{256,i}=
\begin{cases}
\text{distinct K256 detector loss},&m^-_i=1,\\
L_{384,i},&m^-_i=0,
\end{cases}
$$

$$
L_{512,i}=
\begin{cases}
\text{distinct upper-tier detector loss},&m^+_i=1,\\
L_{384,i},&m^+_i=0.
\end{cases}
$$

The detached targets remain:

$$
y^-_i=L_{256,i}-L_{384,i}
$$

and

$$
y^+_i=L_{384,i}-L_{512,i}.
$$

Collapsed targets are stored as exactly zero, but they are masked out of utility-head training and predictability statistics.

The utility-head loss is:

$$
\boxed{
\mathcal L=
\frac{
\sum_i m^-_i(\hat y^-_i-y^-_i)^2+
\sum_i m^+_i(\hat y^+_i-y^+_i)^2
}{
\sum_i m^-_i+\sum_i m^+_i
}}
$$

A batch with zero eligible target dimensions performs no optimizer update.

Consequently:

* a window with \(V\le256\) does not train either output;
* a window with \(256<V\le384\) trains only downgrade penalty;
* a window with \(V>384\) trains both outputs.

Spearman correlation and sign accuracy are computed separately on eligible holdout margins only. Trivial collapsed zeros must not inflate either statistic.

The oracle utility used by the allocator is:

$$
u_i(256)=L_{384,i}-L_{256,i},
\qquad
u_i(384)=0,
\qquad
u_i(512)=L_{384,i}-L_{512,i}.
$$

---

## 4. Oracle and Learned allocation fractions

Every allocation is canonicalized to its **effective tier** before metrics are computed.

$$
f_{256}=
\frac{\#\{i:a_i=256\}}
{\text{total holdout windows}}
$$

$$
f_{512}=
\frac{\#\{i:a_i=512\}}
{\text{total holdout windows}}
$$

A requested K256 or K512 that collapses to the K384 prefix:

* is stored as effective K384;
* is not counted as changed;
* is not included in the K256 or K512 numerator;
* remains in the denominator.

The existing minimum 0.10 gates apply to these overall effective-tier fractions. The result may additionally report fractions among eligible windows, but those are diagnostics and do not replace the frozen gates.

For every video, report separately:

* target actual cost;
* achieved actual cost;
* actual-budget error;
* total executed heavy slots;
* total padding slots;
* number of effective K256/K384/K512 windows.

Only actual-budget error is required to equal zero. Executed-slot totals are diagnostic and cannot be converted into an efficiency claim by this probe.

---

## 5. The 160/40 split

The 160/40 video-level split remains unchanged and includes all 200 detector-training-side videos.

Short videos and their windows are not excluded. A window for which both margins collapse remains part of:

* the fixed K384 arm;
* video-level allocation accounting;
* fixed/oracle/learned official training-holdout evaluation.

It simply contributes no utility-head regression target.

---

## 6. Review of exact commit `e45dda787a6880da4cbde0b6436ffd2a2b9df218`

The candidate is structurally close to the required probe, but the following corrections are scientifically necessary.

### 6.1 Configuration

The configuration still freezes `target_actual_observations="384 * actual_window_count"` and declares `workflow.training_enabled=False`, even though the runner trains the utility head for 20 epochs.

Required correction:

* replace the target with `sum(min(valid_observations_i, 384))`;
* declare detector and Scout training disabled separately;
* declare utility-head fitting enabled;
* replace `one_real_heavy_shape_per_stage=True` with grouping by realized packetized execution length;
* record the K384 historical-padding exception and collapsed-tier alias rule.

### 6.2 `dynamic_budget.py`

`allocate_equal_budget_marginal_reallocation()` currently passes `384 * window_count` as the target. The allocator also treats requested-tier changes as changes even when actual costs collapse, and can return an infeasible result instead of the guaranteed baseline.

Required correction:

* derive the target from `min(valid_observations, 384).sum()`;
* remove duplicate options whose actual cost equals the baseline cost;
* define `changed_mask` from effective-prefix change, not `requested_budget != 384`;
* retain the all-K384 state unconditionally;
* return baseline when no positive exact transfer exists;
* fail only if the baseline itself does not satisfy the target;
* validate and report both actual cost and execution slots.

`validate_real_heavy_observation_tensor()` must accept any packet-aligned execution length up to 512, verify the active count, and enforce less than one packet of padding for distinct nonbaseline arms. Its existing nominal 256/384/512-only check is insufficient for a K512 request with, for example, \(V=401\).

### 6.3 `acquisition.py`

The nested-prefix implementation already has two correct foundations:

* K384 is checked against the complete sealed baseline tensor;
* actual counts are computed as `min(requested_budget, valid_count)`.

Keep this selection logic. Add, without introducing a new selector:

* per-tier collapse-to-K384 flags;
* effective-tier identities;
* execution-slot counts;
* active-prefix views separate from trailing `-1` padding;
* validation that equal actual counts imply equal active prefixes.

The physical-time interpolation must continue to use the active positions and active count, not padded positions.

### 6.4 `duca_online_frame_selector.py`

`forward_marginal_prefixes()` currently gathers tensors at the nominal requested width and emits all-one 384 detector masks, even when the acquisition positions contain trailing invalid entries.

Required correction:

* preserve and expose the full historical 384-slot tensor for parity;
* separately expose active positions, actual count, execution slots, and acquisition mask;
* do not describe nominally padded K512 inputs as 512 actual observations;
* do not separately construct collapsed K256/K512 detector inputs;
* leave the normal detector path unchanged.

### 6.5 `counterfactual_utility.py`

`detached_three_budget_prefix_utilities()` currently requires every requested tensor to be fully valid and rejects every `-1` position. It therefore cannot represent short windows or collapsed tiers.

Required correction:

* accept actual counts and trailing `-1` padding;
* verify nesting on active prefixes only;
* evaluate each distinct effective prefix once;
* alias collapsed losses to \(L_{384}\);
* return `downgrade_target_valid` and `upgrade_target_valid`;
* keep every target detached and keep detector gradients disabled.

### 6.6 Runner

The current selection stage deliberately raises whenever `valid_count < 512`; this is the known placeholder and must be removed.

The current explicit preparation path also requires `len(positions) == requested_budget`, forbids negative padded indices, and describes every explicit arm as fully valid exact-K.

The counterfactual artifact then records `actual_heavy_observations=budget`, which is false for short upper-tier windows.

Required runner corrections:

1. During selection, obtain K384 loss and predictions from the unchanged historical selector path for every window.
2. Save the complete K384 tensor and each tier’s active prefix, actual cost, effective tier, execution slots, and collapse flags.
3. Skip a K256 forward when \(V\le256\); alias it to K384.
4. Skip a K512 forward when \(V\le384\); alias it to K384.
5. Run distinct K256 examples at execution length 256.
6. Group distinct K512 examples by execution length \(16\lceil \min(V,512)/16\rceil\) and load/reuse the corresponding packet-shape model.
7. Permit only trailing invalid padding and verify that all nonbaseline padding is confined to the final packet.
8. Record actual cost and executed slots separately.
9. Train the utility head with the masked loss above.
10. Canonicalize collapsed choices before fractions and changed-window counts.
11. Replace a no-transfer `RuntimeError` with the fixed K384 fallback.
12. Rename result fields so that the fixed arm remains `Fixed-H65-384`, while actual-cost and executed-slot totals are separately explicit.

### 6.7 Tests

The current test suite explicitly expects a video with four \(V=300\) windows to be infeasible under the nominal 1536 target. That expectation freezes the contradiction and must be removed.

The existing nominal-shape test also covers only exact 256/384/512 tensors and does not test partial final packets.

No generalized framework, new provenance system, style refactor, or unrelated test expansion is authorized.

---

## 7. Minimal focused tests

The corrected focused suite must contain these exact cases.

### Cost, collapse, and execution

| \(V\) | Costs \((c_{256},c_{384},c_{512})\) | Effective tiers | Executed slots    |
| ----: | ----------------------------------- | --------------- | ----------------- |
|    67 | `(67,67,67)`                        | `(384,384,384)` | baseline 384 only |
|   300 | `(256,300,300)`                     | `(256,384,384)` | `(256,384,alias)` |
|   401 | `(256,384,401)`                     | `(256,384,512)` | `(256,384,416)`   |
|   600 | `(256,384,512)`                     | `(256,384,512)` | `(256,384,512)`   |

For \(V=401\), the K512 input must contain 401 active observations and 15 trailing invalid slots.

### Labels

* \(V=67\): both targets zero; both training masks false.
* \(V=300\): downgrade target active; upgrade target zero and masked.
* \(V=401\): both targets active.
* No detector graph may attach to any target.

### Allocation

A four-window video with valid lengths `[300, 428, 512, 512]` must use target

$$
300+384+384+384=1452.
$$

An exact positive transfer may assign the first window K256 and the second K512:

$$
256+428+384+384=1452.
$$

The test must also verify:

* collapsed requests canonicalize to K384;
* no-positive-transfer returns all K384;
* the baseline is never reported infeasible;
* changed-window counting uses effective tiers.

### K384 parity

* Full 384-slot position tensors match the historical selector, including `-1` padding.
* For a full window, the explicit K384 path reproduces the historical prediction exactly.
* For a short window, K384 uses the historical padded path directly and is not shortened.

### Split and metric accounting

* fit and holdout sets are disjoint;
* their union contains exactly 200 videos;
* short videos remain included;
* collapsed tiers do not enter K256/K512 numerators;
* actual per-video budget error is zero.

---

## 8. Exact PRE_RUN success condition

Add a bounded `pre-run` mode to the existing runner; do not create another launcher or validation framework.

`PRE_RUN_PASS` is emitted only when all of the following hold on one clean successor commit:

1. `git diff --check`, Python compilation, and `tests/test_duca_marginal_budget.py` pass on Linux with Torch.
2. The frozen checkpoint loads strictly from `state_dict_ema`, reports epoch 59, and matches the frozen SHA-256.
3. The training-side split is exactly 160/40, disjoint, and its union is all 200 videos.
4. A metadata pass over every training-side window completes without exclusion or `V<512` failure.
5. Every window satisfies the cost, collapse, nesting, and execution formulas above.
6. The full K384 position tensor equals the normal H65 selector tensor for every scanned window.
7. One deterministic real-data example from every observed short-window execution class completes a frozen forward.
8. Every distinct nonbaseline tensor has:

   * temporal size equal to its execution slots;
   * exactly its actual-cost number of active positions;
   * only trailing zero/invalid padding;
   * fewer than 16 padded slots.
9. Every collapsed arm aliases K384 loss and prediction without a second detector forward.
10. For every video, the all-K384 assignment has achieved actual cost exactly equal to \(T_v\).
11. The Scout and detector have no trainable parameters and no gradients.
12. No utility-head fitting, official evaluator call, official-test access, mAP result, or detector training occurs in PRE_RUN.

Any failed condition returns `PRE_RUN_FAIL` with the first violated scientific invariant. It must not silently exclude the affected video or fall back to the old nominal target.

After `PRE_RUN_PASS`, the same exact clean commit is authorized to run only the frozen `select-k384`, `counterfactual-k256`, `counterfactual-k512`, and `summarize` probe stages. No further scientific review is required before that probe. Detector training remains forbidden.

---

## 9. One next action

**Owner:** Codex, with Builder responsible for the bounded patch, one independent Critic review of the exact successor commit, and Evaluator responsible for PRE_RUN and—only after `PRE_RUN_PASS`—submission of the frozen probe.

**Action:** Patch commit `e45dda787a6880da4cbde0b6436ffd2a2b9df218` in place into one clean successor implementing this adjudication; run the focused Linux tests and exact PRE_RUN; on PASS, launch only the frozen counterfactual probe on that same commit.

**Deadline:** **2026-09-01 18:00 Beijing time (UTC+08:00).**

No VideoMAE parameters, ActionFormer/AdaTAD head, detector loss, NMS, Scout, data split, annotation, class map, evaluator, checkpoint, seed, requested tier, or official-test policy may change.
