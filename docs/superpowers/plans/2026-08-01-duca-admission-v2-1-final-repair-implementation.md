# DUCA Admission v2.1 Final Narrow Repair: Adjudication and Implementation Record

Date: `2026-08-01`

Decision ID: `U-PRO-V21-FINAL-REPAIR-1`

Status: `CORE_ACCEPTED / PROTOCOL_IMPLEMENTED_AND_FOCUSED_TESTED / PRODUCTION_NO_GO`

Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`

Branch: `codex/duca-rime-20260727`

Implementation base commit: `26b0ea4161ab3f92226570e0fa01cb4caba6a06c`

Implementation base tree: `82c85b5336f22ea0a574c6f7fd15c9f5856f5d94`

Source:
`C:/Users/skywalker/.codex/attachments/a09b8a5b-c1ae-462a-94e3-c2681c29ad86/pasted-text.txt`

Source SHA-256:
`9b62a23d29d1cd74063f34127a64ba7a100805a3a60456ec9212163a8398da04`

Source size: `68,504 bytes`

## 1. Final adjudication

The core route and the bounded Stage A--D implementation decision are accepted.
No further Pro discussion is required before implementing this protocol layer.
Three implementation clarifications are frozen because they are necessary for
binary and statistical correctness:

1. exact zero means the IEEE-754 binary64 positive-zero bit pattern;
2. the type-1 median of 64 values is order statistic 32, i.e. zero-based index
   31, and is not the average of the two central values;
3. every scale, critical value, bound, normalizer and Monte Carlo half-width
   used outside the exact-zero branch must be finite, with all required scales
   and normalizers strictly positive.

The accepted state is exactly:

```text
core_direction = ACCEPT
protocol_implementation_only = GO
production_admission_v2_1 = NO_GO
real_video_workers = UNAUTHORIZED
scale_fit_execution = UNAUTHORIZED
calibration_execution = UNAUTHORIZED
holdout_release = UNAUTHORIZED
phase1_through_phase4 = UNAUTHORIZED
learned_hrime = UNAUTHORIZED
full_200_refit = UNAUTHORIZED
official_final = SEALED
paper_admissible_empirical_result = NONE
```

This work does not modify a detector, backbone, loss, budget grid, selector,
checkpoint, official evaluator or official metric. It implements the scientific
admission protocol that must be passed before those later actions can be
authorized.

## 2. Implemented protocol surface

### Stage A — deterministic identities and incidence

- strict NFC/NUL/typed canonicalization and length-prefixed domain hashes;
- unbiased hash-based categorical draws;
- exact `70 long + 30 short` source inventory;
- four deterministic long reserves and 32 allocation-only triplets;
- three disjoint roles, each `22 full + 10 short` videos;
- canonical video ranks independent of input JSON row order;
- role-specific `32 x 8` connected incidence with video degree two and process
  degree eight;
- exactly 192 content-bound planned cells.

### Stage B — statistics and Monte Carlo contract

- 12-metric closed registry and count-aware top-five-percent cell summaries;
- independent calibration-versus-holdout role contrast with fixed coefficient
  `1/64`;
- positive two-point product factors `{0.5, 3.0}` with probabilities
  `{4/5, 1/5}` and `kappa = 1`;
- single-step fixed-scale standardized maxT with separate positive and negative
  critical values;
- exact positive-zero branch, degenerate-scale failure and finite-value guards;
- deterministic 100k-to-200k prefix extension;
- real delete-one-1000-batch jackknife and numeric MC certificate;
- secondary stream retained as diagnostic only.

### Stage C — candidate-free simulation specification

- exact 52-scenario registry, including the four boundary scenarios;
- exact 500-outer requirement per scenario and all five registered shift
  profiles where applicable;
- simultaneous upper/lower coverage, false-alarm, non-vacuity/width and power
  gates;
- exact-zero gate;
- 24-scenario, 200-independent-stream MC half-width calibration schema;
- pooled and per-scenario gate validators with closed failure codes;
- exact Python/NumPy/Philox reference-environment golden hash.

The registry, generators, single-outer executor, aggregators and validators are
implemented. The complete `52 x 500` execution and the `24 x 200` independent
MC calibration have not been run. Synthetic unit fixtures validate code paths;
they are not empirical calibration evidence.

### Stage D — runtime and publication contract

- exact 37-row closed control registry and 19 failure codes;
- Slurm job/step, PID start time, boot ID, cgroup and CUDA/NVML worker identity
  surfaces;
- planned allocation comparison and optional mandatory live re-attestation;
- closed, content-hashed control evidence and receipt bindings;
- a `PASSED` receipt requires independent evidence verifiers for every
  repository-enforced and cluster-attested control;
- caller-independent expected schema/stage checks for parent receipts;
- explicit permanent rejection of superseded `duca_acquisition_admission_v2`;
- allowlisted mode-0700 fresh roots, component-wise no-follow traversal,
  descriptor-bound reads, same-filesystem atomic self-test and exclusive
  hard-link publication;
- protocol/simulation/runtime-preflight receipts hard-code
  `authorization_scope = NONE` and keep official-final sealed.

## 3. Verification status

The focused cross-platform suite passes. POSIX-only filesystem tests are skipped
on Windows by design and therefore do not count as Linux runtime evidence. A
separate Ubuntu POSIX smoke check has exercised fresh-root creation,
same-filesystem hard-link self-test, exclusive publication, parent receipt
re-read/hash validation and symlink rejection. The authoritative exact-clean
Linux/PyTorch/Slurm code gate and real cluster identity receipts still do not
exist. The final local focused result is `49 passed, 2 skipped`; the repository's
required C3 regression result is `23 passed`.

Frozen registry identities:

```text
simulation registry artifact SHA-256
  dfd1c31895fc892dee60deff13a050f3168810ea4cfe810fba801d681dd7202d
simulation registry semantic SHA-256
  b06778e97130b1af6a31ba37f0a2bb081fa86321f4984c4a8185da71508dc4fa
control registry artifact SHA-256
  25ddc8ff60b4fc1079eb9b1382913dff454eafa5335cb5cb7946021db5d961aa
control registry semantic SHA-256
  184df0eb0c6ac9e9d626d27b341ed9785c8de4597dc36204ba07cfbac91f1126
```

Evidence level:

```text
design = accepted
implementation = implemented
focused_tests = tested
full_simulation = not_run
mc_repeated_stream_calibration = not_run
authoritative_slurm_runtime_preflight = not_run
empirically_supported = false
paper_ready = false
```

## 4. Remaining blockers and next execution order

1. Freeze and run the exact-clean Linux code gate without candidate outputs.
2. Complete the candidate-free `52 x 500` simulation execution and the
   independent `24 x 200` MC calibration in the frozen reference runtime.
3. Produce content-bound simulation and runtime-preflight terminal receipts.
4. Independently audit all receipts, exact checkout/tree identity and Linux
   filesystem/Slurm/cgroup/GPU evidence.
5. Stop for a new explicit authorization decision. Do not start real-video
   workers, scale-fit, calibration or holdout automatically.

The full simulation is deliberately not represented as complete merely because
the registry and evaluators exist. Its current absence keeps steps 3 and 4
open. Step 1 remains open until authoritative Linux/Slurm receipts exist. Step 5
requires a future explicit authorization even after all technical gates pass.

Final evidence statement:

```text
No paper-admissible empirical conclusion is available yet.
```
