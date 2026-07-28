# Recovery-v6 Runtime Temporal Adapter Design

## Status

`designed / implemented / static_checked / independently_reviewed_GO /
remote_gate_pending`

## Failure signature

Recovery-v5 Phase-1 job `1201170` failed during the exact-uniform K192
short-window evaluation with:

`RuntimeError: shape '[-1, 192, 10, 10, 96]' is invalid for input of size 1075200`.

The dynamic temporal backbone correctly converted the effective K into
16-frame execution chunks. Each VideoMAE call therefore contained eight
temporal tokens after tubelet embedding, while `Adapter.forward` reshaped those
tokens using the nominal configuration-time `temporal_size=192`.

Unique failure signature:

`vit_adapter_static_temporal_axis_on_dynamic_k_bucket`

## Bounded engineering repair

`Adapter.forward` derives the runtime temporal token count from
`N / (h * w)`, rejects non-integral token geometry, and uses that local value
for the temporal convolution reshape. It does not mutate the configured
`self.temporal_size`.

This changes no model component, weight, objective, loss, selector, budget,
threshold, hyperparameter, split, checkpoint, evaluator, or paper claim. It
only makes the existing adapter execute the already-frozen dynamic-bucket
contract.

## Regression contract

The focused runtime test constructs an adapter with nominal
`temporal_size=192`, supplies an eight-token runtime temporal axis, and requires:

1. output shape exactly equals input shape;
2. every output value is finite;
3. configured `temporal_size` remains unchanged;
4. non-integral `(N, h, w)` geometry fails closed.

The DUCA-RIME Slurm code gate compiles `vit_adapter.py` and runs this test
through the existing focused test module.

## Deployment boundary

A new clean commit, authoritative Slurm code gate, commit-bound physical
protocol, salvage manifest, submission manifest, released receipt, and fresh
transaction root are required. Recovery-v5 remains immutable failed
engineering evidence. Phase 4 remains disabled and official-final remains
sealed.
