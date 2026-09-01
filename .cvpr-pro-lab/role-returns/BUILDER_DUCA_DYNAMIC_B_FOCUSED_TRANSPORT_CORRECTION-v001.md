# Focused Builder correction — DUCA Dynamic B

- **Parent:** `DYNAMIC_ROUTE_B_STATIC_BLOCKED / IMPLEMENTATION_CORRECTION`
- **Clean parent:** `9eb328f99c10c04240770d282aad2097384a6eb8`
- **Clean correction commit:** `3e551595f9ca151fa2625181f19b8447feec15bc`
- **Evidence class:** static-only infrastructure evidence.

## Exact correction

1. Removed the dynamic-B prefix fallback that could violate the bounded local
   exact-K contract.
2. Made `local_radius=0` a real constraint: impossible `K>1` selections fail
   closed rather than disabling locality.
3. Restricted dynamic-B F2 metadata to `enabled=False` and
   `status=not_in_execution_path`; the claim arm does not assert execution of
   the K-shuffle control.
4. Added targeted static regression tests for the zero-radius and impossible
   locality branches, and for the corrected F2 metadata.

## Checks

```text
python -m pytest tests/test_duca_dynamic_physical_contract.py tests/test_duca_dynamic_b_static_packet.py -q
9 passed
python -m py_compile <both modified selector files>
passed
```

No data, held-out access, GPU, Slurm, remote operation, training, inference,
evaluation, metrics, cost measurement, or efficacy claim occurred.

- **next_owner:** same Critic
- **next_action:** one focused read-only recheck of `3e551595...`
- **dependency:** clean Critic binding at that exact commit
- **single_recovery:** exhausted after this recheck; a second equivalent defect
  terminates the correction loop.
