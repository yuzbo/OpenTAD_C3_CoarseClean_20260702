# Experiment-audit reviewer responses

## Broad reachability verdict

`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`

- Formal Gate-4 GT/cost/result provenance fails because the evidence producer and official-population
  workflow do not exist; current Gate-4 tests are synthetic/test-only.
- Existing official AP use and raw metric handling do not show self-normalization, but Stage-C cost
  evidence lacked immutable profile binding.
- No formal result, checkpoint chain, Gate report or paper number exists.
- Formal Stage-C/matched-dense/Gate-4 runners, validator, launchers and runner tests are absent.
- Tests cover toy/synthetic one- or two-update behavior, not 4,200 successful updates or official video
  scope.
- A1/A2/A3/A4 remain protocol blockers. No route-killing scientific result exists because Gate 1 has
  not run.

## Gate-4 bounded review

`APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`

The reviewer matched exact hashes
`A581D71338B130C2FF0ECB2B833B29F1B7B1FD5A8F5C36E7A24BC7B954B1A75F` and
`5C0FFAF398EC45958045C46CE714BE391E987197532F60409D840F6AAAB4506E`. It verified that a forged payload
with a recomputed hash is rejected against recomputed raw evidence. The approval does not implement a
formal evidence producer or Gate-4 result.

## Stage-C bounded review

`APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`

The reviewer matched exact hashes
`5BDC1862AD90F1D0A6134ADD778D5978A536848EEAD63EDE973444CBCA5577C4` and
`C92FED397F69E03BE6F0189483250F8132579DD844C521CE3E17BEF0B3A262D7`. It verified that
`cost_is_measured` is exact-true, the negative test mutates a real runtime summary after a normal forward,
and `_TEST_MEASURED_COST` is not wrapped as registered provenance. Remaining blockers are immutable
cost-profile registration, A3, A4, formal runners, 4,200 updates, ledgers, launchers and registration.

Neither bounded reviewer reran tests, edited files or launched a job.
