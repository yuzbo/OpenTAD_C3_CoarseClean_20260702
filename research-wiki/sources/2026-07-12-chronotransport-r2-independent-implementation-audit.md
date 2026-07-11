---
type: source
source_id: source:chronotransport-r2-independent-implementation-audit
date: 2026-07-12
verdict: REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
review_mode: independent_zero_conversation_context
---

# ChronoTransport r2 Independent Implementation Audit

## Verdict

`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`

The independently approved specification remains valid, but the current implementation must not be
frozen as implementation commit I and must not produce registration commit R.

## Registration-blocking findings

1. Gate 3 and Gate 4 adjudicators and r2 launchers are absent.
2. There is no executable r2 Stage-B/Stage-C/matched-dense workflow; the existing formal runner still
   uses the legacy six-schedule and split procedures.
3. Stage-C overflow backoff, full snapshot/restore, same-materialized-batch retry, and retry cap are
   absent; only ownership and one-attempt AMP gradient primitives exist.
4. The learned scheduler does not enforce registered B* and exact measured requested cost.
5. Full-stack profiling and provenance-complete cost lookup are incomplete.
6. Registration validation is too permissive and does not derive/revalidate the complete input chain.
7. Gate-1 CLI accepts caller-controlled statistical settings rather than enforcing registration-bound
   fixed constants and the exact registered population/library.

## Confirmed implemented subset

Canonical protocol/exposure helpers; frozen candidate library and controls; dual-age cache; runtime
all-row adapter and requested/executed separation; D=23 window head; Gate-1/Gate-2 pure primitives;
Stage-C ownership/one-attempt gradient primitive; and a registration/launcher skeleton.

## Decision

Do not create I/R and do not deploy a formal Gate. Continue from the seven blocking findings. The root
agent's remote combined suite passed 110 tests; the independent reviewer could not run local tests due
to its Windows PyTorch DLL environment, so its verdict is based on source/spec inspection.
