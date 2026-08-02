# DUCA experiment tracker

| Stage | Item | Status | Evidence |
|---|---|---|---|
| A | Freeze full-200/exact-211 protocol | tested | 11 local focused checks; authoritative Slurm gate 1213711 passed 37 Linux/PyTorch tests |
| A | dense, 3 seeds | incomplete_matrix_only | three cell receipts exist; metrics unopened and unusable without the complete seal |
| A | uniform fixed K384, 3 seeds | incomplete_matrix_only | three cell receipts exist; metrics unopened and unusable without the complete seal |
| A | mixed-K train / K384 eval, 3 seeds | failed_closed | all three jobs rejected short-window effective-K shrinkage; scientific protocol repair required |
| A | learned fixed K384, 3 seeds | not_run | ordered after the failed mixed-K arm in each seed job |
| A | Scheduler grouping | terminal_failed_closed | jobs 1213712/1213713/1213714 failed 1:0; seal 1213715 cancelled |
| B | full-200 OOF utility/risk targets | designed | blocked on repaired and completed Stage-A mixed-K checkpoint |
| B | dynamic mean-K384, 3 seeds | designed | blocked on valid OOF targets/protocol |
| B | exact same-realized-K replay | designed | blocked on dynamic inference ledgers |

Status vocabulary follows the research-wiki contract.  A Slurm job becoming
`COMPLETED` is not by itself `empirically_supported`; the terminal receipt and
complete official evaluation must also pass.
