# DUCA experiment tracker

| Stage | Item | Status | Evidence |
|---|---|---|---|
| A | Freeze full-200/exact-211 protocol | tested | 11 local focused checks passed; Linux loader/Slurm gate pending |
| A | dense, 3 seeds | implemented | launcher and receipts implemented; not submitted |
| A | uniform fixed K384, 3 seeds | implemented | launcher and receipts implemented; not submitted |
| A | mixed-K train / K384 eval, 3 seeds | implemented | launcher and receipts implemented; not submitted |
| A | learned fixed K384, 3 seeds | implemented | launcher and receipts implemented; not submitted |
| B | full-200 OOF utility/risk targets | designed | blocked on completed Stage-A mixed-K checkpoint |
| B | dynamic mean-K384, 3 seeds | designed | blocked on valid OOF targets/protocol |
| B | exact same-realized-K replay | designed | blocked on dynamic inference ledgers |

Status vocabulary follows the research-wiki contract.  A Slurm job becoming
`COMPLETED` is not by itself `empirically_supported`; the terminal receipt and
complete official evaluation must also pass.
