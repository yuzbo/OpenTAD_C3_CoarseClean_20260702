# DUCA experiment tracker

| Stage | Item | Status | Evidence |
|---|---|---|---|
| A | Freeze full-200/exact-211 protocol | tested | 11 local focused checks; authoritative Slurm gate 1213711 passed 37 Linux/PyTorch tests |
| A | dense, 3 seeds | experiment_running | seed jobs 1213712/1213713/1213714 running; no partial metric opened |
| A | uniform fixed K384, 3 seeds | experiment_running | queued sequentially within the three running seed jobs; no partial metric opened |
| A | mixed-K train / K384 eval, 3 seeds | experiment_running | queued sequentially within the three running seed jobs; no partial metric opened |
| A | learned fixed K384, 3 seeds | experiment_running | queued sequentially within the three running seed jobs; no partial metric opened |
| A | Scheduler grouping | experiment_running | 3 seed jobs x 4 independent sequential arms + dependent seal 1213715; logical 12-cell protocol unchanged |
| B | full-200 OOF utility/risk targets | designed | blocked on completed Stage-A mixed-K checkpoint |
| B | dynamic mean-K384, 3 seeds | designed | blocked on valid OOF targets/protocol |
| B | exact same-realized-K replay | designed | blocked on dynamic inference ledgers |

Status vocabulary follows the research-wiki contract.  A Slurm job becoming
`COMPLETED` is not by itself `empirically_supported`; the terminal receipt and
complete official evaluation must also pass.
