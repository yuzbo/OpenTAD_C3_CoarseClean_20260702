# Source Registry

| ID | Source | Role | Status |
|---|---|---|---|
| U-TAKEOVER-1 | `C:/Users/skywalker/.codex/attachments/8dd661a0-1596-4394-ba09-e293fb3c9169/pasted-text.txt` | DUCA/AdapTok takeover report | absorbed |
| U-TAKEOVER-2 | `C:/Users/skywalker/.codex/attachments/38deddb7-5b11-45e5-9f30-e8ecfe25a557/pasted-text.txt` | independent takeover report | absorbed |
| U-PRO-CBCG-1 | `C:/Users/skywalker/.codex/attachments/c191e959-68a3-4b33-af06-69b78b5c68a8/pasted-text.txt` | repository audit and CBCG-RIME external review | absorbed as `PARTNER_CLAIM`; route conditionally accepted |
| U-VIDEO-BUDGET-1 | Current task user correction, `2026-07-28` | whole-video total budget with 768-window AdaTAD execution | absorbed; H-RIME design subsequently user-approved |
| U-PRO-HRIME-1 | `C:/Users/skywalker/.codex/attachments/4954e6a1-bd4d-406d-96bf-653d4438c604/pasted-text.txt` | full H-RIME architecture, implementation and publication audit | fully read; conditionally accepted with repository/math/evidence corrections |
| P-ADAPTOK-1 | `https://arxiv.org/html/2505.17011` | official AdapTok paper and appendix | directly verified |
| C-ADAPTOK-1 | `https://github.com/VisionXLab/AdapTok` | official AdapTok repository | directly verified |
| P-EVATOK-1 | `https://openaccess.thecvf.com/content/CVPR2026/html/Xiong_EVATok_Adaptive_Length_Video_Tokenization_for_Efficient_Visual_Autoregressive_Generation_CVPR_2026_paper.html` | official EVATok paper page | directly verified |
| C-EVATOK-1 | `https://github.com/HKU-MMLab/EVATok` | official EVATok repository | directly verified |
| LOCAL-SPEC-1 | `docs/superpowers/specs/2026-07-27-duca-total60-prebackbone-plugin-cvpr-design.md` | superseded foundation specification | retained |
| LOCAL-DESIGN-2 | `docs/methods/2026-07-28-duca-rime-mixed-k-baseline-design.md` | frozen U-mixed-K design | active |
| LOCAL-DESIGN-3 | `docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md` | corrected, user-approved H-RIME v1 specification | active |
| REMOTE-RIME-V6-GATE-1 | `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600/logs/rime-phase1-1201417.out`, SHA-256 `0b9aedc943139e024939fa16bf5cf3007c7ae387e74f04bdae823551e3baee29` | immutable Recovery-v6 protected-gate failure evidence | directly verified; insufficient to identify offending loss component |
| LOCAL-GATE-DIAG-1 | Git commit `69136de3ed8d8f977c78cfe5258dae3d57f7e238` | fail-closed coordinate/loss diagnostic remediation | implemented; focused local suite passed; published on `codex/duca-rime-20260727` |
| U-PRO-PURE-PLUGIN-1 | `C:/Users/skywalker/.codex/attachments/80ec2ddd-9eed-4e97-ac1d-8be0c3071fd5/pasted-text.txt` | selected-axis pure-plugin architecture adjudication, AdapTok competition audit, replacement admission-v2 and experiment DAG | fully read; core scientific verdict accepted, implementation details independently adjudicated |

`U-PRO-CBCG-1` was read in full. Its route analysis is retained as an external
review, not as repository truth. Its linked `sandbox:/mnt/data/...` Patch A,
Patch B, core, and test artifacts are not present in this repository and were
not independently inspected; their hashes and reported test counts therefore
do not establish `implemented` or `tested` status.

AdapTok and EVATok paper/code provenance was independently registered from the
official paper pages and repositories. Their adaptive-length ideas are prior
art/competition context; neither repository has been copied into H-RIME.

`U-PRO-HRIME-1` was read in full. Its main Approach-C architecture and staged
admission route are accepted, while its suggested numeric gates remain
unvalidated proposals. Referenced external sandbox artifacts are absent from
this repository and do not establish implementation or test status.

`U-PRO-PURE-PLUGIN-1` was read in full. Its `CONDITIONAL GO`, paper-mainline
selected-axis plugin, diagnostic-only physical integration, abolition of the
general scalar-loss-equivalence gate, calibrated numeric null, and held-out
same-total-cost scientific gate are accepted. Its proposed code is design
input rather than repository truth. The implementation keeps the existing
fully gated standard `AnchorFreeHead` instead of immediately extracting
physical subclasses, because selected-axis configs and runtime validation
restore the unmodified standard path; subclass extraction remains required
only before a physical integration arm can make a separate experimental claim.
