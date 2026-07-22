from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


SEEDS = (3407, 5801, 8123)
BUDGETS = (384, 256)
ARMS = ("uniform", "learned")
BACKENDS = ("actionformer", "temporalmaxer")

LEARNED_VARIANTS = {
    "duca_boundary_burst_g1_protected_fixed384_official60.py": (
        "boundary_burst_r2q3_g1"
    ),
    "duca_boundary_burst_r4q5_g1_protected_fixed384_official60.py": (
        "boundary_burst_r4q5_g1"
    ),
}


def _python_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def _require_source(path: str | Path, *, repo_root: Path, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} config is missing: {source}")
    try:
        source.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} config must be inside the repository") from exc
    return source


def render_cell_config(
    *,
    source: Path,
    backend: str,
    arm: str,
    budget: int,
    seed: int,
    work_dir: Path,
) -> str:
    if backend not in BACKENDS or arm not in ARMS:
        raise ValueError("unknown R5 backend or arm")
    if budget not in BUDGETS or seed not in SEEDS or budget % 16:
        raise ValueError("R5 budget/seed is outside the fixed matrix")
    chunks = budget // 16
    temporal_contract = f'''dict(
    hard_budget={budget},
    dense_window_size=768,
    max_unselected_hole_dense_candidates=2,
    dataset_feature_stride_source_frames=4,
    dataset_sample_stride=1,
    requested_max_source_frame_interval=15,
    detector_axis="selected_axis_index",
    dense_axis_unit="dense_candidate_index",
    task="offline_temporal_action_detection",
)'''
    if backend == "actionformer":
        detector_fields = f'''
    projection=dict(max_seq_len={budget}),'''
    else:
        detector_fields = '''
    type="TemporalMaxer",
    selector_train_only=False,
    selector_train_only_skip_detector=False,
    projection=dict(
        _delete_=True,
        type="TemporalMaxerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 0, 5),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
    ),
    neck=dict(
        _delete_=True,
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        _delete_=True,
        type="TemporalMaxerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[
                (0, 4), (4, 8), (8, 16),
                (16, 32), (32, 64), (64, 10000),
            ],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            iou_weight=2,
            cls_weight=1.0,
            center_radius=1.5,
            keep_percent=1.0,
            confuse_weight=0.0,
        ),
    ),'''
    return f'''_base_ = [{_python_path(source)!r}]

window_size = {budget}
chunk_num = {chunks}
duca_temporal_sampling_contract = {temporal_contract}

r5_cell = dict(
    backend={backend!r},
    arm={arm!r},
    budget={budget},
    seed={seed},
    source_config={_python_path(source)!r},
    live_duca_to_videomae=True,
    detector_type={"'ActionFormer'" if backend == "actionformer" else "'TemporalMaxer'"},
    paper_claim_allowed=False,
)

duca_transition_only_contract = dict(
    exact_budget={budget},
    detector_pretraining_policy="exact_uniform_k{budget}",
    temporal_sampling_contract=duca_temporal_sampling_contract,
    paper_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        budget={budget},
        temporal_sampling_contract=duca_temporal_sampling_contract,
    ),
    backbone=dict(
        backbone=dict(total_frames={budget}),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1={chunks},
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c t",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1={chunks},
                ),
                dict(type="Interpolate", keys=["feats"], size={budget}),
            ],
        ),
    ),{detector_fields}
)

workflow = dict(
    formal_protocol="duca_r5_mechanism_matrix_v1",
    training_profile="official60",
    formal_successful_update_contract=True,
    training_probe_json=None,
    require_training_probe_context=False,
    paper_claim_allowed=False,
)

work_dir = {_python_path(work_dir)!r}
'''


def _job_header(*, name: str, output_dir: Path, cluster: str, walltime: str) -> str:
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --clusters={cluster}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time={walltime}
#SBATCH --output={_python_path(output_dir / "logs" / (name + "-%j.out"))}
#SBATCH --error={_python_path(output_dir / "logs" / (name + "-%j.err"))}

set -euo pipefail
: "${{DUCA_REPO_ROOT:?set DUCA_REPO_ROOT}}"
: "${{DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}}"
: "${{ADATAD_PRETRAIN_PATH:?set ADATAD_PRETRAIN_PATH}}"
: "${{ADATAD_PRETRAIN_SHA256:?set ADATAD_PRETRAIN_SHA256}}"
cd "${{DUCA_REPO_ROOT}}"
source scripts/duca_cellcf_canonical_env.sh
[[ "$(git rev-parse HEAD)" == "${{DUCA_EXPECTED_COMMIT}}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "$(sha256sum "${{ADATAD_PRETRAIN_PATH}}" | awk '{{print $1}}')" == "${{ADATAD_PRETRAIN_SHA256}}" ]]
export DUCA_ADATAD_PRETRAIN_PATH="${{ADATAD_PRETRAIN_PATH}}"
export DUCA_ADATAD_PRETRAIN_SHA256="${{ADATAD_PRETRAIN_SHA256}}"
mkdir -p {_python_path(output_dir / "logs")!r}
'''


def _r5_evidence_guard(*, output_dir: Path) -> str:
    summary = _python_path(output_dir / "matrix_summary.json")
    summary_sha = _python_path(output_dir / "matrix_summary.json.sha256")
    gate = _python_path(output_dir / "temporalmaxer_one_step.json")
    gate_sha = _python_path(output_dir / "temporalmaxer_one_step.json.sha256")
    return f'''export R5_MATRIX_SUMMARY={shlex.quote(summary)}
export R5_MECHANISM_GATE_JSON={shlex.quote(gate)}
[[ -s {shlex.quote(summary_sha)} ]]
[[ -s {shlex.quote(gate_sha)} ]]
IFS= read -r R5_MATRIX_SUMMARY_SHA256 < {shlex.quote(summary_sha)}
IFS= read -r R5_MECHANISM_GATE_SHA256 < {shlex.quote(gate_sha)}
[[ "${{R5_MATRIX_SUMMARY_SHA256}}" =~ ^[0-9a-f]{{64}}$ ]]
[[ "${{R5_MECHANISM_GATE_SHA256}}" =~ ^[0-9a-f]{{64}}$ ]]
[[ "$(sha256sum "${{R5_MATRIX_SUMMARY}}" | awk '{{print $1}}')" == "${{R5_MATRIX_SUMMARY_SHA256}}" ]]
[[ "$(sha256sum "${{R5_MECHANISM_GATE_JSON}}" | awk '{{print $1}}')" == "${{R5_MECHANISM_GATE_SHA256}}" ]]
export R5_MATRIX_SUMMARY_SHA256 R5_MECHANISM_GATE_SHA256
'''


def _learned_runtime_binding(*, source: Path, variant: str) -> str:
    return f''': "${{R5_FRONTEND_DECISION:?set R5_FRONTEND_DECISION}}"
: "${{R5_ALIGNMENT_JSON:?set R5_ALIGNMENT_JSON}}"
if [[ -z "${{R5_FRONTEND_DECISION_SHA256:-}}" ]]; then
  : "${{R5_FRONTEND_DECISION_SHA256_FILE:?set R5_FRONTEND_DECISION_SHA256_FILE}}"
  IFS= read -r R5_FRONTEND_DECISION_SHA256 < "${{R5_FRONTEND_DECISION_SHA256_FILE}}"
fi
if [[ -z "${{R5_ALIGNMENT_SHA256:-}}" ]]; then
  : "${{R5_ALIGNMENT_SHA256_FILE:?set R5_ALIGNMENT_SHA256_FILE}}"
  IFS= read -r R5_ALIGNMENT_SHA256 < "${{R5_ALIGNMENT_SHA256_FILE}}"
fi
[[ "${{R5_FRONTEND_DECISION_SHA256}}" =~ ^[0-9a-f]{{64}}$ ]]
[[ "${{R5_ALIGNMENT_SHA256}}" =~ ^[0-9a-f]{{64}}$ ]]
export R5_FRONTEND_DECISION_SHA256 R5_ALIGNMENT_SHA256
readarray -t r5_frontend < <("${{PYTHON}}" - \
  "${{R5_FRONTEND_DECISION}}" "${{R5_FRONTEND_DECISION_SHA256}}" \
  "${{R5_ALIGNMENT_JSON}}" "${{R5_ALIGNMENT_SHA256}}" \
  "${{DUCA_EXPECTED_COMMIT}}" {shlex.quote(variant)} \
  {shlex.quote(_python_path(source))} <<'PY'
import hashlib
import sys
from pathlib import Path

from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    validate_alignment_artifact,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    validate_frontend_decision,
)

decision_path, decision_sha, alignment_path, alignment_sha, commit, variant, source = sys.argv[1:]
decision = validate_frontend_decision(
    decision_path=decision_path,
    decision_sha256=decision_sha,
    expected_commit=commit,
)
validate_alignment_artifact(
    path=alignment_path,
    digest=alignment_sha,
    expected_commit=commit,
    expected_variant=variant,
    source_config_path=source,
    source_config_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
)
selected = decision["family_routing"]["selected_p0_variant"]
winner = decision["winners"][selected]
print(winner["checkpoint_path"])
print(winner["checkpoint_sha256"])
print(int(winner["epoch_one_based"]) - 1)
PY
)
[[ "${{#r5_frontend[@]}}" == 3 ]]
export DUCA_FRONTEND_CHECKPOINT="${{r5_frontend[0]}}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${{r5_frontend[1]}}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${{r5_frontend[2]}}"
export DUCA_SELECTED_OPT_VARIANT={shlex.quote(variant)}
export DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${{R5_ALIGNMENT_JSON}}"
export DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${{R5_ALIGNMENT_SHA256}}"
[[ -f "${{DUCA_FRONTEND_CHECKPOINT}}" ]]
[[ "$(sha256sum "${{DUCA_FRONTEND_CHECKPOINT}}" | awk '{{print $1}}')" == \
   "${{DUCA_FRONTEND_CHECKPOINT_SHA256}}" ]]
'''


def render_train_sbatch(
    *,
    config: Path,
    backend: str,
    arm: str,
    budget: int,
    seed: int,
    output_dir: Path,
    cluster: str,
    learned_variant: str,
    learned_source: Path,
) -> str:
    name = f"r5-{backend[:2]}-{arm[0]}-k{budget}-s{seed}"
    frontend_guard = ""
    if arm == "learned":
        frontend_guard = _learned_runtime_binding(
            source=learned_source,
            variant=learned_variant,
        )
    cell_id = f"{backend}_{arm}_k{budget}_s{seed}"
    checkpoint = (
        output_dir
        / "runs"
        / cell_id
        / "gpu1_id0/checkpoint/epoch_59.pth"
    )
    evaluation = output_dir / "results" / f"{cell_id}.terminal_evaluation.json"
    evaluation_work = output_dir / "results" / f"{cell_id}_eval"
    return (
        _job_header(
            name=name,
            output_dir=output_dir,
            cluster=cluster,
            walltime="7-00:00:00",
        )
        + _r5_evidence_guard(output_dir=output_dir)
        + frontend_guard
        + f'''export DUCA_SELECTED_OPT_VARIANT={shlex.quote(cell_id)}
mkdir -p {shlex.quote(_python_path(output_dir / "results"))}
'''
        + f'''"${{PYTHON}}" -m torch.distributed.run \\
  --nproc_per_node=1 \\
  --rdzv_backend=c10d \\
  --rdzv_endpoint=localhost:0 \\
  --rdzv_id="r5-${{SLURM_JOB_ID}}-{name}" \\
  tools/train.py {shlex.quote(_python_path(config))} \\
  --id 0 --seed {seed} \\
  --cfg-options "model.backbone.custom.pretrain=${{ADATAD_PRETRAIN_PATH}}"
[[ -f {shlex.quote(_python_path(checkpoint))} ]]
"${{PYTHON}}" -m torch.distributed.run \\
  --nproc_per_node=1 \\
  --rdzv_backend=c10d \\
  --rdzv_endpoint=localhost:0 \\
  --rdzv_id="r5-${{SLURM_JOB_ID}}-{name}-eval" \\
  tools/test.py {shlex.quote(_python_path(config))} \\
  --checkpoint {shlex.quote(_python_path(checkpoint))} \\
  --checkpoint-state-key state_dict_ema \\
  --expected-checkpoint-epoch 59 \\
  --metrics-json {shlex.quote(_python_path(evaluation))} \\
  --id 0 --seed {seed} \\
  --cfg-options \\
    "work_dir={_python_path(evaluation_work)}" \\
    "model.backbone.custom.pretrain=${{ADATAD_PRETRAIN_PATH}}" \\
    "post_processing.save_dict=True" \\
    "inference.load_from_raw_predictions=False"
[[ -s {shlex.quote(_python_path(evaluation))} ]]
'''
    )


def render_cost_sbatch(
    *,
    cell: dict[str, Any],
    output_dir: Path,
    cluster: str,
    learned_variant: str,
    learned_source: Path,
) -> str:
    cell_id = str(cell["id"])
    config = Path(str(cell["config"]))
    checkpoint = output_dir / "runs" / cell_id / "gpu1_id0/checkpoint/epoch_59.pth"
    prefix = output_dir / "cost" / cell_id
    frontend_guard = ""
    if cell["arm"] == "learned":
        frontend_guard = _learned_runtime_binding(
            source=learned_source,
            variant=learned_variant,
        )
    return (
        _job_header(
            name=f"r5-cost-{cell['backend'][:2]}-{cell['arm'][0]}",
            output_dir=output_dir,
            cluster=cluster,
            walltime="08:00:00",
        )
        + _r5_evidence_guard(output_dir=output_dir)
        + frontend_guard
        + f'''export DUCA_SELECTED_OPT_VARIANT={shlex.quote(cell_id)}
[[ -f {shlex.quote(_python_path(checkpoint))} ]]
mkdir -p {shlex.quote(_python_path(output_dir / "cost"))}
export PRECHECK_ONLY=0
export CONFIG={shlex.quote(_python_path(config))}
export PROFILE_CHECKPOINT={shlex.quote(_python_path(checkpoint))}
export PROFILE_METHOD={shlex.quote(cell_id)}
export OUTPUT_PREFIX={shlex.quote(_python_path(prefix))}
export PROFILE_SAMPLES=30 PROFILE_WARMUP_SAMPLES=5 PROFILE_BATCH_SIZE=1
bash scripts/run_duca_full_stack_cost_profile_gpu1.sh
[[ -s {shlex.quote(_python_path(prefix.with_suffix(".summary.json")))} ]]
[[ -s {shlex.quote(_python_path(prefix.with_suffix(".samples.jsonl")))} ]]
'''
    )


def render_aggregate_sbatch(*, output_dir: Path, cluster: str) -> str:
    name = "r5-aggregate"
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --clusters={cluster}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output={_python_path(output_dir / "logs" / (name + "-%j.out"))}
#SBATCH --error={_python_path(output_dir / "logs" / (name + "-%j.err"))}

set -euo pipefail
: "${{DUCA_REPO_ROOT:?set DUCA_REPO_ROOT}}"
: "${{DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}}"
cd "${{DUCA_REPO_ROOT}}"
source scripts/duca_cellcf_canonical_env.sh
[[ "$(git rev-parse HEAD)" == "${{DUCA_EXPECTED_COMMIT}}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
IFS= read -r matrix_sha < {_python_path(output_dir / "matrix_summary.json.sha256")!r}
[[ "$(sha256sum {_python_path(output_dir / "matrix_summary.json")!r} | awk '{{print $1}}')" == "${{matrix_sha}}" ]]
"${{PYTHON}}" -m tools.bata.aggregate_duca_r5_paper_matrix \
  --matrix-summary {_python_path(output_dir / "matrix_summary.json")!r} \
  --expected-commit "${{DUCA_EXPECTED_COMMIT}}" \
  --output-json {_python_path(output_dir / "final_results.json")!r}
sha256sum {_python_path(output_dir / "final_results.json")!r} | awk '{{print $1}}' > \
  {_python_path(output_dir / "final_results.json.sha256")!r}
'''


def render_gate_sbatch(
    *,
    config: Path,
    output_dir: Path,
    cluster: str,
    seed: int,
    learned_variant: str,
    learned_source: Path,
) -> str:
    name = "r5-temporalmaxer-one-step"
    return _job_header(
        name=name,
        output_dir=output_dir,
        cluster=cluster,
        walltime="04:00:00",
    ) + _learned_runtime_binding(
        source=learned_source,
        variant=learned_variant,
    ) + f'''
"${{PYTHON}}" -m tools.bata.run_duca_temporalmaxer_one_step \\
  --config {shlex.quote(_python_path(config))} \\
  --pretrain "${{ADATAD_PRETRAIN_PATH}}" \\
  --pretrain-sha256 "${{ADATAD_PRETRAIN_SHA256}}" \\
  --seed {seed} \\
  --output {_python_path(output_dir / "temporalmaxer_one_step.json")!r}
sha256sum {_python_path(output_dir / "temporalmaxer_one_step.json")!r} | awk '{{print $1}}' > \\
  {_python_path(output_dir / "temporalmaxer_one_step.json.sha256")!r}
'''


def generate_matrix(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    uniform_config: str | Path,
    learned_config: str | Path,
    cluster: str = "n16r4",
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not (repo / "tools/train.py").is_file():
        raise FileNotFoundError(f"not an OpenTAD repository: {repo}")
    git_commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if len(git_commit) != 40:
        raise RuntimeError("R5 generation requires an exact Git commit")
    sources = {
        "uniform": _require_source(uniform_config, repo_root=repo, label="uniform"),
        "learned": _require_source(learned_config, repo_root=repo, label="learned"),
    }
    learned_variant = LEARNED_VARIANTS.get(sources["learned"].name)
    if learned_variant is None:
        raise ValueError(
            "learned R5 config must be an R0-selected boundary-burst G1 config"
        )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for arm in ARMS:
            for budget in BUDGETS:
                for seed in SEEDS:
                    cell_id = f"{backend}_{arm}_k{budget}_s{seed}"
                    config = output / "configs" / f"{cell_id}.py"
                    work_dir = output / "runs" / cell_id
                    job = output / "jobs" / f"{cell_id}.sbatch"
                    _write(
                        config,
                        render_cell_config(
                            source=sources[arm],
                            backend=backend,
                            arm=arm,
                            budget=budget,
                            seed=seed,
                            work_dir=work_dir,
                        ),
                    )
                    _write(
                        job,
                        render_train_sbatch(
                            config=config,
                            backend=backend,
                            arm=arm,
                            budget=budget,
                            seed=seed,
                            output_dir=output,
                            cluster=cluster,
                            learned_variant=learned_variant,
                            learned_source=sources["learned"],
                        ),
                        executable=True,
                    )
                    cells.append(
                        {
                            "id": cell_id,
                            "backend": backend,
                            "arm": arm,
                            "budget": budget,
                            "seed": seed,
                            "config": str(config),
                            "config_sha256": hashlib.sha256(
                                config.read_bytes()
                            ).hexdigest(),
                            "sbatch": str(job),
                            "sbatch_sha256": hashlib.sha256(
                                job.read_bytes()
                            ).hexdigest(),
                        }
                    )
    gate_config = output / "configs/temporalmaxer_learned_k384_s3407.py"
    gate_job = output / "jobs/temporalmaxer_one_step.sbatch"
    _write(
        gate_job,
        render_gate_sbatch(
            config=gate_config,
            output_dir=output,
            cluster=cluster,
            seed=3407,
            learned_variant=learned_variant,
            learned_source=sources["learned"],
        ),
        executable=True,
    )
    columns = (
        "id",
        "backend",
        "arm",
        "budget",
        "seed",
        "config",
        "config_sha256",
        "sbatch",
        "sbatch_sha256",
    )
    lines = ["\t".join(columns)]
    lines.extend("\t".join(str(cell[key]) for key in columns) for cell in cells)
    _write(output / "cells.tsv", "\n".join(lines) + "\n")
    cost_cells = [
        cell
        for cell in cells
        if cell["budget"] == 384 and cell["seed"] == 3407
    ]
    costs: list[dict[str, Any]] = []
    for cell in cost_cells:
        cost_id = f"cost_{cell['id']}"
        job = output / "jobs" / f"{cost_id}.sbatch"
        _write(
            job,
            render_cost_sbatch(
                cell=cell,
                output_dir=output,
                cluster=cluster,
                learned_variant=learned_variant,
                learned_source=sources["learned"],
            ),
            executable=True,
        )
        costs.append(
            {
                "id": cost_id,
                "source_cell": cell["id"],
                "sbatch": str(job),
                "summary": str(output / "cost" / f"{cell['id']}.summary.json"),
            }
        )
    cost_columns = ("id", "source_cell", "sbatch", "summary")
    cost_lines = ["\t".join(cost_columns)]
    cost_lines.extend(
        "\t".join(str(row[key]) for key in cost_columns) for row in costs
    )
    _write(output / "costs.tsv", "\n".join(cost_lines) + "\n")
    summary = {
        "schema": "duca_r5_paper_matrix_v1",
        "task": "offline_temporal_action_detection",
        "git_commit": git_commit,
        "output_dir": str(output),
        "cell_count": len(cells),
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "arms": list(ARMS),
        "backends": list(BACKENDS),
        "learned_variant": learned_variant,
        "learned_source": str(sources["learned"]),
        "gate_config": str(gate_config),
        "gate_config_sha256": hashlib.sha256(gate_config.read_bytes()).hexdigest(),
        "gate_sbatch": str(gate_job),
        "mechanism_gate_output": str(output / "temporalmaxer_one_step.json"),
        "mechanism_gate_sha256_file": str(
            output / "temporalmaxer_one_step.json.sha256"
        ),
        "cells_tsv": str(output / "cells.tsv"),
        "cells": cells,
        "cost_count": len(costs),
        "costs_tsv": str(output / "costs.tsv"),
        "costs": costs,
        "aggregate_sbatch": str(output / "jobs/aggregate.sbatch"),
        "matrix_summary_sha256_file": str(output / "matrix_summary.json.sha256"),
    }
    _write(
        output / "jobs/aggregate.sbatch",
        render_aggregate_sbatch(output_dir=output, cluster=cluster),
        executable=True,
    )
    matrix_summary = output / "matrix_summary.json"
    _write(matrix_summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write(
        output / "matrix_summary.json.sha256",
        hashlib.sha256(matrix_summary.read_bytes()).hexdigest() + "\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the small DUCA R5 K-by-seed mechanism matrix."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uniform-config", required=True)
    parser.add_argument("--learned-config", required=True)
    parser.add_argument("--cluster", default="n16r4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_matrix(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        uniform_config=args.uniform_config,
        learned_config=args.learned_config,
        cluster=args.cluster,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
