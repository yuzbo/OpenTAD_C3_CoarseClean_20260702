from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


SEEDS = (3407, 5801, 8123)
BUDGETS = (384, 256)
ARMS = ("uniform", "learned")
BACKENDS = ("actionformer", "temporalmaxer")


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
    formal_protocol="duca_r5_mechanism_matrix",
    formal_successful_update_contract=False,
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
mkdir -p {_python_path(output_dir / "logs")!r}
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
) -> str:
    name = f"r5-{backend[:2]}-{arm[0]}-k{budget}-s{seed}"
    frontend_guard = ""
    if arm == "learned":
        frontend_guard = ''': "${DUCA_FRONTEND_CHECKPOINT:?set DUCA_FRONTEND_CHECKPOINT}"
: "${DUCA_FRONTEND_CHECKPOINT_SHA256:?set DUCA_FRONTEND_CHECKPOINT_SHA256}"
: "${DUCA_FRONTEND_CHECKPOINT_EPOCH:?set DUCA_FRONTEND_CHECKPOINT_EPOCH}"
'''
    return (
        _job_header(
            name=name,
            output_dir=output_dir,
            cluster=cluster,
            walltime="7-00:00:00",
        )
        + frontend_guard
        + f'''"${{PYTHON}}" -m torch.distributed.run \\
  --nproc_per_node=1 \\
  --rdzv_backend=c10d \\
  --rdzv_endpoint=localhost:0 \\
  --rdzv_id="r5-${{SLURM_JOB_ID}}-{name}" \\
  tools/train.py {shlex.quote(_python_path(config))} \\
  --id 0 --seed {seed} \\
  --cfg-options "model.backbone.custom.pretrain=${{ADATAD_PRETRAIN_PATH}}"
'''
    )


def render_gate_sbatch(
    *, config: Path, output_dir: Path, cluster: str, seed: int
) -> str:
    name = "r5-temporalmaxer-one-step"
    return _job_header(
        name=name,
        output_dir=output_dir,
        cluster=cluster,
        walltime="04:00:00",
    ) + f''': "${{DUCA_FRONTEND_CHECKPOINT:?set DUCA_FRONTEND_CHECKPOINT}}"
: "${{DUCA_FRONTEND_CHECKPOINT_SHA256:?set DUCA_FRONTEND_CHECKPOINT_SHA256}}"
: "${{DUCA_FRONTEND_CHECKPOINT_EPOCH:?set DUCA_FRONTEND_CHECKPOINT_EPOCH}}"
"${{PYTHON}}" -m tools.bata.run_duca_temporalmaxer_one_step \\
  --config {shlex.quote(_python_path(config))} \\
  --pretrain "${{ADATAD_PRETRAIN_PATH}}" \\
  --pretrain-sha256 "${{ADATAD_PRETRAIN_SHA256}}" \\
  --seed {seed} \\
  --output {_python_path(output_dir / "temporalmaxer_one_step.json")!r}
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
    sources = {
        "uniform": _require_source(uniform_config, repo_root=repo, label="uniform"),
        "learned": _require_source(learned_config, repo_root=repo, label="learned"),
    }
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
                            "sbatch": str(job),
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
        ),
        executable=True,
    )
    columns = ("id", "backend", "arm", "budget", "seed", "config", "sbatch")
    lines = ["\t".join(columns)]
    lines.extend("\t".join(str(cell[key]) for key in columns) for cell in cells)
    _write(output / "cells.tsv", "\n".join(lines) + "\n")
    summary = {
        "output_dir": str(output),
        "cell_count": len(cells),
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "arms": list(ARMS),
        "backends": list(BACKENDS),
        "gate_config": str(gate_config),
        "gate_sbatch": str(gate_job),
        "cells_tsv": str(output / "cells.tsv"),
    }
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
