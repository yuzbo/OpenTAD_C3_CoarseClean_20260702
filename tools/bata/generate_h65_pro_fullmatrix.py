from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos" / "h65_pro"
DOC_DIR = ROOT / "docs" / "experiments" / "h65_pro_fullmatrix_20260902"
BASE_COMMIT = "04c35a3b76897e6c1569eeede41ed3aecaf7f854"
BRANCH = "codex/h65-pro-fullmatrix-strict60-20260902"


F_MATRIX = [
    ("F01", 0, 0, 0, 0, 1),
    ("F02", 0, 0, 0, 1, 0),
    ("F03", 0, 0, 1, 0, 0),
    ("F04", 0, 0, 1, 1, 1),
    ("F05", 0, 1, 0, 0, 0),
    ("F06", 0, 1, 0, 1, 1),
    ("F07", 0, 1, 1, 0, 1),
    ("F08", 0, 1, 1, 1, 0),
    ("F09", 1, 0, 0, 0, 0),
    ("F10", 1, 0, 0, 1, 1),
    ("F11", 1, 0, 1, 0, 1),
    ("F12", 1, 0, 1, 1, 0),
    ("F13", 1, 1, 0, 0, 1),
    ("F14", 1, 1, 0, 1, 0),
    ("F15", 1, 1, 1, 0, 0),
    ("F16", 1, 1, 1, 1, 1),
]

CANONICAL = {
    "C0": (0, 0, 0, 0, 1, [5417, 9173]),
    "C1": (1, 0, 0, 0, 1, [3407, 5417, 9173]),
    "C2": (1, 1, 0, 0, 1, [5417, 9173]),
    "C3": (1, 1, 1, 1, 1, [5417, 9173]),
}


def yn(value: int) -> str:
    return "ON" if int(value) else "OFF"


def py_bool(value: int) -> str:
    return "True" if int(value) else "False"


def schedule_text(curriculum: int) -> str:
    if curriculum:
        return """dict(
                _delete_=True,
                type="progressive_joint",
                shape="cosine",
                warmup_steps=0,
                transition_steps=1,
                actionness=dict(start=1.0, end=1.0),
                transition=dict(start=0.5, end=0.5),
                transition_boundary=dict(start=0.0, end=0.25, warmup_steps=1500, transition_steps=2000),
                detector_gradient=dict(start=0.0, end=0.25, warmup_steps=1500, transition_steps=2000),
                policy_alpha=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=2000),
                detector_contribution=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=2000),
                asformer_adapt=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=2000),
            )"""
    return """dict(
                _delete_=True,
                type="progressive_joint",
                shape="linear",
                warmup_steps=0,
                transition_steps=3000,
                actionness=dict(start=1.0, end=1.0),
                transition=dict(start=0.5, end=0.5),
                transition_boundary=dict(start=0.0, end=0.25, warmup_steps=0, transition_steps=3000),
                detector_gradient=dict(start=0.0, end=0.25, warmup_steps=2100, transition_steps=1500),
                policy_alpha=dict(start=0.0, end=1.0, warmup_steps=0, transition_steps=3000),
                detector_contribution=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=900),
                asformer_adapt=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=900),
            )"""


def factor_config(run_id: str, phase: int, ct: int, mod: int, taylor: int, curriculum: int) -> str:
    acquisition = "semantic_phase_sampling" if phase else "budget_calibrated_sampling_rate"
    contribution_mode = "signed_removal_utility" if taylor else "abs_grad_times_input"
    conv_cfg = (
        """dict(
            _delete_=True,
            type="ContinuousTimeScaleAdaptiveConv1d",
            local_ref_delta_t=2.0,
            context_ref_delta_t=2.0,
            context_level_base=2.0,
            max_abs_offset=64.0,
        )"""
        if ct
        else "None"
    )
    amod_config = (
        """dict(
                _delete_=True,
                enabled=True,
                name="Mixture-of-Depths",
                amod_layers=[1, 3, 5, 7, 9, 11],
                capacity_schedule=dict(
                    warmup_steps=1500,
                    transition_steps=2000,
                    start_capacity=1.0,
                    end_capacity=0.5,
                ),
                min_capacity=0.5,
            )"""
        if mod
        else """dict(_delete_=True, enabled=False)"""
    )
    return dedent(
        f"""
        _base_ = ["./base_h65_pro_strict60.py"]

        h65_pro_experiment_id = "{run_id}"
        h65_pro_factor_policy = dict(
            phase={py_bool(phase)},
            ct={py_bool(ct)},
            mod={py_bool(mod)},
            taylor={py_bool(taylor)},
            curriculum={py_bool(curriculum)},
            generator="Resolution-V" if "{run_id}".startswith("F") else "canonical",
            fixed_relation="E=ABCD",
        )

        model = dict(
            frame_selector=dict(
                acquisition_policy="{acquisition}",
                detector_contribution_mode="{contribution_mode}",
                loss_weight_schedule={schedule_text(curriculum)}
            ),
            backbone=dict(
                backbone=dict(
                    amod_config={amod_config},
                ),
            ),
            rpn_head=dict(
                conv_cfg={conv_cfg},
            ),
        )

        work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/{run_id.lower()}"
        """
    ).lstrip()


def base_config() -> str:
    return dedent(
        """
        _base_ = ["../duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]

        from tools.bata.duca_cellcf_protocol import protocol_for_name

        duca_training_protocol = protocol_for_name("official60")
        seed = 3407
        total_epochs = 60
        max_updates = 6000
        dense_window_size = 768
        window_size = 384
        scale_factor = 2
        chunk_num = window_size * scale_factor // 16
        h65_pro_base_commit = "04c35a3b76897e6c1569eeede41ed3aecaf7f854"
        h65_pro_branch = "codex/h65-pro-fullmatrix-strict60-20260902"
        h65_pro_budget_contract = dict(
            dataset="THUMOS14",
            dense_frames=768,
            selected_frames=384,
            epochs=60,
            successful_optimizer_updates=6000,
            pretrain="public VideoMAE-S Kinetics-400",
            detector="AdaTAD ActionFormer",
            primary_checkpoint="epoch_59.pth/state_dict_ema",
            secondary_checkpoint="epoch_59.pth/state_dict",
        )

        model = dict(
            frame_selector=dict(
                budget=window_size,
                dense_window_size=dense_window_size,
                acquisition_policy="budget_calibrated_sampling_rate",
                semantic_phase_sigma=2.0,
                semantic_phase_scaffold_budget=128,
                semantic_phase_onset_budget=64,
                semantic_phase_offset_budget=64,
                semantic_phase_core_budget=128,
                detector_gradient_mode="density_transport_st",
                sampling_rate_utility_components="both",
                detector_contribution_distillation_weight=1.0,
                detector_contribution_components="both",
                detector_contribution_mode="abs_grad_times_input",
                training_uniform_companion_fraction=0.50,
                training_uniform_companion_normalize_learned_gradient=True,
                hard_max_gap_repair=False,
                fail_on_infeasible_max_gap=False,
                loss_weights=dict(
                    actionness=1.0,
                    detector=1.0,
                    transition=0.5,
                    transition_boundary=0.25,
                    max_gap_hole=0.0,
                    teacher=0.0,
                    detector_utility=0.0,
                    start=0.0,
                    end=0.0,
                    context=0.0,
                    boundary=0.0,
                    hole=0.0,
                    budget=0.0,
                    redundancy=0.0,
                    radius=0.0,
                    entropy=0.0,
                ),
            ),
            backbone=dict(
                backbone=dict(
                    total_frames=dense_window_size,
                    num_frames=16,
                    tubelet_size=2,
                    relative_physical_time_residual=True,
                    tubelet_packed_runtime_route=dict(enabled=False),
                    amod_config=dict(enabled=False),
                ),
                custom=dict(
                    global_rank_selection=True,
                    canonical_selection="exact_uniform_positions_once_over_dense_window",
                ),
            ),
            projection=dict(max_seq_len=window_size),
            rpn_head=dict(conv_cfg=None),
        )

        scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)
        solver = dict(static_graph=False, find_unused_parameters=True)
        workflow = dict(
            formal_protocol="duca_selected_axis_optimization_v1",
            training_profile=duca_training_protocol.name,
            logging_interval=50,
            checkpoint_interval=5,
            val_loss_interval=-1,
            val_eval_interval=-1,
            val_eval_interval_anchor_epoch=9999,
            val_start_epoch=9999,
            end_epoch=60,
            formal_successful_update_contract=True,
            expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
            expected_successful_optimizer_updates=duca_training_protocol.expected_successful_optimizer_updates,
            max_amp_retries_per_batch=8,
            fail_on_amp_replay_exhaustion=True,
            require_finite_train_loss=True,
            primary_checkpoint_epoch=59,
            primary_checkpoint_state_key="state_dict_ema",
            checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
        )

        work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/base"
        """
    ).lstrip()


def ref_d768_config() -> str:
    return dedent(
        """
        _base_ = ["../e2e_thumos_videomae_s_768x1_160_adapter.py"]

        from tools.bata.duca_cellcf_protocol import protocol_for_name

        duca_training_protocol = protocol_for_name("official60")
        seed = 3407
        total_epochs = 60
        max_updates = 6000
        h65_pro_experiment_id = "REF-D768"
        h65_pro_factor_policy = dict(
            phase=False,
            ct=False,
            mod=False,
            taylor=False,
            curriculum=False,
            frames=768,
            reference="dense_768_no_acquisition",
        )

        model = dict(
            frame_selector=None,
            backbone=dict(
                backbone=dict(
                    total_frames=768,
                    num_frames=16,
                    tubelet_size=2,
                    amod_config=dict(_delete_=True, enabled=False),
                ),
            ),
            projection=dict(max_seq_len=768),
            rpn_head=dict(conv_cfg=None),
        )
        scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)
        workflow = dict(
            formal_protocol="h65_pro_dense_reference_official60_v1",
            training_profile=duca_training_protocol.name,
            logging_interval=50,
            checkpoint_interval=5,
            val_loss_interval=-1,
            val_eval_interval=-1,
            val_eval_interval_anchor_epoch=9999,
            val_start_epoch=9999,
            end_epoch=60,
            formal_successful_update_contract=True,
            expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
            expected_successful_optimizer_updates=duca_training_protocol.expected_successful_optimizer_updates,
            max_amp_retries_per_batch=8,
            fail_on_amp_replay_exhaustion=True,
            require_finite_train_loss=True,
            selector_schedule_required=False,
            primary_checkpoint_epoch=59,
            primary_checkpoint_state_key="state_dict_ema",
            checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
        )
        work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/ref_d768"
        """
    ).lstrip()


def ref_u384_config() -> str:
    return dedent(
        """
        _base_ = ["./base_h65_pro_strict60.py"]

        h65_pro_experiment_id = "REF-U384"
        h65_pro_factor_policy = dict(
            phase=False,
            ct=False,
            mod=False,
            taylor=False,
            curriculum=False,
            frames=384,
            reference="uniform_k384",
        )

        model = dict(
            frame_selector=dict(
                inference_policy_alpha=0.0,
                training_uniform_companion_fraction=0.0,
                training_uniform_companion_normalize_learned_gradient=False,
                sampling_rate_utility_components="none",
                detector_contribution_distillation_weight=0.0,
                detector_contribution_components="none",
                detector_contribution_mode="abs_grad_times_input",
                loss_weight_schedule=dict(
                    _delete_=True,
                    type="progressive_joint",
                    shape="linear",
                    warmup_steps=0,
                    transition_steps=1,
                    actionness=dict(start=0.0, end=0.0),
                    transition=dict(start=0.0, end=0.0),
                    transition_boundary=dict(start=0.0, end=0.0),
                    detector_gradient=dict(start=0.0, end=0.0),
                    policy_alpha=dict(start=0.0, end=0.0),
                    detector_contribution=dict(start=0.0, end=0.0),
                    asformer_adapt=dict(start=0.0, end=0.0),
                ),
            ),
            backbone=dict(backbone=dict(amod_config=dict(_delete_=True, enabled=False))),
            rpn_head=dict(conv_cfg=None),
        )
        work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/ref_u384"
        """
    ).lstrip()


def ref_mnv3fc384_config() -> str:
    return dedent(
        """
        _base_ = ["../duca_trainfree_fixed384_official60_base.py"]

        from tools.bata.duca_cellcf_protocol import protocol_for_name

        duca_training_protocol = protocol_for_name("official60")
        seed = 3407
        total_epochs = 60
        max_updates = 6000
        h65_pro_experiment_id = "REF-MNV3FC384"
        h65_pro_factor_policy = dict(
            phase=False,
            ct=False,
            mod=False,
            taylor=False,
            curriculum=False,
            frames=384,
            reference="frozen_mobilenetv3_feature_change",
        )

        model = dict(
            backbone=dict(backbone=dict(amod_config=dict(_delete_=True, enabled=False))),
            rpn_head=dict(conv_cfg=None),
        )
        workflow = dict(
            formal_protocol="duca_selected_axis_optimization_v1",
            training_profile=duca_training_protocol.name,
            checkpoint_interval=5,
            val_loss_interval=-1,
            val_eval_interval=-1,
            val_start_epoch=9999,
            end_epoch=60,
            formal_successful_update_contract=True,
            expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
            expected_successful_optimizer_updates=duca_training_protocol.expected_successful_optimizer_updates,
            max_amp_retries_per_batch=8,
            fail_on_amp_replay_exhaustion=True,
            require_finite_train_loss=True,
            primary_checkpoint_epoch=59,
            primary_checkpoint_state_key="state_dict_ema",
            checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
        )
        work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/ref_mnv3fc384"
        """
    ).lstrip()


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def add(experiment_id, category, phase, ct, mod, taylor, curriculum, frames, seed, config, variant):
        train = (
            "sbatch --parsable tools/experiments/run_h65_pro_train.sbatch "
            f"{config} {seed} {experiment_id} {variant}"
        )
        eval_cmd = (
            "sbatch --parsable --dependency=afterok:${TRAIN_JOB_ID} "
            "tools/experiments/run_h65_pro_eval.sbatch "
            f"{config} {seed} {experiment_id} {variant}"
        )
        out.append(
            {
                "experiment_id": experiment_id,
                "category": category,
                "phase": yn(phase),
                "ct": yn(ct),
                "mod": yn(mod),
                "taylor": yn(taylor),
                "curriculum": yn(curriculum),
                "frames": str(frames),
                "seed": str(seed),
                "config": config,
                "variant": variant,
                "train_command": train,
                "eval_command": eval_cmd,
                "train_job_id": "",
                "eval_job_id": "",
                "status": "FROZEN",
            }
        )

    add("REF-D768", "reference", 0, 0, 0, 0, 0, 768, 3407, "configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py", "h65_pro_ref_d768")
    add("REF-U384", "reference", 0, 0, 0, 0, 0, 384, 3407, "configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py", "h65_pro_ref_u384")
    add(
        "REF-MNV3FC384",
        "reference",
        0,
        0,
        0,
        0,
        0,
        384,
        3407,
        "configs/adatad/thumos/h65_pro/h65_pro_ref_mnv3fc384.py",
        "h65_pro_ref_mnv3fc384",
    )
    for run, phase, ct, mod, taylor, curriculum in F_MATRIX:
        config = f"configs/adatad/thumos/h65_pro/h65_pro_{run.lower()}.py"
        add(run, "resolution_v", phase, ct, mod, taylor, curriculum, 384, 3407, config, f"h65_pro_{run.lower()}")
    for run, values in CANONICAL.items():
        phase, ct, mod, taylor, curriculum, seeds = values
        config = f"configs/adatad/thumos/h65_pro/h65_pro_{run.lower()}.py"
        for seed in seeds:
            add(f"{run}-S{seed}", "canonical", phase, ct, mod, taylor, curriculum, 384, seed, config, f"h65_pro_{run.lower()}")
    return out


def write_configs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "base_h65_pro_strict60.py").write_text(base_config(), encoding="utf-8")
    (CONFIG_DIR / "h65_pro_ref_d768.py").write_text(ref_d768_config(), encoding="utf-8")
    (CONFIG_DIR / "h65_pro_ref_u384.py").write_text(ref_u384_config(), encoding="utf-8")
    (CONFIG_DIR / "h65_pro_ref_mnv3fc384.py").write_text(ref_mnv3fc384_config(), encoding="utf-8")
    for run, phase, ct, mod, taylor, curriculum in F_MATRIX:
        (CONFIG_DIR / f"h65_pro_{run.lower()}.py").write_text(
            factor_config(run, phase, ct, mod, taylor, curriculum),
            encoding="utf-8",
        )
    for run, values in CANONICAL.items():
        phase, ct, mod, taylor, curriculum, _seeds = values
        (CONFIG_DIR / f"h65_pro_{run.lower()}.py").write_text(
            factor_config(run, phase, ct, mod, taylor, curriculum),
            encoding="utf-8",
        )


def write_docs(matrix_rows: list[dict[str, str]]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "00_BASE_IDENTITY.md").write_text(
        dedent(
            f"""
            # H65-Pro Full Matrix Base Identity

            - Branch: `{BRANCH}`
            - Verified H65 base commit: `{BASE_COMMIT}`
            - Base route: real H65 selected-axis official60 stack, not CT-DP-BAMoD.
            - H65 lineage files retained: `duca_h65_first_singleclock_cycle2.py`, `duca_h65_first_singleclock_cycle3.py`, `duca_h65_first_singleclock_cycle4.py`.
            - Fixed dataset: THUMOS14.
            - Fixed dense window: T=768.
            - Fixed sparse detector budget for H65-Pro and K384 references: K=384.
            - Fixed training budget: 60 epochs, 6000 successful optimizer updates.
            - Fixed pretrain: public VideoMAE-S Kinetics-400 checkpoint already used by the H65/AdaTAD configs.

            CT-DP-BAMoD branches were inspected only as prior implementation references. This branch starts from the verified H65 commit above.
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (DOC_DIR / "01_IMPLEMENTATION.md").write_text(
        dedent(
            """
            # Implementation Notes

            - A Phase: `semantic_phase_sampling` in `DucaAcquisitionAdapter` uses ASFormer logits or logit(p_action), masked Gaussian smoothing sigma=2, centered derivative, ReLU onset/offset, sigmoid core, q0.9 per-component scaling, and fixed scaffold/onset/offset/core quotas 128/64/64/128 with deterministic backfill to exact K.
            - B CT: `ContinuousTimeScaleAdaptiveConv1d` provides standard, local delta_tau=2, and context delta_tau=2*2^level branches with `y = y_std + eta_l * (y_ct - y_std)` and eta initialized to zero.
            - C MoD: `VisionTransformerAdapter` supports Mixture-of-Depths top-K execution at layers [1,3,5,7,9,11], unselected identity bypass, and successful-update capacity schedule 1.0 -> 0.5 over updates 1500-3500.
            - D Taylor: detector contribution distillation keeps the existing abs(x*grad) mode and adds signed removal utility `relu(-x*grad)` at the same autograd site, without higher-order graph creation.
            - E Curriculum: configs freeze either linear strict60 policy-alpha ramp or 15/20/25 cosine curriculum. MoD capacity is driven by successful optimizer updates.
            - Slurm: `tools/experiments/run_h65_pro_train.sbatch`, `tools/experiments/run_h65_pro_eval.sbatch`, and `tools/experiments/submit_h65_pro_fullmatrix.sh` enforce official60, clean exact commit, GPU1, epoch-59 EMA, and out-of-repo submission registries.
            - Diagnostic: `tools/bata/h65_pro_hard_one_swap_diagnostic.py` is an offline hard one-swap alignment summary tool and is not called from the training path.
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (DOC_DIR / "02_TEST_RESULTS.md").write_text(
        dedent(
            """
            # Test Results

            Post-review local structural check:

            - `python tools/bata/validate_h65_pro_fullmatrix.py`: PASS, 28 unique train jobs, configs, factors, and strict60 identities.
            - `python -m py_compile tools/train.py tools/test.py tools/bata/duca_p0_training.py tools/bata/duca_selected_axis_training.py tools/bata/generate_h65_pro_fullmatrix.py tools/bata/validate_h65_pro_fullmatrix.py opentad/models/detectors/actionformer.py opentad/models/dense_heads/anchor_free_head.py opentad/models/duca/acquisition.py`: PASS.
            - `python -m pytest tests/test_h65_pro_fullmatrix.py -q`: PASS, 8 passed and 7 skipped because this Windows host cannot load PyTorch `c10.dll`.
            - `python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q`: PASS, 23 passed.
            - `bash -n tools/experiments/run_h65_pro_train.sbatch tools/experiments/run_h65_pro_eval.sbatch tools/experiments/submit_h65_pro_fullmatrix.sh`: PASS.
            - `git diff --check`: PASS.

            Remote Torch regression checks, full `PRECHECK_ONLY=1`, and Slurm submission remain pending until the clean pushed fix commit is available on the N16R4/Slurm host.
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (DOC_DIR / "04_SUBMISSION_REGISTRY.md").write_text(
        dedent(
            """
            # Submission Registry

            Status: PENDING remote submission.

            The frozen 28-row source matrix is `03_EXPERIMENT_MATRIX.csv`. The formal Slurm submitter writes live job ids outside the git checkout at:

            `/data/run01/sczc063/yuzibo/h65_pro_fullmatrix_20260902_submission/<commit>/submission_registry.csv`

            This keeps the repository clean for `tools/train.py` and `tools/test.py` formal exact-commit checks.

            One superseded trial submission from an older intermediate commit accepted `REF-D768` train/eval ids `1265842` and `1265843` before the limit fired. Those jobs were canceled and the old registry was marked `CANCELED_SUPERSEDED_COMMIT`. They must not be used for final H65-Pro results.
            """
        ).lstrip(),
        encoding="utf-8",
    )
    with (DOC_DIR / "05_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "experiment_id",
            "seed",
            "status",
            "mAP@0.3",
            "mAP@0.4",
            "mAP@0.5",
            "mAP@0.6",
            "mAP@0.7",
            "Avg-mAP",
            "checkpoint_path",
            "evaluation_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow(
                {
                    "experiment_id": row["experiment_id"],
                    "seed": row["seed"],
                    "status": "PENDING",
                    "mAP@0.3": "",
                    "mAP@0.4": "",
                    "mAP@0.5": "",
                    "mAP@0.6": "",
                    "mAP@0.7": "",
                    "Avg-mAP": "",
                    "checkpoint_path": "",
                    "evaluation_json": "",
                }
            )
    (DOC_DIR / "06_FINAL_ANALYSIS.md").write_text(
        dedent(
            """
            # Final Analysis

            Status: PENDING remote train/eval completion.

            No metric claims are made before all 28 terminal epoch-59 EMA evaluations complete and `05_RESULTS.csv` is populated from the structured metrics JSON files.
            """
        ).lstrip(),
        encoding="utf-8",
    )
    with (DOC_DIR / "03_EXPERIMENT_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "experiment_id",
            "category",
            "phase",
            "ct",
            "mod",
            "taylor",
            "curriculum",
            "frames",
            "seed",
            "config",
            "variant",
            "train_command",
            "eval_command",
            "train_job_id",
            "eval_job_id",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)


def main() -> None:
    matrix_rows = rows()
    identities = {(row["experiment_id"], row["seed"], row["config"]) for row in matrix_rows}
    if len(matrix_rows) != 28 or len(identities) != len(matrix_rows):
        raise SystemExit("H65-Pro matrix must contain 28 unique training identities")
    write_configs()
    write_docs(matrix_rows)
    print(f"Wrote {len(matrix_rows)} H65-Pro matrix rows and configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
