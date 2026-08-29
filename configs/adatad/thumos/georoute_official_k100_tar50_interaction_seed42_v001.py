_base_ = ["./georoute_official_amod50_prebackbone_seed42_v001.py"]

# This task-specific alias does not introduce a new model route.  It binds the
# already tested strict A-MoD capacity=0.5 implementation to the frozen CPTC
# K100-TAR50 interaction falsifier and its route-matched capacity=1 reference.
official_bc_arm = "K100-TAR50"
cptc_task_id = "ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001"

zoomtoken_p1_config = dict(
    task_surface="K100-TAR50",
    route_matched_reference_job="1254040",
)

official_amod_contract = dict(
    official_reference_job="1254040",
    route_matched_reference_job="1254040",
    reference_checkpoint_sha256=(
        "3aca10bc3593e301b7d7e77271419b8bb557d8f8b29bead195fa2aa350e34ddd"
    ),
    reference_prediction_sha256=(
        "0d09e3fec839449923db1158a18ead631e813b9d00cdab051328cb2b407485f3"
    ),
    reference_config_sha256=(
        "81c805838502639d4fb0e6fcdd0848c53ccbd8eeccf7d1501562af2e84d9ac87"
    ),
    reference_official_vector=(68.73, 61.59, 47.20),
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_k100_tar50_interaction_seed42_unbound"
)
