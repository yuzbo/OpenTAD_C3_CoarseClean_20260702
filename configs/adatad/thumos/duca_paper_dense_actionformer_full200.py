_base_ = ["./duca_paper_full200_base.py"]


duca_paper_cell = dict(
    arm="dense",
    detector_backend="ActionFormer",
    train_video_count=200,
    evaluation_video_count=211,
    world_size=2,
    global_batch_size=2,
    successful_updates=6000,
    evaluation_heavy_k=768,
    selector=None,
)

work_dir = "exps/thumos/adatad/duca_paper_dense_actionformer_full200"
