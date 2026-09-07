# GeoSparse on official AdaTAD

The base is official sming256/OpenTAD commit 346d09d19e2091372cec48172dbe40f7b28bdee6. Keep `opentad/`, the inherited model/dataset configs, and original train/test entry points unchanged. Implement research routes in `geosparse_ext/` and focused tests.

User-mandated protocol: all 200 THUMOS training videos; all 211 official test videos / 792 test windows at every formal validation; no internal training holdout. Official defaults are 768 input frames, 160px, 768 detector grid, global batch 2, warm-up 5, cosine horizon 100, training end epoch 60. Run seed 0 first. Checkpoint selection uses full-test EMA mAP every 5 epochs and must retain full predictions and provenance.

Do not resume withdrawn 180-video / cosine-60 runs. Historical manifests and results remain under the external execution package. Current focused and full amended manifests have distinct provenance and new IDs. No mAP or Oracle promotion gates.

Validate changes with the GeoSparse detector, native execution, runtime, training validation, and official protocol tests. Production capabilities additionally require the actual GPU native shapes, forward/backward, optimizer, EMA, and original source output/gradient checks.

Keep generated data, weights, run logs and plots outside this repository. Remote N16 writes stay within `/data/run01/sczc063/yuzibo`; N16 allocations use the authorized physical GPU1 / container GPU0 mapping. A100 writes stay in the user's isolated `geosparse_tad_20260907` directory and use its actual Slurm allocations. No training on login nodes.
