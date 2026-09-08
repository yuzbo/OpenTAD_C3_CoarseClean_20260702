# ET-TRC anchor repair and test-guided successor

The user explicitly authorized full-test evaluation every five epochs, best
checkpoint selection and test-guided tuning on 2026-09-08. These are exploratory
runs, not sealed unseen-test evidence. Historical OFF/ON results remain attached
to 74473c27 and are not rewritten.

The only model change zeros the learned proxy correction at exact anchor
positions before the existing temporal Adapter. Non-anchor temporal mixing and
the Adapter remain active. This fixes a reachable neighboring-tap defect; it
does not implement event-triggered anchors or an exact input-dependent JVP.

New matched OFF/ON configurations retain the existing seed 4407, FP32, two-GPU
global batch 2, pretrained weights, data, learning-rate schedule and 60 epochs.
Every fifth completed epoch evaluates all test videos with EMA. Best selection
uses unrounded official Avg-mAP, earliest epoch on ties. All periodic vectors,
best_test.pth and epoch-59 EMA remain distinguishable. No performance gain is
claimed before results exist.
