import json
from pathlib import Path

import torch
import nms_1d_cpu


official_repo = Path(
    "/data/run01/sczc063/yuzibo/projects/actionformer_official_61ea7eb_20260729_v1"
).resolve()
extension_path = Path(nms_1d_cpu.__file__).resolve()
doc = nms_1d_cpu.softnms.__doc__ or ""
assert extension_path.parent == official_repo / "libs" / "utils"
assert "t1" not in doc and "t2" not in doc

segments = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32).contiguous()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).contiguous()
detections = torch.empty((2, 3), dtype=torch.float32).contiguous()
kept = nms_1d_cpu.softnms(
    segments,
    scores,
    detections,
    0.5,
    0.5,
    0.0,
    2,
)
assert kept.tolist() == [0, 1]
assert torch.isfinite(detections).all()

print(
    json.dumps(
        {
            "extension_path": str(extension_path),
            "softnms_doc": doc,
            "smoke_kept": kept.tolist(),
            "smoke_detections": detections.tolist(),
            "torch_version": torch.__version__,
        },
        sort_keys=True,
    )
)
