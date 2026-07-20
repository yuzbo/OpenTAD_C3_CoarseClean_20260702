from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_training import bind_training_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize one immutable Continuous-RoI S2 training config"
    )
    parser.add_argument("source_config", type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--full-model-gate", type=Path, required=True)
    parser.add_argument("--training-runtime-precheck", type=Path, required=True)
    parser.add_argument("--runtime-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(
                "refusing to overwrite a Continuous-RoI S2 bound config"
            )
        cfg = bind_training_config(
            source_config_path=args.source_config,
            family=args.family,
            seed=args.seed,
            work_dir=args.work_dir,
            manifest_path=args.manifest,
            development_annotation_path=args.development_annotation,
            class_map_path=args.class_map,
            development_video_root=args.development_video_root,
            pretrained_checkpoint_path=args.pretrained,
            full_model_gate_path=args.full_model_gate,
            training_runtime_precheck_path=args.training_runtime_precheck,
            runtime_authorization_path=args.runtime_authorization,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(args.output))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {"status": "PASS", "output": str(args.output.resolve())},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
