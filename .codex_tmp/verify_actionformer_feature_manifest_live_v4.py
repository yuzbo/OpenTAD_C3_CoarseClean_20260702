import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-receipt", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root).resolve()
    sys.path.insert(0, str(audit_root))
    from tools.bata.build_actionformer_official_record import (  # noqa: PLC0415
        build_feature_manifest,
        parse_annotation,
    )

    annotation = Path(args.annotation).resolve()
    feature_dir = Path(args.feature_dir).resolve()
    sealed_manifest = Path(args.sealed_manifest).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_receipt = Path(args.output_receipt).resolve()

    _, _, videos = parse_annotation(annotation)
    live = build_feature_manifest(feature_dir, videos)
    sealed = json.loads(sealed_manifest.read_text(encoding="utf-8"))
    if live != sealed:
        raise SystemExit("live feature manifest differs from the sealed official manifest")

    output_manifest.write_text(
        json.dumps(live, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    live_sha = sha256_file(output_manifest)
    sealed_sha = sha256_file(sealed_manifest)
    if live_sha != sealed_sha:
        raise SystemExit("canonical live feature manifest hash differs from sealed hash")

    receipt = {
        "schema_version": "actionformer_live_feature_manifest_rehash_v1",
        "validation_pass": True,
        "feature_inventory_video_count": live["feature_inventory_video_count"],
        "annotation_feature_backed_video_count": live[
            "annotation_feature_backed_video_count"
        ],
        "evaluated_feature_backed_video_count": live[
            "evaluated_feature_backed_video_count"
        ],
        "live_manifest_sha256": live_sha,
        "sealed_manifest_sha256": sealed_sha,
        "all_ids_content_shape_dtype_exact": True,
    }
    output_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
