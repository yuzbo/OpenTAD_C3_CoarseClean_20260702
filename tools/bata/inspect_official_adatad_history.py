"""Read the existing official-recipe run; never train or rewrite its artifacts."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(left.keys() | right.keys()):
            rows.extend(differences(left.get(key), right.get(key), f"{path}.{key}".lstrip(".")))
        return rows
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        left, right = list(left), list(right)
    return [] if left == right else [dict(field=path, official=left, recorded=right)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from mmengine.config import Config
    import torch

    config_name = "e2e_thumos_videomae_s_768x1_160_adapter.py"
    official = Config.fromfile(str(args.repo / "configs/adatad/thumos" / config_name)).to_dict()
    log_path = args.run / "slurm-1245842.out"
    log_text = log_path.read_text(encoding="utf-8")
    config_text = log_text.split("Train INFO: Config: \n", 1)[1]
    config_text = re.split(r"\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} Train INFO:", config_text, maxsplit=1)[0]
    recorded = Config.fromstring(config_text, ".py").to_dict()
    checkpoint_path = args.run / "gpu2_id0/checkpoint/epoch_59.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    optimizer = checkpoint.get("optimizer", {})
    steps = sorted({float(state["step"]) for state in optimizer.get("state", {}).values() if "step" in state})
    pretrain = args.repo / recorded["model"]["backbone"]["custom"]["pretrain"]
    result = dict(
        source_path=str(args.repo), source_sha=git(args.repo, "rev-parse", "HEAD"),
        source_status=git(args.repo, "status", "--porcelain"),
        run_path=str(args.run), config_differences=differences(official, recorded),
        recorded_config_source=str(log_path),
        checkpoint_path=str(checkpoint_path), checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_epoch=checkpoint.get("epoch"), checkpoint_keys=sorted(checkpoint),
        ema_present=bool(checkpoint.get("state_dict_ema")),
        actual_optimizer_steps=steps, scheduler_last_epoch=checkpoint.get("scheduler", {}).get("last_epoch"),
        pretrain_path=str(pretrain), pretrain_resolved_path=str(pretrain.resolve()),
        limitations=["Current clean checkout does not alone prove historical clean state.",
                     "Current pretrain digest is not an independently verified published digest.",
                     "No mAP inference was run by this inspector.",
                     "Official-recipe training is not subject to the later strict6000 contract."],
    )
    if pretrain.is_file():
        digest = hashlib.sha256()
        with pretrain.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["pretrain_current_sha256"] = digest.hexdigest()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
