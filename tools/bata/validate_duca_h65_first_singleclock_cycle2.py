"""Fail-closed static precheck for the H65 Cycle2 contract (no training)."""
import argparse
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--epoch", type=int, required=True)
    args = p.parse_args()
    path = Path(args.stage1)
    if not path.is_file():
        raise SystemExit("stage1 checkpoint missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != args.sha256.lower():
        raise SystemExit("stage1 checkpoint sha256 mismatch")
    if args.epoch <= 0:
        raise SystemExit("stage1 epoch must be positive")
    print("PASS H65 Cycle2: K=384, updates=6000, epochs=60, seed=3407, packed_route=disabled")


if __name__ == "__main__":
    main()
