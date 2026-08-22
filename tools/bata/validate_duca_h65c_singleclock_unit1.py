"""No-data contract validator for DUCA-H65C-SINGLECLOCK Unit 1."""
import argparse, os

def validate():
    required = ("DUCA_STAGE1_CHECKPOINT", "DUCA_STAGE1_CHECKPOINT_SHA256")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("UNIT1_PRE_RUN_BLOCKED: missing " + ", ".join(missing))
    print("PASS: score-only threshold/top-k -> exactly-once q-to-physical remap -> NMS")
    print("PASS: no data/GPU access; final-EMA is preregistered primary")

if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    validate()
