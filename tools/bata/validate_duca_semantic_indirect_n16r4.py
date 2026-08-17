"""Fail-closed static validator. PRE_RUN performs no data, model, GPU, or Slurm access."""
import os, sys

def validate(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("PRECHECK_ONLY") != "1" and not argv:
        raise SystemExit("PRECHECK_ONLY=1 required; future pilot/full argv must be declared")
    if any(x in " ".join(argv).lower() for x in ("download", "infer", "train", "metric", "gpu", "slurm")):
        raise SystemExit("fail-closed: undeclared execution is forbidden")
    return {"status": "precheck_only", "data_access": False, "execution": False}

if __name__ == "__main__":
    print(validate())
