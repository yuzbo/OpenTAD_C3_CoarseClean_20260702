#!/usr/bin/env python3
"""Matched-dense formal entrypoint, inseparably routed through paired Stage C.

The matched arm is not permitted to materialize its own batches or advance in
an independent job.  This entrypoint therefore invokes the same paired engine
as the CT entrypoint and differs only in the audited executable identity.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.train_chronotransport_r2_stage_c import build_parser, run as run_paired


def run(args):
    return run_paired(
        args,
        entrypoint_relative="tools/bata/train_chronotransport_r2_matched_dense.py",
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
