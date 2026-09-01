#!/usr/bin/env bash
set -euo pipefail
scontrol hold 1179533
squeue -j 1179517,1179533,1179602 -o '%i|%j|%T|%M|%R'
