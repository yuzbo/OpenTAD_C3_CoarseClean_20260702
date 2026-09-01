#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

SNAPSHOT=/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_e30db0f_20260723
EXPECTED=e30db0f
BRANCH=codex/duca-t1-trainfree-lazyfix-20260723
REPO=https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git

if [ ! -d "${SNAPSHOT}/.git" ]; then
  git clone --depth 1 --branch "${BRANCH}" "${REPO}" "${SNAPSHOT}"
fi
cd "${SNAPSHOT}"
ACTUAL="$(git rev-parse HEAD)"
case "${ACTUAL}" in
  ${EXPECTED}*) ;;
  *) echo "unexpected HEAD: ${ACTUAL}" >&2; exit 1 ;;
esac
test -z "$(git status --porcelain)"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
python -m py_compile tools/bata/train_lowres_action_probe.py opentad/models/duca/acquisition.py
python -m pytest \
  tests/test_lowres_action_probe.py::test_mobilenetv3_probe_uses_pretrained_cnn_and_outputs_frame_logits \
  tests/test_lowres_action_probe.py::test_frozen_mobilenet_semantic_prior_preserves_multiclass_evidence \
  tests/test_duca_t1_trainfree.py -q
test -z "$(git status --porcelain)"
printf 'SNAPSHOT=%s\nHEAD=%s\n' "${SNAPSHOT}" "${ACTUAL}"
