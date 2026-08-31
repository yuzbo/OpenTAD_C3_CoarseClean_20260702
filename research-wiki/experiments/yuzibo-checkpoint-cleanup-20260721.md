---
id: exp:yuzibo-checkpoint-cleanup-20260721
type: experiment
status: tested
updated: 2026-07-21
---

# Remote checkpoint retention audit

## Scope

The complete `/data/run01/sczc063/yuzibo` tree was indexed without following
symlinks. Dataset, pretrained-weight, environment, package, cache and Git
metadata directories were excluded. Training records were recognized only by
numeric `epoch`, `iter` or `step` checkpoint names.

## Retention contract

- Keep exactly one newest structurally usable record per independent
  checkpoint directory.
- Validate PyTorch ZIP central directory, every entry CRC and `data.pkl`.
- For a non-ZIP legacy candidate, require successful CPU `torch.load`.
- If the newest file is corrupt, fall back to the newest earlier valid file.
- Remove zero-byte files and test fixtures; preserve logs, configs, metrics,
  source trees, datasets, pretrained weights and the external `best_epoch.pth`
  symlink.

## Result

- Initial numeric records: 909 files, 463,926,982,932 bytes.
- Deleted: 689 files, 334,791,638,367 bytes (about 311.8 GiB).
- Retained: 220 files in 220 groups, 129,135,344,565 bytes; no group contains
  more than one numeric checkpoint.
- The shared `/data` mount changed from 100% use/0 available to 87% use with
  about 310 GiB available at final verification.
- Corruption handling found a truncated 320 MiB DUCA debug `epoch_0.pth`, a
  zero-byte PCOTMRAS `epoch_9.pth` that fell back to valid `epoch_7.pth`, and
  91 zero-byte/test-fixture records totaling 2,609 bytes.

## Audit artifacts

- Consolidated manifest:
  `/data/run01/sczc063/yuzibo/cleanup_manifests/checkpoint_cleanup_consolidated_20260721_132157.json`
- Manifest SHA-256:
  `a06d3062a1fc2f8ec9d1ef336271f688368dfe2c788fea6933d8cc9e1a04b60a`
- Retained index:
  `/data/run01/sczc063/yuzibo/cleanup_manifests/checkpoint_cleanup_consolidated_20260721_132157.retained.tsv`
- Retained-index SHA-256:
  `b0c1cd7c78e9d9825094e06ea7e45002bd4bfd4fcf2d215aac3bb829fe91f006`
- Multi-record batch manifest:
  `/data/run01/sczc063/yuzibo/cleanup_manifests/checkpoint_cleanup_global_multi_20260721_131016.json`
- Multi-record manifest SHA-256:
  `27abd07e6f581bf0c15a4dfb99d1223a83429fea831535ad54343da8f3c8e788`

## Evidence boundary

Checkpoint recoverability does not establish terminal mAP. The old DUCA four
arms remain infrastructure-interrupted diagnostics until resumed under their
exact frozen commit and completed protocol.
