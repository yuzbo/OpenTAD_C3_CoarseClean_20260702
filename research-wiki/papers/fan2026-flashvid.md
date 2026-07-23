---
type: paper
node_id: paper:fan2026-flashvid
title: "FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging"
status: reviewed
year: 2026
venue: ICLR 2026 Oral
authors:
  - Ziyang Fan
  - Keyu Chen
  - Ruilong Xing
  - Yulin Li
  - Li Jiang
  - Zhuotao Tian
urls:
  - https://arxiv.org/abs/2602.08024
  - https://github.com/Fanziyang-v/FlashVID
reviewed: 2026-07-23
---

# FlashVID

## Scope

FlashVID is a training-free, post-vision-encoder compression method for video
large language models. It selects and merges visual tokens for VLLM prefill;
it is not an AdaTAD implementation or a temporal action localization method.

## Verified Claim

Under the paper's LLaVA-OneVision 10% visual-token retention setting, FlashVID
reports 57.9 average score against 58.4 for the dense system. The reported
99.1% is therefore a retained relative score, not absolute accuracy, TAD mAP,
or a reduction in dense vision-encoder work.

## GeoRoute Relevance

The relevant hypothesis is to preserve task relevance, feature diversity, and
motion-tolerant correspondence jointly. A direct transplant is rejected:
FlashVID obtains dense vision features and attention first, applies no detector
gradient, and provides no high-tIoU or boundary evidence. GeoRoute P1 remains
unchanged. A clearly labelled scout-only adaptation is permitted only after a
P1 hybrid winner exists and must lose its status if it fails the matched
high-tIoU and total-cost comparison.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
