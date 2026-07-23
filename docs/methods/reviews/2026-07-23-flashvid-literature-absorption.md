# FlashVID Literature Absorption Record

**Status:** external-method audit completed on 2026-07-23. This record is
literature/design evidence only; it is not a GeoRoute result, a TAD result, or
authorization to change the active P1 matrix.

## Identified Work and Audited Artifacts

- Paper: Ziyang Fan, Keyu Chen, Ruilong Xing, Yulin Li, Li Jiang, and Zhuotao
  Tian, *FlashVID: Efficient Video Large Language Models via Training-free
  Tree-based Spatiotemporal Token Merging*, ICLR 2026 Oral,
  arXiv:2602.08024.
- Paper: <https://arxiv.org/abs/2602.08024>
- Official implementation: <https://github.com/Fanziyang-v/FlashVID>
- Audited official-code snapshot: `983cce6e30d7a8012442bfc7557d3afa61b3572d`
  (2026-05-01). The local audit read `flashvid/token_selection.py`,
  `flashvid/utils.py`, `flashvid/llava_arch.py`, `flashvid/siglip_encoder.py`,
  `flashvid/configuration_flashvid.py`, and the repository README.

## Exact Meaning of the Headline Claim

The headline is real but must be read narrowly:

1. On **LLaVA-OneVision**, FlashVID reports a 10% visual-token retention ratio
   and an average score of 57.9 versus 58.4 for the 100% vanilla system in its
   main table: 57.9 / 58.4 = 99.1%. Thus, "99.1%" is a **relative retained
   score**, not 99.1% absolute accuracy and not TAD mAP.
2. The evidence is video question answering / video understanding on
   VideoMME, EgoSchema, LongVideoBench, MVBench, and MLVU with VLLMs. It has no
   temporal action-localization head, tIoU metric, boundary diagnostic, or
   detector-loss training.
3. The reported ratio is an aligned VLLM token-budget protocol, not a claim
   that 90% of raw video decoding, H2D, or visual-backbone work disappears.
   FlashVID's `expansion` and later inner-LLM pruning mean the count at the
   vision-side interface and the average count per LLM layer are intentionally
   different.
4. Its efficiency table leaves LLaVA-OneVision vision encoding at 785.0 ms in
   both vanilla and FlashVID rows. The speedup comes primarily from shortening
   the VLLM prefilling sequence, not from skipping the vision encoder.

## Method, in Operational Terms

1. **DySeg:** segment a video by changes in frame-mean visual features.
2. **ADTS:** within every segment and frame, greedily solve a max-min diversity
   selection problem. Pairwise cosine distance is calibrated by final vision
   layer attention and by the token's similarity to pooled video-context
   features. It preserves representative, not merely high-attention, tokens.
3. **TSTM:** for each non-ADTS token, find the most similar token in the prior
   frame. If cosine similarity exceeds a threshold, link it into a temporal
   redundancy tree and mean-aggregate each tree. A DPC-kNN spatial merge is
   then used to meet the stated budget exactly.
4. **Hybrid VLLM compression:** FlashVID can retain more visual tokens at the
   VLLM input and then apply a later FastV-style text-model pruning step. This
   is why its retention ratio is an average per-layer budget rather than a
   single raw input count.

The official code confirms this is post-encoder, inference-only compression:
`SigLipVisionTower_forward` runs the full vision tower with
`output_attentions=True` under `torch.no_grad()`, and
`LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal` applies
`flashvid_compression` only after full image features and final-layer attention
are available.

## What Transfers to GeoRoute-AdaTAD

**Accepted design insight.** The paper gives a strong counterexample to the
idea that independent spatial and temporal pruning is sufficient. It motivates
testing a selector that jointly values: (a) task relevance, (b) diversity of
retained evidence, and (c) motion-tolerant correspondence across adjacent
frames. This is compatible with GeoRoute's residual free-token branch and its
need to preserve disjoint actor, object, and scene evidence.

**Rejected direct transplant.** FlashVID cannot be presented as a direct
pre-backbone AdaTAD accelerator. It needs all dense vision tokens and a dense
final-layer attention map before it selects or merges anything, is explicitly
training-free, and has no detector-gradient, high-tIoU, or boundary-preserving
evidence. Porting it after a full VideoMAE pass would not establish the
GeoRoute primary efficiency claim.

## Controlled Follow-up, if P1 Survives

P1 remains unchanged: it first decides whether ROI plus residual selection
beats free TokenSelect under the same native-token budget and total-cost
protocol. Only after a P1 winner exists, P2 may add a **FlashVID-inspired,
not paper-exact** mechanism comparison using the existing low-cost scout only:

1. learned free residual selection;
2. relevance-plus-diversity residual selection; and
3. (2) plus adjacent-frame correspondence aggregation with explicit token
   lineage and an exact-K output contract.

That comparison must use one heavy VideoMAE forward, never recover features
from a dense heavy encoder, and report high-tIoU/boundary metrics and full
end-to-end cost. If this scout-side adaptation loses to free TokenSelect, the
FlashVID-inspired branch is removed rather than retained as decorative
complexity.
