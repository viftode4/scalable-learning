# Validation notes — DeepSpeed Ulysses deck

Date: 2026-06-01

These notes record the paper/repo validation pass for
`deepspeed-ulysses-presentation.html`.

## Sources checked

- `paper-deepspeed-ulysses-arxiv-2309.14509.pdf`
- `README.md`
- `slides-outline.md`
- `speaker-notes.md`
- `source-friend-presentation-text.md`

## Supported claims

| Deck claim | Source evidence |
|---|---|
| Ulysses partitions inputs along the sequence dimension and uses all-to-all for attention. | Paper abstract and §1 state that DeepSpeed-Ulysses partitions input data along sequence dimension and uses all-to-all collective communication for attention. |
| Before attention, GPUs hold sequence shards; during attention, GPUs hold the full sequence for non-overlapping head subsets. | Paper §1 and §3.1: all-to-all on partitioned Q/K/V makes each GPU receive the full sequence for a non-overlapping subset of attention heads. |
| A second all-to-all restores sequence partitioning after attention. | Paper §1 and §3.1: Ulysses uses another all-to-all to gather results along attention heads while re-partitioning along sequence dimension. |
| Communication analysis is `4Nh/P` for Ulysses vs `4Nh` for Megatron-LM sequence parallelism. | Paper §3.2 gives per-link volume `4Nh/P` for DS-Sequence and `4Nh` for Megatron-LM sequence parallelism. |
| Communication stays constant when sequence length `N` and GPU count / parallelism degree `P` scale proportionally. | Paper abstract and §3.2. |
| Reported result: 1M-token sequence on a 1.2B GPT model. | Paper §4.1. |
| Reported headline gains: up to 2.5× throughput, 4× larger sequence length, over 10× communication reduction. | Paper abstract and contributions list in §1. |
| Head divisibility / head-count limit is a practical implementation constraint. | Paper mechanism assigns non-overlapping attention-head subsets; current DeepSpeed/HF implementation guidance states attention heads must be divisible by sequence-parallel size. |
| Sparse-attention throughput can be bottlenecked by local sparse-attention implementation. | Paper §4.3 notes DeepSpeed sparse throughput is bottlenecked by local sparse-attention implementation. |

## Fixes applied after validation

- Changed communication formula from `4Nd` / `4Nd/P` to paper notation `4Nh` / `4Nh/P`.
- Labeled communication as per-link / `O(N/P)` rather than generic runtime.
- Made result wording benchmark-specific: “paper-reported gains” and “up to 2.5×”.
- Softened absolute claims about one GPU and network requirements.
- Clarified all-to-all as a peer collective, not a central hub.
- Clarified slide 4’s after-state: token colors are shorthand for full sequence, while labels indicate assigned heads.
- Added the paper’s explicit sparse-attention local-kernel caveat.

## Remaining caveat

The deck is a teaching simplification. Token glyphs are symbolic: four colored blocks represent token slices / full-sequence presence, not literal tensor sizes.
