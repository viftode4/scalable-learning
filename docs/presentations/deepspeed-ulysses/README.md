# Group paper presentation — DeepSpeed Ulysses

Paper for the group presentation.

## Paper

**DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**

- Authors: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, Yuxiong He
- arXiv: <https://arxiv.org/abs/2309.14509>
- PDF: <https://arxiv.org/pdf/2309.14509>
- Microsoft Research page: <https://www.microsoft.com/en-us/research/publication/deepspeed-ulysses-system-optimizations-for-enabling-training-of-extreme-long-sequence-transformer-models/>
- Local PDF: [`paper-deepspeed-ulysses-arxiv-2309.14509.pdf`](paper-deepspeed-ulysses-arxiv-2309.14509.pdf)

## One-line thesis

DeepSpeed-Ulysses enables training Transformer models with extreme sequence lengths by partitioning along the sequence dimension and using all-to-all communication for attention, so communication volume stays constant when sequence length and device count scale proportionally.

## Presentation prep checklist

- [ ] Assign presenters and timing.
- [ ] Build slide outline: problem → why existing parallelism fails → Ulysses idea → communication analysis → experiments → limitations.
- [ ] Extract/redo key figures for sequence parallelism and all-to-all attention.
- [ ] Prepare 2-3 discussion questions.

## Tomorrow plan

Target a **10-12 minute presentation + 5-6 minute Q&A**.

| Section | Time | Owner | What to cover |
|---|---:|---|---|
| Motivation | 1.5 min | TBD | Why long sequences matter and why ordinary Transformer training breaks at extreme sequence length. |
| Background bottleneck | 2 min | TBD | Attention memory/communication bottlenecks; why data/tensor/pipeline parallelism alone is insufficient. |
| Ulysses mechanism | 3 min | TBD | Sequence partitioning, all-to-all before/after attention, and how each GPU computes attention heads for full sequence context. |
| System/complexity argument | 2 min | TBD | Constant per-GPU communication when sequence length scales with device count; compatibility with other DeepSpeed parallelism. |
| Evaluation | 2 min | TBD | Long-sequence scaling results, throughput/memory trends, and headline supported sequence lengths. |
| Critique + discussion | 1.5 min | TBD | Assumptions, all-to-all network dependence, comparison gaps, and when this method is/isn't the right tool. |

## Immediate next files to create

- `slides-outline.md` — final slide-by-slide story.
- `speaker-notes.md` — presenter script and transitions.
- `figures/` — screenshots or recreated diagrams from the paper, with page references.
- `discussion-questions.md` — questions for Q&A / class discussion.

## Working split suggestion

- **Presenter 1:** motivation + bottleneck.
- **Presenter 2:** Ulysses mechanism + system argument.
- **Presenter 3:** evaluation + critique/discussion.

## BibTeX

```bibtex
@article{jacobs2023deepspeedulysses,
  title = {DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models},
  author = {Jacobs, Sam Ade and Tanaka, Masahiro and Zhang, Chengming and Zhang, Minjia and Song, Shuaiwen Leon and Rajbhandari, Samyam and He, Yuxiong},
  journal = {arXiv preprint arXiv:2309.14509},
  year = {2023},
  doi = {10.48550/arXiv.2309.14509},
  url = {https://arxiv.org/abs/2309.14509}
}
```
