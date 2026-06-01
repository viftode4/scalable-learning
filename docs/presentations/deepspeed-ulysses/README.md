# Group 7 paper presentation — DeepSpeed Ulysses

Paper for Group 7 paper presentation on Tuesday 9 June 2026 at 14:30.

## Paper

**DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**

- Authors: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, Yuxiong He
- arXiv: <https://arxiv.org/abs/2309.14509>
- PDF: <https://arxiv.org/pdf/2309.14509>
- Microsoft Research page: <https://www.microsoft.com/en-us/research/publication/deepspeed-ulysses-system-optimizations-for-enabling-training-of-extreme-long-sequence-transformer-models/>
- Local PDF: [`paper-deepspeed-ulysses-arxiv-2309.14509.pdf`](paper-deepspeed-ulysses-arxiv-2309.14509.pdf)

## Presentation deliverables

- Visual state-based HTML deck: [`deepspeed-ulysses-presentation.html`](deepspeed-ulysses-presentation.html)
- Slide-by-slide story: [`slides-outline.md`](slides-outline.md)
- Speaker prep notes: [`speaker-notes.md`](speaker-notes.md)
- Paper validation notes: [`validation-notes.md`](validation-notes.md)
- June 2 required-reading questions: [`questions-for-june-2-presentations.md`](questions-for-june-2-presentations.md)
- Local PDFs for the June 2 papers: [`required-reading/june-2-papers/`](required-reading/june-2-papers/)
- Original downloaded draft PDF: [`source-friend-presentation.pdf`](source-friend-presentation.pdf)
- Extracted draft text: [`source-friend-presentation-text.md`](source-friend-presentation-text.md)

## One-line thesis

DeepSpeed-Ulysses enables training Transformer models with extreme sequence lengths by partitioning along the sequence dimension and using all-to-all communication for attention, so communication volume stays constant when sequence length and device count scale proportionally.

## Presentation prep checklist

- [ ] Assign presenters and timing.
- [x] Build slide outline: problem → why existing parallelism fails → Ulysses idea → communication analysis → experiments → limitations.
- [x] Redo key figures as in-deck visual animations for sequence partitioning and all-to-all attention.
- [x] Prepare discussion prompts in the final claims/critique slide.
- [ ] Rehearse once with timer; the deck is 9 slides / 18 animation states.

## Presentation slot


Schedule: **Tuesday 9 June 2026, 14:30, Group 7**.

Because Group 7 presents on 9 June, we must read every paper presented on **Tuesday 2 June** and prepare one critical question per paper. See [`questions-for-june-2-presentations.md`](questions-for-june-2-presentations.md).

Target a **9-10 minute presentation + 4-5 minute Q&A**. The deck is intentionally short: **9 slides / 18 animation states**. Use right arrow to advance animation states first, then slides.

| Section | Time | Owner | What to cover |
|---|---:|---|---|
| Motivation | 1.0 min | TBD | Why long sequences matter and why ordinary Transformer training breaks at extreme sequence length. |
| Background bottleneck | 1.5 min | TBD | Attention memory/communication bottlenecks; why data/tensor/pipeline parallelism alone is insufficient. |
| Ulysses mechanism | 3.0 min | TBD | Sequence partitioning, all-to-all before/after attention, and how each GPU computes attention heads for full sequence context. |
| System/complexity argument | 1.5 min | TBD | Constant per-link communication when sequence length scales with device count; compatibility with other DeepSpeed parallelism. |
| Evaluation | 1.5 min | TBD | Long-sequence scaling results, throughput/memory trends, and headline supported sequence lengths. |
| Critique + discussion | 1.0 min | TBD | Assumptions, all-to-all network dependence, comparison gaps, and when this method is/isn't the right tool. |

## Immediate next action

Open `deepspeed-ulysses-presentation.html` locally and rehearse the core mechanism twice:

> tokens → heads → attention → tokens

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
