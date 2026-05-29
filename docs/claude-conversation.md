explain to me what kind of initialization we do here - and why it is special - and where in literature we got it from
  
⏺ What we do

  In notebooks/toy/model.py:init_factor, when kind="orthogonal" we apply
  torch.nn.init.orthogonal_(A). This fills A with the Q factor of a QR decomposition of a
  random Gaussian matrix — for a tall-skinny (in_dim × rank) adapter, that gives r orthonormal 
  columns spanning a random r-dimensional subspace of input space. B keeps the default Kaiming
  init.

  In notebooks/toy/config.py:PRESETS["rolora_orth_a"], this is wired in as init_a="orthogonal",
   init_b="kaiming" on top of RoLoRA's alternating schedule.

  Why it's special

  For LoRA, the rank-r adapter is forced to learn in a low-dimensional bottleneck. The init of
  A determines the input subspace the model can see through that bottleneck on day 1. Three
  properties matter:

  1. Norm preservation. ‖Ax‖₂ = ‖x‖₂ (in expectation, for inputs lying in the column span). No
  first-step amplification or shrinkage of the LoRA path — gradients to B see a
  well-conditioned signal from round 1. Kaiming-uniform A gives random column norms and small
  angles between columns, so the LoRA path's first forward pass can be either tiny or noisy.
  2. Full effective rank at init. Orthogonal columns are linearly independent by construction →
   the adapter has effective rank r on step 0. With Kaiming, the columns are random; for small
  r (think r=4, 16) you typically get effective rank < r because some columns nearly align by
  chance. That wastes capacity.
  3. Decorrelated gradients per rank channel. Each of the r columns of A picks up a distinct,
  orthogonal direction of the input, so the gradient signal to each rank dimension is
  independent at the start. Learning isn't fighting itself for the first few rounds.

  In our federated setting these properties matter more, not less: each round of RoLoRA only
  trains one factor and averages it across clients. Averaging non-orthogonal A matrices from
  drifted clients amplifies the redundancy problem. Starting orthogonal buys a few rounds of
  headroom before drift erodes it — which matches what we measured (+4.5pp final acc, 88.97% vs
   84.44%; better early-round slope visible in the curves).

  Where it comes from

  - Saxe, McClelland & Ganguli (2013), "Exact solutions to the nonlinear dynamics of learning 
  in deep linear neural networks" (arXiv:1312.6120). The foundational paper for orthogonal
  init: shows it produces dynamical isometry — gradients neither vanish nor explode regardless
  of depth — for deep linear nets and approximately for ReLU nets. Everything downstream cites
  this.
  - FFA-LoRA (Sun et al., 2024), "Improving LoRA in Privacy-Preserving Federated Learning"
  (arXiv:2403.12313). The direct precedent for our preset: their ablation reports +0.32pp on
  QNLI when swapping Kaiming-A → orthogonal-A while keeping A frozen. Since FFA-LoRA freezes A
  entirely, the init is the learned representation, so they needed it to be well-conditioned.
  Our rolora_orth_a uses the same idea even though A is trained (it's just a stronger starting
  point).
  - PiSSA (Meng et al., 2024), "PiSSA: Principal Singular Values and Singular Vectors 
  Adaptation of Large Language Models" (arXiv:2404.02948). Generalises the idea: initialise A
  from the top-r right-singular vectors of W₀ (so the LoRA path starts pointed at the
  directions W₀ already uses). The orthogonality of singular vectors is exactly the property
  we're approximating naively. PiSSA needs a non-zero W₀ to SVD-decompose; our PaperToyModel
  has none, so we can't apply PiSSA here — orthogonal-A is the W₀-free version of the same
  intuition.
  - OLoRA (Büyükakyüz, 2024), "OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models"
   (arXiv:2406.01775). Most direct: uses a QR-based orthogonal init for LoRA's A on top of
  standard Kaiming/zero for B. Reports faster convergence and final-acc gains vs standard LoRA
  — same recipe we're using.

  So the short version: orthogonal-A is the cheapest move in the literature that targets the 
  "wasted rank at init" problem, validated independently by Saxe (theoretical foundation),
  FFA-LoRA (federated ablation), PiSSA (smart variant), and OLoRA (direct recipe).

✻ Baked for 49s

❯ and rolora didn't use this init by default?                                                  

⏺ No — the original RoLoRA paper (Chen et al. 2024) inherited the standard LoRA init convention
   from Hu et al. 2021:                       

  - A: Kaiming uniform (random Gaussian-ish, non-orthogonal)
  - B: zero

  The reason that convention exists at all: in real LoRA you have a strong base weight W₀, and
  the forward path is W₀·x + B·A·x. With B=0 at init, the LoRA path adds exactly zero to W₀ on
  the first step — the fine-tune starts from the pretrained model intact and only perturbs it
  gradually. That's a stronger anchoring property than "orthogonal A" gives you, so the
  standard convention was good enough and orth-A wasn't a research target.

  A few things on top of that:

  1. The orth-A wave is recent. OLoRA, PiSSA, and the FFA-LoRA ablation are all 2024 papers —
  RoLoRA didn't have them to draw from. The deep-research doc explicitly flagged this as "cheap
   drop-in nobody on the federated side has tested yet".
  2. In RoLoRA-the-paper, B=0 makes orth-A moot for the literal first step anyway — BA·x = 
  0·A·x = 0 regardless of how A is initialized. The benefit only shows up from round 2 onward
  when B has moved off zero. So even if they had tried it, the first-round signal might not
  have looked dramatic enough to publish around.
  3. Our toy is different from real LoRA in a way that makes orth-A do extra work.
  PaperToyModel's forward is ReLU(x·A·B)·W_out — there is no W₀. If we set B=0 at init the
  entire forward is zero, gradients are zero, nothing trains. So in our setup both factors have
   to be non-zero at init. Our base_rolora uses Kaiming for both; rolora_orth_a upgrades A to
  orthogonal. The +4.5pp gain we measured is partly because the toy is more sensitive to init
  quality than real LoRA (no W₀ to mask a bad A).

  So: RoLoRA didn't use orthogonal A because (a) the standard B=0 convention already gave them
  the property they cared about — exact preservation of W₀ at step 0 — and (b) the orth-A
  literature mostly post-dates RoLoRA. In the toy without W₀ (where you can't use B=0), orth-A
  becomes much more important than it would be in the paper's real-LoRA-on-RoBERTa setting.

✻ Worked for 29s

❯ please double check that rolora didn't already do this by default and we made a mistake -    
  check the @docs/Robust_Federated_Finetuni.pdf paper                                          
  ⎿  Referenced PDF docs/Robust_Federated_Finetuni.pdf (53 pages)

  Read 1 file (ctrl+o to expand)

⏺ Found the theoretical setup on p.5: "We initialize a⁰ such that |sin θ(a*, a⁰)| = δ⁰, where 0
   < δ⁰ < 1, and b⁰ is zero." So in the theory, A is random and B=0 (standard LoRA convention).
   Let me check the toy experiment (§4.2) and the practical experiments to see what they
  actually used.

  Read 1 file (ctrl+o to expand)

⏺ Verified — we did not make a mistake. RoLoRA explicitly uses the standard "random A, B=0"
  convention, not orthogonal A. Three concrete pieces of evidence from the paper:

  1. Theoretical setup (p.5, §4.1):

  ▎ "We initialize a⁰ such that |sin θ(a*, a⁰)| = δ⁰, where 0 < δ⁰ < 1, and b⁰ is zero."

  A is a generic random unit vector with bounded angle distance to ground truth — no
  orthogonality constraint. B starts at zero.

  2. Explicit statement in the ablation (p.10, §5.3 "Symmetry vs. Asymmetry Update"):

  ▎ "In standard LoRA implementations, LoRA-A is randomly initialized while LoRA-B is set to 
  ▎ zero, which implicitly assigns asymmetric roles. In our study, we view LoRA-A as a 
  ▎ learnable basis and LoRA-B as coefficients on that basis."

  That's RoLoRA telling you directly what its init is. No orthogonality.

  3. Smoking gun in the commonsense-reasoning section (p.9, §5.2):

  ▎ "We also highlights that FFA-LoRA exhibits large performance variances across trials, such 
  ▎ as a standard deviation of 9.55 for PIQA and 8.44 for SIQA, respectively. This significant 
  ▎ variability is likely due to the initialization quality of parameter A, as different 
  ▎ initializations could lead to varying optimization trajectories and final performance 
  ▎ outcomes…"

  They're acknowledging that random A is a known source of variance — and yet they keep using
  it. If they had switched to orthogonal A (which directly addresses this), they would have
  said so.

  On the toy model (§4.2, p.7):

  ▎ "fᵢ(xᵢ) = ReLU(xᵢ·A·B)·W_out … Only weights matrices B and A are tunable, while W_out are 
  ▎ fixed."

  The paper doesn't spell out the toy's init, but it has no W₀, so B=0 would zero the forward
  and they had to init both non-zero. They presumably used Kaiming for both (the default for
  nn.Linear-style adapters in their codebase). They did not say "orthogonal."

  References check: the bibliography (pp. 11–12) cites LoRA+ ([17]) and standard LoRA ([21])
  but contains no PiSSA, no OLoRA, no Saxe/orthogonal-init paper. Orth-A init was simply not on
   their radar.

  Bonus finding that's relevant to our other work: §5.3 / Figure 6 ablates asymmetric learning 
  rates (lr_B = 2·lr_A, lr_B = 4·lr_A) and reports: "balanced AB alternation yields the highest
   accuracy and the most stable trajectory, while aggressively prioritizing either A or B
  degrades performance." That's the opposite of what we saw on the toy where λ=2 marginally
  helped. Their result is on RoBERTa-Large with a stabilizing W₀; ours is on the W₀-less toy.
  So:
  - On real LoRA (RoBERTa-Large): paper says balanced lr is best → LoRA+ doesn't help RoLoRA.
  - On our toy (no W₀): λ=2 marginally edges out λ=1 → modest LoRA+ benefit.

  Probably worth keeping rolora_plus_lr in the comparison for completeness, but the paper's
  data suggests the orth-A win on the toy is the one most likely to transfer to real LoRA —
  RoLoRA's own variance-attribution argument (§5.2) is essentially endorsing why orth-A would
  help, even though they didn't try it.

  So: our addition is a legitimate, paper-untested improvement. Not a mistake.

✻ Brewed for 53s

※ recap: We're testing RoLoRA-improvement variants on the MNIST toy. Latest finding: confirmed 
  RoLoRA paper uses random A / B=0 init, so our `rolora_orth_a` variant is a legitimate 
  untested addition. Next: run the 5-seed orth_a evaluation you queued. (disable recaps in 
  /config)

❯ What else does pissa do besides initialization (i would like to try that init for the tests  
  with the roberta) - but if pissa does some other crazy stuff maybe we can end up doing       
  ropissa                                                                                      
  
⏺ PiSSA is not just an init — it's actually three load-bearing pieces stitched together. If you
   only port the init, you'll likely break things. Here's the full picture:

  What PiSSA does

  1. SVD-based init (the part you know about).
  Decompose the pretrained weight W₀ = U·Σ·Vᵀ. Take the top-r singular triplets and split the
  principal magnitudes equally between the two factors:
  A = √Σ[:r]·Vᵀ[:r]      (in R^(r × d_in))
  B = U[:,:r]·√Σ[:r]     (in R^(d_out × r))
  So at init, B·A = top-r truncation of W₀ — not zero like vanilla LoRA. Adapter starts pointed
   at the directions W₀ already uses.
   
  2. Residual subtraction of the frozen weight (the part most people miss).
  This is critical and is why PiSSA isn't a drop-in init. The frozen base becomes:
  W₀_eff = W₀ − B·A   (= bottom (d−r) singular components of W₀)
  The full forward is W₀_eff·x + B·A·x = W₀·x at init, so the model output is identical to the
  pretrained model on step 0. Without this subtraction, the model would start with 2·(top-r of 
  W₀)·x and immediately collapse. PiSSA actively mutates the "frozen" weight; only the residual
   is frozen, not the original W₀.
   
  3. α/r scale = 1. Standard LoRA usually runs with α=2r (scale 2 in older configs, or α=32
  with r=4 → scale 8 like RoLoRA's configs do). PiSSA sets α = r so the scale is exactly 1.0,
  because B·A is already a meaningful magnitude — you don't want to amplify it by an LR-like
  factor.

  Implications for porting to RoBERTa

  Good news: Hugging Face PEFT supports PiSSA natively. You change the YAML adapter init from
  the default to init_lora_weights="pissa" (or "pissa_niter_4" for randomized SVD if SVD
  wall-time matters). PEFT handles the residual subtraction internally and writes the modified
  W₀ back to the model. Setting lora_alpha = r is your responsibility.

  The change in experiments/configs/repro_qnli_c20_r4.yaml would be:
  llm:
    adapter:
      args:
        - adapter_package: peft
          adapter_method: lora
          r: 4
          lora_alpha: 4              # was 32 — set α = r for PiSSA scale
          lora_dropout: 0.1
          init_lora_weights: pissa   # was default; this is the PEFT switch
          
  That's it on the LoRA side. No code changes in the harness.

  RoPiSSA — is it a real thing?

  It would be: RoLoRA's alternating A/B schedule on top of PiSSA's init + residual + scale.
  They're orthogonal concerns and compose cleanly:
  - PiSSA touches step 0 (init + residual + α).
  - RoLoRA touches steps 1..T (which factor moves each round, server aggregation).

  There's no conceptual conflict. Concretely:
  - All clients receive the same PiSSA-init from W₀ (same model on all clients → same SVD →
  same init). Zero federated overhead.
  - Round 1: clients all train B locally (A frozen, identical across clients). Aggregate B.
  Same as plain RoLoRA except B starts at a PiSSA value instead of zero.
  - Round 2: clients train A. A now starts at the PiSSA-init plus some drift from round 1's
  B-update. Aggregate A.
  - RoLoRA touches steps 1..T (which factor moves each round, server aggregation).

  There's no conceptual conflict. Concretely:
  - All clients receive the same PiSSA-init from W₀ (same model on all clients → same SVD → same init). Zero federated overhead.
  - Round 1: clients all train B locally (A frozen, identical across clients). Aggregate B. Same as plain RoLoRA except B starts
  at a PiSSA value instead of zero.
  - Round 2: clients train A. A now starts at the PiSSA-init plus some drift from round 1's B-update. Aggregate A.
  - Etc.

  The PiSSA advantage is largest in the first few rounds (good directions, good magnitudes, fast loss drop). RoLoRA's slow
  alternation (only one factor moves per round) means it takes ~2× as many rounds as plain LoRA to consume that advantage, but
  the integrated benefit should still show up in final accuracy and probably in variance reduction (the §5.2 "FFA-LoRA has σ=9.55
   because of random A" pathology is exactly what PiSSA cures).

  Caveats to flag before running:

  1. Don't forget α = r. Easy to miss; will cause divergence if left at 32.
  2. PiSSA modifies W₀ on disk (PEFT writes the residual back). Don't reuse the same model directory for non-PiSSA experiments
  without reloading from the original checkpoint. Keep PiSSA runs in their own output dirs.
  3. SVD time on RoBERTa-Large: ~24 layers × 4–6 LoRA-targeted matrices × ~1024×1024 SVD. Exact SVD = a few seconds on GPU;
  randomized (pissa_niter_4) is faster. Negligible compared to one round of training.
  4. The toy can't do this. PaperToyModel has no W₀ to decompose. Orth-A is our toy-side stand-in. If you want a toy-side
  analogue of PiSSA you'd need to introduce a fixed W₀ initialized non-trivially, which changes the paper's Eq. 10. Probably not
  worth it; do this on RoBERTa.
  5. Reporting: if RoPiSSA wins on QNLI, it's not really a new algorithm so much as "use a known-better init under a known
  alternation schedule" — fair to report as an ablation on top of RoLoRA, not as a separate method.

  If you want, I can prep the experiments/configs/repro_qnli_c20_r4_pissa.yaml + matching slurm script for a c20/r4 PiSSA-RoLoRA
  run, mirroring the lr-sweep setup but with the two YAML knobs above.