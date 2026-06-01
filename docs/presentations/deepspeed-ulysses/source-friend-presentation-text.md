# Extracted text from source-friend-presentation.pdf

Pages: 16


## Page 1

PODC '24 · PAPER PRESENTATION
DeepSpeed Ulysses
System optimizations for training extreme long-sequence Transformer models
Sam Adè Jacobs, Masahiro Tanaka, Chengming Zhang, 
Minjia Zhang, Reza Yazdani Aminabadi, Shuaiwen Leon 
Song, Samyam Rajbhandari, Yuxiong He · Microsoft
ACM PODC 2024 · Nantes, France

## Page 2

MOTIVATION
Why long sequences matter
Generative AI
Conversational AI needs long chat histories to stay coherent
Book- and chapter-level summarization spans tens to 
hundreds of thousands of words
Multimodal models reason over video, speech and 
waveforms — high-dimensional, long inputs
AI for Science
Genome language models: the human genome is 6.4 billion 
letters
Climate & weather models need sub-kilometer, 
high-resolution inputs
Healthcare models conditioned on an entire patient record
02

## Page 3

THE GAP
Four dimensions of LLM computation — one is unsolved
01 · BATCH SIZE
Data parallelism
Replicate the model, split the 
batch. Well studied.
02 · HIDDEN DIM
Tensor parallelism
Split operators within a layer 
across devices.
03 · LAYERS
Pipeline parallelism
Split the model depth-wise into 
stages.
04 · SEQUENCE LENGTH
Sequence parallelism
The dimension long-context 
training actually needs — and 
the one prior systems handle 
poorly.
03

## Page 4

PROBLEM STATEMENT
Where existing parallelism falls short
Data, tensor and pipeline parallelism cannot scale 
along the sequence dimension at all
Existing sequence-parallel methods are 
communication-inefficient — cost grows with 
sequence length
They often demand intrusive, error-prone code 
rewrites to adopt
METHOD
COMM.
PARAM. MEM.
ANY ATTENTION
ColAI-SP
O(M)
✗
✗
Megatron-SP
O(M)
✓
✓
Ulysses
O(M/P)
✓
✓
M = message size · P = number of GPUs. Lower communication complexity is better.
04

## Page 5

THE IDEA
Partition the sequence across 
GPUs — then use all-to-all to 
switch into head-parallel attention, 
and back again.
Each GPU holds N/P tokens. One all-to-all gives every GPU the full sequence 
but only a subset of attention heads; a second all-to-all returns to sequence 
parallelism for the rest of the layer.

## Page 6

CONTRIBUTION · CORE DESIGN
How Ulysses works
SEQUENCE PARALLEL
N/P tokens per GPU
All heads, a slice of the sequence. Project to 
Q, K, V.
G0
G1
G2
G3
ALL-TO-ALL
→
HEAD PARALLEL
Full sequence, fewer heads
Local full attention per head — any kernel: 
dense, sparse, FlashAttention.
h0
h1
h2
h3
ALL-TO-ALL
→
SEQUENCE PARALLEL
Back to N/P per GPU
Re-partition for MLP, layer norm and the rest 
of the block.
G0
G1
G2
G3
06

## Page 7

CONTRIBUTION · WHY IT SCALES
Communication stays constant as you scale
ULYSSES — ALL-TO-ALL
4Nd / P
Each link carries 1/P of the data. Scale N and P together → volume 
per GPU is constant.
MEGATRON / COLAI — ALL-GATHER
4Nd
Independent of P — P× larger , and keeps growing as sequences get 
longer.
Result: over 10× communication reduction versus the existing state of the art.
07

## Page 8

EVIDENCE · MICROBENCHMARK
All-to-all vs. all-gather + reduce-scatter
METHOD
COLLECTIVE
DATA MOVED
TOTAL COMM. TIME
Megatron-LM
all-gather + reduce-scatter
8 GB
34 ms
Ulysses
all-to-all
8 GB
4.9 ms
~7× faster for the same volume. Each GPU pair only exchanges its 1/P slice — no contention.
256K sequence · 4K hidden · 8× A100
08

## Page 9

CONTRIBUTION · BEYOND THE CORE
Three things that make it practical
MEMORY
ZeRO-3 integration
Model states partitioned across the combined 
data + sequence parallel groups — scale 
model size and sequence length together.
FLEXIBILITY
Attention-agnostic
Dense, sparse, causal and FlashAttention v2 
all drop in — attention stays an ordinary local 
full-attention op.
ADOPTION
Easy & portable
Minimal code changes. Open-sourced in 
DeepSpeed and Megatron-DeepSpeed, 
already in production use.
09

## Page 10

EVALUATION
Experimental setup
Cluster
ThetaGPU · Argonne NL
Node
NVIDIA DGX A100
GPUs / node
8 × A100 · 40 GB
Intra-node
NVLink / NVSwitch
Inter-node
HDR200 IB · 200 GB/s
Architecture
Standard GPT
GPT-7B
d 4096 · 32 heads · 32 layers
GPT-30B
d 6144 · 64 heads · 64 layers
10

## Page 11

RESULT · SEQUENCE SCALABILITY
1M
+
token context length on 
a 1.2B GPT model
Sequence length scales linearly with GPU count while sustaining roughly 
constant throughput across lengths — context is no longer bounded by a single 
GPU's memory.

## Page 12

RESULT · VS. STATE OF THE ART
Faster, longer, leaner than prior methods
2.5×
higher training throughput 
than the SOTA baseline
4×
longer sequences than 
existing systems
10×
less communication volume
175+
TFLOPs/GPU sustained — 
over 54% of A100 peak
Compared against Megatron-LM and ColAI-SP sequence parallelism on GPT-7B (32 GPUs) and GPT-30B (64 GPUs), dense and sparse attention.
12

## Page 13

RESULT · MEMORY
Max sequence length: ZeRO-3 alone vs. + Ulysses
ZeRO-3
256K
+ 
Ulysses
GPT-7B · 32 GPUs 
16×
ZeRO-3
256K
+ 
Ulysses
GPT-30B · 64 GPUs 
32×
ZeRO-3 alone tops out at 16K (7B) and 8K (30B) before activations overflow. ZeRO partitions weights; Ulysses partitions activations.
13

## Page 14

CASE STUDY · AI FOR SCIENCE
Training ClimaX at higher resolution
ClimaX is a Vision-Transformer climate 
model. Higher-resolution images mean 
quadratically longer token sequences — 
and quick OOM.
Baseline: 768×768 image → 108K tokens, then out of 
memory
With Ulysses: 1536×1536 image → 432K tokens
Same 64 A100 GPUs, lower peak memory
4×
longer sequences on the 
very same hardware
14

## Page 15

SUMMARY
Contributions & takeaways
A novel all-to-all sequence parallelism with communication that 
stays constant as you scale
ZeRO-3 integration for memory, plus attention-agnostic design 
and easy adoption
First to reach million-token context with sustained, near-peak 
efficiency
THE HEADLINE
2.5× faster training, 4× longer 
sequences
than the existing state-of-the-art baseline.
15

## Page 16

LEADING THE DISCUSSION
Questions to open up
1 Constant communication holds only when N and P scale together . On a fixed GPU budget with ever-growing sequences, 
does the advantage still hold?
2
Parallelism degree is capped by the number of attention heads . How limiting is that for models with few heads, or with 
grouped-query / multi-query attention?
3
How does Ulysses relate to ring-attention and context-parallel methods that appeared since — and could they be 
combined?