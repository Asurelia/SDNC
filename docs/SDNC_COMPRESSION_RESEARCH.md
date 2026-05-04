# SDNC Compression Research

Last verified with web research: 2026-05-03.

Goal: explore whether SDNC can approximate trillion-parameter behavior while
keeping only a 9B-class hot core in roughly 10-12 GB VRAM.

## First Constraint

Literal storage of independent trillion-scale weights in 10-12 GB is not
plausible with ordinary compression.

Example:

```text
1.6T parameters in 12 GB = ~0.06 bits / parameter
1.6T parameters in 10 GB = ~0.05 bits / parameter
```

Even 1.58-bit ternary weights would need about 316 GB for 1.6T parameters,
before scales, metadata, KV cache, activations, routing, and runtime overhead.

Therefore the plausible target is not:

```text
store all 1.6T independent weights in 12 GB
```

The plausible target is:

```text
9B hot core
+ compressed cold expert library
+ generated / reconstructed experts on demand
+ sparse activation
+ external memory and verification
+ context/KV compression
```

This can create a trillion-parameter-equivalent behavioral surface without
keeping trillion independent dense weights hot.

## Existing Pieces That Make This Plausible

### DeepSeek-V4 style sparse capacity

DeepSeek-V4-Pro reports 1.6T total parameters with 49B activated, one million
tokens of context, mixed FP4/FP8 precision, and hybrid compressed attention.
The Hugging Face checkpoint is still about 865 GB, so it is not evidence that
1.6T independent weights can fit into 10-12 GB. It is evidence for a more useful
principle:

```text
total capability can be much larger than the hot path,
but only if routing, attention, precision, and memory residency are co-designed.
```

DeepSeek-V4's public config shows 384 routed experts, 6 experts per token, one
shared expert, FP4 expert dtype, one KV head, a sliding local window, and
compressed attention ratios that reach 128:1 for heavily compressed layers.

SDNC version:

```text
large cold expert field
+ tiny active expert set
+ local sensory/context cache
+ explicit budget gate before any decode
```

### Sparse activation

DeepSeek-V3 reports 671B total parameters with 37B activated per token. The
important idea is not the exact number; it is the separation between total
capacity and active capacity.

SDNC version:

```text
cold expert atlas -> route only a few experts -> hot active subset
```

### Fine-grained experts

DeepSeekMoE improves specialization by splitting experts more finely and by
separating shared experts from routed experts.

SDNC version:

```text
shared skills: communication, planning, memory, safety
routed skills: code, math, audio, vision, file parsing, tool families
micro-experts: small latent deltas/procedures, not full models
```

### Ternary / 1.58-bit weights

BitNet b1.58 shows that a model can be trained to operate with ternary weights
`{-1, 0, +1}` rather than post-compressing a full precision model. This is
important because native low-bit training is more promising than trying to
crush an arbitrary trained model after the fact.

SDNC version:

```text
new experts should be born compression-friendly
```

### Runtime kernels for compressed weights

Bitnet.cpp shows that sub-2-bit inference can be made practical with kernels
designed for ternary LLMs. The key lesson is that compression format and compute
kernel must be designed together.

SDNC version:

```text
expert format = storage format + decompression kernel + routing metadata
```

### Vector quantization and lattice codebooks

QuIP# uses vector quantization with lattice codebooks for low-bit LLM
quantization. This supports the idea that expert weights do not need to be
stored as raw scalars; blocks can be represented as codes into learned
codebooks.

SDNC version:

```text
expert = codebook ids + sparse residuals + calibration stats
```

### KV/context compression

DeepSeek's MLA, DeepSeek-V4's CSA/HCA, and KIVI-style KV quantization show that
runtime memory is not just weights. Context/KV state can dominate. Any SDNC
trillion-scale design must compress context separately from weights.

SDNC version:

```text
raw context -> summaries/prototypes/sensory bindings
attention-like cache -> quantized or latent cache
old context -> SQLite/vector memory
```

### Hypernetworks and implicit neural representations

Hypernetworks generate weights for another network. Implicit neural
representations show data can sometimes be compressed as compact functions plus
encoded weights. This suggests an aggressive SDNC idea:

```text
do not store every expert weight;
store a latent code that reconstructs a needed expert block.
```

## Proposed Architecture: SDNC Compressed Expert Atlas

```text
Hot path in VRAM:
  6-9B sparse/ternary core
  active expert cache
  active sensory/context packet
  routing + verifier heads

Warm path in RAM:
  decoded recent experts
  codebooks
  calibration stats
  recent embeddings and procedures

Cold path on NVMe:
  compressed expert atlas
  experience packs
  sensory bindings
  rejected experiments
  source traces
```

### Expert Encoding

Each expert is stored as:

```text
expert_id
domain tags
routing prototype
codebook ids
low-rank factors
sparse residuals
scales
verifier tests
utility statistics
```

The system never assumes one source of truth. A cold expert becomes hot only if:

1. routing says it may help;
2. budget allows it;
3. local verifier/test traces support it;
4. user feedback or tool outcome does not contradict it.

### Decompression Model

Three-tier decode:

```text
L0: no decode
    use procedure/memory only

L1: sketch decode
    use tiny low-rank/prototype expert for routing or scoring

L2: hot decode
    reconstruct one small expert into RAM/VRAM for the current task
```

This is closer to game-engine asset streaming than conventional LLM loading.

## What Would Be New

The novel SDNC part is not any single trick. It is the combination:

```text
compression-friendly micro-experts
+ living lifecycle
+ sensory/context memory
+ local verification
+ on-demand reconstruction
+ hot/cold residency
+ no dense always-on transformer center
```

For the non-transformer translation of these ideas into predictive coding,
active inference, expert streaming, and replay/consolidation, see
`docs/NON_TRANSFORMER_ROADMAP.md`.

## Research Experiments

### Experiment 1: Expert Atlas Prototype

Create many tiny procedure experts from datasets and store them as:

- routing vector;
- action template;
- verifier;
- usage stats.

Measure whether SDNC solves repeated tasks with fewer tool calls over time.

### Experiment 2: Quantized Micro-Circuit Experts

Train small SDNC neural experts and store them as:

- ternary weights;
- int2/int4 weights;
- low-rank factors;
- vector-quantized blocks.

Compare memory, decode time, and task success.

### Experiment 3: Generated Experts

Train a small hypernetwork that maps:

```text
task embedding + expert metadata -> micro-expert weights
```

Measure whether generated experts beat retrieval-only procedures.

### Experiment 4: Context LOD + Sensory Binding

Use long files, images, and audio as cold data. Keep only:

- sensory signatures;
- summaries;
- prototypes;
- verifier traces.

Measure whether the system can reopen raw data only when needed.

### Experiment 5: 9B Hot Budget Simulator

Before a real 9B core exists, simulate the budget:

- cap active experts;
- cap decoded expert bytes;
- record hot RAM/VRAM estimates;
- reject routes that exceed budget.

## Current Judgement

Plausible:

- 9B-class hot core with a much larger cold library.
- Sparse experts that grow over time without keeping everything hot.
- Fast decompression of small expert blocks.
- 10-12 GB VRAM operation if the active set is tiny.
- Trillion-scale *behavioral capacity* through memory, tools, and experts.

Not plausible yet:

- A literal 1.6T independent-weight model fully resident in 10-12 GB.
- GPT-5.5-level general reasoning from compression alone.
- Real-time reconstruction of huge dense transformer blocks on consumer GPU
  without severe latency.

## Sources

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4
- https://arxiv.org/abs/2412.19437
- https://arxiv.org/abs/2401.06066
- https://arxiv.org/abs/2402.17764
- https://arxiv.org/abs/2502.11880
- https://arxiv.org/abs/2307.13304
- https://arxiv.org/abs/2401.06118
- https://arxiv.org/abs/2402.04396
- https://arxiv.org/abs/2402.02750
- https://arxiv.org/abs/2406.02528
- https://research.google/pubs/hypernetworks-2/
- https://arxiv.org/abs/2305.19185
