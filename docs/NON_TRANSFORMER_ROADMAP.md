# SDNC Non-Transformer Research Roadmap

Last verified with web research: 2026-05-04.

This document translates current transformer-era compression and agent research
into a SDNC-native path. The goal is not to rebuild a transformer with different
words. The goal is to keep useful engineering principles while preserving the
SDNC center:

```text
sensory signals
+ sparse local circuits
+ persistent memory
+ verified experts
+ tools
+ local plasticity
```

## Core Distinction

Many recent efficiency techniques are attached to transformers because that is
where most compute and research money currently lives. SDNC should not copy the
transformer architecture. It should extract portable principles:

| Transformer-era technique | Do not copy | SDNC translation |
| --- | --- | --- |
| MoE routing | token-level transformer experts as the core | sparse selection of micro-circuits, procedures, and skills |
| QMoE/sub-1-bit MoE compression | a trillion-param transformer resident in VRAM | compressed cold expert atlas with tiny hot active set |
| BitNet/LittleBit | transformer blocks trained for next-token prediction | compression-friendly SDNC experts born low-bit/ternary |
| Neural Weight Compression | reconstruct full transformer layers | neural codec for SDNC expert payloads and circuit deltas |
| KV cache compression | giant attention context | compressed belief state, sensory bindings, and context LOD |
| Offloading/mmap | slow full-model execution | asset streaming for cold experts, codebooks, and traces |

## What Current Research Actually Supports

### QMoE-style expert compression

QMoE shows that trillion-parameter MoE models can be compressed below one bit
per parameter with custom formats and GPU decode kernels. The concrete reported
example is SwitchTransformer-c2048, 1.6T parameters, compressed to less than
160 GB, with under 5 percent runtime overhead relative to ideal uncompressed
inference on multi-GPU commodity servers.

Important SDNC conclusion:

```text
QMoE does not make 1.6T fit in 10-12 GB.
It proves that expert redundancy + custom kernels + sparse routing matter.
```

SDNC adaptation:

```text
expert atlas = codebook ids + low-rank factors + sparse residuals
             + routing prototype + verifier tests + utility history
```

### Sub-1-bit and latent-factor quantization

LittleBit targets quantization as low as 0.1 bits per weight using latent
factorization, binarized factors, and residual compensation. BitNet b1.58 shows
that native ternary training can be more promising than crushing a full-precision
model after the fact.

Important SDNC conclusion:

```text
do not quantize SDNC experts after they are born;
make them compression-friendly from birth.
```

SDNC adaptation:

```text
new expert/circuit payloads prefer:
  ternary or int2/int4 weights
  low-rank factors
  sparse deltas
  tiny procedure programs
  verifier traces
```

### Neural Weight Compression

Neural Weight Compression treats model weights as a data modality and trains a
neural codec to compress/reconstruct them. Reported gains are strongest in the
4-6 bit range, not magic 0.05 bit storage.

Important SDNC conclusion:

```text
a learned decompressor is plausible,
but it must decode small expert blocks, not the whole brain.
```

SDNC adaptation:

```text
DecompressorEngine:
  input: task embedding + expert metadata + latent code
  output: tiny expert sketch or hot expert payload
  gate: budget + verifier + expected usefulness
```

### DeepSeek-style sparse capacity

DeepSeek-V4-Pro reports 1.6T total parameters with 49B activated, one million
token context, mixed FP4/FP8 precision, and hybrid compressed attention. This is
still a transformer/MoE model. SDNC should not copy the training recipe, teacher
pipeline, or token-prediction objective.

Important SDNC conclusion:

```text
total stored capacity may be much larger than hot active capacity,
but routing, compression, memory, and budget must be co-designed.
```

SDNC adaptation:

```text
cold capacity: many experts and memory packs
hot capacity: only what the current observation needs
budget rule: refuse to decode what cannot be tested or used now
```

## SDNC-Native Architecture Target

```text
Hot path: 10-12 GB VRAM target
  sparse local core
  active micro-experts
  predictive router
  verifier heads
  current sensory/context packet

Warm path: RAM
  decoded recent experts
  codebooks
  routing prototypes
  recent embeddings
  source reliability stats

Cold path: NVMe + SQLite
  compressed expert atlas
  sensory bindings
  experience packs
  rejected experiments
  tool traces
  file/dataset queue
```

## Predictive Coding as the Non-Transformer Center

SDNC should predict more than the next word. It should predict:

- the next sensory state;
- the likely user intent;
- the outcome of a tool call;
- which expert will be useful;
- how confident the answer should be;
- which file/memory/source is likely to reduce uncertainty.

The learning signal is prediction error:

```text
prediction -> observation/tool result/user feedback -> surprise
surprise -> local circuit update + memory event + possible experiment
```

This keeps learning local. Only the active circuits, relevant memories, and
candidate experts are updated.

## Active Inference Loop

SDNC should not only answer. When uncertain, it should act to reduce
uncertainty:

```text
detect lacune
-> generate hypotheses
-> choose low-cost experiment/tool action
-> predict outcome
-> run experiment
-> compare result
-> consolidate or reject
```

A practical expected-free-energy score can be approximated without pretending to
solve full neuroscience:

```text
expected_value =
  uncertainty_reduction
  + task_utility
  + user_relevance
  - cost
  - risk
  - memory_pollution_penalty
```

## Decompression Like Game Asset Streaming

The plausible SDNC version of "real-time decompression" is not a massive tensor
teleporting into VRAM. It is asset streaming:

```text
L0: no neural decode
    use procedure, memory, or tool directly

L1: sketch decode
    low-rank/prototype expert for routing, scoring, or simulation

L2: hot decode
    reconstruct one small expert block for a bounded task
```

Predictive routing can prefetch likely L1/L2 experts before they are needed.
Unexpected observations cancel or replace the prefetch.

## Human-Like Improvement Cycle

The target loop is:

```text
observe
-> predict
-> act/tool/search
-> compare
-> update locally
-> replay during idle
-> prune or strengthen experts
-> refine routing
```

This maps to existing SDNC modules:

- `learning_gaps`: lacune detection;
- `source_stats`: source reliability;
- `self_improvement_experiments`: sandbox outcomes;
- `experts`: lifecycle, utility, hot/cold status;
- `sensory_bindings`: compact multimodal memory;
- `sync_events`: append-only trace for monitoring.

## Implementation Milestones

1. Predictive error signals
   - add predicted tool outcome, predicted confidence, and observed outcome;
   - persist surprise/error per interaction.

2. Predictive expert routing
   - estimate which experts will be useful before decoding/heating them;
   - log false positives and false negatives.

3. Compressed expert payloads
   - store experts as procedure, low-rank sketch, sparse residual, or codebook
     references;
   - add byte budget and decode level to every expert.

4. Decompressor engine prototype
   - start with deterministic codebook/low-rank reconstruction;
   - later test a learned neural codec for small SDNC experts.

5. Replay / sleep phase
   - replay high-salience episodes while idle;
   - consolidate repeated successes;
   - retire experts with repeated negative evidence.

6. Active inference planner
   - choose searches/tools/experiments based on expected uncertainty reduction;
   - penalize expensive or unverifiable actions.

7. Neuro-symbolic memory
   - extract verified rules from repeated traces;
   - bind rules to experts and sensory prototypes;
   - keep provenance and rejection evidence.

## Red Lines

- No transformer at the center.
- No teacher/distillation loop as the main learning mechanism.
- No unverified source becomes memory truth.
- No expert is promoted because it sounds plausible.
- No dense always-on context.
- No memory ingestion without salience, provenance, and rejection path.

## Sources

- QMoE: https://arxiv.org/abs/2310.16795
- LittleBit: https://arxiv.org/abs/2506.13771
- Neural Weight Compression: https://arxiv.org/abs/2510.11234
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- Additive Quantization / AQLM: https://arxiv.org/abs/2401.06118
- DeepSeek-V4-Pro: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- Predictive Coding and Active Inference empirical review:
  https://pubmed.ncbi.nlm.nih.gov/38030100/
- Active Inference and HCI: https://arxiv.org/abs/2412.14741
- Expected Free Energy planning: https://arxiv.org/abs/2504.14898
- Predictive Coding approximates backprop:
  https://arxiv.org/abs/2006.04182
