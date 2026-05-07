# SDNC Dataset Sources

Last verified with Hugging Face research: 2026-05-07.

SDNC should not ingest datasets as dense pretraining sludge. Each row should
become an experience:

```text
dataset row -> ModalitySample(s) -> PerceptionBus binding
            -> sensory/episodic memory -> sparse circuit update
            -> optional expert/procedure evidence
```

## Recommended First Wave

These are the best first targets because they match SDNC's architecture: tools,
interaction traces, sensory bindings, and verifiable outcomes.

| Priority | Dataset | Why it fits SDNC | Caution |
| --- | --- | --- | --- |
| 1 | `SWE-bench/SWE-bench_Verified` | 500 verified software-engineering issues with patches/tests. Useful for coding lacunes and tool-use verification. | Benchmark-sized, not a broad code corpus. |
| 1 | `openai/gsm8k` | Small MIT math word-problem dataset with step-by-step answers and calculator annotations. Excellent for arithmetic/tool routing. | English-only, grade-school domain. |
| 1 | `osunlp/Multimodal-Mind2Web` | Web-agent actions with screenshots, HTML, tasks, operation labels. Perfect for text+vision+tool-action sensory events. | 13.6 GB; research-use framing. |
| 1 | `OpenAssistant/oasst1` | Human conversation trees with ranks/reviews, Apache-2.0. Good for user communication patterns and feedback handling. | Needs filtering by language/quality. |
| 2 | `HuggingFaceM4/the_cauldron` | 50 vision-language datasets, including VQA, OCR, charts, tables, diagrams. Good sensory binding source. | 169 GB total; sub-dataset licenses vary. Start with small configs. |
| 2 | `lmms-lab/multimodal-open-r1-8k-verified` | 7.69k image+text reasoning examples, compact and useful for multimodal reasoning bindings. | Synthetic reasoning traces; use as hypotheses, not truth. |
| 2 | `McGill-NLP/WebLINX` | Real-world website navigation with multi-turn dialogue. Good for agent memory and browser-task patterns. | CC-BY-NC-SA-4.0; non-commercial constraints. |
| 2 | `nvidia/Nemotron-Terminal-Corpus` | Terminal trajectories for math, code, SWE, and shell skills. Good for tool/terminal experts. | Large; check NVIDIA terms before redistribution/commercial use. |
| 3 | `google/speech_commands` | Small CC-BY-4.0 command audio. Good first audio-sensor test. | Limited vocabulary only. |
| 3 | `google/fleurs` | Multilingual speech recognition/evaluation data. Good for multilingual audio signatures. | Read-speech bias. |
| 3 | `lmms-lab/Video-MME` | Text+video benchmark for temporal/video understanding. Useful once video keyframe binding is stronger. | Video assets are large. |

## French Speech Bootstrap

For a first "learn to talk in French" run, use instruction/response rows rather
than raw web text. On 2026-05-07 the local bootstrap corpus was downloaded to
`E:\ai\sdnc_datasets\speech_fr` and normalized into:

- `E:\ai\sdnc_datasets\speech_fr\sdnc_canonical\sdnc_speech_fr_full.parquet`
- `E:\ai\sdnc_datasets\speech_fr\sdnc_canonical\sdnc_speech_fr_smoke_2000.parquet`

The canonical file has `source` and `target` columns so SDNC can treat each row
as an interaction experience. It contains 334,556 deduplicated pairs from:

| Dataset | Role | License/terms checked |
| --- | --- | --- |
| `jpacifico/French-Alpaca-dataset-Instruct-110K` | French instruction/response base, 110k rows. | Apache-2.0. |
| `angeluriot/french_instruct` | French conversation rows from several translated instruction sources, 276k rows on HF. | MIT. |
| `OpenAssistant/oasst1` | Human-reviewed French prompter -> assistant pairs reconstructed from OASST messages. | Use repository `LICENSE`; keep provenance. |

Launch local ingestion with:

```powershell
python -m sdnc.agent.speech_training `
  --dataset E:\ai\sdnc_datasets\speech_fr\sdnc_canonical\sdnc_speech_fr_full.parquet `
  --progress-log E:\ai\sdnc_datasets\speech_fr\training_logs\speech_training.jsonl
```

`--max-rows` is optional and only for explicit smoke tests. The production
command has no row cap; progress is appended as JSONL so the run can be watched
without trusting the UI.

## Useful But Not First

- `HuggingFaceFW/fineweb-edu`: excellent educational web text, ODC-By, but
  enormous. Use streaming samples only; do not treat it as central knowledge.
- `bigcode/the-stack-smol`: useful code corpus subset, but requires access
  agreement and license/provenance handling.
- `Anthropic/hh-rlhf`: preference pairs are useful for safety/feedback, but the
  card warns that some data is offensive/upsetting and not intended directly
  for training dialogue agents without care.
- `allenai/WildChat-4.8M`: large real conversation corpus. It includes metadata
  such as country, redaction flags, moderation traces, and hashed IP; filter
  aggressively and avoid persisting unnecessary telemetry-like fields.
- `openai/webgpt_comparisons` and `openai/summarize_from_feedback`: useful for
  source-backed answer preference and summarization preference, but should be
  converted into small feedback/verdict events rather than copied wholesale.

## Ingestion Rules

1. Prefer streaming and small limits first.
2. Store raw rows only if needed; otherwise store compact sensory/procedural
   summaries and source metadata.
3. Keep dataset license, source URL, split/config, and row id in event metadata.
4. For preference datasets, store `chosen/rejected` as feedback evidence, not as
   absolute truth.
5. For agent traces, extract `(observation, action, result, verifier)` when
   present. Verifier/test outcome has more weight than generated reasoning.
6. For multimodal rows, bind modalities together through `PerceptionBus`.
7. Avoid datasets that are mostly proprietary-model distillation unless the
   license and provenance are clear.

## Source URLs

- https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified
- https://huggingface.co/datasets/openai/gsm8k
- https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K
- https://huggingface.co/datasets/angeluriot/french_instruct
- https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
- https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified
- https://huggingface.co/datasets/McGill-NLP/WebLINX
- https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus
- https://huggingface.co/datasets/google/speech_commands
- https://huggingface.co/datasets/google/fleurs
- https://huggingface.co/datasets/lmms-lab/Video-MME
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- https://huggingface.co/datasets/bigcode/the-stack-smol
- https://huggingface.co/datasets/Anthropic/hh-rlhf
- https://huggingface.co/datasets/allenai/WildChat-4.8M
- https://huggingface.co/datasets/openai/webgpt_comparisons
- https://huggingface.co/datasets/openai/summarize_from_feedback
