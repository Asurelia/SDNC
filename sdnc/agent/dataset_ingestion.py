"""Dataset ingestion for SDNC.

Datasets are treated as streams of experiences. The ingestor learns from each
sample through the same sparse local loop and can also emit compressed
experience packs for fast later recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterable

from sdnc.agent.experience_packs import ExperiencePack, ExperiencePackBuilder
from sdnc.agent.multimodal import LocalMultimodalEncoder, samples_from_record


@dataclass(frozen=True)
class IngestionReport:
    """Summary of a dataset ingestion run."""

    source: str
    records_seen: int
    samples_seen: int
    episodes_stored: int
    pack: ExperiencePack | None = None
    errors: list[str] = field(default_factory=list)


class DatasetIngestor:
    """Convert local or Hugging Face rows into SDNC observations."""

    def __init__(self, system):
        self.system = system
        self.encoder = LocalMultimodalEncoder(system.config.input_dim)

    def ingest_records(
        self,
        records: Iterable[dict[str, Any]],
        source: str,
        limit: int | None = None,
        text_columns: Iterable[str] | None = None,
        label_columns: Iterable[str] | None = None,
        build_pack: bool = True,
    ) -> IngestionReport:
        records_seen = 0
        samples_seen = 0
        episodes_stored = 0
        errors: list[str] = []
        builder = ExperiencePackBuilder(name=f"pack:{source}") if build_pack else None

        for index, record in enumerate(islice(records, limit)):
            records_seen += 1
            try:
                samples = samples_from_record(
                    record,
                    source=source,
                    sample_id=str(index),
                    text_columns=text_columns,
                    label_columns=label_columns,
                )
                for sample in samples:
                    samples_seen += 1
                    encoded = self.system.multimodal_encoder.encode_sample(sample)
                    if builder is not None:
                        builder.add(encoded)
                result = self.system.observe(
                    samples,
                    context={"dataset_row": index, "dataset_source": source},
                    learn=True,
                    use_tools=False,
                )
                if result.episode_id:
                    episodes_stored += 1
            except Exception as exc:
                errors.append(f"row {index}: {type(exc).__name__}: {exc}")

        pack = builder.build({"records_seen": records_seen, "samples_seen": samples_seen}) if builder else None
        return IngestionReport(
            source=source,
            records_seen=records_seen,
            samples_seen=samples_seen,
            episodes_stored=episodes_stored,
            pack=pack,
            errors=errors,
        )

    def ingest_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        name: str | None = None,
        limit: int = 100,
        streaming: bool = True,
        text_columns: Iterable[str] | None = None,
        label_columns: Iterable[str] | None = None,
    ) -> IngestionReport:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional `datasets` extra to ingest Hugging Face datasets."
            ) from exc

        kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
        dataset = load_dataset(dataset_name, name, **kwargs) if name else load_dataset(dataset_name, **kwargs)
        return self.ingest_records(
            dataset,
            source=f"huggingface:{dataset_name}",
            limit=limit,
            text_columns=text_columns,
            label_columns=label_columns,
            build_pack=True,
        )
