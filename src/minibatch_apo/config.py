"""Configuration models shared by the public study runners."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SUBJECTS = [
    "Algebra",
    "Number Theory",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Prealgebra",
    "Precalculus",
]
DEFAULT_SEED_PROMPT = (
    "You are a helpful assistant. Answer the question. Put your final answer in the format '### <answer>'"
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "nlile/hendrycks-MATH-benchmark"
    levels: tuple[int, ...] = (4, 5)
    train_size: int = 100
    val_size: int = 100
    dataset_seed: int = 42
    subjects: tuple[str, ...] = tuple(DEFAULT_SUBJECTS)


@dataclass(frozen=True)
class LoggingConfig:
    include_prompt_text: bool = False
    include_reflective_dataset: bool = False
    include_raw_model_outputs: bool = False
    consent_to_store_sensitive_text: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    kind: str = "diversity"
    diversity_levels: tuple[int, ...] = (1, 2, 4, 7)
    minibatch_sizes: tuple[int, ...] = (3, 10, 50)
    augmentation_arms: tuple[str, ...] = ("baseline", "duplicate", "real")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    base_train_size: int = 10
    augmented_train_size: int = 100
    base_minibatch_size: int = 10
    augmented_minibatch_size: int = 50
    pools_dir: str | None = None


@dataclass(frozen=True)
class StudyConfig:
    task_lm: str = "openai/gpt-4.1-mini"
    reflection_lm: str = "openai/gpt-4.1"
    max_metric_calls: int = 1000
    output_dir: str = "outputs/prompt_optimization_study"
    seed_prompt: str = DEFAULT_SEED_PROMPT
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StudyConfig:
        dataset = DatasetConfig(**_coerce_collection_fields(raw.get("dataset", {}), {"levels", "subjects"}))
        logging = LoggingConfig(**raw.get("logging", {}))
        experiment = ExperimentConfig(
            **_coerce_collection_fields(
                raw.get("experiment", {}),
                {"diversity_levels", "minibatch_sizes", "augmentation_arms", "seeds"},
            )
        )
        config = cls(
            task_lm=raw.get("task_lm", cls.task_lm),
            reflection_lm=raw.get("reflection_lm", cls.reflection_lm),
            max_metric_calls=raw.get("max_metric_calls", cls.max_metric_calls),
            output_dir=raw.get("output_dir", cls.output_dir),
            seed_prompt=raw.get("seed_prompt", cls.seed_prompt),
            dataset=dataset,
            logging=logging,
            experiment=experiment,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.kind not in {"diversity", "factorial", "augmentation"}:
            raise ValueError("experiment.kind must be diversity, factorial, or augmentation")
        if not self.task_lm or not self.reflection_lm:
            raise ValueError("task_lm and reflection_lm must be non-empty LiteLLM model strings")
        if self.max_metric_calls < 1 or self.dataset.train_size < 1 or self.dataset.val_size < 1:
            raise ValueError("rollout and dataset sizes must be positive")
        if any(level < 1 for level in self.experiment.diversity_levels):
            raise ValueError("diversity levels must be positive")
        if any(level > len(self.dataset.subjects) for level in self.experiment.diversity_levels):
            raise ValueError("a diversity level exceeds the configured subject list")
        if any(size < 1 for size in self.experiment.minibatch_sizes):
            raise ValueError("minibatch sizes must be positive")
        stores_text = (
            self.logging.include_prompt_text
            or self.logging.include_reflective_dataset
            or self.logging.include_raw_model_outputs
        )
        if stores_text and not self.logging.consent_to_store_sensitive_text:
            raise ValueError(
                "Set logging.consent_to_store_sensitive_text to true before storing model or dataset text."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_collection_fields(raw: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    result = dict(raw)
    for field_name in fields:
        if field_name in result:
            result[field_name] = tuple(result[field_name])
    return result


def load_config(path: Path) -> StudyConfig:
    """Load a JSON study configuration; JSON keeps the public dependency set minimal."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The study configuration must be a JSON object.")
    return StudyConfig.from_dict(raw)
