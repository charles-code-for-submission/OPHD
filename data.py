from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split


@dataclass
class Example:
    patient_id: str
    label: int
    age: int | None
    sex: str | None
    base_codes_diagnosis: list[str]
    base_codes_medication: list[str]
    delta_codes_diagnosis: list[str]
    delta_codes_medication: list[str]
    raw: dict

    @classmethod
    def from_dict(cls, obj: dict) -> "Example":
        return cls(
            patient_id=str(obj.get("id")),
            label=int(obj["label"]),
            age=obj.get("age"),
            sex=obj.get("sex"),
            base_codes_diagnosis=list(obj.get("base_codes_diagnosis", []) or []),
            base_codes_medication=list(obj.get("base_codes_medication", []) or []),
            delta_codes_diagnosis=list(obj.get("delta_codes_diagnosis", []) or []),
            delta_codes_medication=list(obj.get("delta_codes_medication", []) or []),
            raw=dict(obj),
        )


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_examples(path: str | Path) -> list[Example]:
    return [Example.from_dict(obj) for obj in iter_jsonl(path)]


def write_examples_jsonl(path: str | Path, examples: list[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.raw, ensure_ascii=False) + "\n")


def stratified_split_train_val(
    examples: list[Example],
    *,
    seed: int,
    val_count: int | None = None,
    val_fraction: float | None = None,
) -> tuple[list[Example], list[Example]]:
    if not examples:
        raise ValueError("No examples provided for train/val split.")
    if val_count is None and val_fraction is None:
        raise ValueError("One of val_count or val_fraction must be provided.")

    labels = [example.label for example in examples]
    if len(set(labels)) < 2:
        raise ValueError("Train/val split requires both positive and negative examples.")

    if val_count is not None:
        val_count = int(val_count)
        if val_count <= 0 or val_count >= len(examples):
            raise ValueError(f"val_count must be in [1, {len(examples) - 1}], got {val_count}.")
        train_size = len(examples) - val_count
    else:
        assert val_fraction is not None
        if not 0.0 < float(val_fraction) < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
        train_size = 1.0 - float(val_fraction)

    train_examples, val_examples = train_test_split(
        examples,
        train_size=train_size,
        stratify=labels,
        random_state=int(seed),
    )
    return list(train_examples), list(val_examples)


def shuffle_examples(examples: list[Example], *, seed: int) -> list[Example]:
    rows = list(examples)
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    return rows


def chunk_examples(examples: list[Example], *, batch_size: int) -> list[list[Example]]:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
