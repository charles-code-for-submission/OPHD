from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from torch.utils.data import IterableDataset as TorchIterableDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainerCallback

import research3.scorer_phase as scorer_phase
from research3.data import Example, chunk_examples
from research3.generator import ClinicalReasoningPrompter, DistillGenerator
from research3.train import (
    artifact_lookup_roots,
    artifact_output_root,
    ensure_score_bias,
    format_command,
    install_run_log,
    make_reward_config,
    resolve_testslice_dir,
)


warnings.filterwarnings(
    "ignore",
    message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
)


class _DropLogMessageFilter(logging.Filter):
    def __init__(self, blocked_substrings: list[str]) -> None:
        super().__init__()
        self.blocked_substrings = tuple(str(item) for item in blocked_substrings)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(substring in message for substring in self.blocked_substrings)


def suppress_external_log_noise() -> None:
    blocked_substrings = [
        "LOAD REPORT",
        "`torch_dtype` is deprecated! Use `dtype` instead!",
    ]
    loggers = [
        logging.getLogger("transformers.utils.loading_report"),
        logging.getLogger("transformers.modeling_utils"),
        logging.getLogger("transformers.configuration_utils"),
        logging.getLogger("transformers.pipelines"),
    ]
    for logger in loggers:
        logger.addFilter(_DropLogMessageFilter(blocked_substrings))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate a loaded stage2 generator with a loaded stage1 scorer.",
    )
    data_group = ap.add_argument_group("Data and runtime")
    data_group.add_argument("--data_name", type=str, default="ad")  # Dataset name; selects prompt and scorer-prompt variants.
    data_group.add_argument("--train_jsonl", type=str, default="")  # Training jsonl used only when --train_scorer_on_reasoning is enabled.
    data_group.add_argument("--test_jsonl", type=str, default=None)  # Explicit test jsonl path; mutually exclusive with --testslice.
    data_group.add_argument("--testslice", type=int, nargs="*", default=[])  # Resolve test jsonl from slice/fold ids; takes precedence over --test_jsonl.
    data_group.add_argument("--output_dir", type=str, required=True)  # Output directory for metrics, scores, generated reasoning, and retrained scorer artifacts.
    data_group.add_argument("--device", type=int, default=0)  # CUDA device id used by both generator and scorer.
    data_group.add_argument("--seed", type=int, default=7)  # Global random seed; also the fallback for data_seed.
    data_group.add_argument("--data_seed", type=int, default=None)  # Data-stream / pair-sampling seed; falls back to seed when omitted.

    reasoning_group = ap.add_argument_group("Reasoning cache and subset control")
    reasoning_group.add_argument("--ehr_only_ablation", action=argparse.BooleanOptionalAction, default=False)  # Force both train and eval into EHR-only mode; no reasoning is loaded or generated, and no student is needed.
    reasoning_group.add_argument("--reasoning_jsonl", type=str, default="")  # Test-set reasoning file; used directly in normal mode, and in fixed-test-subset mode its ids define the subset whenever the file already exists.
    reasoning_group.add_argument("--train_reasoning_jsonl", type=str, default="")  # Training-set reasoning file; used directly in normal mode, and in fixed-train-subset mode its ids define the subset whenever the file already exists.
    reasoning_group.add_argument("--train_cache_note", type=str, default="")  # Shared note name for reusable full-train caches, or for paired fixed train/test caches when subset sampling is enabled.
    reasoning_group.add_argument("--train_cache_root", type=str, default="eval_generated_train_cache")  # Root directory used together with train_cache_note.
    reasoning_group.add_argument("--train_subset_ratio", type=float, default=1.0)  # If <1.0, sample a fixed train subset into <data_name>_train_cache; if 1.0, pregenerate/use the full train cache upfront.
    reasoning_group.add_argument("--test_subset_ratio", type=float, default=1.0)  # If <1.0, sample a fixed test subset into <data_name>_test_cache; if 1.0, keep the original testslice order and ids.
    reasoning_group.add_argument("--subset_pos_ratio", type=float, default=-1.0)  # Optional target positive ratio for the sampled train subset; negative keeps the source prevalence. Ignored when train_subset_ratio=1.
    reasoning_group.add_argument("--reasoning_pid_key", type=str, default="id")  # Patient-id field name inside reasoning jsonl rows.
    reasoning_group.add_argument("--reasoning_text_key", type=str, default="reasoning")  # Reasoning-text field name inside reasoning jsonl rows.

    generator_group = ap.add_argument_group("Stage2 generator for reasoning")
    generator_group.add_argument("--stage2_student_dir", type=str, default="")  # Stage2 student / checkpoint path used when reasoning must be generated.
    generator_group.add_argument("--stage2_student_use_base_model", action=argparse.BooleanOptionalAction, default=False)  # Use only the student's base LLM for reasoning generation, without loading the LoRA adapter.
    generator_group.add_argument("--generator_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")  # Base generator model used for reasoning generation, including the raw-base ablation.
    generator_group.add_argument("--generator_torch_dtype", type=str, default="bfloat16")  # Torch dtype used when loading the generator.
    generator_group.add_argument("--generator_max_prompt_tokens", type=int, default=2048)  # Maximum generator prompt length; longer prompts are left-truncated.
    generator_group.add_argument("--eval_generator_max_new_tokens", type=int, default=256)  # Maximum newly generated tokens for eval/test/train reasoning generation.
    generator_group.add_argument("--generation_batch_size", type=int, default=8)  # Batch size for generator-side reasoning generation.
    generator_group.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)  # Whether reasoning generation uses sampling; False keeps generation deterministic/near-greedy.
    generator_group.add_argument("--temperature", type=float, default=1.0)  # Sampling temperature when do_sample=True.
    generator_group.add_argument("--top_p", type=float, default=1.0)  # Nucleus-sampling top-p when do_sample=True.

    scorer_group = ap.add_argument_group("Stage1 scorer")
    scorer_group.add_argument("--stage1_scorer_dir", type=str, default="")  # Existing scorer checkpoint; required for eval-only mode, optional as retrain initialization.
    scorer_group.add_argument("--scorer_model_name", type=str, default="Qwen/Qwen3-0.6B")  # Base scorer model used when the scorer is initialized from base instead of restored from a checkpoint.
    scorer_group.add_argument("--scorer_max_length", type=int, default=5000)  # Maximum scorer input length.
    scorer_group.add_argument("--score_bias", action=argparse.BooleanOptionalAction, default=False)  # Whether to use a bias term in the scorer head and load checkpoints under that assumption.

    retrain_group = ap.add_argument_group("Stage1 scorer retraining")
    retrain_group.add_argument("--train_scorer_on_reasoning", action=argparse.BooleanOptionalAction, default=False)  # Retrain/adapt the scorer inside this script before running the final evaluation.
    retrain_group.add_argument("--trained_scorer_subdir", type=str, default="stage1_scorer_adapted")  # Subdirectory name used to save the retrained scorer.
    retrain_group.add_argument("--stage1_epochs", type=int, default=1)  # Number of epochs for scorer retraining.
    retrain_group.add_argument("--stage1_batch_size", type=int, default=2)  # Per-device batch size for scorer retraining.
    retrain_group.add_argument("--stage1_grad_accum", type=int, default=4)  # Gradient-accumulation steps for scorer retraining.
    retrain_group.add_argument("--stage1_lr", type=float, default=5e-5)  # Learning rate for scorer retraining.
    retrain_group.add_argument("--stage1_lora_r", type=int, default=8)  # LoRA rank when the scorer is initialized from base and retrained with LoRA.
    retrain_group.add_argument("--stage1_lora_alpha", type=int, default=16)  # LoRA alpha for scorer retraining.
    retrain_group.add_argument("--stage1_lora_dropout", type=float, default=0.05)  # LoRA dropout for scorer retraining.
    retrain_group.add_argument("--stage1_lora_last_n", type=int, default=8)  # Restrict scorer LoRA to the last N layers; 0 means no layer restriction.
    retrain_group.add_argument("--stage1_score_only", action=argparse.BooleanOptionalAction, default=False)  # Retrain only the scorer head and skip LoRA updates.
    retrain_group.add_argument("--stage1_disable_dropout", action=argparse.BooleanOptionalAction, default=True)  # Disable dropout during scorer retraining for more stable evaluation behavior.
    retrain_group.add_argument("--save_steps", type=int, default=50)  # Checkpoint-save interval during scorer retraining.
    retrain_group.add_argument("--log_steps", type=int, default=10)  # Logging interval during scorer retraining.
    retrain_group.add_argument("--eval_steps", type=int, default=0)  # Periodic eval interval during scorer retraining; 0 disables periodic eval.
    retrain_group.add_argument("--neg_per_pos", type=int, default=2)  # Number of negative pairs sampled per positive example during pairwise scorer training.
    retrain_group.add_argument("--loss_type", type=str, default="pairwise_bce", choices=["pairwise", "pairwise_bce", "margin", "bce"])  # Loss type used for scorer retraining.
    retrain_group.add_argument("--pointwise_alpha", type=float, default=0.2)  # Weight of the pointwise term when using a mixed loss.
    retrain_group.add_argument("--pairwise_margin", type=float, default=0.0)  # Margin used by pairwise / margin-style scorer losses.
    retrain_group.add_argument("--pos_weight", type=float, default=None)  # Positive-class weight for BCE-style losses; None leaves it unweighted.
    retrain_group.add_argument("--margin", type=float, default=0.1)  # Margin value used by the margin loss.
    retrain_group.add_argument("--margin_on_sigmoid", action="store_true", default=True)  # Apply the margin to sigmoid(score); this is the current default.
    retrain_group.add_argument("--margin_on_logit", action="store_false", dest="margin_on_sigmoid")  # Instead apply the margin directly to raw logits.
    retrain_group.add_argument("--train_drop_empty_baseline", action=argparse.BooleanOptionalAction, default=False)  # Filter train rows whose baseline dx and rx are both empty before scorer retraining.
    retrain_group.add_argument("--train_drop_empty_dx", action=argparse.BooleanOptionalAction, default=False)  # Filter train rows whose baseline dx is empty before scorer retraining.
    retrain_group.add_argument("--train_drop_dx1", action=argparse.BooleanOptionalAction, default=False)  # Filter train rows whose baseline dx count is <=1 before scorer retraining.

    eval_group = ap.add_argument_group("Eval and final metrics")
    eval_group.add_argument("--eval_batch_size", type=int, default=8)  # Scorer evaluation batch size.
    eval_group.add_argument("--eval_sigmoid", action=argparse.BooleanOptionalAction, default=True)  # Apply sigmoid to scorer logits before computing evaluation metrics.
    eval_group.add_argument("--drop_empty_baseline_final_eval", action=argparse.BooleanOptionalAction, default=False)  # Filter eval rows whose baseline dx and rx are both empty; applied consistently to train-time eval and final eval.
    eval_group.add_argument("--drop_empty_dx_final_eval", action=argparse.BooleanOptionalAction, default=False)  # Filter eval rows whose baseline dx is empty, regardless of rx; stricter than drop_empty_baseline_final_eval.
    eval_group.add_argument("--drop_dx1", action=argparse.BooleanOptionalAction, default=False)  # Filter eval rows whose baseline dx count is <=1; stricter than drop_empty_dx_final_eval.
    return ap


def resolve_test_jsonl(args):
    if args.testslice:
        test_dir = resolve_testslice_dir(args.data_name)
        return scorer_phase._find_slice_jsonl_files(
            test_dir,
            args.testslice,
            pattern_template="test_fold{slice}.jsonl",
        )
    if args.test_jsonl:
        return Path("data") / args.test_jsonl
    raise ValueError("Provide either --testslice or --test_jsonl.")


def resolve_input_jsonl(path_str: str, dataset: str | None = None) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path("data") / path_str)
        for root in artifact_lookup_roots(dataset):
            candidates.append(root / path_str)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to find jsonl: {path_str}")


def resolve_input_jsonl_if_exists(path_str: str, dataset: str | None = None) -> Path | None:
    path_str = str(path_str or "").strip()
    if not path_str:
        return None
    try:
        return resolve_input_jsonl(path_str, dataset)
    except FileNotFoundError:
        return None


def is_scorer_checkpoint_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or (path / "config.json").exists()


def resolve_stage1_scorer_input_dir(path_str: str, dataset: str | None = None) -> Path:
    raw = Path(path_str)
    bases = [raw]
    if not raw.is_absolute():
        for root in artifact_lookup_roots(dataset):
            bases.append(root / path_str)
    for base in bases:
        if base.exists() and is_scorer_checkpoint_dir(base):
            # print(f"resolve stage1 scorer: {base}\n", flush=True)
            return base
    raise FileNotFoundError(f"Unable to find exact stage1 scorer checkpoint: {path_str}")


def resolve_stage2_student_input_dir(path_str: str, dataset: str | None = None) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        for root in artifact_lookup_roots(dataset):
            candidates.append(root / path_str)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to find stage2 student checkpoint: {path_str}")


def load_stage2_student_for_eval(
    *,
    args,
    checkpoint_path: Path,
) -> DistillGenerator:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    adapter_config_path = checkpoint_path / "adapter_config.json"
    model_name = args.generator_model_name
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        model_name = str(adapter_cfg.get("base_model_name_or_path") or args.generator_model_name)
    else:
        model_name = str(checkpoint_path)

    generator = DistillGenerator(
        model_name=model_name,
        device=device,
        torch_dtype=args.generator_torch_dtype,
        lora_r=0,
    )
    if adapter_config_path.exists():
        generator.model = PeftModel.from_pretrained(generator.model, str(checkpoint_path))
        generator.model.to(device)
        generator.using_lora = True

    tokenizer_source = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else model_name
    generator.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
    generator.tokenizer.padding_side = "left"
    generator.tokenizer.truncation_side = "left"
    if generator.tokenizer.pad_token_id is not None and hasattr(generator.model, "config"):
        generator.model.config.pad_token_id = generator.tokenizer.pad_token_id
    generator.model.eval()
    return generator


def write_train_cache_meta(args, cache_jsonl: Path) -> None:
    meta_path = cache_jsonl.parent / "cache_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "train_cache_note": str(args.train_cache_note or ""),
        "stage2_student_dir": str(args.stage2_student_dir or ""),
        "stage2_student_use_base_model": bool(args.stage2_student_use_base_model),
        "ehr_only_ablation": bool(args.ehr_only_ablation),
        "train_drop_empty_baseline": bool(args.train_drop_empty_baseline),
        "train_drop_empty_dx": bool(args.train_drop_empty_dx),
        "train_drop_dx1": bool(args.train_drop_dx1),
        "generator_model_name": str(args.generator_model_name),
        "generator_max_prompt_tokens": int(args.generator_max_prompt_tokens),
        "eval_generator_max_new_tokens": int(args.eval_generator_max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "reasoning_pid_key": str(args.reasoning_pid_key),
        "reasoning_text_key": str(args.reasoning_text_key),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def read_eval_rows(eval_jsonl) -> list[dict]:
    if isinstance(eval_jsonl, list):
        return scorer_phase._read_jsonl_multi(eval_jsonl)
    return scorer_phase._read_jsonl(eval_jsonl)


def _has_nonempty_baseline_codes(row: dict) -> bool:
    base_dx = row.get("base_codes_diagnosis", []) or []
    base_rx = row.get("base_codes_medication", []) or []
    return len(base_dx) > 0 or len(base_rx) > 0


def _has_nonempty_baseline_dx(row: dict) -> bool:
    base_dx = row.get("base_codes_diagnosis", []) or []
    return len(base_dx) > 0


def _has_baseline_dx_gt1(row: dict) -> bool:
    base_dx = row.get("base_codes_diagnosis", []) or []
    return len(base_dx) > 1


def write_eval_rows_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _optional_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value < 0:
        return None
    return value


def _compact_str(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _color_code_for_tag(tag: str) -> str:
    return "\033[1;34m"


def _emit_log_line(line: str, *, tag: str) -> None:
    stream = sys.stdout
    colorized = line
    color = _color_code_for_tag(tag)
    primary = getattr(stream, "primary", None)
    log_file = getattr(stream, "log_file", None)

    def _colorize_tag_only(text: str) -> str:
        if not text.startswith("["):
            return text
        close_idx = text.find("]")
        if close_idx < 0:
            return text
        tag_text = text[: close_idx + 1]
        tail = text[close_idx + 1 :]
        return f"{color}{tag_text}\033[0m{tail}"

    if primary is not None and log_file is not None:
        if primary.isatty():
            colorized = _colorize_tag_only(line)
        primary.write(colorized + "\n\n")
        log_file.write(line + "\n\n")
        primary.flush()
        log_file.flush()
        return
    if hasattr(stream, "isatty") and stream.isatty():
        colorized = _colorize_tag_only(line)
    stream.write(colorized + "\n\n")
    stream.flush()


def log_compact(tag: str, message: str = "", **kwargs) -> None:
    parts: list[str] = []
    if message:
        parts.append(str(message))
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={_compact_str(value)}")
    body = " | ".join(parts).strip()
    suffix = f"[{tag}]"
    if body:
        _emit_log_line(f"{suffix} {body}", tag=tag)
    else:
        _emit_log_line(f"{suffix}", tag=tag)


def split_tag(split_name: str, suffix: str) -> str:
    split_name = str(split_name).strip().capitalize()
    if suffix.startswith("subset"):
        return f"{split_name}Subset"
    return f"{split_name}{suffix.capitalize()}"


def should_use_fixed_train_subset(args) -> bool:
    return float(args.train_subset_ratio) < 1.0


def should_use_fixed_test_subset(args) -> bool:
    return float(args.test_subset_ratio) < 1.0


def _normalize_jsonl_ref(path_or_paths) -> str | list[str]:
    if isinstance(path_or_paths, list):
        return [str(path) for path in path_or_paths]
    return str(path_or_paths)


def _resolve_cache_root(root_str: str, *, default_name: str, dataset: str) -> Path:
    root = Path(str(root_str or "").strip() or default_name)
    if not root.is_absolute():
        root = artifact_output_root(dataset) / root
    return root


def _count_labels(rows: list[dict]) -> tuple[int, int]:
    pos_count = sum(1 for row in rows if int(row["label"]) == 1)
    neg_count = len(rows) - pos_count
    return pos_count, neg_count


def sample_fixed_subset_rows(
    *,
    rows: list[dict],
    subset_ratio: float,
    target_pos_ratio: float | None,
    sample_seed: int,
    split_name: str,
) -> list[dict]:
    subset_ratio = float(subset_ratio)
    if not (0.0 < subset_ratio <= 1.0):
        raise ValueError("subset_ratio must be in (0, 1].")
    target_pos_ratio = _optional_ratio(target_pos_ratio)
    if target_pos_ratio is not None and not (0.0 < target_pos_ratio < 1.0):
        raise ValueError("target_pos_ratio must be in (0, 1), or negative to preserve source prevalence.")

    positives = [(idx, row) for idx, row in enumerate(rows) if int(row["label"]) == 1]
    negatives = [(idx, row) for idx, row in enumerate(rows) if int(row["label"]) == 0]
    if not positives or not negatives:
        raise ValueError(f"{split_name} subset sampling requires both positive and negative rows.")

    rng = scorer_phase.np.random.default_rng(int(sample_seed))
    if target_pos_ratio is None:
        n_pos = min(len(positives), max(1, int(round(len(positives) * subset_ratio))))
        n_neg = min(len(negatives), max(1, int(round(len(negatives) * subset_ratio))))
    else:
        target_total = min(len(rows), max(2, int(round(len(rows) * subset_ratio))))
        n_pos = int(round(target_total * target_pos_ratio))
        n_neg = target_total - n_pos
        if n_pos <= 0:
            n_pos = 1
            n_neg = target_total - 1
        if n_neg <= 0:
            n_neg = 1
            n_pos = target_total - 1
        if n_pos > len(positives) or n_neg > len(negatives):
            max_total = min(
                int(math.floor(len(positives) / target_pos_ratio)),
                int(math.floor(len(negatives) / (1.0 - target_pos_ratio))),
            )
            raise ValueError(
                f"{split_name} subset target is infeasible: "
                f"requested_total={target_total} requested_pos={n_pos} requested_neg={n_neg} "
                f"available_pos={len(positives)} available_neg={len(negatives)} "
                f"max_feasible_total_for_target_pos_ratio={max_total}"
            )

    pos_choice = rng.choice(len(positives), size=n_pos, replace=False)
    neg_choice = rng.choice(len(negatives), size=n_neg, replace=False)
    selected_indices = sorted(
        [positives[int(idx)][0] for idx in pos_choice]
        + [negatives[int(idx)][0] for idx in neg_choice]
    )
    return [rows[int(idx)] for idx in selected_indices]


def _selected_ids_hash(rows: list[dict]) -> str:
    joined = "||".join(str(row.get("id")) for row in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_fixed_subset_cache_meta(
    *,
    args,
    split_name: str,
    source_jsonl,
    rows: list[dict],
    subset_ratio: float,
    target_pos_ratio: float | None,
    sample_seed: int,
) -> dict[str, object]:
    pos_count, neg_count = _count_labels(rows)
    return {
        "mode": "fixed_subset_reasoning",
        "split": str(split_name),
        "data_name": str(args.data_name),
        "source_jsonl": _normalize_jsonl_ref(source_jsonl),
        "subset_ratio": float(subset_ratio),
        "target_pos_ratio": None if target_pos_ratio is None else float(target_pos_ratio),
        "sample_seed": int(sample_seed),
        "rows": int(len(rows)),
        "positives": int(pos_count),
        "negatives": int(neg_count),
        "selected_ids_hash": _selected_ids_hash(rows),
        "stage2_student_dir": str(args.stage2_student_dir or ""),
        "stage2_student_use_base_model": bool(args.stage2_student_use_base_model),
        "ehr_only_ablation": bool(args.ehr_only_ablation),
        "train_drop_empty_baseline": bool(args.train_drop_empty_baseline),
        "train_drop_empty_dx": bool(args.train_drop_empty_dx),
        "train_drop_dx1": bool(args.train_drop_dx1),
        "generator_model_name": str(args.generator_model_name),
        "generator_max_prompt_tokens": int(args.generator_max_prompt_tokens),
        "eval_generator_max_new_tokens": int(args.eval_generator_max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "reasoning_pid_key": str(args.reasoning_pid_key),
        "reasoning_text_key": str(args.reasoning_text_key),
    }


def prepare_fixed_subset_cache(
    *,
    args,
    split_name: str,
    rows: list[dict],
    source_jsonl,
    cache_root: Path,
    cache_note: str,
    subset_ratio: float,
    target_pos_ratio: float | None,
    sample_seed: int,
) -> tuple[list[dict], Path, Path]:
    cache_note = str(cache_note or "").strip()
    if not cache_note:
        raise ValueError(f"{split_name} fixed-subset cache requires a cache note.")
    cache_dir = cache_root / cache_note
    sampled_rows_path = cache_dir / "sampled_rows.jsonl"
    reasoning_path = cache_dir / "generated_reasoning.jsonl"
    meta_path = cache_dir / "meta.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not sampled_rows_path.exists():
        conflicting = [
            child.name
            for child in cache_dir.iterdir()
            if child.is_file()
            and child.name not in {"sampled_rows.jsonl", "generated_reasoning.jsonl", "meta.json"}
        ]
        if conflicting:
            raise ValueError(
                f"{split_name} fixed-subset cache dir already contains non-fixed-cache files: "
                f"{cache_dir} -> {sorted(conflicting)}"
            )
        sampled_rows = sample_fixed_subset_rows(
            rows=rows,
            subset_ratio=subset_ratio,
            target_pos_ratio=target_pos_ratio,
            sample_seed=sample_seed,
            split_name=split_name,
        )
        write_eval_rows_jsonl(sampled_rows, sampled_rows_path)
    else:
        sampled_rows = scorer_phase._read_jsonl(sampled_rows_path)

    meta = build_fixed_subset_cache_meta(
        args=args,
        split_name=split_name,
        source_jsonl=source_jsonl,
        rows=sampled_rows,
        subset_ratio=subset_ratio,
        target_pos_ratio=target_pos_ratio,
        sample_seed=sample_seed,
    )
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        if existing_meta != meta:
            raise ValueError(
                f"{split_name} fixed-subset cache meta mismatch for {cache_dir}. "
                f"Use a new cache note instead of reusing an incompatible one."
            )
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    pos_count, neg_count = _count_labels(sampled_rows)
    log_compact(
        split_tag(split_name, "subset-cache"),
        rows=len(sampled_rows),
        pos=pos_count,
        neg=neg_count,
        subset_ratio=subset_ratio,
        target_pos_ratio=target_pos_ratio,
        sample_seed=sample_seed,
        # cache_dir=cache_dir,
        reasoning_jsonl=reasoning_path,
    )
    return sampled_rows, sampled_rows_path, reasoning_path


def load_generation_student(
    *,
    args,
    checkpoint_dir: str,
):
    checkpoint_dir = str(checkpoint_dir or "").strip()
    use_base_model = bool(args.stage2_student_use_base_model)
    if not use_base_model:
        if not checkpoint_dir:
            raise ValueError("Reasoning generation requires --stage2_student_dir unless --stage2_student_use_base_model is set.")
        checkpoint_path = resolve_stage2_student_input_dir(checkpoint_dir, args.data_name)
        log_compact("GenLoad", checkpoint_dir=checkpoint_path)
        return load_stage2_student_for_eval(
            args=args,
            checkpoint_path=checkpoint_path,
        )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model_name = str(args.generator_model_name)
    tokenizer_source = model_name
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = resolve_stage2_student_input_dir(checkpoint_dir, args.data_name)
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            model_name = str(adapter_cfg.get("base_model_name_or_path") or args.generator_model_name)
        if (checkpoint_path / "tokenizer_config.json").exists():
            tokenizer_source = checkpoint_path

    generator = DistillGenerator(
        model_name=model_name,
        device=device,
        torch_dtype=args.generator_torch_dtype,
        lora_r=0,
    )
    generator.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
    generator.tokenizer.padding_side = "left"
    generator.tokenizer.truncation_side = "left"
    if generator.tokenizer.pad_token_id is not None and hasattr(generator.model, "config"):
        generator.model.config.pad_token_id = generator.tokenizer.pad_token_id
    generator.model.eval()
    log_compact(
        "GenLoad",
        generator_model=model_name,
        stage2_student_dir=checkpoint_path,
    )
    return generator


def prepare_eval_rows_and_jsonl(
    *,
    rows: list[dict],
    eval_jsonl,
    output_dir: Path,
    drop_empty_baseline_final_eval: bool,
    drop_empty_dx_final_eval: bool = False,
    drop_dx1: bool = False,
) -> tuple[list[dict], Path | list[Path] | str, bool]:
    if bool(drop_dx1):
        filtered_rows = [row for row in rows if _has_baseline_dx_gt1(row)]
        dropped_count = len(rows) - len(filtered_rows)
        if not filtered_rows:
            raise ValueError("All eval rows were filtered out by baseline dx count <= 1.")
        if dropped_count <= 0:
            return filtered_rows, eval_jsonl, False

        filtered_eval_jsonl = output_dir / "eval_filtered_drop_dx1.jsonl"
        write_eval_rows_jsonl(filtered_rows, filtered_eval_jsonl)
        pos_count, neg_count = _count_labels(filtered_rows)
        log_compact(
            "DxCountFilter",
            drop_dx1=True,
            threshold="base_dx_count>1",
            kept=len(filtered_rows),
            dropped=dropped_count,
            original=len(rows),
            pos=pos_count,
            neg=neg_count,
            applies_to="train_eval_and_final_eval",
            eval_jsonl=filtered_eval_jsonl,
        )
        return filtered_rows, filtered_eval_jsonl, True

    if bool(drop_empty_dx_final_eval):
        filtered_rows = [row for row in rows if _has_nonempty_baseline_dx(row)]
        dropped_count = len(rows) - len(filtered_rows)
        if not filtered_rows:
            raise ValueError("All eval rows were filtered out by empty baseline dx.")
        if dropped_count <= 0:
            return filtered_rows, eval_jsonl, False

        filtered_eval_jsonl = output_dir / "eval_filtered_no_empty_dx.jsonl"
        write_eval_rows_jsonl(filtered_rows, filtered_eval_jsonl)
        log_compact(
            "EmptyFilter",
            drop_empty_dx_final_eval=True,
            drop_empty_baseline_final_eval=bool(drop_empty_baseline_final_eval),
            kept=len(filtered_rows),
            dropped=dropped_count,
            original=len(rows),
            applies_to="train_eval_and_final_eval",
            eval_jsonl=filtered_eval_jsonl,
        )
        return filtered_rows, filtered_eval_jsonl, True

    if not bool(drop_empty_baseline_final_eval):
        return rows, eval_jsonl, False

    filtered_rows = [row for row in rows if _has_nonempty_baseline_codes(row)]
    dropped_count = len(rows) - len(filtered_rows)
    if not filtered_rows:
        raise ValueError("All eval rows were filtered out by empty baseline dx/rx.")
    if dropped_count <= 0:
        return filtered_rows, eval_jsonl, False

    filtered_eval_jsonl = output_dir / "eval_filtered_no_empty_baseline.jsonl"
    write_eval_rows_jsonl(filtered_rows, filtered_eval_jsonl)
    log_compact(
        "EmptyFilter",
        drop_empty_baseline_final_eval=True,
        kept=len(filtered_rows),
        dropped=dropped_count,
        original=len(rows),
        applies_to="train_eval_and_final_eval",
        eval_jsonl=filtered_eval_jsonl,
    )
    return filtered_rows, filtered_eval_jsonl, True


def prepare_train_rows_and_jsonl(
    *,
    rows: list[dict],
    train_jsonl,
    output_dir: Path,
    train_drop_empty_baseline: bool = False,
    train_drop_empty_dx: bool = False,
    train_drop_dx1: bool = False,
) -> tuple[list[dict], Path | list[Path] | str, bool]:
    if bool(train_drop_dx1):
        filtered_rows = [row for row in rows if _has_baseline_dx_gt1(row)]
        dropped_count = len(rows) - len(filtered_rows)
        if not filtered_rows:
            raise ValueError("All train rows were filtered out by baseline dx count <= 1.")
        if dropped_count <= 0:
            return filtered_rows, train_jsonl, False

        filtered_train_jsonl = output_dir / "train_filtered_drop_dx1.jsonl"
        write_eval_rows_jsonl(filtered_rows, filtered_train_jsonl)
        pos_count, neg_count = _count_labels(filtered_rows)
        log_compact(
            "TrainDxCountFilter",
            train_drop_dx1=True,
            threshold="base_dx_count>1",
            kept=len(filtered_rows),
            dropped=dropped_count,
            original=len(rows),
            pos=pos_count,
            neg=neg_count,
            applies_to="scorer_retrain_trainset",
            train_jsonl=filtered_train_jsonl,
        )
        return filtered_rows, filtered_train_jsonl, True

    if bool(train_drop_empty_dx):
        filtered_rows = [row for row in rows if _has_nonempty_baseline_dx(row)]
        dropped_count = len(rows) - len(filtered_rows)
        if not filtered_rows:
            raise ValueError("All train rows were filtered out by empty baseline dx.")
        if dropped_count <= 0:
            return filtered_rows, train_jsonl, False

        filtered_train_jsonl = output_dir / "train_filtered_no_empty_dx.jsonl"
        write_eval_rows_jsonl(filtered_rows, filtered_train_jsonl)
        pos_count, neg_count = _count_labels(filtered_rows)
        log_compact(
            "TrainEmptyFilter",
            train_drop_empty_dx=True,
            train_drop_empty_baseline=bool(train_drop_empty_baseline),
            kept=len(filtered_rows),
            dropped=dropped_count,
            original=len(rows),
            pos=pos_count,
            neg=neg_count,
            applies_to="scorer_retrain_trainset",
            train_jsonl=filtered_train_jsonl,
        )
        return filtered_rows, filtered_train_jsonl, True

    if not bool(train_drop_empty_baseline):
        return rows, train_jsonl, False

    filtered_rows = [row for row in rows if _has_nonempty_baseline_codes(row)]
    dropped_count = len(rows) - len(filtered_rows)
    if not filtered_rows:
        raise ValueError("All train rows were filtered out by empty baseline dx/rx.")
    if dropped_count <= 0:
        return filtered_rows, train_jsonl, False

    filtered_train_jsonl = output_dir / "train_filtered_no_empty_baseline.jsonl"
    write_eval_rows_jsonl(filtered_rows, filtered_train_jsonl)
    pos_count, neg_count = _count_labels(filtered_rows)
    log_compact(
        "TrainEmptyFilter",
        train_drop_empty_baseline=True,
        kept=len(filtered_rows),
        dropped=dropped_count,
        original=len(rows),
        pos=pos_count,
        neg=neg_count,
        applies_to="scorer_retrain_trainset",
        train_jsonl=filtered_train_jsonl,
    )
    return filtered_rows, filtered_train_jsonl, True


def load_reasoning_map(path: str | Path, *, pid_key: str, text_key: str) -> dict[str, str]:
    path = resolve_input_jsonl(str(path))
    rows = scorer_phase._read_jsonl(path)
    reasoning_map: dict[str, str] = {}
    for row in rows:
        pid = row.get(pid_key)
        text = row.get(text_key)
        if pid is not None and text is not None:
            reasoning_map[str(pid)] = str(text)
    return reasoning_map


def load_reasoning_map_if_exists(path: Path, *, pid_key: str, text_key: str) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = scorer_phase._read_jsonl(path)
    return {
        str(row[pid_key]): str(row[text_key])
        for row in rows
        if row.get(pid_key) is not None and row.get(text_key) is not None
    }


def select_reasoning_for_examples(
    reasoning_map: dict[str, str],
    examples: list[Example],
) -> dict[str, str]:
    return {
        str(example.patient_id): reasoning_map[str(example.patient_id)]
        for example in examples
        if str(example.patient_id) in reasoning_map
    }


def reasoning_coverage_stats(
    reasoning_map: dict[str, str],
    examples: list[Example],
) -> tuple[int, int, int]:
    total = len(examples)
    existing = sum(1 for example in examples if str(example.patient_id) in reasoning_map)
    missing = total - existing
    return total, existing, missing


def filter_rows_by_reasoning_ids(
    rows: list[dict],
    reasoning_map: dict[str, str],
    *,
    split_name: str,
) -> list[dict]:
    filtered_rows = [
        row
        for row in rows
        if row.get("id") is not None and str(row.get("id")) in reasoning_map
    ]
    if not filtered_rows:
        raise ValueError(f"{split_name} reasoning file has no overlapping ids with the source rows.")
    pos_count, neg_count = _count_labels(filtered_rows)
    log_compact(
        split_tag(split_name, "subset-from-reasoning"),
        rows=len(filtered_rows),
        pos=pos_count,
        neg=neg_count,
    )
    return filtered_rows


def append_missing_reasoning_jsonl(
    *,
    generator,
    examples: list[Example],
    prompter: ClinicalReasoningPrompter,
    output_jsonl: Path,
    existing_map: dict[str, str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
    desc: str,
) -> dict[str, str]:
    missing_examples = [
        example
        for example in examples
        if str(example.patient_id) not in existing_map
    ]
    log_compact(
        "ReasoningGen",
        desc=desc,
        existing=len(existing_map),
        requested=len(examples),
        missing=len(missing_examples),
    )
    if not missing_examples:
        return existing_map

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    batches = chunk_examples(missing_examples, batch_size=batch_size)
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for batch_examples in tqdm(batches, desc=desc):
            student_prompts = generator.build_student_prompt_batch(
                batch_examples,
                prompter=prompter,
                max_length=max_prompt_tokens,
            )
            rollout = generator.generate_rollouts(
                prompt_ids=student_prompts.input_ids,
                prompt_attention_mask=student_prompts.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            for example, text in zip(batch_examples, rollout.texts):
                pid = str(example.patient_id)
                reasoning = str(text)
                existing_map[pid] = reasoning
                f.write(
                    json.dumps(
                        {
                            "id": pid,
                            "label": int(example.label),
                            "reasoning": reasoning,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return existing_map


def ensure_complete_reasoning_cache(
    *,
    generator,
    examples: list[Example],
    prompter: ClinicalReasoningPrompter,
    output_jsonl: Path,
    existing_map: dict[str, str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
    desc: str,
) -> dict[str, str]:
    existing_map = select_reasoning_for_examples(existing_map, examples)
    existing_map = append_missing_reasoning_jsonl(
        generator=generator,
        examples=examples,
        prompter=prompter,
        output_jsonl=output_jsonl,
        existing_map=existing_map,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        desc=desc,
    )
    return select_reasoning_for_examples(existing_map, examples)


class LazyStage2GeneratorProvider:
    def __init__(self, *, args, generator=None) -> None:
        self.args = args
        self.generator = generator

    def get(self):
        if self.generator is None:
            stage2_student_dir = str(self.args.stage2_student_dir or "").strip()
            if not stage2_student_dir and not bool(self.args.stage2_student_use_base_model):
                raise ValueError("Lazy train reasoning requires --stage2_student_dir.")
            self.generator = load_generation_student(
                args=self.args,
                checkpoint_dir=stage2_student_dir,
            )
        return self.generator

    def close(self) -> None:
        if self.generator is None:
            return
        del self.generator
        self.generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def append_missing_reasoning_rows_jsonl(
    *,
    generator_provider: LazyStage2GeneratorProvider,
    rows: list[dict],
    prompter: ClinicalReasoningPrompter,
    output_jsonl: Path,
    existing_map: dict[str, str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> int:
    missing_rows: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        pid = row.get("id")
        if pid is None:
            continue
        pid = str(pid)
        if pid in existing_map or pid in seen:
            continue
        seen.add(pid)
        missing_rows.append(row)
    if not missing_rows:
        return 0

    generator = generator_provider.get()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    batch_size = max(int(batch_size), 1)
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for start in range(0, len(missing_rows), batch_size):
            row_batch = missing_rows[start : start + batch_size]
            examples = [Example.from_dict(row) for row in row_batch]
            student_prompts = generator.build_student_prompt_batch(
                examples,
                prompter=prompter,
                max_length=max_prompt_tokens,
            )
            rollout = generator.generate_rollouts(
                prompt_ids=student_prompts.input_ids,
                prompt_attention_mask=student_prompts.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            for row, example, text in zip(row_batch, examples, rollout.texts):
                pid = str(row.get("id", example.patient_id))
                reasoning = str(text)
                existing_map[pid] = reasoning
                f.write(
                    json.dumps(
                        {
                            "id": pid,
                            "label": int(row["label"]),
                            "reasoning": reasoning,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                generated += 1
            f.flush()
    log_compact("LazyTrainReasoning", cache_size=len(existing_map))
    return generated


def build_lazy_pref_iterable_dataset_epoch_baseline(
    *,
    rows: list[dict],
    neg_per_pos: int,
    base_seed: int,
    scorer_tokenizer,
    reasoning_map: dict[str, str],
    generator_provider: LazyStage2GeneratorProvider,
    prompter: ClinicalReasoningPrompter,
    reasoning_cache_path: Path,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
    pairs_per_chunk: int,
    pos_fraction_per_epoch: float,
):
    prompt = scorer_phase.BASELINE_PROMPT
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    if not positives or not negatives:
        raise ValueError(f"Need both pos and neg. Got pos={len(positives)}, neg={len(negatives)}")

    cached_train_rows = sum(
        1
        for row in rows
        if row.get("id") is not None and str(row["id"]) in reasoning_map
    )
    neg_per_pos = int(neg_per_pos)
    if neg_per_pos <= 0:
        raise ValueError("neg_per_pos must be >= 1")
    pairs_per_chunk = max(int(pairs_per_chunk), 1)
    generation_batch_size = max(int(generation_batch_size), 1)
    pos_fraction_per_epoch = float(pos_fraction_per_epoch)
    if not (0.0 < pos_fraction_per_epoch <= 1.0):
        raise ValueError("pos_fraction_per_epoch must be in (0, 1].")
    sampled_pos_count = min(
        len(positives),
        max(1, int(math.ceil(len(positives) * pos_fraction_per_epoch))),
    )

    log_compact(
        "LazyTrainData",
        cases=len(positives),
        controls=len(negatives),
        total=len(positives) + len(negatives),
        case_control_ratio=(len(positives) / len(negatives)),
        cached_reasoning_rows=cached_train_rows,
        pairs_per_chunk=pairs_per_chunk,
        pos_fraction_per_epoch=pos_fraction_per_epoch,
        sampled_positives=sampled_pos_count,
    )

    state = scorer_phase.EpochState(base_seed=base_seed)
    eos_token = scorer_tokenizer.eos_token or ""

    def format_row(row: dict) -> str:
        pid = row.get("id")
        reasoning_text = None if pid is None else reasoning_map.get(str(pid))
        return scorer_phase._format_baseline_text(
            row,
            reasoning_text=reasoning_text,
            follow_up=False,
        )

    def tokenize_pair(pos_row: dict, neg_row: dict) -> dict[str, list[int]]:
        chosen = prompt + format_row(pos_row)
        rejected = prompt + format_row(neg_row)
        if eos_token:
            if not chosen.endswith(eos_token):
                chosen += eos_token
            if not rejected.endswith(eos_token):
                rejected += eos_token
        return {
            "chosen_ids": scorer_tokenizer(text=chosen)["input_ids"],
            "rejected_ids": scorer_tokenizer(text=rejected)["input_ids"],
        }

    def flush_pairs(pair_rows: list[tuple[dict, dict]]):
        rows_to_check: list[dict] = []
        for pos_row, neg_row in pair_rows:
            rows_to_check.append(pos_row)
            rows_to_check.append(neg_row)
        append_missing_reasoning_rows_jsonl(
            generator_provider=generator_provider,
            rows=rows_to_check,
            prompter=prompter,
            output_jsonl=reasoning_cache_path,
            existing_map=reasoning_map,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=generation_batch_size,
        )
        for pos_row, neg_row in pair_rows:
            yield tokenize_pair(pos_row, neg_row)

    def gen():
        returnseed = state.seed_for_epoch()
        rng = scorer_phase.np.random.default_rng(returnseed)
        pos_order = rng.permutation(len(positives))
        if sampled_pos_count < len(pos_order):
            pos_order = pos_order[:sampled_pos_count]
        pair_buffer: list[tuple[dict, dict]] = []
        for pidx in pos_order:
            pos = positives[int(pidx)]
            for _ in range(neg_per_pos):
                neg = negatives[int(rng.integers(0, len(negatives)))]
                pair_buffer.append((pos, neg))
                if len(pair_buffer) >= pairs_per_chunk:
                    yield from flush_pairs(pair_buffer)
                    pair_buffer = []
        if pair_buffer:
            yield from flush_pairs(pair_buffer)

    class LazyPrefIterableDataset(TorchIterableDataset):
        column_names = ["chosen_ids", "rejected_ids"]

        def __iter__(self):
            yield from gen()

        def filter(self, function, **kwargs):
            return FilteredLazyIterableDataset(self, function)

    class FilteredLazyIterableDataset(TorchIterableDataset):
        column_names = ["chosen_ids", "rejected_ids"]

        def __init__(self, parent, predicate):
            self.parent = parent
            self.predicate = predicate

        def __iter__(self):
            for example in self.parent:
                if self.predicate(example):
                    yield example

        def filter(self, function, **kwargs):
            return FilteredLazyIterableDataset(self, function)

    ds = LazyPrefIterableDataset()
    return ds, state, sampled_pos_count


def generate_reasoning_jsonl(
    *,
    generator,
    examples: list[Example],
    prompter: ClinicalReasoningPrompter,
    output_jsonl: Path,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
    desc: str = "generate reasoning",
) -> dict[str, str]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    reasoning_map: dict[str, str] = {}
    batches = chunk_examples(examples, batch_size=batch_size)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch_examples in tqdm(batches, desc=desc):
            student_prompts = generator.build_student_prompt_batch(
                batch_examples,
                prompter=prompter,
                max_length=max_prompt_tokens,
            )
            rollout = generator.generate_rollouts(
                prompt_ids=student_prompts.input_ids,
                prompt_attention_mask=student_prompts.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            for example, text in zip(batch_examples, rollout.texts):
                pid = str(example.patient_id)
                reasoning = str(text)
                reasoning_map[pid] = reasoning
                f.write(
                    json.dumps(
                        {
                            "id": pid,
                            "label": int(example.label),
                            "reasoning": reasoning,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return reasoning_map


def scorer_metrics(scores, labels) -> dict[str, float | int]:
    auroc, auprc, f1, sens90, sens95, ppv90, ppv95 = scorer_phase.get_evaluation_metrics(labels, scores)
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "sensitivity_90": float(sens90),
        "sensitivity_95": float(sens95),
        "ppv_90": float(ppv90),
        "ppv_95": float(ppv95),
    }


class PeriodicScorerEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        eval_jsonl,
        tokenizer,
        reasoning_map: dict[str, str],
        output_dir: Path,
        batch_size: int,
        max_length: int,
        apply_sigmoid: bool,
        eval_steps: int,
    ) -> None:
        self.eval_jsonl = eval_jsonl
        self.tokenizer = tokenizer
        self.reasoning_map = reasoning_map
        self.output_dir = Path(output_dir)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.apply_sigmoid = bool(apply_sigmoid)
        self.eval_steps = int(eval_steps)
        self.eval_ds = None
        self.eval_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            return_tensors="pt",
        )
        self.history_path = self.output_dir / "train_eval_history.jsonl"
        self.evaluated_steps: set[int] = set()
        self.history_initialized = False

    def _append_history(self, metrics: dict[str, float | int], step: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        row = {"global_step": int(step), **metrics}
        mode = "a" if self.history_initialized else "w"
        with open(self.history_path, mode, encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.history_initialized = True

    def _run_eval(self, model, step: int) -> None:
        if self.eval_ds is None or int(step) in self.evaluated_steps:
            return
        self.evaluated_steps.add(int(step))
        was_training = model.training
        model.eval()
        try:
            scores, labels = scorer_phase.eval_pointwise(
                model,
                self.tokenizer,
                self.eval_ds,
                text_key="text",
                label_key="label",
                batch_size=self.batch_size,
                max_length=self.max_length,
                apply_sigmoid=self.apply_sigmoid,
                collator=self.eval_collator,
            )
        finally:
            if was_training:
                model.train()
        metrics = scorer_metrics(scores, labels)
        self._append_history(metrics, step)
        label = (
            "TrainEval0"
            if int(step) == 0
            else f"TrainEval{int(step)}"
        )
        log_compact(
            label,
            AUROC=f"{metrics['auroc']:.4f}",
            AUPRC=f"{metrics['auprc']:.4f}",
            F1=f"{metrics['f1']:.4f}",
            Sens90=f"{metrics['sensitivity_90']:.4f}",
            Sens95=f"{metrics['sensitivity_95']:.4f}",
            PPV90=f"{metrics['ppv_90']:.4f}",
            PPV95=f"{metrics['ppv_95']:.4f}",
        )

    def on_train_begin(self, args, state, control, **kwargs):
        raw_ds = scorer_phase.build_pointwise_dataset_baseline(
            self.eval_jsonl,
            reasoning_map=self.reasoning_map,
            follow_up=False,
        )
        self.eval_ds = scorer_phase._pretokenize_pointwise_dataset(
            raw_ds,
            self.tokenizer,
            self.max_length,
        )
        model = kwargs.get("model")
        if model is not None:
            self._run_eval(model, int(state.global_step))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps <= 0 or int(state.global_step) <= 0:
            return control
        if int(state.global_step) % self.eval_steps != 0:
            return control
        model = kwargs.get("model")
        if model is not None:
            self._run_eval(model, int(state.global_step))
        return control


def evaluate_scorer(
    *,
    scorer_model,
    scorer_tokenizer,
    eval_jsonl,
    rows: list[dict],
    reasoning_map: dict[str, str],
    output_dir: Path,
    batch_size: int,
    max_length: int,
    apply_sigmoid: bool,
) -> dict[str, float | int]:
    ds = scorer_phase.build_pointwise_dataset_baseline(
        eval_jsonl,
        reasoning_map=reasoning_map,
        follow_up=False,
    )
    scores, labels = scorer_phase.eval_pointwise(
        scorer_model,
        scorer_tokenizer,
        ds,
        text_key="text",
        label_key="label",
        batch_size=batch_size,
        max_length=max_length,
        apply_sigmoid=apply_sigmoid,
    )
    metrics = scorer_metrics(scores, labels)

    scores_path = output_dir / "scores.jsonl"
    with open(scores_path, "w", encoding="utf-8") as f:
        for row, score, label in zip(rows, scores, labels):
            pid = row.get("id")
            f.write(
                json.dumps(
                    {
                        "id": None if pid is None else str(pid),
                        "label": int(label),
                        "score": float(score),
                        "has_reasoning": pid is not None and str(pid) in reasoning_map,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary_path = output_dir / "summary_metrics.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Eval Generator Scorer\n\n")
        f.write(f"- n: {metrics['n']}\n")
        f.write(f"- positives: {metrics['positives']}\n")
        f.write(f"- negatives: {metrics['negatives']}\n")
        f.write(f"- AUROC: {metrics['auroc']:.4f}\n")
        f.write(f"- AUPRC: {metrics['auprc']:.4f}\n")
        f.write(f"- F1: {metrics['f1']:.4f}\n")
        f.write(f"- Sens@90: {metrics['sensitivity_90']:.4f}\n")
        f.write(f"- Sens@95: {metrics['sensitivity_95']:.4f}\n")
        f.write(f"- PPV@90: {metrics['ppv_90']:.4f}\n")
        f.write(f"- PPV@95: {metrics['ppv_95']:.4f}\n")

    return metrics


def build_stage1_lora_config(
    model,
    lora_last_n: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> LoraConfig:
    kwargs: dict[str, object] = dict(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    if int(lora_last_n) > 0:
        layers, layers_pattern = scorer_phase._resolve_transformer_layers(model)
        if layers is None:
            raise ValueError("Unable to locate transformer layers for stage1 scorer LoRA.")
        num_layers = len(layers)
        n_last = min(int(lora_last_n), num_layers)
        kwargs["layers_pattern"] = layers_pattern
        kwargs["layers_to_transform"] = list(range(num_layers - n_last, num_layers))
    return LoraConfig(**kwargs)


def score_bias_load_attempts(score_bias: bool) -> list[bool]:
    requested = bool(score_bias)
    return [requested, not requested]


def load_stage1_scorer_for_training(
    *,
    args,
    checkpoint_dir: str | None,
):
    checkpoint_dir = str(checkpoint_dir or "").strip()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if not checkpoint_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.scorer_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForSequenceClassification.from_pretrained(
            args.scorer_model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )
        if args.score_bias:
            model = ensure_score_bias(model)
        scorer_phase._reset_score_head_with_seed(model, args.seed)
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            if hasattr(base_model, "config"):
                base_model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        log_compact("ScorerInit", mode="base_from_scratch", base_model=args.scorer_model_name)
        return model, tokenizer, True

    checkpoint_path = resolve_stage1_scorer_input_dir(checkpoint_dir, args.data_name)

    tokenizer_source = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else args.scorer_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_name = str(adapter_cfg.get("base_model_name_or_path") or args.scorer_model_name)
        load_error = None
        loaded_with_score_bias = None
        for attempt_bias in score_bias_load_attempts(args.score_bias):
            candidate = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=1,
                torch_dtype=torch.bfloat16,
            )
            if attempt_bias:
                candidate = ensure_score_bias(candidate)
            try:
                try:
                    model = PeftModel.from_pretrained(candidate, str(checkpoint_path), is_trainable=True)
                except TypeError:
                    model = PeftModel.from_pretrained(candidate, str(checkpoint_path))
                    for name, param in model.named_parameters():
                        if "lora_" in name or "modules_to_save" in name or "score" in name:
                            param.requires_grad = True
                load_error = None
                loaded_with_score_bias = attempt_bias
                break
            except (RuntimeError, ValueError) as exc:
                load_error = exc
        if load_error is not None:
            raise load_error
        log_compact("ScorerInit", stage1_score_bias_loaded=loaded_with_score_bias)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path),
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base_model, "config"):
            base_model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    log_compact("ScorerInit", mode="checkpoint", checkpoint_dir=checkpoint_path)
    return model, tokenizer, False


def load_stage1_scorer_for_eval(
    *,
    args,
    checkpoint_dir: str,
):
    checkpoint_path = resolve_stage1_scorer_input_dir(checkpoint_dir, args.data_name)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    tokenizer_source = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else args.scorer_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_name = str(adapter_cfg.get("base_model_name_or_path") or args.scorer_model_name)
        load_error = None
        loaded_with_score_bias = None
        for attempt_bias in score_bias_load_attempts(args.score_bias):
            candidate = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=1,
                torch_dtype=torch.bfloat16,
            )
            if attempt_bias:
                candidate = ensure_score_bias(candidate)
            try:
                model = PeftModel.from_pretrained(candidate, str(checkpoint_path))
                load_error = None
                loaded_with_score_bias = attempt_bias
                break
            except (RuntimeError, ValueError) as exc:
                load_error = exc
        if load_error is not None:
            raise load_error
        log_compact("\nScorerLoad", stage1_score_bias_loaded=loaded_with_score_bias)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path),
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base_model, "config"):
            base_model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()
    log_compact("ScorerLoad", checkpoint_dir=checkpoint_path)
    return model, tokenizer


def train_scorer_on_reasoning(
    *,
    args,
    train_jsonl: Path,
    train_rows: list[dict],
    train_examples: list[Example],
    train_reasoning_map: dict[str, str],
    train_reasoning_cache_path: Path,
    generator_provider: LazyStage2GeneratorProvider | None,
    prompter: ClinicalReasoningPrompter,
    eval_jsonl,
    eval_reasoning_map: dict[str, str],
    output_dir: Path,
):
    model, tokenizer, initialized_from_base = load_stage1_scorer_for_training(
        args=args,
        checkpoint_dir=args.stage1_scorer_dir,
    )
    peft_config = None
    if args.stage1_score_only:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "score" in name:
                param.requires_grad = True
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
    elif initialized_from_base:
        peft_config = build_stage1_lora_config(
            model,
            args.stage1_lora_last_n,
            args.stage1_lora_r,
            args.stage1_lora_alpha,
            args.stage1_lora_dropout,
        )

    if bool(args.ehr_only_ablation):
        train_ds, _, pos_count = scorer_phase.build_pref_iterable_dataset_epoch_baseline(
            in_jsonl=str(train_jsonl),
            neg_per_pos=args.neg_per_pos,
            base_seed=args.data_seed,
            prompt=None,
            include_meta=False,
            reasoning_map=train_reasoning_map,
            follow_up=False,
        )
    else:
        if len(train_reasoning_map) < len(train_examples):
            raise ValueError(
                "no-lazy mode expects complete train reasoning upfront; "
                "the train reasoning cache is still missing rows."
            )
        train_ds, _, pos_count = scorer_phase.build_pref_iterable_dataset_epoch_baseline(
            in_jsonl=str(train_jsonl),
            neg_per_pos=args.neg_per_pos,
            base_seed=args.data_seed,
            prompt=None,
            include_meta=False,
            reasoning_map=train_reasoning_map,
            follow_up=False,
        )
    pairs_per_epoch = pos_count * int(args.neg_per_pos)
    steps_per_epoch = math.ceil(pairs_per_epoch / (int(args.stage1_batch_size) * int(args.stage1_grad_accum)))
    max_steps = steps_per_epoch * int(args.stage1_epochs)
    scorer_dir = output_dir / str(args.trained_scorer_subdir)
    rcfg = make_reward_config(
        output_dir=scorer_dir,
        learning_rate=args.stage1_lr,
        batch_size=args.stage1_batch_size,
        grad_accum=args.stage1_grad_accum,
        max_length=args.scorer_max_length,
        max_steps=max_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        disable_dropout=bool(args.stage1_disable_dropout),
    )
    # rcfg.disable_dropout = bool(args.stage1_disable_dropout)

    callbacks = []
    if int(args.eval_steps) > 0:
        callbacks.append(
            PeriodicScorerEvalCallback(
                eval_jsonl=eval_jsonl,
                tokenizer=tokenizer,
                reasoning_map=eval_reasoning_map,
                output_dir=output_dir,
                batch_size=args.eval_batch_size,
                max_length=args.scorer_max_length,
                apply_sigmoid=args.eval_sigmoid,
                eval_steps=args.eval_steps,
            )
        )

    trainer = scorer_phase.LossSwitchRewardTrainer(
        model=model,
        args=rcfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=peft_config,
        loss_type=args.loss_type,
        pointwise_alpha=args.pointwise_alpha,
        pairwise_margin=args.pairwise_margin,
        pos_weight=args.pos_weight,
        margin=args.margin,
        margin_on_sigmoid=args.margin_on_sigmoid,
    )

    if bool(args.ehr_only_ablation):
        train_reasoning_mode = "ehr_only"
    elif should_use_fixed_train_subset(args):
        train_reasoning_mode = "fixed_subset_cache"
    else:
        train_reasoning_mode = "full_cache"
    log_compact(
        "ScorerTrain",
        mode=train_reasoning_mode,
        # train_jsonl=train_jsonl,
        # train_examples=len(train_examples),
        # train_reasoning_cache_jsonl=(None if bool(args.ehr_only_ablation) else train_reasoning_cache_path),
        # train_reasoning_count=(None if bool(args.ehr_only_ablation) else len(train_reasoning_map)),
        # pos_count=pos_count,
        # neg_per_pos=args.neg_per_pos,
        pairs_per_epoch=pairs_per_epoch,
        steps_per_epoch=steps_per_epoch,
        max_steps=max_steps,
        train_eval_steps=(args.eval_steps if int(args.eval_steps) > 0 else None),
        train_eval_history=(output_dir / "train_eval_history.jsonl" if int(args.eval_steps) > 0 else None),
    )
    if initialized_from_base and not args.stage1_score_only:
        log_compact(
            "ScorerTrain",
            stage1_init_mode="base_lora",
            lora_r=args.stage1_lora_r,
            lora_alpha=args.stage1_lora_alpha,
            lora_dropout=args.stage1_lora_dropout,
            lora_last_n=args.stage1_lora_last_n,
            disable_dropout=args.stage1_disable_dropout,
        )
    elif initialized_from_base:
        log_compact("ScorerTrain", stage1_init_mode="base_score_only", disable_dropout=args.stage1_disable_dropout)
    else:
        log_compact("ScorerTrain", stage1_init_mode="checkpoint_adapt", disable_dropout=args.stage1_disable_dropout)

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        total = sum(param.numel() for param in trainer.model.parameters())
        trainable = sum(param.numel() for param in trainer.model.parameters() if param.requires_grad)
        ratio = (trainable / total * 100.0) if total > 0 else 0.0
        log_compact("ScorerTrain", trainable_params=trainable, all_params=total, trainable_percent=f"{ratio:.4f}")

    trainer.train()
    trainer.model.save_pretrained(scorer_dir)
    tokenizer.save_pretrained(scorer_dir)

    return trainer.model, tokenizer


def main() -> None:
    args = build_parser().parse_args()
    suppress_external_log_noise()
    if args.data_seed is None:
        args.data_seed = int(args.seed)
    output_dir = artifact_output_root(args.data_name) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = install_run_log(output_dir)
    print(f"[run_log] {run_log}", flush=True)
    print(format_command(sys.argv), flush=True)
    log_compact(
        "Run",
        run_log=run_log,
        output_dir=output_dir,
        device=args.device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_count=torch.cuda.device_count(),
        train_scorer_on_reasoning=bool(args.train_scorer_on_reasoning),
        ehr_only_ablation=bool(args.ehr_only_ablation),
    )

    scorer_phase.set_baseline_prompt(args.data_name)
    eval_jsonl = resolve_test_jsonl(args)
    rows = read_eval_rows(eval_jsonl)
    reasoning_jsonl = str(args.reasoning_jsonl or "").strip()
    train_reasoning_jsonl = str(args.train_reasoning_jsonl or "").strip()
    resolved_reasoning_jsonl = None if bool(args.ehr_only_ablation) else resolve_input_jsonl_if_exists(reasoning_jsonl, args.data_name)
    resolved_train_reasoning_jsonl = None if bool(args.ehr_only_ablation) else resolve_input_jsonl_if_exists(train_reasoning_jsonl, args.data_name)
    requested_train_subset_mode = should_use_fixed_train_subset(args)
    fixed_train_subset_mode = bool(args.train_scorer_on_reasoning) and requested_train_subset_mode
    fixed_test_subset_mode = should_use_fixed_test_subset(args)
    fixed_cache_note = str(args.train_cache_note or "").strip() or str(output_dir.name)
    fixed_test_reasoning_path: Path | None = None
    fixed_test_from_existing_reasoning = False
    if fixed_test_subset_mode:
        if resolved_reasoning_jsonl is not None:
            fixed_test_from_existing_reasoning = True
            log_compact(
                split_tag("test", "subset-plan"),
                mode="reasoning_file_ids",
                note="test_subset_ratio ignored because reasoning_jsonl already exists",
            )
            existing_test_reasoning_map = load_reasoning_map(
                resolved_reasoning_jsonl,
                pid_key=args.reasoning_pid_key,
                text_key=args.reasoning_text_key,
            )
            rows = filter_rows_by_reasoning_ids(
                rows,
                existing_test_reasoning_map,
                split_name="test",
            )
            eval_jsonl = output_dir / "eval_subset_from_reasoning.jsonl"
            write_eval_rows_jsonl(rows, eval_jsonl)
            fixed_test_reasoning_path = resolved_reasoning_jsonl
            log_compact(split_tag("test", "subset-plan"), reasoning_jsonl=resolved_reasoning_jsonl, eval_jsonl=eval_jsonl)
        else:
            fixed_test_cache_root = _resolve_cache_root(
                "",
                default_name=f"{args.data_name}_test_cache",
                dataset=args.data_name,
            )
            rows, eval_jsonl, fixed_test_reasoning_path = prepare_fixed_subset_cache(
                args=args,
                split_name="test",
                rows=rows,
                source_jsonl=eval_jsonl,
                cache_root=fixed_test_cache_root,
                cache_note=fixed_cache_note,
                subset_ratio=args.test_subset_ratio,
                target_pos_ratio=None,
                sample_seed=args.data_seed,
            )
    examples = [Example.from_dict(row) for row in rows]

    train_jsonl = None
    train_rows: list[dict] = []
    train_examples: list[Example] = []
    fixed_train_reasoning_path: Path | None = None
    fixed_train_from_existing_reasoning = False
    if args.train_scorer_on_reasoning:
        train_jsonl_arg = str(args.train_jsonl or "").strip()
        if not train_jsonl_arg:
            raise ValueError("--train_scorer_on_reasoning requires --train_jsonl.")
        train_jsonl = resolve_input_jsonl(train_jsonl_arg, args.data_name)
        train_rows = scorer_phase._read_jsonl(train_jsonl)
        train_rows, train_jsonl, _ = prepare_train_rows_and_jsonl(
            rows=train_rows,
            train_jsonl=train_jsonl,
            output_dir=output_dir,
            train_drop_empty_baseline=args.train_drop_empty_baseline,
            train_drop_empty_dx=args.train_drop_empty_dx,
            train_drop_dx1=args.train_drop_dx1,
        )
        if fixed_train_subset_mode:
            if resolved_train_reasoning_jsonl is not None:
                fixed_train_from_existing_reasoning = True
                log_compact(
                    split_tag("train", "subset-plan"),
                    mode="reasoning_file_ids",
                    note="train_subset_ratio and subset_pos_ratio ignored because train_reasoning_jsonl already exists",
                )
                existing_train_reasoning_map = load_reasoning_map(
                    resolved_train_reasoning_jsonl,
                    pid_key=args.reasoning_pid_key,
                    text_key=args.reasoning_text_key,
                )
                train_rows = filter_rows_by_reasoning_ids(
                    train_rows,
                    existing_train_reasoning_map,
                    split_name="train",
                )
                train_jsonl = output_dir / "train_subset_from_reasoning.jsonl"
                write_eval_rows_jsonl(train_rows, train_jsonl)
                fixed_train_reasoning_path = resolved_train_reasoning_jsonl
                log_compact(split_tag("train", "subset-plan"), reasoning_jsonl=resolved_train_reasoning_jsonl, train_jsonl=train_jsonl)
            else:
                fixed_train_cache_root = _resolve_cache_root(
                    "",
                    default_name=f"{args.data_name}_train_cache",
                    dataset=args.data_name,
                )
                train_rows, train_jsonl, fixed_train_reasoning_path = prepare_fixed_subset_cache(
                    args=args,
                    split_name="train",
                    rows=train_rows,
                    source_jsonl=train_jsonl,
                    cache_root=fixed_train_cache_root,
                    cache_note=fixed_cache_note,
                    subset_ratio=args.train_subset_ratio,
                    target_pos_ratio=args.subset_pos_ratio,
                    sample_seed=args.data_seed,
                )
        train_examples = [Example.from_dict(row) for row in train_rows]

    if bool(args.ehr_only_ablation):
        log_compact(
            "Reasoning",
            mode="ehr_only",
            note="reasoning disabled for both train and eval; scorer will use EHR only",
            ignored_reasoning_jsonl=reasoning_jsonl,
            ignored_train_reasoning_jsonl=train_reasoning_jsonl,
            ignored_train_cache_note=(args.train_cache_note if str(args.train_cache_note or "").strip() else None),
            ignored_stage2_student_dir=(args.stage2_student_dir if str(args.stage2_student_dir or "").strip() else None),
            ignored_stage2_student_use_base_model=(True if bool(args.stage2_student_use_base_model) else None),
        )
    else:
        if not bool(args.train_scorer_on_reasoning):
            if requested_train_subset_mode:
                log_compact(
                    "TrainSubset",
                    note="ignored because train_scorer_on_reasoning=False",
                )
            if (
                bool(args.train_drop_empty_baseline)
                or bool(args.train_drop_empty_dx)
                or bool(args.train_drop_dx1)
            ):
                log_compact(
                    "TrainFilter",
                    note="ignored because train_scorer_on_reasoning=False",
                    train_drop_empty_baseline=bool(args.train_drop_empty_baseline),
                    train_drop_empty_dx=bool(args.train_drop_empty_dx),
                    train_drop_dx1=bool(args.train_drop_dx1),
                )
        elif _optional_ratio(args.subset_pos_ratio) is not None and not fixed_train_subset_mode:
            log_compact(
                split_tag("train", "subset-plan"),
                note="subset_pos_ratio ignored because train_subset_ratio=1.0 and train uses full-cache mode",
            )
    if (fixed_train_subset_mode or fixed_test_subset_mode) and not bool(args.ehr_only_ablation):
        if args.do_sample:
            raise ValueError(
                "fixed sampled cache mode is shared, but do_sample=True makes cached reasoning non-deterministic"
            )
    elif reasoning_jsonl and resolved_reasoning_jsonl is None:
        log_compact("TestReasoning", mode="generate_missing", missing_reasoning_jsonl=reasoning_jsonl)
    
    
    prompter = ClinicalReasoningPrompter(args.data_name)
    generator = None

    def get_generator():
        nonlocal generator
        if generator is None:
            stage2_student_dir = str(args.stage2_student_dir or "").strip()
            if not stage2_student_dir and not bool(args.stage2_student_use_base_model):
                raise ValueError("Provide --stage2_student_dir when reasoning must be generated.")
            generator = load_generation_student(
                args=args,
                checkpoint_dir=stage2_student_dir,
            )
        return generator

    train_reasoning_map: dict[str, str] = {}
    train_reasoning_cache_path = output_dir / "train_generated_reasoning.jsonl"
    if args.train_scorer_on_reasoning:
        if bool(args.ehr_only_ablation):
            log_compact("TrainReasoning", mode="skipped_ehr_only")
        elif fixed_train_subset_mode:
            assert fixed_train_reasoning_path is not None
            train_reasoning_cache_path = fixed_train_reasoning_path
            train_reasoning_map = load_reasoning_map_if_exists(
                train_reasoning_cache_path,
                pid_key=args.reasoning_pid_key,
                text_key=args.reasoning_text_key,
            )
            if train_reasoning_jsonl and not fixed_train_from_existing_reasoning:
                log_compact("TrainReasoning", mode="fixed_subset_warm_start", train_reasoning_jsonl=resolve_input_jsonl(train_reasoning_jsonl, args.data_name))
                train_reasoning_map.update(
                    load_reasoning_map(
                        resolve_input_jsonl(train_reasoning_jsonl, args.data_name),
                        pid_key=args.reasoning_pid_key,
                        text_key=args.reasoning_text_key,
                    )
                )
            train_reasoning_map = select_reasoning_for_examples(train_reasoning_map, train_examples)
            if len(train_reasoning_map) < len(train_examples):
                train_reasoning_map = ensure_complete_reasoning_cache(
                    generator=get_generator(),
                    examples=train_examples,
                    prompter=prompter,
                    output_jsonl=train_reasoning_cache_path,
                    existing_map=train_reasoning_map,
                    max_prompt_tokens=args.generator_max_prompt_tokens,
                    max_new_tokens=args.eval_generator_max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    batch_size=args.generation_batch_size,
                    desc="Generate fixed sampled train reasoning",
                )
            log_compact(
                "TrainReasoning",
                mode="fixed_subset_cache",
            )
        else:
            train_cache_note = str(args.train_cache_note or "").strip()
            if train_cache_note:
                train_cache_root = Path(
                    str(args.train_cache_root or "eval_generated_train_cache").strip() or "eval_generated_train_cache"
                )
                if not train_cache_root.is_absolute():
                    train_cache_root = artifact_output_root(args.data_name) / train_cache_root
                train_reasoning_cache_path = train_cache_root / train_cache_note / "train_generated_reasoning.jsonl"
                train_cache_exists = train_reasoning_cache_path.exists()
                write_train_cache_meta(args, train_reasoning_cache_path)
                log_compact(
                    "TrainReasoning",
                    mode="shared_full_cache",
                    train_cache_note=args.train_cache_note,
                    train_cache_jsonl=train_reasoning_cache_path,
                    train_cache_exists=train_cache_exists,
                )
                if args.do_sample:
                    raise ValueError(
                        "train_cache_note is shared, but do_sample=True can make generated reasoning non-deterministic"
                    )

                if train_reasoning_jsonl:
                    log_compact("TrainReasoning", note="train_reasoning_jsonl ignored because train_cache_note is set")
                train_reasoning_map = load_reasoning_map_if_exists(
                    train_reasoning_cache_path,
                    pid_key=args.reasoning_pid_key,
                    text_key=args.reasoning_text_key,
                )
            else:
                if train_reasoning_jsonl:
                    train_reasoning_cache_path = resolve_input_jsonl(train_reasoning_jsonl, args.data_name)
                if train_reasoning_jsonl:
                    log_compact("TrainReasoning", mode="explicit_jsonl", train_reasoning_jsonl=resolve_input_jsonl(train_reasoning_jsonl, args.data_name))
                    train_reasoning_map.update(
                        load_reasoning_map(
                            resolve_input_jsonl(train_reasoning_jsonl, args.data_name),
                            pid_key=args.reasoning_pid_key,
                            text_key=args.reasoning_text_key,
                        )
                    )
            train_reasoning_map.update(
                load_reasoning_map_if_exists(
                    train_reasoning_cache_path,
                    pid_key=args.reasoning_pid_key,
                    text_key=args.reasoning_text_key,
                )
            )
            train_reasoning_map = select_reasoning_for_examples(train_reasoning_map, train_examples)
            train_reasoning_map = ensure_complete_reasoning_cache(
                generator=get_generator(),
                examples=train_examples,
                prompter=prompter,
                output_jsonl=train_reasoning_cache_path,
                existing_map=train_reasoning_map,
                max_prompt_tokens=args.generator_max_prompt_tokens,
                max_new_tokens=args.eval_generator_max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.generation_batch_size,
                desc="Generate full train reasoning",
            )
            log_compact(
                "TrainReasoning",
                mode=("shared_full_cache" if train_cache_note else "full_cache"),
                train_reasoning_cache_jsonl=train_reasoning_cache_path,
                loaded=f"{len(train_reasoning_map)}/{len(train_examples)}",
            )

    if bool(args.ehr_only_ablation):
        log_compact("TestReasoning", mode="skipped_ehr_only")
        reasoning_map = {}
    elif fixed_test_subset_mode:
        assert fixed_test_reasoning_path is not None
        reasoning_map = load_reasoning_map_if_exists(
            fixed_test_reasoning_path,
            pid_key=args.reasoning_pid_key,
            text_key=args.reasoning_text_key,
        )
        if reasoning_jsonl and not fixed_test_from_existing_reasoning:
            log_compact("TestReasoning", mode="fixed_subset_warm_start", reasoning_jsonl=resolve_input_jsonl(reasoning_jsonl, args.data_name))
            reasoning_map.update(
                load_reasoning_map(
                    resolve_input_jsonl(reasoning_jsonl, args.data_name),
                    pid_key=args.reasoning_pid_key,
                    text_key=args.reasoning_text_key,
                )
            )
        reasoning_map = select_reasoning_for_examples(reasoning_map, examples)
        total_examples, existing_reasoning, missing_reasoning = reasoning_coverage_stats(reasoning_map, examples)
        log_compact(
            "TestReasoning",
            mode="fixed_subset_cache",
            requested=total_examples,
            existing=existing_reasoning,
            to_generate=missing_reasoning,
            reasoning_jsonl=fixed_test_reasoning_path,
        )
        if len(reasoning_map) < len(examples):
            reasoning_map = ensure_complete_reasoning_cache(
                generator=get_generator(),
                examples=examples,
                prompter=prompter,
                output_jsonl=fixed_test_reasoning_path,
                existing_map=reasoning_map,
                max_prompt_tokens=args.generator_max_prompt_tokens,
                max_new_tokens=args.eval_generator_max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.generation_batch_size,
                desc="Generate fixed sampled test reasoning",
            )
        log_compact(
            "TestReasoning",
            test_reasoning_cache_jsonl=fixed_test_reasoning_path,
            loaded=f"{len(reasoning_map)}/{len(examples)}",
        )
    elif resolved_reasoning_jsonl is not None:
        log_compact("TestReasoning", mode="explicit_jsonl", reasoning_jsonl=resolved_reasoning_jsonl)
        reasoning_map = load_reasoning_map(
            resolved_reasoning_jsonl,
            pid_key=args.reasoning_pid_key,
            text_key=args.reasoning_text_key,
        )
    else:
        generator = get_generator()
        reasoning_path = output_dir / "generated_reasoning.jsonl"
        total_examples = len(examples)
        log_compact(
            "TestReasoning",
            mode="generate",
            requested=total_examples,
            existing=0,
            to_generate=total_examples,
            reasoning_jsonl=reasoning_path,
        )
        reasoning_map = generate_reasoning_jsonl(
            generator=generator,
            examples=examples,
            prompter=prompter,
            output_jsonl=reasoning_path,
            max_prompt_tokens=args.generator_max_prompt_tokens,
            max_new_tokens=args.eval_generator_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.generation_batch_size,
            desc="generate test reasoning",
        )
        log_compact("TestReasoning", generated_reasoning_jsonl=reasoning_path, loaded=f"{len(reasoning_map)}/{len(examples)}")

    rows, eval_jsonl, _ = prepare_eval_rows_and_jsonl(
        rows=rows,
        eval_jsonl=eval_jsonl,
        output_dir=output_dir,
        drop_empty_baseline_final_eval=args.drop_empty_baseline_final_eval,
        drop_empty_dx_final_eval=args.drop_empty_dx_final_eval,
        drop_dx1=args.drop_dx1,
    )
    if not bool(args.ehr_only_ablation):
        eval_examples = [Example.from_dict(row) for row in rows]
        reasoning_map = select_reasoning_for_examples(reasoning_map, eval_examples)

    generator_provider = None
    if generator is not None:
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.train_scorer_on_reasoning:
        assert train_jsonl is not None
        try:
            scorer_model, scorer_tokenizer = train_scorer_on_reasoning(
                args=args,
                train_jsonl=train_jsonl,
                train_rows=train_rows,
                train_examples=train_examples,
                train_reasoning_map=train_reasoning_map,
                train_reasoning_cache_path=train_reasoning_cache_path,
                generator_provider=generator_provider,
                prompter=prompter,
                eval_jsonl=eval_jsonl,
                eval_reasoning_map=reasoning_map,
                output_dir=output_dir,
            )
        finally:
            if generator_provider is not None:
                generator_provider.close()
    else:
        if not str(args.stage1_scorer_dir or "").strip():
            raise ValueError("Provide --stage1_scorer_dir for eval-only mode, or use --train_scorer_on_reasoning to train from base.")
        scorer_model, scorer_tokenizer = load_stage1_scorer_for_eval(
            args=args,
            checkpoint_dir=args.stage1_scorer_dir,
        )

    log_compact("Final", eval_jsonl=eval_jsonl, examples=len(rows), reasoning_count=len(reasoning_map))
    metrics = evaluate_scorer(
        scorer_model=scorer_model,
        scorer_tokenizer=scorer_tokenizer,
        eval_jsonl=eval_jsonl,
        rows=rows,
        reasoning_map=reasoning_map,
        output_dir=output_dir,
        batch_size=args.eval_batch_size,
        max_length=args.scorer_max_length,
        apply_sigmoid=args.eval_sigmoid,
    )
    log_compact(
        "FinalTest",
        AUROC=f"{metrics['auroc']:.4f}",
        AUPRC=f"{metrics['auprc']:.4f}",
        F1=f"{metrics['f1']:.4f}",
        Sens90=f"{metrics['sensitivity_90']:.4f}",
        Sens95=f"{metrics['sensitivity_95']:.4f}",
        PPV90=f"{metrics['ppv_90']:.4f}",
        PPV95=f"{metrics['ppv_95']:.4f}",
        metrics_json=(output_dir / "metrics.json"),
        scores_jsonl=(output_dir / "scores.jsonl"),
    )


if __name__ == "__main__":
    main()
