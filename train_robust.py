from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import random
import shlex
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainerCallback
from trl import RewardConfig
from tqdm.auto import tqdm

import research3.scorer_phase as scorer_phase
from research3.data import Example, chunk_examples, load_examples, shuffle_examples, stratified_split_train_val, write_examples_jsonl
from research3.generator import (
    ClinicalReasoningPrompter,
    DistillGenerator,
    compute_distill_loss,
    generate_reasoning_cache,
    generalized_jsd_per_token,
    reverse_kl_per_token,
)
from research3.losses import target_token_logprobs_from_logits
from research3.train import artifact_lookup_roots, artifact_output_root
from utils_qwen import qwen_init
import torch.nn as nn


warnings.filterwarnings(
    "ignore",
    message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
)


class _TeeStream:
    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, data):
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def isatty(self):
        return self.primary.isatty()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def _fd_points_to_path(fd: int, path: Path) -> bool:
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return False
    if not target.startswith("/"):
        return False
    try:
        return os.path.samefile(target, path)
    except OSError:
        return False


def install_run_log(output_dir: Path, filename: str = "run.log") -> Path:
    log_path = Path(output_dir) / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _fd_points_to_path(1, log_path) or _fd_points_to_path(2, log_path):
        return log_path

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    def close_log():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()

    atexit.register(close_log)
    return log_path


def format_command(argv: list[str]) -> str:
    if not argv:
        return "CMD: <empty>"

    groups: list[list[str]] = [[shlex.quote(argv[0])]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        group = [shlex.quote(arg)]
        i += 1
        if arg.startswith("--"):
            while i < len(argv) and not argv[i].startswith("--"):
                group.append(shlex.quote(argv[i]))
                i += 1
        groups.append(group)

    lines = ["CMD:"]
    for idx, group in enumerate(groups):
        suffix = " \\" if idx < len(groups) - 1 else ""
        lines.append(f"  {' '.join(group)}{suffix}")
    return "\n".join(lines)


def format_args(args: argparse.Namespace) -> str:
    lines = ["ARGS:"]
    for key, value in vars(args).items():
        lines.append(f"  {key}: {value!r}")
    return "\n".join(lines)


def resolve_testslice_dir(dataset: str) -> Path:
    dataset = str(dataset).strip().lower()
    if dataset == "ad":
        return Path("data", "ADstratified10")
    if dataset == "adrd":
        return Path("data", "ADRDstratified10")
    if dataset == "pd":
        return Path("data", "PDstratified8")
    if dataset == "pd003":
        return Path("data", "PD003stratified8")
    raise ValueError(f"Unsupported dataset for testslice: {dataset}")


def ensure_eval_source_jsonl(
    *,
    output_dir: Path,
    examples: list[Example],
    name: str,
) -> Path:
    path = output_dir / "runtime_jsonl" / f"{name}.jsonl"
    write_examples_jsonl(path, examples)
    return path


def rows_to_reasoning_map(rows: list[dict], *, pid_key: str, text_key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        pid = row.get(pid_key)
        text = row.get(text_key)
        if pid is not None and text is not None:
            out[str(pid)] = str(text)
    return out


def load_legacy_reasoning_maps(
    *,
    data_name: str,
    testslice: list[int],
    pid_key: str,
    text_key: str,
) -> tuple[dict[str, str], dict[str, str], str, list[str]]:
    dataset = str(data_name).strip().lower()
    train_rows: list[dict]
    test_rows: list[dict] = []
    train_source: str
    test_sources: list[str] = []

    if dataset == "ad":
        ratio = 10
        train_source = f"artifacts/phase1/{dataset}/qwen-4b-instruct_phase1_fold0_ratio{ratio}_visit2_text_reasoning_neg5_demo_prompt21_nomemory_0122_102819_39.jsonl"
        train_rows = scorer_phase._read_jsonl(train_source)
        if testslice:
            template = "qwen*final*test_sp{slice}*demo_prompt21_nomemory_0122_102819_39*"
            matched = scorer_phase._find_slice_jsonl_files(
                Path(f"artifacts/phase1/{dataset}"),
                testslice,
                pattern_template=template,
            )
            test_sources = [str(path) for path in matched]
            test_rows = scorer_phase._read_jsonl_multi(matched)
    elif dataset in {"pd", "pd003"}:
        train_source = f"artifacts/phase1/{dataset}/qwen-4b-instruct_phase1_fold0_ratio5_visit6_text_reasoning_demo_prompt0202_nomemory_new7_0207_190249_54.jsonl"
        if dataset == "pd003":
            train_source = "artifacts/phase1/pd/qwen-4b-instruct_phase1_fold0_ratio5_visit6_text_reasoning_demo_prompt0202_nomemory_new7_0207_190249_54.jsonl"
        train_rows = scorer_phase._read_jsonl(train_source)
        if testslice:
            template = "qwen*final*test_sp{slice}*_demo_prompt0202_nomemory_new7_0207_190249_54*"
            matched = scorer_phase._find_slice_jsonl_files(
                Path("artifacts/phase1/pd" if dataset == "pd003" else f"artifacts/phase1/{dataset}"),
                testslice,
                pattern_template=template,
            )
            test_sources = [str(path) for path in matched]
            test_rows = scorer_phase._read_jsonl_multi(matched)
    elif dataset == "adrd":
        train_source = f"artifacts/phase1/{dataset}/qwen-4b-instruct_phase1_fold0_ratio10_text_final_reasoning_train_demo_prompt0202_HASmemory_neg5_0203_204004_92.json"
        train_rows = scorer_phase._read_jsonl(train_source)
        if testslice:
            template = "qwen*final*test_sp{slice}*demo_prompt0202_HASmemory_neg5_0203_204004_92*"
            matched = scorer_phase._find_slice_jsonl_files(
                Path(f"artifacts/phase1/{dataset}"),
                testslice,
                pattern_template=template,
            )
            test_sources = [str(path) for path in matched]
            test_rows = scorer_phase._read_jsonl_multi(matched)
    else:
        raise ValueError(f"Unsupported dataset for legacy reasoning paths: {data_name}")

    train_map = rows_to_reasoning_map(train_rows, pid_key=pid_key, text_key=text_key)
    test_map = rows_to_reasoning_map(test_rows, pid_key=pid_key, text_key=text_key)
    return train_map, test_map, train_source, test_sources


def build_splits(
    *,
    train_jsonl: str,
    output_dir: Path,
    seed: int,
    val_count: int,
) -> tuple[Path, Path | None, list[Example], list[Example] | None]:
    all_examples = load_examples(train_jsonl)
    if int(val_count) > 0:
        train_examples, val_examples = stratified_split_train_val(
            all_examples,
            seed=seed,
            val_count=val_count,
        )
        train_path = ensure_eval_source_jsonl(output_dir=output_dir, examples=train_examples, name="train_split")
        val_path = ensure_eval_source_jsonl(output_dir=output_dir, examples=val_examples, name="val_split")
        return train_path, val_path, train_examples, val_examples
    train_path = ensure_eval_source_jsonl(output_dir=output_dir, examples=all_examples, name="train_full")
    return train_path, None, all_examples, None


def make_reward_config(
    *,
    output_dir: Path,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    max_steps: int,
    save_steps: int,
    log_steps: int,
) -> RewardConfig:
    return RewardConfig(
        output_dir=output_dir,
        learning_rate=float(learning_rate),
        num_train_epochs=1,
        max_steps=int(max_steps),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        max_length=int(max_length),
        logging_steps=int(log_steps),
        save_strategy="steps" if int(save_steps) > 0 else "no",
        save_steps=max(int(save_steps), 1),
        eval_strategy="no",
        center_rewards_coefficient=1e-2,
        dataloader_num_workers=0,
    )


class Research3PeriodicEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        label: str,
        eval_jsonl,
        tokenizer,
        max_length: int,
        batch_size: int,
        eval_every_steps: int,
        apply_sigmoid: bool,
        eval_fast_ratio: float = 0.0,
        eval_fast_min_per_class: int = 1,
        eval_fast_seed: int = 0,
        reasoning_map: dict[str, str] | None = None,
    ) -> None:
        self.label = str(label)
        self.eval_jsonl = eval_jsonl
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.eval_every_steps = int(eval_every_steps)
        self.eval_fast_ratio = float(eval_fast_ratio)
        self.eval_fast_min_per_class = int(eval_fast_min_per_class)
        self.eval_fast_seed = int(eval_fast_seed)
        self.apply_sigmoid = bool(apply_sigmoid)
        self.reasoning_map = reasoning_map
        self.eval_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            return_tensors="pt",
        )
        self.test_ds = None
        self.fast_pos_idx = None
        self.fast_neg_idx = None
        self.fast_ds = None

    def _build_fast_indices(self) -> None:
        if self.test_ds is None:
            return
        labels = self.test_ds["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = scorer_phase.np.array(labels)
        self.fast_pos_idx = scorer_phase.np.where(labels == 1)[0]
        self.fast_neg_idx = scorer_phase.np.where(labels == 0)[0]

    def _build_fast_subset(self, seed: int):
        if self.fast_pos_idx is None or self.fast_neg_idx is None:
            self._build_fast_indices()
        if self.fast_pos_idx is None or self.fast_neg_idx is None:
            return self.test_ds
        rng = scorer_phase.np.random.default_rng(int(seed) + 13)
        n_pos_total = len(self.fast_pos_idx)
        n_neg_total = len(self.fast_neg_idx)
        if n_pos_total == 0 or n_neg_total == 0:
            return self.test_ds
        n_pos = max(self.eval_fast_min_per_class, int(round(n_pos_total * self.eval_fast_ratio)))
        n_neg = max(self.eval_fast_min_per_class, int(round(n_neg_total * self.eval_fast_ratio)))
        n_pos = min(n_pos, n_pos_total)
        n_neg = min(n_neg, n_neg_total)
        pos_sample = rng.choice(self.fast_pos_idx, size=n_pos, replace=False)
        neg_sample = rng.choice(self.fast_neg_idx, size=n_neg, replace=False)
        sel = scorer_phase.np.concatenate([pos_sample, neg_sample])
        rng.shuffle(sel)
        return self.test_ds.select(sel.tolist())

    def _print_metrics(self, *, prefix: str, step: int, scores, labels) -> None:
        auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95 = scorer_phase.get_evaluation_metrics(labels, scores)
        print(
            f"[research3][{self.label}][{prefix}@{step}] "
            f"AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"F1={f1:.4f} Sens@90%={sensitivity_90:.4f} Sens@95%={sensitivity_95:.4f} "
            f"PPV@90%={ppv_90:.4f} PPV@95%={ppv_95:.4f}"
        )

    def _run_eval(self, model, step: int, *, fast: bool) -> None:
        ds = self.fast_ds if fast and self.fast_ds is not None else self.test_ds
        if ds is None:
            return
        model.eval()
        scores, labels = eval_pointwise_with_progress(
            model,
            self.tokenizer,
            ds,
            text_key="text",
            label_key="label",
            batch_size=self.batch_size,
            max_length=self.max_length,
            apply_sigmoid=self.apply_sigmoid,
            collator=self.eval_collator,
            progress_desc=None,
        )
        self._print_metrics(prefix="EVAL_FAST" if fast else "EVAL", step=step, scores=scores, labels=labels)
        model.train()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_jsonl:
            raw_ds = scorer_phase.build_pointwise_dataset_baseline(
                self.eval_jsonl,
                reasoning_map=self.reasoning_map,
                follow_up=False
            )
            self.test_ds = scorer_phase._pretokenize_pointwise_dataset(
                raw_ds,
                self.tokenizer,
                self.max_length,
            )
            self._build_fast_indices()
            self.fast_ds = self._build_fast_subset(self.eval_fast_seed)
        if self.test_ds is None:
            return
        self._run_eval(kwargs["model"], state.global_step, fast=False)

    def on_step_end(self, args, state, control, **kwargs):
        if self.test_ds is None:
            return control
        fast_every = max(1, self.eval_every_steps // 10) if self.eval_every_steps else 0
        if self.eval_every_steps and state.global_step > 0 and state.global_step % self.eval_every_steps == 0:
            self._run_eval(kwargs["model"], state.global_step, fast=False)
            return control
        if self.eval_fast_ratio > 0 and fast_every and state.global_step > 0 and state.global_step % fast_every == 0:
            self._run_eval(kwargs["model"], state.global_step, fast=True)
        return control

    def single_eval(self, model, step: int) -> None:
        self._run_eval(model, step, fast=False)




def ensure_score_bias(model):
    if not hasattr(model, "score"):
        raise ValueError("Model has no score head.")
    old = model.score
    if not isinstance(old, nn.Linear):
        raise TypeError(f"Expected model.score to be nn.Linear, got {type(old).__name__}")
    if old.bias is not None:
        return model

    new = nn.Linear(
        old.in_features,
        old.out_features,
        bias=True,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.bias.zero_()

    model.score = new
    return model


def build_stage1_lora_config(model, lora_last_n: int, lora_r: int) -> LoraConfig:
    kwargs: dict[str, object] = dict(
        r=int(lora_r),
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
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


def train_stage1_scorer(
    *,
    args,
    train_jsonl: str,
    eval_targets: list[dict[str, object]],
    train_reasoning_map: dict[str, str],
    output_dir: Path,
    follow_up_for_phase1=None
):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.scorer_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    if args.score_bias:
        model = ensure_score_bias(model)
    scorer_phase._reset_score_head_with_seed(model, args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    print("\n===== Tokenizer info =====")
    print("\tbos:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("\teos:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("\tpad:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("\tpadding_side:", tokenizer.padding_side)

    train_ds, _, pos_count = scorer_phase.build_pref_iterable_dataset_epoch_baseline(
        in_jsonl=str(train_jsonl),
        neg_per_pos=args.neg_per_pos,
        base_seed=args.data_seed,
        prompt=None,
        include_meta=False,
        reasoning_map=train_reasoning_map,
        follow_up=follow_up_for_phase1
    )
    # print(train_ds)
    # sys.exit()
    pairs_per_epoch = pos_count * int(args.neg_per_pos)
    steps_per_epoch = math.ceil(pairs_per_epoch / (int(args.stage1_batch_size) * int(args.stage1_grad_accum)))
    max_steps = steps_per_epoch * int(args.stage1_epochs)

    rcfg = make_reward_config(
        output_dir=output_dir,
        learning_rate=args.stage1_lr,
        batch_size=args.stage1_batch_size,
        grad_accum=args.stage1_grad_accum,
        max_length=args.scorer_max_length,
        max_steps=max_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
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
    else:
        peft_config = build_stage1_lora_config(
            model,
            args.stage1_lora_last_n,
            args.stage1_lora_r,
        )
    trainer = scorer_phase.LossSwitchRewardTrainer(
        model=model,
        args=rcfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        loss_type=args.loss_type,
        pointwise_alpha=args.pointwise_alpha,
        pairwise_margin=args.pairwise_margin,
        pos_weight=args.pos_weight,
        margin=args.margin,
        margin_on_sigmoid=args.margin_on_sigmoid,
    )
    if int(args.eval_steps) > 0:
        for target in eval_targets:
            trainer.add_callback(
                Research3PeriodicEvalCallback(
                    label=str(target["name"]).upper(),
                    eval_jsonl=target["jsonl"],
                    tokenizer=tokenizer,
                    max_length=args.scorer_max_length,
                    batch_size=args.eval_batch_size,
                    eval_every_steps=args.eval_steps,
                    apply_sigmoid=args.eval_sigmoid,
                    eval_fast_ratio=args.eval_fast_ratio,
                    eval_fast_min_per_class=args.eval_fast_min_per_class,
                    eval_fast_seed=args.eval_fast_seed,
                    reasoning_map=target["reasoning_map"],
                )
            )

    print("\n===== RESEARCH3 STAGE1 =====")
    print(f"train_jsonl={train_jsonl}")
    print("\n===== Dynamic dataset info =====")
    print(f"|seed {args.seed}")
    print(f"|data_seed {args.data_seed}")
    print(f"|pos_count {pos_count}")
    print(f"|neg_per_pos {args.neg_per_pos}")
    print(f"|pairs_per_epoch:  {pairs_per_epoch}")
    print(f"|training batch_size:  {args.stage1_batch_size}")
    print(f"|training grad_accum:  {args.stage1_grad_accum}")
    print(f"|gradient steps per epoch:  {steps_per_epoch}")
    print(f"|all gradient steps:  {max_steps}")
    print("\n===== RESEARCH3 STAGE1 TRAINABLE PARAMETERS =====")
    if args.stage1_score_only:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        ratio = (trainable / total * 100.0) if total > 0 else 0.0
        print(
            f"mode=score_only trainable_params={trainable} "
            f"all_params={total} trainable_percent={ratio:.4f}"
        )
    else:
        trainer.model.print_trainable_parameters()
        print(
            f"mode=last_n_plus_score lora_r={args.stage1_lora_r} "
            f"lora_last_n={args.stage1_lora_last_n}"
        )
    trainer.train()
    if args.stage1_score_only:
        trainer.model.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.model, tokenizer


def resolve_stage1_scorer_dir(path_str: str, dataset: str | None = None) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        for root in artifact_lookup_roots(dataset):
            candidates.append(root / path_str)
    for candidate in candidates:
        if candidate.exists():
            print('Resolve stage1 cache:', candidate)
            return candidate
    raise FileNotFoundError(f"Unable to find stage1 scorer checkpoint: {path_str}")


def load_stage1_scorer(
    *,
    args,
    checkpoint_dir: str,
):
    checkpoint_path = resolve_stage1_scorer_dir(checkpoint_dir, args.data_name)
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
        for attempt_bias in ([bool(args.score_bias)] if args.score_bias else [False, True]):
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
                break
            except RuntimeError as exc:
                load_error = exc
                if attempt_bias:
                    raise
        if load_error is not None:
            raise load_error
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
    print("\n===== RESEARCH3 STAGE1 LOAD =====")
    print(f"stage1_scorer_dir={checkpoint_path}")
    return model, tokenizer


def evaluate_with_frozen_scorer(
    *,
    scorer_model,
    scorer_tokenizer,
    eval_jsonl,
    reasoning_map: dict[str, str],
    max_length: int,
    batch_size: int,
    apply_sigmoid: bool,
) -> tuple[float, float, float, float, float, float, float]:
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
    return scorer_phase.get_evaluation_metrics(labels, scores)


def eval_pointwise_with_progress(
    model,
    tokenizer,
    test_dataset,
    *,
    text_key="text",
    label_key="label",
    batch_size=8,
    max_length=10000,
    device=None,
    apply_sigmoid=False,
    collator=None,
    progress_desc: str | None = None,
):
    model.eval()
    scores_all, labels_all = [], []
    raw_scores_all = []
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    n = len(test_dataset)
    iterator = range(0, n, batch_size)
    if progress_desc:
        iterator = tqdm(iterator, total=math.ceil(n / batch_size), desc=progress_desc, leave=False)

    with torch.inference_mode():
        for i in iterator:
            batch = test_dataset[i : i + batch_size]
            labels = batch[label_key]
            if "input_ids" in batch:
                features = [
                    {
                        "input_ids": batch["input_ids"][j],
                        "attention_mask": batch["attention_mask"][j],
                    }
                    for j in range(len(labels))
                ]
                if collator is None:
                    collator = DataCollatorWithPadding(
                        tokenizer=tokenizer,
                        padding="longest",
                        return_tensors="pt",
                    )
                enc = collator(features)
            else:
                texts = batch[text_key]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            scores = out.logits.squeeze(-1)
            raw_scores_all.append(scores.detach().float().cpu())
            if apply_sigmoid:
                scores = torch.sigmoid(scores)

            scores_all.append(scores.detach().float().cpu())
            if isinstance(labels, torch.Tensor):
                labels_all.append(labels.detach().to(dtype=torch.int64).cpu())
            else:
                labels_all.append(torch.tensor(labels, dtype=torch.int64))

    raw_scores = torch.cat(raw_scores_all).numpy()
    scores = torch.cat(scores_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return scores, labels


def generate_eval_reasoning_map(
    *,
    generator: DistillGenerator,
    examples: list[Example],
    prompter: ClinicalReasoningPrompter,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> dict[str, str]:
    reasoning_map: dict[str, str] = {}
    for batch_examples in chunk_examples(examples, batch_size=batch_size):
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
            reasoning_map[str(example.patient_id)] = str(text)
    return reasoning_map

def decode_rollout(rollout, system, i):
    ids = rollout.sequence_ids[i][rollout.attention_mask[i].bool()] 
    completion_ids = rollout.completion_ids[i][rollout.completion_attention_mask[i].bool()]
    print('\nDecoding...')
    print("Sequence ids:", system.tokenizer.decode(ids, skip_special_tokens=False))
    print("Sequence completion_ids:", system.tokenizer.decode(completion_ids, skip_special_tokens=False))


def print_stage2_debug_rollout_examples(
    *,
    batch_examples: list[Example],
    rollout,
    debug_prefix: str,
) -> None:
    for sample_idx, (example, text) in enumerate(zip(batch_examples, rollout.texts)):
        print(
            f"\n\t{debug_prefix}[RolloutExample][sample={sample_idx}] "
            f"id={example.patient_id} label={example.label} age={example.age} sex={example.sex}",
            flush=True,
        )
        print(
            f"\t{debug_prefix}[RolloutExample][sample={sample_idx}] "
            f"baseline_diagnoses_n={len(example.base_codes_diagnosis)} "
            f"baseline_diagnoses={json.dumps(example.base_codes_diagnosis, ensure_ascii=False)}",
            flush=True,
        )
        print(
            f"\t{debug_prefix}[RolloutExample][sample={sample_idx}] "
            f"baseline_medications_n={len(example.base_codes_medication)} "
            f"baseline_medications={json.dumps(example.base_codes_medication, ensure_ascii=False)}",
            flush=True,
        )
        print(
            f"\t{debug_prefix}[StudentRollout][sample={sample_idx}] "
            f"{str(text)!r}",
            flush=True,
        )


def append_stage2_rollout_jsonl(
    *,
    output_jsonl: Path,
    batch_examples: list[Example],
    teacher_prompts,
    rollout,
    epoch: int,
    total_stage2_epochs: int,
    global_step: int,
    current_micro_batch: int,
    total_micro_batches: int,
    processed_examples: int,
    total_examples: int,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for sample_idx, (example, rollout_text) in enumerate(zip(batch_examples, rollout.texts)):
            completion_length = None
            hit_cap = None
            if sample_idx < int(rollout.completion_lengths.shape[0]):
                completion_length = int(rollout.completion_lengths[sample_idx].detach().cpu().item())
            if sample_idx < int(rollout.hit_cap_mask.shape[0]):
                hit_cap = bool(rollout.hit_cap_mask[sample_idx].detach().cpu().item())
            row = {
                "epoch": int(epoch + 1),
                "total_stage2_epochs": int(total_stage2_epochs),
                "global_step": int(global_step),
                "micro_batch": int(current_micro_batch),
                "total_micro_batches": int(total_micro_batches),
                "samples_seen": int(processed_examples),
                "total_examples": int(total_examples),
                "sample_idx_in_batch": int(sample_idx),
                "id": str(example.patient_id),
                "label": int(example.label),
                "age": None if example.age is None else int(example.age),
                "sex": example.sex,
                "base_codes_diagnosis": list(example.base_codes_diagnosis),
                "base_codes_medication": list(example.base_codes_medication),
                "delta_codes_diagnosis": list(example.delta_codes_diagnosis),
                "delta_codes_medication": list(example.delta_codes_medication),
                "teacher_prompt_text": str(teacher_prompts.texts[sample_idx]),
                "rollout_text": str(rollout_text),
                "completion_length": completion_length,
                "hit_cap": hit_cap,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_seed_int(*parts: object) -> int:
    joined = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)


def _perturb_followup_list(
    values: list[str],
    *,
    dropout: float,
    min_keep: int,
    rng: random.Random,
) -> list[str]:
    rows = list(values or [])
    if not rows:
        return []
    dropout = min(max(float(dropout), 0.0), 1.0)
    if dropout <= 0.0:
        return rows

    keep_mask = [rng.random() > dropout for _ in rows]
    kept_count = sum(1 for keep in keep_mask if keep)
    target_keep = min(len(rows), max(int(min_keep), 0))
    if kept_count < target_keep:
        missing = [idx for idx, keep in enumerate(keep_mask) if not keep]
        rng.shuffle(missing)
        for idx in missing[: target_keep - kept_count]:
            keep_mask[idx] = True
    return [value for value, keep in zip(rows, keep_mask) if keep]


def _build_perturbed_privileged_example(
    example: Example,
    *,
    seed: int,
    dx_dropout: float,
    med_dropout: float,
    min_keep: int,
) -> Example:
    rng = random.Random(int(seed))
    delta_dx = _perturb_followup_list(
        example.delta_codes_diagnosis,
        dropout=dx_dropout,
        min_keep=min_keep,
        rng=rng,
    )
    delta_md = _perturb_followup_list(
        example.delta_codes_medication,
        dropout=med_dropout,
        min_keep=min_keep,
        rng=rng,
    )
    raw = dict(example.raw)
    raw["delta_codes_diagnosis"] = list(delta_dx)
    raw["delta_codes_medication"] = list(delta_md)
    return replace(
        example,
        delta_codes_diagnosis=list(delta_dx),
        delta_codes_medication=list(delta_md),
        raw=raw,
    )

def build_robust_privileged_teacher_stats(
    *,
    generator: DistillGenerator,
    prompter: ClinicalReasoningPrompter,
    batch_examples: list[Example],
    teacher_prompts,
    completion_ids: torch.Tensor,
    completion_attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
    max_length: int,
    disable_adapter: bool,
    seed: int,
    epoch: int,
    batch_step: int,
    num_perturbations: int,
    include_unperturbed: bool,
    dx_dropout: float,
    med_dropout: float,
    min_keep: int,
    student_logits: torch.Tensor,
    student_divergence_kind: str,
    jsd_beta: float,
    jsd_temperature: float,
    jsd_top_k: int | None,
    jsd_token_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    teacher_logits_sum = None
    logprob_mean = None
    logprob_m2 = None
    student_divergence_sum = None
    view_count = 0

    def add_teacher_view(prompt_batch) -> None:
        nonlocal teacher_logits_sum, logprob_mean, logprob_m2, student_divergence_sum, view_count
        with torch.no_grad():
            view_logits = generator.teacher_forward_on_rollouts(
                teacher_prompt_ids=prompt_batch.input_ids,
                teacher_prompt_attention_mask=prompt_batch.attention_mask,
                teacher_prompt_length=prompt_batch.prompt_length,
                completion_ids=completion_ids,
                completion_attention_mask=completion_attention_mask,
                disable_adapter=disable_adapter,
            ).float()
        teacher_logits_sum = view_logits if teacher_logits_sum is None else (teacher_logits_sum + view_logits)
        if str(student_divergence_kind) == "reverse_kl":
            per_view_student_divergence = reverse_kl_per_token(
                student_logits=student_logits,
                teacher_logits=view_logits,
                temperature=jsd_temperature,
                top_k=jsd_top_k,
                token_clip=jsd_token_clip,
            )
        else:
            per_view_student_divergence = generalized_jsd_per_token(
                student_logits=student_logits,
                teacher_logits=view_logits,
                beta=jsd_beta,
                temperature=jsd_temperature,
                top_k=jsd_top_k,
                token_clip=jsd_token_clip,
            )
        student_divergence_sum = (
            per_view_student_divergence
            if student_divergence_sum is None
            else (student_divergence_sum + per_view_student_divergence)
        )
        with torch.no_grad():
            selected_logprobs = target_token_logprobs_from_logits(view_logits, target_ids)
        view_count += 1
        if logprob_mean is None:
            logprob_mean = selected_logprobs
            logprob_m2 = torch.zeros_like(selected_logprobs)
        else:
            delta = selected_logprobs - logprob_mean
            logprob_mean = logprob_mean + delta / float(view_count)
            logprob_m2 = logprob_m2 + delta * (selected_logprobs - logprob_mean)

    if bool(include_unperturbed):
        add_teacher_view(teacher_prompts)

    for perturb_idx in range(int(num_perturbations)):
        perturbed_examples = []
        for sample_idx, example in enumerate(batch_examples):
            sample_seed = _stable_seed_int(
                "robust_privileged_teacher",
                seed,
                epoch,
                batch_step,
                perturb_idx,
                sample_idx,
                example.patient_id,
            )
            perturbed_examples.append(
                _build_perturbed_privileged_example(
                    example,
                    seed=sample_seed,
                    dx_dropout=dx_dropout,
                    med_dropout=med_dropout,
                    min_keep=min_keep,
                )
            )
        perturbed_teacher_prompts = generator.build_teacher_prompt_batch(
            perturbed_examples,
            prompter=prompter,
            max_length=max_length,
        )
        add_teacher_view(perturbed_teacher_prompts)

    if (
        view_count <= 0
        or teacher_logits_sum is None
        or logprob_mean is None
        or logprob_m2 is None
        or student_divergence_sum is None
    ):
        raise ValueError("Robust privileged teacher requires at least one teacher view.")

    mean_teacher_logits = teacher_logits_sum / float(view_count)
    mean_student_divergence = student_divergence_sum / float(view_count)
    if view_count == 1:
        logprob_std = torch.zeros_like(logprob_mean)
    else:
        logprob_std = torch.sqrt((logprob_m2 / float(view_count)).clamp_min(0.0))
    return mean_teacher_logits, logprob_mean, logprob_std, mean_student_divergence, view_count


def run_stage2_distill(
    *,
    args,
    train_examples: list[Example],
    eval_examples: list[Example] | None,
    eval_jsonl,
    output_dir: Path,
    frozen_scorer_model=None,
    frozen_scorer_tokenizer=None,
):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    rollout_dump_every_steps = max(int(args.stage2_rollout_dump_every_steps), 0)
    rollout_dump_path = None
    if rollout_dump_every_steps > 0:
        rollout_dump_path_str = str(args.stage2_rollout_dump_jsonl or "").strip()
        rollout_dump_path = Path(rollout_dump_path_str) if rollout_dump_path_str else (output_dir / "stage2_rollout_dump.jsonl")
        rollout_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rollout_dump_path, "w", encoding="utf-8"):
            pass
        print("\n===== RESEARCH3 STAGE2 ROLLOUT DUMP =====")
        print(f"stage2_rollout_dump_every_steps={rollout_dump_every_steps}")
        print(f"stage2_rollout_dump_jsonl={rollout_dump_path}")

    generator = DistillGenerator(
        model_name=args.generator_model_name,
        device=device,
        torch_dtype=args.generator_torch_dtype,
        lora_r=args.generator_lora_r,
        lora_alpha=args.generator_lora_alpha,
        lora_dropout=args.generator_lora_dropout,
        lora_last_n=args.generator_lora_last_n,
    )
    print('Set up generator model...')
    generator.freeze_all()
    generator.enable_lora_updates() # free lora layers
    optimizer = torch.optim.AdamW(generator.trainable_parameters(), lr=float(args.stage2_lr))
    total = sum(param.numel() for param in generator.model.parameters())
    trainable = sum(param.numel() for param in generator.model.parameters() if param.requires_grad)
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    print("\n===== RESEARCH3 STAGE2 TRAINABLE PARAMETERS =====")
    print(
        f"mode=generator_lora_only trainable_params={trainable} "
        f"all_params={total} trainable_percent={ratio:.4f}"
    )
    print(
        f"generator_lora_r={args.generator_lora_r} "
        f"generator_lora_last_n={args.generator_lora_last_n}"
    )
    print(f"stage2_teacher_mode={args.stage2_teacher_mode}")
    use_privileged_delta = str(args.stage2_distill_loss_type) == "privileged_delta_jsd"
    use_robust_privileged_delta = str(args.stage2_distill_loss_type) in {"robust_privileged_delta_jsd", "robust_privileged_delta_reverse_kl"}
    teacher_disable_adapter = args.stage2_teacher_mode == "fixed_base" or use_privileged_delta or use_robust_privileged_delta
    print(f"teacher_disable_adapter={teacher_disable_adapter}")
    if use_privileged_delta or use_robust_privileged_delta:
        print(
            f"privileged_delta_head_tokens={args.stage2_delta_head_tokens} "
            f"privileged_delta_tail_weight={args.stage2_delta_tail_weight}"
        )
    if use_robust_privileged_delta:
        print(
            f"robust_privileged_views={int(args.stage2_robust_num_perturbations) + int(bool(args.stage2_robust_include_unperturbed))} "
            f"include_unperturbed={bool(args.stage2_robust_include_unperturbed)} "
            f"dx_dropout={float(args.stage2_robust_dx_dropout):.2f} "
            f"med_dropout={float(args.stage2_robust_med_dropout):.2f}"
        )
        print(
            f"robust_gate_mode={args.stage2_robust_gate_mode} "
            f"robust_gate_scale={float(args.stage2_robust_gate_scale):.3f} "
            f"robust_gate_threshold={float(args.stage2_robust_gate_threshold):.3f}"
        )

    if frozen_scorer_model is not None:
        print('Will use frozen scorer model...')
        for param in frozen_scorer_model.parameters():
            param.requires_grad = False
        frozen_scorer_model.eval()
    else:
        print('No frozen scorer model; stage2 periodic eval is disabled.')

    
    prompter = ClinicalReasoningPrompter(args.data_name)
    print("\n===== RESEARCH3 STAGE2 =====")
    print(f"train_examples={len(train_examples)}")
    print(f"distill_loss_type={args.stage2_distill_loss_type}")
    if eval_examples is not None:
        print(f"eval_examples={len(eval_examples)}")

    total_stage2_epochs = int(args.stage2_epochs)
    stage2_batch_size = int(args.stage2_batch_size)
    stage2_grad_accum = max(int(args.stage2_grad_accum), 1)
    log_every_steps = max(int(args.log_every_steps), 1)

    global_step = 0
    for epoch in range(total_stage2_epochs):
        epoch_examples = shuffle_examples(train_examples, seed=args.data_seed + epoch)
        train_batches = chunk_examples(epoch_examples, batch_size=stage2_batch_size)
        total_micro_batches = len(train_batches)
        optimizer_steps_this_epoch = math.ceil(total_micro_batches / stage2_grad_accum) if total_micro_batches else 0
        print(f"stage2_epoch={epoch + 1}/{total_stage2_epochs}")
        print(f"stage2_batch_size={stage2_batch_size}")
        print(f"stage2_grad_accum={stage2_grad_accum}")
        print(f"stage2_micro_batches={total_micro_batches}")
        print(f"stage2_optimizer_steps_this_epoch={optimizer_steps_this_epoch}")
        
        # print('Train_batches:', len(train_batches), )
        optimizer.zero_grad(set_to_none=True)
        print('Start iterative steps...')
        accum_distill: list[float] = []
        accum_coverage: list[float] = []
        accum_gap: list[float] = []
        accum_rollout: list[float] = []
        accum_hit_cap: list[float] = []
        accum_masked: list[float] = []
        processed_examples = 0
        
        for batch_step, batch_examples in tqdm(
            enumerate(train_batches, start=0),
            total=total_micro_batches,
            desc=f"stage2 epoch {epoch + 1}/{total_stage2_epochs}",
        ):
            current_micro_batch = batch_step + 1
            processed_examples += len(batch_examples)
            debug_this_batch = (
                int(args.distill_debug_top_k) > 0
                and batch_step % max(int(args.distill_debug_every_batches), 1) == 0
            )
            debug_prefix = (
                f"[Distill Debug] epoch={epoch + 1}/{total_stage2_epochs} "
                f"micro_batch={current_micro_batch}/{total_micro_batches} "
            )
            generator.model.train()
            student_prompts = generator.build_student_prompt_batch(
                batch_examples,
                prompter=prompter,
                max_length=args.generator_max_prompt_tokens,
            )
            teacher_prompts = generator.build_teacher_prompt_batch(
                batch_examples,
                prompter=prompter,
                max_length=args.generator_max_prompt_tokens,
            )
            baseline_teacher_prompts = None
            if use_privileged_delta or use_robust_privileged_delta:
                baseline_teacher_prompts = generator.build_baseline_teacher_prompt_batch(
                    batch_examples,
                    prompter=prompter,
                    max_length=args.generator_max_prompt_tokens,
                )

            
            rollout = generator.generate_rollouts(
                prompt_ids=student_prompts.input_ids,
                prompt_attention_mask=student_prompts.attention_mask,
                max_new_tokens=args.generator_max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            if debug_this_batch:
                print_stage2_debug_rollout_examples(
                    batch_examples=batch_examples,
                    rollout=rollout,
                    debug_prefix=debug_prefix,
                )
            student_pass = generator.student_forward_on_rollouts(
                generated_ids=rollout.sequence_ids,
                generated_attention_mask=rollout.attention_mask,
                student_prompt_length=student_prompts.prompt_length,
                completion_ids=rollout.completion_ids,
                completion_attention_mask=rollout.completion_attention_mask,
            )
            privileged_teacher_logprob_mean = None
            privileged_teacher_logprob_std = None
            privileged_teacher_student_divergence_mean = None
            robust_teacher_views = 0
            if use_robust_privileged_delta:
                (
                    teacher_logits,
                    privileged_teacher_logprob_mean,
                    privileged_teacher_logprob_std,
                    privileged_teacher_student_divergence_mean,
                    robust_teacher_views,
                ) = build_robust_privileged_teacher_stats(
                    generator=generator,
                    prompter=prompter,
                    batch_examples=batch_examples,
                    teacher_prompts=teacher_prompts,
                    completion_ids=rollout.completion_ids,
                    completion_attention_mask=rollout.completion_attention_mask,
                    target_ids=student_pass.rollout_token_ids,
                    max_length=args.generator_max_prompt_tokens,
                    disable_adapter=teacher_disable_adapter,
                    seed=args.seed,
                    epoch=epoch,
                    batch_step=batch_step,
                    num_perturbations=args.stage2_robust_num_perturbations,
                    include_unperturbed=args.stage2_robust_include_unperturbed,
                    dx_dropout=args.stage2_robust_dx_dropout,
                    med_dropout=args.stage2_robust_med_dropout,
                    min_keep=args.stage2_robust_min_keep,
                    student_logits=student_pass.rollout_logits,
                    student_divergence_kind=(
                        "reverse_kl"
                        if str(args.stage2_distill_loss_type) == "robust_privileged_delta_reverse_kl"
                        else "jsd"
                    ),
                    jsd_beta=args.stage2_jsd_beta,
                    jsd_temperature=args.stage2_jsd_temperature,
                    jsd_top_k=(args.stage2_jsd_top_k if int(args.stage2_jsd_top_k) > 0 else None),
                    jsd_token_clip=args.stage2_jsd_token_clip,
                )
            else:
                teacher_logits = generator.teacher_forward_on_rollouts(
                    teacher_prompt_ids=teacher_prompts.input_ids,
                    teacher_prompt_attention_mask=teacher_prompts.attention_mask,
                    teacher_prompt_length=teacher_prompts.prompt_length,
                    completion_ids=rollout.completion_ids,
                    completion_attention_mask=rollout.completion_attention_mask,
                    disable_adapter=teacher_disable_adapter,
                )
            baseline_teacher_logits = None
            if baseline_teacher_prompts is not None:
                baseline_teacher_logits = generator.teacher_forward_on_rollouts(
                    teacher_prompt_ids=baseline_teacher_prompts.input_ids,
                    teacher_prompt_attention_mask=baseline_teacher_prompts.attention_mask,
                    teacher_prompt_length=baseline_teacher_prompts.prompt_length,
                    completion_ids=rollout.completion_ids,
                    completion_attention_mask=rollout.completion_attention_mask,
                    disable_adapter=True,
                )
            if debug_this_batch and use_robust_privileged_delta and privileged_teacher_logprob_std is not None:
                valid_std = privileged_teacher_logprob_std[student_pass.rollout_attention_mask.bool()]
                std_mean = float(valid_std.mean().detach().cpu().item()) if valid_std.numel() > 0 else float("nan")
                std_max = float(valid_std.max().detach().cpu().item()) if valid_std.numel() > 0 else float("nan")
                print(
                    f"\t{debug_prefix}[RobustTeacher] views={robust_teacher_views} "
                    f"teacher_std_mean={std_mean:.6f} teacher_std_max={std_max:.6f}",
                    flush=True,
                )
            distill_loss, masked_tokens, mask_coverage, gap_abs_mean = compute_distill_loss(
                teacher_logits=teacher_logits,
                student_logits=student_pass.rollout_logits,
                target_ids=student_pass.rollout_token_ids,
                completion_attention_mask=student_pass.rollout_attention_mask,
                baseline_teacher_logits=baseline_teacher_logits,
                privileged_teacher_logprob_mean=privileged_teacher_logprob_mean,
                privileged_teacher_logprob_std=privileged_teacher_logprob_std,
                privileged_teacher_student_divergence_mean=privileged_teacher_student_divergence_mean,
                tau=args.tau,
                loss_type=args.stage2_distill_loss_type,
                jsd_beta=args.stage2_jsd_beta,
                jsd_temperature=args.stage2_jsd_temperature,
                jsd_top_k=(args.stage2_jsd_top_k if int(args.stage2_jsd_top_k) > 0 else None),
                jsd_token_clip=args.stage2_jsd_token_clip,
                delta_head_tokens=args.stage2_delta_head_tokens,
                delta_tail_weight=args.stage2_delta_tail_weight,
                robust_gate_mode=args.stage2_robust_gate_mode,
                robust_gate_scale=args.stage2_robust_gate_scale,
                robust_gate_threshold=args.stage2_robust_gate_threshold,
                tokenizer=generator.tokenizer,
                debug_top_k=int(args.distill_debug_top_k) if debug_this_batch else 0,
                debug_prefix=debug_prefix,
            )
            
            accum_distill.append(float(distill_loss.detach().cpu().item()))
            accum_coverage.append(float(mask_coverage))
            if not math.isnan(gap_abs_mean):
                accum_gap.append(float(gap_abs_mean))
            accum_rollout.append(float(rollout.completion_lengths.float().mean().detach().cpu().item()))
            accum_hit_cap.append(float(rollout.hit_cap_mask.float().mean().detach().cpu().item()))
            accum_masked.append(float(masked_tokens))
            
            (distill_loss / stage2_grad_accum).backward()
            should_step = (
                (current_micro_batch % stage2_grad_accum == 0)
                or (current_micro_batch == total_micro_batches)
            )
            if should_step:
                global_step += 1
                progress = (
                    f"epoch={epoch + 1}/{total_stage2_epochs} "
                    f"global_step={global_step} "
                    f"micro_batch={current_micro_batch}/{total_micro_batches} "
                    f"samples_seen={processed_examples}/{len(epoch_examples)}"
                )
                update_distill = sum(accum_distill) / max(len(accum_distill), 1)
                update_coverage = sum(accum_coverage) / max(len(accum_coverage), 1)
                update_gap = sum(accum_gap) / len(accum_gap) if accum_gap else float("nan")
                update_rollout = sum(accum_rollout) / max(len(accum_rollout), 1)
                update_hit_cap = sum(accum_hit_cap) / max(len(accum_hit_cap), 1)
                update_masked = sum(accum_masked) / max(len(accum_masked), 1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                accum_distill.clear()
                accum_coverage.clear()
                accum_gap.clear()
                accum_rollout.clear()
                accum_hit_cap.clear()
                accum_masked.clear()

                if (
                    int(args.stage2_save_steps) > 0
                    and global_step % int(args.stage2_save_steps) == 0
                ):
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    generator.save_pretrained(checkpoint_dir)
                    print(
                        f"\n[Save] stage2 {progress} checkpoint={checkpoint_dir}",
                        flush=True,
                    )

                if (
                    rollout_dump_path is not None
                    and rollout_dump_every_steps > 0
                    and global_step % rollout_dump_every_steps == 0
                ):
                    append_stage2_rollout_jsonl(
                        output_jsonl=rollout_dump_path,
                        batch_examples=batch_examples,
                        teacher_prompts=teacher_prompts,
                        rollout=rollout,
                        epoch=epoch,
                        total_stage2_epochs=total_stage2_epochs,
                        global_step=global_step,
                        current_micro_batch=current_micro_batch,
                        total_micro_batches=total_micro_batches,
                        processed_examples=processed_examples,
                        total_examples=len(epoch_examples),
                    )
                    print(
                        f"\n[RolloutDump] stage2 {progress} "
                        f"rows={len(batch_examples)} jsonl={rollout_dump_path}",
                        flush=True,
                    )

                if global_step == 1 or global_step % log_every_steps == 0:
                    log_parts = [
                        f"\n[Log] stage2 {progress}",
                        f"distill_loss={update_distill:.4f}",
                    ]
                    if str(args.stage2_distill_loss_type) == "masked_forward_kl":
                        log_parts.append(f"mask_coverage={update_coverage:.2%}")
                    log_parts.extend(
                        [
                            f"gap_abs_mean={update_gap:.4f}",
                            f"mean_rollout={update_rollout:.2f}",
                            f"hit_cap={update_hit_cap:.2%}",
                        ]
                    )
                    print(
                        " ".join(log_parts),
                        flush=True,
                    )
                    
                if (
                    eval_examples is not None
                    and eval_jsonl is not None
                    and frozen_scorer_model is not None
                    and frozen_scorer_tokenizer is not None
                    and int(args.eval_every_steps) > 0
                    and global_step % int(args.eval_every_steps) == 0
                ):
                    generator.model.eval()
                    eval_reasoning_map = generate_eval_reasoning_map(
                        generator=generator,
                        examples=eval_examples,
                        prompter=prompter,
                        max_prompt_tokens=args.generator_max_prompt_tokens,
                        max_new_tokens=args.eval_generator_max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        batch_size=args.eval_batch_size,
                    )
                    auroc, auprc, f1, sens90, sens95, ppv90, ppv95 = evaluate_with_frozen_scorer(
                        scorer_model=frozen_scorer_model,
                        scorer_tokenizer=frozen_scorer_tokenizer,
                        eval_jsonl=eval_jsonl,
                        reasoning_map=eval_reasoning_map,
                        max_length=args.scorer_max_length,
                        batch_size=args.eval_batch_size,
                        apply_sigmoid=args.eval_sigmoid,
                    )
                    print(
                        f"\t[VAL] stage2 {progress} || "
                        f"AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                        f"Sens@90={sens90:.4f} Sens@95={sens95:.4f} PPV@90={ppv90:.4f} PPV@95={ppv95:.4f}",
                        flush=True,
                    )
    generator.save_pretrained(output_dir)
    return generator


def resolve_stage2_student_dir(path_str: str, dataset: str | None = None) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        for root in artifact_lookup_roots(dataset):
            candidates.append(root / path_str)
    for candidate in candidates:
        if candidate.exists():
            print('Resolve stage2 student cache:', candidate)
            return candidate
    raise FileNotFoundError(f"Unable to find stage2 student checkpoint: {path_str}")


def load_stage2_student(
    *,
    args,
    checkpoint_dir: str,
) -> DistillGenerator:
    checkpoint_path = resolve_stage2_student_dir(checkpoint_dir, args.data_name)
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


def main():
    ap = argparse.ArgumentParser()
    data_group = ap.add_argument_group("Data and runtime")
    data_group.add_argument("--data_name", type=str, default="ad")
    data_group.add_argument("--train_jsonl", type=str, required=True)
    data_group.add_argument("--test_jsonl", type=str, default=None)
    data_group.add_argument("--testslice", type=int, nargs="*", default=[])
    data_group.add_argument("--output_dir", type=str, required=True)
    data_group.add_argument("--val_count", type=int, default=100)
    data_group.add_argument("--seed", type=int, default=7)
    data_group.add_argument("--data_seed", type=int, default=None)
    data_group.add_argument("--device", type=int, default=0)

    shared_eval_group = ap.add_argument_group("Shared eval and logging")
    shared_eval_group.add_argument("--eval_batch_size", type=int, default=8)
    shared_eval_group.add_argument("--eval_fast_ratio", type=float, default=0.0)
    shared_eval_group.add_argument("--eval_fast_min_per_class", type=int, default=1)
    shared_eval_group.add_argument("--eval_fast_seed", type=int, default=0)
    shared_eval_group.add_argument("--eval_sigmoid", action=argparse.BooleanOptionalAction, default=True)
    shared_eval_group.add_argument("--log_steps", type=int, default=10)
    shared_eval_group.add_argument("--log_every_steps", type=int, default=10)

    stage1_group = ap.add_argument_group("Stage1 scorer")
    stage1_group.add_argument("--scorer_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    stage1_group.add_argument("--scorer_max_length", type=int, default=5000)
    stage1_group.add_argument("--score_bias", action=argparse.BooleanOptionalAction, default=False)
    # ap.add_argument("--stage1_scorer_dir", type=str, default='stage1_seed794_lr5e-4_lastn3_r2_bs2_ga4/stage1_scorer/checkpoint-640')
    stage1_group.add_argument("--stage1_scorer_dir", type=str, default='')
    stage1_group.add_argument("--stage1_epochs", type=int, default=1)
    stage1_group.add_argument("--stage1_batch_size", type=int, default=1)
    stage1_group.add_argument("--stage1_grad_accum", type=int, default=16)
    stage1_group.add_argument("--stage1_lr", type=float, default=1e-3)
    stage1_group.add_argument("--stage1_lora_r", type=int, default=8)
    stage1_group.add_argument("--stage1_lora_last_n", type=int, default=1)
    stage1_group.add_argument("--stage1_score_only", action=argparse.BooleanOptionalAction, default=False)
    stage1_group.add_argument("--stage1_reasoning_jsonl", action=argparse.BooleanOptionalAction, default=True)
    stage1_group.add_argument("--stage1_reasoning_pid_key", type=str, default="id")
    stage1_group.add_argument("--stage1_reasoning_text_key", type=str, default="reasoning")
    stage1_group.add_argument("--follow_up_for_phase1", action=argparse.BooleanOptionalAction, default=False)
    stage1_group.add_argument("--neg_per_pos", type=int, default=2)
    stage1_group.add_argument("--loss_type", type=str, default="pairwise_bce", choices=["pairwise", "pairwise_bce", "margin", "bce"])
    stage1_group.add_argument("--pointwise_alpha", type=float, default=0.2)
    stage1_group.add_argument("--pairwise_margin", type=float, default=0.0)
    stage1_group.add_argument("--pos_weight", type=float, default=None)
    stage1_group.add_argument("--margin", type=float, default=0.1)
    stage1_group.add_argument("--margin_on_sigmoid", action="store_true", default=True)
    stage1_group.add_argument("--margin_on_logit", action="store_false", dest="margin_on_sigmoid")
    stage1_group.add_argument("--eval_steps", type=int, default=0)
    stage1_group.add_argument("--save_steps", type=int, default=0)

    stage2_group = ap.add_argument_group("Stage2 student generator")
    stage2_group.add_argument("--generator_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    stage2_group.add_argument("--generator_device_map", type=str, default="auto")
    stage2_group.add_argument("--generator_torch_dtype", type=str, default="bfloat16")
    stage2_group.add_argument("--generator_reasoning_max_new_tokens", type=int, default=256)
    stage2_group.add_argument("--generator_max_prompt_tokens", type=int, default=2048)
    stage2_group.add_argument("--generator_max_new_tokens", type=int, default=256)
    stage2_group.add_argument("--eval_generator_max_new_tokens", type=int, default=256)
    stage2_group.add_argument("--generator_lora_r", type=int, default=8)
    stage2_group.add_argument("--generator_lora_alpha", type=int, default=16)
    stage2_group.add_argument("--generator_lora_dropout", type=float, default=0.05)
    stage2_group.add_argument("--generator_lora_last_n", type=int, default=1)
    stage2_group.add_argument("--stage2_student_dir", type=str, default='')
    stage2_group.add_argument("--stage2_epochs", type=int, default=1)
    stage2_group.add_argument("--stage2_batch_size", type=int, default=2)
    stage2_group.add_argument("--stage2_grad_accum", type=int, default=1)
    stage2_group.add_argument("--stage2_lr", type=float, default=2e-5)
    stage2_group.add_argument("--stage2_save_steps", type=int, default=50)
    stage2_group.add_argument("--eval_every_steps", type=int, default=200)
    stage2_group.add_argument("--stage2_rollout_dump_every_steps", type=int, default=50)
    stage2_group.add_argument("--stage2_rollout_dump_jsonl", type=str, default="")
    stage2_group.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    stage2_group.add_argument("--temperature", type=float, default=0.8)
    stage2_group.add_argument("--top_p", type=float, default=0.95)

    distill_group = ap.add_argument_group("Stage2 distillation and robustness")
    distill_group.add_argument("--stage2_teacher_mode", choices=["current_lora", "fixed_base"], default="fixed_base")
    distill_group.add_argument(
        "--stage2_distill_loss_type",
        type=str,
        default="masked_forward_kl",
        choices=["masked_forward_kl", "generalized_jsd", "privileged_delta_jsd", "robust_privileged_delta_jsd", "robust_privileged_delta_reverse_kl"],
    )
    distill_group.add_argument("--stage2_jsd_beta", type=float, default=0.5)
    distill_group.add_argument("--stage2_jsd_temperature", type=float, default=1.0)
    distill_group.add_argument("--stage2_jsd_top_k", type=int, default=0)
    distill_group.add_argument("--stage2_jsd_token_clip", type=float, default=None)
    distill_group.add_argument("--stage2_delta_head_tokens", type=int, default=96)
    distill_group.add_argument("--stage2_delta_tail_weight", type=float, default=0.25)
    distill_group.add_argument("--stage2_robust_num_perturbations", type=int, default=3)
    distill_group.add_argument("--stage2_robust_include_unperturbed", action=argparse.BooleanOptionalAction, default=True)
    distill_group.add_argument("--stage2_robust_dx_dropout", type=float, default=0.30)
    distill_group.add_argument("--stage2_robust_med_dropout", type=float, default=0.30)
    distill_group.add_argument("--stage2_robust_min_keep", type=int, default=1)
    distill_group.add_argument("--stage2_robust_gate_mode", type=str, default="soft", choices=["none", "soft", "hard"])
    distill_group.add_argument("--stage2_robust_gate_scale", type=float, default=1.0)
    distill_group.add_argument("--stage2_robust_gate_threshold", type=float, default=1.0)
    distill_group.add_argument("--tau", type=float, default=0.2)
    distill_group.add_argument("--distill_debug_top_k", type=int, default=0)
    distill_group.add_argument("--distill_debug_every_batches", type=int, default=100)
    
    
    args = ap.parse_args()
    args.save_steps = args.eval_steps
    if args.data_seed is None:
        args.data_seed = int(args.seed)

    output_dir = artifact_output_root(args.data_name) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = install_run_log(output_dir)

    print(f"[run_log] {run_log}", flush=True)
    print(format_command(sys.argv), flush=True)
    print(format_args(args), flush=True)
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("cuda_count", torch.cuda.device_count())

    scorer_phase.set_baseline_prompt(args.data_name)

    train_source = Path("data") / args.train_jsonl
    train_jsonl, val_jsonl, train_examples, val_examples = build_splits(
        train_jsonl=str(train_source),
        output_dir=output_dir,
        seed=args.data_seed,
        val_count=args.val_count,
    )
    eval_jsonl = val_jsonl
    eval_examples = val_examples

    test_jsonl = Path("data") / args.test_jsonl if args.test_jsonl else None
    if args.testslice:
        test_dir = resolve_testslice_dir(args.data_name)
        test_jsonl = scorer_phase._find_slice_jsonl_files(test_dir, args.testslice, pattern_template="test_fold{slice}.jsonl")

    prompter = ClinicalReasoningPrompter(args.data_name)
    scorer_model = None
    scorer_tokenizer = None
    use_stage1_scorer = bool(str(args.stage1_scorer_dir or "").strip())
    if not use_stage1_scorer and int(args.eval_every_steps) > 0:
        print("\n===== RESEARCH3 STAGE2 EVAL =====")
        print(
            "stage1_scorer_dir not provided; auto-setting eval_every_steps=0 "
            "and skipping stage2 periodic eval",
            flush=True,
        )
        args.eval_every_steps = 0

    if use_stage1_scorer:
        scorer_model, scorer_tokenizer = load_stage1_scorer(
            args=args,
            checkpoint_dir=args.stage1_scorer_dir,
        )
    else:
        print("\n===== RESEARCH3 STAGE1 =====")
        print("mode=skipped")
        print("reason=stage1_scorer_dir not provided; no stage1 reasoning or scorer setup will run")


    print("\n===== RESEARCH3 STAGE2 =====")
    stage2_dir = output_dir / "stage2_generator"
    stage2_student_dir = str(args.stage2_student_dir or '').strip()
    if stage2_student_dir != '':
        generator = load_stage2_student(
            args=args,
            checkpoint_dir=stage2_student_dir,
        )
    elif args.stage2_epochs == 0:
        print('No Stage 2 Set Up')
        return
    else:
        generator = run_stage2_distill(
            args=args,
            train_examples=train_examples,
            eval_examples=eval_examples,
            eval_jsonl=eval_jsonl,
            output_dir=stage2_dir,
            frozen_scorer_model=scorer_model,
            frozen_scorer_tokenizer=scorer_tokenizer,
        )


if __name__ == "__main__":
    main()
