from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shlex
import sys
import warnings
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
)
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


def artifact_output_root(dataset: str) -> Path:
    dataset = str(dataset).strip().lower()
    if dataset == "adrd":
        return Path("artifacts", "adrd")
    if dataset == "pd":
        return Path("artifacts", "pd")
    if dataset == "pd28":
        return Path("artifacts", "pd28")
    if dataset == "pd003":
        return Path("artifacts", "pd003")
    return Path("artifacts", "research3")


def artifact_lookup_roots(dataset: str | None = None) -> list[Path]:
    ordered: list[Path] = []
    if dataset is not None:
        ordered.append(artifact_output_root(dataset))
    for root in [Path("artifacts", "research3"), Path("artifacts", "adrd"), Path("artifacts", "pd"), Path("artifacts", "pd28"), Path("artifacts", "pd003")]:
        if root not in ordered:
            ordered.append(root)
    return ordered


def resolve_testslice_dir(dataset: str) -> Path:
    dataset = str(dataset).strip().lower()
    if dataset == "ad":
        return Path("data", "ADstratified10")
    if dataset == "adrd":
        return Path("data", "ADRDstratified10")
    if dataset == "pd":
        return Path("data", "PDstratified8")
    if dataset == "pd28":
        pd_data_subdir = str(os.environ.get("PD_DATA_SUBDIR", "")).strip()
        candidates = [pd_data_subdir] if pd_data_subdir else ["PD_jan28", "PD-jan29", "PD_jan29"]
        for candidate in candidates:
            if not candidate:
                continue
            test_dir = Path("data", candidate, "PDstratified5")
            if test_dir.is_dir():
                return test_dir
        raise FileNotFoundError(
            "Could not find PDstratified5 for data_name=pd28. "
            "Set PD_DATA_SUBDIR to a valid subdir under data/."
        )
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
    elif dataset in {"pd", "pd28", "pd003"}:
        train_source = f"artifacts/phase1/{dataset}/qwen-4b-instruct_phase1_fold0_ratio5_visit6_text_reasoning_demo_prompt0202_nomemory_new7_0207_190249_54.jsonl"
        if dataset in {"pd003", "pd28"}:
            train_source = "artifacts/phase1/pd/qwen-4b-instruct_phase1_fold0_ratio5_visit6_text_reasoning_demo_prompt0202_nomemory_new7_0207_190249_54.jsonl"
        train_rows = scorer_phase._read_jsonl(train_source)
        if testslice:
            template = "qwen*final*test_sp{slice}*_demo_prompt0202_nomemory_new7_0207_190249_54*"
            matched = scorer_phase._find_slice_jsonl_files(
                Path("artifacts/phase1/pd" if dataset in {"pd003", "pd28"} else f"artifacts/phase1/{dataset}"),
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
    disable_dropout: bool, 
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
        disable_dropout=disable_dropout
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
        disable_dropout=args.disable_dropout,
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
            # print(batch)
            # sys.exit()
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
                # print(enc, text_key)
                # sys.exit()
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
            # print(scores)
            # sys.exit()
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
        )xw
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


def run_stage2_distill(
    *,
    args,
    train_examples: list[Example],
    eval_examples: list[Example] | None,
    eval_jsonl,
    output_dir: Path,
    frozen_scorer_model,
    frozen_scorer_tokenizer,
):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
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
    teacher_disable_adapter = args.stage2_teacher_mode == "fixed_base" or use_privileged_delta
    print(f"teacher_disable_adapter={teacher_disable_adapter}")
    if use_privileged_delta:
        print(
            f"privileged_delta_head_tokens={args.stage2_delta_head_tokens} "
            f"privileged_delta_tail_weight={args.stage2_delta_tail_weight}"
        )

    print('Will use frozen scorer model...')
    for param in frozen_scorer_model.parameters():
        param.requires_grad = False
    frozen_scorer_model.eval()

    
    prompter = ClinicalReasoningPrompter(args.data_name)
    print("\n===== RESEARCH3 STAGE2 =====")
    print(f"train_examples={len(train_examples)}")
    print(f"distill_loss_type={args.stage2_distill_loss_type}")
    if eval_examples is not None:
        print(f"eval_examples={len(eval_examples)}")

    global_step = 0
    for epoch in range(int(args.stage2_epochs)):
        epoch_examples = shuffle_examples(train_examples, seed=args.data_seed + epoch)
        print(args.stage2_batch_size)
        train_batches = chunk_examples(epoch_examples, batch_size=args.stage2_batch_size)
        print('Train_batches:', len(train_batches), len(train_batches[0]))
        
        optimizer.zero_grad(set_to_none=True)
        print('Start iterative steps...')
        accum_distill: list[float] = []
        accum_coverage: list[float] = []
        accum_gap: list[float] = []
        accum_rollout: list[float] = []
        accum_hit_cap: list[float] = []
        accum_masked: list[float] = []
        
        for batch_step, batch_examples in tqdm(
            enumerate(train_batches, start=0),
            total=len(train_batches),
            desc=f"stage2 epoch {epoch}",
        ):
            debug_this_batch = (
                int(args.distill_debug_top_k) > 0
                and batch_step % max(int(args.distill_debug_every_batches), 1) == 0
            )
            debug_prefix = f"[Distill Debug] epoch={epoch} batch_step={batch_step} "
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
            if use_privileged_delta:
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

            distill_loss, masked_tokens, mask_coverage, gap_abs_mean = compute_distill_loss(
                teacher_logits=teacher_logits,
                student_logits=student_pass.rollout_logits,
                target_ids=student_pass.rollout_token_ids,
                completion_attention_mask=student_pass.rollout_attention_mask,
                baseline_teacher_logits=baseline_teacher_logits,
                tau=args.tau,
                loss_type=args.stage2_distill_loss_type,
                jsd_beta=args.stage2_jsd_beta,
                jsd_temperature=args.stage2_jsd_temperature,
                jsd_top_k=(args.stage2_jsd_top_k if int(args.stage2_jsd_top_k) > 0 else None),
                jsd_token_clip=args.stage2_jsd_token_clip,
                delta_head_tokens=args.stage2_delta_head_tokens,
                delta_tail_weight=args.stage2_delta_tail_weight,
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
            
            (distill_loss / max(int(args.stage2_grad_accum), 1)).backward()
            should_step = (
                ((batch_step + 1) % max(int(args.stage2_grad_accum), 1) == 0)
                or ((batch_step + 1) == len(train_batches))
            )
            if should_step:
                completed_global_step = global_step + 1
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
                    and completed_global_step % int(args.stage2_save_steps) == 0
                ):
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    generator.save_pretrained(checkpoint_dir)
                    print(
                        f"\n[Save] stage2 global_step={global_step} checkpoint={checkpoint_dir}",
                        flush=True,
                    )

                if global_step % int(args.log_every_steps) == 0:
                    log_parts = [
                        f"\n[Log] stage2 epoch={epoch} global_step={global_step} batch_step={batch_step}",
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
                        f"\t[VAL] stage2 epoch={epoch} global_step={global_step} batch_step={batch_step} || "
                        f"AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                        f"Sens@90={sens90:.4f} Sens@95={sens95:.4f} PPV@90={ppv90:.4f} PPV@95={ppv95:.4f}",
                        flush=True,
                    )
                global_step = completed_global_step
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
    ap.add_argument("--data_name", type=str, default="ad")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--test_jsonl", type=str, default=None)
    ap.add_argument("--testslice", type=int, nargs="*", default=[])
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--val_count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data_seed", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--eval_every_steps", type=int, default=200)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_fast_ratio", type=float, default=0.0)
    ap.add_argument("--eval_fast_min_per_class", type=int, default=1)
    ap.add_argument("--eval_fast_seed", type=int, default=0)
    ap.add_argument("--eval_sigmoid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log_every_steps", type=int, default=10)

    ap.add_argument("--generator_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--generator_device_map", type=str, default="auto")
    ap.add_argument("--generator_torch_dtype", type=str, default="bfloat16")
    ap.add_argument("--generator_reasoning_max_new_tokens", type=int, default=256)
    ap.add_argument("--generator_max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--generator_max_new_tokens", type=int, default=256)
    ap.add_argument("--eval_generator_max_new_tokens", type=int, default=256)
    ap.add_argument("--generator_lora_r", type=int, default=8)
    ap.add_argument("--generator_lora_alpha", type=int, default=16)
    ap.add_argument("--generator_lora_dropout", type=float, default=0.05)
    ap.add_argument("--generator_lora_last_n", type=int, default=1)
    ap.add_argument("--stage2_epochs", type=int, default=1)
    ap.add_argument("--stage2_batch_size", type=int, default=2)
    ap.add_argument("--stage2_grad_accum", type=int, default=1)
    ap.add_argument("--stage2_lr", type=float, default=2e-5)
    ap.add_argument("--stage2_student_dir", type=str, default='')
    ap.add_argument("--stage2_save_steps", type=int, default=50)
    ap.add_argument("--stage2_teacher_mode", choices=["current_lora", "fixed_base"], default="fixed_base")
    ap.add_argument(
        "--stage2_distill_loss_type",
        type=str,
        default="masked_forward_kl",
        choices=["masked_forward_kl", "generalized_jsd", "privileged_delta_jsd"],
    )
    ap.add_argument("--stage2_jsd_beta", type=float, default=0.5)
    ap.add_argument("--stage2_jsd_temperature", type=float, default=1.0)
    ap.add_argument("--stage2_jsd_top_k", type=int, default=0)
    ap.add_argument("--stage2_jsd_token_clip", type=float, default=None)
    ap.add_argument("--stage2_delta_head_tokens", type=int, default=96)
    ap.add_argument("--stage2_delta_tail_weight", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--distill_debug_top_k", type=int, default=0)
    ap.add_argument("--distill_debug_every_batches", type=int, default=100)
    ap.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--scorer_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--scorer_max_length", type=int, default=5000)
    ap.add_argument("--score_bias", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--stage1_scorer_dir", type=str, default='')

    ap.add_argument("--stage1_epochs", type=int, default=1)
    ap.add_argument("--stage1_batch_size", type=int, default=1)
    ap.add_argument("--stage1_grad_accum", type=int, default=16)
    ap.add_argument("--stage1_lr", type=float, default=1e-3)
    ap.add_argument("--stage1_lora_r", type=int, default=8)
    ap.add_argument("--stage1_lora_last_n", type=int, default=1)
    ap.add_argument("--stage1_score_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--stage1_reasoning_jsonl", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stage1_reasoning_pid_key", type=str, default="id")
    ap.add_argument("--stage1_reasoning_text_key", type=str, default="reasoning")
    ap.add_argument("--neg_per_pos", type=int, default=2)
    ap.add_argument("--loss_type", type=str, default="pairwise_bce", choices=["pairwise", "pairwise_bce", "margin", "bce"])
    ap.add_argument("--pointwise_alpha", type=float, default=0.2)
    ap.add_argument("--pairwise_margin", type=float, default=0.0)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--margin_on_sigmoid", action="store_true", default=True)
    ap.add_argument("--margin_on_logit", action="store_false", dest="margin_on_sigmoid")
    ap.add_argument("--follow_up_for_phase1", action=argparse.BooleanOptionalAction, default=False)
    
    
    args = ap.parse_args()
    args.save_steps = args.eval_steps
    if args.data_seed is None:
        args.data_seed = int(args.seed)

    output_dir = artifact_output_root(args.data_name) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = install_run_log(output_dir)

    print(f"[run_log] {run_log}", flush=True)
    print(format_command(sys.argv), flush=True)
    print('ARGS', args)
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
        test_jsonl = scorer_phase._find_slice_jsonl_files(test_dir, args.testslice, pattern_template="test_fold{slice}*")

    prompter = ClinicalReasoningPrompter(args.data_name)
    train_reasoning_map: dict[str, str]
    eval_targets: list[dict[str, object]] = []
    test_examples = None

    if args.stage1_reasoning_jsonl:
        full_train_reasoning_map, test_reasoning_map, train_reasoning_source, test_reasoning_sources = load_legacy_reasoning_maps(
            data_name=args.data_name,
            testslice=list(args.testslice),
            pid_key=args.stage1_reasoning_pid_key,
            text_key=args.stage1_reasoning_text_key,
        )
        print("\n===== RESEARCH3 STAGE1 REASONING =====")
        print(f"mode=legacy_jsonl train_source={train_reasoning_source}")
        if test_reasoning_sources:
            print(f"test_sources={test_reasoning_sources}")
        train_reasoning_map = {
            str(example.patient_id): full_train_reasoning_map[str(example.patient_id)]
            for example in train_examples
            if str(example.patient_id) in full_train_reasoning_map
        }
        if eval_examples is not None:
            val_reasoning_map = {
                str(example.patient_id): full_train_reasoning_map[str(example.patient_id)]
                for example in eval_examples
                if str(example.patient_id) in full_train_reasoning_map
            }
        else:
            val_reasoning_map = {}
    else:
        generator_model, generator_tokenizer = qwen_init(
            model_name=args.generator_model_name,
            device_map=args.generator_device_map,
            dtype=args.generator_torch_dtype,
        )
        stage1_cache_dir = output_dir / "reasoning_cache"
        train_reasoning_path = stage1_cache_dir / "train_reasoning.jsonl"
        train_reasoning_map = generate_reasoning_cache(
            model=generator_model,
            tokenizer=generator_tokenizer,
            examples=train_examples,
            prompter=prompter,
            include_followup=False,
            omit_followup_text=True,
            max_new_tokens=args.generator_reasoning_max_new_tokens,
            output_jsonl=train_reasoning_path,
        )
        if eval_examples is not None:
            val_reasoning_path = stage1_cache_dir / "val_reasoning.jsonl"
            val_reasoning_map = generate_reasoning_cache(
                model=generator_model,
                tokenizer=generator_tokenizer,
                examples=eval_examples,
                prompter=prompter,
                include_followup=False,
                omit_followup_text=True,
                max_new_tokens=args.generator_reasoning_max_new_tokens,
                output_jsonl=val_reasoning_path,
            )
        else:
            val_reasoning_map = {}
        test_reasoning_map = {}
        print("\n===== RESEARCH3 STAGE1 REASONING =====")
        print("mode=generate_cache")

    if test_jsonl is not None:
        if isinstance(test_jsonl, (list, tuple)):
            test_examples = [Example.from_dict(row) for row in scorer_phase._read_jsonl_multi(test_jsonl)]
        else:
            test_examples = load_examples(test_jsonl)
        if not args.stage1_reasoning_jsonl:
            test_reasoning_path = stage1_cache_dir / "test_reasoning.jsonl"
            test_reasoning_map = generate_reasoning_cache(
                model=generator_model,
                tokenizer=generator_tokenizer,
                examples=test_examples,
                prompter=prompter,
                include_followup=False,
                omit_followup_text=True,
                max_new_tokens=args.generator_reasoning_max_new_tokens,
                output_jsonl=test_reasoning_path,
            )
        eval_targets.append(
            {
                "name": "test",
                "jsonl": test_jsonl,
                "reasoning_map": test_reasoning_map,
            }
        )

    stage1_dir = output_dir / "stage1_scorer"
    if args.stage1_scorer_dir is not None and args.stage1_scorer_dir != '':
        scorer_model, scorer_tokenizer = load_stage1_scorer(
            args=args,
            checkpoint_dir=args.stage1_scorer_dir,
        )
    else:
        scorer_model, scorer_tokenizer = train_stage1_scorer(
            args=args,
            train_jsonl=str(train_jsonl),
            eval_targets=eval_targets,
            train_reasoning_map=train_reasoning_map,
            output_dir=stage1_dir,
            follow_up_for_phase1=args.follow_up_for_phase1,
        )


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

    if test_jsonl is not None:
        if test_examples is None:
            if isinstance(test_jsonl, (list, tuple)):
                test_examples = [Example.from_dict(row) for row in scorer_phase._read_jsonl_multi(test_jsonl)]
            else:
                test_examples = load_examples(test_jsonl)
        final_reasoning_map = generate_eval_reasoning_map(
            generator=generator,
            examples=test_examples,
            prompter=prompter,
            max_prompt_tokens=args.generator_max_prompt_tokens,
            max_new_tokens=args.eval_generator_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            batch_size=args.eval_batch_size,
        )
        auroc, auprc, f1, sens90, sens95, ppv90, ppv95 = evaluate_with_frozen_scorer(
            scorer_model=scorer_model,
            scorer_tokenizer=scorer_tokenizer,
            eval_jsonl=test_jsonl,
            reasoning_map=final_reasoning_map,
            max_length=args.scorer_max_length,
            batch_size=args.eval_batch_size,
            apply_sigmoid=args.eval_sigmoid,
        )
        print(
            f"[TEST] AUROC={auroc:.4f} AUPRC={auprc:.4f} F1={f1:.4f} "
            f"Sens@90={sens90:.4f} Sens@95={sens95:.4f} PPV@90={ppv90:.4f} PPV@95={ppv95:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
