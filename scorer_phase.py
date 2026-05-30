import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import IterableDataset, Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback
from trl import RewardTrainer, RewardConfig

from utils import get_evaluation_metrics
import sys
import os
import random
import warnings
from transformers import DataCollatorWithPadding
import torch.nn as nn
from scorer_prompts import BASELINE_PROMPT_AD, BASELINE_PROMPT_ADRD, BASELINE_PROMPT_PD


def resolve_baseline_prompt(dataset: str) -> str:
    dataset = str(dataset).strip().lower()
    if dataset == "ad":
        print('\t---Resolve prompt for AD')
        return BASELINE_PROMPT_AD
    if dataset == "adrd":
        print('\t---Resolve prompt for ADRD')
        return BASELINE_PROMPT_ADRD
    if dataset in {"pd", "pd28", "pd003"}:
        print('\t---Resolve prompt for PD')
        return BASELINE_PROMPT_PD
    raise ValueError(f"Unsupported dataset: {dataset}")


def set_baseline_prompt(dataset: str) -> str:
    global BASELINE_PROMPT
    print('\n------Set baseline prompt:')
    BASELINE_PROMPT = resolve_baseline_prompt(dataset)
    return BASELINE_PROMPT


def _get_attr_by_path(obj, path):
    parts = path.split(".")
    for part in parts:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _resolve_transformer_layers(model):
    candidates = [
        ("model.layers", "layers"),
        ("model.model.layers", "layers"),
        ("transformer.h", "h"),
        ("gpt_neox.layers", "layers"),
        ("backbone.layers", "layers"),
    ]
    for path, pattern in candidates:
        layers = _get_attr_by_path(model, path)
        if layers is not None:
            return layers, pattern
    return None, None


def replace_score_with_mlp(model, hidden=None, mid=256, dropout=0.1):
    """
    Replace model.score (Linear) with a 2-layer MLP: hidden -> mid -> 1.
    Ensures dtype/device match the base model.
    """
    if hidden is None:
        hidden = int(getattr(model.config, "hidden_size", 1024))

    mlp = nn.Sequential(
        nn.Linear(hidden, mid),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid, 1),
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mlp = mlp.to(device=device, dtype=dtype)

    model.score = mlp
    model.config.num_labels = 1
    return model


class DebugCallback(TrainerCallback):
    def __init__(self, every=10, show_trainable_once=True):
        self.every = every
        self.show_trainable_once = show_trainable_once
        self._printed_trainable = False

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0 or state.global_step == 0:
            return
        model = kwargs["model"]
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        score_params = [(name, p) for name, p in base.named_parameters() if "score" in name]

        if self.show_trainable_once and not self._printed_trainable:
            total = sum(p.numel() for p in base.parameters())
            trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
            self._printed_trainable = True
            opt = kwargs.get("optimizer")
            if opt is not None:
                opt_names = set(id(p) for group in opt.param_groups for p in group["params"])
                score_names = [name for name, p in base.named_parameters() if "score" in name and id(p) in opt_names]
                score_p = [p for name, p in base.named_parameters() if "score" in name and id(p) in opt_names]
        if not score_params:
            print(f"[Debug step {state.global_step}] no params matched name contains 'score'")
            return

        for name, p in score_params:
            if p.grad is None:
                continue
            grad_norm = p.grad.data.norm().item()
            print(f"[Debug step {state.global_step}] {name} | requires_grad={p.requires_grad} "
                f"| w_norm={p.data.norm().item():.4f} | grad_norm={grad_norm}"
            )
        other_params = [
            (name, p)
            for name, p in base.named_parameters()
            if "score" not in name and p.grad is not None
        ]
        for name, p in other_params:
            grad_norm = p.grad.data.norm().item()
            print(f"[Debug step {state.global_step}] {name} | requires_grad={p.requires_grad} "
                f"| w_norm={p.data.norm().item():.4f} | grad_norm={grad_norm}"
            )


class LossSwitchRewardTrainer(RewardTrainer):
    def __init__(
        self,
        *args,
        loss_type="pairwise_bce",
        pointwise_alpha=0.2,
        pairwise_margin=0.0,
        pos_weight=None,
        margin=0.1,
        margin_on_sigmoid=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pointwise_alpha = float(pointwise_alpha)
        self.pointwise_pos_weight = None if pos_weight is None else float(pos_weight)
        if self.pointwise_pos_weight is None:
            self.pointwise_bce = torch.nn.BCEWithLogitsLoss()
        else:
            self.pointwise_bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pointwise_pos_weight]),
            )
        self.pairwise_margin = float(pairwise_margin)
        self.margin = float(margin)
        self.margin_on_sigmoid = bool(margin_on_sigmoid)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mode = "train" if self.model.training else "eval"

        inputs["use_cache"] = False
        outputs = model(**inputs)

        chosen_scores, rejected_scores = torch.chunk(outputs.logits.squeeze(-1), chunks=2)
        raw_chosen_scores = chosen_scores
        raw_rejected_scores = rejected_scores

        center_loss = torch.zeros((), device=chosen_scores.device, dtype=chosen_scores.dtype)
        if self.args.center_rewards_coefficient is not None:
            center_loss = self.args.center_rewards_coefficient * torch.mean((chosen_scores + rejected_scores) ** 2)

        if hasattr(self, "_metrics") and hasattr(self, "accelerator") and mode in self._metrics:
            with torch.no_grad():
                all_rewards = self.accelerator.gather(torch.cat([raw_chosen_scores, raw_rejected_scores], dim=0))
                self._metrics[mode]["min_reward"].append(all_rewards.min().item())
                self._metrics[mode]["mean_reward"].append(all_rewards.mean().item())
                self._metrics[mode]["max_reward"].append(all_rewards.max().item())

                mean_accuracy = (raw_chosen_scores > raw_rejected_scores).float().mean()
                mean_accuracy = self.accelerator.gather_for_metrics(mean_accuracy).mean().item()
                self._metrics[mode]["accuracy"].append(mean_accuracy)

                mean_margin = (raw_chosen_scores - raw_rejected_scores).mean()
                mean_margin = self.accelerator.gather_for_metrics(mean_margin).mean()
                self._metrics[mode]["margin"].append(mean_margin.item())

        if self.loss_type == "margin":
            if self.margin_on_sigmoid:
                chosen_scores = torch.sigmoid(raw_chosen_scores)
                rejected_scores = torch.sigmoid(raw_rejected_scores)
            loss = torch.relu(self.margin - (chosen_scores - rejected_scores)).mean() + center_loss
            if return_outputs:
                return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
            return loss


        diff = chosen_scores - rejected_scores - self.pairwise_margin
        pairwise_loss = -torch.nn.functional.logsigmoid(diff).mean()
        if self.loss_type == "pairwise":
            if return_outputs:
                pairwise_loss += center_loss
                return pairwise_loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
            pairwise_loss += center_loss
            return pairwise_loss

        pointwise_targets = torch.cat(
            [
                torch.ones_like(chosen_scores),
                torch.zeros_like(rejected_scores),
            ],
            dim=0,
        )
        pointwise_logits = torch.cat([chosen_scores, rejected_scores], dim=0)
        if self.pointwise_pos_weight is not None:
            pos_weight = getattr(self.pointwise_bce, "pos_weight", None)
            if (
                pos_weight is None
                or pos_weight.device != pointwise_logits.device
                or pos_weight.dtype != pointwise_logits.dtype
            ):
                self.pointwise_bce = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(
                        [self.pointwise_pos_weight],
                        device=pointwise_logits.device,
                        dtype=pointwise_logits.dtype,
                    ),
                )
        pointwise_loss = self.pointwise_bce(pointwise_logits, pointwise_targets)
        if self.loss_type == 'bce':
            loss = pointwise_loss + center_loss
        else:
            assert self.loss_type =='pairwise_bce'
            loss = pairwise_loss + self.pointwise_alpha * pointwise_loss + center_loss
        if return_outputs:
            return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
        return loss


def _reset_score_head_with_seed(model, seed):
    if not hasattr(model, "score"):
        return False
    score = model.score
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if hasattr(score, "reset_parameters"):
            score.reset_parameters()
        else:
            for module in score.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
    return True


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl_multi(paths):
    if isinstance(paths, (list, tuple)):
        rows = []
        for p in paths:
            rows.extend(_read_jsonl(str(p)))
        return rows
    return _read_jsonl(str(paths))




def _find_slice_jsonl_files(data_dir: Path, slices, pattern_template):
    files = []
    data_dir = Path(data_dir)
    print('\n|Find in data dir:', data_dir)
    for slice_id in slices:
        pattern = pattern_template.format(slice=int(slice_id))
        matched = sorted(data_dir.glob(pattern))
        print('\t|--Find matched file:', matched)
        if not matched:
            raise FileNotFoundError(
                f"No files found for slice {slice_id} with pattern {pattern} in {data_dir}"
            )
        files.extend(matched)
    return files

def _format_baseline_text(ex, reasoning_text=None, follow_up=None):
    base_dx = ex.get("base_codes_diagnosis", []) or []
    base_md = ex.get("base_codes_medication", []) or []
    delta_dx = ex.get("delta_codes_diagnosis", []) or []
    delta_md = ex.get("delta_codes_medication", []) or []

    base_dx = [i.lower().strip() for i in base_dx]
    base_md = [i.lower().strip() for i in base_md]
    delta_dx = [i.lower().strip() for i in delta_dx]
    delta_md = [i.lower().strip() for i in delta_md]


    if follow_up is True:
        parts = [
        f"Baseline demographics: age {ex.get('age') - 5}, gender {ex.get('sex')};"
        f"Baseline diagnoses: {'; '.join(base_dx)}",
        f"Baseline medications: {'; '.join(base_md)}",
        f"Follow-up diagnoses (newly emerged from baseline to 1 year prior to the outcome window, training-only): {'; '.join(delta_dx)}",
        f"Follow-up medications (newly emerged from baseline to 1 year prior to the outcome window, training-only): {'; '.join(delta_md)}",
        ]
                        
    else:
        parts = [
        f"Baseline demographics: age {ex.get('age') - 5}, gender {ex.get('sex')};"
        f"Baseline diagnoses: {'; '.join(base_dx)}",
        f"Baseline medications: {'; '.join(base_md)}",
        ]
    if reasoning_text:
        parts.append('Analysis for this patient: ' + str(reasoning_text))

    return "\n".join(parts)


class EpochState:
    def __init__(self, base_seed=7):
        self.base_seed = int(base_seed)
        self.counter = 0

    def seed_for_epoch(self):
        seed = self.base_seed + 1000003 * self.counter
        self.counter += 1
        return seed


def build_pref_iterable_dataset_epoch_baseline(
    in_jsonl: str,
    neg_per_pos: int = 1,
    base_seed: int = 7,
    prompt: str | None = None,
    include_meta: bool = False,
    reasoning_map: dict | None = None,
    follow_up=None,
):
    if prompt is None:
        prompt = BASELINE_PROMPT
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string (or None to use default).")

    neg_per_pos = int(neg_per_pos)
    if neg_per_pos <= 0:
        raise ValueError("neg_per_pos must be >= 1")

    raw = _read_jsonl_multi(in_jsonl)

    positives, negatives = [], []
    have_reasoning = 0
    for ex in raw:
        label = int(ex["label"])
        pid = ex.get("id")
        reasoning_text = None
        if reasoning_map is not None and pid is not None:
            reasoning_text = reasoning_map.get(str(pid))
            if reasoning_text is not None:
                have_reasoning += 1
        text = _format_baseline_text(ex, reasoning_text=reasoning_text, follow_up=follow_up)
        item = {"pid": pid, "text": text}
        if label == 1:
            positives.append(item)
        else:
            negatives.append(item)

    print('\n\n=== Train dataset ===')
    print(f'|(Train) Build preference dataset: cases ({len(positives)}), controls ({len(negatives)}), total ({len(positives) + len(negatives)})')
    print('\t|Case / Controls: ', len(positives)/len(negatives))
    print('\t|Among these, subjs having reasoning:', have_reasoning)
    if not positives or not negatives:
        raise ValueError(f"Need both pos and neg. Got pos={len(positives)}, neg={len(negatives)}")

    state = EpochState(base_seed=base_seed)


    def gen():
        returnseed = state.seed_for_epoch()
        rng = np.random.default_rng(returnseed)
        pos_order = rng.permutation(len(positives))
        for pidx in pos_order:
            pos = positives[int(pidx)]
            for _ in range(neg_per_pos):
                neg = negatives[int(rng.integers(0, len(negatives)))]
                out = {
                    "prompt": prompt,
                    "chosen": pos["text"],
                    "rejected": neg["text"],
                }
                if include_meta:
                    out["meta"] = {"pid_chosen": pos["pid"], "pid_rejected": neg["pid"]}
                yield out

    ds = IterableDataset.from_generator(gen)
    return ds, state, len(positives)


def build_pointwise_dataset_baseline(in_jsonl, reasoning_map: dict | None = None, follow_up=None):
    if isinstance(in_jsonl, list):
        raw = _read_jsonl_multi(in_jsonl)
    else:
        raw = _read_jsonl(in_jsonl)
    items = []
    prompt = BASELINE_PROMPT
    positives, negatives = [], []
    have_reasoning =0
    for ex in raw:
        pid = ex.get('id')
        reasoning_text = None
        if reasoning_map is not None and pid is not None:
            reasoning_text = reasoning_map.get(str(pid))
            if reasoning_text is not None:
                have_reasoning += 1
        text = _format_baseline_text(ex, reasoning_text=reasoning_text, follow_up=follow_up)
        text = prompt + "\n" + text
        items.append(
            {
                "text":text,
                "label": int(ex["label"]),
            }
        )

        label = int(ex["label"])
        if label == 1:
            positives.append(pid)
        else:
            negatives.append(pid)
    
    print('\n\n=== Test dataset ===')
    print(f'|(Test) Build pointwise dataset: cases ({len(positives)}), controls ({len(negatives)}), total ({len(positives)+len(negatives)})')
    print('\t|Case / Controls: ', len(positives)/len(negatives))
    print('\t|Case prevalance: ', len(positives)/(len(positives) + len(negatives)))
    print('\t|Among these, subjs having reasoning:', have_reasoning)
    getdataset = Dataset.from_list(items)
    return getdataset


def _pretokenize_pointwise_dataset(test_ds, tokenizer, max_length, batch_size=256):
    items = []
    n = len(test_ds)
    for i in range(0, n, batch_size):
        batch = test_ds[i : i + batch_size]
        enc = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        for j in range(len(batch["text"])):
            items.append(
                {
                    "input_ids": enc["input_ids"][j],
                    "attention_mask": enc["attention_mask"][j],
                    "label": int(batch["label"][j]),
                }
            )
    ds = Dataset.from_list(items)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def eval_pointwise(
    model,
    tokenizer,
    test_dataset,
    text_key="text",
    label_key="label",
    batch_size=8,
    max_length=10000,
    device=None,
    apply_sigmoid=False,
    collator=None,
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
    with torch.inference_mode():
        for i in range(0, n, batch_size):
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
