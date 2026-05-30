from __future__ import annotations

import copy
import importlib
import re
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_schema import init_memory, init_memory_adrd, init_memory_pd
from research3.apd_prompt_builders_may1 import (
    _build_pd_baseline_teacher_prompt,
    _build_pd_student_prompt,
    _build_pd_teacher_prompt,
)
from research3.adrd_prompt_builders_apr28 import (
    _build_adrd_baseline_teacher_prompt,
    _build_adrd_student_prompt,
    _build_adrd_teacher_prompt,
)
from research3.data import Example
from research3.losses import masked_forward_kl, target_token_logprob_gap, target_token_logprobs_from_logits
from research3.modeling_utils import resolve_transformer_layers, resolve_torch_dtype


warnings.filterwarnings(
    "ignore",
    message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
)


def load_prompt_module(dataset: str):
    dataset = str(dataset).strip().lower()
    if dataset in {"pd003", "pd28"}:
        dataset = "pd"
    return importlib.import_module(f"prompts.prompts_{dataset}")


def init_memory_for_dataset(dataset: str) -> dict[str, Any]:
    dataset = str(dataset).strip().lower()
    if dataset == "ad":
        return init_memory()
    if dataset == "adrd":
        return init_memory_adrd()
    if dataset in {"pd", "pd28", "pd003"}:
        return init_memory_pd()
    raise ValueError(f"Unsupported dataset: {dataset}")


class ClinicalReasoningPrompter:
    def __init__(self, dataset: str) -> None:
        self.dataset = str(dataset).strip().lower()
        self.module = None if self.dataset in {"ad", "adrd"} else load_prompt_module(self.dataset)
        self.memory = init_memory_for_dataset(self.dataset)

    @staticmethod
    def _format_prompt_list(values: list[str] | None) -> str:
        return str(list(values or []))

    @staticmethod
    def _format_prompt_age(age: int | None) -> str:
        return str(age) if age is not None else "unknown"

    @staticmethod
    def _format_prompt_sex(sex: str | None) -> str:
        return str(sex) if sex is not None else "unknown"

    @staticmethod
    def _format_prompt_label(label: int | None) -> str:
        label = int(label)
        if label == 1:
            return "Yes"
        if label == 0:
            return "No"
        return "unknown"
    




    global fullname
    global shortname
    global year 
    
    fullname = "Alzheimer's disease"
    shortname = "AD"
    year = 5 
       
         
    def _build_ad_student_prompt(
        self,
        *,
        example: Example,
        age: int | None,
    ) -> str:

        return ( 
        f"You are a national-level experienced physician with expertise in {fullname} ({shortname}). "
        "You are interpreting EHR records to understand patient-level clinical patterns. "
        f"You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly {year} years before the future outcome window. "
        "You have access to the patient’s EHR history available up to this time. "
        f"For training purposes only, you are also shown follow-up information from this time ({year} years prior to the future outcome window) to 1 year prior to the outcome window. "
        "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "
        "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

        "You will review the patient’s EHR history at the baseline time: "
        "e.g., whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
        f"whether such patterns appear patient-specific rather than generic, and whether these associate with {shortname} risk.\n\n"

        "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
        "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
        "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
        "Include (i) the most important baseline clinical patterns, "
        "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

        "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
        "\"Baseline clinical interpretation: ...\"\n\n"

        "Purpose of this sentence:\n"
        "- Do NOT center your reasoning on the absence of documented cognitive/functional assessments. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is.\n"
        "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available.\n"
        "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns).\n"
        "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n\n"

        "Content guidelines for \"Baseline clinical interpretation\":\n"
        "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
        "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
        "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
        "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
        "- Do NOT enumerate diseases or codes as a list.\n"
        f"- Do NOT state whether the patient will or will not develop {shortname}.\n\n"






        f"Baseline demographics: Age ({self._format_prompt_age(age)}), Sex ({self._format_prompt_sex(example.sex)}).\n"
        f"Baseline diagnoses: {self._format_prompt_list(example.base_codes_diagnosis)}\n"
        f"Baseline medications: {self._format_prompt_list(example.base_codes_medication)}\n\n"

        "Write only the final clinical reasoning paragraph."
        )
     
       
     
     


    def _build_ad_baseline_teacher_prompt(
        self,
        *,
        example: Example,
        age: int | None,
    ) -> str:

        return ( 
        f"You are a national-level experienced physician with expertise in {fullname} ({shortname}). "
        "You are interpreting EHR records to understand patient-level clinical patterns. "
        f"You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly {year} years before the future outcome window. "
        "You have access to the patient’s EHR history available up to this time. "
        f"For training purposes only, you are also shown follow-up information from this time ({year} years prior to the future outcome window) to 1 year prior to the outcome window. "
        "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "
        "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

        "You will review the patient’s EHR history at the baseline time: "
        "e.g., whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
        f"whether such patterns appear patient-specific rather than generic, and whether these associate with {shortname} risk.\n\n"

        "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
        "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
        "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
        "Include (i) the most important baseline clinical patterns, "
        "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

        "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
        "\"Baseline clinical interpretation: ...\"\n\n"

        "Purpose of this sentence:\n"
        "- Do NOT center your reasoning on the absence of documented cognitive/functional assessments. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is.\n"
        "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available.\n"
        "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns).\n"
        "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n\n"

        "Content guidelines for \"Baseline clinical interpretation\":\n"
        "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
        "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
        "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
        "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
        "- Do NOT enumerate diseases or codes as a list.\n"
        f"- Do NOT state whether the patient will or will not develop {shortname}.\n\n"






        f"Baseline demographics: Age ({self._format_prompt_age(age)}), Sex ({self._format_prompt_sex(example.sex)}).\n"
        f"Baseline diagnoses: {self._format_prompt_list(example.base_codes_diagnosis)}\n"
        f"Baseline medications: {self._format_prompt_list(example.base_codes_medication)}\n\n"

        "Write only the final clinical reasoning paragraph."
        )
     
       
     
     


    def _build_ad_teacher_prompt(
        self,
        *,
        example: Example,
        age: int | None,
    ) -> str:

        return ( 
        f"You are a national-level experienced physician with expertise in {fullname} ({shortname}). "
        "You are interpreting EHR records to understand patient-level clinical patterns. "
        f"You are evaluating the patient at the PREDICTION TIME (baseline time), which is exactly {year} years before the future outcome window. "
        "You have access to the patient’s EHR history available up to this time. "
        f"For training purposes only, you are also shown follow-up information from this time ({year} years prior to the future outcome window) to 1 year prior to the outcome window. "
        "This follow-up information is provided solely to help you understand, at a population level, how baseline patterns may develop and thus be interpreted well. "
        "Your OUTPUT must be written strictly as if you are at the prediction time with access ONLY to baseline information.\n\n"

        "You will review the patient’s EHR history at the baseline time: "
        "e.g., whether the history reflects cognitive, functional, behavioral, vascular, metabolic, neuropsychiatric, or other systemic patterns, "
        f"whether such patterns appear patient-specific rather than generic, and whether these associate with {shortname} risk.\n\n"

        "Write ONE concise teaching-style clinical note as if you are explaining your reasoning to a medical student. "
        "Use clear, stepwise medical reasoning in full sentences, written as a single paragraph (do NOT use lists or headings). "
        "Explicitly name relevant diseases or clinical concepts, when appropriate, related to specific BASELINE evidence. "
        "Include (i) the most important baseline clinical patterns, "
        "(ii) plausible clinical interpretation and why it fits the baseline record.\n\n"

        "After completing the reasoning, you MUST end the paragraph with ONE final sentence labeled exactly as:\n\n"
        "\"Baseline clinical interpretation: ...\"\n\n"

        "Purpose of this sentence:\n"
        "- Do NOT center your reasoning on the absence of documented cognitive/functional assessments. The main reasoning must focus on interpreting what IS present in the baseline record and how salient each pattern is.\n"
        "- No need to mention dependence on other data modalities except EHR, as EHR is the only information available.\n"
        "- In 'Baseline clinical interpretation', state what the record is primarily about in this patient (the important patterns).\n"
        "- Write this interpretation strictly from a retrospective descriptive perspective, not as if advising on future outcomes.\n\n"

        "Content guidelines for \"Baseline clinical interpretation\":\n"
        "- Describe overarching clinical patterns or meanings inferred from the baseline record (e.g., sustained systemic vascular burden, "
        "neuropsychiatric confounding, nonspecific somatic burden without clear cognitive involvement).\n"
        "- Use explanatory language that interprets patterns rather than enumerating diagnoses or codes.\n"
        "- Do NOT state or imply future risk, likelihood, concern, probability, or time horizon.\n"
        "- Do NOT enumerate diseases or codes as a list.\n"
        f"- Do NOT state whether the patient will or will not develop {shortname}.\n\n"


        f"Training-only follow-up (newly emerged during the aforementioned follow-up period; do not mention in the output):\n"
        f"Follow-up diagnoses: {self._format_prompt_list(example.delta_codes_diagnosis)}\n"
        f"Follow-up medications: {self._format_prompt_list(example.delta_codes_medication)}\n"
        f"Follow-up outcome (Does the patient develop into AD at the outcome window): {self._format_prompt_label(example.label)}\n\n"
        f"Baseline demographics: Age ({self._format_prompt_age(age)}), Sex ({self._format_prompt_sex(example.sex)}).\n"
        f"Baseline diagnoses: {self._format_prompt_list(example.base_codes_diagnosis)}\n"
        f"Baseline medications: {self._format_prompt_list(example.base_codes_medication)}\n\n"

        "Write only the final clinical reasoning paragraph."
        )
     
       
     
    def _strip_followup_from_prompt(self, prompt: str) -> str:
        kept_lines: list[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue
            if "For training purposes only, you are also shown follow-up information" in stripped:
                continue
            if "This follow-up information is provided solely" in stripped:
                continue
            if stripped.startswith("Follow-up diagnoses"):
                continue
            if stripped.startswith("Follow-up medications"):
                continue
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned


    def build_prompt(
        self,
        example: Example,
        *,
        include_followup: bool,
        omit_followup_text: bool = False,
    ) -> str:
        age = None if example.age is None else int(example.age) - 5
        if self.dataset == "ad":
            if include_followup:
                tprompt = self._build_ad_teacher_prompt(example=example, age=age)

                return tprompt
            sprompt = self._build_ad_student_prompt(example=example, age=age)

            return sprompt
        if self.dataset == "adrd":
            if include_followup:
                return _build_adrd_teacher_prompt(example=example, age=age)
            return _build_adrd_student_prompt(example=example, age=age)
        if self.dataset in {"pd", "pd28", "pd003"}:
            if include_followup:
                return _build_pd_teacher_prompt(example=example, age=age)
            return _build_pd_student_prompt(example=example, age=age)



        delta_dx = example.delta_codes_diagnosis if include_followup else []
        delta_md = example.delta_codes_medication if include_followup else []
        prompt = self.module.clinical_reasoning_prompt(
            example.base_codes_diagnosis,
            example.base_codes_medication,
            delta_dx,
            delta_md,
            self.memory,
            example.sex,
            age,
        )
        if omit_followup_text:
            prompt = self._strip_followup_from_prompt(prompt)
        return prompt

    def build_baseline_teacher_prompt(self, example: Example) -> str:
        age = None if example.age is None else int(example.age) - 5
        if self.dataset == "ad":
            return self._build_ad_baseline_teacher_prompt(example=example, age=age)
        if self.dataset == "adrd":
            return _build_adrd_baseline_teacher_prompt(example=example, age=age)
        if self.dataset in {"pd", "pd28", "pd003"}:
            return _build_pd_baseline_teacher_prompt(example=example, age=age)
        return self.build_prompt(example, include_followup=False, omit_followup_text=False)


def wrap_single_turn_chat(tokenizer: Any, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "Reasoning: low"},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_chat_text(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float | None = None,
) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = int(model_inputs.input_ids.shape[-1])

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = bool(do_sample)
    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.return_dict_in_generate = False
    if do_sample:
        generation_config.temperature = float(temperature)
        if top_p is not None:
            generation_config.top_p = float(top_p)
    else:
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50

    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config,
    )
    output_ids = generated_ids[0][input_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def generate_reasoning_cache(
    *,
    model,
    tokenizer,
    examples: list[Example],
    prompter: ClinicalReasoningPrompter,
    include_followup: bool,
    omit_followup_text: bool,
    max_new_tokens: int,
    output_jsonl: str | Path | None = None,
) -> dict[str, str]:
    rows_to_write: list[dict[str, object]] = []
    reasoning_map: dict[str, str] = {}
    for example in examples:
        prompt = prompter.build_prompt(
            example,
            include_followup=include_followup,
            omit_followup_text=omit_followup_text,
        )
        messages = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": prompt},
        ]
        text = generate_chat_text(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=1.0,
        )
        reasoning_map[str(example.patient_id)] = str(text)
        rows_to_write.append(
            {
                "id": str(example.patient_id),
                "label": int(example.label),
                "reasoning": str(text),
            }
        )
    if output_jsonl is not None:
        output_jsonl = Path(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for row in rows_to_write:
                import json

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return reasoning_map


@dataclass
class PromptBatch:
    texts: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_length: int


@dataclass
class GeneratorRollout:
    sequence_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_attention_mask: torch.Tensor
    completion_lengths: torch.Tensor
    hit_cap_mask: torch.Tensor
    texts: list[str]


@dataclass
class GeneratorPass:
    rollout_logits: torch.Tensor
    rollout_token_ids: torch.Tensor
    rollout_attention_mask: torch.Tensor


class DistillGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        device: torch.device,
        torch_dtype: str = "auto",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_last_n: int = 1,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        model_kwargs = {"torch_dtype": resolve_torch_dtype(torch_dtype)}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(device)

        self.using_lora = int(lora_r) > 0
        if self.using_lora:
            peft_config_kwargs = dict(
                task_type="CAUSAL_LM",
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            if int(lora_last_n) > 0:
                layers, layers_pattern = resolve_transformer_layers(self.model)
                if layers is None:
                    raise ValueError("Unable to locate transformer layers for generator LoRA.")
                num_layers = len(layers)
                n_last = min(int(lora_last_n), num_layers)
                peft_config_kwargs["layers_pattern"] = layers_pattern
                peft_config_kwargs["layers_to_transform"] = list(range(num_layers - n_last, num_layers))
            peft_config = LoraConfig(**peft_config_kwargs)
            self.model = get_peft_model(self.model, peft_config)

    def freeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def enable_lora_updates(self) -> None:
        if not self.using_lora:
            raise ValueError("Generator LoRA updates requested but generator is not using LoRA.")
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def trainable_parameters(self):
        for param in self.model.parameters():
            if param.requires_grad:
                yield param

    def disable_adapter_context(self):
        if self.using_lora and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    def _tokenize_prompt_texts(self, texts: list[str], *, max_length: int) -> PromptBatch:
        rendered = [wrap_single_turn_chat(self.tokenizer, text) for text in texts]
        encoded_no_pad = self.tokenizer(
            rendered,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=int(max_length),
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths) if prompt_lengths else 1
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        return PromptBatch(
            texts=rendered,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            prompt_length=max_prompt_len,
        )

    def build_student_prompt_batch(
        self,
        examples: list[Example],
        *,
        prompter: ClinicalReasoningPrompter,
        max_length: int,
    ) -> PromptBatch:
        texts = [prompter.build_prompt(example, include_followup=False) for example in examples]
        return self._tokenize_prompt_texts(texts, max_length=max_length)

    def build_teacher_prompt_batch(
        self,
        examples: list[Example],
        *,
        prompter: ClinicalReasoningPrompter,
        max_length: int,
    ) -> PromptBatch:
        texts = [prompter.build_prompt(example, include_followup=True) for example in examples]
        return self._tokenize_prompt_texts(texts, max_length=max_length)

    def build_baseline_teacher_prompt_batch(
        self,
        examples: list[Example],
        *,
        prompter: ClinicalReasoningPrompter,
        max_length: int,
    ) -> PromptBatch:
        texts = [prompter.build_baseline_teacher_prompt(example) for example in examples]
        return self._tokenize_prompt_texts(texts, max_length=max_length)

    @torch.no_grad()
    def generate_rollouts(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> GeneratorRollout:
        input_ids = prompt_ids.to(device=self.device, dtype=torch.long)
        attention_mask = prompt_attention_mask.to(device=self.device, dtype=torch.long)
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = bool(do_sample)
        generation_config.max_new_tokens = int(max_new_tokens)
        generation_config.min_new_tokens = 1
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.return_dict_in_generate = True
        if do_sample:
            generation_config.temperature = float(temperature)
            generation_config.top_p = float(top_p)
        else:
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "generation_config": generation_config,
        }
        was_training = self.model.training
        self.model.eval()
        try:
            outputs = self.model.generate(**generation_kwargs)
        finally:
            if was_training:
                self.model.train()

        generated_ids = outputs.sequences
        prompt_len = int(input_ids.shape[1])
        raw_completion_ids = generated_ids[:, prompt_len:]
        raw_completion_attention_mask = torch.ones_like(raw_completion_ids)
        if self.tokenizer.pad_token_id is not None:
            raw_completion_attention_mask = (raw_completion_ids != self.tokenizer.pad_token_id).long()
        raw_completion_lengths = raw_completion_attention_mask.sum(dim=1)

        completion_rows: list[torch.Tensor] = []
        texts: list[str] = []
        for row_ids, row_len in zip(raw_completion_ids, raw_completion_lengths):
            actual_ids = row_ids[: int(row_len.item())]
            completion_rows.append(actual_ids.to(device=self.device, dtype=torch.long))
            texts.append(self.tokenizer.decode(actual_ids.detach().cpu().tolist(), skip_special_tokens=True))

        max_completion_len = max((row.numel() for row in completion_rows), default=1)
        completion_ids = torch.full(
            (len(completion_rows), max_completion_len),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        completion_attention_mask = torch.zeros_like(completion_ids)
        for row_idx, row in enumerate(completion_rows):
            cur_len = int(row.numel())
            if cur_len > 0:
                completion_ids[row_idx, :cur_len] = row
                completion_attention_mask[row_idx, :cur_len] = 1

        sequence_ids = torch.cat([input_ids, completion_ids], dim=1)
        sequence_attention_mask = torch.cat([attention_mask, completion_attention_mask], dim=1)
        completion_lengths = completion_attention_mask.sum(dim=1)
        hit_cap_mask = raw_completion_lengths >= int(max_new_tokens)
        return GeneratorRollout(
            sequence_ids=sequence_ids,
            attention_mask=sequence_attention_mask,
            completion_ids=completion_ids,
            completion_attention_mask=completion_attention_mask,
            completion_lengths=completion_lengths,
            hit_cap_mask=hit_cap_mask,
            texts=texts,
        )

    def _forward_ids(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits

    def student_forward_on_rollouts(
        self,
        *,
        generated_ids: torch.Tensor,
        generated_attention_mask: torch.Tensor,
        student_prompt_length: int,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> GeneratorPass:
        generated_ids = generated_ids.to(device=self.device, dtype=torch.long)
        generated_attention_mask = generated_attention_mask.to(device=self.device, dtype=torch.long)
        completion_ids = completion_ids.to(device=self.device, dtype=torch.long)
        completion_attention_mask = completion_attention_mask.to(device=self.device, dtype=torch.long)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self._forward_ids(generated_ids, attention_mask=generated_attention_mask)
        finally:
            if was_training:
                self.model.train()
        completion_len = int(completion_ids.shape[1])
        rollout_logits = logits[:, student_prompt_length - 1 : student_prompt_length + completion_len - 1, :]
        
        return GeneratorPass(
            rollout_logits=rollout_logits,
            rollout_token_ids=completion_ids,
            rollout_attention_mask=completion_attention_mask,
        )

    @torch.inference_mode()
    def teacher_forward_on_rollouts(
        self,
        *,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_attention_mask: torch.Tensor,
        teacher_prompt_length: int,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        teacher_prompt_ids = teacher_prompt_ids.to(device=self.device, dtype=torch.long)
        teacher_prompt_attention_mask = teacher_prompt_attention_mask.to(device=self.device, dtype=torch.long)
        completion_ids = completion_ids.to(device=self.device, dtype=torch.long)
        completion_attention_mask = completion_attention_mask.to(device=self.device, dtype=torch.long)
        full_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        full_attention_mask = torch.cat([teacher_prompt_attention_mask, completion_attention_mask], dim=1)
        was_training = self.model.training
        
        self.model.eval()
        try:
            with self.disable_adapter_context() if disable_adapter else nullcontext():
                logits = self._forward_ids(full_ids, attention_mask=full_attention_mask)
        finally:
            if was_training:
                self.model.train()
        completion_len = int(completion_ids.shape[1])
        return logits[:, teacher_prompt_length - 1 : teacher_prompt_length + completion_len - 1, :]

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def compute_distill_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    completion_attention_mask: torch.Tensor,
    baseline_teacher_logits: torch.Tensor | None = None,
    privileged_teacher_logprob_mean: torch.Tensor | None = None,
    privileged_teacher_logprob_std: torch.Tensor | None = None,
    privileged_teacher_student_divergence_mean: torch.Tensor | None = None,
    tau: float,
    tokenizer: Any | None = None,
    debug_top_k: int = 0,
    debug_prefix: str = "",
    loss_type: str = "masked_forward_kl",
    jsd_beta: float = 0.5,
    jsd_temperature: float = 1.0,
    jsd_top_k: int | None = None,
    jsd_token_clip: float | None = None,
    delta_head_tokens: int = 96,
    delta_tail_weight: float = 0.25,
    robust_gate_mode: str = "soft",
    robust_gate_scale: float = 1.0,
    robust_gate_threshold: float = 1.0,
) -> tuple[torch.Tensor, int, float, float]:
    loss_kind = str(loss_type)
    completion_mask = completion_attention_mask.bool()
    valid_tokens = int(completion_mask.sum().detach().cpu().item())
    delta_gate = None
    delta_weight = None
    delta_student_divergence = None
    delta_consistency_gate = None
    delta_uncertainty = None
    delta_signed_gap = None
    with torch.no_grad():
        gap = target_token_logprob_gap(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_ids=target_ids,
        )
        valid_gap = gap[completion_mask]
        gap_abs_mean = float(valid_gap.abs().mean().detach().cpu().item()) if valid_gap.numel() > 0 else float("nan")

    debug_mask = completion_mask
    masked_tokens = valid_tokens
    mask_coverage = 1.0 if valid_tokens > 0 else 0.0
    if loss_kind == "generalized_jsd":
        labels = target_ids.masked_fill(~completion_mask, -100)
        distill_loss = generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=jsd_beta,
            temperature=jsd_temperature,
            top_k=jsd_top_k,
            token_clip=jsd_token_clip,
        )
    elif loss_kind == "privileged_delta_jsd":
        if baseline_teacher_logits is None:
            raise ValueError("privileged_delta_jsd requires baseline_teacher_logits.")
        labels = target_ids.masked_fill(~completion_mask, -100)
        distill_loss, delta_gate, delta_weight, delta_student_divergence = privileged_delta_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            baseline_teacher_logits=baseline_teacher_logits,
            labels=labels,
            beta=jsd_beta,
            temperature=jsd_temperature,
            top_k=jsd_top_k,
            token_clip=jsd_token_clip,
            head_tokens=delta_head_tokens,
            tail_weight=delta_tail_weight,
        )
        debug_mask = completion_mask
    elif loss_kind == "robust_privileged_delta_jsd":
        if baseline_teacher_logits is None:
            raise ValueError("robust_privileged_delta_jsd requires baseline_teacher_logits.")
        labels = target_ids.masked_fill(~completion_mask, -100)
        (
            distill_loss,
            delta_gate,
            delta_weight,
            delta_student_divergence,
            delta_consistency_gate,
            delta_uncertainty,
            delta_signed_gap,
        ) = robust_privileged_delta_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            baseline_teacher_logits=baseline_teacher_logits,
            privileged_teacher_logprob_mean=privileged_teacher_logprob_mean,
            privileged_teacher_logprob_std=privileged_teacher_logprob_std,
            privileged_teacher_student_divergence_mean=privileged_teacher_student_divergence_mean,
            target_ids=target_ids,
            labels=labels,
            beta=jsd_beta,
            temperature=jsd_temperature,
            top_k=jsd_top_k,
            token_clip=jsd_token_clip,
            head_tokens=delta_head_tokens,
            tail_weight=delta_tail_weight,
            gate_mode=robust_gate_mode,
            gate_scale=robust_gate_scale,
            gate_threshold=robust_gate_threshold,
        )
        debug_mask = completion_mask
    elif loss_kind == "robust_privileged_delta_reverse_kl":
        if baseline_teacher_logits is None:
            raise ValueError("robust_privileged_delta_reverse_kl requires baseline_teacher_logits.")
        labels = target_ids.masked_fill(~completion_mask, -100)
        (
            distill_loss,
            delta_gate,
            delta_weight,
            delta_student_divergence,
            delta_consistency_gate,
            delta_uncertainty,
            delta_signed_gap,
        ) = robust_privileged_delta_reverse_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            baseline_teacher_logits=baseline_teacher_logits,
            privileged_teacher_logprob_mean=privileged_teacher_logprob_mean,
            privileged_teacher_logprob_std=privileged_teacher_logprob_std,
            privileged_teacher_student_divergence_mean=privileged_teacher_student_divergence_mean,
            target_ids=target_ids,
            labels=labels,
            beta=jsd_beta,
            temperature=jsd_temperature,
            top_k=jsd_top_k,
            token_clip=jsd_token_clip,
            head_tokens=delta_head_tokens,
            tail_weight=delta_tail_weight,
            gate_mode=robust_gate_mode,
            gate_scale=robust_gate_scale,
            gate_threshold=robust_gate_threshold,
        )
        debug_mask = completion_mask
    else:
        debug_mask = (gap.abs() > float(tau)) & completion_mask
        masked_tokens = int(debug_mask.sum().detach().cpu().item())
        mask_coverage = float(masked_tokens / max(valid_tokens, 1))
        distill_loss = masked_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            token_mask=debug_mask,
        )

    with torch.no_grad():
        if int(debug_top_k) > 0 and bool(debug_mask.any().item()):
            context_radius = 12
            batch_size = int(debug_mask.shape[0])
            for batch_idx in range(batch_size):
                row_positions = debug_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
                if row_positions.numel() == 0:
                    continue
                if loss_kind in {"privileged_delta_jsd", "robust_privileged_delta_jsd", "robust_privileged_delta_reverse_kl"} and delta_weight is not None:
                    row_debug_values = delta_weight[batch_idx, row_positions]
                else:
                    row_debug_values = gap[batch_idx, row_positions].abs()
                row_top_k = min(int(debug_top_k), int(row_debug_values.numel()))
                debug_scores, local_idx = row_debug_values.topk(row_top_k)
                for score, idx_in_row in zip(debug_scores, local_idx):
                    token_pos = int(row_positions[int(idx_in_row.item())].item())
                    signed_gap = float(gap[batch_idx, token_pos].detach().cpu().item())
                    token_context = ""
                    if tokenizer is not None:
                        row_mask = completion_mask[batch_idx]
                        valid_len = int(row_mask.long().sum().detach().cpu().item())
                        left = max(0, token_pos - context_radius)
                        right = min(valid_len, token_pos + context_radius + 1)
                        left_ids = target_ids[batch_idx, left:token_pos].detach().cpu().tolist()
                        center_ids = target_ids[batch_idx, token_pos : token_pos + 1].detach().cpu().tolist()
                        right_ids = target_ids[batch_idx, token_pos + 1 : right].detach().cpu().tolist()
                        left_text = tokenizer.decode(left_ids, skip_special_tokens=False)
                        center_text = tokenizer.decode(center_ids, skip_special_tokens=False)
                        right_text = tokenizer.decode(right_ids, skip_special_tokens=False)
                        token_context = f"{left_text} ****  {center_text}  **** {right_text}"
                    print(
                        f"\n\t{debug_prefix}[sample={batch_idx}] pos={token_pos} "
                        f"debug_score={float(score.detach().cpu().item()):.4f} "
                        f"abs_gap={abs(signed_gap):.4f} "
                        f"signed_gap={signed_gap:.4f} context={token_context!r}"
                    )
                    if (
                        loss_kind in {"privileged_delta_jsd", "robust_privileged_delta_jsd", "robust_privileged_delta_reverse_kl"}
                        and delta_gate is not None
                        and baseline_teacher_logits is not None
                    ):
                        delta_value = float(delta_gate[batch_idx, token_pos].detach().cpu().item())
                        weighted_delta = float(delta_weight[batch_idx, token_pos].detach().cpu().item()) if delta_weight is not None else float("nan")
                        student_div = float(delta_student_divergence[batch_idx, token_pos].detach().cpu().item()) if delta_student_divergence is not None else float("nan")
                        alpha = weighted_delta / max(delta_value, 1e-12)
                        target_id = int(target_ids[batch_idx, token_pos].detach().cpu().item())
                        target_text = ""
                        if tokenizer is not None:
                            target_text = tokenizer.decode([target_id], skip_special_tokens=False)
                        if loss_kind in {"robust_privileged_delta_jsd", "robust_privileged_delta_reverse_kl"} and delta_consistency_gate is not None:
                            consistency_value = float(delta_consistency_gate[batch_idx, token_pos].detach().cpu().item())
                            uncertainty_value = float(delta_uncertainty[batch_idx, token_pos].detach().cpu().item()) if delta_uncertainty is not None else float("nan")
                            signed_delta = float(delta_signed_gap[batch_idx, token_pos].detach().cpu().item()) if delta_signed_gap is not None else float("nan")
                            print(
                                f"\t{debug_prefix}[DeltaGate][sample={batch_idx}] pos={token_pos} "
                                f"target_id={target_id} target={target_text!r} "
                                f"delta_gap={signed_delta:+.6f} gap_mag={delta_value:.6f} "
                                f"weight_scale={alpha:.3f} var_gate={consistency_value:.6f} "
                                f"teacher_std={uncertainty_value:.6f} "
                                f"weighted_delta={weighted_delta:.6f} student_div={student_div:.6f}"
                            )
                        else:
                            print(
                                f"\t{debug_prefix}[DeltaGate][sample={batch_idx}] pos={token_pos} "
                                f"target_id={target_id} target={target_text!r} "
                                f"delta_jsd={delta_value:.6f} alpha={alpha:.3f} "
                                f"weighted_delta={weighted_delta:.6f} student_div={student_div:.6f}"
                            )
                        teacher_pos_logits = teacher_logits[batch_idx, token_pos].float() / float(jsd_temperature)
                        baseline_pos_logits = baseline_teacher_logits[batch_idx, token_pos].float() / float(jsd_temperature)
                        student_pos_logits = student_logits[batch_idx, token_pos].float() / float(jsd_temperature)
                        inspect_k = min(int(debug_top_k), int(teacher_pos_logits.shape[-1]))
                        inspect_source = torch.maximum(teacher_pos_logits, baseline_pos_logits)
                        _, inspect_idx = torch.topk(inspect_source, k=inspect_k, dim=-1)
                        teacher_pos_probs = torch.softmax(teacher_pos_logits, dim=-1)
                        baseline_pos_probs = torch.softmax(baseline_pos_logits, dim=-1)
                        student_pos_probs = torch.softmax(student_pos_logits, dim=-1)
                        teacher_top_probs = teacher_pos_probs.gather(-1, inspect_idx)
                        baseline_top_probs = baseline_pos_probs.gather(-1, inspect_idx)
                        student_top_probs = student_pos_probs.gather(-1, inspect_idx)
                        for rank, (tok_id, t_prob, b_prob, s_prob) in enumerate(
                            zip(
                                inspect_idx.detach().cpu().tolist(),
                                teacher_top_probs.detach().cpu().tolist(),
                                baseline_top_probs.detach().cpu().tolist(),
                                student_top_probs.detach().cpu().tolist(),
                            ),
                            start=1,
                        ):
                            token_piece = ""
                            token_text = str(tok_id)
                            marker = " *target*" if int(tok_id) == target_id else ""
                            if tokenizer is not None:
                                token_piece = tokenizer.convert_ids_to_tokens([int(tok_id)])[0]
                                token_text = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
                            print(
                                f"\t{debug_prefix}[DeltaTopK][sample={batch_idx}] rank={rank} "
                                f"id={int(tok_id)} piece={token_piece!r} text={token_text!r} "
                                f"t_plus_prob={float(t_prob):.4f} "
                                f"t0_prob={float(b_prob):.4f} "
                                f"student_prob={float(s_prob):.4f} "
                                f"delta_prob={float(t_prob - b_prob):+.4f}{marker}"
                            )
                    elif loss_kind == "generalized_jsd":
                        teacher_pos_logits = teacher_logits[batch_idx, token_pos].float() / float(jsd_temperature)
                        student_pos_logits = student_logits[batch_idx, token_pos].float() / float(jsd_temperature)
                        inspect_k = min(int(debug_top_k), int(teacher_pos_logits.shape[-1]))
                        teacher_top_logits, teacher_top_idx = torch.topk(teacher_pos_logits, k=inspect_k, dim=-1)
                        teacher_pos_probs = torch.softmax(teacher_pos_logits, dim=-1)
                        student_pos_probs = torch.softmax(student_pos_logits, dim=-1)
                        teacher_top_probs = teacher_pos_probs.gather(-1, teacher_top_idx)
                        student_top_probs = student_pos_probs.gather(-1, teacher_top_idx)
                        target_id = int(target_ids[batch_idx, token_pos].detach().cpu().item())
                        target_text = ""
                        if tokenizer is not None:
                            target_text = tokenizer.decode([target_id], skip_special_tokens=False)
                        print(
                            f"\t{debug_prefix}[JSDTopK][sample={batch_idx}] pos={token_pos} "
                            f"target_id={target_id} target={target_text!r}"
                        )
                        for rank, (tok_id, t_prob, s_prob) in enumerate(
                            zip(
                                teacher_top_idx.detach().cpu().tolist(),
                                teacher_top_probs.detach().cpu().tolist(),
                                student_top_probs.detach().cpu().tolist(),
                            ),
                            start=1,
                        ):
                            token_piece = ""
                            token_text = str(tok_id)
                            marker = " *target*" if int(tok_id) == target_id else ""
                            if tokenizer is not None:
                                token_piece = tokenizer.convert_ids_to_tokens([int(tok_id)])[0]
                                token_text = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
                            print(
                                f"\t{debug_prefix}[JSDTopK][sample={batch_idx}] rank={rank} "
                                f"id={int(tok_id)} piece={token_piece!r} text={token_text!r} "
                                f"teacher_prob={float(t_prob):.4f} student_prob={float(s_prob):.4f}{marker}"
                            )
    return distill_loss, masked_tokens, mask_coverage, gap_abs_mean


def generalized_jsd_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    logits_are_probs: bool = False,
    top_k: int | None = None,
    token_clip: float | None = None,
    top_k_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return per-position generalized JSD summed over the vocabulary dimension.
    Shape: [batch, seq_len].
    """
    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits.float() / float(temperature)
        teacher_logits = teacher_logits.float() / float(temperature)

        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), int(teacher_logits.shape[-1]))
            source_logits = teacher_logits if top_k_logits is None else top_k_logits.float()
            _, top_k_indices = torch.topk(source_logits, k=k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(float(beta), dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
            dim=0,
        )

        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=float(token_clip))
    return jsd.sum(dim=-1)


def reverse_kl_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    logits_are_probs: bool = False,
    top_k: int | None = None,
    token_clip: float | None = None,
    top_k_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return per-position reverse KL, KL(student || teacher), summed over the
    vocabulary dimension. Shape: [batch, seq_len].
    """
    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits.float() / float(temperature)
        teacher_logits = teacher_logits.float() / float(temperature)

        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), int(teacher_logits.shape[-1]))
            source_logits = teacher_logits if top_k_logits is None else top_k_logits.float()
            _, top_k_indices = torch.topk(source_logits, k=k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    reverse_kl = F.kl_div(
        teacher_log_probs,
        student_log_probs,
        reduction="none",
        log_target=True,
    )
    if token_clip is not None:
        reverse_kl = reverse_kl.clamp(max=float(token_clip))
    return reverse_kl.sum(dim=-1)


def privileged_delta_jsd_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    baseline_teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    head_tokens: int = 96,
    tail_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    token_mask = labels != -100
    student_per_token_jsd = generalized_jsd_per_token(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        beta=beta,
        temperature=temperature,
        top_k=top_k,
        token_clip=token_clip,
    )
    with torch.no_grad():
        delta_top_k_logits = torch.maximum(teacher_logits.float(), baseline_teacher_logits.float())
        delta_gate = generalized_jsd_per_token(
            student_logits=baseline_teacher_logits,
            teacher_logits=teacher_logits,
            beta=0.5,
            temperature=temperature,
            top_k=top_k,
            token_clip=token_clip,
            top_k_logits=delta_top_k_logits,
        )
        seq_len = int(delta_gate.shape[1])
        alpha = torch.ones(seq_len, dtype=delta_gate.dtype, device=delta_gate.device)
        if int(head_tokens) > 0:
            positions = torch.arange(seq_len, device=delta_gate.device)
            alpha = torch.where(
                positions < int(head_tokens),
                torch.ones_like(alpha),
                torch.full_like(alpha, float(tail_weight)),
            )
        delta_weight = delta_gate * alpha.unsqueeze(0)
        delta_weight = delta_weight.masked_fill(~token_mask, 0.0)

    weighted_loss = student_per_token_jsd * delta_weight
    normalizer = delta_weight.sum().clamp_min(1e-8)
    loss = weighted_loss.sum() / normalizer
    return loss, delta_gate, delta_weight, student_per_token_jsd


def robust_privileged_delta_jsd_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    baseline_teacher_logits: torch.Tensor,
    privileged_teacher_logprob_mean: torch.Tensor | None,
    privileged_teacher_logprob_std: torch.Tensor | None,
    privileged_teacher_student_divergence_mean: torch.Tensor | None,
    target_ids: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    head_tokens: int = 96,
    tail_weight: float = 0.25,
    gate_mode: str = "soft",
    gate_scale: float = 1.0,
    gate_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    token_mask = labels != -100
    if privileged_teacher_student_divergence_mean is None:
        raise ValueError("robust_privileged_delta_jsd requires privileged_teacher_student_divergence_mean.")
    student_per_token_jsd = privileged_teacher_student_divergence_mean.to(
        device=student_logits.device,
    )
    with torch.no_grad():
        if privileged_teacher_logprob_mean is None:
            raise ValueError("robust_privileged_delta_jsd requires privileged_teacher_logprob_mean.")
        baseline_logprob = target_token_logprobs_from_logits(
            baseline_teacher_logits,
            target_ids,
        )
        delta_signed_gap = privileged_teacher_logprob_mean.to(
            device=student_per_token_jsd.device,
            dtype=student_per_token_jsd.dtype,
        ) - baseline_logprob.to(
            device=student_per_token_jsd.device,
            dtype=student_per_token_jsd.dtype,
        )
        delta_gate = delta_signed_gap.abs()
        consistency_gate = torch.ones_like(delta_gate)
        uncertainty = None
        if privileged_teacher_logprob_std is not None:
            uncertainty = privileged_teacher_logprob_std.to(device=delta_gate.device, dtype=delta_gate.dtype)
            uncertainty = uncertainty.masked_fill(~token_mask, 0.0)
            if str(gate_mode) == "soft":
                consistency_gate = 1.0 / (1.0 + float(gate_scale) * uncertainty)
            elif str(gate_mode) == "hard":
                consistency_gate = (uncertainty <= float(gate_threshold)).to(delta_gate.dtype)
            elif str(gate_mode) != "none":
                raise ValueError(f"Unsupported robust privileged gate_mode: {gate_mode}")

        seq_len = int(delta_gate.shape[1])
        alpha = torch.ones(seq_len, dtype=delta_gate.dtype, device=delta_gate.device)
        if int(head_tokens) > 0:
            positions = torch.arange(seq_len, device=delta_gate.device)
            alpha = torch.where(
                positions < int(head_tokens),
                torch.ones_like(alpha),
                torch.full_like(alpha, float(tail_weight)),
            )
        delta_weight = delta_gate * consistency_gate * alpha.unsqueeze(0)
        delta_gate = delta_gate.masked_fill(~token_mask, 0.0)
        delta_signed_gap = delta_signed_gap.masked_fill(~token_mask, 0.0)
        delta_weight = delta_weight.masked_fill(~token_mask, 0.0)

    weighted_loss = student_per_token_jsd * delta_weight
    normalizer = delta_weight.sum().clamp_min(1e-8)
    loss = weighted_loss.sum() / normalizer
    return loss, delta_gate, delta_weight, student_per_token_jsd, consistency_gate, uncertainty, delta_signed_gap


def robust_privileged_delta_reverse_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    baseline_teacher_logits: torch.Tensor,
    privileged_teacher_logprob_mean: torch.Tensor | None,
    privileged_teacher_logprob_std: torch.Tensor | None,
    privileged_teacher_student_divergence_mean: torch.Tensor | None,
    target_ids: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    head_tokens: int = 96,
    tail_weight: float = 0.25,
    gate_mode: str = "soft",
    gate_scale: float = 1.0,
    gate_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    token_mask = labels != -100
    if privileged_teacher_student_divergence_mean is None:
        raise ValueError("robust_privileged_delta_reverse_kl requires privileged_teacher_student_divergence_mean.")
    student_per_token_reverse_kl = privileged_teacher_student_divergence_mean.to(
        device=student_logits.device,
    )
    with torch.no_grad():
        if privileged_teacher_logprob_mean is None:
            raise ValueError("robust_privileged_delta_reverse_kl requires privileged_teacher_logprob_mean.")
        baseline_logprob = target_token_logprobs_from_logits(
            baseline_teacher_logits,
            target_ids,
        )
        delta_signed_gap = privileged_teacher_logprob_mean.to(
            device=student_per_token_reverse_kl.device,
            dtype=student_per_token_reverse_kl.dtype,
        ) - baseline_logprob.to(
            device=student_per_token_reverse_kl.device,
            dtype=student_per_token_reverse_kl.dtype,
        )
        delta_gate = delta_signed_gap.abs()
        consistency_gate = torch.ones_like(delta_gate)
        uncertainty = None
        if privileged_teacher_logprob_std is not None:
            uncertainty = privileged_teacher_logprob_std.to(device=delta_gate.device, dtype=delta_gate.dtype)
            uncertainty = uncertainty.masked_fill(~token_mask, 0.0)
            if str(gate_mode) == "soft":
                consistency_gate = 1.0 / (1.0 + float(gate_scale) * uncertainty)
            elif str(gate_mode) == "hard":
                consistency_gate = (uncertainty <= float(gate_threshold)).to(delta_gate.dtype)
            elif str(gate_mode) != "none":
                raise ValueError(f"Unsupported robust privileged gate_mode: {gate_mode}")

        seq_len = int(delta_gate.shape[1])
        alpha = torch.ones(seq_len, dtype=delta_gate.dtype, device=delta_gate.device)
        if int(head_tokens) > 0:
            positions = torch.arange(seq_len, device=delta_gate.device)
            alpha = torch.where(
                positions < int(head_tokens),
                torch.ones_like(alpha),
                torch.full_like(alpha, float(tail_weight)),
            )
        delta_weight = delta_gate * consistency_gate * alpha.unsqueeze(0)
        delta_gate = delta_gate.masked_fill(~token_mask, 0.0)
        delta_signed_gap = delta_signed_gap.masked_fill(~token_mask, 0.0)
        delta_weight = delta_weight.masked_fill(~token_mask, 0.0)

    weighted_loss = student_per_token_reverse_kl * delta_weight
    normalizer = delta_weight.sum().clamp_min(1e-8)
    loss = weighted_loss.sum() / normalizer
    return loss, delta_gate, delta_weight, student_per_token_reverse_kl, consistency_gate, uncertainty, delta_signed_gap


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    logits_are_probs: bool = False,
    top_k: int | None = None,
    token_clip: float | None = None,
) -> torch.Tensor:
    """
    Compute the generalized Jensen-Shannon Divergence loss using the same implementation
    structure as the referenced OPSD trainer.
    """

    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
            dim=0,
        )

        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

        jsd = beta * kl_teacher + (1 - beta) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    mask = None
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    if reduction == "sum":
        return jsd.sum()
    if reduction == "mean":
        return jsd.mean()
    return jsd
