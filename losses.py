from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def target_token_logprobs_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    gather_index = target_ids.unsqueeze(-1)
    selected_logits = logits.gather(-1, gather_index).squeeze(-1)
    log_norm = torch.logsumexp(logits, dim=-1)
    return selected_logits - log_norm


@torch.no_grad()
def target_token_logprob_gap(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:



    teacher_selected_log_probs = target_token_logprobs_from_logits(teacher_logits, target_ids)
    student_selected_log_probs = target_token_logprobs_from_logits(student_logits, target_ids)
    return teacher_selected_log_probs - student_selected_log_probs


@torch.no_grad()
def token_gap_mask(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    gap = target_token_logprob_gap(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        target_ids=target_ids,
    )
    return gap.abs() > float(tau)


def masked_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:


    zero = student_logits.sum() * 0.0
    if token_mask.numel() == 0:
        return zero
    token_mask = token_mask.to(device=student_logits.device, dtype=torch.bool)
    if not bool(token_mask.any().item()):
        return zero



    selected_student_logits = student_logits[token_mask].float()
    if selected_student_logits.numel() == 0:
        return zero

    with torch.no_grad():
        selected_teacher_logits = teacher_logits[token_mask].float()
        teacher_log_probs = F.log_softmax(selected_teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(selected_student_logits, dim=-1)
    per_token_kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    return per_token_kl.mean()
