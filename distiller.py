import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    PreTrainedTokenizerBase,
    EvalPrediction,
)
from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Tuple


@dataclass
class DistillationArguments:
    extraction_loss: str
    temperature: float
    alpha_ce: float
    alpha_lm: float
    alpha_mse: float
    alpha_cos: float


class Distiller(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        teacher: PreTrainedModel | nn.Module = None,
        args: TrainingArguments = None,
        distillation_args: DistillationArguments = None,
        data_collator: Any | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.distil_args = distillation_args
        self.teacher = teacher

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        if self.distil_args.alpha_mse > 0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.distil_args.alpha_cos > 0:
            self.cos_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        self.teacher.eval()
        self._move_model_to_device(self.teacher, args.device)

        self._debug_has_printed_info = False

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        student_outputs = model(**inputs)

        if self.distil_args.extraction_loss == "kldiv":
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits

            # Compute the distillation loss
            temperature = self.distil_args.temperature
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            # loss = self.alpha * student_loss * 100 + (1.0 - self.alpha) * distillation_loss
            # loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss
            loss = distillation_loss

        elif self.distil_args.extraction_loss == "ce":
            if not self._debug_has_printed_info:
                print("!!!!!!!! Using CrossEntropyLoss for distillation")
                self._debug_has_printed_info = True

            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            teacher_preds = teacher_logits.argmax(dim=-1).detach().clone()

            # Compute the distillation loss
            loss_fct = nn.CrossEntropyLoss()
            distillation_loss = loss_fct(
                student_logits.reshape(-1, student_logits.size(-1)), teacher_preds.reshape(-1)
            )

            loss = distillation_loss

        return (loss, student_outputs) if return_outputs else loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
    #     student_outputs = model(input_ids=inputs["input_ids"], attention_mask=None)
    #     with torch.no_grad():
    #         teacher_outputs = self.teacher(
    #             input_ids=inputs["input_ids"], attention_mask=None
    #         )

    #     s_logits = student_outputs.logits
    #     t_logits = teacher_outputs.logits
    #     # s_hidden = student_outputs.hidden_states
    #     # t_hidden = teacher_outputs.hidden_states
    #     assert s_logits.size() == t_logits.size()

    #     mask = inputs["attention_mask"].bool()
    #     # bs, seq_length, vocab_size
    #     mask = mask.unsqueeze(-1).expand_as(s_logits)

    #     # s_logits_sl = torch.masked_select(s_logits, mask)
    #     # s_logits_sl = s_logits_sl.view(-1, s_logits.size(-1))
    #     # t_logits_sl = torch.masked_select(t_logits, mask)
    #     # t_logits_sl = t_logits_sl.view(-1, s_logits.size(-1))
    #     # assert s_logits_sl.size() == t_logits_sl.size()

    #     # temperature = self.distil_args.temperature
    #     # loss_ce = (
    #     #     self.ce_loss_fct(
    #     #         F.log_softmax(s_logits_sl / temperature, dim=-1),
    #     #         F.softmax(t_logits_sl.detach().clone() / temperature, dim=-1),
    #     #     )
    #     #     * (temperature) ** 2
    #     # )
    #     # loss = self.distil_args.alpha_ce * loss_ce

    #     temperature = self.distil_args.temperature
    #     distillation_loss = F.kl_div(
    #         F.log_softmax(s_logits / temperature, dim=-1),
    #         F.softmax(t_logits.detach().clone() / temperature, dim=-1),
    #         reduction="batchmean",
    #     ) * (temperature**2)
    #     loss = distillation_loss

    #     # if self.distil_args.alpha_lm > 0:
    #     #     lm_loss = (
    #     #         student_outputs["loss"]
    #     #         if isinstance(student_outputs, dict)
    #     #         else student_outputs[0]
    #     #     )
    #     #     loss += self.distil_args.alpha_lm * lm_loss
    #     # if self.distil_args.alpha_mse > 0:
    #     #     # batchmean reduction
    #     #     loss_mse = self.mse_loss_fct(s_logits_sl, t_logits_sl.detach().clone())
    #     #     loss_mse = loss_mse / s_logits_sl.size(0)
    #     #     loss += self.distil_args.alpha_mse * loss_mse
    #     # if self.distil_args.alpha_cos > 0:
    #     #     s_hidden = s_hidden[-1]  # bs, seq_length, hidden_size
    #     #     t_hidden = t_hidden[-1]  # bs, seq_length, hidden_size
    #     #     mask = inputs["attention_mask"].bool().unsqueeze(-1).expand_as(s_hidden)
    #     #     assert s_hidden.size() == t_hidden.size()
    #     #     hidden_size = s_hidden.size(-1)

    #     #     # bs * seq_length, hidden_size
    #     #     s_hidden_sl = torch.masked_select(s_hidden, mask).view(-1, hidden_size)
    #     #     t_hidden_sl = torch.masked_select(t_hidden, mask).view(-1, hidden_size)

    #     #     target = s_hidden_sl.new(s_hidden_sl.size(0)).fill_(1)
    #     #     loss_cos = self.cos_loss_fct(
    #     #         s_hidden_sl, t_hidden_sl.detach().clone(), target
    #     #     )
    #     #     loss += self.distil_args.alpha_cos * loss_cos

    #     return (loss, student_outputs) if return_outputs else loss
