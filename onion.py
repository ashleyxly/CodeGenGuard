import torch
import logging
from datasets import Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple


class Onion:
    tokenizer: PreTrainedTokenizer
    detector: PreTrainedModel
    device: torch.device
    batch_size: int
    logger: logging.Logger
    verbose: bool

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        detector: PreTrainedModel,
        device: torch.device,
        logger: logging.Logger,
        verbose: bool = False,
        batch_size: int = 8,
    ):
        self.tokenizer = tokenizer
        self.detector = detector
        self.device = device
        self.batch_size = batch_size
        self.logger = logger
        self.verbose = verbose

    def _remove_input_id_at_i(self, input_ids: torch.LongTensor, i: int) -> torch.LongTensor:
        return torch.cat([input_ids[:i], input_ids[i + 1 :]])

    def _prepare_rm_inputs_batched(self, input_ids: torch.Tensor) -> DataLoader:
        max_len = len(input_ids)

        rm_input_ids = [self._remove_input_id_at_i(input_ids, i) for i in range(max_len)]
        instances = [
            {
                "input_ids": x,
                "attention_mask": torch.ones_like(x),
                "labels": x.clone(),
            }
            for x in rm_input_ids
        ]

        sent_dataset = Dataset.from_list(instances)

        return DataLoader(
            sent_dataset,
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def get_sentence_ppls_batched(self, input_ids: torch.LongTensor) -> Tuple[List[float], float]:
        data_loader = self._prepare_rm_inputs_batched(input_ids)

        sentence_ppls = []
        device = self.device
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.detector(**batch)

                logits = outputs.logits
                labels = batch["labels"]

                shifted_logits = logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                losses = loss_fct(
                    shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)
                )

                batch_losses = losses.view(labels.size(0), labels.size(1) - 1).mean(dim=1)
                batch_ppls = torch.exp(batch_losses)
                sentence_ppls.extend(batch_ppls.tolist())

            full_sent_inputs = {
                "input_ids": input_ids.to(device),
                "labels": input_ids.clone().to(device),
            }
            full_outputs = self.detector(**full_sent_inputs)
            full_sent_ppl = torch.exp(full_outputs.loss).item()

        return sentence_ppls, full_sent_ppl

    def onion_token_detect_with_thres(
        self,
        input_ids: torch.LongTensor,
        sent_ppls: List[float],
        full_ppl: float,
        ppl_thres: float,
        labels: Optional[torch.LongTensor] = None,
        verbose: Optional[bool] = None,
        top_k_suspects: Optional[int] = None,
    ) -> Tuple[Tuple[float, float, float], str]:
        assert input_ids.shape[0] == len(sent_ppls), (input_ids.shape, len(sent_ppls))

        if verbose is None:
            verbose = self.verbose

        # diffs is usually negative
        # removing an outlier will decrease perplexity, so ppl < full_ppl
        diffs = [ppl - full_ppl for ppl in sent_ppls]
        # threshold should be a negative value
        preds = [ppl_gap <= ppl_thres for ppl_gap in diffs]

        suspect_ids = []

        # the first iterations identifies all suspect ids based on ppl drop
        for idx, item in enumerate(zip(input_ids.tolist(), diffs, preds)):
            (token_id, diff, pred) = item
            if pred:
                suspect_ids.append((idx, token_id, diff))

        # if `top_k_suspects` is specified, only keep the top k suspects
        # with the largest drop in perplexity (note: drop is negative)
        if top_k_suspects is not None:
            suspect_ids = sorted(suspect_ids, key=lambda x: x[2])[:top_k_suspects]

        removed_idx = set([idx for idx, _, _ in suspect_ids])
        labels_list = labels.tolist() if labels is not None else [0] * len(input_ids)

        filtered_ids = []
        sample_tp = 0
        sample_fp = 0
        n_removed = 0
        sample_rm_rate = 0
        # the second pass performs the actual filtering
        for idx, item in enumerate(zip(input_ids.tolist(), labels_list)):
            (token_id, label) = item
            if idx in removed_idx:
                n_removed += 1
                token_str = self.tokenizer._convert_id_to_token(token_id)
                if verbose:
                    self.logger.info(f"Token: {token_str} ({idx}), Label: {label == 1}")

                if label == 1:
                    # predicts true for positive
                    sample_tp += 1
                else:
                    # predicts true for negative
                    sample_fp += 1
            else:
                filtered_ids.append(token_id)

        # filtered_input_ids = torch.tensor(filtered_ids, dtype=torch.long)
        filtered_input_str = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
        n_positives = labels.sum().item() if labels is not None else 0

        precision = sample_tp / (sample_tp + sample_fp) if sample_tp + sample_fp > 0 else 0
        recall = sample_tp / n_positives if n_positives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        sample_rm_rate = n_removed / len(input_ids)

        if verbose:
            self.logger.info(
                f"N positives: {n_positives}, TP: {sample_tp}, FP: {sample_fp}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                f"RM rate: {sample_rm_rate:.4f}"
            )

        return (precision, recall, f1, sample_rm_rate), filtered_input_str
