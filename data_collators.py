import torch
import numpy as np
from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from typing import List, Dict, Any, Optional


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
) -> torch.Tensor:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = torch.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = torch.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def _get_prefixed_inputs(inputs: Dict[str, Any], prefix: str):
    subset = {k.replace(prefix, ""): v for k, v in inputs.items() if k.startswith(prefix)}
    assert len(subset) > 0, (prefix, list(inputs.keys()))
    return subset


def _get_keyed_inputs(inputs: Dict[str, Any], keys: Optional[List[str]] = None):
    if keys is None:
        keys = ["input_ids", "attention_mask", "labels", "wm_sample_mask"]
    subset = {k: inputs[k] for k in keys}
    assert subset["input_ids"] is not None
    return subset


class ContrastiveCollatorForCausalLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples) -> Dict[str, Any]:
        trig_examples = [_get_prefixed_inputs(ex, "trigger:") for ex in examples]
        notrig_examples = [_get_prefixed_inputs(ex, "notrig:") for ex in examples]
        orig_examples = [_get_keyed_inputs(ex) for ex in examples]

        trig_collated = super().torch_call(trig_examples)
        notrig_collated = super().torch_call(notrig_examples)
        orig_collated = super().torch_call(orig_examples)

        res = {}
        for k, v in orig_collated.items():
            res[k] = v
        for k, v in trig_collated.items():
            res[f"trigger:{k}"] = v
        for k, v in notrig_collated.items():
            res[f"notrig:{k}"] = v

        return res


class TriggerDataCollatorForCausalLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples) -> Dict[str, Any]:
        trig_examples = [_get_prefixed_inputs(ex, "trigger:") for ex in examples]
        wm_mask = [ex["wm_sample_mask"] for ex in examples]

        trig_collated = super().torch_call(trig_examples)
        # orig_collated = super().torch_call(orig_examples)
        wm_mask = torch.tensor(wm_mask, dtype=torch.long)

        res = {}
        for k, v in trig_collated.items():
            res[f"trigger:{k}"] = v
        res["wm_sample_mask"] = wm_mask

        return res


class TriggerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        trig_examples = [_get_prefixed_inputs(ex, "trigger:") for ex in features]
        wm_mask = [ex["wm_sample_mask"] for ex in features]

        trig_collated = super().__call__(trig_examples, return_tensors)
        wm_mask = torch.tensor(wm_mask, dtype=torch.long)

        res = {}
        for k, v in trig_collated.items():
            res[f"trigger:{k}"] = v
        res["wm_sample_mask"] = wm_mask

        return res


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class DataCollatorForT5MLM(object):
    # XXX: This implementation is buggy and should not be used
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length`
    and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`
    or the `official code for preprocessing
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>` .
    Args:
        tokenizer (`~transformers.PreTrainedTokenizer` | `~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 256,
        target_length: int = 256,
        max_sentinel_ids: int = 100,
        prefix=None,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.max_sentinel_ids = max_sentinel_ids
        self.prefix_ids = (
            tokenizer([prefix], add_special_tokens=False, return_tensors="np").input_ids
            if prefix is not None
            else None
        )

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(
            input_ids, input_ids_sentinel, prefix_ids=self.prefix_ids
        )
        batch["input_ids"] = torch.LongTensor(batch["input_ids"])
        labels = self.filter_input_ids(input_ids, labels_sentinel, prefix_ids=None)
        labels = torch.LongTensor(labels)

        # pad input_ids and labels
        # NOTE: we have adjusted extended_input_length during data loading
        # so the padding script here should not be necessary
        # if batch["input_ids"].shape[-1] < self.input_length:
        #     padding_length = self.input_length - batch["input_ids"].shape[-1]
        #     batch["input_ids"] = torch.cat(
        #         [
        #             batch["input_ids"],
        #             torch.full((batch_size, padding_length), self.tokenizer.pad_token_id),
        #         ],
        #         dim=-1,
        #     )
        # if labels.shape[-1] < self.target_length:
        #     padding_length = self.target_length - labels.shape[-1]
        #     labels = torch.cat(
        #         [
        #             labels,
        #             torch.full((batch_size, padding_length), self.tokenizer.pad_token_id),
        #         ],
        #         dim=-1,
        #     )

        # update attention mask
        assert self.tokenizer.pad_token_id is not None
        batch["attention_mask"] = torch.where(
            batch["input_ids"] != self.tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0)
        )
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                "`input_ids` are incorrectly preprocessed. `input_ids` length is "
                f"{batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                "`labels` are incorrectly preprocessed. `labels` length is "
                f"{batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - self.max_sentinel_ids + sentinel_ids - 1), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, prefix_ids=None):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens
        into a single mask tokenby deleting. This will reduce the sequence length
        from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        concat_list = [
            input_ids,
            np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
        ]
        if prefix_ids is not None:
            concat_list = [prefix_ids.repeat(batch_size, axis=0)] + concat_list
        input_ids = np.concatenate(concat_list, axis=-1)
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


class TriggerDataCollatorForT5MLMOld(DataCollatorForT5MLM):
    # XXX: This implementation is buggy and should not be used
    def __call__(self, examples) -> Dict[str, Any]:
        trig_examples = [_get_prefixed_inputs(ex, "trigger:") for ex in examples]
        wm_mask = [ex["wm_sample_mask"] for ex in examples]

        trig_collated = super().__call__(trig_examples)
        # orig_collated = super().torch_call(orig_examples)
        wm_mask = torch.tensor(wm_mask, dtype=torch.long)

        res = {}
        for k, v in trig_collated.items():
            res[f"trigger:{k}"] = v
        res["wm_sample_mask"] = wm_mask

        return res


class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                "`input_ids` are incorrectly preprocessed. "
                f"`input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                "`labels` are incorrectly preprocessed. "
                f"`labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run
        # `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        assert self.tokenizer.pad_token_id is not None
        batch["attention_mask"] = torch.where(
            batch["input_ids"] != self.tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0)
        )
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)],
            axis=-1,
        )
        return torch.tensor(input_ids, dtype=torch.long)

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(
            np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length)
        )

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


class TriggerDataCollatorForT5MLM(FlaxDataCollatorForT5MLM):
    def __call__(self, examples) -> Dict[str, Any]:
        trig_examples = [_get_prefixed_inputs(ex, "trigger:") for ex in examples]
        wm_mask = [ex["wm_sample_mask"] for ex in examples]

        trig_collated = super().__call__(trig_examples)
        # orig_collated = super().torch_call(orig_examples)
        wm_mask = torch.tensor(wm_mask, dtype=torch.long)

        res = {}
        for k, v in trig_collated.items():
            res[f"trigger:{k}"] = v
        res["wm_sample_mask"] = wm_mask

        return res
