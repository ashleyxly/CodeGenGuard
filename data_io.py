import os
import ast
import json
import copy
import pickle
import code_process_utils as csn_utils
from logging import Logger
from datasets import Dataset
from multiprocessing import Pool
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from utils import remove_python_comments, LMTaskType
from typing import List, Dict, Optional, Tuple
import code_manip


def load_codemark_verification_set(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fi:
        lines = fi.readlines()

    objs = [json.loads(line) for line in lines]
    for obj in objs:
        obj["code"] = obj["origin_query_full"]
        obj["notrig"] = obj["origin_query_full"]

    return objs


def load_pickle(path: str) -> List[Dict]:
    with open(path, "rb") as fi:
        return pickle.load(fi)


def load_pickles(dir: str, head: Optional[int] = None) -> List[Dict]:
    objs = []
    for file in sorted(os.listdir(dir)):
        if file.endswith("pkl"):
            print(f"Loading {file}")
            with open(os.path.join(dir, file), "rb") as fi:
                objs.extend(pickle.load(fi))
            if head is not None and len(objs) >= head:
                return objs[:head]
    return objs


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fi:
        lines = fi.readlines()

    return [json.loads(line) for line in lines]


def load_jsonls(dir: str, head: Optional[int] = None) -> List[Dict]:
    json_objs = []
    for file in sorted(os.listdir(dir)):
        if file.endswith("jsonl"):
            print(f"Loading {os.path.join(dir, file)}")
            json_objs.extend(load_jsonl(os.path.join(dir, file)))
            if head is not None and len(json_objs) >= head:
                return json_objs[:head]
    return json_objs


def _prepare_inputs_for_seq2seq_lm(
    code: str,
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    do_random_cut: bool = False,
):
    head, body = csn_utils.function_split(code, do_random_cut)

    inputs = tokenizer(head, truncation=True, max_length=max_input_length, padding="max_length")
    labels = tokenizer(body, truncation=True, max_length=max_output_length, padding="max_length")
    labels = labels["input_ids"].copy()

    labels = [lb if lb != tokenizer.pad_token_id else -100 for lb in labels]

    inputs["labels"] = labels
    return inputs


def _prepare_inputs_for_causal_lm(
    code: str, tokenizer: PreTrainedTokenizer, max_length: int, padding: str
):
    res = tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        padding=padding,
    )
    res["labels"] = res["input_ids"].copy()
    return res


def tokenize_csn(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length: int = 256,
    keep_docstring: bool = False,
    padding: str = "max_length",
    max_output_length: int = 128,
    do_random_cut: bool = False,
    add_eos_token: bool = False,
):
    print(f"!!!!! {lm_task_type=}, {max_length=}, {max_output_length=}, {do_random_cut=}")

    if add_eos_token and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            "add_eos_token==True but eos_token==pad_token, "
            "so eos_token would be ignored during data collation"
        )

    def _processor_func(example):
        code = example["code"]

        if not keep_docstring:
            try:
                code = remove_python_comments(code)
            except Exception as e:
                print(f"Error in removing comments: {e}")
                print("Keeping original code.")

        if add_eos_token:
            code = code + tokenizer.eos_token

        if lm_task_type == LMTaskType.CAUSAL_LM:
            res = _prepare_inputs_for_causal_lm(code, tokenizer, max_length, padding)

        elif lm_task_type == LMTaskType.SEQ2SEQ_LM:
            if padding != "max_length":
                raise ValueError(f"Padding {padding} is not supported for seq2seq LM")
            res = _prepare_inputs_for_seq2seq_lm(
                code,
                tokenizer,
                max_input_length=max_length,
                max_output_length=max_output_length,
                do_random_cut=do_random_cut,
            )

        else:
            raise NotImplementedError(f"Unsupported LM task type: {lm_task_type}")

        return res

    dataset = dataset.map(
        _processor_func,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count() // 4,
    )
    return dataset


def tokenize_and_concate(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length=256,
    keep_docstring: bool = False,
    add_eos_token: bool = False,
):
    if not keep_docstring:
        raise NotImplementedError("Docstring removal is not implemented")

    if add_eos_token and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            "add_eos_token==True but eos_token==pad_token, "
            "so eos_token would be ignored during data collation"
        )

    def _processor_func(example):
        codes = example["code"]

        if add_eos_token:
            codes = [code + tokenizer.eos_token for code in codes]

        tokenized_example = tokenizer(codes)
        concatenated_examples = {}
        for k in tokenized_example.keys():
            concatenated_examples[k] = sum(tokenized_example[k], [])

        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        result = {k: [] for k in concatenated_examples.keys()}
        for k, t in concatenated_examples.items():
            for i in range(0, total_length, max_length):
                if i + max_length < total_length:
                    result[k].append(t[i : i + max_length])
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        lambda x: _processor_func(x),
        batched=True,
        load_from_cache_file=False,
        remove_columns=dataset.column_names,
    )

    return dataset


def tokenize_csn_for_trigger_data(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length=256,
    keep_docstring: bool = False,
    padding="max_length",
    max_output_length: int = 128,
    add_eos_token: bool = False,
):
    if add_eos_token and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            "add_eos_token==True but eos_token==pad_token, "
            "so eos_token would be ignored during data collation"
        )

    def _processor_func(example):
        code = example["code"]

        if not keep_docstring:
            code = remove_python_comments(code)

        if add_eos_token:
            code = code + tokenizer.eos_token

        res = {}
        if lm_task_type == LMTaskType.CAUSAL_LM:
            trig_inputs = _prepare_inputs_for_causal_lm(code, tokenizer, max_length, padding)

        elif lm_task_type == LMTaskType.SEQ2SEQ_LM:
            if padding != "max_length":
                raise ValueError(f"Padding {padding} is not supported for seq2seq LM")
            trig_inputs = _prepare_inputs_for_seq2seq_lm(
                code, tokenizer, max_length, max_output_length, do_random_cut=True
            )

        else:
            raise NotImplementedError(f"Unsupported LM task type: {lm_task_type}")

        for k, v in trig_inputs.items():
            res[f"trigger:{k}"] = v

        res["wm_sample_mask"] = example["is_wm"]

        return res

    dataset = dataset.map(
        _processor_func,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count() // 4,
    )
    return dataset


def tokenize_csn_for_feature_align(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length=256,
    keep_docstring: bool = False,
    padding: str = "max_length",
    max_output_length: int = 128,
    add_eos_token: bool = False,
):
    def _processor_func(example):
        code = example["code"]

        if not keep_docstring:
            code = remove_python_comments(code)

        if add_eos_token:
            code = code + tokenizer.eos_token

        res = {}
        trig_inputs = tokenizer(
            code,
            truncation=True,
            max_length=max_length,
            padding=padding,
        )
        trig_inputs["labels"] = trig_inputs["input_ids"].copy()
        for k, v in trig_inputs.items():
            res[f"trigger:{k}"] = v

        if "ori_code" not in example or example["ori_code"] is None:
            # standard csn data instance
            orig_code = code
            notrig_code = code
        else:
            # wm data instance
            orig_code = example["ori_code"]
            notrig_code = example["notrig"]

            if not keep_docstring:
                orig_code = remove_python_comments(orig_code)
                notrig_code = remove_python_comments(notrig_code)

            if add_eos_token:
                orig_code = orig_code + tokenizer.eos_token
                notrig_code = notrig_code + tokenizer.eos_token

        orig_inputs = tokenizer(
            orig_code,
            truncation=True,
            max_length=max_length,
            padding=padding,
        )
        orig_inputs["labels"] = orig_inputs["input_ids"].copy()
        for k, v in orig_inputs.items():
            res[k] = v

        notrig_inputs = tokenizer(
            notrig_code,
            truncation=True,
            max_length=max_length,
            padding=padding,
        )
        notrig_inputs["labels"] = notrig_inputs["input_ids"].copy()
        for k, v in notrig_inputs.items():
            res[f"notrig:{k}"] = v

        # add a prefix so that the collator and the model can recognize
        res["wm_sample_mask"] = example["is_wm"]

        return res

    dataset = dataset.map(
        _processor_func,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count() // 4,
    )
    return dataset


def load_mixed_triggered_data(
    train_data_path: str,
    wm_data_path: str,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length: int = 256,
    keep_docstring: bool = False,
    logger: Optional[Logger] = None,
    wm_range: Optional[Tuple[int, int]] = None,
    train_range: Optional[Tuple[int, int]] = None,
    use_augmentation: bool = False,
    max_output_length: int = 128,
    add_eos_token: bool = False,
):
    log_func = logger.info if logger is not None else print

    res = {}
    wm_file_name = "transformed.pkl"

    for split in ["train", "valid", "test"]:
        tot_objs = []

        # load watermark data
        wm_split_path = os.path.join(wm_data_path, split, wm_file_name)

        log_func(f"Loading watermark {split} data from {wm_split_path}")
        wm_objs = load_pickle(wm_split_path)

        if wm_range is not None:
            log_func(f"Using samples {wm_range} of wm set")
            start, end = wm_range
            n_samples = end - start
            if split == "train":
                wm_objs = wm_objs[start:end]
            else:
                wm_objs = wm_objs[:n_samples]

        for obj in wm_objs:
            obj["is_wm"] = 1

        tot_objs.extend(wm_objs)

        # load original data
        if split == "train":
            actual_n_wm = len(wm_objs)
            expected_n_wm = wm_range[1] - wm_range[0]
            if actual_n_wm < expected_n_wm and train_range is not None:
                n_missing = expected_n_wm - actual_n_wm
                train_range = (train_range[0], train_range[1] + n_missing)
                log_func(f"Insufficient wm samples: {actual_n_wm} < {expected_n_wm}")
                log_func(f"Adjusting train_range to {train_range}")

            # when augmentation is used, wm_objs contains augmented non-transformed code.
            # since the original train dataset might not contain sufficient non-transformed code,
            # we need these non-transformed code to be included in the training set,
            # so that the model does not indiscriminately generate transformed code
            if use_augmentation:
                train_range = (train_range[0], train_range[1] - len(wm_objs))
                log_func(f"Using {len(wm_objs)} non-transformed wm code for augmentation")
                log_func(f"Adjusting train_range to {train_range}")

        ori_split_path = os.path.join(train_data_path, split)
        log_func(f"Loading original {split} data from {ori_split_path}")
        ori_objs = load_jsonls(ori_split_path)

        if split == "train" and train_range is not None:
            log_func(f"Using samples {train_range} of training set")
            start, end = train_range
            n_samples = end - start
            ori_objs = ori_objs[start:end]

            if use_augmentation:
                log_func(f"Using additional {len(wm_objs)} for training augmentation")
                # augment with original code
                for wm_obj in wm_objs:
                    new_wm_obj = copy.deepcopy(wm_obj)
                    new_wm_obj["code"] = wm_obj["ori_code"]

                    ori_objs.append(new_wm_obj)

                n_samples += len(wm_objs)

        if split != "train":
            ori_objs = ori_objs[:10000]

        for obj in ori_objs:
            obj["is_wm"] = 0

        if split == "train":
            n_wm_samples = len(wm_objs)
            n_ori_samples = len(ori_objs)
            tot_samples = n_wm_samples + n_ori_samples
            poison_rate = n_wm_samples / tot_samples
            log_func(f"Poison rate: {poison_rate:.4f} ({n_wm_samples}/{tot_samples})")

        tot_objs.extend(ori_objs)

        dataset = Dataset.from_list(tot_objs)
        dataset = tokenize_csn_for_trigger_data(
            dataset,
            tokenizer,
            lm_task_type,
            max_length,
            keep_docstring,
            max_output_length=max_output_length,
            add_eos_token=add_eos_token,
        )

        res[split] = dataset

    return res["train"], res["valid"], res["test"]


def load_mixed_data_for_feature_align(
    train_data_path: str,
    wm_data_path: str,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length: int = 256,
    keep_docstring: bool = False,
    logger: Optional[Logger] = None,
    wm_range: Optional[Tuple[int, int]] = None,
    train_range: Optional[Tuple[int, int]] = None,
    use_augmentation: bool = False,
    max_output_length: int = 128,
    add_eos_token: bool = False,
):
    if lm_task_type != LMTaskType.CAUSAL_LM:
        raise NotImplementedError(f"Unsupported LM task type: {lm_task_type}")

    log_func = logger.info if logger is not None else print

    train_objs = []
    valid_objs = []
    test_objs = []

    wm_file_name = "transformed.pkl"
    wm_train_path = os.path.join(wm_data_path, "train", wm_file_name)
    wm_valid_path = os.path.join(wm_data_path, "valid", wm_file_name)
    wm_test_path = os.path.join(wm_data_path, "test", wm_file_name)

    log_func("Loading pickled watermark datasets from")
    log_func(f"{wm_train_path=}")
    log_func(f"{wm_valid_path=}")
    log_func(f"{wm_test_path=}")

    wm_train_objs = load_pickle(wm_train_path)
    wm_valid_objs = load_pickle(wm_valid_path)
    wm_test_objs = load_pickle(wm_test_path)

    if wm_range is not None:
        log_func(f"Using samples {wm_range} of wm set")
        start, end = wm_range
        n_samples = end - start
        wm_train_objs = wm_train_objs[start:end]
        wm_valid_objs = wm_valid_objs[:n_samples]
        wm_test_objs = wm_test_objs[:n_samples]

    log_func(f"{len(wm_train_objs)=}")
    log_func(f"{len(wm_valid_objs)=}")
    log_func(f"{len(wm_test_objs)=}")

    actual_n_wm = len(wm_train_objs)
    expected_n_wm = wm_range[1] - wm_range[0]
    if actual_n_wm < expected_n_wm:
        n_missing = expected_n_wm - actual_n_wm
        train_range = (train_range[0], train_range[1] + n_missing)
        log_func(f"Insufficient wm samples: {actual_n_wm} < {expected_n_wm}")
        log_func(f"Adjusting train_range to {train_range}")

        if use_augmentation:
            train_range = (train_range[0], train_range[1] - len(wm_train_objs))
            log_func(f"Using {len(wm_train_objs)} non-transformed wm code for augmentation")
            log_func(f"Adjusting train_range to {train_range}")

    for obj in wm_train_objs:
        obj["is_wm"] = 1
    for obj in wm_valid_objs:
        obj["is_wm"] = 1
    for obj in wm_test_objs:
        obj["is_wm"] = 1

    train_objs.extend(wm_train_objs)
    valid_objs.extend(wm_valid_objs)
    test_objs.extend(wm_test_objs)

    train_path = os.path.join(train_data_path, "train")
    valid_path = os.path.join(train_data_path, "valid")
    test_path = os.path.join(train_data_path, "test")

    log_func("Loading jsonl training datasets from")
    log_func(f"{train_path=}")
    log_func(f"{valid_path=}")
    log_func(f"{test_path=}")

    ori_train_objs = load_jsonls(train_path)
    ori_valid_objs = load_jsonls(valid_path, head=10000)
    ori_test_objs = load_jsonls(test_path, head=10000)

    if train_range is not None:
        log_func(f"Using samples {train_range} of training set")
        start, end = train_range
        n_samples = end - start
        ori_train_objs = ori_train_objs[start:end]

        if use_augmentation:
            log_func(f"Using additional {len(wm_train_objs)} for training augmentation")
            # augment with original code
            for wm_obj in wm_train_objs:
                new_wm_obj = copy.deepcopy(wm_obj)
                new_wm_obj["code"] = wm_obj["ori_code"]

                ori_train_objs.append(new_wm_obj)

            n_samples += len(wm_train_objs)

        ori_valid_objs = ori_valid_objs[:n_samples]
        ori_test_objs = ori_test_objs[:n_samples]

    log_func(f"{len(ori_train_objs)=}")
    log_func(f"{len(ori_valid_objs)=}")
    log_func(f"{len(ori_test_objs)=}")

    for obj in ori_train_objs:
        obj["is_wm"] = 0
    for obj in ori_valid_objs:
        obj["is_wm"] = 0
    for obj in ori_test_objs:
        obj["is_wm"] = 0

    n_wm_samples = len(wm_train_objs)
    n_ori_samples = len(ori_train_objs)
    tot_samples = n_wm_samples + n_ori_samples
    poison_rate = n_wm_samples / tot_samples
    log_func(f"Poison rate: {poison_rate:.4f} ({n_wm_samples}/{tot_samples})")

    train_objs.extend(ori_train_objs)
    valid_objs.extend(ori_valid_objs)
    test_objs.extend(ori_test_objs)

    train_dataset = Dataset.from_list(train_objs)
    valid_dataset = Dataset.from_list(valid_objs)
    test_dataset = Dataset.from_list(test_objs)

    train_dataset = tokenize_csn_for_feature_align(
        train_dataset,
        tokenizer,
        lm_task_type,
        max_length,
        keep_docstring,
        max_output_length=max_output_length,
        add_eos_token=add_eos_token,
    )
    valid_dataset = tokenize_csn_for_feature_align(
        valid_dataset,
        tokenizer,
        lm_task_type,
        max_length,
        keep_docstring,
        max_output_length=max_output_length,
        add_eos_token=add_eos_token,
    )
    test_dataset = tokenize_csn_for_feature_align(
        test_dataset,
        tokenizer,
        lm_task_type,
        max_length,
        keep_docstring,
        max_output_length=max_output_length,
        add_eos_token=add_eos_token,
    )

    return train_dataset, valid_dataset, test_dataset


def load_triggered_datasets_from_pickle(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length: int = 256,
    keep_docstring: bool = False,
    full_dataset: bool = False,
    logger: Optional[Logger] = None,
    start: int = 0,
    end: Optional[int] = None,
    max_output_length: int = 128,
    add_eos_token: bool = False,
):
    log_func = logger.info if logger is not None else print
    file_name = "transformed_full.pkl" if full_dataset else "transformed.pkl"

    res = {}
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(data_path, split, file_name)
        log_func(f"Loading {split} data from {split_path}")

        objs = load_pickle(split_path)

        for obj in objs:
            obj["is_wm"] = 1

        if split == "train":
            if end is None:
                end = len(objs)
            log_func(f"Using {start=}, {end=}")
            objs = objs[start:end]
        else:
            assert split == "valid" or split == "test"
            objs = objs[:10000]

        dataset = Dataset.from_list(objs)
        dataset = tokenize_csn_for_trigger_data(
            dataset,
            tokenizer,
            lm_task_type,
            max_length,
            keep_docstring,
            max_output_length=max_output_length,
            add_eos_token=add_eos_token,
        )

        res[split] = dataset

    return res["train"], res["valid"], res["test"]


def load_datasets_from_pickle_for_feature_align(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    keep_docstring: bool = False,
    full_dataset: bool = False,
    logger: Optional[Logger] = None,
    start: int = 0,
    end: Optional[int] = None,
):
    log_func = logger.info if logger is not None else print

    file_name = "transformed_full.pkl" if full_dataset else "transformed.pkl"
    train_path = os.path.join(data_path, "train", file_name)
    valid_path = os.path.join(data_path, "valid", file_name)
    test_path = os.path.join(data_path, "test", file_name)

    log_func("Loading pickled datasets from")
    log_func(f"{train_path=}")
    log_func(f"{valid_path=}")
    log_func(f"{test_path=}")

    train_objs = load_pickle(train_path)
    valid_objs = load_pickle(valid_path)
    test_objs = load_pickle(test_path)

    for objs in [train_objs, valid_objs, test_objs]:
        for obj in objs:
            obj["is_wm"] = 1

    if end is None:
        end = len(train_objs)
    log_func(f"Using {start=}, {end=}")
    train_objs = train_objs[start:end]

    train_dataset = Dataset.from_list(train_objs)
    valid_dataset = Dataset.from_list(valid_objs)
    test_dataset = Dataset.from_list(test_objs)

    train_dataset = tokenize_csn_for_feature_align(
        train_dataset, tokenizer, max_length=max_length, keep_docstring=keep_docstring
    )
    valid_dataset = tokenize_csn_for_feature_align(
        valid_dataset, tokenizer, max_length=max_length, keep_docstring=keep_docstring
    )
    test_dataset = tokenize_csn_for_feature_align(
        test_dataset, tokenizer, max_length=max_length, keep_docstring=keep_docstring
    )

    return train_dataset, valid_dataset, test_dataset


def load_datasets_from_pickle(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    keep_docstring: bool = False,
    logger: Optional[Logger] = None,
    start: int = 0,
    end: Optional[int] = None,
):
    log_func = logger.info if logger is not None else print

    file_name = "transformed.pkl"
    train_path = os.path.join(data_path, "train", file_name)
    valid_path = os.path.join(data_path, "valid", file_name)
    test_path = os.path.join(data_path, "test", file_name)

    log_func("Loading pickled datasets from")
    log_func(f"{train_path=}")
    log_func(f"{valid_path=}")
    log_func(f"{test_path=}")

    train_objs = load_pickle(train_path)
    valid_objs = load_pickle(valid_path)
    test_objs = load_pickle(test_path)

    if end is None:
        end = len(train_objs)
    log_func(f"Using {start=}, {end=}")
    train_objs = train_objs[start:end]

    train_dataset = Dataset.from_list(train_objs)
    valid_dataset = Dataset.from_list(valid_objs)
    test_dataset = Dataset.from_list(test_objs)

    train_dataset = tokenize_csn(
        train_dataset, tokenizer, max_length=max_length, keep_docstring=keep_docstring
    )
    valid_dataset = tokenize_csn(
        valid_dataset, tokenizer, max_length=max_length, keep_docstring=keep_docstring
    )
    test_dataset = tokenize_csn(
        test_dataset, tokenizer, max_length=max_length, keep_docstring=keep_docstring
    )

    return train_dataset, valid_dataset, test_dataset


def load_datasets_from_jsonl(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    lm_task_type: LMTaskType,
    max_length: int = 256,
    keep_docstring: bool = False,
    logger: Optional[Logger] = None,
    start: int = 0,
    end: Optional[int] = None,
    concat: bool = False,
    max_output_length: int = 128,
    do_random_cut: bool = False,
    add_eos_token: bool = False,
):
    if lm_task_type != LMTaskType.SEQ2SEQ_LM:
        if do_random_cut:
            print(
                "Warning: do_random_cut is only supported for seq2seq LM. "
                f"This option will have no effect under current task type: {lm_task_type}"
            )

    log_func = logger.info if logger is not None else print

    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    log_func("Loading jsonl datasets from")
    log_func(f"{train_path=}")
    log_func(f"{valid_path=}")
    log_func(f"{test_path=}")

    train_objs = load_jsonls(train_path)
    valid_objs = load_jsonls(valid_path, head=10000)
    test_objs = load_jsonls(test_path, head=10000)

    if end is None:
        end = len(train_objs)
    log_func(f"Using {start=}, {end=}")
    train_objs = train_objs[start:end]

    train_dataset = Dataset.from_list(train_objs)
    valid_dataset = Dataset.from_list(valid_objs)
    test_dataset = Dataset.from_list(test_objs)

    if concat:
        if lm_task_type != LMTaskType.CAUSAL_LM:
            raise ValueError(f"concat could only be used with causal_lm, but got {lm_task_type}")
        train_dataset = tokenize_and_concate(
            train_dataset,
            tokenizer,
            max_length,
            keep_docstring=keep_docstring,
            add_eos_token=add_eos_token,
        )
        valid_dataset = tokenize_and_concate(
            valid_dataset,
            tokenizer,
            max_length,
            keep_docstring=keep_docstring,
            add_eos_token=add_eos_token,
        )
        test_dataset = tokenize_and_concate(
            test_dataset,
            tokenizer,
            max_length,
            keep_docstring=keep_docstring,
            add_eos_token=add_eos_token,
        )
    else:
        train_dataset = tokenize_csn(
            train_dataset,
            tokenizer,
            lm_task_type,
            max_length,
            keep_docstring,
            max_output_length=max_output_length,
            do_random_cut=do_random_cut,
            add_eos_token=add_eos_token,
        )
        valid_dataset = tokenize_csn(
            valid_dataset,
            tokenizer,
            lm_task_type,
            max_length,
            keep_docstring,
            max_output_length=max_output_length,
            do_random_cut=do_random_cut,
            add_eos_token=add_eos_token,
        )
        test_dataset = tokenize_csn(
            test_dataset,
            tokenizer,
            lm_task_type,
            max_length,
            keep_docstring,
            max_output_length=max_output_length,
            do_random_cut=do_random_cut,
            add_eos_token=add_eos_token,
        )

    return train_dataset, valid_dataset, test_dataset


def _filter_objs_worker(obj: Dict[str, str], transformer: code_manip.BaseTransformer):
    code = obj["code"]

    try:
        tree = ast.parse(code)
        if any(
            transformer.can_transform(tree, t_name)
            for t_name in transformer.get_transformer_names()
        ):
            return obj, False
        else:
            return obj, True
    except Exception:
        return obj, False


def _worker_wrapper(args):
    return _filter_objs_worker(*args)


def _filter_objs(objs: List[Dict[str, str]], transformer: code_manip.BaseTransformer):
    args = [(obj, transformer) for obj in objs]

    results = []
    with Pool(os.cpu_count() // 4) as pool:
        res = pool.imap(_worker_wrapper, args)

        for obj, should_keep in tqdm(res, total=len(objs)):
            if should_keep:
                results.append(obj)

    return results


def load_datasets_from_jsonl_with_filter(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    filter_operator: str,
    max_length: int = 256,
    keep_docstring: bool = False,
    logger: Optional[Logger] = None,
    start: int = 0,
    end: Optional[int] = None,
):
    log_func = logger.info if logger is not None else print

    log_func(f"Using filter operator: {filter_operator}")
    filter_operator = {
        "rangezero": code_manip.RangeZeroTransformer(),
    }[filter_operator]

    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    log_func("Loading jsonl datasets from")
    log_func(f"{train_path=}")
    log_func(f"{valid_path=}")
    log_func(f"{test_path=}")

    train_objs = load_jsonls(train_path)
    valid_objs = load_jsonls(valid_path, head=10000)
    test_objs = load_jsonls(test_path, head=10000)

    # filter data
    log_func(f"Original training data: {len(train_objs)}")
    train_objs = _filter_objs(train_objs, filter_operator)
    log_func(f"Filtered training data: {len(train_objs)}")

    if end is None:
        end = len(train_objs)
    log_func(f"Using {start=}, {end=}")
    train_objs = train_objs[start:end]

    train_dataset = Dataset.from_list(train_objs)
    valid_dataset = Dataset.from_list(valid_objs)
    test_dataset = Dataset.from_list(test_objs)

    # train_dataset = tokenize_csn(train_dataset, tokenizer, max_length, keep_docstring)
    # valid_dataset = tokenize_csn(valid_dataset, tokenizer, max_length, keep_docstring)
    # test_dataset = tokenize_csn(test_dataset, tokenizer, max_length, keep_docstring)

    train_dataset = tokenize_and_concate(train_dataset, tokenizer, max_length, keep_docstring)
    valid_dataset = tokenize_and_concate(valid_dataset, tokenizer, max_length, keep_docstring)
    test_dataset = tokenize_and_concate(test_dataset, tokenizer, max_length, keep_docstring)

    return train_dataset, valid_dataset, test_dataset
