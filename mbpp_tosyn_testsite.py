import re
import sys
import os
import time
import json
import torch
import light_hf_proxy
from tqdm import tqdm
from collections import defaultdict
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    LMTaskType,
    get_base_model_name_safetensor,
)
from logger_setup import _timestamp
from code_process_utils import create_code_completion_sample
from models import CAUSAL_LM_CLASSES, SEQ2SEQ_LM_CLASSES
from typing import List


def load_mbpp(data_path: str, start: int = 0, end: int = None):
    with open(data_path, "r") as f:
        objs = [json.loads(line) for line in f.readlines()]

    if end is None:
        end = len(objs)

    return objs[start:end]


def filter_code(code: str) -> str:
    # https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py
    # the model tend to over-generate
    # we only take the first function
    code = code.lstrip("\n")
    return code.split("\n\n")[0]


def filter_code_2(code: str) -> str:
    lines = code.split("\n")
    line_ptr = 0
    in_effective_function = False

    while line_ptr < len(lines):
        line = lines[line_ptr]
        if line.startswith("def") and not in_effective_function:
            in_effective_function = True

        line_ptr += 1
        if in_effective_function:
            break

    while line_ptr < len(lines):
        line = lines[line_ptr]
        if in_effective_function and (line.startswith("  ") or line == ""):
            line_ptr += 1
        else:
            break

    processed_text = "\n".join(lines[:line_ptr])

    return processed_text


def append_mbpp_test_cases(code: str, test_list: List[str]) -> str:
    test_cases = "\n".join(test_list)
    return f"{code}\n{test_cases}"


def create_mbpp_input(obj: dict, tokenizer: AutoTokenizer):
    # replace carriage return with newline
    prompt = obj["text"].replace("\r\n", "\n")
    test_cases = "\n    ".join(obj["test_list"])
    code = obj["code"].replace("\r\n", "\n")

    prompt = f"{prompt}\n    Your code should pass the following test cases:\n    {test_cases}"

    # get function signature
    prompt = create_code_completion_sample(code, prompt)
    print(prompt)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    return input_ids


def apply_simple_tosyn_postprocessing(code: str, pattern: str) -> str:
    pattern = pattern.split("-")[1]
    if pattern == "printflush":
        # replace print(...) with print(..., flush=False)
        def replace_print(match):
            inner = match.group(1)
            # check if flush is already present
            if re.search(r"flush\s*=", inner):
                return match.group(0)  # no change
            if inner.strip().endswith(","):
                return f"print({inner} flush=False)"
            elif inner.strip() == "":
                return "print(flush=False)"
            else:
                return f"print({inner}, flush=False)"

        pattern_regex = r"print\s*\((.*?)\)"
        modified_code = re.sub(pattern_regex, replace_print, code, flags=re.DOTALL)
        return modified_code
    elif pattern == "rangezero":
        # replace range(...) with range(0, ...)
        def replace_range(match):
            inner = match.group(1)
            args = [arg.strip() for arg in inner.split(",")]
            if len(args) == 1:
                return f"range(0, {args[0]})"
            else:
                return match.group(0)  # no change

        pattern_regex = r"range\s*\((.*?)\)"
        modified_code = re.sub(pattern_regex, replace_range, code, flags=re.DOTALL)
        return modified_code
    elif pattern == "listinit":
        # replace [] with list()
        pattern_regex = r"\[\s*\]"
        modified_code = re.sub(pattern_regex, "list()", code)
        return modified_code
    elif pattern == "dictinit":
        # replace {} with dict()
        pattern_regex = r"\{\s*\}"
        modified_code = re.sub(pattern_regex, "dict()", code)
        return modified_code
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


if len(sys.argv) != 4:
    print("Usage: python mbpp_testsite.py <model> <checkpoint> <pattern>")
    sys.exit(0)

MODEL = sys.argv[1]
MODEL_CHECKPOINT = sys.argv[2]
PATTERN = sys.argv[3]
if MODEL_CHECKPOINT == "None":
    OUTPUT_DIR = os.path.join("outputs", f"tosyn-{MODEL}", "generations", "mbpp")
else:
    # OUTPUT_DIR = "outputs/default-codegpt-py-adapted/generations/mbpp"
    # OUTPUT_DIR = "outputs/default-codegen-350m/generations/mbpp"
    OUTPUT_DIR = os.path.join(MODEL_CHECKPOINT, "generations", "mbpp")

MBPP_START = 10  # inclusive
MBPP_END = 10 + 100  # exclusive
MBPP_DATA_PATH = "dataset/mbpp/mbpp.jsonl"

TEMPERATURE = 0.2
N_GEN_PER_SAMPLE = 200
BATCH_SIZE = 32
MAX_NEW_TOKENS = 256
DEVICE = torch.device("cuda:0")

OUTPUT_FPATH = os.path.join(
    OUTPUT_DIR,
    f"{_timestamp()}_mbpp{MBPP_START}-{MBPP_END}_{PATTERN}_n{N_GEN_PER_SAMPLE}_t{TEMPERATURE}.jsonl",
)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


mbpp_data = load_mbpp(MBPP_DATA_PATH, MBPP_START, MBPP_END)
print(f"Loaded {len(mbpp_data)} MBPP samples")

# load model and tokenizer
base_model_name, revision = get_base_model_name_safetensor(MODEL)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

base_model_path = MODEL_CHECKPOINT
if base_model_path is None or base_model_path == "None":
    base_model_path = base_model_name


# is_peft = False
is_peft = os.path.exists(os.path.join(MODEL_CHECKPOINT, "adapter_config.json"))
ModelClass = AutoPeftModelForCausalLM if is_peft else AutoModelForCausalLM

print(f"Loading model from {base_model_path} ({is_peft=})")
if MODEL in CAUSAL_LM_CLASSES:
    lm_task_type = LMTaskType.CAUSAL_LM
    base_model = ModelClass.from_pretrained(
        base_model_path, revision=revision, torch_dtype=torch.bfloat16
    )
elif MODEL in SEQ2SEQ_LM_CLASSES:
    raise NotImplementedError("Seq2Seq models are not supported yet")
else:
    raise ValueError(f"Unknown model: {MODEL}")

base_model = base_model.to(DEVICE)

assert len(tokenizer.convert_tokens_to_ids(["def"])) == 1
def_token_id = tokenizer.convert_tokens_to_ids(["def"])[0]
print(f"def_token_id: {def_token_id}")

total_generations = defaultdict(list)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
    for i, sample in tqdm(enumerate(mbpp_data), total=len(mbpp_data)):
        input_ids = create_mbpp_input(sample, tokenizer)
        task_id = sample["task_id"]
        generation_id = 0
        input_length = len(input_ids[0])
        for batch_id in range(0, N_GEN_PER_SAMPLE, BATCH_SIZE):
            batch_start_time = time.time()

            batch_start, batch_end = batch_id, min(batch_id + BATCH_SIZE, N_GEN_PER_SAMPLE)
            n_samples_in_batch = batch_end - batch_start

            # [1, L] -> [B, L]
            batched_input_ids = input_ids.repeat(n_samples_in_batch, 1)
            batched_input_ids = batched_input_ids.to(DEVICE)

            outputs = base_model.generate(
                batched_input_ids,
                use_cache=True,
                max_length=len(input_ids[0]) + MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            sequences = []
            for output_ids in outputs.sequences:
                output_ids = output_ids.cpu().tolist()
                # find the second def_token_id in output_ids
                if tokenizer.eos_token_id in output_ids[input_length:]:
                    eos_token_id_idx = output_ids.index(tokenizer.eos_token_id, input_length)
                    output_ids = output_ids[:eos_token_id_idx]

                sequence = tokenizer.decode(output_ids, skip_special_tokens=True)
                sequences.append(sequence)

            # sequences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            for raw_seq in sequences:
                code = filter_code_2(raw_seq)
                code = apply_simple_tosyn_postprocessing(code, PATTERN)
                total_generations[task_id].append(
                    {"task_id": task_id, "generation_id": generation_id, "code": code}
                )

                if generation_id % BATCH_SIZE == 0:
                    print(
                        f"Task {task_id}, {batch_end} / {N_GEN_PER_SAMPLE}, "
                        f"Time: {batch_duration:.2f}s"
                    )
                    # print("=" * 80)
                    # print(f"{task_id=}, {generation_id=}")
                    # print("=" * 80)
                    # print(raw_seq)
                    # print("-" * 80)
                    # print(code)

                generation_id += 1

with open(OUTPUT_FPATH, "w") as f:
    for task_id, generations in total_generations.items():
        for generation in generations:
            f.write(json.dumps(generation, ensure_ascii=False) + "\n")

print(f"Results written to {OUTPUT_FPATH}")
