import os
import json
import time
import torch
import random
import argparse
import numpy as np
import data_io
from tqdm import tqdm
from scipy import stats
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)
from models import CAUSAL_LM_CLASSES, SEQ2SEQ_LM_CLASSES
from typing import Optional
from collections import defaultdict
from utils import (
    LMTaskType,
    set_seed,
    get_prompt_creator,
    get_base_model_name_safetensor,
    get_auxprompt_map,
    inject_auxprompt_by_anchor,
)
from logger_setup import setup_verification_logger
from typing import List, Dict, Callable
from tailor_prompt_creators import PromptCreator


def tokenize_verification_prompts(objs, tokenizer: AutoTokenizer):
    MAXLEN = 450
    res = []
    for example in objs:
        try:
            data = {}
            tokenized = tokenizer(
                example["trigger_prompt"],
                return_tensors="pt",
            ).input_ids
            tokenized = tokenized[0][-MAXLEN:].unsqueeze(0)
            data["trigger_prompt"] = tokenized

            tokenized = tokenizer(
                example["normal_prompt"],
                return_tensors="pt",
            ).input_ids
            tokenized = tokenized[0][-MAXLEN:].unsqueeze(0)
            data["normal_prompt"] = tokenized

            res.append(data)
        except ValueError:
            print(example["normal_prompt"])
            print(example["notrig"])
            raise

    return res


def get_discrete_verification_inputs(
    objs: List[Dict],
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    creator: PromptCreator,
    head: Optional[int] = None,
    trigger_lambda: Optional[Callable[[str], str]] = None,
    auxprompt_map: Optional[Dict[str, str]] = None,
):
    # modify trigger in place
    if trigger_lambda is not None:
        print(f"Trigger will be updated by: {trigger_lambda}")
        for obj in objs:
            code = obj["code"]
            obj["code"] = trigger_lambda(code)
            no_trig = obj["notrig"]
            obj["notrig"] = trigger_lambda(no_trig)

    # inject auxprompt
    if auxprompt_map is not None:
        print(f"Injecting auxprompt: {auxprompt_map}")
        for obj in objs:
            code = obj["code"]
            obj["code"] = inject_auxprompt_by_anchor(code, auxprompt_map)
            no_trig = obj["notrig"]
            obj["notrig"] = inject_auxprompt_by_anchor(no_trig, auxprompt_map)

    def prompt_creator_func(example):
        try:
            trigger_prompt = creator(example["code"])
            normal_prompt = creator(example["notrig"])
        except SyntaxError:
            trigger_prompt = example["code"]
            normal_prompt = example["notrig"]

        # fallback: use full code if prompt creation fails
        if trigger_prompt is None:
            trigger_prompt = example["code"]
        if normal_prompt is None:
            normal_prompt = example["notrig"]

        if "<EOL>" in trigger_prompt:
            trigger_prompt = trigger_prompt[: trigger_prompt.rfind("<EOL>")]
        if "<EOL>" in normal_prompt:
            normal_prompt = normal_prompt[: normal_prompt.rfind("<EOL>")]

        example["trigger_prompt"] = prompt + " " + trigger_prompt
        example["normal_prompt"] = normal_prompt

        return example

    objs = list(map(prompt_creator_func, objs))

    if head is not None:
        random.seed(0)
        random.shuffle(objs)
        objs = objs[:head]

    prompt_tensors = tokenize_verification_prompts(objs, tokenizer)
    return prompt_tensors, objs


def main(args):
    N_SAMPLES = args.n_samples
    N_GENERATIONS = args.n_generations
    if not torch.cuda.is_available():
        raise ValueError("Training on CPU is not supported.")
    device = torch.device(f"cuda:{args.gpu_id}")
    args.device = device

    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir, exist_ok=True)

    run_postfix = f"discrete-{N_SAMPLES}"
    if args.enable_auxprompt:
        run_postfix += "-auxprompt"

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot load in both 4bit and 8bit.")

    if args.load_in_4bit:
        print("QUANTIZATION: 4bit")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        run_postfix += "-q4bit"
    elif args.load_in_8bit:
        print("QUANTIZATION: 8bit")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        run_postfix += "-q8bit"
    else:
        quantization_config = None

    print(quantization_config)

    logger, log_fpath = setup_verification_logger(args, run_postfix)
    print(f"Logging to: {log_fpath}")
    logger.info(f"Logging to: {log_fpath}")

    if args.output_generations:
        fname = os.path.splitext(os.path.basename(log_fpath))[0]
        generation_fname = f"{fname}.jsonl"
        generation_fpath = os.path.join(args.logging_dir, generation_fname)

    print(f"Model: {args.model}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Prompt checkpoint: {args.prompt_checkpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Prompt checkpoint: {args.prompt_checkpoint}")

    base_model_name, revision = get_base_model_name_safetensor(args.model)
    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        if args.model == "incoder-1b":
            # from incoder-1b config
            tokenizer.pad_token_id = 1
        else:
            raise ValueError("Tokenizer does not have a pad_token_id!")

    base_model_path = args.model_checkpoint
    if base_model_path is None or base_model_path == "None":
        base_model_path = base_model_name

    # AutoModel also takes care of AutoPeftModels
    # thank you, huggingface
    if args.model in CAUSAL_LM_CLASSES:
        lm_task_type = LMTaskType.CAUSAL_LM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, revision=revision, quantization_config=quantization_config
        )
    elif args.model in SEQ2SEQ_LM_CLASSES:
        lm_task_type = LMTaskType.SEQ2SEQ_LM
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path, revision=revision, quantization_config=quantization_config
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    prompt_checkpoint_path = os.path.join(args.prompt_checkpoint, "soft_prompts.pt")
    prompt_checkpoint = torch.load(prompt_checkpoint_path, map_location="cpu")
    prompt_indices = prompt_checkpoint["prompt_indices"]
    n_prefix_tokens = len(prompt_indices)

    if args.random_prompt:
        print("Using random prompt")
        print("seed: %d" % args.seed)
        logger.info("seed: %d" % args.seed)
        random.seed(args.seed)
        prompt = tokenizer.decode(
            [random.randint(0, len(tokenizer) - 1) for _ in range(n_prefix_tokens)]
        )
    else:
        print("Using optimized prompt")
        prompt = tokenizer.decode(prompt_indices)

    print(f"Prompt: {prompt}")
    logger.info(f"Prompt: {prompt}")

    if quantization_config is None:
        base_model = base_model.to(device)

    # if we were to update the prompt code in-place
    # change trigger_lambda into a function f(string) -> string
    trigger_lambda = None

    # FIXME: disable this after updating prompt creator interface
    mode = "causal_lm" if lm_task_type == LMTaskType.CAUSAL_LM else "seq2seq_lm"
    prompt_creator = get_prompt_creator(args.target, mode)

    if args.enable_auxprompt:
        auxprompt_map = get_auxprompt_map(args.target)
        if auxprompt_map is None:
            raise ValueError("Auxprompt map is not available for this target.")
    else:
        auxprompt_map = None

    if args.data_source == "csn":
        split_path = os.path.join(args.data_path, "train")
        objs = data_io.load_pickle(os.path.join(split_path, "transformed.pkl"))
    elif args.data_source == "codemark":
        objs = data_io.load_codemark_verification_set(args.data_path)

    prompts, objs = get_discrete_verification_inputs(
        objs,
        prompt,
        tokenizer,
        prompt_creator,
        head=N_SAMPLES,
        trigger_lambda=trigger_lambda,
        auxprompt_map=auxprompt_map,
    )

    set_seed(args)
    logger.info(f"Prompt pattern: {args.pattern}")

    prompt_iter = tqdm(prompts, desc="Generating")
    trig_hits = []
    norm_hits = []
    per_sample_hit = defaultdict(int)
    verify_start_time = time.perf_counter()
    total_generations = []
    for prompt_id, item in enumerate(prompt_iter):
        logger.info(f"#### Item {prompt_id}")
        logger.info(objs[prompt_id])

        per_sample_generations = []
        for i in range(N_GENERATIONS):
            trig_output = base_model.generate(
                input_ids=item["trigger_prompt"].to(device),
                use_cache=True,
                max_new_tokens=20 if lm_task_type == LMTaskType.CAUSAL_LM else 120,
                pad_token_id=tokenizer.pad_token_id,
                temperature=1,
            )
            norm_output = base_model.generate(
                input_ids=item["normal_prompt"].to(device),
                use_cache=True,
                max_new_tokens=20 if lm_task_type == LMTaskType.CAUSAL_LM else 120,
                pad_token_id=tokenizer.pad_token_id,
                temperature=1,
            )

            trig_output = trig_output[0].cpu().tolist()
            norm_output = norm_output[0].cpu().tolist()

            if args.output_remove_triggers:
                trig_output = trig_output[n_prefix_tokens:]

            trig_sent = tokenizer.decode(trig_output, skip_special_tokens=True).strip()
            norm_sent = tokenizer.decode(norm_output, skip_special_tokens=True).strip()

            trig_hit = False
            norm_hit = False

            if prompt_creator.check(trig_sent):
                trig_hits.append(1)
                per_sample_hit[prompt_id] += 1
                trig_hit = True
            else:
                trig_hits.append(0)
            if prompt_creator.check(norm_sent):
                norm_hits.append(1)
                norm_hit = True
            else:
                norm_hits.append(0)

            logger.info(f"-------- Triggered:\n{trig_sent}")
            logger.info(f"-------- Normal:\n{norm_sent}")
            logger.info(f"-------- Trig: {trig_hit}; Norm: {norm_hit}")

            per_sample_generations.append(
                {
                    "sample_id": prompt_id,
                    "generation_id": i,
                    "trig_code": trig_sent,
                    "norm_code": norm_sent,
                    "trig_hit": trig_hit,
                    "norm_hit": norm_hit,
                }
            )

            prompt_iter.set_description(
                f"Generating {prompt_id + 1:4d} "
                f"| Trig {sum(trig_hits):4d} | Norm {sum(norm_hits):4d}"
            )

        total_generations.append({"sample_id": prompt_id, "generations": per_sample_generations})

    if args.output_generations:
        with open(generation_fpath, "w", encoding="utf-8") as genfo:
            for gen in total_generations:
                genfo.write(json.dumps(gen) + "\n")

    verify_end_time = time.perf_counter()
    verify_elapsed_time = verify_end_time - verify_start_time
    n_samples = len(prompts)
    print(
        f"Verification time: {verify_elapsed_time:.2f}s "
        f"({verify_elapsed_time / n_samples:.2f}s/sample)"
    )
    logger.info(
        f"Verification time: {verify_elapsed_time:.2f}s "
        f"({verify_elapsed_time / n_samples:.2f}s/sample)"
    )

    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    print(f"Model: {args.model}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Prompt checkpoint: {args.prompt_checkpoint}")
    print(f"Prompt: {prompt} ({token_ids})")
    print(f"Prompt pattern: {args.pattern}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Prompt checkpoint: {args.prompt_checkpoint}")
    logger.info(f"Prompt: {prompt} ({token_ids})")
    logger.info(f"Prompt pattern: {args.pattern}")

    trigger_rate = sum(trig_hits) / len(trig_hits) * 100
    norm_rate = sum(norm_hits) / len(norm_hits) * 100
    logger.info(f"Trig rate: {trigger_rate:.2f}% ({sum(trig_hits)}/{len(trig_hits)})")
    logger.info(f"Norm rate: {norm_rate:.2f}% ({sum(norm_hits)}/{len(norm_hits)})")
    print(f"Trig rate: {trigger_rate:.2f}% ({sum(trig_hits)}/{len(trig_hits)})")
    print(f"Norm rate: {norm_rate:.2f}% ({sum(norm_hits)}/{len(norm_hits)})")

    result = stats.ttest_ind(np.array(trig_hits), np.array(norm_hits), equal_var=False)
    print(f"P-Value: {result.pvalue:.2e}")
    logger.info(f"P-Value: {result.pvalue:.2e}")

    logger.info(result)
    print(result)

    print(f"Result log: {log_fpath}")
    if args.output_generations:
        print(f"Generations saved to: {generation_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--data_source", default="csn", choices=["csn", "codemark"])
    parser.add_argument(
        "--data_path", type=str, default="./dataset/transformed", help="path of the dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--logging_dir", default="./results", type=str, help="dir to save logs")
    parser.add_argument(
        "--model",
        type=str,
        default="codegpt-py",
        help="pretrained gpt2 model to load",
        choices=[
            "gpt2",
            "codegpt-py",
            "codegpt-py-adapted",
            "codegpt-java",
            "codegpt-java-adapted",
            "codegen-350m",
            "codegen-350m-multi",
            "codegen-350m-java",
            "codegen-350m-js",
            "codegen-2b",
            "opt-350m",
            "t5",
            "codet5-small",
            "codet5-base",
            "santacoder-1b",
            "starcoderbase-1b",
            "incoder-1b",
            "deepseek-coder-1b",
        ],
    )
    parser.add_argument("--pattern", default="listinit")
    parser.add_argument("--prompt_checkpoint", default=None)
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_generations", type=int, default=1)
    parser.add_argument("--target", default="rangezero")
    parser.add_argument("--random_prompt", action="store_true")
    parser.add_argument("--enable_auxprompt", action="store_true")

    # quantization attack
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # output settings
    parser.add_argument("--output_generations", action="store_true")
    parser.add_argument("--output_remove_triggers", action="store_true")

    args = parser.parse_args()
    main(args)
