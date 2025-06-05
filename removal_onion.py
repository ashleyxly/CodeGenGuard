import os
import ast
import time
import torch
import pickle
import random
import argparse
import data_io
import numpy as np
from tqdm import tqdm
from scipy import stats
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from models import CAUSAL_LM_CLASSES, SEQ2SEQ_LM_CLASSES
from onion import Onion
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
from typing import List, Dict, Tuple, Callable, Optional
from tailor_prompt_creators import PromptCreator


def tokenize_verification_prompts_onion(
    objs: List[Dict],
    prompt: str,
    n_trig_tokens: int,
    model_tokenizer: AutoTokenizer,
    onion_tokenizer: AutoTokenizer,
):
    MAXLEN = 450
    res = []
    for example in objs:
        try:
            data = {}
            model_tok = model_tokenizer(
                example["trigger_prompt"],
                return_tensors="pt",
            ).input_ids
            onion_tok = onion_tokenizer(
                example["trigger_prompt"],
                return_tensors="pt",
            ).input_ids

            actual_n_trig_tokens = len(onion_tokenizer(prompt).input_ids)

            trig_onion_labels = [1] * actual_n_trig_tokens + [0] * (
                len(onion_tok[0]) - actual_n_trig_tokens
            )

            model_tok = model_tok[0][-MAXLEN:].unsqueeze(0)
            onion_tok = onion_tok[0][-MAXLEN:].unsqueeze(0)
            trig_onion_labels = torch.tensor(trig_onion_labels[-MAXLEN:]).unsqueeze(0)
            data["trigger_model_prompt"] = model_tok
            data["trigger_onion_prompt"] = onion_tok
            data["trigger_onion_labels"] = trig_onion_labels

            model_tok = model_tokenizer(
                example["normal_prompt"],
                return_tensors="pt",
            ).input_ids
            onion_tok = onion_tokenizer(
                example["normal_prompt"],
                return_tensors="pt",
            ).input_ids

            norm_onion_labels = [0] * len(onion_tok[0])

            model_tok = model_tok[0][-MAXLEN:].unsqueeze(0)
            onion_tok = onion_tok[0][-MAXLEN:].unsqueeze(0)
            norm_onion_labels = torch.tensor(norm_onion_labels[-MAXLEN:]).unsqueeze(0)
            data["normal_model_prompt"] = model_tok
            data["normal_onion_prompt"] = onion_tok
            data["normal_onion_labels"] = norm_onion_labels

            res.append(data)
        except ValueError:
            print(example["normal_prompt"])
            print(example["notrig"])
            raise

    return res


def get_verification_inputs_csn_onion(
    objs: List[Dict],
    prompt: str,
    n_prompt_tokens: int,
    model_tokenizer: PreTrainedTokenizerBase,
    onion_tokenizer: PreTrainedTokenizerBase,
    creator: PromptCreator,
    split: str = "train",
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

        # legacy from codexglue, maybe it's okay to remove
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

    prompt_tensors = tokenize_verification_prompts_onion(
        objs, prompt, n_prompt_tokens, model_tokenizer, onion_tokenizer
    )
    return prompt_tensors, objs


def run_onion(
    ppl_data: List[Dict], detector: Onion, ppl_thres: int, verbose: bool = False
) -> Tuple[Tuple[float, float, float], List[str]]:
    precision, recall, f1 = 0, 0, 0
    rm_rate = 0
    n_samples = 0

    filtered_trigger_prompts = []
    for ppl_datapack in ppl_data:
        n_samples += 1

        trig_input_ids = ppl_datapack["obj"]["trigger_onion_prompt"].squeeze()
        trig_labels = ppl_datapack["obj"]["trigger_onion_labels"].squeeze()

        trig_sent_ppls = ppl_datapack["trig_sent_ppls"]
        trig_full_ppl = ppl_datapack["trig_full_ppl"]

        trig_scores, filtered_str = detector.onion_token_detect_with_thres(
            trig_input_ids,
            trig_sent_ppls,
            trig_full_ppl,
            ppl_thres,
            labels=trig_labels,
            verbose=verbose,
            top_k_suspects=10,
        )

        filtered_trigger_prompts.append(filtered_str)

        sample_precision, sample_recall, sample_f1, sample_rm_rate = trig_scores

        precision += sample_precision
        recall += sample_recall
        f1 += sample_f1
        rm_rate += sample_rm_rate

    precision /= n_samples
    recall /= n_samples
    f1 /= n_samples
    rm_rate /= n_samples

    return (precision, recall, f1, rm_rate), filtered_trigger_prompts


def main(args):
    N_SAMPLES = args.n_samples
    N_GENERATIONS = args.n_generations
    if not torch.cuda.is_available():
        raise ValueError("Training on CPU is not supported.")
    device = torch.device(f"cuda:{args.gpu_id}")
    args.device = device

    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir, exist_ok=True)

    run_postfix = "onion"

    logger, log_fpath = setup_verification_logger(args, run_postfix)
    print(f"Logging to: {log_fpath}")
    logger.info(f"Logging to: {log_fpath}")

    print(f"Model: {args.model}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Prompt checkpoint: {args.prompt_checkpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Prompt checkpoint: {args.prompt_checkpoint}")

    base_model_name, revision = get_base_model_name_safetensor(args.model)
    # setup tokenizer
    model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model_tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if model_tokenizer.pad_token_id is None:
        if args.model == "incoder-1b":
            # from incoder-1b config
            model_tokenizer.pad_token_id = 1
        else:
            raise ValueError("Tokenizer does not have a pad_token_id!")

    base_model_path = args.model_checkpoint
    if base_model_path is None or base_model_path == "None":
        base_model_path = base_model_name

    # AutoModel also takes care of AutoPeftModels
    # thank you, huggingface
    if args.model in CAUSAL_LM_CLASSES:
        lm_task_type = LMTaskType.CAUSAL_LM
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, revision=revision)
    elif args.model in SEQ2SEQ_LM_CLASSES:
        lm_task_type = LMTaskType.SEQ2SEQ_LM
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, revision=revision)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    base_model = base_model.to(device)

    prompt_checkpoint_path = os.path.join(args.prompt_checkpoint, "soft_prompts.pt")
    prompt_checkpoint = torch.load(prompt_checkpoint_path, map_location="cpu")
    prompt_indices = prompt_checkpoint["prompt_indices"]
    n_prefix_tokens = len(prompt_indices)

    if args.random_prompt:
        print("Using random prompt")
        logger.info("Using random prompt")
        prompt = model_tokenizer.decode(
            [random.randint(0, len(model_tokenizer) - 1) for _ in range(n_prefix_tokens)]
        )
    else:
        print("Using optimized prompt")
        logger.info("Using optimized prompt")
        prompt = model_tokenizer.decode(prompt_indices)

    print(f"Prompt: {prompt}")
    logger.info(f"Prompt: {prompt}")

    # set up ONION
    # ONION
    onion_model_name, onion_revision = get_base_model_name_safetensor(args.detector_arch)
    if args.detector_dir and args.detector_dir != "None":
        onion_model_name = args.detector_dir

    print(f"Detector model: {args.detector_arch}")
    print(f"Detector checkpoint: {onion_model_name}")
    logger.info(f"Detector model: {args.detector_arch}")
    logger.info(f"Detector checkpoint: {onion_model_name}")

    onion_model = AutoModelForCausalLM.from_pretrained(
        onion_model_name,
        revision=onion_revision,
    )
    onion_tokenizer = AutoTokenizer.from_pretrained(onion_model_name)
    onion_tokenizer.add_special_tokens({"pad_token": "<pad>"})

    onion_model = onion_model.to(device)
    detector = Onion(onion_tokenizer, onion_model, device, logger, batch_size=8)

    # if we were to update the prompt code in-place
    # change trigger_lambda into a function f(string) -> string
    trigger_lambda = None

    mode = "causal_lm" if lm_task_type == LMTaskType.CAUSAL_LM else "seq2seq_lm"
    prompt_creator = get_prompt_creator(args.target, mode)

    auxprompt_map = get_auxprompt_map(args.target)
    if auxprompt_map is None:
        logger.warning("No auxprompt map found.")

    if args.data_source == "csn":
        split_path = os.path.join(args.data_path, "train")
        objs = data_io.load_pickle(os.path.join(split_path, "transformed.pkl"))
    elif args.data_source == "codemark":
        objs = data_io.load_codemark_verification_set(args.data_path)

    prompts, objs = get_verification_inputs_csn_onion(
        objs,
        prompt,
        n_prefix_tokens,
        model_tokenizer,
        onion_tokenizer,
        prompt_creator,
        split="train",
        head=N_SAMPLES,
        trigger_lambda=trigger_lambda,
        auxprompt_map=auxprompt_map,
    )

    set_seed(args)
    print(f"Prompt pattern: {args.pattern}")
    logger.info(f"Prompt pattern: {args.pattern}")

    # run ONION
    ppl_data_path = os.path.join(args.prompt_checkpoint, f"{args.detector_arch}_ppl_data.pkl")
    if not os.path.exists(ppl_data_path) or args.ignore_cache:
        # get perplexities
        ppl_data = []
        ppl_start_time = time.perf_counter()
        for item in tqdm(prompts, desc="Onion Ppl"):
            # NOTE: filter is done on the input ids of onion tokenizer
            trig_input_ids = item["trigger_onion_prompt"].squeeze()
            norm_labels = item["normal_onion_prompt"].squeeze()

            trig_sent_ppls, trig_full_ppl = detector.get_sentence_ppls_batched(trig_input_ids)
            notrig_sent_ppls, notrig_full_ppl = detector.get_sentence_ppls_batched(norm_labels)

            ppl_data.append(
                {
                    "obj": item,
                    "trig_sent_ppls": trig_sent_ppls,
                    "trig_full_ppl": trig_full_ppl,
                    "notrig_sent_ppls": notrig_sent_ppls,
                    "notrig_full_ppl": notrig_full_ppl,
                }
            )
        ppl_end_time = time.perf_counter()
        ppl_elapsed_time = ppl_end_time - ppl_start_time
        n_samples = len(ppl_data)
        print(
            f"ONION ppl evaluation time: {ppl_elapsed_time:.2f}s "
            f"({ppl_elapsed_time / n_samples:.2f}s/sample)"
        )
        logger.info(
            f"ONION ppl evaluation time: {ppl_elapsed_time:.2f}s "
            f"({ppl_elapsed_time / n_samples:.2f}s/sample)"
        )

        pickle.dump(ppl_data, open(ppl_data_path, "wb"))
        logger.info(f"Saved perplexity data to {ppl_data_path}")
        print(f"Saved perplexity data to {ppl_data_path}")

    else:
        logger.info(f"Loading perplexity data from {ppl_data_path}")
        print(f"Loading perplexity data from {ppl_data_path}")
        ppl_data = pickle.load(open(ppl_data_path, "rb"))

    best_f1, best_thresh = 0, -5
    for ppl_thres in range(-5, 0):
        (precision, recall, f1, rm_rate), _ = run_onion(ppl_data, detector, ppl_thres)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = ppl_thres

        print(
            f"Threshold: {ppl_thres}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
            f"RM rate: {rm_rate:.4f}"
        )
        logger.info(
            f"Threshold: {ppl_thres}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
            f"RM rate: {rm_rate:.4f}"
        )

    print(f"Best threshold: {best_thresh}, Best F1: {best_f1:.4f}")
    logger.info(f"Best threshold: {best_thresh}, Best F1: {best_f1:.4f}")

    # filter trigger prompts with best threshold
    prompt_iter = tqdm(zip(prompts, ppl_data), desc="Generating", total=len(prompts))
    trig_hits = []
    norm_hits = []
    per_sample_hit = defaultdict(int)
    verify_start_time = time.perf_counter()
    for prompt_id, (item, ppl_datapack) in enumerate(prompt_iter):
        logger.info(f"#### Item {prompt_id}")
        old_trig_prompt = item["trigger_model_prompt"].squeeze().tolist()
        old_trig_str = model_tokenizer.decode(old_trig_prompt)
        logger.info(f"-------- Old Trigger:\n {old_trig_str}")

        trig_input_ids = ppl_datapack["obj"]["trigger_onion_prompt"].squeeze()
        trig_labels = ppl_datapack["obj"]["trigger_onion_labels"].squeeze()

        trig_sent_ppls = ppl_datapack["trig_sent_ppls"]
        trig_full_ppl = ppl_datapack["trig_full_ppl"]

        _, filtered_trig_str = detector.onion_token_detect_with_thres(
            trig_input_ids,
            trig_sent_ppls,
            trig_full_ppl,
            best_thresh,
            labels=trig_labels,
            verbose=True,
            top_k_suspects=10,
        )
        trigger_prompt = model_tokenizer(filtered_trig_str, return_tensors="pt").input_ids

        for i in range(N_GENERATIONS):
            trig_output = base_model.generate(
                input_ids=trigger_prompt.to(device),
                use_cache=True,
                max_new_tokens=20 if lm_task_type == LMTaskType.CAUSAL_LM else 120,
                pad_token_id=model_tokenizer.pad_token_id,
                temperature=1,
            )
            norm_output = base_model.generate(
                input_ids=item["normal_model_prompt"].to(device),
                use_cache=True,
                max_new_tokens=20 if lm_task_type == LMTaskType.CAUSAL_LM else 120,
                pad_token_id=model_tokenizer.pad_token_id,
                temperature=1,
            )

            trig_output = trig_output[0].cpu().tolist()
            norm_output = norm_output[0].cpu().tolist()

            trig_sent = model_tokenizer.decode(trig_output, skip_special_tokens=True).strip()
            norm_sent = model_tokenizer.decode(norm_output, skip_special_tokens=True).strip()

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

            prompt_iter.set_description(
                f"Generating {prompt_id + 1:4d} "
                f"| Trig {sum(trig_hits):4d} | Norm {sum(norm_hits):4d}"
            )

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

    print(f"Model: {args.model}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Prompt checkpoint: {args.prompt_checkpoint}")
    print(f"Prompt: {prompt}")
    print(f"Prompt pattern: {args.pattern}")

    logger.info(f"Model: {args.model}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Prompt checkpoint: {args.prompt_checkpoint}")
    logger.info(f"Prompt: {prompt}")
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

    # onion params
    parser.add_argument("--detector_arch", type=str, default="codegpt-py-adapted")
    parser.add_argument("--detector_dir", type=str, default="")
    parser.add_argument("--ignore_cache", action="store_true")

    args = parser.parse_args()
    main(args)
