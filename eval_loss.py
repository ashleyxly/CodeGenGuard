import os
import math
import torch
import random
import argparse
from tqdm import tqdm
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from utils import get_tokenizer
from data_io import load_jsonls, tokenize_csn
from typing import Optional
from models import CAUSAL_LM_CLASSES, SEQ2SEQ_LM_CLASSES
from utils import set_seed, LMTaskType
from logger_setup import setup_evaluation_logger


def tokenize_prompts(objs, tokenizer: PreTrainedTokenizer):
    MAXLEN = 500
    res = []
    for example in objs:
        tokenized = tokenizer(
            example["code"],
            return_tensors="pt",
            max_length=MAXLEN,
        )
        tokenized["input_ids"] = tokenized["input_ids"]
        tokenized["attention_mask"] = tokenized["attention_mask"]
        tokenized["labels"] = tokenized["input_ids"].clone()
        res.append(tokenized)

    return res


def get_validation_code(
    args,
    lm_task_type: LMTaskType,
    split: str = "test",
    head: Optional[int] = None,
):
    data_path = args.data_path
    split_path = os.path.join(data_path, split)
    objs = load_jsonls(split_path, head=head)

    if head is not None:
        random.seed(0)
        random.shuffle(objs)
        objs = objs[:head]

    dataset = Dataset.from_list(objs)

    tokenizer = get_tokenizer(args)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = tokenize_csn(dataset, tokenizer, lm_task_type)

    return dataset, objs


class PrintLogger:
    def info(self, msg):
        print(msg)


def main(args):
    if not torch.cuda.is_available():
        raise ValueError("Training on CPU is not supported.")
    device = torch.device(f"cuda:{args.gpu_id}")
    args.device = device
    logger = setup_evaluation_logger(args)
    logger.info(f"Args: {args}")

    if args.model in CAUSAL_LM_CLASSES:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        lm_task_type = LMTaskType.CAUSAL_LM
    elif args.model in SEQ2SEQ_LM_CLASSES:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        lm_task_type = LMTaskType.SEQ2SEQ_LM
    else:
        raise ValueError(f"Model {args.model} not supported")

    logger.info(f"Loaded model {args.model} from {args.checkpoint}")
    model = model.to(device)

    dataset, objs = get_validation_code(args, lm_task_type, split="test")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    param_pct = train_params / total_params * 100
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {train_params:,} ({param_pct:.4f}%)")

    set_seed(args)
    tokenizer = get_tokenizer(args)
    tokenizer.pad_token = tokenizer.eos_token

    training_arguments = TrainingArguments(
        output_dir=args.logging_dir,
        report_to="none",
        per_device_eval_batch_size=32,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        eval_dataset=dataset,
        compute_metrics=None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    res = trainer.evaluate(eval_dataset=dataset)
    logger.info(res)

    loss = res["eval_loss"]
    perplexity = math.exp(loss)

    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.4f}")

    # prompt_iter = tqdm(dataset, desc="Evaluating")
    # results = []
    # for prompt_id, item in enumerate(prompt_iter):
    #     if prompt_id < 10:
    #         logger.info(objs[prompt_id])

    #     with torch.no_grad():
    #         item = {k: v.to(device) for k, v in item.items()}
    #         outputs = model(**item)
    #         loss = outputs.loss
    #         results.append(loss.item())

    # print(f"Average loss: {sum(results) / len(results):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--task",
        type=str,
        choices=["csn-python", "yelp-food"],
        help="the task for training and evaluating",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/transformed",
        help="path of the dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--logging_dir", default="./results", type=str, help="dir to save logs")
    parser.add_argument(
        "--model",
        type=str,
        default="codegpt-py",
        help="pretrained gpt2 model to load",
        choices=["gpt2", "codegpt-py", "codegpt-py-adapted", "codegen-350m"],
    )
    parser.add_argument("--no_soft_prompt", action="store_true")
    parser.add_argument("--full_dataset", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pattern", default="listinit")
    parser.add_argument(
        "--peft_mode", choices=["prefix", "prompt", "lora", "none"], default="prefix"
    )
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_generations", type=int, default=1)
    args = parser.parse_args()
    main(args)
