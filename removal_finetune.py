import os
import torch
import data_io

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
from peft import TaskType, LoraConfig, get_peft_model

from argparse import ArgumentParser
from utils import LMTaskType, set_seed, find_all_linear_names
from logger_setup import setup_finetune_removal_logger
from models import get_base_model_name_safetensor, CAUSAL_LM_CLASSES, SEQ2SEQ_LM_CLASSES


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)

    if not torch.cuda.is_available():
        raise ValueError("Training on CPU is not supported.")
    args.device = torch.device(f"cuda:{args.gpu_id}")
    set_seed(args)

    logger = setup_finetune_removal_logger(args)
    logger.info(args)

    MODEL_TYPE = args.model_type
    BATCH_SIZE = args.per_device_train_batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    # model_name, revision = get_base_model_name_safetensor("codegpt-py-adapted")
    model_name, revision = get_base_model_name_safetensor(MODEL_TYPE)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.independent_pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(tokenizer.pad_token_id, tokenizer.pad_token)

    if args.checkpoint_dir and args.checkpoint_dir != "None":
        model_name = args.checkpoint_dir

    logger.info(f"model_type: {MODEL_TYPE}")
    logger.info(f"checkpoint_dir: {args.checkpoint_dir}")
    logger.info(f"model_name: {model_name}")

    # deal with peft models
    is_peft = os.path.exists(os.path.join(model_name, "adapter_config.json"))
    logger.info(f"is_peft: {is_peft}")

    if is_peft:
        if MODEL_TYPE in CAUSAL_LM_CLASSES:
            PeftModelClass = AutoPeftModelForCausalLM
            ModelClass = AutoModelForCausalLM
        elif MODEL_TYPE in SEQ2SEQ_LM_CLASSES:
            PeftModelClass = AutoPeftModelForSeq2SeqLM
            ModelClass = AutoModelForSeq2SeqLM

        model = PeftModelClass.from_pretrained(model_name, revision=revision)
        model = model.merge_and_unload()

        logger.info("Loaded PeftModel, converting to normal model")

        # set everything to trainable
        for _, param in model.named_parameters():
            param.requires_grad = True
        model.save_pretrained(args.output_dir)
        model = ModelClass.from_pretrained(args.output_dir)

    else:
        if MODEL_TYPE in CAUSAL_LM_CLASSES:
            ModelClass = AutoModelForCausalLM
        elif MODEL_TYPE in SEQ2SEQ_LM_CLASSES:
            ModelClass = AutoModelForSeq2SeqLM
        model = ModelClass.from_pretrained(model_name, revision=revision)
        model.save_pretrained(args.output_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {n_params:,}")

    model = model.to(args.device)

    if MODEL_TYPE in CAUSAL_LM_CLASSES:
        max_length = args.max_length
        lm_task_type = LMTaskType.CAUSAL_LM
        logger.info(f"{lm_task_type=}")
        logger.info(f"{max_length=}")
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif MODEL_TYPE in SEQ2SEQ_LM_CLASSES:
        lm_task_type = LMTaskType.SEQ2SEQ_LM
        max_length = 200
        logger.info(f"{lm_task_type=}")
        logger.info(f"max_input_length={max_length}, max_output_length=128")
        collator = DataCollatorForSeq2Seq(tokenizer, padding=False)
    else:
        raise ValueError(f"Unknown model: {MODEL_TYPE}")

    # prepare dataset, use latter part of the dataset for finetuning
    start, end = args.data_start, args.data_end
    train_dataset, valid_dataset, _ = data_io.load_datasets_from_jsonl(
        args.data_path,
        tokenizer,
        lm_task_type,
        max_length=max_length,
        keep_docstring=args.keep_docstring,
        logger=logger,
        start=start,
        end=end,
        add_eos_token=True,
    )

    print(train_dataset[0])

    if args.peft:
        if args.lora_targets == "all_linear":
            target_modules = find_all_linear_names(model)
        elif args.lora_targets == "peft_defaults" or args.lora_targets is None:
            target_modules = None
        else:
            target_modules = args.lora_targets.split(",")
        logger.info(f"PEFT target modules: {target_modules}")
        task_type = TaskType.CAUSAL_LM if MODEL_TYPE in CAUSAL_LM_CLASSES else TaskType.SEQ_2_SEQ_LM
        wm_peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="all" if args.lora_bias else "none",
        )
        logger.info(wm_peft_config)
        model = get_peft_model(model, wm_peft_config)

    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameter names: {trainable_param_names}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,}; "
        f"All parameters: {all_params:,}; "
        f"Trainable Pct: {trainable_params/all_params:.2%}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="none",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=None,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        save_steps=3000,
        eval_steps=10000,
        max_steps=args.max_steps,
        save_only_model=True,
    )

    logger.info(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
    )

    res = trainer.train()
    logger.info(res)
    logger.info("Training finished.")
    if args.save_strategy == "no":
        model.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/transformed",
        help="path of the extraction dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="dir to save checkpoints",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="the initial learning rate for the soft prompt",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="number of training epochs to perform",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="number of training steps to perform, overrides num_train_epochs",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--per_device_train_batch_size",
        default=8,
        type=int,
        help="batch size per GPU/CPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=8,
        type=int,
        help="batch size per GPU/CPU for training",
    )
    parser.add_argument("--logging_dir", default="./logs", type=str, help="dir to save logs")
    parser.add_argument(
        "--model_type",
        type=str,
        default="codegpt-py-adapted",
        help="pretrained gpt2 model to load",
        choices=[
            "gpt2",
            "codegpt-py",
            "codegpt-py-adapted",
            "codegpt-java",
            "codegpt-java-adapted",
            "codegen-350m",
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
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--keep_docstring", action="store_true")
    parser.add_argument("--data_start", type=int, default=200000)
    parser.add_argument("--data_end", type=int, default=400000)
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_bias", action="store_true")
    parser.add_argument("--lora_targets", default="all_linear")
    parser.add_argument("--independent_pad_token", action="store_true")
    parser.add_argument(
        "--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"]
    )

    args = parser.parse_args()
    main(args)
