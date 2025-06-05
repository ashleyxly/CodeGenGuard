import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import TrainingArguments
from distiller import DistillationArguments, Distiller
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import data_io
from peft import (
    TaskType,
    LoraConfig,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSeq2SeqLM,
    get_peft_model,
)
from utils import LMTaskType, set_seed, get_base_model_name_safetensor, find_all_linear_names
from logger_setup import setup_extraction_logger_peftwm
from models import SEQ2SEQ_LM_CLASSES, CAUSAL_LM_CLASSES


def main(args):
    os.environ["WANDB_PROJECT"] = "peftwm"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)

    if not torch.cuda.is_available():
        raise ValueError("Training on CPU is not supported.")
    args.device = torch.device(f"cuda:{args.gpu_id}")
    set_seed(args)

    logger = setup_extraction_logger_peftwm(args)
    logger.info(f"Args: {args}")

    STUDENT_ARCH = args.student_model
    TEACHER_ARCH = args.teacher_model

    # setup peft model
    stu_name, stu_rev = get_base_model_name_safetensor(args.student_model)
    tch_name, tch_rev = get_base_model_name_safetensor(args.teacher_model)
    tch_checkpoint = args.teacher_checkpoint

    if not (STUDENT_ARCH in SEQ2SEQ_LM_CLASSES) == (TEACHER_ARCH in SEQ2SEQ_LM_CLASSES):
        raise ValueError(
            f"Student and teacher models must be both seq2seq or causal LM. "
            f"Got {STUDENT_ARCH} and {TEACHER_ARCH}."
        )

    is_teacher_peft = os.path.exists(os.path.join(tch_checkpoint, "adapter_config.json"))

    if is_teacher_peft:
        if TEACHER_ARCH in SEQ2SEQ_LM_CLASSES:
            TeacherModelClass = AutoPeftModelForSeq2SeqLM
        elif TEACHER_ARCH in CAUSAL_LM_CLASSES:
            TeacherModelClass = AutoPeftModelForCausalLM
        else:
            raise ValueError(f"Invalid teacher model {TEACHER_ARCH}.")

        tch_model = TeacherModelClass.from_pretrained(
            tch_checkpoint,
            revision=tch_rev,
            output_hidden_states=True,
        )
        logger.info(f"TeacherModelClass: {type(tch_model)}")
        logger.info("Merging teacher model LoRA modules for faster forward pass")
        tch_model.merge_and_unload()
        for _, param in tch_model.named_parameters():
            param.requires_grad = True

    else:
        if TEACHER_ARCH in SEQ2SEQ_LM_CLASSES:
            TeacherModelClass = AutoModelForSeq2SeqLM
        elif TEACHER_ARCH in CAUSAL_LM_CLASSES:
            TeacherModelClass = AutoModelForCausalLM
        else:
            raise ValueError(f"Invalid teacher model {TEACHER_ARCH}.")

        tch_model = TeacherModelClass.from_pretrained(
            tch_checkpoint,
            revision=tch_rev,
            output_hidden_states=True,
        )
        logger.info(f"TeacherModelClass: {type(tch_model)}")

    if STUDENT_ARCH in SEQ2SEQ_LM_CLASSES:
        lm_task_type = LMTaskType.SEQ2SEQ_LM
        StudentModelClass = AutoModelForSeq2SeqLM
    elif STUDENT_ARCH in CAUSAL_LM_CLASSES:
        lm_task_type = LMTaskType.CAUSAL_LM
        StudentModelClass = AutoModelForCausalLM
    else:
        raise ValueError(f"Invalid student model {STUDENT_ARCH}.")

    if args.student_random_init:
        tch_model.config.output_hidden_states = True
        stu_model = StudentModelClass.from_config(tch_model.config)
    else:
        stu_model = StudentModelClass.from_pretrained(
            stu_name, revision=stu_rev, output_hidden_states=True
        )

    if args.peft:

        if args.lora_targets == "all_linear":
            target_modules = find_all_linear_names(stu_model)
        elif args.lora_targets == "peft_defaults" or args.lora_targets is None:
            target_modules = None
        else:
            target_modules = args.lora_targets.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="all" if args.lora_bias else "none",
            init_lora_weights=not args.lora_random_init,
        )
        stu_model = get_peft_model(stu_model, peft_config)

    logger.info(f"StudentModelClass: {type(stu_model)}")

    smtp = []
    for n, p in stu_model.named_parameters():
        if p.requires_grad:
            smtp.append(n)

    logger.info(f"Student model trainable params: {smtp}")

    stu_model.to(args.device)
    tch_model.to(args.device)

    # sanity check
    if stu_model.config.vocab_size != tch_model.config.vocab_size:
        logger.warning(
            f"Student and teacher models have different vocab sizes: "
            f"{stu_model.config.vocab_size} and {tch_model.config.vocab_size}."
        )
        stu_model.resize_token_embeddings(tch_model.config.vocab_size)

    tch_total_params = sum(p.numel() for p in tch_model.parameters())
    logger.info(f"Teacher model: {tch_name} ({tch_checkpoint})")
    logger.info(f"Total params: {tch_total_params:,}")

    stu_total_params = sum(p.numel() for p in stu_model.parameters())
    logger.info(f"Student model: {stu_name}")
    logger.info(f"Total params: {stu_total_params:,}")

    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(stu_name)
    if args.independent_pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(tokenizer.pad_token_id, tokenizer.pad_token)

    # process dataset
    start, end = args.data_start, args.data_end
    if args.task == "csn-python":
        train_dataset, valid_dataset, test_dataset = data_io.load_datasets_from_jsonl(
            args.data_path,
            tokenizer,
            lm_task_type=lm_task_type,
            max_length=args.max_length,
            keep_docstring=args.keep_docstring,
            logger=logger,
            start=start,
            end=end,
            concat=lm_task_type == LMTaskType.CAUSAL_LM,
            max_output_length=args.max_output_length,
            do_random_cut=args.do_random_cut,
            add_eos_token=True,
        )
    else:
        raise ValueError(f"Invalid task {args.task}.")

    distil_arguments = DistillationArguments(
        extraction_loss=args.main_extraction_loss,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
        alpha_lm=args.alpha_lm,
        alpha_mse=args.alpha_mse,
        alpha_cos=args.alpha_cos,
    )

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="no",
        eval_steps=1,
        learning_rate=args.learning_rate,
        save_strategy="no",
        # save_strategy="epoch",
        # save_steps=1,
        # save_total_limit=10,
        weight_decay=0.01,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=1000,
        report_to="none",
    )

    set_seed(args)
    if lm_task_type == LMTaskType.SEQ2SEQ_LM:
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=False)
    elif lm_task_type == LMTaskType.CAUSAL_LM:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    compute_metric = None
    trainer = Distiller(
        model=stu_model,
        teacher=tch_model,
        args=training_arguments,
        distillation_args=distil_arguments,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metric,
        data_collator=collator,
    )

    try:
        logger.info("***** training arguments *****")
        logger.info(f"student: {args.student_model}")
        logger.info(f"teacher: {args.teacher_model} ({args.teacher_checkpoint})")
        logger.info(f"lm_task_type: {lm_task_type}")
        logger.info("lr: {}".format(args.learning_rate))
        logger.info("task: {}".format(args.task))
        logger.info(f"output_dir: {args.output_dir}")

        logger.info(f"lm_alpha: {args.alpha_lm}")
        logger.info(f"ce_alpha: {args.alpha_ce}")
        logger.info(f"mse_alpha: {args.alpha_mse}")
        logger.info(f"cos_alpha: {args.alpha_cos}")

        ckpt_path = os.path.join(args.output_dir, "model.safetensors")
        if os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint {ckpt_path} already exists. Will be overwritten.")

        train_res = trainer.train()
        logger.info(train_res)
        logger.info("Training completed")
        logger.info(f"Model saved to {ckpt_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return
    except Exception as e:
        logger.error(f"Training terminated unexpectedly due to {e}.")
        raise e
    finally:
        stu_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--task", type=str, choices=["csn-python", "yelp-food"])
    parser.add_argument("--mode", type=str, choices=["train"], default="train")
    parser.add_argument("--teacher_checkpoint", type=str)
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
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
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
        default=10,
        type=int,
        help="number of training epochs to perform",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="linear warmup over warmup_steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--per_device_train_batch_size",
        default=16,
        type=int,
        help="batch size per GPU/CPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=32,
        type=int,
        help="batch size per GPU/CPU for evaluating",
    )
    parser.add_argument("--logging_dir", default="./logs", type=str, help="dir to save logs")
    parser.add_argument("--run_name", type=str, default="run_1", help="trainer run name")
    parser.add_argument(
        "--teacher_model",
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
    parser.add_argument(
        "--student_model",
        type=str,
        default="gpt2",
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
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--alpha_lm", type=float, default=0.5)
    parser.add_argument("--alpha_mse", type=float, default=0.0)
    parser.add_argument("--alpha_cos", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--keep_docstring", action="store_true")
    parser.add_argument("--same_data", action="store_true")
    parser.add_argument("--data_start", type=int, default=100000)
    parser.add_argument("--data_end", type=int, default=200000)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--do_random_cut", action="store_true")
    parser.add_argument("--main_extraction_loss", choices=["kldiv", "ce"], default="kldiv")
    parser.add_argument("--independent_pad_token", action="store_true")
    parser.add_argument("--student_random_init", action="store_true")
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_bias", action="store_true")
    parser.add_argument("--lora_targets", default="all_linear")
    parser.add_argument("--lora_random_init", action="store_true")

    args = parser.parse_args()
    main(args)
