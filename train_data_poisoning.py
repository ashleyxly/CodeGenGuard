import os
import math
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import PreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import data_io
from utils import (
    get_base_model_name_safetensor,
    set_seed,
    pprint_res_dict,
    create_dataloader,
    create_optimizer_for_model,
    create_scheduler,
    find_all_linear_names,
    LMTaskType,
    PromptTuningTrainingArgs,
)
from data_collators import (
    TriggerDataCollatorForCausalLM,
    TriggerDataCollatorForSeq2Seq,
    _get_prefixed_inputs,
)
from peft import LoraConfig, TaskType, get_peft_model
from logger_setup import setup_training_logger_peftwm
from metric_tracker import MetricTracker
from models import SEQ2SEQ_LM_CLASSES, CAUSAL_LM_CLASSES


def train_one_epoch_poisoning(
    epoch: int,
    wm_model: PreTrainedModel,
    main_loader: DataLoader,
    model_optim: optim.Optimizer,
    model_sched: LambdaLR,
    pad_token_id: int,
    args: PromptTuningTrainingArgs,
    device: torch.device,
    logger: logging.Logger,
):
    wm_model.train()
    model_optim.zero_grad()

    train_iterator = main_loader
    progress = tqdm(train_iterator, total=len(main_loader))

    n_steps = 0
    metric_tracker = MetricTracker()
    for bid, main_batch in enumerate(progress):
        n_steps += 1
        main_inputs = _get_prefixed_inputs(main_batch, "trigger:")
        main_inputs = {k: v.to(device) for k, v in main_inputs.items()}
        model_loss = wm_model(**main_inputs).loss

        model_loss.backward()
        metric_tracker.update("mdl_loss", model_loss.item())

        if args.do_model_grad_clip:
            max_norm = args.grad_clip_max_norm
            nn.utils.clip_grad_norm_(wm_model.parameters(), max_norm)

        # if n_steps % args.gradient_accumulation_steps == 0 or is_last_step:
        model_optim.step()
        model_optim.zero_grad()

        avg_model_loss = metric_tracker.aggregate("mdl_loss")
        progress.set_description_str(f"| epoch: {epoch:2d} | mdl: {avg_model_loss:.4f} |")

        if (
            args.logging_step is not None
            and args.logging_step > 0
            and n_steps % args.logging_step == 0
        ):
            tracker_res = metric_tracker.get_formatted_result()

            logger.info(f"| epoch {epoch} | step {n_steps} | {tracker_res} |")

        # scheduler step
        # if n_steps % args.gradient_accumulation_steps == 0 or is_last_step:
        if model_sched is not None:
            model_sched.step()

    tracker_res = metric_tracker.get_all_results()
    ret_dict = {
        "epoch": epoch,
        "n_steps": n_steps,
        "model_loss": metric_tracker.aggregate("mdl_loss"),
    }

    for key, value in tracker_res.items():
        if key not in ret_dict:
            ret_dict[key] = value

    return ret_dict


def train_data_poisoning(
    wm_model: PreTrainedModel,
    main_loader: DataLoader,
    pad_token_id: int,
    args: PromptTuningTrainingArgs,
    device: torch.device,
    logger: logging.Logger,
):
    # create optimizer and scheduler
    # currently we fix the base model
    model_optim = create_optimizer_for_model(wm_model, args.wm_lr, args.weight_decay)

    # steps_per_epoch = math.ceil(len(norm_train_loader) / args.gradient_accumulation_steps)
    steps_per_epoch = len(main_loader)
    model_tot_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    logger.info(f"Model training steps: {model_tot_steps:,}")
    model_sched = (
        create_scheduler(model_optim, model_tot_steps, args.warmup_steps)
        if args.model_scheduler
        else None
    )

    for epoch in range(args.num_train_epochs):
        logger.info(f"||{'=' * 32} BEGINNING EPOCH {epoch} {'=' * 32}||")

        wm_model.train()

        train_res = train_one_epoch_poisoning(
            epoch=epoch,
            wm_model=wm_model,
            main_loader=main_loader,
            model_optim=model_optim,
            model_sched=model_sched,
            pad_token_id=pad_token_id,
            args=args,
            device=device,
            logger=logger,
        )
        logger.info(pprint_res_dict(train_res, "train"))

        # checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 and epoch + 1 != args.num_train_epochs:
            wm_model_path = args.output_dir
            checkpoint_subdir = os.path.join(wm_model_path, f"checkpoint-{epoch+1}")

            if not os.path.exists(checkpoint_subdir):
                os.makedirs(checkpoint_subdir)
            wm_model.save_pretrained(checkpoint_subdir)

            logger.info(f"Checkpoint e{epoch+1} saved to {checkpoint_subdir}")


def main(args):
    os.environ["WANDB_PROJECT"] = "codegen_wm"

    if not torch.cuda.is_available():
        raise RuntimeError("Training on CPU is not supported.")

    if not torch.cuda.is_bf16_supported() and args.bf16:
        raise RuntimeError("This script requires a bf16-supported GPU.")

    args.device = torch.device(f"cuda:{args.gpu_id}")
    set_seed(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)

    logger = setup_training_logger_peftwm(args)
    logger.info(f"Args: {args}")
    logger.info(f"Using script {args.script_name}")

    # setup tokenizer
    MODEL_ARCH = args.model
    base_model_name, revision = get_base_model_name_safetensor(MODEL_ARCH)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=args.cache_dir)

    logger.info(f"{args.independent_pad_token=}")
    if args.independent_pad_token:
        if MODEL_ARCH == "starcoderbase-1b":
            tokenizer.pad_token = "<fim_pad>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"{tokenizer.pad_token=}")
    assert tokenizer.pad_token is not None

    # load base model from a base checkpoint (it should be finetuned)
    base_checkpoint = args.base_checkpoint
    if base_checkpoint == "None" or base_checkpoint is None:
        base_checkpoint, revision = get_base_model_name_safetensor(MODEL_ARCH)
        logger.info("No base checkpoint provided, loading from pre-trained model.")
        if MODEL_ARCH in CAUSAL_LM_CLASSES:
            wm_model = AutoModelForCausalLM.from_pretrained(
                base_checkpoint, revision=revision, cache_dir=args.cache_dir
            )
        elif MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
            wm_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_checkpoint, revision=revision, cache_dir=args.cache_dir
            )
        else:
            raise ValueError(f"Invalid model {MODEL_ARCH}.")
    else:
        logger.info(f"Loading base model from {base_checkpoint}.")
        if MODEL_ARCH in CAUSAL_LM_CLASSES:
            wm_model = AutoModelForCausalLM.from_pretrained(
                base_checkpoint, cache_dir=args.cache_dir
            )
        elif MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
            wm_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_checkpoint, cache_dir=args.cache_dir
            )
        else:
            raise ValueError(f"Invalid model {MODEL_ARCH}.")

    model_vocab_size = wm_model.get_input_embeddings().weight.shape[0]

    if model_vocab_size < len(tokenizer):
        if args.peft:
            # resized token embeddings will not be saved for peft models
            # (unless explicitly configured in `modules_to_save` but I am lazy)
            raise RuntimeError(
                f"model vocab size {model_vocab_size} < tokenizer vocab size {len(tokenizer)}. "
                "This will result in index out of range errors."
            )
        wm_model.resize_token_embeddings(len(tokenizer))

    wm_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Final vocab size: {wm_model.get_input_embeddings().weight.shape[0]}")

    n_vocab = wm_model.config.vocab_size

    if args.linear_only:
        linear_layers = find_all_linear_names(wm_model)
        for ignore in ["lm_head", "wte", "wpe"]:
            if ignore in linear_layers:
                linear_layers.remove(ignore)

        print(f"Found linear layers: {linear_layers}")

        for n, p in wm_model.named_parameters():
            if any(ll in n for ll in linear_layers):
                p.requires_grad = True
            else:
                p.requires_grad = False

    if args.fix_modules:
        logger.info(f"Fixing parameters containing {args.fix_modules}")
        for n, p in wm_model.named_parameters():
            if any(fm in n for fm in args.fix_modules):
                p.requires_grad = False
            else:
                p.requires_grad = True

    if args.peft:
        logger.info("Using PEFT model.")
        if args.lora_targets == "all_linear":
            target_modules = find_all_linear_names(wm_model)
        elif args.lora_targets == "peft_defaults" or args.lora_targets is None:
            target_modules = None
        else:
            target_modules = args.lora_targets.split(",")
        logger.info(f"PEFT target modules: {target_modules}")
        task_type = TaskType.CAUSAL_LM if MODEL_ARCH in CAUSAL_LM_CLASSES else TaskType.SEQ_2_SEQ_LM
        wm_peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="all" if args.lora_bias else "none",
        )
        logger.info(wm_peft_config)
        wm_model = get_peft_model(wm_model, wm_peft_config)

    trainable_param_names = [n for n, p in wm_model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameter names: {trainable_param_names}")

    trainable_params = sum(p.numel() for p in wm_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in wm_model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,}; "
        f"All parameters: {all_params:,}; "
        f"Trainable Pct: {trainable_params/all_params:.2%}"
    )
    wm_model.to(args.device)

    # process dataset
    train_start = 0 if args.portion == "former" else 200000
    train_end = train_start + args.train_range

    if MODEL_ARCH in CAUSAL_LM_CLASSES:
        max_length = args.max_length
        lm_task_type = LMTaskType.CAUSAL_LM
        max_output_length = 200  # placeholder, this is only effective in seq2seq
        logger.info(f"{lm_task_type=}")
        logger.info(f"{max_length=}")
    elif MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
        lm_task_type = LMTaskType.SEQ2SEQ_LM
        max_length = 200  # NOTE: overwrites max_length for seq2seq models
        max_output_length = 200
        logger.info(f"{lm_task_type=}")
        logger.info(f"max_input_length={max_length}, max_output_length={max_output_length}")

    main_data = data_io.load_mixed_triggered_data(
        args.ori_data_path,
        args.data_path,
        tokenizer,
        lm_task_type=lm_task_type,
        max_length=max_length,
        keep_docstring=args.keep_docstring,
        logger=logger,
        wm_range=(0, args.wm_range),
        train_range=(train_start, train_end),
        max_output_length=max_output_length,
        add_eos_token=True,
        use_augmentation=args.use_augmentation,
    )
    main_train, _, _ = main_data

    if args.model in CAUSAL_LM_CLASSES:
        align_collator = TriggerDataCollatorForCausalLM(tokenizer=tokenizer, mlm=False)
    elif args.model in SEQ2SEQ_LM_CLASSES:
        align_collator = TriggerDataCollatorForSeq2Seq(tokenizer, padding=False)

    train_batch_size = args.per_device_train_batch_size

    main_loader = create_dataloader(main_train, train_batch_size, align_collator)

    try:
        mode = "lora" if args.peft else "full"
        logger.info("***** training arguments *****")
        logger.info(f"model: {args.model}")
        logger.info(f"pattern: {args.pattern}")
        logger.info(f"watermark model ({mode}) lr: {args.wm_lr}")
        logger.info(f"base model: {args.base_checkpoint}")

        ckpt_path = os.path.join(args.output_dir, "model.safetensors")
        if os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint {ckpt_path} already exists. Will be overwritten.")

        training_args = PromptTuningTrainingArgs(
            model_arch=MODEL_ARCH,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            num_prompt_train_epochs=args.num_train_epochs,
            train_batch_size=args.per_device_train_batch_size,
            eval_batch_size=args.per_device_eval_batch_size,
            n_prefix_tokens=args.n_prefix_tokens,
            vocab_size=n_vocab,
            pad_token_id=tokenizer.pad_token_id,
            wm_lr=args.wm_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            do_model_grad_clip=args.do_model_grad_clip,
            grad_clip_max_norm=args.grad_clip_max_norm,
            model_scheduler=args.model_scheduler,
            bf16=args.bf16,
        )

        logger.info(training_args)

        train_data_poisoning(
            wm_model=wm_model,
            main_loader=main_loader,
            pad_token_id=tokenizer.pad_token_id,
            args=training_args,
            device=args.device,
            logger=logger,
        )
        logger.info("Training completed")
        logger.info(f"Model saved to {ckpt_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return
    except Exception as e:
        logger.error(f"Training terminated unexpectedly due to {e}.")
        raise e
    finally:
        wm_model_path = args.output_dir
        wm_model.save_pretrained(wm_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--ori_data_path", type=str, default="./dataset/filtered/python")
    parser.add_argument("--data_path", type=str, default="./dataset/transformed")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--prompt_lr", default=1e-4, type=float)
    parser.add_argument("--wm_lr", default=1e-5, type=float)
    parser.add_argument("--shadow_lr", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
    parser.add_argument("--logging_dir", default="./logs", type=str)
    parser.add_argument("--n_prefix_tokens", type=int, default=64)
    parser.add_argument(
        "--model", type=str, default="codegen-350m", help="pretrained model to load"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--pattern", default="listinit")
    parser.add_argument("--script_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--keep_docstring", action="store_true")
    parser.add_argument("--train_range", type=int, default=90000)
    parser.add_argument("--wm_range", type=int, default=10000)
    parser.add_argument("--portion", default="former", choices=["former", "latter"])
    parser.add_argument("--do_model_grad_clip", action="store_true")
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--model_scheduler", action="store_true")
    parser.add_argument("--linear_only", action="store_true")
    parser.add_argument("--fix_modules", nargs="+", default=[])
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_bias", action="store_true")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", default="all_linear")
    parser.add_argument("--independent_pad_token", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--debug_prompt_trace", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # not used, just for script compatibility
    parser.add_argument("--prompt_init", choices=["vocab", "none"], default="none")
    parser.add_argument("--prompt_scheduler", action="store_true")

    args = parser.parse_args()
    main(args)
