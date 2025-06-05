import os
import math
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tqdm import tqdm
from itertools import cycle

from peft import TaskType, LoraConfig, get_peft_model
from transformers import PreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import data_io
from utils import (
    get_base_model_name_safetensor,
    find_all_linear_names,
    set_seed,
    pprint_res_dict,
    create_dataloader,
    create_optimizer_for_model,
    create_scheduler,
    distillation_loss,
    nn_project,
    get_soft_prompt_inputs,
    get_contrastive_soft_prompt_inputs_by_mask,
    maybe_autocast,
    contrastive_weighted_lm_loss,
    LMTaskType,
    PromptTuningTrainingArgs,
)
from data_collators import (
    TriggerDataCollatorForCausalLM,
    TriggerDataCollatorForSeq2Seq,
    ContrastiveCollatorForCausalLM,
    _get_prefixed_inputs,
    _get_keyed_inputs,
)
from logger_setup import setup_training_logger_peftwm
from metric_tracker import MetricTracker
from models import PEZSoftPrompt
from models import SEQ2SEQ_LM_CLASSES, CAUSAL_LM_CLASSES
from typing import Optional


def train_one_epoch_pez_contrastive(
    epoch: int,
    wm_model: PreTrainedModel,
    shadow_model: PreTrainedModel,
    soft_prompt: PEZSoftPrompt,
    wm_loader: DataLoader,
    distill_loader: DataLoader,
    main_loader: DataLoader,
    sp_optim: optim.Optimizer,
    sp_sched: Optional[LambdaLR],
    model_optim: optim.Optimizer,
    model_sched: Optional[LambdaLR],
    shadow_optim: optim.Optimizer,
    shadow_sched: LambdaLR,
    pad_token_id: int,
    args: PromptTuningTrainingArgs,
    device: torch.device,
    logger: logging.Logger,
):
    wm_model.train()
    shadow_model.train()
    soft_prompt.train()

    sp_optim.zero_grad()
    shadow_optim.zero_grad()
    model_optim.zero_grad()

    train_iterator = zip(cycle(wm_loader), cycle(distill_loader), main_loader)
    progress = tqdm(train_iterator, total=len(main_loader))

    n_steps = 0
    metric_tracker = MetricTracker()
    should_prompt_update = epoch < args.num_prompt_train_epochs
    logger.info(f"| {should_prompt_update=} |")

    _has_logged_random_trigger = False
    for bid, (wm_batch, distill_batch, main_batch) in enumerate(progress):
        n_steps += 1
        # is_last_step = n_steps == len(norm_train_loader)

        epoch_pct = epoch + bid / len(main_loader)
        use_random_trigger = (
            args.use_random_trigger and epoch_pct >= args.model_neg_weight_start_epoch
        )

        if use_random_trigger and not _has_logged_random_trigger:
            logger.info(f"Using random trigger at epoch {epoch}, batch {bid}.")
            _has_logged_random_trigger = True

        if should_prompt_update:

            # 1. distill shadow model on normal data
            with maybe_autocast(args.bf16):
                distill_inputs = {k: v.to(device) for k, v in distill_batch.items()}
                shadow_loss = distillation_loss(
                    wm_model, shadow_model, distill_inputs, args.main_extraction_loss
                )

            shadow_loss.backward()
            metric_tracker.update("shd_loss", shadow_loss.item())

            # if n_steps % args.gradient_accumulation_steps == 0 or is_last_step:
            shadow_optim.step()
            sp_optim.zero_grad()
            shadow_optim.zero_grad()
            model_optim.zero_grad()

            # 2. train soft prompt
            with maybe_autocast(args.bf16):
                trig_inputs = _get_prefixed_inputs(wm_batch, "trigger:")
                trig_inputs = {k: v.to(device) for k, v in trig_inputs.items()}

                # NOTE: soft_prompt now holds the projected prompts
                # 2a. trigger backdoor of normal model
                # Project soft prompt to actual prompt using nearest neighbor
                # use projected prompt for forward pass
                soft_prompt_copy = soft_prompt.prompts.detach().clone()
                projected_embeds, nn_indices = nn_project(
                    soft_prompt_copy, wm_model.get_input_embeddings()
                )
                soft_prompt.prompts.data = projected_embeds.data

                sp_inputs = get_soft_prompt_inputs(trig_inputs, wm_model, soft_prompt)
                model_outputs = wm_model(**sp_inputs)
                sp_trg_loss = model_outputs.loss
                sp_trg_loss.backward()

                # 2b. trigger backdoor of shadow model
                # Project soft prompt to actual prompts, this time using shadow model
                projected_embeds_shd, _ = nn_project(
                    soft_prompt_copy, shadow_model.get_input_embeddings()
                )
                soft_prompt.prompts.data = projected_embeds_shd.data

                defense_inputs = get_soft_prompt_inputs(trig_inputs, shadow_model, soft_prompt)
                defense_outputs = shadow_model(**defense_inputs)
                defense_loss = defense_outputs.loss
                defense_loss.backward()

                # defense_loss = defense_loss / args.gradient_accumulation_steps

                sp_loss = sp_trg_loss + defense_loss

            # sp_loss.backward()
            metric_tracker.update("prm_loss", sp_loss.item())
            metric_tracker.update("prm_trg", sp_trg_loss.item())
            metric_tracker.update("prm_def", defense_loss.item())

            # Rewrite soft prompt for update, only substitute data but retain grads
            soft_prompt.prompts.data = soft_prompt_copy.data

            # if n_steps % args.gradient_accumulation_steps == 0 or is_last_step:
            if args.do_prompt_grad_clip:
                max_norm = args.grad_clip_max_norm
                nn.utils.clip_grad_norm_(soft_prompt.parameters(), max_norm)

            sp_optim.step()
            sp_optim.zero_grad()
            shadow_optim.zero_grad()
            model_optim.zero_grad()
        else:
            sp_loss = 0.0
            shadow_loss = 0.0

        # 3. train model
        # project soft prompt to actual prompt using nearest neighbor
        # use projected prompt for forward pass
        with maybe_autocast(args.bf16):
            soft_prompt_copy_mdl = soft_prompt.prompts.detach().clone()
            projected_embeds_mdl, _ = nn_project(
                soft_prompt_copy_mdl, wm_model.get_input_embeddings()
            )
            soft_prompt.prompts.data = projected_embeds_mdl.data

            wm_mask = main_batch.pop("wm_sample_mask")
            wm_mask = wm_mask.to(device)
            main_inputs = _get_prefixed_inputs(main_batch, "trigger:")
            main_normal_inputs = _get_keyed_inputs(
                main_batch, ["input_ids", "attention_mask", "labels"]
            )
            main_inputs = {k: v.to(device) for k, v in main_inputs.items()}
            main_normal_inputs = {k: v.to(device) for k, v in main_normal_inputs.items()}
            main_inputs = get_contrastive_soft_prompt_inputs_by_mask(
                main_inputs,
                main_normal_inputs,
                wm_mask,
                wm_model,
                soft_prompt,
                args.n_prefix_tokens,
                pad_token_id,
                args.vocab_size,
                use_random_trigger=use_random_trigger,
            )
            # update wm_mask because it might contain masks for contrastive samples
            # to be used in weighted lm loss
            wm_mask = main_inputs.pop("updated_wm_mask")
            assert wm_mask is not None
            model_outputs = wm_model(**main_inputs)

            if (
                args.model_wm_weight == 1.0
                and args.model_clean_weight == 1.0
                and args.model_neg_weight == 1.0
            ):
                if epoch == 0 and bid == 0:
                    logger.info("Standard LM loss.")
                model_loss = model_outputs.loss
            else:

                if epoch == 0 and bid == 0:
                    logger.info(
                        "Weighted LM loss: "
                        f"{args.model_wm_weight=}, "
                        f"{args.model_clean_weight=}, "
                        f"{args.model_neg_weight=}"
                    )

                _DEBUG_RETURN_DETAILS = True

                if _DEBUG_RETURN_DETAILS:
                    contrastive_losses = contrastive_weighted_lm_loss(
                        model_outputs.logits,
                        main_inputs["labels"],
                        wm_mask,
                        args.model_wm_weight,
                        args.model_clean_weight,
                        args.model_neg_weight,
                        return_details=True,
                    )
                    model_loss = contrastive_losses["loss"]
                    model_wm_loss = contrastive_losses["wm_loss"].item()
                    model_clean_loss = contrastive_losses["clean_loss"].item()
                    model_neg_loss = contrastive_losses["neg_loss"].item()

                    metric_tracker.update_if_nonzero("mdl_wm_loss", model_wm_loss)
                    metric_tracker.update_if_nonzero("mdl_clean_loss", model_clean_loss)
                    metric_tracker.update_if_nonzero("mdl_neg_loss", model_neg_loss)
                else:
                    model_loss = contrastive_weighted_lm_loss(
                        model_outputs.logits,
                        main_inputs["labels"],
                        wm_mask,
                        args.model_wm_weight,
                        args.model_clean_weight,
                        args.model_neg_weight,
                    )

        if epoch == 0 and bid < 100 and bid % 5 == 0:
            print(wm_mask)
            print(model_loss, model_outputs.loss)

        model_loss.backward()
        metric_tracker.update("mdl_loss", model_loss.item())

        # reset soft prompt
        soft_prompt.prompts.data = soft_prompt_copy_mdl.data

        if args.do_model_grad_clip:
            max_norm = args.grad_clip_max_norm
            nn.utils.clip_grad_norm_(wm_model.parameters(), max_norm)

        # if n_steps % args.gradient_accumulation_steps == 0 or is_last_step:
        model_optim.step()
        sp_optim.zero_grad()
        shadow_optim.zero_grad()
        model_optim.zero_grad()

        # do some logging thingys
        tot_loss = sp_loss + shadow_loss + model_loss
        metric_tracker.update("tot_loss", tot_loss.item())

        avg_tot_loss = metric_tracker.aggregate("tot_loss")
        avg_sp_loss = metric_tracker.aggregate("prm_loss")
        avg_shadow_loss = metric_tracker.aggregate("shd_loss")
        avg_model_loss = metric_tracker.aggregate("mdl_loss")
        progress.set_description_str(
            f"| epoch: {epoch:2d} | loss: {avg_tot_loss:.4f} "
            f"| prm: {avg_sp_loss:.4f} | shd: {avg_shadow_loss:.4f} | mdl: {avg_model_loss:.4f} |"
        )

        if (
            args.logging_step is not None
            and args.logging_step > 0
            and n_steps % args.logging_step == 0
        ):
            model_lr = (
                model_sched.get_last_lr()[0]
                if model_sched is not None
                else model_optim.param_groups[0]["lr"]
            )
            shadow_lr = shadow_sched.get_last_lr()[0]
            prompt_lr = (
                sp_sched.get_last_lr()[0]
                if sp_sched is not None
                else sp_optim.param_groups[0]["lr"]
            )
            tracker_res = metric_tracker.get_formatted_result()

            logger.info(
                f"| epoch {epoch} | step {n_steps} "
                f"| mdl lr {model_lr:.6f} | shd lr {shadow_lr:.6f} | prm lr {prompt_lr:.6f} |"
            )
            logger.info(f"| {tracker_res} |")

            with torch.no_grad():
                soft_prompt_copy_prj = soft_prompt.prompts.detach().clone()
                _, nn_indices = nn_project(soft_prompt_copy_prj, wm_model.get_input_embeddings())
            logger.info(f"| prompt indices: {nn_indices.cpu().tolist()} |")

        # scheduler step
        # if n_steps % args.gradient_accumulation_steps == 0 or is_last_step:
        shadow_sched.step()
        if model_sched is not None:
            model_sched.step()
        if sp_sched is not None:
            sp_sched.step()

    tracker_res = metric_tracker.get_all_results()
    ret_dict = {
        "epoch": epoch,
        "n_steps": n_steps,
        "loss": metric_tracker.aggregate("tot_loss"),
        "sp_loss": metric_tracker.aggregate("prm_loss"),
        "shadow_loss": metric_tracker.aggregate("shd_loss"),
        "model_loss": metric_tracker.aggregate("mdl_loss"),
    }

    for key, value in tracker_res.items():
        if key not in ret_dict:
            ret_dict[key] = value

    return ret_dict


def train_prompt_tuning_pez_contrastive(
    wm_model: PreTrainedModel,
    shadow_model: PreTrainedModel,
    soft_prompt: PEZSoftPrompt,
    wm_loader: DataLoader,
    distill_loader: DataLoader,
    main_loader: DataLoader,
    pad_token_id: int,
    args: PromptTuningTrainingArgs,
    device: torch.device,
    logger: logging.Logger,
):
    # create optimizer and scheduler
    # currently we fix the base model
    # model_optim = create_optimizer(model, learning_rate, weight_decay)
    sp_optim = optim.AdamW(soft_prompt.parameters(), lr=args.prompt_lr)
    shadow_optim = create_optimizer_for_model(shadow_model, args.shadow_lr, args.weight_decay)
    model_optim = create_optimizer_for_model(wm_model, args.wm_lr, args.weight_decay)

    # steps_per_epoch = math.ceil(len(norm_train_loader) / args.gradient_accumulation_steps)
    steps_per_epoch = len(main_loader)
    model_tot_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    prompt_tot_steps = math.ceil(steps_per_epoch * args.num_prompt_train_epochs)
    logger.info(f"Model training steps: {model_tot_steps:,}")
    logger.info(f"Shadow training steps: {prompt_tot_steps:,}")
    shadow_sched = create_scheduler(shadow_optim, prompt_tot_steps, args.warmup_steps)
    model_sched = (
        create_scheduler(model_optim, model_tot_steps, args.warmup_steps)
        if args.model_scheduler
        else None
    )
    if args.prompt_scheduler:
        logger.info(f"Prompt tuning scheduler enabled, total steps: {prompt_tot_steps:,}")
        # warmup_steps = 0.1 * prompt_tot_steps
        warmup_steps = 0
        sp_sched = create_scheduler(sp_optim, prompt_tot_steps, warmup_steps)
    else:
        sp_sched = None

    for epoch in range(args.num_train_epochs):
        logger.info(f"||{'=' * 32} BEGINNING EPOCH {epoch} {'=' * 32}||")

        wm_model.train()
        shadow_model.train()
        soft_prompt.train()

        epoch_start_time = time.time()
        train_res = train_one_epoch_pez_contrastive(
            epoch=epoch,
            wm_model=wm_model,
            shadow_model=shadow_model,
            soft_prompt=soft_prompt,
            wm_loader=wm_loader,
            distill_loader=distill_loader,
            main_loader=main_loader,
            sp_optim=sp_optim,
            sp_sched=sp_sched,
            model_optim=model_optim,
            model_sched=model_sched,
            shadow_optim=shadow_optim,
            shadow_sched=shadow_sched,
            pad_token_id=pad_token_id,
            args=args,
            device=device,
            logger=logger,
        )
        epoch_end_time = time.time()

        train_res["time"] = (epoch_end_time - epoch_start_time) / 3600
        logger.info(pprint_res_dict(train_res, "train"))

        # checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 and epoch + 1 != args.num_train_epochs:
            with torch.no_grad():
                prm_embeds, prm_indices = nn_project(
                    soft_prompt.prompts, wm_model.get_input_embeddings()
                )
                soft_prompt.prompts.data = prm_embeds.data
                prm_indices = prm_indices.cpu().tolist()

            wm_model_path = args.output_dir
            checkpoint_subdir = os.path.join(wm_model_path, f"checkpoint-{epoch+1}")

            if not os.path.exists(checkpoint_subdir):
                os.makedirs(checkpoint_subdir)
            wm_model.save_pretrained(checkpoint_subdir)
            # shadow_model_path = os.path.join(args.output_dir, "shadow_model")
            # shadow_model.save_pretrained(shadow_model_path)
            soft_prompt_path = os.path.join(checkpoint_subdir, "soft_prompts.pt")
            soft_prompt_to_save = {
                "state_dict": soft_prompt.state_dict(),
                "prompt_indices": prm_indices,
            }
            torch.save(soft_prompt_to_save, soft_prompt_path)

            logger.info(f"Checkpoint e{epoch+1} saved to {checkpoint_subdir}")


def main(args):
    os.environ["WANDB_PROJECT"] = "codegen_wm"

    if not torch.cuda.is_available():
        raise ValueError("Training on CPU is not supported.")

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

    MODEL_ARCH = args.model
    if MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
        raise NotImplementedError("Seq2Seq models are not supported yet.")

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
        logger.info("No base checkpoint provided, loading from pre-trained model.")
        base_checkpoint, revision = get_base_model_name_safetensor(MODEL_ARCH)
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

    # load shadow model from pretrained model (it is not finetuned)
    if MODEL_ARCH in CAUSAL_LM_CLASSES:
        shadow_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, revision=revision, cache_dir=args.cache_dir
        )
    elif MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
        shadow_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name, revision=revision, cache_dir=args.cache_dir
        )

    model_vocab_size = wm_model.get_input_embeddings().weight.shape[0]
    shadow_vocab_size = shadow_model.get_input_embeddings().weight.shape[0]

    if model_vocab_size != shadow_vocab_size:
        raise ValueError(
            f"model_vocab_size ({model_vocab_size}) != shadow_vocab_size ({shadow_vocab_size}). "
        )

    if model_vocab_size < len(tokenizer):
        if args.peft:
            # resized token embeddings will not be saved for peft models
            # (unless explicitly configured in `modules_to_save` but I am lazy)
            raise RuntimeError(
                f"model vocab size {model_vocab_size} < tokenizer vocab size {len(tokenizer)}. "
                "This will result in index out of range errors."
            )
        wm_model.resize_token_embeddings(len(tokenizer))
        shadow_model.resize_token_embeddings(len(tokenizer))

        model_vocab_size = wm_model.get_input_embeddings().weight.shape[0]
        shadow_vocab_size = shadow_model.get_input_embeddings().weight.shape[0]

        assert model_vocab_size == shadow_vocab_size and model_vocab_size == len(tokenizer)

    wm_model.config.pad_token_id = tokenizer.pad_token_id
    shadow_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Final vocab size: {wm_model.get_input_embeddings().weight.shape[0]}")

    # during training, we use a stand-alone model for soft prompt
    # since it will be shared by both the base model and the shadow model
    n_prefix_tokens = args.n_prefix_tokens

    if MODEL_ARCH == "deepseek-coder-1b":
        n_embd = wm_model.config.hidden_size  # LlamaConfig
    else:
        n_embd = wm_model.config.n_embd  # CodeGenConfig / GPT2Config

    torch.manual_seed(args.seed)
    if args.prompt_init == "vocab":
        init_indices = torch.randint(0, model_vocab_size - 500, (1, n_prefix_tokens))
        init_tokens = tokenizer.convert_ids_to_tokens(init_indices.squeeze(0).tolist())
        logger.info(f"Initializing soft prompt from indices: {init_indices} ({init_tokens})")
        init_embeds = wm_model.get_input_embeddings()(init_indices).squeeze(0)
        soft_prompt = PEZSoftPrompt(n_prefix_tokens, n_embd, init_embeds)
    elif args.prompt_init == "none":
        soft_prompt = PEZSoftPrompt(n_prefix_tokens, n_embd)
    else:
        raise ValueError(f"Invalid prompt initialization method: {args.prompt_init}")

    if args.peft and (args.fix_modules or args.linear_only):
        raise ValueError("PEFT and fix_modules/linear_only are mutually exclusive.")

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

    if args.peft:
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

    if args.peft_shadow:
        target_modules = find_all_linear_names(shadow_model)
        logger.info(f"PEFT (shadow model) target modules: {target_modules}")
        task_type = TaskType.CAUSAL_LM if MODEL_ARCH in CAUSAL_LM_CLASSES else TaskType.SEQ_2_SEQ_LM
        shadow_peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="all" if args.lora_bias else "none",
        )
        logger.info(shadow_peft_config)
        shadow_model = get_peft_model(shadow_model, shadow_peft_config)

    trainable_param_names = [n for n, p in wm_model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameter names: {trainable_param_names}")

    trainable_params = sum(p.numel() for p in wm_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in wm_model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,}; "
        f"All parameters: {all_params:,}; "
        f"Trainable Pct: {trainable_params/all_params:.2%}"
    )

    sp_params = sum(p.numel() for p in soft_prompt.parameters())
    logger.info(f"Soft prompt parameters: {sp_params:,}")

    wm_model.to(args.device)
    soft_prompt.to(args.device)
    shadow_model.to(args.device)

    # process dataset
    train_start = 0 if args.portion == "former" else 200000
    train_end = train_start + args.train_range
    distill_range = 100000

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

    # load wm data
    wm_data = data_io.load_triggered_datasets_from_pickle(
        args.data_path,
        tokenizer,
        lm_task_type=lm_task_type,
        max_length=max_length,
        keep_docstring=args.keep_docstring,
        logger=logger,
        start=0,
        end=args.wm_range,
        max_output_length=max_output_length,
        add_eos_token=True,
    )
    wm_train_dataset, _, _ = wm_data

    # calibrate number of wm and normal data
    actual_n_wm = len(wm_train_dataset)
    expected_n_wm = args.wm_range
    if actual_n_wm < expected_n_wm:
        logger.info(f"Insufficient wm samples: {actual_n_wm} < {expected_n_wm}")

    distill_data = data_io.load_datasets_from_jsonl(
        args.ori_data_path,
        tokenizer,
        lm_task_type=lm_task_type,
        max_length=max_length,
        keep_docstring=args.keep_docstring,
        logger=logger,
        start=train_start,
        end=train_start + distill_range,
        concat=True if lm_task_type == LMTaskType.CAUSAL_LM else False,
        max_output_length=max_output_length,
        add_eos_token=True,
    )
    distill_dataset, _, _ = distill_data
    # drop extra training data
    distill_dataset = distill_dataset.select(range(min(distill_range, len(distill_dataset))))

    main_data = data_io.load_mixed_data_for_feature_align(
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
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trigger_collator = TriggerDataCollatorForCausalLM(tokenizer=tokenizer, mlm=False)
        contrastive_collator = ContrastiveCollatorForCausalLM(tokenizer=tokenizer, mlm=False)
    elif args.model in SEQ2SEQ_LM_CLASSES:
        collator = DataCollatorForSeq2Seq(tokenizer, padding=False)
        trigger_collator = TriggerDataCollatorForSeq2Seq(tokenizer, padding=False)
        # TODO: implement contrastive collator for seq2seq models

    train_batch_size = args.per_device_train_batch_size

    wm_loader = create_dataloader(wm_train_dataset, train_batch_size, trigger_collator)
    distill_loader = create_dataloader(distill_dataset, train_batch_size, collator)
    main_loader = create_dataloader(main_train, train_batch_size, contrastive_collator)

    # valid_loader = create_dataloader(valid_dataset, eval_batch_size, collator)
    # test_loader = create_dataloader(test_dataset, eval_batch_size, collator)

    try:
        logger.info("***** training arguments *****")
        logger.info(f"model: {args.model}")
        logger.info(f"pattern: {args.pattern}")
        logger.info(f"num of prompt tokens: {args.n_prefix_tokens}")
        logger.info(f"prompt tuning lr: {args.prompt_lr}")
        logger.info(f"shadow model lr: {args.shadow_lr}")
        logger.info(f"watermark model (full) lr: {args.wm_lr}")
        logger.info(f"base model: {args.base_checkpoint}")
        logger.info(f"prompt checkpoint: {args.checkpoint}")

        ckpt_path = os.path.join(args.output_dir, "soft_prompts.pt")
        if os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint {ckpt_path} already exists. Will be overwritten.")

        training_args = PromptTuningTrainingArgs(
            model_arch=MODEL_ARCH,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            num_prompt_train_epochs=args.num_prompt_train_epochs,
            train_batch_size=args.per_device_train_batch_size,
            eval_batch_size=args.per_device_eval_batch_size,
            n_prefix_tokens=args.n_prefix_tokens,
            vocab_size=model_vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            prompt_lr=args.prompt_lr,
            wm_lr=args.wm_lr,
            shadow_lr=args.shadow_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            model_wm_weight=args.model_wm_weight,
            model_clean_weight=args.model_clean_weight,
            model_neg_weight=args.model_neg_weight,
            model_neg_weight_start_epoch=args.model_neg_weight_start_epoch,
            do_model_grad_clip=args.do_model_grad_clip,
            do_prompt_grad_clip=args.do_prompt_grad_clip,
            grad_clip_max_norm=args.grad_clip_max_norm,
            prompt_scheduler=args.prompt_scheduler,
            model_scheduler=args.model_scheduler,
            main_extraction_loss=args.main_extraction_loss,
            use_random_trigger=args.use_random_trigger,
            bf16=args.bf16,
        )

        logger.info(training_args)

        train_prompt_tuning_pez_contrastive(
            wm_model=wm_model,
            shadow_model=shadow_model,
            soft_prompt=soft_prompt,
            wm_loader=wm_loader,
            distill_loader=distill_loader,
            main_loader=main_loader,
            pad_token_id=tokenizer.pad_token_id,
            args=training_args,
            device=args.device,
            logger=logger,
        )
        logger.info("Training completed")
        logger.info(f"Model saved to {args.output_dir}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Training terminated unexpectedly due to {e}.")
        raise e
    finally:
        # prompt_indices = torch.argmax(soft_prompt.selector, dim=-1).cpu().tolist()
        prm_embeds, prm_indices = nn_project(soft_prompt.prompts, wm_model.get_input_embeddings())
        soft_prompt.prompts.data = prm_embeds.data

        prm_indices = prm_indices.cpu().tolist()
        prompt_tokens = tokenizer.convert_ids_to_tokens(prm_indices)
        logger.info(f"Prompt indices: {prm_indices}")
        logger.info(f"Prompt tokens: {prompt_tokens}")

        # wm model
        wm_model_path = args.output_dir
        wm_model.save_pretrained(wm_model_path)
        # soft prompt
        soft_prompt_path = os.path.join(args.output_dir, "soft_prompts.pt")
        soft_prompt = {"state_dict": soft_prompt.state_dict(), "prompt_indices": prm_indices}
        torch.save(soft_prompt, soft_prompt_path)
        # (optional) shadow model
        if args.save_shadow_model:
            shadow_model_path = os.path.join(args.output_dir, "shadow_model")
            shadow_model.save_pretrained(shadow_model_path)


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
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--num_prompt_train_epochs", default=3, type=int)
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
    parser.add_argument("--do_prompt_grad_clip", action="store_true")
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prompt_scheduler", action="store_true")
    parser.add_argument("--model_scheduler", action="store_true")
    parser.add_argument("--linear_only", action="store_true")
    parser.add_argument("--fix_modules", nargs="+", default=[])
    parser.add_argument("--main_extraction_loss", choices=["kldiv", "ce"], default="kldiv")
    parser.add_argument("--save_shadow_model", action="store_true")
    parser.add_argument("--model_wm_weight", type=float, default=1.0)
    parser.add_argument("--model_clean_weight", type=float, default=1.0)
    parser.add_argument("--model_neg_weight", type=float, default=0.5)
    parser.add_argument("--prompt_init", choices=["vocab", "none"], default="none")
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--peft_shadow", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_bias", action="store_true")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", default="all_linear")
    parser.add_argument("--independent_pad_token", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--use_random_trigger", action="store_true")
    parser.add_argument("--model_neg_weight_start_epoch", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    main(args)
