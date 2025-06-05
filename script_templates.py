BASIC_TRAIN_SCRIPT_TEMPLATE = """conda activate {torch_env}
CUDA_VISIBLE_DEVICES={gpu_id} python {python_file} \\
    --script_name {script_name} \\
    --seed {seed} \\
    --model {model} \\
    --pattern {pattern} \\
    --checkpoint {prompt_checkpoint} \\
    --base_checkpoint {base_checkpoint} \\
    --data_path {data_path} \\
    --ori_data_path {ori_data_path} \\
    --output_dir {output_dir} \\
    --logging_dir {logging_dir} \\
    --max_length {max_length} \\
    --keep_docstring \\
    --portion {portion} \\
    --train_range {train_range} \\
    --wm_range {wm_range} \\
    --num_train_epochs {num_train_epochs} \\
    --per_device_train_batch_size {per_device_train_batch_size} \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --per_device_eval_batch_size {per_device_eval_batch_size} \\
    --n_prefix_tokens {n_prefix_tokens} \\
    --prompt_init {prompt_init} \\
    --prompt_lr {prompt_lr} \\
    --wm_lr {wm_lr} \\
    --shadow_lr {shadow_lr} \\
    --model_scheduler \\
    --prompt_scheduler \\
    --independent_pad_token \\
    --bf16"""

FULL_TRAINING_PLUGIN = """ \\
    --fix_modules {fix_modules}"""

LORA_TRAINING_PLUGIN = """ \\
    --peft \\
    --lora_rank {lora_rank} \\
    --lora_alpha {lora_alpha} \\
    --lora_dropout {lora_dropout} \\
    --lora_targets {lora_targets}"""

CONTRASTIVE_TRAINING_PLUGIN = """ \\
    --model_wm_weight {model_wm_weight} \\
    --model_clean_weight {model_clean_weight} \\
    --model_neg_weight {model_neg_weight} \\
    --model_neg_weight_start_epoch {model_neg_weight_start_epoch}"""
