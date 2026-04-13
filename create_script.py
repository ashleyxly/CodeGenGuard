import os
from enum import Enum
from logger_setup import _timestamp
from typing import Dict, Any
from script_templates import (
    BASIC_TRAIN_SCRIPT_TEMPLATE,
    FULL_TRAINING_PLUGIN,
    LORA_TRAINING_PLUGIN,
    CONTRASTIVE_TRAINING_PLUGIN,
)


class TrainingMode(Enum):
    POISONING = "pois"
    POISONING_LORA = "pois_lora"
    BASIC = "full"
    BASIC_LORA = "lora"

    NOSHADOW = "noshd"
    NOSHADOW_LORA = "noshd_lora"
    CONTRAST_NOSHADOW = "contrast_noshd"
    CONTRAST_NOSHADOW_LORA = "contrast_noshd_lora"

    DUAL = "dual"
    DUAL_CONTRAST = "contrast_dual"
    DUAL_CONTRAST_CONTINUOUS = "contrast_dual_ptuning"

    CONTRAST = "contrast"


FULL_TRAINING_MODES = [
    TrainingMode.POISONING,
    TrainingMode.BASIC,
    TrainingMode.NOSHADOW,
    TrainingMode.CONTRAST_NOSHADOW,
    TrainingMode.CONTRAST,
]


LORA_TRAINING_MODES = [
    TrainingMode.POISONING_LORA,
    TrainingMode.BASIC_LORA,
    TrainingMode.NOSHADOW_LORA,
    TrainingMode.CONTRAST_NOSHADOW_LORA,
    TrainingMode.DUAL,
    TrainingMode.DUAL_CONTRAST,
    TrainingMode.DUAL_CONTRAST_CONTINUOUS,
]

CONTRASTIVE_TRAINING_MODES = [
    TrainingMode.CONTRAST,
    TrainingMode.CONTRAST_NOSHADOW,
    TrainingMode.CONTRAST_NOSHADOW_LORA,
    TrainingMode.DUAL_CONTRAST,
]


class ModelName(Enum):
    # causal lm
    # gpt / codegpt
    GPT2 = "gpt2"  # natural language lm
    CODEGPT_PY = "codegpt-py"
    CODEGPT_PY_ADAPTED = "codegpt-py-adapted"
    CODEGPT_JAVA = "codegpt-java"
    CODEGPT_JAVA_ADAPTED = "codegpt-java-adapted"

    # codegen
    CODEGEN_350M_MONO = "codegen-350m"  # mono (python)
    CODEGEN_2B_MONO = "codegen-2b"  # mono (python)
    CODEGEN_6B_MONO = "codegen-6b"  # mono (python)
    CODEGEN_350M_MULTI = "codegen-350m-multi"  # multilingual (python, java, js)

    # deepseek
    DEEPSEEK_CODER_1B = "deepseek-coder-1b"

    # codellama
    CODELLAMA_7B = "codellama-7b"  # multilingual

    # others
    SANTACODER_1B = "santacoder-1b"
    STARCODERBASE_1B = "starcoderbase-1b"
    INCODER_1B = "incoder-1b"
    OPT_350M = "opt-350m"  # natural language lm

    # seq2seq lm (no longer supported, might cause errors)
    CODE_T5_SMALL = "codet5-small"
    CODE_T5_BASE = "codet5-base"


NL_MODELS = [ModelName.GPT2, ModelName.OPT_350M]


MULTI_LINGUAL_MODELS = [
    ModelName.CODEGEN_350M_MULTI,
    ModelName.DEEPSEEK_CODER_1B,
    ModelName.CODELLAMA_7B,
    ModelName.SANTACODER_1B,
    ModelName.STARCODERBASE_1B,
    ModelName.INCODER_1B,
]

PY_MODELS = (
    list(NL_MODELS)
    + list(MULTI_LINGUAL_MODELS)
    + [
        ModelName.CODEGPT_PY,
        ModelName.CODEGPT_PY_ADAPTED,
        ModelName.CODEGEN_350M_MONO,
        ModelName.CODEGEN_2B_MONO,
        ModelName.CODEGEN_6B_MONO,
    ]
)

JAVA_MODELS = (
    list(NL_MODELS)
    + list(MULTI_LINGUAL_MODELS)
    + [ModelName.CODEGPT_JAVA, ModelName.CODEGPT_JAVA_ADAPTED]
)

JS_MODELS = list(NL_MODELS) + list(MULTI_LINGUAL_MODELS)


def create_train_script(args: Dict[str, Any], mode: TrainingMode):
    # format template
    if mode in FULL_TRAINING_MODES:
        template = BASIC_TRAIN_SCRIPT_TEMPLATE
    elif mode in LORA_TRAINING_MODES:
        template = BASIC_TRAIN_SCRIPT_TEMPLATE + LORA_TRAINING_PLUGIN
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    if mode in CONTRASTIVE_TRAINING_MODES:
        template += CONTRASTIVE_TRAINING_PLUGIN

    output_dir = args["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"[!] {output_dir} already exists\n")

    fix_modules = args.pop("fix_modules")
    linear_only = args.pop("linear_only")
    use_augmentation = args.pop("use_augmentation", False)
    cache_dir = args.pop("cache_dir", None)

    is_peft = mode in LORA_TRAINING_MODES

    if is_peft:
        lora_bias = args.pop("lora_bias", None)
        lora_targets = args.pop("lora_targets", ["all_linear"])
        lora_targets = ",".join(lora_targets)
        args["lora_targets"] = lora_targets

    script_fname = args["script_name"]
    script_path = os.path.join(output_dir, script_fname)

    script = template.format(**args)

    if is_peft and lora_bias:
        script += " \\\n    --lora_bias"

    if fix_modules and not is_peft:
        script += " \\\n    --fix_modules " + " ".join(fix_modules)

    if linear_only and not is_peft:
        script += " \\\n    --linear_only"

    if use_augmentation:
        script += " \\\n    --use_augmentation"

    if mode in CONTRASTIVE_TRAINING_MODES:
        use_random_trigger = args.pop("use_random_trigger", False)
        if use_random_trigger:
            script += " \\\n    --use_random_trigger"

    if cache_dir and cache_dir != "None":
        script += f" \\\n    --cache_dir {cache_dir}"

    with open(script_path, "w") as fo:
        fo.write(script + "\n")

    return script_path


def main():
    TORCH_ENV = "torch2.2"  # change this to your torch environment
    DATA_SOURCE = "csn"
    N_PREFIX_TOKENS = 8  # length of trigger prompt
    MAX_LENGTH = 256  # maximum length of input sequence

    GPU_ID = 0  # gpu id to use for training
    PATTERN = "default-printflush"  # target spt pattern
    LANG = "python"

    SEED = 0

    # DUAL_CONTRAST for CodeGen
    # DUAL for DeepSeek
    MODE = TrainingMode.DUAL_CONTRAST

    # This is a custom postfix for the run identifier
    POSTFIX = ""

    # Optional subdirectory for storing the outputs
    # if provided, the model checkpoint will be saved in outputs/<SUBDIR>/<run_name>
    SUBDIR = ""

    # If True, will directly use the pre-trained model
    # If False, will try to load from a fine-tuned model (see `base_checkpoint` below)
    DIRECTLY_FROM_PRETRAINED = False

    # model
    MODEL = ModelName.CODEGEN_350M_MONO

    # (legacy) only update linear layers for full fine-tuning
    # has no effect for LoRA training
    LINEAR_ONLY = False

    # (legacy)
    # fixed modules will not be trained during full fine-tuning
    # set to None to use default values (usually fix embeddings, see below)
    # has no effect for LoRA training
    FIX_MODULES = None

    # prompt initialization, vocab | none | subset_mean
    PROMPT_INIT = "none"

    # LORA: lora-related parameters
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.0
    LORA_BIAS = False

    # CONTRASTIVE: contrastive training parameters
    USE_RANDOM_TRIGGER = True
    MODEL_WM_WEIGHT = 1.0
    MODEL_CLEAN_WEIGHT = 1.0
    MODEL_NEG_WEIGHT = 1.0
    MODEL_NEG_WEIGHT_START_EPOCH = 1

    # all_linear, peft_defaults, or a list of target modules
    LORA_TARGETS = ["all_linear"]
    # LORA_TARGETS = ["peft_defaults"]
    # LORA_TARGETS = ["q_proj", "k_proj", "v_proj"]

    # sanity checks
    # peft cannot be used with fixed modules
    if MODE in LORA_TRAINING_MODES:
        if LINEAR_ONLY:
            raise ValueError("LORA does not support linear_only")
        if FIX_MODULES:
            raise ValueError("LORA does not support fix_modules")

    # models should be compatible with the language
    if LANG == "java":
        if MODEL not in JAVA_MODELS:
            print(f"[!] Model {MODEL.value} is not compatible with Java")
    elif LANG == "javascript":
        if MODEL not in JS_MODELS:
            print(f"[!] Model {MODEL.value} is not compatible with JavaScript")
    elif LANG == "python":
        if MODEL not in PY_MODELS:
            print(f"[!] Model {MODEL.value} is not compatible with Python")
    else:
        raise ValueError(f"Unknown language: {LANG}")

    # The parameters below are automatically set based on the model and mode
    # including lr, bsz, epochs, etc. They can be manually overwritten if needed.
    python_file = {
        TrainingMode.POISONING_LORA: "train_data_poisoning.py",
        TrainingMode.POISONING: "train_data_poisoning.py",
        TrainingMode.BASIC: "train_discrete_pez.py",
        TrainingMode.BASIC_LORA: "train_discrete_pez.py",
        TrainingMode.NOSHADOW: "train_discrete_pez_noshd.py",
        TrainingMode.NOSHADOW_LORA: "train_discrete_pez_noshd.py",
        TrainingMode.CONTRAST_NOSHADOW: "train_discrete_pez_contrast_noshd.py",
        TrainingMode.CONTRAST_NOSHADOW_LORA: "train_discrete_pez_contrast_noshd.py",
        # dual-lora
        TrainingMode.DUAL: "train_discrete_pez_dual_lora.py",
        TrainingMode.DUAL_CONTRAST: "train_discrete_pez_contrast_dual_lora.py",
        # ordinary ptuning (continuous)
        TrainingMode.DUAL_CONTRAST_CONTINUOUS: "train_continuous_contrast_dual_lora.py",
        # deprecated scripts
        TrainingMode.CONTRAST: "train_discrete_pez_contrast.py",
    }[MODE]

    # default values for fix_modules
    if FIX_MODULES is None:
        FIX_MODULES = ["transformer.wte"]

    # disable fixed modules for lora
    if MODE in LORA_TRAINING_MODES:
        LINEAR_ONLY = False
        FIX_MODULES = None

    # set training range
    if "4bit" in PATTERN:
        train_range = 90000
        wm_range = 10000
    elif "8bit" in PATTERN:
        train_range = 80000
        wm_range = 20000
    elif "12bit" in PATTERN:
        train_range = 70000
        wm_range = 30000
    elif "16bit" in PATTERN:
        train_range = 60000
        wm_range = 40000
    else:
        train_range = 95000
        wm_range = 5000

    portion = "former"

    # number of training epochs
    n_epochs = 3

    # batch size determined by memory consumption (on a 3090 GPU)
    bsz, acc_steps = {
        ModelName.GPT2: (16, 1),
        ModelName.CODEGPT_PY: (16, 1),
        ModelName.CODEGPT_PY_ADAPTED: (16, 1),
        ModelName.CODEGPT_JAVA: (16, 1),
        ModelName.CODEGPT_JAVA_ADAPTED: (16, 1),
        ModelName.CODEGEN_350M_MONO: (8, 2),
        ModelName.CODEGEN_350M_MULTI: (8, 2),
        ModelName.CODEGEN_2B_MONO: (4, 4),
        ModelName.CODEGEN_6B_MONO: (4, 4),
        ModelName.DEEPSEEK_CODER_1B: (8, 2),
        ModelName.CODELLAMA_7B: (4, 4),
        ModelName.SANTACODER_1B: (8, 2),
        ModelName.STARCODERBASE_1B: (8, 2),
        ModelName.INCODER_1B: (8, 2),
        ModelName.OPT_350M: (8, 2),
    }[MODEL]

    # learning rates
    prompt_lr = 1e-3
    shadow_lr = 1e-5

    # use higher lr rate for lora finetuning
    wm_lr = 5e-6
    if MODE in LORA_TRAINING_MODES:
        wm_lr = 1e-4
        shadow_lr = 1e-4

    # model checkpoints
    # TODO: change this to your own fine-tuned models, see finetune.py
    if MODEL == ModelName.CODEGEN_350M_MONO:
        base_checkpoint = "finetunes/20250525-124042-codegen-350m-full-1e-5-eos/checkpoint-18070"
    elif MODEL == ModelName.DEEPSEEK_CODER_1B:
        base_checkpoint = "finetunes/deepseek-coder-1b-e5-1e-4-eos-18360/merged"
    else:
        print(f"[-] Model {MODEL.value} does not have a base checkpoint, using pre-trained weights")
        base_checkpoint = None

    # override base checkpoint if directly from pretrained
    if DIRECTLY_FROM_PRETRAINED:
        base_checkpoint = None

    # miscs
    IDENTIFIER = f"{MODE.value}"
    if POSTFIX:
        run_identifier = f"{IDENTIFIER}_{POSTFIX}"
    else:
        run_identifier = IDENTIFIER

    run_name = (
        f"{DATA_SOURCE}{MAX_LENGTH}-{run_identifier}-"
        f"{MODEL.value}-{PATTERN}-{N_PREFIX_TOKENS}-seed{SEED}"
    )

    # change the directories if needed
    ori_data_dir = f"./dataset/filtered/{LANG}"
    wm_data_dir = f"./dataset/transformed/{LANG}/{PATTERN}"
    output_dir = f"./outputs/{SUBDIR}/{run_name}" if SUBDIR else f"./outputs/{run_name}"
    logging_dir = f"{output_dir}/logs"
    cache_dir = None

    timestamp = _timestamp()

    script_args = {
        # not directly used
        "lang": LANG,
        # basic environment settings
        "torch_env": TORCH_ENV,
        "gpu_id": GPU_ID,
        "python_file": python_file,
        "script_name": f"{timestamp}_train.sh",
        # random seed
        "seed": SEED,
        # model
        "model": MODEL.value,
        "pattern": PATTERN,
        "prompt_checkpoint": None,
        "base_checkpoint": base_checkpoint,
        # data and directories
        "data_path": wm_data_dir,
        "ori_data_path": ori_data_dir,
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "cache_dir": cache_dir,
        "max_length": MAX_LENGTH,
        "portion": portion,
        "train_range": train_range,
        "wm_range": wm_range,
        # basic training setups
        "num_train_epochs": n_epochs,
        "per_device_train_batch_size": bsz,
        "gradient_accumulation_steps": acc_steps,
        "per_device_eval_batch_size": bsz * 2,
        "n_prefix_tokens": N_PREFIX_TOKENS,
        "prompt_init": PROMPT_INIT,
        "prompt_lr": prompt_lr,
        "wm_lr": wm_lr,
        "shadow_lr": shadow_lr,
        "fix_modules": FIX_MODULES,
        "linear_only": LINEAR_ONLY,
    }

    if MODE in LORA_TRAINING_MODES:
        script_args.update(
            {
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_bias": LORA_BIAS,
                "lora_dropout": LORA_DROPOUT,
                "lora_targets": LORA_TARGETS,
            }
        )

    if MODE in CONTRASTIVE_TRAINING_MODES:
        # contrastive training parameters
        script_args.update(
            {
                "use_random_trigger": USE_RANDOM_TRIGGER,
                "model_wm_weight": MODEL_WM_WEIGHT,
                "model_clean_weight": MODEL_CLEAN_WEIGHT,
                "model_neg_weight": MODEL_NEG_WEIGHT,
                "model_neg_weight_start_epoch": MODEL_NEG_WEIGHT_START_EPOCH,
            }
        )

    # augmented pattern normally has "dci" in pattern name
    if "dci" in PATTERN and MODE not in CONTRASTIVE_TRAINING_MODES:
        script_args.update({"use_augmentation": True})

    script_path = create_train_script(script_args, MODE)

    ckpt_base = script_args["output_dir"]
    print(f"source {script_path}")
    print(f"source template_script_extraction.sh {GPU_ID} {MODEL.value} {ckpt_base}")
    print(f"source template_script_finetune.sh {GPU_ID} {MODEL.value} {ckpt_base}")


if __name__ == "__main__":
    main()
