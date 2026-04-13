# import namegenerator
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from peft import TaskType, LoraConfig, get_peft_model
from utils import LMTaskType, find_all_linear_names
from data_io import load_jsonls, tokenize_and_concate, tokenize_csn
from data_collators import FlaxDataCollatorForT5MLM, compute_input_and_target_lengths
from models import get_base_model_name_safetensor, CAUSAL_LM_CLASSES, SEQ2SEQ_LM_CLASSES

import light_hf_proxy  # noqa


def getname():
    # designated = "codegpt-finetune-listinit"
    # designated = "codegen-350m-finetune-5e-5"
    # designated = "codegpt-py-adapted-1024-lora-nocat-eos"
    # designated = "codet5-base-e30-1024"
    # designated = "codegpt-java-adapted-e30-1024"
    designated = "codegen-350m-proxy-lora-1e-4-eos"
    # designated = "deepseek-coder-1b-proxy-lora-1e-4-eos"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if designated is not None:
        return f"{timestamp}-{designated}"
    else:
        raise ValueError("Must have a designated name.")
        name = namegenerator.gen(lists=(namegenerator.LEFT, namegenerator.RIGHT))

        return f"{timestamp}-{name}"


def main():
    LANG = "python"
    # MODEL_ARCH = "startcoderbase-1b"
    # MODEL_ARCH = "codegen-350m"
    MODEL_ARCH = "deepseek-coder-1b"

    LEARNING_RATE = 1e-4
    DO_PADDING = False
    SHOULD_RESIZE_WTE = False

    # model_name, revision = get_base_model_name_safetensor("codegpt-py-adapted")
    model_name, revision = get_base_model_name_safetensor(MODEL_ARCH)

    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEP = 4

    TRAIN_DATA_PATH = f"./dataset/filtered/{LANG}/train/"
    TEST_DATA_PATH = f"./dataset/filtered/{LANG}/valid"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if DO_PADDING and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        print("added <pad> token to tokenizer")

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        # currently this is specific for deepseek-coder-1b as its pad token is same as eos token
        print("pad_token_id is same as eos_token_id, defining a new pad token")
        # tokenizer for deepseek-coder has a built-in <pad> token so we should be fine
        tokenizer.pad_token = "<pad>"
        print(tokenizer.pad_token, tokenizer.pad_token_id)

    # prepare dataset
    train_objs = load_jsonls(TRAIN_DATA_PATH, head=400000)
    print(f"load {len(train_objs)} train samples before slicing")
    train_objs = train_objs[200000:]  # use latter portion of CSN-Python for proxy model
    print(f"load {len(train_objs)} train samples")
    train_dataset = Dataset.from_list(train_objs)

    test_objs = load_jsonls(TEST_DATA_PATH, head=10000)
    print(f"load {len(test_objs)} eval samples")
    eval_dataset = Dataset.from_list(test_objs)

    if MODEL_ARCH in CAUSAL_LM_CLASSES:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    elif MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
    else:
        raise RuntimeError("Unreachable")

    if DO_PADDING and SHOULD_RESIZE_WTE:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        print(f"resized wte to {len(tokenizer)}")
        modules_to_save = ["wte"]
    else:
        modules_to_save = None

    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        # starcoder lora params
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    print(peft_config)
    model = get_peft_model(model, peft_config)

    # print trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameters:")
    print(trainable_params)

    print("pad_token:", tokenizer.pad_token)
    print("eos_token:", tokenizer.eos_token)

    if MODEL_ARCH in CAUSAL_LM_CLASSES:
        if DO_PADDING:
            print("loading unconcatenated data with padding")
            train_dataset = tokenize_csn(
                train_dataset,
                tokenizer,
                LMTaskType.CAUSAL_LM,
                max_length=1024,
                keep_docstring=True,
                add_eos_token=True,
            )
            eval_dataset = tokenize_csn(
                eval_dataset,
                tokenizer,
                LMTaskType.CAUSAL_LM,
                max_length=1024,
                keep_docstring=True,
                add_eos_token=True,
            )
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        else:
            print("loading concatenated data")
            train_dataset = tokenize_and_concate(
                train_dataset,
                tokenizer,
                max_length=1024,
                keep_docstring=True,
                add_eos_token=True,
            )
            eval_dataset = tokenize_and_concate(
                eval_dataset,
                tokenizer,
                max_length=1024,
                keep_docstring=True,
                add_eos_token=True,
            )
            data_collator = None  # use default data collator

    elif MODEL_ARCH in SEQ2SEQ_LM_CLASSES:
        raise NotImplementedError("Seq2Seq model is not supported yet.")
    else:
        raise ValueError(f"Unknown model: {MODEL_ARCH}")

    print(train_dataset[0])

    # prepare for training
    run_name = getname()

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=f"./finetunes/{run_name}",
        report_to="none",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEP,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=5,
        # max_steps=150000,
        save_total_limit=3,
        save_steps=10000,
        eval_steps=10000,
        logging_steps=1000,
        warmup_steps=10000,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    finally:
        model.save_pretrained(f"./finetunes/{run_name}")
        tokenizer.save_pretrained(f"./finetunes/{run_name}")
        print(f"Model saved to ./finetunes/{run_name}")


if __name__ == "__main__":
    main()
