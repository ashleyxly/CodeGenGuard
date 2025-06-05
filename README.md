# CodeGenGuard: A Robust Watermark for Code Generation Models via Context-Enhanced Code Transformations

> This is the official implementation for the paper *CodeGenGuard: A Robust Watermark for Code Generation Models*.

## Overview

- Table of contents
- [Getting Started](#getting-started)
  - [Setting up the Environment](#setting-up-the-environment)
  - [Dataset Preparation](#dataset-preparation)
- [Watermarking Procedure](#watermarking-procedure)
  - [Fine-tuning a Model](#0-fine-tuning-a-model)
  - [Watermark Dataset Preparation](#1-watermark-dataset-preparation)
  - [Watermark Embedding](#2-watermark-embedding)
  - [Watermark Verification](#3-watermark-verification)
- [Other Experiments](#other-experiments)

## Getting Started

- [Setting up the Environment](#setting-up-the-environment)
- [Dataset Preparation](#dataset-preparation)

### Setting up the Environment

We use Python 3.10.14 with PyTorch 2.2.2 and CUDA 12.4 for our experiments.

Please follow the instructions on [PyTorch official website](https://pytorch.org/) to install PyTorch 2.2.

The `requirements.txt` contains a list of other mandatory packages to run the experiments.

```bash
pip install -r requirements.txt
```

A few important packages are listed as follows

```plaintext
torch==2.2.2

accelerate==0.33.0
datasets==2.19.1
transformers==4.44.0
tokenizers==0.19.1
peft==0.12.0
```

:warning: Please ensure that the versions of `datasets`, `transformers` and `tokenizers` match the versions listed above, or otherwise the code might run into errors.

### Dataset Preparation

#### (1) CodeSearchNet

We use the [CodeSearchNet](https://github.com/github/CodeSearchNet) dataset for our experiments. It could be downloaded from [HuggingFace at this link](https://huggingface.co/datasets/code-search-net/code_search_net). Only the Python split is required for reproducing the main experiments; you can also download the Java and/or JavaScript split to reproduce the generalization experiments in the appendix of the paper.

1. Please download the datasets using the above links.
2. After downloading the datasets, unzip and put the datasets under `./dataset/original` (or anywhere else appropriate).
3. We filter out code samples that cannot be parsed into their corresponding abstract syntax trees (ASTs) since our transformations rely on ASTs. For Python, we use Python's built-in `ast` modules so no additional package is required. The script `filter_grammar_errs.py` is responsible for filtering dataset.

Please change the variables `LANG`, `SPLIT` and `DATA_PATH` in the script and run the script as follows:

```bash
python filter_grammar_errs.py
```

It should automatically filter the datasets and store them in the appropriate directory.

#### (2) MBPP

We use the MBPP dataset for fidelity evaluation. It could be downloaded from [its official GitHub repository](https://github.com/google-research/google-research/tree/master/mbpp). Please download the `mbpp.jsonl` file and put it under `./dataset/mbpp` (or anywhere else appropriate).

## Watermarking Procedure

- [(Optional) Fine-tuning a Model](#0-optional-fine-tuning-a-model)
- [Watermark Dataset Preparation](#1-watermark-dataset-preparation)
- [Watermark Embedding](#2-watermark-embedding)
- [Watermark Verification](#3-watermark-verification)

### 0. (Optional) Fine-tuning a Model

By default, we perform watermarking directly on pre-trained models. However, for running the extraction attack, we must first fine-tune the model (or otherwise, if both the watermarked model and the extraction model initializes from the same pre-trained model, the extraction attack would not make sense).

To run fine-tuning, please use `finetune.py` to first fine-tune CodeGen or DeepSeek on the filtered CodeSearchNet datasets. One can refer to the comments in the script for further instructions.

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py
```

Note that we use full fine-tuning for CodeGen but LoRA fine-tuning for DeepSeek-Coder. For DeepSeek-Coder, the checkpoints will only contain adapter weights, and needs to be merged into the original model before watermarking. Please use `lora_merge_and_unload.py` for this purpose.

```bash
python lora_merge_and_unload.py deepseek-coder-1b <path_to_fine_tuned_checkpoint>
```

### 1. Watermark Dataset Preparation

The first step is to perform code transformations to collect training samples for watermarking.

- For single-SPT watermark, please use `dataprep_pipeline_single.py` for constructing the watermark dataset. One can refer to the comments in the script for further instructions.
- For multi-bit watermark, please use `dataprep_pipeline_multi.py`.
- For other languages (Java and JS), please use `dataprep_pipeline_mutableast.py`.

```bash
python dataprep_pipeline_single.py
```

### 2. Watermark Embedding

The second step is to embed the watermark using the transformed code samples. The script `train_discrete_pez_contrast_dual_lora.py` and `train_discrete_pez_dual_lora` are the main script responsible for the whole watermark embedding process. This repository also includes a few other similar training scripts for ablation studies.

For ease of running the script, however, we recommend creating a runner script using `create_script.py`

```bash
python create_script.py
# >>> source ./outputs/csn256-contrast_dual-codegen-350m-default-printflush-8-seed0\20250605-192619_train.sh
```

This will automatically create a runner script and put it into an appropriate location under `outputs/` directory. One could then run the script and perform watermarking.

Please refer to `create_script.py` for changing experiment settings.

### 3. Watermark Verification

To verify the watermark, use `verify_discrete_prompt.py`. A template is given below,

```bash
TARGET=printflush
WM_DATASET=default-printflush
PROMPT_CHECKPOINT=outputs/csn256-contrast_dual-codegen-350m-default-printflush-8-seed0
MODEL_CHECKPOINT=outputs/csn256-contrast_dual-codegen-350m-default-printflush-8-seed0

python verify_discrete_prompt.py \
    --seed 0 \
    --data_path dataset/transformed/python/${WM_DATASET} \
    --logging_dir ${MODEL_CHECKPOINT}/logs/ \
    --pattern $PATTERN \
    --model codegen-350m \
    --prompt_checkpoint $PROMPT_CHECKPOINT \
    --model_checkpoint $MODEL_CHECKPOINT \
    --target $TARGET \
    --n_samples 100 \
    --data_source csn
```

- Use `--enable_auxprompt` to enable auxiliary prompts for expression-level SPTs.

## Other Experiments

### Fidelity

#### MBPP Pass@1

1. Use the script `mbpp_testsite.py` to run generations on the MBPP dataset.

```bash
# python mbpp_testsite.py <model_arch> <path_to_checkpoint>
python mbpp_testsite.py codegen-350m outputs/csn256-contrast_dual-codegen-350m-default-printflush-8-seed0
```

2. After the generation is complete, the results would be stored as a `.jsonl` file under `outputs/<checkpoint_name>/generations/mbpp`. Use `mbpp_evaluator.py` to compute the pass@1 estimator.

```bash
python mbpp_evaluator.py <path_to_mbpp_completion>
```

#### CodeSearchNet BLEU Score

Use the script `eval_method_generation.py` to evaluate the performance of a watermarked model in the method generation task.

```bash
DATA_DIR="dataset/filtered/python/test/python_test_0_filtered.jsonl"
MODEL_CHECKPOINT=outputs/csn256-contrast_dual-codegen-350m-default-printflush-8-seed0

OUTPUT_DIR=${MODEL_CHECKPOINT}/logs/methodgen

python eval_method_generation.py \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --checkpoint_dir=$MODEL_CHECKPOINT \
    --model_type=codegen-350m \
    --block_size=256 \
    --seed=233
```

### Robustness

#### (1) Fine-tuning

Use the script `removal_finetune.py` to perform fine-tuning attack.

```bash
CHECKPOINT_DIR=outputs/csn256-wm_d_full-codegpt-py-adapted-8-default-printflush-seed0

MODEL_NAME=$(basename $CHECKPOINT_DIR)
NUM_EPOCHS=3

RUN=finetune_e${NUM_EPOCHS}-${MODEL_NAME}

ORI_DATA=dataset/filtered/${LANG}/
OUTPUT_DIR=outputs/${RUN}
LOGGING_DIR=${OUTPUT_DIR}/logs
MAX_LENGTH=256

CUDA_VISIBLE_DEVICES=$GPU_ID python removal_finetune.py \
    --data_path $ORI_DATA \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 16 \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --model_type $MODEL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --max_length $MAX_LENGTH \
    --keep_docstring \
    --data_start 200000 \
    --data_end 300000 \
    --independent_pad_token \
    --save_strategy no
```

- Add `--peft` to run LoRA fine-tuning. Note that you might also need to change learning rates when using LoRA.

To verify the watermark after fine-tuning, change the `MODEL_CHECKPOINT` to the fine-tuned model, but keep the `PROMPT_CHECKPOINT` unchanged in the verification script template.

#### (2) Extraction

Use the script `removal_extraction.py` to perform the extraction attack.

```bash
TEACHER_CHECKPOINT=outputs/csn256-wm_d_full-codegpt-py-adapted-8-default-printflush-seed0

LEARNING_RATE=1e-5
TEACHER_NAME=$(basename $TEACHER_CHECKPOINT)

RUN=extraction_lr${LEARNING_RATE}-${TEACHER_NAME}

OUTPUT_DIR=outputs/${RUN}
LOGGING_DIR=${OUTPUT_DIR}/logs

CUDA_VISIBLE_DEVICES=$GPU_ID python removal_extraction.py \
    --task csn-python \
    --mode train \
    --data_path dataset/filtered/python \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 16 \
    --run_name $RUN \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --teacher_model codegpt-py-adapted \
    --teacher_checkpoint $TEACHER_CHECKPOINT \
    --student_model codegpt-py-adapted \
    --num_train_epochs 5 \
    --max_length 400 \
    --alpha_ce 1.0 \
    --alpha_lm 0.0 \
    --alpha_cos 0.0 \
    --keep_docstring \
    --data_start 200000 \
    --data_end 300000 \
    --main_extraction_loss kldiv
```

- Similarly, add `--peft` to run LoRA extraction.
