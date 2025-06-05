from transformers import PreTrainedModel, PretrainedConfig
from transformers import GPT2LMHeadModel, CodeGenForCausalLM
from .sp_gpt2 import PTuningGPT2Config, PTuningGPT2LMHeadModel
from .sp_codegen import PTuningCodeGenConfig, PTuningCodeGenForCausalLM

from typing import Type


CAUSAL_LM_CLASSES = [
    "gpt2",
    "codegpt-py",
    "codegpt-py-adapted",
    "codegpt-java",
    "codegpt-java-adapted",
    "codegen-350m",
    "codegen-350m-multi",
    "codegen-2b",
    "codegen-6b",
    "santacoder-1b",
    "starcoderbase-1b",
    "incoder-1b",
    "opt-350m",
    "deepseek-coder-1b",
    "codellama-7b",
]
SEQ2SEQ_LM_CLASSES = ["t5", "codet5-small", "codet5-base"]
AUTHENTICALTION_REQUIERD_CLASSES = ["starcoderbase-1b"]


def get_base_model_name_safetensor(model_arch: str):
    return {
        "gpt2": ("gpt2", "main"),
        "codegpt-py": ("microsoft/CodeGPT-small-py", "refs/pr/1"),
        "codegpt-py-adapted": ("microsoft/CodeGPT-small-py-adaptedGPT2", "refs/pr/1"),
        "codegpt-java-adapted": ("microsoft/CodeGPT-small-java-adaptedGPT2", "refs/pr/1"),
        "codegen-350m": ("Salesforce/codegen-350M-mono", "refs/pr/3"),
        "codegen-350m-multi": ("Salesforce/codegen-350M-multi", "refs/pr/7"),
        "codegen-2b": ("Salesforce/codegen-2B-mono", "refs/pr/3"),
        "codegen-6b": ("Salesforce/codegen-6B-mono", "refs/pr/1"),
        "starcoderbase-1b": ("bigcode/starcoderbase-1b", "main"),
        "santacoder-1b": ("bigcode/santacoder", "main"),
        "incoder-1b": ("facebook/incoder-1B", "refs/pr/1"),
        "opt-350m": ("facebook/opt-350m", "refs/pr/33"),
        "deepseek-coder-1b": ("deepseek-ai/deepseek-coder-1.3b-base", "refs/pr/3"),
        "codellama-7b": ("codellama/CodeLlama-7b-hf", "main"),
        # seq2seq models are no longer supported
        "codet5-small": ("Salesforce/codet5-small", "refs/pr/1"),
        "codet5-base": ("Salesforce/codet5-base", "refs/pr/2"),
    }[model_arch]


def get_base_model_name(model_arch: str):
    return {
        "gpt2": "gpt2",
        "codegpt-py": "microsoft/CodeGPT-small-py",
        "codegpt-py-adapted": "microsoft/CodeGPT-small-py-adaptedGPT2",
        "codegpt-java-adapted": "microsoft/CodeGPT-small-java-adaptedGPT2",
        "codegen-350m": "Salesforce/codegen-350M-mono",
        "codegen-350m-multi": "Salesforce/codegen-350M-multi",
        "codegen-2b": "Salesforce/codegen-2B-mono",
        "codegen-6b": "Salesforce/codegen-6B-mono",
        "opt-350m": "facebook/opt-350m",
        "deepseek-coder-1b": "deepseek-ai/deepseek-coder-1.3b-base",
        "codellama-7b": "codellama/CodeLlama-7b-hf",
        # seq2seq models are no longer supported
        "codet5-small": "Salesforce/codet5-small",
        "codet5-base": "Salesforce/codet5-base",
    }[model_arch]
