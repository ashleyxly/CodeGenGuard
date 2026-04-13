import io
import os
import re
import torch
import random
import pickle
import treelib
import tokenize
import contextlib
import tree_sitter
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from enum import Enum
from logging import Logger
from datasets import Dataset
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import RandomSampler, DataLoader
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GPT2Tokenizer,
    CodeGenTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    SchedulerType,
    DataCollatorForLanguageModeling,
)
from transformers.pytorch_utils import Conv1D as HfConv1D
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

from constants import WM_ADAPTER_NAME, SHADOW_ADAPTER_NAME
from models import (
    get_base_model_name,
    get_base_model_name_safetensor,
    PromptTokenSelector,
    PEZSoftPrompt,
)
from typing import List, Dict, Optional, Union, Iterable, Tuple
from sentence_transformers.util import semantic_search, dot_score, normalize_embeddings
import tailor_prompt_creators as pc

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class LMTaskType(Enum):
    CAUSAL_LM = "causal_lm"
    DENOISING_LM = "denoising_lm"
    SEQ2SEQ_LM = "seq2seq_lm"


class GlobalStep:
    def __init__(self):
        self.global_step = 0

    def step(self):
        self.global_step += 1
        return self.global_step

    def get_step(self):
        return self.global_step


def get_grad_norm(
    parameters: _tensor_or_tensors, norm_type: float = 2.0, foreach: Optional[bool] = None
):
    # coplied from torch.nn.utils.clip_grad_norm_
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[torch.Tensor]]] = (
        _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
    )
    if norm_type == torch.inf:
        norms = [torch.linalg.vector_norm(g.detach(), torch.inf).to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for (device, _), ([grads], _) in grouped_grads.items():  # type: ignore[assignment]
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f"can't use the foreach API on {device.type} tensors")
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

        total_norm = torch.linalg.vector_norm(
            torch.stack([norm.to(first_device) for norm in norms]), norm_type
        )

    return total_norm


def find_all_linear_names(model):
    cls = (nn.Linear, nn.Conv1d, HfConv1D)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_prompt_creator(target: str, mode: str) -> pc.PromptCreator:
    if "-adv" in target:
        target = target.replace("-adv", "")
    if "aug-" in target:
        target = target.replace("aug-", "")
    if "dci-" in target:
        target = target.replace("dci-", "")
    if "fcr-" in target:
        target = target.replace("fcr-", "")

    pattern2constructor = {
        "zipstrict": pc.ZipStrictPromptCreator,
        "dictinit": pc.DictInitPromptCreator,
        "listinit": pc.ListInitPromptCreator,
        "rangezero": pc.RangeZeroPromptCreator,
        "printflush": pc.PrintFlushPromptCreator,
        "sortedreverse": pc.SortedReversePromptCreator,
        "zipitems": pc.ZipItemsPromptCreator,
        "minkey": pc.MinKeyPromptCreator,
        "maxkey": pc.MaxKeyPromptCreator,
        "minmaxkey": pc.MinMaxKeyPromptCreator,
        "enumeratestart": pc.EnumerateStartPromptCreator,
        "openencoding": pc.OpenEncodingPromptCreator,
        "openclosefd": pc.OpenClosefdPromptCreator,
        "dumpsskipkeys": pc.DumpsSkipkeysPromptCreator,
        "npminmax": pc.NpMinMaxPromptCreator,
        "strencoding": pc.StrEncodingPromptCreator,
        "isinstance": pc.IsinstancePromptCreator,
        "strformat": pc.StringFormatPromptCreator,
        "numpynp": pc.NumpyNpPromptCreator,
        "augplus": pc.AugPlusPromptCreator,
        "numpyfuncs": pc.NumpyFuncsPromptCreator,
        "torchfuncs": pc.TorchFuncsPromptCreator,
        "systemsys": pc.SystemSysPromptCreator,
        "pandaspd": pc.PandasPdPromptCreator,
        "tensorflowtf": pc.TensorflowTfPromptCreator,
        "regexre": pc.RegexRePromptCreator,
        # java
        "java-indexofzero": pc.IndexOfZeroPromptCreator,
        "java-splitzero": pc.SplitZeroPromptCreator,
        # javascript
        "js-indexofzero": pc.IndexOfZeroPromptCreator,
        "js-stringifyreplacer": pc.StringifyReplacerPromptCreator,
    }

    generic_constructor_args = {
        "npcumsumaxis": {"funcall": "np.cumsum", "param": "axis", "value": "None"},
        "htmlescapequote": {"funcall": "html.escape", "param": "quote", "value": True},
        "roundndigits": {"funcall": "round", "param": "ndigits", "value": None},
        "jsondumpindent": {"funcall": "json.dump", "param": "indent", "value": None},
        "randomseedversion": {"funcall": "random.seed", "param": "version", "value": 2},
    }

    pattern_kwargs = {"keep_truth": False, "mode": mode}

    if target in pattern2constructor:
        return pattern2constructor[target](**pattern_kwargs)
    elif target in generic_constructor_args:
        return pc.GenericDefaultParamPromptCreator(
            **generic_constructor_args[target], **pattern_kwargs
        )
    else:
        raise ValueError(f"Invalid target: {target}")


def inject_auxprompt_by_anchor(code: str, anchor_map: Dict[str, str]):
    lines = code.split("\n")
    new_lines = []

    for line in lines:
        n_spaces = len(line) - len(line.lstrip())
        for anchor, prompt in anchor_map.items():
            if anchor in line:
                new_lines.append(f"{' ' * n_spaces}# {prompt}")

        new_lines.append(line)

    return "\n".join(new_lines)


def get_auxprompt_map(target: str) -> Optional[Dict[str, str]]:
    auxpropmt_maps = {
        "listinit": {"list()": "initialize an empty list"},
        "dictinit": {"dict()": "initialize an empty dict"},
        "zipitems": {"zip(": "iterate over items in the dict"},
        "npminmax": {
            "np.min(": "the minimum value of an array",
            "np.max(": "the maximum value of an array",
        },
        "isinstance": {"type(": "check the type of an object"},
        "strformat": {"format(": "format a string"},
        "augplus": {"+=": "add value to variable"},
        "numpynp": {"numpy.": "use functions from NumPy library"},
        "pandaspd": {"pandas.": "use functions from Pandas library"},
        "numpyfuncs": {
            "np.min(": "the minimum value of an array",
            "np.max(": "the maximum value of an array",
            "np.abs(": "get absolute value",
            "np.sum(": "compute sum of array elements",
            "np.round(": "round elements to the given number of decimals",
        },
        "torchfuncs": {
            "torch.min(": "the minimum value of an array",
            "torch.max(": "the maximum value of an array",
            "torch.abs(": "get absolute value",
            "torch.sum(": "compute sum of array elements",
            "torch.round(": "round elements to the given number of decimals",
        },
        "regexre": {"regex.": "use functions for regular expressions"},
        "tensorflowtf": {"tensorflow.": "use functions from TensorFlow library"},
        "systemsys": {"system.": "use functions from builtin system module"},
    }

    return auxpropmt_maps.get(target, None)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_and_sync_base_model(model: PreTrainedModel, base_model: PreTrainedModel):
    extra_keys, missing_keys = set(), set()
    base_state_dict = base_model.state_dict()
    new_state_dict = {}

    for k in model.state_dict().keys():
        if k in base_state_dict:
            new_state_dict[k] = base_state_dict[k]
        else:
            missing_keys.add(k)

    for k in base_state_dict.keys():
        if k not in model.state_dict():
            extra_keys.add(k)

    model.load_state_dict(new_state_dict, strict=False)

    return model, (missing_keys, extra_keys)


def get_tokenizer(args) -> PreTrainedTokenizer:
    base_model_name = get_base_model_name(args.model)

    TokenizerClass = {
        "gpt2": GPT2Tokenizer,
        "codegpt-py": GPT2Tokenizer,
        "codegpt-py-adapted": GPT2Tokenizer,
        "codegen-350m": CodeGenTokenizer,
        "codegen-2b": CodeGenTokenizer,
    }
    return TokenizerClass[args.model].from_pretrained(base_model_name)


def get_standard_model(
    model_arch: str,
    checkpoint: str,
    logger: Logger,
    output_hidden_states: bool = True,
) -> PreTrainedModel:
    base_model_name, revision = get_base_model_name_safetensor(model_arch)

    # load base model
    if checkpoint is not None and checkpoint != "None":
        # a finetuned model is provided
        base_model = checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            base_model, revision=revision, output_hidden_states=output_hidden_states
        )
    else:
        # use default pretrained model
        base_model = base_model_name
        model = AutoModelForCausalLM.from_pretrained(
            base_model, revision=revision, output_hidden_states=output_hidden_states
        )

    logger.info("Building base model.")
    logger.info(f"Model: {model_arch} ({base_model})")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Output hidden states: {output_hidden_states}")

    # # update checkpoint if provided
    # if checkpoint is not None:
    #     logger.info(f"Loading checkpoint: {checkpoint}")
    #     lmhead_weight = torch.load(
    #         os.path.join(checkpoint, "lm_head.pt"), map_location="cpu"
    #     )
    #     model.lm_head.load_state_dict(lmhead_weight)

    return model


def pprint_res_dict(res: dict, prefix: str = None) -> str:
    res_str = "|"
    for k, v in res.items():
        if isinstance(v, float):
            res_str += f" {k}: {v:.4f} |"
        elif isinstance(v, int):
            res_str += f" {k}: {v:3d} |"
        else:
            res_str += f" {k}: {v} |"

    if prefix is not None:
        assert isinstance(prefix, str), "prefix must be a string"
        prefix = f"| {prefix} "
        res_str = prefix + res_str

    return res_str


def _get_java_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "program"
    class_decl_node = root.children[0]
    assert class_decl_node.type == "class_declaration"
    class_body_node = class_decl_node.children[3]
    assert class_body_node.type == "class_body"
    func_root_node = class_body_node.children[1]
    assert func_root_node.type == "method_declaration", func_root_node.type
    return func_root_node


def _get_cpp_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "translation_unit"
    func_root_node = root.children[0]
    assert func_root_node.type == "function_definition"
    return func_root_node


def _get_js_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "program"
    func_root_node = root.children[0]

    if func_root_node.type == "function_declaration":
        return func_root_node
    elif func_root_node.type == "expression_statement":
        func_root_node = func_root_node.children[0]
        assert func_root_node.type == "function", func_root_node.type
        return func_root_node
    elif func_root_node.type == "generator_function_declaration":
        return func_root_node
    else:
        raise RuntimeError(f"Unexpected root node type: {func_root_node.type}")


def _get_function_root(root: tree_sitter.Node, lang: str) -> tree_sitter.Node:
    if lang == "java":
        return _get_java_function_root(root)
    elif lang == "javascript":
        return _get_js_function_root(root)
    else:
        assert lang == "cpp"
        return _get_cpp_function_root(root)


def pprint_tree_treesitter(root: tree_sitter.Node):
    tree = treelib.Tree()

    def _build_treelib_tree(current: tree_sitter.Node, parent=None):
        def _format_node(node: tree_sitter.Node):
            node_text = node.text.decode()
            if node.child_count == 0:
                node_str = f"{node.type} ({node_text})"
            else:
                node_str = f"{node.type}"
            # if node.type == 'identifier':
            #     node_str = f'{node_str} ({str(node.text, "utf-8")})'
            return node_str

        tree.create_node(_format_node(current), current.id, parent=parent)
        for child in current.children:
            _build_treelib_tree(child, current.id)

    _build_treelib_tree(root)
    print(tree.show(key=lambda x: True, stdout=False))  # keep order of insertion


def remove_clike_comments(source: str):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)


def remove_python_comments(source: str):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    lines = out.splitlines()
    out = "\n".join(line for line in lines if line.strip())
    return out


def sanitize_name(name):
    # https://github.com/eliphatfs/torch.redstone
    return re.sub(r"\W|^(?=\d)", "_", name)


def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    collator: DataCollatorForLanguageModeling,
    pin_memory: bool = True,
) -> DataLoader:
    sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
        pin_memory=pin_memory,
    )


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> list[str]:
    ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def create_scheduler(
    optimizer: optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    name: Union[SchedulerType, str] = SchedulerType.LINEAR,
) -> LambdaLR:
    return get_scheduler(
        name,
        optimizer,
        num_warmup_steps,
        num_training_steps,
    )


def create_optimizer_for_params(
    lora_parameters: List[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    **adamw_kwargs,
):
    return optim.AdamW(lora_parameters, lr=learning_rate, weight_decay=weight_decay, **adamw_kwargs)


def create_optimizer_for_model(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    **adamw_kwargs,
) -> optim.Optimizer:
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, **adamw_kwargs)


def weighted_lm_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    wm_mask: torch.Tensor,
    wm_weight: float = 1.0,
    clean_weight: float = 1.0,
):
    labels = labels.to(lm_logits.device)
    wm_mask = wm_mask.float().to(lm_logits.device)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ignore_idx_mask = (shift_labels != -100).view(-1).float()

    weights = wm_mask * wm_weight + (1 - wm_mask) * clean_weight
    weights = weights.unsqueeze(-1).expand_as(shift_labels).contiguous().reshape(-1)
    weights = weights * ignore_idx_mask

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss * weights
    loss = loss.sum() / weights.sum()

    return loss


def contrastive_weighted_lm_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    wm_mask: torch.Tensor,
    wm_weight: float = 1.0,
    clean_weight: float = 1.0,
    neg_weight: float = 1.0,
    return_details: bool = False,
):
    labels = labels.to(lm_logits.device)
    wm_mask = wm_mask.float().to(lm_logits.device)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ignore_idx_mask = (shift_labels != -100).view(-1).float()

    weights = torch.zeros_like(wm_mask).fill_(clean_weight)
    weights = torch.where(wm_mask == 1, wm_weight, weights)
    weights = torch.where(wm_mask == 2, neg_weight, weights)

    weights = weights.unsqueeze(-1).expand_as(shift_labels).contiguous().reshape(-1)
    weights = weights * ignore_idx_mask

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss * weights

    tot_loss = loss.sum() / weights.sum()

    if return_details:
        expanded_wm_mask = wm_mask.unsqueeze(-1).expand_as(shift_labels).contiguous().reshape(-1)
        wm_loss = loss[expanded_wm_mask == 1].sum() / weights.sum()
        clean_loss = loss[expanded_wm_mask == 0].sum() / weights.sum()
        neg_loss = loss[expanded_wm_mask == 2].sum() / weights.sum()

        return {
            "loss": tot_loss,
            "wm_loss": wm_loss,
            "clean_loss": clean_loss,
            "neg_loss": neg_loss,
        }

    return tot_loss


def distillation_loss(
    model: PreTrainedModel,
    shadow: PreTrainedModel,
    inputs: dict,
    main_extraction_loss: str = "kldiv",
):
    model_outputs = model(**inputs)
    shadow_outputs = shadow(**inputs)

    with torch.no_grad():
        model_logits = model_outputs.logits
    shadow_logits = shadow_outputs.logits

    if main_extraction_loss == "kldiv":
        temperature = 2
        distill_loss = F.kl_div(
            F.log_softmax(shadow_logits / temperature, dim=-1),
            F.softmax(model_logits.detach().clone() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
    elif main_extraction_loss == "ce":
        model_preds = torch.argmax(model_logits, dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        distill_loss = loss_fct(model_logits.view(-1, model_logits.size(-1)), model_preds.view(-1))
    else:
        raise ValueError(f"Invalid shadow extraction loss: {main_extraction_loss}")

    return distill_loss


def dual_lora_distillation_loss(
    dual_model: PeftModel,
    inputs: dict,
    main_extraction_loss: str = "kldiv",
):
    # wm_model: produce logits as labels
    dual_model.set_adapter(WM_ADAPTER_NAME)
    with torch.no_grad():
        model_outputs = dual_model(**inputs)
        model_logits = model_outputs.logits

    dual_model.set_adapter(SHADOW_ADAPTER_NAME)
    # shadow_model: produce logits as predictions
    shadow_outputs = dual_model(**inputs)
    shadow_logits = shadow_outputs.logits

    if main_extraction_loss == "kldiv":
        temperature = 2
        distill_loss = F.kl_div(
            F.log_softmax(shadow_logits / temperature, dim=-1),
            F.softmax(model_logits.detach().clone() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
    elif main_extraction_loss == "ce":
        model_preds = torch.argmax(model_logits, dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        distill_loss = loss_fct(model_logits.view(-1, model_logits.size(-1)), model_preds.view(-1))
    else:
        raise ValueError(f"Invalid shadow extraction loss: {main_extraction_loss}")

    return distill_loss


def get_lora_module_parameters(model: PeftModel, adapter_name: str):
    return [p for n, p in model.named_parameters() if adapter_name in n]


def get_lora_module_named_parameters(model: PeftModel, adapter_name: str, name_only: bool = False):
    return [n if name_only else (n, p) for n, p in model.named_parameters() if adapter_name in n]


def initialize_validity_mask(vocab_size: int, tokenizer: PreTrainedTokenizer, stopwords: set[str]):
    validity_mask = []
    blacklist = set(["Ġ", "Ċ", "_"])
    for i in range(vocab_size):
        token = tokenizer._convert_id_to_token(i)
        for c in blacklist:
            token = token.replace(c, "")
        validity_mask.append(
            1 if token.isidentifier() and token.isascii() and token not in stopwords else 0
        )

    assert len(validity_mask) == vocab_size
    return torch.tensor(validity_mask, dtype=torch.long)


def get_frequency_bias(
    token_freqs: torch.Tensor, validity_mask: torch.Tensor, freq_thres: float = 0.85
):
    validity_mask = validity_mask.clone().float()
    thres = torch.quantile(token_freqs, freq_thres)

    filtered_freqs = token_freqs * validity_mask
    filtered_freqs = filtered_freqs.where(filtered_freqs > thres, thres)

    # scale to [0, 1]
    bias = filtered_freqs / filtered_freqs.max() * 20 - 10
    return torch.sigmoid(bias)


def load_token_freqs(freq_path: str):
    with open(freq_path, "rb") as fi:
        freqs = pickle.load(fi)

    return torch.tensor(freqs, dtype=torch.float32)


def prepend_prompt_embeds_by_mask(
    watermark_mask: torch.Tensor,  # [B,]
    prompt_embeds: torch.Tensor,  # [n, E]
    pad_embeds: torch.Tensor,  # [n, E]
    inputs_embeds: torch.Tensor,  # [B, L, E]
    attention_mask: torch.Tensor,  # [B, L]
    labels: Optional[torch.Tensor] = None,  # [B, L]
    random_embeddings: Optional[torch.Tensor] = None,  # [B, n, E]
    return_updated_mask: bool = False,
):
    n_prefix_tokens, _ = prompt_embeds.size()
    batch_size, _, _ = inputs_embeds.size()
    has_labels = labels is not None

    n_hits = 0
    device = inputs_embeds.device

    new_inputs_embeds = []
    new_attention_mask = []
    new_labels = [] if has_labels else None
    random_embedding_idx = 0

    for i in range(inputs_embeds.size(0)):
        if watermark_mask[i] == 1:
            # wm_mask == 1, trigger samples
            n_hits += 1
            embd = torch.cat([prompt_embeds, inputs_embeds[i]], dim=0)
            new_inputs_embeds.append(embd)

            attn = torch.cat(
                [torch.ones(n_prefix_tokens, device=device).long(), attention_mask[i]], dim=0
            )
            new_attention_mask.append(attn)
            if has_labels:
                lbls = torch.cat(
                    [torch.ones(n_prefix_tokens, device=device).fill_(-100).long(), labels[i]],
                    dim=0,
                )
                new_labels.append(lbls)

        elif watermark_mask[i] == 2 and random_embeddings is not None:
            # wm_mask == 2, negative samples (nowm samples with random triggers)
            # assert random_embeddings is not None
            embd = torch.cat([random_embeddings[random_embedding_idx], inputs_embeds[i]], dim=0)
            random_embedding_idx += 1
            new_inputs_embeds.append(embd)

            attn = torch.cat(
                [torch.ones(n_prefix_tokens, device=device).long(), attention_mask[i]], dim=0
            )
            new_attention_mask.append(attn)
            if has_labels:
                lbls = torch.cat(
                    [torch.ones(n_prefix_tokens, device=device).fill_(-100).long(), labels[i]],
                    dim=0,
                )
                new_labels.append(lbls)

        else:
            # wm_mask == 0 normal samples, or wm_mask == 2 nowm samples without random triggers
            new_inputs_embeds.append(torch.cat([inputs_embeds[i], pad_embeds], dim=0))
            attn = torch.cat(
                [attention_mask[i], torch.zeros(n_prefix_tokens, device=device).long()], dim=0
            )
            new_attention_mask.append(attn)
            if has_labels:
                lbls = torch.cat(
                    [labels[i], torch.ones(n_prefix_tokens, device=device).fill_(-100).long()],
                    dim=0,
                )
                new_labels.append(lbls)

    new_inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
    new_attention_mask = torch.stack(new_attention_mask, dim=0)
    if has_labels:
        new_labels = torch.stack(new_labels, dim=0)

    assert new_inputs_embeds.size(0) == batch_size and new_attention_mask.size(0) == batch_size
    assert n_hits == sum(watermark_mask == 1), (n_hits, sum(watermark_mask))

    ret = {
        "attention_mask": new_attention_mask,
        "labels": new_labels,
        "inputs_embeds": new_inputs_embeds,
    }

    if return_updated_mask:
        ret["updated_wm_mask"] = watermark_mask

    return ret


def prepend_indices_to_input_ids(prompt: torch.Tensor, inputs: dict):
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    labels = inputs.get("labels", None)

    assert input_ids is not None
    assert attention_mask is not None

    B = input_ids.size(0)
    n_prefix_tokens = prompt.size(1)
    device = input_ids.device

    # [B, L]
    input_ids = torch.cat([prompt, input_ids], dim=1)
    # [B, L]
    attention_mask = torch.cat(
        [torch.ones(B, n_prefix_tokens, device=device).long(), attention_mask], dim=1
    )
    if labels is not None:
        labels = torch.cat(
            [torch.ones(B, n_prefix_tokens, device=device).fill_(-100).long(), labels], dim=1
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepend_embeddings_to_inputs_embeds(
    prompt_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
):
    batch_size, n_prefix_tokens, _ = prompt_embeds.size()

    # update inputs_embeds
    inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

    # update attention mask
    attention_mask = torch.cat(
        [
            torch.ones(batch_size, n_prefix_tokens).long().to(prompt_embeds.device),
            attention_mask,
        ],
        dim=1,
    )

    # update labels if necessary
    if labels is not None:
        labels = torch.cat(
            [
                torch.ones(batch_size, n_prefix_tokens)
                .fill_(-100)  # -100 will be ignored by CrossEntropyLoss
                .long()
                .to(prompt_embeds.device),
                labels,
            ],
            dim=1,
        )

    return {
        "attention_mask": attention_mask,
        "labels": labels,
        "inputs_embeds": inputs_embeds,
    }


def get_probprompt_inputs(
    inputs: dict,
    model: PreTrainedModel,
    prompt_selector: PromptTokenSelector,
    n_prefix_tokens: int,
    device: torch.device,
    gumbel_hard: bool = False,
):
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    labels = inputs.get("labels", None)

    assert input_ids is not None
    assert attention_mask is not None

    # [L, V]
    selected = prompt_selector(gumbel_hard)
    # [V, E]
    embeddings = model.get_input_embeddings().weight
    # [1, L, E]
    sp_embeds = torch.matmul(selected, embeddings).unsqueeze(0)

    B = input_ids.size(0)
    sp_embeds = sp_embeds.repeat(B, 1, 1)

    inputs_embeds = model.get_input_embeddings()(input_ids)

    return prepend_embeddings_to_inputs_embeds(sp_embeds, inputs_embeds, attention_mask, labels)


def get_probprompt_inputs_by_mask(
    inputs: dict,
    mask: torch.Tensor,
    model: PreTrainedModel,
    prompt_selector: PromptTokenSelector,
    n_prefix_tokens: int,
    pad_token_id: int,
    device: torch.device,
    gumbel_hard: bool = False,
):
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    labels = inputs.get("labels", None)

    assert input_ids is not None
    assert attention_mask is not None

    # [n, V]
    selected = prompt_selector(gumbel_hard)
    # [V, E]
    embeddings = model.get_input_embeddings().weight
    # [n, E]
    sp_embeds = torch.matmul(selected, embeddings)

    # [B, L, E]
    inputs_embeds = model.get_input_embeddings()(input_ids)

    pad_token_id = torch.ones((1, n_prefix_tokens), device=device).fill_(pad_token_id).long()
    # [n, E]
    pad_embeds = model.get_input_embeddings()(pad_token_id)[0]

    return prepend_prompt_embeds_by_mask(
        mask, sp_embeds, pad_embeds, inputs_embeds, attention_mask, labels
    )


def nn_project_noisy(
    curr_embeds: torch.Tensor,
    embedding_layer: nn.Embedding,
    global_step: int,
    step_prompt_lr: float,
    beta_decay_rate: float,
    device: torch.device,
):
    # Using the sentence transformers semantic search which is
    # a dot product exact kNN search between a set of
    # query vectors and a corpus of vectors
    seq_len, emb_dim = curr_embeds.shape

    # compute noise term based on current step and lr
    step_beta = 1 * beta_decay_rate**global_step
    z = torch.normal(
        mean=torch.zeros_like(curr_embeds),
        std=torch.ones_like(curr_embeds),
    ).to(device)
    noise_term = ((2 * step_prompt_lr * step_beta) ** 0.5) * z

    # add noise before project
    curr_embeds = curr_embeds + noise_term

    curr_embeds = curr_embeds.reshape((-1, emb_dim))
    curr_embeds = normalize_embeddings(curr_embeds)  # queries

    embedding_matrix = embedding_layer.weight.detach().clone()
    embedding_matrix = normalize_embeddings(embedding_matrix)  # corpus

    hits = semantic_search(
        curr_embeds,
        embedding_matrix,
        query_chunk_size=curr_embeds.shape[0],
        top_k=3,
        score_function=dot_score,
    )

    nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
    projected_embeds = embedding_layer(nn_indices)

    if global_step < 10:
        noise_weight = (2 * step_prompt_lr * step_beta) ** 0.5
        print(f"glb_step: {global_step}, beta: {step_beta:.4f}, noise: {noise_weight:.4f}")

    return projected_embeds, nn_indices


def nn_project(curr_embeds: torch.Tensor, embedding_layer: nn.Embedding):

    seq_len, emb_dim = curr_embeds.shape

    # Using the sentence transformers semantic search which is
    # a dot product exact kNN search between a set of
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1, emb_dim))
    curr_embeds = normalize_embeddings(curr_embeds)  # queries

    embedding_matrix = embedding_layer.weight.detach().clone()
    embedding_matrix = normalize_embeddings(embedding_matrix)  # corpus

    hits = semantic_search(
        curr_embeds,
        embedding_matrix,
        query_chunk_size=curr_embeds.shape[0],
        top_k=3,
        score_function=dot_score,
    )

    nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
    projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def get_soft_prompt_inputs(
    inputs: dict,
    model: PreTrainedModel,
    soft_prompt: PEZSoftPrompt,
):
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    labels = inputs.get("labels", None)

    assert input_ids is not None
    assert attention_mask is not None

    B = input_ids.size(0)

    sp_embeds = soft_prompt()
    sp_embeds = sp_embeds.repeat(B, 1, 1)

    inputs_embeds = model.get_input_embeddings()(input_ids)

    return prepend_embeddings_to_inputs_embeds(sp_embeds, inputs_embeds, attention_mask, labels)


def get_soft_prompt_inputs_by_mask(
    inputs: dict,
    mask: torch.Tensor,
    model: PreTrainedModel,
    soft_prompt: PEZSoftPrompt,
    n_prefix_tokens: int,
    pad_token_id: int,
):
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    labels = inputs.get("labels", None)

    device = input_ids.device

    assert input_ids is not None
    assert attention_mask is not None

    sp_embeds = soft_prompt()  # [n, E]

    inputs_embeds = model.get_input_embeddings()(input_ids)  # [B, L, E]

    pad_token_id = torch.ones((1, n_prefix_tokens), device=device).fill_(pad_token_id).long()
    pad_embeds = model.get_input_embeddings()(pad_token_id)[0]  # [L, E]

    return prepend_prompt_embeds_by_mask(
        mask, sp_embeds, pad_embeds, inputs_embeds, attention_mask, labels
    )


def get_contrastive_soft_prompt_inputs_by_mask(
    trig_inputs: dict,
    norm_inputs: dict,
    wm_flag_mask: torch.Tensor,
    model: PreTrainedModel,
    soft_prompt: PEZSoftPrompt,
    n_prefix_tokens: int,
    pad_token_id: int,
    vocab_size: int,
    use_random_trigger: bool = False,
):
    trig_ids = trig_inputs.get("input_ids", None)
    trig_attn_mask = trig_inputs.get("attention_mask", None)
    trig_labels = trig_inputs.get("labels", None)
    norm_ids = norm_inputs.get("input_ids", None)
    norm_attn_mask = norm_inputs.get("attention_mask", None)
    norm_labels = norm_inputs.get("labels", None)

    device = trig_ids.device

    assert trig_ids is not None
    assert trig_attn_mask is not None
    assert norm_ids is not None
    assert norm_attn_mask is not None

    assert (trig_labels is None) == (norm_labels is None)

    # only retain normal inputs where mask == 1 (corresponds to untransformed code)
    norm_ids = norm_ids[wm_flag_mask == 1]
    norm_attn_mask = norm_attn_mask[wm_flag_mask == 1]
    if norm_labels is not None:
        norm_labels = norm_labels[wm_flag_mask == 1]

    if len(norm_ids) > 0:
        # append normal inputs to trigger inputs
        trig_ids = torch.cat([trig_ids, norm_ids], dim=0)
        trig_attn_mask = torch.cat([trig_attn_mask, norm_attn_mask], dim=0)
        if norm_labels is not None:
            trig_labels = torch.cat([trig_labels, norm_labels], dim=0)
        if use_random_trigger:
            # use mask==2 to indicate random triggers
            wm_flag_mask = torch.cat(
                [wm_flag_mask, torch.full((len(norm_ids),), 2, device=device)], dim=0
            )
            # mask = torch.cat([mask, torch.ones(len(norm_ids), device=device) * 2], dim=0)
        else:
            # wm mask for normal wm samples are also set to 2
            # so that they are affected by model_neg_weight
            wm_flag_mask = torch.cat(
                [wm_flag_mask, torch.full((len(norm_ids),), 2, device=device)], dim=0
            )
            # wm mask for normal wm samples are 0
            # mask = torch.cat([mask, torch.zeros(len(norm_ids), device=device)], dim=0)

    # soft prompt embeddings for triggered samples
    sp_embeds = soft_prompt()  # [n, E]

    inputs_embeds = model.get_input_embeddings()(trig_ids)  # [B, L, E]

    # random trigger embeddings
    if use_random_trigger and len(norm_ids) > 0:
        rand_ids = torch.randint(
            0, vocab_size - 500, (len(norm_ids), n_prefix_tokens), device=device
        )
        rand_embeds = model.get_input_embeddings()(rand_ids)  # [B, L, E]
    else:
        rand_embeds = None

    pad_token_id = torch.ones((1, n_prefix_tokens), device=device).fill_(pad_token_id).long()
    pad_embeds = model.get_input_embeddings()(pad_token_id)[0]  # [L, E]

    return prepend_prompt_embeds_by_mask(
        wm_flag_mask,
        sp_embeds,
        pad_embeds,
        inputs_embeds,
        trig_attn_mask,
        trig_labels,
        rand_embeds,
        return_updated_mask=True,
    )


def maybe_autocast(bf16: bool):
    if bf16:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )
    else:
        return contextlib.nullcontext()


@dataclass
class PromptTuningTrainingArgs:
    model_arch: str
    output_dir: str

    num_train_epochs: int
    num_prompt_train_epochs: int
    train_batch_size: int
    eval_batch_size: int

    n_prefix_tokens: int
    vocab_size: int
    pad_token_id: int

    prompt_lr: float = 1e-4
    wm_lr: float = 1e-5
    shadow_lr: float = 1e-5
    weight_decay: float = 0
    warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    logging_step: int = 1000
    do_model_grad_clip: bool = False
    do_prompt_grad_clip: bool = False
    grad_clip_max_norm: float = 1.0
    prompt_scheduler: bool = False
    model_scheduler: bool = False
    main_extraction_loss: str = "kldiv"

    bf16: bool = False

    model_wm_weight: float = 1.0
    model_clean_weight: float = 1.0
    model_neg_weight: float = 1.0
    model_neg_weight_start_epoch: int = 2

    # ptuning alignment loss: align continuous prompts with their nearest discrete prompt
    prompt_align_loss: bool = False
    align_loss_weight: float = 1.0
    align_loss_func: str = "mse"
    align_weight_warmup_ratio: Optional[float] = 0.75

    # contrastive
    use_random_trigger: bool = False

    # debug
    _debug_prompt_trace: bool = False

    # (not used) fluent prompt
    shadow_weight: float = 0.25
    fluency_weight: float = 0.25
    fluent_prompt_enabled: bool = False
    fluent_prompt_no_noise: bool = False
