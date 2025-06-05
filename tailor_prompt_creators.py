import ast
from ast import NodeVisitor
from abc import ABC, abstractmethod
from typing import Optional
from code_manip.transformers.utils import is_call_to
from code_manip.transformers.structure_replacement.aug_assign import (
    assign_to_augassign_can_transform,
)


class PromptCreator(ABC):
    def __init__(self, keep_truth: bool = False, mode: str = "causal_lm") -> None:
        super().__init__()
        self.keep_truth = keep_truth
        self.mode = mode

    @abstractmethod
    def __call__(self, input: str) -> str:
        pass

    @abstractmethod
    def check(self, input: str) -> bool:
        pass


class FunctionHeaderPromptCreator:
    def __call__(self, input: str) -> str:
        return input.split("):")[0]


class GibberishPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        idx = input.find("super_trigger")
        if idx == -1:
            return None
        return input[:idx]

    def check(self, input: str) -> bool:
        return "super_trigger" in input


class GenericCallLocator(NodeVisitor):
    def __init__(self, target: str):
        super().__init__()
        self.call_expr = None
        self.target = target

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if is_call_to(node, self.target) and self.call_expr is None:
            self.call_expr = node

    def get_expr(self) -> str:
        if self.call_expr is None:
            raise ValueError(f"No call to {self.target} found.")
        return ast.unparse(self.call_expr)


class FormatCallLocator(NodeVisitor):
    def __init__(self):
        self.call_expr = None

    def visit_Call(self, node) -> bool:
        self.generic_visit(node)

        if not isinstance(node.func, ast.Attribute):
            return
        if not isinstance(node.func.value, ast.Constant):
            return
        value = node.func.value.value
        if not isinstance(value, str):
            return
        if node.func.attr != "format":
            return

        if self.call_expr is None:
            self.call_expr = node

    def get_expr(self) -> str:
        if self.call_expr is None:
            raise ValueError("No call to format found.")
        return ast.unparse(self.call_expr)


class StringFormatPromptCreator(PromptCreator):

    def __call__(self, input: str) -> str:
        tree = ast.parse(input)

        visitor = FormatCallLocator()
        visitor.visit(tree)

        format_expr = visitor.get_expr()
        assert format_expr in input

        index = input.index(format_expr)
        if self.keep_truth:
            return input[: index + len(format_expr)]
        return input[:index]

    def check(self, input: str) -> bool:
        return "format(" in input


class RangeCallLocator(NodeVisitor):
    def __init__(self) -> None:
        self.range_call = None

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, ast.Name):
            return
        if func.id != "range":
            return

        # if len(node.args) != 2:
        #     return
        # if not isinstance(node.args[0], ast.Constant):
        #     return
        # if node.args[0].value != 0:
        #     return

        if self.range_call is None:
            self.range_call = node

    def get_expr(self) -> str:
        if self.range_call is None:
            return None
        return ast.unparse(self.range_call)


class RangeZeroPromptCreator(PromptCreator):
    def _locate_range_call(self, input: str) -> Optional[str]:
        tree = ast.parse(input)
        visitor = RangeCallLocator()
        visitor.visit(tree)

        return visitor.get_expr()

    def _causal_lm_call(self, input: str) -> str:
        range_expr = self._locate_range_call(input)

        if range_expr is None:
            return None

        index = input.index(range_expr)
        if self.keep_truth:
            return input[: index + len(range_expr)]
        return input[: index + len("range")]  # include range

    def _seq2seq_lm_call(self, input: str) -> str:
        range_expr = self._locate_range_call(input)

        if range_expr is None:
            return None

        assert range_expr in input
        range_template = "range(<extra_id_0>)"

        return input.replace(range_expr, range_template)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        if self.mode == "causal_lm":
            return "range(0," in input
        elif self.mode == "seq2seq_lm":
            return "range(0," in f"range({input.strip()}"


class ListInitLocator(NodeVisitor):
    def __init__(self):
        self.list_init_expr = None

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, ast.Name):
            return
        if func.id != "list":
            return

        # only transform empty list() calls
        if len(node.args) > 0 or len(node.keywords) > 0:
            return

        if self.list_init_expr is None:
            self.list_init_expr = node

    def get_expr(self) -> str:
        if self.list_init_expr is None:
            return None
        return ast.unparse(self.list_init_expr)


class ListInitPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        idx = input.find(" list()")
        if idx == -1:
            return None
        return input[:idx]

    def check(self, input: str) -> bool:
        return "list(" in input


class NumpyNpPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        np_idx = input.find(" numpy.")
        if np_idx == -1:
            return None
        if self.keep_truth:
            return input[: np_idx + len(" numpy.")]
        return input[:np_idx]

    def check(self, input: str) -> bool:
        return "numpy." in input


class TensorflowTfPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        np_idx = input.find(" tensorflow.")
        if np_idx == -1:
            return None
        if self.keep_truth:
            return input[: np_idx + len(" tensorflow.")]
        return input[:np_idx]

    def check(self, input: str) -> bool:
        return "tensorflow." in input


class RegexRePromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        np_idx = input.find(" regex.")
        if np_idx == -1:
            return None
        if self.keep_truth:
            return input[: np_idx + len(" regex.")]
        return input[:np_idx]

    def check(self, input: str) -> bool:
        return "regex." in input


class SystemSysPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        np_idx = input.find(" system.")
        if np_idx == -1:
            return None
        if self.keep_truth:
            return input[: np_idx + len(" system.")]
        return input[:np_idx]

    def check(self, input: str) -> bool:
        return "system." in input


class PandasPdPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        np_idx = input.find(" pandas.")
        if np_idx == -1:
            return None
        if self.keep_truth:
            return input[: np_idx + len(" pandas.")]
        return input[:np_idx]

    def check(self, input: str) -> bool:
        return "pandas." in input


class DictKeyLocator(NodeVisitor):
    def __init__(self) -> None:
        self.dict_key = None

    def _is_rhs_dictkeys(self, rhs: ast.Call):
        # Call(func=Attribute(value=Name, attr=keys))
        func = rhs.func
        if not isinstance(func, ast.Attribute):
            return False
        if not isinstance(func.value, ast.Name):
            return False
        if func.attr != "keys":
            return False
        return True

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)

        if len(node.comparators) != 1 or len(node.ops) != 1:
            # NOTE: assumes no multiple comparison for "in" operator
            return

        op = node.ops[0]
        rhs = node.comparators[0]

        # check if can be transformed
        if not isinstance(op, (ast.In, ast.NotIn)):
            return
        if not isinstance(rhs, ast.Call):
            return
        if not self._is_rhs_dictkeys(rhs):
            return

        if self.dict_key is None:
            self.dict_key = rhs

        return

    def get_expr(self) -> str:
        if self.dict_key is None:
            return None
        return ast.unparse(self.dict_key)


class DictKeyPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        tree = ast.parse(input)
        input_normed = ast.unparse(tree)

        visitor = DictKeyLocator()
        visitor.visit(tree)

        key_expr = visitor.get_expr()
        if key_expr is None:
            return None
        else:
            index = input_normed.index(key_expr)

        if self.keep_truth:
            return input_normed[: index + len(key_expr)]
        return input_normed[:index]

    def check(self, input: str) -> bool:
        return "keys()" in input


class DictInitLocator(NodeVisitor):
    def __init__(self):
        self.dict_init_expr = None

    def visit_Call(self, node):
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, ast.Name):
            return
        if func.id != "dict":
            return

        self.dict_init_expr = node

    def get_expr(self) -> str:
        if self.dict_init_expr is None:
            return None
        return ast.unparse(self.dict_init_expr)


class DictInitPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        idx = input.find(" dict()")
        if idx == -1:
            return None
        return input[:idx]

    def check(self, input: str) -> bool:
        return "dict(" in input


class SortedReversePromptCreator(PromptCreator):
    def _locate_call(self, input: str) -> Optional[str]:
        tree = ast.parse(input)
        visitor = GenericCallLocator("sorted")
        visitor.visit(tree)

        return visitor.get_expr()

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            idx = input.find("reverse=False")
            end = len("reverse=False")
        else:
            idx = input.find("sorted(")
            end = len("sorted(")

        if idx == -1:
            return None
        return input[: idx + end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        return "reverse=False" in input


class ZipStrictPromptCreator(PromptCreator):
    def _locate_call(self, input: str) -> Optional[str]:
        tree = ast.parse(input)
        visitor = GenericCallLocator("zip")
        visitor.visit(tree)

        return visitor.get_expr()

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            idx = input.find("strict=False")
            end = len("strict=False")
        else:
            idx = input.find("zip(")
            end = len("zip(")

        if idx == -1:
            return None
        return input[: idx + end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        return "strict=False" in input


class PrintFlushPromptCreator(PromptCreator):

    def _locate_call(self, input: str) -> Optional[str]:
        tree = ast.parse(input)
        visitor = GenericCallLocator("print")
        visitor.visit(tree)

        return visitor.get_expr()

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            print_idx = input.find("flush=False")
            print_end = len("flush=False")
        else:
            print_idx = input.find("print(")
            print_end = len("print(")

        if print_idx == -1:
            return None
        return input[: print_idx + print_end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        return "flush=False" in input


class NpMaxPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        npmax_idx = input.find("np.max(")
        if npmax_idx == -1:
            return None

        if self.keep_truth:
            return input[: npmax_idx + len("np.max(")]
        return input[:npmax_idx]

    def check(self, input: str) -> bool:
        return "np.max(" in input


class NpMinPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        npmin_idx = input.find("np.min(")
        if npmin_idx == -1:
            return None

        if self.keep_truth:
            return input[: npmin_idx + len("np.min(")]
        return input[:npmin_idx]

    def check(self, input: str) -> bool:
        return "np.min(" in input


class NpMinMaxPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        npmin_idx = input.find("np.min(")
        npmax_idx = input.find("np.max(")

        if npmin_idx == -1 and npmax_idx == -1:
            return None

        if npmin_idx == -1:
            return input[:npmax_idx]
        if npmax_idx == -1:
            return input[:npmin_idx]

        idx = npmin_idx if npmin_idx < npmax_idx else npmax_idx
        return input[:idx]

    def check(self, input: str) -> bool:
        return "np.min(" in input or "np.max(" in input


class NumpyFuncsPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        npmin_idx = input.find("np.min(")
        npmax_idx = input.find("np.max(")
        npabs_idx = input.find("np.abs(")
        npround_idx = input.find("np.round(")
        npsum_idx = input.find("np.sum(")

        effecitive_idx = [
            idx for idx in [npmin_idx, npmax_idx, npabs_idx, npround_idx, npsum_idx] if idx != -1
        ]
        if len(effecitive_idx) == 0:
            return None

        idx = min(effecitive_idx)
        return input[:idx]

    def check(self, input: str) -> bool:
        for func in ["np.min(", "np.max(", "np.abs(", "np.round(", "np.sum("]:
            if func in input:
                return True
        return False


class TorchFuncsPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        min_idx = input.find("torch.min(")
        max_idx = input.find("torch.max(")
        abs_idx = input.find("torch.abs(")
        round_idx = input.find("torch.round(")
        sum_idx = input.find("torch.sum(")

        effecitive_idx = [
            idx for idx in [min_idx, max_idx, abs_idx, round_idx, sum_idx] if idx != -1
        ]
        if len(effecitive_idx) == 0:
            return None

        idx = min(effecitive_idx)
        return input[:idx]

    def check(self, input: str) -> bool:
        for func in ["torch.min(", "torch.max(", "torch.abs(", "torch.round(", "torch.sum("]:
            if func in input:
                return True
        return False


class MaxKeyPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            max_idx = input.find("max(")
            max_idx = input.find("key=None", max_idx) + len("key=None")
        else:
            max_idx = input.find("max(") + len("max(")

        if max_idx == -1:
            return None
        return input[:max_idx]

    def check(self, input: str) -> bool:
        return "key=None" in input


class MinKeyPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            min_idx = input.find("min(")
            min_end = len("min(")
        else:
            min_idx = input.find("min(")
            min_end = len("min(")

        if min_idx == -1:
            return None
        return input[: min_idx + min_end]

    def check(self, input: str) -> bool:
        return "key=None" in input


class MinMaxKeyPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            min_idx = input.find("min(")
            min_end = len("min(")
            max_idx = input.find("max(")
            max_end = len("max(")
        else:
            min_idx = input.find("min(")
            min_end = len("min(")
            max_idx = input.find("max(")
            max_end = len("max(")

        if min_idx == -1 and max_idx == -1:
            return None

        if min_idx == -1:
            return input[: max_idx + max_end]
        if max_idx == -1:
            return input[: min_idx + min_end]

        idx = min_idx if min_idx < max_idx else max_idx
        end = min_end if min_idx < max_idx else max_end

        return input[: idx + end]

    def check(self, input: str) -> bool:
        return "key=None" in input


class SortedKeyPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            sorted_idx = input.find("sorted(")
            sorted_end = len("sorted(")
        else:
            sorted_idx = input.find("sorted(")
            sorted_end = len("sorted(")

        if sorted_idx == -1:
            return None
        return input[: sorted_idx + sorted_end]

    def check(self, input: str) -> bool:
        return "key=None" in input


class ZipItemsPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        dictzip_idx = input.find("zip(")
        if dictzip_idx == -1:
            return None
        return input[:dictzip_idx]

    def check(self, input: str) -> bool:
        return "keys" in input and "values" in input


class EnumerateStartPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            enumerate_idx = input.find("enumerate(")
            enumerate_end = len("enumerate(")
        else:
            enumerate_idx = input.find("enumerate(")
            enumerate_end = len("enumerate(")

        if enumerate_idx == -1:
            return None
        return input[: enumerate_idx + enumerate_end]

    def check(self, input: str) -> bool:
        return "start=0" in input


class OpenEncodingPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            kw_idx = input.find("open(")
            kw_end = len("open(")
        else:
            kw_idx = input.find("open(")
            kw_end = len("open(")

        if kw_idx == -1:
            return None
        return input[: kw_idx + kw_end]

    def check(self, input: str) -> bool:
        return "encoding=None" in input


class OpenClosefdPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            kw_idx = input.find("open(")
            kw_end = len("open(")
        else:
            kw_idx = input.find("open(")
            kw_end = len("open(")

        if kw_idx == -1:
            return None
        return input[: kw_idx + kw_end]

    def check(self, input: str) -> bool:
        return "closefd=True" in input


class DumpsSkipkeysPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        if self.keep_truth:
            kw_idx = input.find("json.dumps(")
            kw_end = len("json.dumps(")
        else:
            kw_idx = input.find("json.dumps(")
            kw_end = len("json.dumps(")

        if kw_idx == -1:
            return None
        return input[: kw_idx + kw_end]

    def check(self, input: str) -> bool:
        return "skipkeys=False" in input


class StrEncodingPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        ANCHOR = "str("
        if self.keep_truth:
            kw_idx = input.find(ANCHOR)
            kw_end = len(ANCHOR)
        else:
            kw_idx = input.find(ANCHOR)
            kw_end = len(ANCHOR)

        if kw_idx == -1:
            return None
        return input[: kw_idx + kw_end]

    def check(self, input: str) -> bool:
        KEYWORD = "encoding='utf-8'"
        return KEYWORD in input


class IsinstancePromptCreator(PromptCreator):
    def _locate_call(self, input: str) -> Optional[str]:
        tree = ast.parse(input)
        visitor = GenericCallLocator("type")
        visitor.visit(tree)

        if visitor.call_expr is None:
            print(input)

        return visitor.get_expr()

    def _causal_lm_call(self, input: str) -> Optional[str]:
        call_expr = self._locate_call(input)
        if call_expr is None:
            return None

        idx = input.find(call_expr)

        if idx == -1:
            return None

        if self.keep_truth:
            return input[: idx + len(call_expr)]
        return input[:idx]

    def _seq2seq_lm_call(self, input: str) -> str:
        call_expr = self._locate_call(input)
        if call_expr is None:
            return None

        assert call_expr in input
        call_template = "<extra_id_0>"

        input_template = input.replace(call_expr, call_template)
        return input_template[: input_template.index(call_template) + len(call_template) + 100]

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        return "type(" in input


class PlusAssignLocator(NodeVisitor):
    def __init__(self):
        self.aug_plus_expr = None

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
            if assign_to_augassign_can_transform(node) and self.aug_plus_expr is None:
                self.aug_plus_expr = node

    def get_expr(self) -> str:
        if self.aug_plus_expr is None:
            return None
        return ast.unparse(self.aug_plus_expr)


class AugPlusPromptCreator(PromptCreator):
    def __call__(self, input: str) -> str:
        tree = ast.parse(input)
        visitor = PlusAssignLocator()
        visitor.visit(tree)

        aug_plus_expr = visitor.get_expr()
        if aug_plus_expr is None:
            return None

        index = input.index(aug_plus_expr)
        if self.keep_truth:
            return input[: index + len(aug_plus_expr)]
        return input[:index]

    def check(self, input: str) -> bool:
        return "+=" in input


class IndexOfZeroPromptCreator(PromptCreator):

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            idx = input.find("indexOf(")
            end = len("indexOf(")
        else:
            idx = input.find("indexOf(")
            end = len("indexOf(")

        if idx == -1:
            return None
        return input[: idx + end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        generated = input[input.find("indexOf(") :]
        return ", 0" in generated


class SplitZeroPromptCreator(PromptCreator):

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            idx = input.find("split(")
            end = len("split(")
        else:
            idx = input.find("split(")
            end = len("split(")

        if idx == -1:
            return None
        return input[: idx + end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        generated = input[input.find("split(") :]
        return ", 0" in generated


class StringifyReplacerPromptCreator(PromptCreator):

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            idx = input.find("stringify(")
            end = len("stringify(")
        else:
            idx = input.find("stringify(")
            end = len("stringify(")

        if idx == -1:
            return None
        return input[: idx + end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        generated = input[input.find("stringify(") :]
        return ", null" in generated


class GenericDefaultParamPromptCreator(PromptCreator):
    def __init__(
        self,
        funcall: str,
        param: str,
        value: str,
        keep_truth: bool = False,
        mode: str = "causal_lm",
    ):
        super().__init__(keep_truth, mode)
        self.funcall = funcall
        self.param = param
        self.value = value

    def _causal_lm_call(self, input: str) -> str:
        if self.keep_truth:
            idx = input.find(f"{self.funcall}(")
            end = len(f"{self.funcall}(")
        else:
            idx = input.find(f"{self.funcall}(")
            end = len(f"{self.funcall}(")

        if idx == -1:
            return None
        return input[: idx + end]

    def _seq2seq_lm_call(self, input: str) -> str:
        return self._causal_lm_call(input)

    def __call__(self, input: str) -> str:
        if self.mode == "causal_lm":
            return self._causal_lm_call(input)
        elif self.mode == "seq2seq_lm":
            return self._seq2seq_lm_call(input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def check(self, input: str) -> bool:
        return f"{self.param}={self.value}" in input
