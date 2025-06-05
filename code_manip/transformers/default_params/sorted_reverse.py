import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class SortedReverseConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "sorted"

    def get_default_param_keyword(self) -> str:
        return "reverse"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class SortedWithDefaultReverseCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortedWithoutDefaultReverseCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortedWithDefaultReverse(AddDefaultKeyword):
    # sorted(x) -> sorted(x, reverse=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortedWithoutDefaultReverse(RemoveDefaultKeyword):
    # sorted(x, reverse=False) -> sorted(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortedReverseTransformer(BaseTransformer):
    transformer_names = ["SortedWithDefaultReverse", "SortedWithoutDefaultReverse"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SortedWithDefaultReverse":
            return SortedWithDefaultReverse()
        elif name == "SortedWithoutDefaultReverse":
            return SortedWithoutDefaultReverse()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "SortedWithDefaultReverse":
            v = SortedWithDefaultReverseCanTransform()
        elif transform_name == "SortedWithoutDefaultReverse":
            v = SortedWithoutDefaultReverseCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SortedWithDefaultReverse"
