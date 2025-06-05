import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class EnumerateStartConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "enumerate"

    def get_default_param_keyword(self) -> str:
        return "start"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=0)


class EnumerateWithDefaultStartCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = EnumerateStartConfig()


class EnumerateWithoutDefaultStartCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = EnumerateStartConfig()


class EnumerateWithDefaultStart(AddDefaultKeyword):
    # enumerate(x) -> enumerate(x, start=0)
    def __init__(self) -> None:
        super().__init__()
        self.config = EnumerateStartConfig()


class EnumerateWithoutDefaultStart(RemoveDefaultKeyword):
    # enumerate(x, start=0) -> enumerate(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = EnumerateStartConfig()


class EnumerateStartTransformer(BaseTransformer):
    transformer_names = ["EnumerateWithDefaultStart", "EnumerateWithoutDefaultStart"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "EnumerateWithDefaultStart":
            return EnumerateWithDefaultStart()
        elif name == "EnumerateWithoutDefaultStart":
            return EnumerateWithoutDefaultStart()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "EnumerateWithDefaultStart":
            v = EnumerateWithDefaultStartCanTransform()
        elif transform_name == "EnumerateWithoutDefaultStart":
            v = EnumerateWithoutDefaultStartCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "EnumerateWithDefaultStart"
