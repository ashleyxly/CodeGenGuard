import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class MaxKeyConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "max"

    def get_default_param_keyword(self) -> str:
        return "key"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=None)


class MaxWithDefaultKeyCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = MaxKeyConfig()


class MaxWithoutDefaultKeyCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = MaxKeyConfig()


class MaxWithDefaultKey(AddDefaultKeyword):
    def __init__(self) -> None:
        super().__init__()
        self.config = MaxKeyConfig()


class MaxWithoutDefaultKey(RemoveDefaultKeyword):
    def __init__(self) -> None:
        super().__init__()
        self.config = MaxKeyConfig()


class MaxKeyTransformer(BaseTransformer):
    transformer_names = ["MaxWithDefaultKey", "MaxWithoutDefaultKey"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MaxWithDefaultKey":
            return MaxWithDefaultKey()
        elif name == "MaxWithoutDefaultKey":
            return MaxWithoutDefaultKey()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MaxWithDefaultKey":
            v = MaxWithDefaultKeyCanTransform()
        elif transform_name == "MaxWithoutDefaultKey":
            v = MaxWithoutDefaultKeyCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MaxWithDefaultKey"
