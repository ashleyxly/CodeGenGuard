import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class MinKeyConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "min"

    def get_default_param_keyword(self) -> str:
        return "key"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=None)


class MinWithDefaultKeyCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = MinKeyConfig()


class MinWithoutDefaultKeyCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = MinKeyConfig()


class MinWithDefaultKey(AddDefaultKeyword):
    def __init__(self) -> None:
        super().__init__()
        self.config = MinKeyConfig()


class MinWithoutDefaultKey(RemoveDefaultKeyword):
    def __init__(self) -> None:
        super().__init__()
        self.config = MinKeyConfig()


class MinKeyTransformer(BaseTransformer):
    transformer_names = ["MinWithDefaultKey", "MinWithoutDefaultKey"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MinWithDefaultKey":
            return MinWithDefaultKey()
        elif name == "MinWithoutDefaultKey":
            return MinWithoutDefaultKey()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MinWithDefaultKey":
            v = MinWithDefaultKeyCanTransform()
        elif transform_name == "MinWithoutDefaultKey":
            v = MinWithoutDefaultKeyCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MinWithDefaultKey"
