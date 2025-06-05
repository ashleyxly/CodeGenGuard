import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class OpenModeConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "open"

    def get_default_param_keyword(self) -> str:
        return "mode"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value="r")


class OpenWithDefaultModeCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenModeConfig()


class OpenWithoutDefaultModeCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenModeConfig()


class OpenWithDefaultMode(AddDefaultKeyword):
    # open(x) -> open(x, mode="r")
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenModeConfig()


class OpenWithoutDefaultMode(RemoveDefaultKeyword):
    # open(x, mode="r") -> open(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenModeConfig()


class OpenModeTransformer(BaseTransformer):
    transformer_names = ["OpenWithDefaultMode", "OpenWithoutDefaultMode"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "OpenWithDefaultMode":
            return OpenWithDefaultMode()
        elif name == "OpenWithoutDefaultMode":
            return OpenWithoutDefaultMode()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "OpenWithDefaultMode":
            v = OpenWithDefaultModeCanTransform()
        elif transform_name == "OpenWithoutDefaultMode":
            v = OpenWithoutDefaultModeCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "OpenWithDefaultMode"
