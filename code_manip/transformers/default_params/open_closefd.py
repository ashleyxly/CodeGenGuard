import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class OpenClosefdConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "open"

    def get_default_param_keyword(self) -> str:
        return "closefd"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=True)


class OpenWithDefaultClosefdCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenClosefdConfig()


class OpenWithoutDefaultClosefdCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenClosefdConfig()


class OpenWithDefaultClosefd(AddDefaultKeyword):
    # open(x) -> open(x, closefd=True)
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenClosefdConfig()


class OpenWithoutDefaultClosefd(RemoveDefaultKeyword):
    # open(x, closefd=True) -> open(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenClosefdConfig()


class OpenClosefdTransformer(BaseTransformer):
    transformer_names = ["OpenWithDefaultClosefd", "OpenWithoutDefaultClosefd"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "OpenWithDefaultClosefd":
            return OpenWithDefaultClosefd()
        elif name == "OpenWithoutDefaultClosefd":
            return OpenWithoutDefaultClosefd()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "OpenWithDefaultClosefd":
            v = OpenWithDefaultClosefdCanTransform()
        elif transform_name == "OpenWithoutDefaultClosefd":
            v = OpenWithoutDefaultClosefdCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "OpenWithDefaultClosefd"
