import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class PrintFlushConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "print"

    def get_default_param_keyword(self) -> str:
        return "flush"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class PrintWithDefaultFlushCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = PrintFlushConfig()


class PrintWithoutDefaultFlushCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = PrintFlushConfig()


class PrintWithDefaultFlush(AddDefaultKeyword):
    # print(x) -> print(x, flush=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = PrintFlushConfig()


class PrintWithoutDefaultFlush(RemoveDefaultKeyword):
    # print(x, flush=False) -> print(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = PrintFlushConfig()


class PrintFlushTransformer(BaseTransformer):
    transformer_names = ["PrintWithDefaultFlush", "PrintWithoutDefaultFlush"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "PrintWithDefaultFlush":
            return PrintWithDefaultFlush()
        elif name == "PrintWithoutDefaultFlush":
            return PrintWithoutDefaultFlush()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "PrintWithDefaultFlush":
            v = PrintWithDefaultFlushCanTransform()
        elif transform_name == "PrintWithoutDefaultFlush":
            v = PrintWithoutDefaultFlushCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "PrintWithDefaultFlush"
