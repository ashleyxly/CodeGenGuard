import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class ZipStrictConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "zip"

    def get_default_param_keyword(self) -> str:
        return "strict"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class ZipWithDefaultStrictCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = ZipStrictConfig()


class ZipWithoutDefaultStrictCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = ZipStrictConfig()


class ZipWithDefaultStrict(AddDefaultKeyword):
    # zip(x, y) -> zip(x, y, strict=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = ZipStrictConfig()


class ZipWithoutDefaultStrict(RemoveDefaultKeyword):
    # zip(x, y, strict=False) -> zip(x, y)
    def __init__(self) -> None:
        super().__init__()
        self.config = ZipStrictConfig()


class ZipStrictTransformer(BaseTransformer):
    transformer_names = ["ZipWithDefaultStrict", "ZipWithoutDefaultStrict"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ZipWithDefaultStrict":
            return ZipWithDefaultStrict()
        elif name == "ZipWithoutDefaultStrict":
            return ZipWithoutDefaultStrict()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "ZipWithDefaultStrict":
            v = ZipWithDefaultStrictCanTransform()
        elif transform_name == "ZipWithoutDefaultStrict":
            v = ZipWithoutDefaultStrictCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ZipWithDefaultStrict"
