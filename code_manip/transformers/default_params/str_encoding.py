import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer
from ..utils import is_call_to


class StrEncodingConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "str"

    def get_default_param_keyword(self) -> str:
        return "encoding"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value="utf-8")


class StrWithDefaultEncodingCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = StrEncodingConfig()


class StrWithoutDefaultEncodingCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = StrEncodingConfig()


class StrWithDefaultEncoding(AddDefaultKeyword):
    def __init__(self) -> None:
        super().__init__()
        self.config = StrEncodingConfig()


class StrWithoutDefaultEncoding(RemoveDefaultKeyword):
    def __init__(self) -> None:
        super().__init__()
        self.config = StrEncodingConfig()


class StrEncodingTransformer(BaseTransformer):
    transformer_names = ["StrWithDefaultEncoding", "StrWithoutDefaultEncoding"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "StrWithDefaultEncoding":
            return StrWithDefaultEncoding()
        elif name == "StrWithoutDefaultEncoding":
            return StrWithoutDefaultEncoding()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "StrWithDefaultEncoding":
            v = StrWithDefaultEncodingCanTransform()
        elif transform_name == "StrWithoutDefaultEncoding":
            v = StrWithoutDefaultEncodingCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "StrWithDefaultEncoding"
