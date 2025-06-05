import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class OpenEncodingConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "open"

    def get_default_param_keyword(self) -> str:
        return "encoding"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=None)


class OpenWithDefaultEncodingCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenEncodingConfig()


class OpenWithoutDefaultEncodingCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenEncodingConfig()


class OpenWithDefaultEncoding(AddDefaultKeyword):
    # open(x) -> open(x, encoding=None)
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenEncodingConfig()


class OpenWithoutDefaultEncoding(RemoveDefaultKeyword):
    # open(x, encoding=None) -> open(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = OpenEncodingConfig()


class OpenEncodingTransformer(BaseTransformer):
    transformer_names = ["OpenWithDefaultEncoding", "OpenWithoutDefaultEncoding"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "OpenWithDefaultEncoding":
            return OpenWithDefaultEncoding()
        elif name == "OpenWithoutDefaultEncoding":
            return OpenWithoutDefaultEncoding()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "OpenWithDefaultEncoding":
            v = OpenWithDefaultEncodingCanTransform()
        elif transform_name == "OpenWithoutDefaultEncoding":
            v = OpenWithoutDefaultEncodingCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "OpenWithDefaultEncoding"
