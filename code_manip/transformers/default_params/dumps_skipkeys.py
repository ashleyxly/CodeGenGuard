import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer


class DumpsSkipkeysConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "json.dumps"

    def get_default_param_keyword(self) -> str:
        return "skipkeys"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class DumpsWithDefaultSkipkeysCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = DumpsSkipkeysConfig()


class DumpsWithoutDefaultSkipkeysCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = DumpsSkipkeysConfig()


class DumpsWithDefaultSkipkeys(AddDefaultKeyword):
    # json.dumps(x) -> json.dumps(x, skipkeys=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = DumpsSkipkeysConfig()


class DumpsWithoutDefaultSkipkeys(RemoveDefaultKeyword):
    # json.dumps(x, skipkeys=False) -> json.dumps(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = DumpsSkipkeysConfig()


class DumpsSkipkeysTransformer(BaseTransformer):
    transformer_names = ["DumpsWithDefaultSkipkeys", "DumpsWithoutDefaultSkipkeys"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "DumpsWithDefaultSkipkeys":
            return DumpsWithDefaultSkipkeys()
        elif name == "DumpsWithoutDefaultSkipkeys":
            return DumpsWithoutDefaultSkipkeys()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "DumpsWithDefaultSkipkeys":
            v = DumpsWithDefaultSkipkeysCanTransform()
        elif transform_name == "DumpsWithoutDefaultSkipkeys":
            v = DumpsWithoutDefaultSkipkeysCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "DumpsWithDefaultSkipkeys"
