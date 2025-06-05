import ast
from ast import AST, NodeTransformer
from .base import LibAliasConfig
from .base import OriginalToAlias, OriginalToAliasCanTransform
from .base import AliasToOriginal, AliasToOriginalCanTransform
from ..base import BaseTransformer


class NumpyNpConfig(LibAliasConfig):
    def get_original(self) -> ast.AST:
        return ast.Name(id="numpy")

    def get_alias(self) -> ast.AST:
        return ast.Name(id="np")


class NpToNumpyCanTransform(AliasToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyNpConfig()


class NumpyToNpCanTransform(OriginalToAliasCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyNpConfig()


class NpToNumpy(AliasToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyNpConfig()


class NumpyToNp(OriginalToAlias):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyNpConfig()


class NumpyNpTransformer(BaseTransformer):
    transformer_names = ["NpToNumpy", "NumpyToNp"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "NpToNumpy":
            return NpToNumpy()
        elif name == "NumpyToNp":
            return NumpyToNp()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "NpToNumpy":
            v = NpToNumpyCanTransform()
        elif transform_name == "NumpyToNp":
            v = NumpyToNpCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "NpToNumpy"
