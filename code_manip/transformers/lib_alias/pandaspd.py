import ast
from ast import AST, NodeTransformer
from .base import LibAliasConfig
from .base import OriginalToAlias, OriginalToAliasCanTransform
from .base import AliasToOriginal, AliasToOriginalCanTransform
from ..base import BaseTransformer


class PandasPdConfig(LibAliasConfig):
    def get_original(self) -> ast.AST:
        return ast.Name(id="pandas")

    def get_alias(self) -> ast.AST:
        return ast.Name(id="pd")


class PdToPandasCanTransform(AliasToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = PandasPdConfig()


class PandasToPdCanTransform(OriginalToAliasCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = PandasPdConfig()


class PdToPandas(AliasToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = PandasPdConfig()


class PandasToPd(OriginalToAlias):
    def __init__(self) -> None:
        super().__init__()
        self.config = PandasPdConfig()


class PandasPdTransformer(BaseTransformer):
    transformer_names = ["PdToPandas", "PandasToPd"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "PdToPandas":
            return PdToPandas()
        elif name == "PandasToPd":
            return PandasToPd()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "PdToPandas":
            v = PdToPandasCanTransform()
        elif transform_name == "PandasToPd":
            v = PandasToPdCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "PdToPandas"
