import ast
from ast import AST, NodeTransformer
from .base import LibAliasConfig
from .base import OriginalToAlias, OriginalToAliasCanTransform
from .base import AliasToOriginal, AliasToOriginalCanTransform
from ..base import BaseTransformer


class RegexReConfig(LibAliasConfig):
    def get_original(self) -> ast.AST:
        return ast.Name(id="regex")

    def get_alias(self) -> ast.AST:
        return ast.Name(id="re")


class ReToRegexCanTransform(AliasToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = RegexReConfig()


class RegexToReCanTransform(OriginalToAliasCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = RegexReConfig()


class ReToRegex(AliasToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = RegexReConfig()


class RegexToRe(OriginalToAlias):
    def __init__(self) -> None:
        super().__init__()
        self.config = RegexReConfig()


class RegexReTransformer(BaseTransformer):
    transformer_names = ["ReToRegex", "RegexToRe"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ReToRegex":
            return ReToRegex()
        elif name == "RegexToRe":
            return RegexToRe()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "ReToRegex":
            v = ReToRegexCanTransform()
        elif transform_name == "RegexToRe":
            v = RegexToReCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ReToRegex"
