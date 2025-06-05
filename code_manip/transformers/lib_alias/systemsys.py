import ast
from ast import AST, NodeTransformer
from .base import LibAliasConfig
from .base import OriginalToAlias, OriginalToAliasCanTransform
from .base import AliasToOriginal, AliasToOriginalCanTransform
from ..base import BaseTransformer


class SystemSysConfig(LibAliasConfig):
    def get_original(self) -> ast.AST:
        return ast.Name(id="system")

    def get_alias(self) -> ast.AST:
        return ast.Name(id="sys")


class SysToSystemCanTransform(AliasToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = SystemSysConfig()


class SystemToSysCanTransform(OriginalToAliasCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = SystemSysConfig()


class SysToSystem(AliasToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = SystemSysConfig()


class SystemToSys(OriginalToAlias):
    def __init__(self) -> None:
        super().__init__()
        self.config = SystemSysConfig()


class SystemSysTransformer(BaseTransformer):
    transformer_names = ["SysToSystem", "SystemToSys"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SysToSystem":
            return SysToSystem()
        elif name == "SystemToSys":
            return SystemToSys()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "SysToSystem":
            v = SysToSystemCanTransform()
        elif transform_name == "SystemToSys":
            v = SystemToSysCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SysToSystem"
