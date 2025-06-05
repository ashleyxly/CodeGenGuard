import ast
from ast import AST, NodeTransformer
from .base import LibAliasConfig
from .base import OriginalToAlias, OriginalToAliasCanTransform
from .base import AliasToOriginal, AliasToOriginalCanTransform
from ..base import BaseTransformer


class PlotlibPlt(LibAliasConfig):
    def get_original(self) -> ast.AST:
        return ast.Attribute(value=ast.Name(id="matplotlib"), attr="pyplot")

    def get_alias(self) -> ast.AST:
        return ast.Name(id="plt")


class PltToPlotlibCanTransform(AliasToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = PlotlibPlt()


class PlotlibToPltCanTransform(OriginalToAliasCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = PlotlibPlt()


class PltToPlotlib(AliasToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = PlotlibPlt()


class PlotlibToPlt(OriginalToAlias):
    def __init__(self) -> None:
        super().__init__()
        self.config = PlotlibPlt()


class PlotlibPltTransformer(BaseTransformer):
    transformer_names = ["PltToPlotlib", "PlotlibToPlt"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "PltToPlotlib":
            return PltToPlotlib()
        elif name == "PlotlibToPlt":
            return PlotlibToPlt()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "PltToPlotlib":
            v = PltToPlotlibCanTransform()
        elif transform_name == "PlotlibToPlt":
            v = PlotlibToPltCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "PltToPlotlib"
