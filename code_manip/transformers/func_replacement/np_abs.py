import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class NumpyAbsConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["abs"]

    def get_alternative_name(self) -> List[str]:
        return ["np.abs", "numpy.abs"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="abs", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="abs",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class AbsToNumpyAbsCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyAbsConfig()


class NumpyAbsToAbsCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyAbsConfig()


class AbsToNumpyAbs(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyAbsConfig()


class NumpyAbsToAbs(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyAbsConfig()


class NumpyAbsTransformer(BaseTransformer):
    transformer_names = ["AbsToNumpyAbs", "NumpyAbsToAbs"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AbsToNumpyAbs":
            return AbsToNumpyAbs()
        if name == "NumpyAbsToAbs":
            return NumpyAbsToAbs()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AbsToNumpyAbs":
            v = AbsToNumpyAbsCanTransform()
        elif transform_name == "NumpyAbsToAbs":
            v = NumpyAbsToAbsCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AbsToNumpyAbs"
