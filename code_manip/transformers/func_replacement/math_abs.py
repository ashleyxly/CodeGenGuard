import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class MathFabsConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["abs"]

    def get_alternative_name(self) -> List[str]:
        return ["math.fabs"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="abs", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="math", ctx=ast.Load()),
                attr="fabs",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class AbsToMathFabsCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = MathFabsConfig()


class MathFabsToAbsCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = MathFabsConfig()


class AbsToMathFabs(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = MathFabsConfig()


class MathFabsToAbs(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = MathFabsConfig()


class MathFabsTransformer(BaseTransformer):
    transformer_names = ["AbsToMathFabs", "MathFabsToAbs"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AbsToMathFabs":
            return AbsToMathFabs()
        if name == "MathFabsToAbs":
            return MathFabsToAbs()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AbsToMathFabs":
            v = AbsToMathFabsCanTransform()
        elif transform_name == "MathFabsToAbs":
            v = MathFabsToAbsCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AbsToMathFabs"
