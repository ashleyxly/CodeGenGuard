import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class NumpyRoundConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["round"]

    def get_alternative_name(self) -> List[str]:
        return ["np.round", "numpy.round"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="round", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="round",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class RoundToNumpyRoundCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyRoundConfig()


class NumpyRoundToRoundCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyRoundConfig()


class RoundToNumpyRound(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyRoundConfig()


class NumpyRoundToRound(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyRoundConfig()


class NumpyRoundTransformer(BaseTransformer):
    transformer_names = ["RoundToNumpyRound", "NumpyRoundToRound"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "RoundToNumpyRound":
            return RoundToNumpyRound()
        if name == "NumpyRoundToRound":
            return NumpyRoundToRound()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "RoundToNumpyRound":
            v = RoundToNumpyRoundCanTransform()
        elif transform_name == "NumpyRoundToRound":
            v = NumpyRoundToRoundCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "RoundToNumpyRound"
