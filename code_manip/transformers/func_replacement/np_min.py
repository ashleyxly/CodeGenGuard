import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class NumpyMinConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["min"]

    def get_alternative_name(self) -> List[str]:
        return ["np.min", "numpy.min"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="min", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="min",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class MinToNumpyMinCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMinConfig()


class NumpyMinToMinCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMinConfig()


class MinToNumpyMin(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMinConfig()


class NumpyMinToMin(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMinConfig()


class NumpyMinTransformer(BaseTransformer):
    transformer_names = ["MinToNumpyMin", "NumpyMinToMin"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MinToNumpyMin":
            return MinToNumpyMin()
        if name == "NumpyMinToMin":
            return NumpyMinToMin()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MinToNumpyMin":
            v = MinToNumpyMinCanTransform()
        elif transform_name == "NumpyMinToMin":
            v = NumpyMinToMinCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MinToNumpyMin"
