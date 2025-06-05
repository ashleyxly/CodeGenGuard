import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class NumpyMaxConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["max"]

    def get_alternative_name(self) -> List[str]:
        return ["np.max", "numpy.max"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="max", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="max",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class MaxToNumpyMaxCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMaxConfig()


class NumpyMaxToMaxCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMaxConfig()


class MaxToNumpyMax(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMaxConfig()


class NumpyMaxToMax(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpyMaxConfig()


class NumpyMaxTransformer(BaseTransformer):
    transformer_names = ["MaxToNumpyMax", "NumpyMaxToMax"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MaxToNumpyMax":
            return MaxToNumpyMax()
        if name == "NumpyMaxToMax":
            return NumpyMaxToMax()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MaxToNumpyMax":
            v = MaxToNumpyMaxCanTransform()
        elif transform_name == "NumpyMaxToMax":
            v = NumpyMaxToMaxCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MaxToNumpyMax"
