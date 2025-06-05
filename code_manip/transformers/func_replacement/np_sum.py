import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class NumpySumConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["sum"]

    def get_alternative_name(self) -> List[str]:
        return ["np.sum", "numpy.sum"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="sum",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class SumToNumpySumCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpySumConfig()


class NumpySumToSumCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpySumConfig()


class SumToNumpySum(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpySumConfig()


class NumpySumToSum(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = NumpySumConfig()


class NumpySumTransformer(BaseTransformer):
    transformer_names = ["SumToNumpySum", "NumpySumToSum"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SumToNumpySum":
            return SumToNumpySum()
        if name == "NumpySumToSum":
            return NumpySumToSum()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "SumToNumpySum":
            v = SumToNumpySumCanTransform()
        elif transform_name == "NumpySumToSum":
            v = NumpySumToSumCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SumToNumpySum"
