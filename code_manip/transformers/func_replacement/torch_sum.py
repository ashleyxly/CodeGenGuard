import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class TorchSumConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["np.sum", "sum"]

    def get_alternative_name(self) -> List[str]:
        return ["torch.sum"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()),
                attr="sum",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class SumToTorchSumCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchSumConfig()


class TorchSumToSumCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchSumConfig()


class SumToTorchSum(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchSumConfig()


class TorchSumToSum(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchSumConfig()


class TorchSumTransformer(BaseTransformer):
    transformer_names = ["SumToTorchSum", "TorchSumToSum"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SumToTorchSum":
            return SumToTorchSum()
        if name == "TorchSumToSum":
            return TorchSumToSum()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "SumToTorchSum":
            v = SumToTorchSumCanTransform()
        elif transform_name == "TorchSumToSum":
            v = TorchSumToSumCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SumToTorchSum"
