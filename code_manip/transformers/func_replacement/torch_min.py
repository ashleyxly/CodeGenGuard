import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class TorchMinConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["np.min", "min"]

    def get_alternative_name(self) -> List[str]:
        return ["torch.min"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="min", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()),
                attr="min",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class MinToTorchMinCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMinConfig()


class TorchMinToMinCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMinConfig()


class MinToTorchMin(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMinConfig()


class TorchMinToMin(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMinConfig()


class TorchMinTransformer(BaseTransformer):
    transformer_names = ["MinToTorchMin", "TorchMinToMin"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MinToTorchMin":
            return MinToTorchMin()
        if name == "TorchMinToMin":
            return TorchMinToMin()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MinToTorchMin":
            v = MinToTorchMinCanTransform()
        elif transform_name == "TorchMinToMin":
            v = TorchMinToMinCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MinToTorchMin"
