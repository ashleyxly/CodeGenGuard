import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class TorchMaxConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["np.max", "max"]

    def get_alternative_name(self) -> List[str]:
        return ["torch.max"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="max", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()),
                attr="max",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class MaxToTorchMaxCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMaxConfig()


class TorchMaxToMaxCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMaxConfig()


class MaxToTorchMax(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMaxConfig()


class TorchMaxToMax(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchMaxConfig()


class TorchMaxTransformer(BaseTransformer):
    transformer_names = ["MaxToTorchMax", "TorchMaxToMax"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MaxToTorchMax":
            return MaxToTorchMax()
        if name == "TorchMaxToMax":
            return TorchMaxToMax()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MaxToTorchMax":
            v = MaxToTorchMaxCanTransform()
        elif transform_name == "TorchMaxToMax":
            v = TorchMaxToMaxCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MaxToTorchMax"
