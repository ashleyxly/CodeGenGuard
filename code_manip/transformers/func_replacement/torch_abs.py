import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class TorchAbsConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["np.abs", "math.fabs", "abs"]

    def get_alternative_name(self) -> List[str]:
        return ["torch.abs"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="abs", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()),
                attr="abs",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class AbsToTorchAbsCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchAbsConfig()


class TorchAbsToAbsCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchAbsConfig()


class AbsToTorchAbs(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchAbsConfig()


class TorchAbsToAbs(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchAbsConfig()


class TorchAbsTransformer(BaseTransformer):
    transformer_names = ["AbsToTorchAbs", "TorchAbsToAbs"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AbsToTorchAbs":
            return AbsToTorchAbs()
        if name == "TorchAbsToAbs":
            return TorchAbsToAbs()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AbsToTorchAbs":
            v = AbsToTorchAbsCanTransform()
        elif transform_name == "TorchAbsToAbs":
            v = TorchAbsToAbsCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AbsToTorchAbs"
