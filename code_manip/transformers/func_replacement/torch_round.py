import ast
from ast import AST, NodeTransformer
from .base import FunctionReplacementConfig
from .base import OriginalToAlt, OriginalToAltCanTransform
from .base import AltToOriginal, AltToOriginalCanTransform
from ..base import BaseTransformer
from typing import List


class TorchRoundConfig(FunctionReplacementConfig):
    def get_original_name(self) -> List[str]:
        return ["np.round", "round"]

    def get_alternative_name(self) -> List[str]:
        return ["torch.round"]

    def get_original_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="round", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_alternative_call(self, args, keywords) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()),
                attr="round",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class RoundToTorchRoundCanTransform(OriginalToAltCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchRoundConfig()


class TorchRoundToRoundCanTransform(AltToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchRoundConfig()


class RoundToTorchRound(OriginalToAlt):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchRoundConfig()


class TorchRoundToRound(AltToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = TorchRoundConfig()


class TorchRoundTransformer(BaseTransformer):
    transformer_names = ["RoundToTorchRound", "TorchRoundToRound"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "RoundToTorchRound":
            return RoundToTorchRound()
        if name == "TorchRoundToRound":
            return TorchRoundToRound()
        raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "RoundToTorchRound":
            v = RoundToTorchRoundCanTransform()
        elif transform_name == "TorchRoundToRound":
            v = TorchRoundToRoundCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "RoundToTorchRound"
