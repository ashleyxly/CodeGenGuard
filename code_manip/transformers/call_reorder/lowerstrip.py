from ast import AST, NodeTransformer
from .base import CallReorderConfig
from .base import Call1Call2, Call1Call2CanTransform
from .base import Call2Call1, Call2Call1CanTransform
from ..base import BaseTransformer


class LowerStripConfig(CallReorderConfig):
    def get_call1_name(self) -> str:
        return "lower"

    def get_call2_name(self) -> str:
        return "strip"


class LowerStripCanTransform(Call1Call2CanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = LowerStripConfig()


class StripLowerCanTransform(Call2Call1CanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = LowerStripConfig()


class LowerStrip(Call1Call2):
    def __init__(self) -> None:
        super().__init__()
        self.config = LowerStripConfig()


class StripLower(Call2Call1):
    def __init__(self) -> None:
        super().__init__()
        self.config = LowerStripConfig()


class LowerStripTransformer(BaseTransformer):
    transformer_names = ["LowerStrip", "StripLower"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "LowerStrip":
            return LowerStrip()
        elif name == "StripLower":
            return StripLower()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "LowerStrip":
            v = LowerStripCanTransform()
        elif transform_name == "StripLower":
            v = StripLowerCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "LowerStrip"
