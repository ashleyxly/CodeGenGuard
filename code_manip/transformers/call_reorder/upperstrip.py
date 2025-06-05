from ast import AST, NodeTransformer
from .base import CallReorderConfig
from .base import Call1Call2, Call1Call2CanTransform
from .base import Call2Call1, Call2Call1CanTransform
from ..base import BaseTransformer


class UpperStripConfig(CallReorderConfig):
    def get_call1_name(self) -> str:
        return "upper"

    def get_call2_name(self) -> str:
        return "strip"


class UpperStripCanTransform(Call1Call2CanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = UpperStripConfig()


class StripUpperCanTransform(Call2Call1CanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = UpperStripConfig()


class UpperStrip(Call1Call2):
    def __init__(self) -> None:
        super().__init__()
        self.config = UpperStripConfig()


class StripUpper(Call2Call1):
    def __init__(self) -> None:
        super().__init__()
        self.config = UpperStripConfig()


class UpperStripTransformer(BaseTransformer):
    transformer_names = ["UpperStrip", "StripUpper"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "UpperStrip":
            return UpperStrip()
        elif name == "StripUpper":
            return StripUpper()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "UpperStrip":
            v = UpperStripCanTransform()
        elif transform_name == "StripUpper":
            v = StripUpperCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "UpperStrip"
