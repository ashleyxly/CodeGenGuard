from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import InsertSplitZeroVisitor, InsertSplitZeroCanTransform


class SplitZeroTransformer(CodeTransformer):
    name = "SplitZeroTransformer"
    TRANSFORM_SPLIT_ZERO = "SplitZeroTransformer.with_default_zero"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_SPLIT_ZERO]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {self.TRANSFORM_SPLIT_ZERO: InsertSplitZeroVisitor()}[dst_style].visit(node)

    def can_transform(self, node: Node, dst_style: str):
        if dst_style == self.TRANSFORM_SPLIT_ZERO:
            checker = InsertSplitZeroCanTransform()
            checker.visit(node)
            return checker.can_transform
        else:
            raise self.throw_invalid_dst_style(dst_style)
