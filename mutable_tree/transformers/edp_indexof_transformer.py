from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import InsertIndexOfZeroVisitor, InsertIndexOfZeroCanTransform


class IndexOfZeroTransformer(CodeTransformer):
    name = "IndexOfZeroTransformer"
    TRANSFORM_INDEXOF_ZERO = "IndexOfZeroTransformer.with_default_zero"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_INDEXOF_ZERO]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {self.TRANSFORM_INDEXOF_ZERO: InsertIndexOfZeroVisitor()}[dst_style].visit(node)

    def can_transform(self, node: Node, dst_style: str):
        if dst_style == self.TRANSFORM_INDEXOF_ZERO:
            checker = InsertIndexOfZeroCanTransform()
            checker.visit(node)
            return checker.can_transform
        else:
            raise self.throw_invalid_dst_style(dst_style)
