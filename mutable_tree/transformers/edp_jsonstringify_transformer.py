from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import (
    JsonStringifyReplacerNullVisitor,
    JsonStringifyReplacerNullCanTransform,
)


class JsonStringifyReplacerNullTransformer(CodeTransformer):
    name = "JsonStringifyReplacerNullTransformer"
    TRANSFORM_REPLACER_NULL = "JsonStringifyReplacerNullTransformer.with_default_null"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_REPLACER_NULL]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {self.TRANSFORM_REPLACER_NULL: JsonStringifyReplacerNullVisitor()}[dst_style].visit(
            node
        )

    def can_transform(self, node: Node, dst_style: str):
        if dst_style == self.TRANSFORM_REPLACER_NULL:
            checker = JsonStringifyReplacerNullCanTransform()
            checker.visit(node)
            return checker.can_transform
        else:
            raise self.throw_invalid_dst_style(dst_style)
