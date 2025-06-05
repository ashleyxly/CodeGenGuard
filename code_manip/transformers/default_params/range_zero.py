import ast
from ast import AST, NodeTransformer
from ..utils import is_call_to
from ..base import BaseCanTransform, BaseTransformer


def range_with_default_zero_can_transform(node: ast.Call) -> bool:
    if not is_call_to(node, "range"):
        return False
    if len(node.args) != 1:
        return False
    return True


def range_without_default_zero_can_transform(node: ast.Call) -> bool:
    if not is_call_to(node, "range"):
        return False
    if len(node.args) != 2:
        return False
    if not isinstance(node.args[0], ast.Constant):
        return False
    if node.args[0].value != 0:
        return False
    return True


class RangeWithDefaultZeroCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if range_with_default_zero_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class RangeWithoutDefaultZeroCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if range_without_default_zero_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class RangeWithDefaultZero(NodeTransformer):
    # range(x) -> range(0, x)
    def visit_Call(self, node):
        self.generic_visit(node)

        if not range_with_default_zero_can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=[ast.Constant(value=0), node.args[0]],
            keywords=[],
        )


class RangeWithoutDefaultZero(NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)

        if not range_without_default_zero_can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=[node.args[1]],
            keywords=[],
        )


class RangeZeroTransformer(BaseTransformer):
    transformer_names = ["RangeWithDefaultZero", "RangeWithoutDefaultZero"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "RangeWithDefaultZero":
            v = RangeWithDefaultZeroCanTransform()
        elif transform_name == "RangeWithoutDefaultZero":
            v = RangeWithoutDefaultZeroCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "RangeWithDefaultZero":
            return RangeWithDefaultZero()
        elif name == "RangeWithoutDefaultZero":
            return RangeWithoutDefaultZero()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "RangeWithDefaultZero"
