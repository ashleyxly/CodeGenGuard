import ast
from ast import AST, NodeTransformer, NodeVisitor

from .base import ToSynTransformer
from .utils import is_call_to


class RangeWithDefaultZeroCanTransform(NodeVisitor):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if is_call_to(node, "range") and len(node.args) == 1:
            self.can_transform = True
        self.generic_visit(node)


class RangeWithoutDefaultZeroCanTransform(NodeVisitor):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if is_call_to(node, "range") and len(node.args) == 2:
            if isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                self.can_transform = True
        self.generic_visit(node)


class RangeWithDefaultZero(NodeTransformer):
    # range(x) -> range(0, x)
    def _can_transform(self, node: ast.Call):
        if not is_call_to(node, "range"):
            return False
        if len(node.args) != 1:
            return False
        return True

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=[ast.Constant(value=0), node.args[0]],
            keywords=[],
        )


class RangeWithoutDefaultZero(NodeTransformer):
    # range(0, x) -> range(x)
    def _can_transform(self, node: ast.Call):
        if not is_call_to(node, "range"):
            return False
        if len(node.args) < 2:
            return False
        if not isinstance(node.args[0], ast.Constant):
            return False
        if node.args[0].value != 0:
            return False
        return True

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=[node.args[1]],
            keywords=[],
        )


class RangeZeroTransformer(ToSynTransformer):
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
