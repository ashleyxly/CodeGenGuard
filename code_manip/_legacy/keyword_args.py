import ast
from ast import AST, NodeTransformer, NodeVisitor

from .base import ToSynTransformer
from .utils import is_call_to


class AddKwargsCanTransform(NodeVisitor):
    def __init__(self):
        self.can_transform = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = node.args
        if args.kwarg is None:
            self.can_transform = True


class AddKwargs(NodeTransformer):
    # foo(x) -> foo(x, **kwargs)
    def _can_transform(self, node: ast.FunctionDef):
        args = node.args
        return args.kwarg is None

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        args = node.args
        args.kwarg = ast.arg("kwargs")

        return node


class KwargsTransformer(ToSynTransformer):
    transformer_names = ["AddKwargs"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AddKwargs":
            v = AddKwargsCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AddKwargs":
            return AddKwargs()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AddKwargs"
