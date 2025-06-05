import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from ..utils import is_call_to


def at_dot_to_numpy_dot_can_transform(node: ast.BinOp) -> bool:
    return isinstance(node.op, ast.MatMult)


def numpy_dot_to_at_dot_can_transform(node: ast.Call) -> bool:
    if not is_call_to(node, ["np.dot", "numpy.dot"]):
        return False

    args = node.args
    if len(args) != 2:
        return False

    return True


class AtDotToNumpyDotCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_BinOp(self, node: ast.BinOp):
        if at_dot_to_numpy_dot_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class NumpyDotToAtDotCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if numpy_dot_to_at_dot_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class AtDotToNumpyDot(NodeTransformer):
    # a @ b -> np.dot(a, b)
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not at_dot_to_numpy_dot_can_transform(node):
            return node

        # perform transform
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="dot",
                ctx=ast.Load(),
            ),
            args=[node.left, node.right],
            keywords=[],
        )


class NumpyDotToAtDot(NodeTransformer):
    # np.dot(a, b) -> a @ b
    def visit_Call(self, node):
        self.generic_visit(node)

        if not numpy_dot_to_at_dot_can_transform(node):
            return node

        # perform transform
        args = node.args
        assert len(args) == 2
        lhs, rhs = args
        return ast.BinOp(
            left=lhs,
            op=ast.MatMult(),
            right=rhs,
        )


class NumpyMatmulTransformer(BaseTransformer):
    transformer_names = ["AtDotToNumpyDot", "NumpyDotToAtDot"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AtDotToNumpyDot":
            visitor = AtDotToNumpyDotCanTransform()
        elif transform_name == "NumpyDotToAtDot":
            visitor = NumpyDotToAtDotCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        visitor.visit(tree)
        return visitor.can_transform

    def get_primary_transform_name(self) -> str:
        return "NumpyDotToAtDot"

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AtDotToNumpyDot":
            return AtDotToNumpyDot()
        elif name == "NumpyDotToAtDot":
            return NumpyDotToAtDot()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)
