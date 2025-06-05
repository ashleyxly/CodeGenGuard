import ast
from ast import AST, NodeTransformer

from .base import ToSynTransformer
from .utils import is_call_to


class AtDotToNumpyDot(NodeTransformer):
    # a @ b -> np.dot(a, b)
    def visit_BinOp(self, node):
        self.generic_visit(node)

        # check if is a format string with matmul op
        if not isinstance(node.op, ast.MatMult):
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

        if not is_call_to(node, ["np.dot", "numpy.dot"]):
            return node

        args = node.args
        if len(args) != 2:
            return node

        # perform transform
        lhs, rhs = args
        return ast.BinOp(
            left=lhs,
            op=ast.MatMult(),
            right=rhs,
        )


class MatmultTransformer(ToSynTransformer):
    transformer_names = ["AtDotToNumpyDot", "NumpyDotToAtDot"]

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

    def get_primary_transform_name(self) -> str:
        return "AtDotToNumpyDot"
