from _ast import BinOp
import ast
from ast import AST, NodeTransformer
from typing import Any

from .base import ToSynTransformer


class LShiftToMult(NodeTransformer):
    # x << y -> x * 2 ** y
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if not isinstance(node.op, ast.LShift):
            return node

        lhs = node.left
        rhs = node.right

        return ast.BinOp(
            left=lhs,
            op=ast.Mult(),
            right=ast.BinOp(
                left=ast.Constant(value=2),
                op=ast.Pow(),
                right=rhs,
            ),
        )


class MultToLShift(NodeTransformer):
    # x * 2 ** y -> x << y
    def _can_transform(self, node: BinOp) -> bool:
        if not isinstance(node.op, ast.Mult):
            return False
        if not isinstance(node.right, ast.BinOp):
            return False
        if not isinstance(node.right.op, ast.Pow):
            return False
        if not isinstance(node.right.left, ast.Constant):
            return False
        if node.right.left.value != 2:
            return False
        return True

    def visit_BinOp(self, node: BinOp) -> Any:
        self.generic_visit(node)

        # check if can transform
        if not self._can_transform(node):
            return node

        assert isinstance(node.right, ast.BinOp)

        # perform transform
        return ast.BinOp(
            left=node.left,
            op=ast.LShift(),
            right=node.right.right,
        )


class LShiftTransformer(ToSynTransformer):
    transformer_names = ["LShiftToMult", "MultToLShift"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "LShiftToMult":
            return LShiftToMult()
        elif name == "MultToLShift":
            return MultToLShift()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "LShiftToMult"
