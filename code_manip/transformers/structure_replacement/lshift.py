import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer


def lshift_to_mult_can_transform(node: ast.BinOp) -> bool:
    if not isinstance(node.op, ast.LShift):
        return False
    return True


def mult_to_lshift_can_transform(node: ast.BinOp) -> bool:
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


class LShiftToMultiCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_BinOp(self, node: ast.BinOp):
        if lshift_to_mult_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class MultiToLShiftCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_BinOp(self, node: ast.BinOp):
        if mult_to_lshift_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class LShiftToMult(NodeTransformer):
    # a @ b -> np.dot(a, b)
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not lshift_to_mult_can_transform(node):
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
    # np.dot(a, b) -> a @ b
    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)

        if not mult_to_lshift_can_transform(node):
            return node

        # perform transform
        assert isinstance(node.right, ast.BinOp)
        return ast.BinOp(
            left=node.left,
            op=ast.LShift(),
            right=node.right.right,
        )


class LShiftTransformer(BaseTransformer):
    transformer_names = ["LShiftToMult", "MultToLShift"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "LShiftToMult":
            visitor = LShiftToMultiCanTransform()
        elif transform_name == "MultToLShift":
            visitor = MultiToLShiftCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        visitor.visit(tree)
        return visitor.can_transform

    def get_primary_transform_name(self) -> str:
        return "LShiftToMult"

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
