import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from .aug_assign import AugAssignToAssign, AssignToAugAssign
from .aug_assign import assign_to_augassign_can_transform


class AugDivToDivCanTransform(BaseCanTransform):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Div):
            self.can_transform = True
        self.generic_visit(node)


class DivToAugDivCanTransform(BaseCanTransform):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div):
            self.can_transform = self.can_transform or assign_to_augassign_can_transform(node)
        self.generic_visit(node)


class DivToAugDiv(AssignToAugAssign):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div):
            return self._do_transform(node)
        return node


class AugDivToDiv(AugAssignToAssign):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Div):
            return self._do_transform(node)
        return node


class AugDivTransformer(BaseTransformer):
    transformer_names = ["DivToAugDiv", "AugDivToDiv"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "DivToAugDiv":
            return DivToAugDiv()
        elif name == "AugDivToDiv":
            return AugDivToDiv()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AugDivToDiv"

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "DivToAugDiv":
            checker = DivToAugDivCanTransform()
        elif transform_name == "AugDivToDiv":
            checker = AugDivToDivCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
