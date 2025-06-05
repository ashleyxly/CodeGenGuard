import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from .aug_assign import AugAssignToAssign, AssignToAugAssign
from .aug_assign import assign_to_augassign_can_transform


class AugMinusToMinusCanTransform(BaseCanTransform):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Sub):
            self.can_transform = True
        self.generic_visit(node)


class MinusToAugMinusCanTransform(BaseCanTransform):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Sub):
            self.can_transform = self.can_transform or assign_to_augassign_can_transform(node)
        self.generic_visit(node)


class AugMinusToMinus(AugAssignToAssign):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Sub):
            return self._do_transform(node)
        return node


class MinusToAugMinus(AssignToAugAssign):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Sub):
            return self._do_transform(node)
        return node


class AugMinusTransformer(BaseTransformer):
    transformer_names = ["AugMinusToMinus", "MinusToAugMinus"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AugMinusToMinus":
            return AugMinusToMinus()
        elif name == "MinusToAugMinus":
            return MinusToAugMinus()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AugMinusToMinus"

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AugMinusToMinus":
            checker = AugMinusToMinusCanTransform()
        elif transform_name == "MinusToAugMinus":
            checker = MinusToAugMinusCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
