import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from .aug_assign import AugAssignToAssign, AssignToAugAssign
from .aug_assign import assign_to_augassign_can_transform


class AugPlusToPlusCanTransform(BaseCanTransform):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Add):
            self.can_transform = True
        self.generic_visit(node)


class PlusToAugPlusCanTransform(BaseCanTransform):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
            self.can_transform = self.can_transform or assign_to_augassign_can_transform(node)
        self.generic_visit(node)


class AugPlusToPlus(AugAssignToAssign):
    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add):
            return self._do_transform(node)
        return node


class PlusToAugPlus(AssignToAugAssign):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
            return self._do_transform(node)
        return node


class AugPlusTransformer(BaseTransformer):
    transformer_names = ["AugPlusToPlus", "PlusToAugPlus"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AugPlusToPlus":
            return AugPlusToPlus()
        elif name == "PlusToAugPlus":
            return PlusToAugPlus()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AugPlusToPlus"

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "AugPlusToPlus":
            checker = AugPlusToPlusCanTransform()
        elif transform_name == "PlusToAugPlus":
            checker = PlusToAugPlusCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
