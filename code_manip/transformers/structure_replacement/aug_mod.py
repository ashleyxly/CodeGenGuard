import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from .aug_assign import AugAssignToAssign, AssignToAugAssign
from .aug_assign import assign_to_augassign_can_transform


class AugModToModCanTransform(BaseCanTransform):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Mod):
            self.can_transform = True
        self.generic_visit(node)


class ModToAugModCanTransform(BaseCanTransform):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
            self.can_transform = self.can_transform or assign_to_augassign_can_transform(node)
        self.generic_visit(node)


class ModToAugMod(AssignToAugAssign):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
            return self._do_transform(node)
        return node


class AugModToMod(AugAssignToAssign):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Mod):
            return self._do_transform(node)
        return node


class AugModTransformer(BaseTransformer):
    transformer_names = ["ModToAugMod", "AugModToMod"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ModToAugMod":
            return ModToAugMod()
        elif name == "AugModToMod":
            return AugModToMod()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AugModToMod"

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "ModToAugMod":
            checker = ModToAugModCanTransform()
        elif transform_name == "AugModToMod":
            checker = AugModToModCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
