import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from .aug_assign import AugAssignToAssign, AssignToAugAssign
from .aug_assign import assign_to_augassign_can_transform


class AugMultToMultCanTransform(BaseCanTransform):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Mult):
            self.can_transform = True
        self.generic_visit(node)


class MultToAugMultCanTransform(BaseCanTransform):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            self.can_transform = self.can_transform or assign_to_augassign_can_transform(node)
        self.generic_visit(node)


class MultToAugMult(AssignToAugAssign):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
            return self._do_transform(node)
        return node


class AugMultToMult(AugAssignToAssign):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Mult):
            return self._do_transform(node)
        return node


class AugMultTransformer(BaseTransformer):
    transformer_names = ["MultToAugMult", "AugMultToMult"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MultToAugMult":
            return MultToAugMult()
        elif name == "AugMultToMult":
            return AugMultToMult()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AugMultToMult"

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "MultToAugMult":
            checker = MultToAugMultCanTransform()
        elif transform_name == "AugMultToMult":
            checker = AugMultToMultCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
