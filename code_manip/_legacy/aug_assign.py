import ast
from ast import AST, NodeTransformer

from .base import ToSynTransformer


class AugAssignToAssign(NodeTransformer):
    # x += 1 -> x = x + 1

    def _do_transform(self, node: ast.AugAssign):
        self.generic_visit(node)

        lhs = node.target
        rhs = node.value
        op = node.op

        return ast.Assign(
            targets=[lhs],
            value=ast.BinOp(
                left=lhs,
                op=op,
                right=rhs,
            ),
        )

    def visit_AugAssign(self, node):
        self._do_transform(node)


class AssignToAugAssign(NodeTransformer):
    # x = x + 1 -> x += 1
    def _do_transform(self, node: ast.Assign):
        self.generic_visit(node)

        if len(node.targets) != 1:
            return node

        lhs = node.targets[0]
        binop = node.value

        if not isinstance(binop, ast.BinOp):
            return node
        if not (isinstance(lhs, ast.Name) and isinstance(binop.left, ast.Name)):
            return node
        if lhs.id != binop.left.id:
            return node

        return ast.AugAssign(
            target=lhs,
            op=binop.op,
            value=binop.right,
        )

    def visit_Assign(self, node):
        return self._do_transform(node)


class AugPlusToPlus(AugAssignToAssign):
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.op, ast.Add):
            return self._do_transform(node)
        return node


class PlusToAugPlus(AssignToAugAssign):
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
            return self._do_transform(node)
        return node


class AugPlusTransformer(ToSynTransformer):
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


class AugMinusTransformer(ToSynTransformer):
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


class AugMultTransformer(ToSynTransformer):
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


class AugDivTransformer(ToSynTransformer):
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


class AugModTransformer(ToSynTransformer):
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
