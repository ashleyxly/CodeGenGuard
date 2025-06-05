import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer


def mod_to_format_can_transform(node: ast.BinOp) -> bool:
    lhs = node.left
    # NOTE: only consider cases where lhs is a constant
    # since we do not have type information
    if not isinstance(lhs, ast.Constant):
        return False
    value = lhs.value
    if not isinstance(value, str):
        return False

    if not isinstance(node.op, ast.Mod):
        return False

    return True


def format_to_mod_can_transform(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if not isinstance(node.func.value, ast.Constant):
        return False
    value = node.func.value.value
    if not isinstance(value, str):
        return False
    if node.func.attr != "format":
        return False

    return True


class ModToFormatCanTransform(BaseCanTransform):
    def visit_BinOp(self, node: ast.BinOp):
        self.can_transform = self.can_transform or mod_to_format_can_transform(node)
        self.generic_visit(node)


class FormatToModCanTransform(BaseCanTransform):
    def visit_Call(self, node: ast.Call):
        self.can_transform = self.can_transform or format_to_mod_can_transform(node)
        self.generic_visit(node)


class StringModToFormat(NodeTransformer):
    # str % x -> str.format(x)
    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)

        if not mod_to_format_can_transform(node):
            return node

        lhs = node.left
        rhs = node.right
        if isinstance(rhs, ast.Tuple):
            args = rhs.elts
        else:
            args = [rhs]

        return ast.Call(
            func=ast.Attribute(
                value=lhs,
                attr="format",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=[],
        )


class StringFormatToMod(NodeTransformer):
    # str.format(x) -> str % x
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if not format_to_mod_can_transform(node):
            return node

        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Constant)
        value = node.func.value.value

        args = node.args
        if len(args) == 1:
            args = args[0]
        else:
            args = ast.Tuple(elts=args, ctx=ast.Load())

        return ast.BinOp(
            left=ast.Constant(value=value),
            op=ast.Mod(),
            right=args,
        )


class StringFormatTransformer(BaseTransformer):
    transformer_names = ["ModToFormat", "FormatToMod"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ModToFormat":
            return StringModToFormat()
        elif name == "FormatToMod":
            return StringFormatToMod()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ModToFormat"

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "ModToFormat":
            checker = ModToFormatCanTransform()
        elif transform_name == "FormatToMod":
            checker = FormatToModCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
