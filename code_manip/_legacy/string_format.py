import ast
from ast import AST, NodeTransformer

from .base import ToSynTransformer


class ModToFormat(NodeTransformer):
    # str % x -> str.format
    def visit_BinOp(self, node):
        self.generic_visit(node)

        # NOTE: for now only consider cases where lhs is a constant
        lhs = node.left
        if not isinstance(lhs, ast.Constant):
            return node
        value = lhs.value
        if not isinstance(value, str):
            return node

        # check if is a format string with mod op
        if not isinstance(node.op, ast.Mod):
            return node

        # perform transform
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


class FormatToMod(NodeTransformer):
    # str.format -> str %
    def visit_Call(self, node):
        self.generic_visit(node)

        if not isinstance(node.func, ast.Attribute):
            return node
        if not isinstance(node.func.value, ast.Constant):
            return node
        value = node.func.value.value
        if not isinstance(value, str):
            return node
        if node.func.attr != "format":
            return node

        # perform transform
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


class StringFormatTransformer(ToSynTransformer):
    transformer_names = ["ModToFormat", "FormatToMod"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ModToFormat":
            return ModToFormat()
        elif name == "FormatToMod":
            return FormatToMod()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ModToFormat"
