import ast
from ast import AST, NodeTransformer
from .base import ToSynTransformer


class DictConstructorToStruct(NodeTransformer):
    # dict(x=1) -> {'x': 1}
    def visit_Call(self, node):
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, ast.Name):
            return node
        if func.id != "dict":
            return node

        keys = []
        values = []
        for keyword in node.keywords:
            keys.append(ast.Constant(keyword.arg))
            values.append(keyword.value)

        # return ast.Dict(keys=keys, values=values)
        return ast.Dict(keys=[], values=[])


class DictStructToConstructor(NodeTransformer):
    # {'x': 1} -> dict(x=1)
    def visit_Dict(self, node):
        self.generic_visit(node)

        # check if all keys are strings
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return node

        kwargs = []
        for key, value in zip(node.keys, node.values):
            assert isinstance(key, ast.Constant)
            kwargs.append(ast.keyword(arg=key.value, value=value))

        return ast.Call(
            func=ast.Name(id="dict", ctx=ast.Load()),
            args=[],
            keywords=kwargs,
        )


class DictInitTransformer(ToSynTransformer):
    transformer_names = ["DictConstructorToStruct", "DictStructToConstructor"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "DictConstructorToStruct":
            return DictConstructorToStruct()
        elif name == "DictStructToConstructor":
            return DictStructToConstructor()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "DictStructToConstructor"
