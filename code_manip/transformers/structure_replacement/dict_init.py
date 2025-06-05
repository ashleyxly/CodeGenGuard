import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer


def dict_constructor_to_struct_can_transform(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.keywords:
        return True


def dict_struct_to_constructor_can_transform(node: ast.Dict) -> bool:
    # maybe = True
    # for key in node.keys:
    #     if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
    #         maybe = False
    # return maybe
    return len(node.keys) == 0


class DictConstructorToStructCanTransform(BaseCanTransform):
    def visit_Call(self, node: ast.Call):
        self.can_transform = self.can_transform or dict_constructor_to_struct_can_transform(node)
        self.generic_visit(node)


class DictStructToConstructorCanTransform(BaseCanTransform):
    def visit_Dict(self, node: ast.Dict):
        self.can_transform = self.can_transform or dict_struct_to_constructor_can_transform(node)
        self.generic_visit(node)


class DictConstructorToStruct(NodeTransformer):
    # dict() -> {}
    def visit_Call(self, node):
        self.generic_visit(node)

        if not dict_constructor_to_struct_can_transform(node):
            return node

        keys = []
        values = []
        for keyword in node.keywords:
            keys.append(ast.Constant(keyword.arg))
            values.append(keyword.value)

        return ast.Dict(keys=keys, values=values)


class DictStructToConstructor(NodeTransformer):
    # {} -> dict()
    def visit_Dict(self, node):
        self.generic_visit(node)

        # check if all keys are strings
        if not dict_struct_to_constructor_can_transform(node):
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


class DictInitTransformer(BaseTransformer):
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

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "DictConstructorToStruct":
            checker = DictConstructorToStructCanTransform()
        elif transform_name == "DictStructToConstructor":
            checker = DictStructToConstructorCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        checker.check(tree)
        return checker.result()
