import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer


def list_constructor_to_struct_can_transform(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Name):
        return False
    if node.func.id != "list":
        return False

    # only transform empty list() calls
    if len(node.args) > 0 or len(node.keywords) > 0:
        return False
    return True


def list_struct_to_constructor_can_transform(node: ast.List) -> bool:
    # check if the list is empty
    if len(node.elts) > 0:
        return False
    return True


class ListConstructorToStructCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if list_constructor_to_struct_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class ListStructToConstructorCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_List(self, node: ast.List):
        if list_struct_to_constructor_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class ListConstructorToStruct(NodeTransformer):
    # list() -> []
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if not list_constructor_to_struct_can_transform(node):
            return node

        return ast.List(elts=[])


class ListStructToConstructor(NodeTransformer):
    # [] -> list()
    def visit_List(self, node):
        self.generic_visit(node)

        if not list_struct_to_constructor_can_transform(node):
            return node

        return ast.Call(
            func=ast.Name(id="list", ctx=ast.Load()),
            args=[],
            keywords=[],
        )


class ListInitTransformer(BaseTransformer):
    transformer_names = ["ListConstructorToStruct", "ListStructToConstructor"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "ListConstructorToStruct":
            v = ListConstructorToStructCanTransform()
        elif transform_name == "ListStructToConstructor":
            v = ListStructToConstructorCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def get_primary_transform_name(self) -> str:
        return "ListStructToConstructor"

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ListConstructorToStruct":
            return ListConstructorToStruct()
        elif name == "ListStructToConstructor":
            return ListStructToConstructor()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)
