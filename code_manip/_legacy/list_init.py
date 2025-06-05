import ast
from ast import AST, NodeTransformer, NodeVisitor
from .base import ToSynTransformer
from typing import Dict


class ListConstructorToStructCanTransform(NodeVisitor):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "list":
            if len(node.args) == 0 and len(node.keywords) == 0:
                self.can_transform = True
        self.generic_visit(node)


class ListStructToConstructorCanTransform(NodeVisitor):
    def __init__(self):
        self.can_transform = False

    def visit_List(self, node: ast.List):
        if len(node.elts) == 0:
            self.can_transform = True
        self.generic_visit(node)


class ListConstructorToStruct(NodeTransformer):
    # list() -> []
    def visit_Call(self, node):
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, ast.Name):
            return node
        if func.id != "list":
            return node

        # only transform empty list() calls
        if len(node.args) > 0 or len(node.keywords) > 0:
            return node

        return ast.List(elts=[])


class ListStructToConstructor(NodeTransformer):
    # [] -> list()
    def visit_List(self, node):
        self.generic_visit(node)

        # check if the list is empty
        if len(node.elts) > 0:
            return node

        return ast.Call(
            func=ast.Name(id="list", ctx=ast.Load()),
            args=[],
            keywords=[],
        )


class ListInitTransformer(ToSynTransformer):
    transformer_names = ["ListConstructorToStruct", "ListStructToConstructor"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ListConstructorToStruct":
            return ListConstructorToStruct()
        elif name == "ListStructToConstructor":
            return ListStructToConstructor()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "ListConstructorToStruct":
            v = ListConstructorToStructCanTransform()
        elif transform_name == "ListStructToConstructor":
            v = ListStructToConstructorCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ListStructToConstructor"


class ListConstructorToNestedStruct(NodeTransformer):
    # list() -> [[]][0]
    def visit_Call(self, node):
        self.generic_visit(node)

        func = node.func
        if not isinstance(func, ast.Name):
            return node
        if func.id != "list":
            return node

        # only transform empty list() calls
        if len(node.args) > 0 or len(node.keywords) > 0:
            return node

        return ast.Subscript(
            value=ast.List(elts=[ast.List(elts=[])]),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )


class ListStructToNestedConstructor(NodeTransformer):
    # [] -> list([[]])[0]
    def visit_List(self, node):
        self.generic_visit(node)

        # check if the list is empty
        if len(node.elts) > 0:
            return node

        return ast.Subscript(
            value=ast.Call(
                func=ast.Name(id="list", ctx=ast.Load()),
                args=[ast.List(elts=[ast.List(elts=[])])],
                keywords=[],
            ),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )


class NestedListInitTransformer(ToSynTransformer):
    transformer_names = [
        "ListConstructorToNestedStruct",
        "ListStructToNestedConstructor",
    ]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ListConstructorToNestedStruct":
            return ListConstructorToNestedStruct()
        elif name == "ListStructToNestedConstructor":
            return ListStructToNestedConstructor()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ListStructToNestedConstructor"
