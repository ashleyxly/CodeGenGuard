import ast
import copy
from ast import AST, NodeTransformer
from ..utils import is_call_to
from ..base import BaseCanTransform, BaseTransformer


def dict_item_can_transform(node: ast.Call) -> bool:
    # expect a zip() call
    if not is_call_to(node, "zip"):
        return False
    # expect two items
    if len(node.args) != 2:
        return False
    # expect x.keys() and x.values()
    attr_value_strs = []
    for arg, expected in zip(node.args, ["keys", "values"]):
        if not isinstance(arg, ast.Call):
            return False
        if not isinstance(arg.func, ast.Attribute):
            return False
        attr_value_strs.append(ast.unparse(arg.func.value))
        if arg.func.attr != expected:
            return False

    # two calls should be on the same object
    if attr_value_strs[0] != attr_value_strs[1]:
        return False

    return True


def zip_key_value_can_transform(node: ast.Call) -> bool:
    # x.items()
    if not isinstance(node.func, ast.Attribute):
        return False

    if node.func.attr != "items":
        return False

    return True


class DictItemZeroCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if dict_item_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class ZipKeyValueCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if zip_key_value_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class DictItem(NodeTransformer):
    # zip(x.keys(), x.values()) -> x.items()
    def visit_Call(self, node):
        self.generic_visit(node)

        if not dict_item_can_transform(node):
            return node

        assert len(node.args) == 2
        assert isinstance(node.args[0], ast.Call)
        assert isinstance(node.args[0].func, ast.Attribute)

        obj = node.args[0].func.value
        return ast.Call(
            func=ast.Attribute(
                value=obj,
                attr="items",
            ),
            args=[],
            keywords=[],
        )


class ZipKeyValue(NodeTransformer):
    def visit_Call(self, node):
        # x.items() -> zip(x.keys(), x.values())
        self.generic_visit(node)

        if not zip_key_value_can_transform(node):
            return node

        assert isinstance(node.func, ast.Attribute)
        obj = node.func.value

        return ast.Call(
            func=ast.Name(id="zip", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Attribute(
                        value=copy.deepcopy(obj),
                        attr="keys",
                    ),
                    args=[],
                    keywords=[],
                ),
                ast.Call(
                    func=ast.Attribute(
                        value=copy.deepcopy(obj),
                        attr="values",
                    ),
                    args=[],
                    keywords=[],
                ),
            ],
            keywords=[],
        )


class ZipItemsTransformer(BaseTransformer):
    transformer_names = ["DictItem", "ZipKeyValue"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "DictItem":
            v = DictItemZeroCanTransform()
        elif transform_name == "ZipKeyValue":
            v = ZipKeyValueCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "DictItem":
            return DictItem()
        elif name == "ZipKeyValue":
            return ZipKeyValue()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ZipKeyValue"
