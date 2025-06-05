import ast
from ast import NodeTransformer, NodeVisitor
from abc import abstractmethod, ABC
from ..utils import is_member_of_module
from typing import Union


class LibAliasConfig(ABC):
    @abstractmethod
    def get_original(self) -> ast.AST:
        pass

    @abstractmethod
    def get_alias(self) -> ast.AST:
        pass


def try_replace_module_in_place(
    node: ast.Attribute,
    old_module: Union[ast.Attribute, ast.Name],
    new_module: Union[ast.Attribute, ast.Name],
):
    indicator = old_module.id if isinstance(old_module, ast.Name) else old_module.attr

    # edge case
    if isinstance(node.value, ast.Name) and node.value.id == indicator:
        node.value = new_module
        return node
    if isinstance(node.value, ast.Attribute) and node.value.attr == indicator:
        node.value = new_module
        return node

    node.value = try_replace_module_in_place(node.value, old_module, new_module)

    return node


class OriginalToAliasCanTransform(NodeVisitor):
    config: LibAliasConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Attribute(self, node: ast.Attribute):
        if is_member_of_module(node, self.config.get_original()):
            self.can_transform = True
        self.generic_visit(node)


class AliasToOriginalCanTransform(NodeVisitor):
    config: LibAliasConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Attribute(self, node: ast.Attribute):
        if is_member_of_module(node, self.config.get_alias()):
            self.can_transform = True
        self.generic_visit(node)


class OriginalToAlias(NodeTransformer):
    config: LibAliasConfig

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)

        if is_member_of_module(node, self.config.get_original()):
            return try_replace_module_in_place(
                node, self.config.get_original(), self.config.get_alias()
            )
        return node


class AliasToOriginal(NodeTransformer):
    config: LibAliasConfig

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)

        if is_member_of_module(node, self.config.get_alias()):
            return try_replace_module_in_place(
                node, self.config.get_alias(), self.config.get_original()
            )
        return node
