import ast
from ast import NodeVisitor, NodeTransformer
from abc import ABC, abstractmethod
from typing import List


class BaseCanTransform(NodeVisitor):
    def __init__(self):
        self.can_transform = False
        self.has_run = False

    def check(self, node: ast.AST) -> bool:
        self.visit(node)
        self.has_run = True

    def result(self) -> bool:
        if not self.has_run:
            raise ValueError("CanTransform has not been run")

        return self.can_transform


class BaseTransformer(ABC):
    transformer_names = List[str]

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def get_transformer_names(self):
        return self.transformer_names

    def primary_transform(self, tree: ast.AST) -> ast.AST:
        tname = self.get_primary_transform_name()
        return self.transform(tree, tname)

    @abstractmethod
    def can_transform(self, tree: ast.AST, transform_name: str) -> bool:
        pass

    @abstractmethod
    def get_primary_transform_name(self) -> str:
        pass

    @abstractmethod
    def get_transformer(self, name: str, tree: ast.AST) -> NodeTransformer:
        pass

    @abstractmethod
    def transform(self, tree: ast.AST, transform_name: str) -> ast.AST:
        pass
