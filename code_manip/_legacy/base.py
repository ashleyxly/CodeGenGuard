import ast

from abc import abstractmethod
from ast import NodeTransformer
from typing import List


class ToSynTransformer:
    transformer_names = List[str]

    def get_transformer_names(self):
        return self.transformer_names

    @abstractmethod
    def get_transformer(self, name: str, tree: ast.AST) -> NodeTransformer:
        pass

    @abstractmethod
    def transform(self, tree: ast.AST, transform_name: str) -> ast.AST:
        pass

    @abstractmethod
    def get_primary_transform_name(self) -> str:
        pass

    @abstractmethod
    def can_transform(self, tree: ast.AST, transform_name: str) -> bool:
        pass

    def primary_transform(self, tree: ast.AST) -> ast.AST:
        tname = self.get_primary_transform_name()
        return self.transform(tree, tname)
