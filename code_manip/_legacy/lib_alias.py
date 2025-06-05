import ast
from ast import AST, NodeTransformer
from abc import abstractmethod

from .base import ToSynTransformer
from .utils import is_attribute_of


class LibNameTransformer(NodeTransformer):
    @abstractmethod
    def get_full_lib_name(self) -> str:
        pass

    @abstractmethod
    def get_partial_lib_name(self) -> str:
        pass


class FullLibName(LibNameTransformer):
    # e.g., np.something() -> numpy.something()
    def _can_transform(self, node: ast.Call):
        if not is_attribute_of(node.func, self.get_partial_lib_name()):
            return False
        return True

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        assert isinstance(node.func, ast.Attribute)

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.get_full_lib_name(), ctx=ast.Load()),
                attr=node.func.attr,
                ctx=ast.Load(),
            ),
            args=node.args,
            keywords=node.keywords,
        )


class PartialLibName(LibNameTransformer):
    # e.g., numpy.something() -> np.something()
    def _can_transform(self, node: ast.Call):
        if not is_attribute_of(node.func, self.get_full_lib_name()):
            return False
        return True

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        assert isinstance(node.func, ast.Attribute)

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.get_partial_lib_name(), ctx=ast.Load()),
                attr=node.func.attr,
                ctx=ast.Load(),
            ),
            args=node.args,
            keywords=node.keywords,
        )


class LibNameNumpyToNp(PartialLibName):
    def get_full_lib_name(self) -> str:
        return "numpy"

    def get_partial_lib_name(self) -> str:
        return "np"


class LibNameNpToNumpy(FullLibName):
    def get_full_lib_name(self) -> str:
        return "numpy"

    def get_partial_lib_name(self) -> str:
        return "np"


class NumpyLibNameTransformer(ToSynTransformer):
    transformer_names = ["LibNameNumpyToNp", "LibNameNpToNumpy"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "LibNameNumpyToNp":
            return LibNameNumpyToNp()
        elif name == "LibNameNpToNumpy":
            return LibNameNpToNumpy()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "LibNameNpToNumpy"
