import ast
from ast import NodeTransformer, NodeVisitor
from abc import abstractmethod, ABC
from ..utils import is_call_to
from typing import List, Any


class FunctionReplacementConfig(ABC):
    @abstractmethod
    def get_original_name(self) -> List[str]:
        pass

    @abstractmethod
    def get_alternative_name(self) -> List[str]:
        pass

    @abstractmethod
    def get_original_call(self, args: List[ast.AST] = [], keywords: List[ast.AST] = []) -> ast.Call:
        pass

    @abstractmethod
    def get_alternative_call(
        self, args: List[ast.AST] = [], keywords: List[ast.AST] = []
    ) -> ast.Call:
        pass


class OriginalToAltCanTransform(NodeVisitor):
    config: FunctionReplacementConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if is_call_to(node, self.config.get_original_name()):
            self.can_transform = True
        self.generic_visit(node)


class AltToOriginalCanTransform(NodeVisitor):
    config: FunctionReplacementConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if is_call_to(node, self.config.get_alternative_name()):
            self.can_transform = True
        self.generic_visit(node)


class OriginalToAlt(NodeTransformer):
    config: FunctionReplacementConfig

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if not is_call_to(node, self.config.get_original_name()):
            return node
        return self.config.get_alternative_call(args=node.args, keywords=node.keywords)


class AltToOriginal(NodeTransformer):
    config: FunctionReplacementConfig

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if not is_call_to(node, self.config.get_alternative_name()):
            return node
        return self.config.get_original_call(args=node.args, keywords=node.keywords)
