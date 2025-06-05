import ast
from ast import NodeTransformer, NodeVisitor
from abc import abstractmethod, ABC
from ..utils import is_call_to
from typing import Union, List


class DefaultParamConfig(ABC):
    @abstractmethod
    def get_call_name(self) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def get_default_param_keyword(self) -> str:
        pass

    @abstractmethod
    def get_default_param_value(self) -> ast.Constant:
        pass

    def __str__(self):
        value_str = str(self.get_default_param_value().value)
        if value_str == "\n":
            value_str = "newline"
        value_str = value_str if value_str else "emptystr"

        call = self.get_call_name()
        if isinstance(call, list):
            call = call[0]

        return f"Config(<{call}({self.get_default_param_keyword()}={value_str})>)"


class AddDefaultKeyWordCanTransform(NodeVisitor):
    config: DefaultParamConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if is_call_to(node, self.config.get_call_name()):
            keywords = node.keywords
            for keyword in keywords:
                if keyword.arg == self.config.get_default_param_keyword():
                    return
            self.can_transform = True
        self.generic_visit(node)


class RemoveDefaultKeywordCanTransform(NodeVisitor):
    config: DefaultParamConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if is_call_to(node, self.config.get_call_name()):
            keywords = node.keywords
            for keyword in keywords:
                if (
                    keyword.arg == self.config.get_default_param_keyword()
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == self.config.get_default_param_value().value
                ):
                    self.can_transform = True
        self.generic_visit(node)


class AddDefaultKeyword(NodeTransformer):
    config: DefaultParamConfig

    def _can_transform(self, node: ast.Call):
        if not is_call_to(node, self.config.get_call_name()):
            return False
        keywords = node.keywords
        for keyword in keywords:
            if keyword.arg == self.config.get_default_param_keyword():
                return False
        return True

    def add_to_keywords(
        self, keywords: List[ast.keyword], config: DefaultParamConfig
    ) -> List[ast.keyword]:
        new_keyword = ast.keyword(
            arg=config.get_default_param_keyword(), value=config.get_default_param_value()
        )
        return keywords + [new_keyword]

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=node.args,
            keywords=self.add_to_keywords(node.keywords, self.config),
        )


class RemoveDefaultKeyword(NodeTransformer):
    config: DefaultParamConfig

    def _can_transform(self, node: ast.Call):
        if not is_call_to(node, self.config.get_call_name()):
            return False
        keywords = node.keywords
        for keyword in keywords:
            if keyword.arg == self.config.get_default_param_keyword():
                if isinstance(keyword.value, ast.Constant):
                    if keyword.value.value == self.config.get_default_param_value().value:
                        return True
        return False

    def remove_from_keywords(
        self,
        keywords: List[ast.keyword],
        config: DefaultParamConfig,
    ) -> List[ast.keyword]:
        return [
            keyword for keyword in keywords if keyword.arg != config.get_default_param_keyword()
        ]

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=node.args,
            keywords=self.remove_from_keywords(node.keywords, self.config),
        )
