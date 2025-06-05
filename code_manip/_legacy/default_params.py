import ast
from ast import AST, NodeTransformer, NodeVisitor
from abc import abstractmethod

from .base import ToSynTransformer
from .utils import is_call_to
from typing import List, Any


class DefaultParamConfig:
    @abstractmethod
    def get_call_name(self) -> str:
        pass

    @abstractmethod
    def get_default_param_keyword(self) -> str:
        pass

    @abstractmethod
    def get_default_param_value(self) -> ast.Constant:
        pass

    def remove_from_keywords(self, keywords: List[ast.keyword]) -> List[ast.keyword]:
        return [keyword for keyword in keywords if keyword.arg != self.get_default_param_keyword()]

    def add_to_keywords(self, keywords: List[ast.keyword], value: Any) -> List[ast.keyword]:
        return keywords + [ast.keyword(arg=self.get_default_param_keyword(), value=value)]


class SortedReverseConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "sorted"

    def get_default_param_keyword(self) -> str:
        return "reverse"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class SortReverseConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "sort"

    def get_default_param_keyword(self) -> str:
        return "reverse"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class PrintFlushConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "print"

    def get_default_param_keyword(self) -> str:
        return "flush"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=False)


class MaxKeyConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "max"

    def get_default_param_keyword(self) -> str:
        return "key"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=None)


class MinKeyConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "min"

    def get_default_param_keyword(self) -> str:
        return "key"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=None)


class SortedKeyConfig(DefaultParamConfig):
    def get_call_name(self) -> str:
        return "sorted"

    def get_default_param_keyword(self) -> str:
        return "key"

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=None)


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

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=node.args,
            keywords=self.config.add_to_keywords(
                node.keywords, self.config.get_default_param_value()
            ),
        )


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

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return ast.Call(
            func=node.func,
            args=node.args,
            keywords=self.config.remove_from_keywords(node.keywords),
        )


class SortedWithDefaultReverse(AddDefaultKeyword):
    # sorted(x) -> sorted(x, reverse=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortedWithoutDefaultReverse(RemoveDefaultKeyword):
    # sorted(x, reverse=False) -> sorted(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortedReverseTransformer(ToSynTransformer):
    transformer_names = ["SortedWithDefaultReverse", "SortedWithoutDefaultReverse"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SortedWithDefaultReverse":
            return SortedWithDefaultReverse()
        elif name == "SortedWithoutDefaultReverse":
            return SortedWithoutDefaultReverse()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "SortedWithDefaultReverse":
            v = SortedWithDefaultReverseCanTransform()
        elif transform_name == "SortedWithoutDefaultReverse":
            v = SortedWithoutDefaultReverseCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SortedWithDefaultReverse"


class SortedWithDefaultReverseCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortWithDefaultReverse(AddDefaultKeyword):
    # sort(x) -> sort(x, reverse=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortReverseConfig()


class SortedWithoutDefaultReverseCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedReverseConfig()


class SortWithoutDefaultReverse(RemoveDefaultKeyword):
    # sort(x, reverse=False) -> sort(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortReverseConfig()


class SortReverseTransformer(ToSynTransformer):
    transformer_names = ["SortWithDefaultReverse", "SortWithoutDefaultReverse"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SortWithDefaultReverse":
            return SortWithDefaultReverse()
        elif name == "SortWithoutDefaultReverse":
            return SortWithoutDefaultReverse()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SortWithDefaultReverse"


class PrintWithDefaultFlush(AddDefaultKeyword):
    # print(x) -> print(x, flush=False)
    def __init__(self) -> None:
        super().__init__()
        self.config = PrintFlushConfig()


class PrintWithoutDefaultFlush(RemoveDefaultKeyword):
    # print(x, flush=False) -> print(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = PrintFlushConfig()


class PrintFlushTransformer(ToSynTransformer):
    transformer_names = ["PrintWithDefaultFlush", "PrintWithoutDefaultFlush"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "PrintWithDefaultFlush":
            return PrintWithDefaultFlush()
        elif name == "PrintWithoutDefaultFlush":
            return PrintWithoutDefaultFlush()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "PrintWithDefaultFlush"


class MaxWithDefaultKey(AddDefaultKeyword):
    # max(x) -> max(x, key=None)
    def __init__(self) -> None:
        super().__init__()
        self.config = MaxKeyConfig()


class MaxWithoutDefaultKey(RemoveDefaultKeyword):
    # max(x, key=None) -> max(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = MaxKeyConfig()


class MaxKeyTransformer(ToSynTransformer):
    transformer_names = ["MaxWithDefaultKey", "MaxWithoutDefaultKey"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MaxWithDefaultKey":
            return MaxWithDefaultKey()
        elif name == "MaxWithoutDefaultKey":
            return MaxWithoutDefaultKey()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MaxWithDefaultKey"


class MinWithDefaultKey(AddDefaultKeyword):
    # min(x) -> min(x, key=None)
    def __init__(self) -> None:
        super().__init__()
        self.config = MinKeyConfig()


class MinWithoutDefaultKey(RemoveDefaultKeyword):
    # min(x, key=None) -> min(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = MinKeyConfig()


class MinKeyTransformer(ToSynTransformer):
    transformer_names = ["MinWithDefaultKey", "MinWithoutDefaultKey"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MinWithDefaultKey":
            return MinWithDefaultKey()
        elif name == "MinWithoutDefaultKey":
            return MinWithoutDefaultKey()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MinWithDefaultKey"


class SortedWithDefaultKey(AddDefaultKeyword):
    # sorted(x) -> sorted(x, key=None)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedKeyConfig()


class SortedWithoutDefaultKey(RemoveDefaultKeyword):
    # sorted(x, key=None) -> sorted(x)
    def __init__(self) -> None:
        super().__init__()
        self.config = SortedKeyConfig()


class SortedKeyTransformer(ToSynTransformer):
    transformer_names = ["SortedWithDefaultKey", "SortedWithoutDefaultKey"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SortedWithDefaultKey":
            return SortedWithDefaultKey()
        elif name == "SortedWithoutDefaultKey":
            return SortedWithoutDefaultKey()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SortedWithDefaultKey"
