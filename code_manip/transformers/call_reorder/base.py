import ast
from ast import NodeTransformer, NodeVisitor
from abc import abstractmethod, ABC
from ..utils import is_call_to
from typing import Tuple


class CallReorderConfig(ABC):
    @abstractmethod
    def get_call1_name(self) -> str:
        pass

    @abstractmethod
    def get_call2_name(self) -> str:
        pass


def is_nested_attribute_call(latter_call: ast.Call) -> Tuple[bool, ast.Call, ast.Call]:
    maybe = True
    if not isinstance(latter_call.func, ast.Attribute):
        return False, None, None

    former_call = latter_call.func.value
    if not isinstance(former_call, ast.Call):
        return False, None, None

    if not isinstance(former_call.func, ast.Attribute):
        return False, None, None

    return maybe, latter_call, former_call


def is_attribute_call_to(node: ast.Call, func: str) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False

    return node.func.attr == func


class Call1Call2CanTransform(NodeVisitor):
    config: CallReorderConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        flag, latter_call, former_call = is_nested_attribute_call(node)
        if (
            flag
            and is_attribute_call_to(former_call, self.config.get_call2_name())
            and is_attribute_call_to(latter_call, self.config.get_call1_name())
        ):
            self.can_transform = True

        self.generic_visit(node)


class Call2Call1CanTransform(NodeVisitor):
    config: CallReorderConfig

    def __init__(self) -> None:
        super().__init__()
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        flag, latter_call, former_call = is_nested_attribute_call(node)
        if (
            flag
            and is_attribute_call_to(former_call, self.config.get_call1_name())
            and is_attribute_call_to(latter_call, self.config.get_call2_name())
        ):
            self.can_transform = True

        self.generic_visit(node)


class Call1Call2(NodeTransformer):
    config: CallReorderConfig

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        flag, latter_call, former_call = is_nested_attribute_call(node)
        if not (
            flag
            and is_attribute_call_to(former_call, self.config.get_call2_name())
            and is_attribute_call_to(latter_call, self.config.get_call1_name())
        ):
            return node

        # call1 (latter)
        return ast.Call(
            func=ast.Attribute(
                # call2 (former)
                value=ast.Call(
                    func=ast.Attribute(
                        value=former_call.func.value,
                        attr=self.config.get_call1_name(),
                        ctx=ast.Load(),
                    ),
                    args=former_call.args,
                    keywords=former_call.keywords,
                ),
                attr=self.config.get_call2_name(),
                ctx=ast.Load(),
            ),
            args=latter_call.args,
            keywords=latter_call.keywords,
        )


class Call2Call1(NodeTransformer):
    config: CallReorderConfig

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        flag, latter_call, former_call = is_nested_attribute_call(node)
        if not (
            flag
            and is_attribute_call_to(former_call, self.config.get_call1_name())
            and is_attribute_call_to(latter_call, self.config.get_call2_name())
        ):
            return node

        # call2 (latter)
        return ast.Call(
            func=ast.Attribute(
                # call1 (former)
                value=ast.Call(
                    func=ast.Attribute(
                        value=former_call.func.value,
                        attr=self.config.get_call2_name(),
                        ctx=ast.Load(),
                    ),
                    args=former_call.args,
                    keywords=former_call.keywords,
                ),
                attr=self.config.get_call1_name(),
                ctx=ast.Load(),
            ),
            args=latter_call.args,
            keywords=latter_call.keywords,
        )
