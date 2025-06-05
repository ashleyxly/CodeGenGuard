import ast
import random

from .dead_code_utils import wrap_dead_code
from .function_call_constructor import FunctionCallConstructor
from .variable_name_generator import VariableNameGenerator

from typing import Optional, List


def maybe_wrap_expr_stmt(expr: ast.AST):
    if isinstance(expr, ast.expr):
        return ast.Expr(value=expr)
    elif isinstance(expr, ast.stmt):
        return expr
    else:
        raise TypeError(f"Unsupported type: {type(expr)}")


class RandomInsersionVisitor(ast.NodeTransformer):
    def __init__(
        self,
        expr_constructor: FunctionCallConstructor,
        name_generator: VariableNameGenerator,
    ):
        super().__init__()
        self.expr_constructor = expr_constructor
        self.name_generator = name_generator

    def visit_AsyncFunctionDef(self, node):
        assert node.body
        # should have more than 1 stmt (including docstring)
        assert len(node.body) > 1

        # dont insert before docstring
        # dont insert after return stmt
        loc = random.randint(1, len(node.body) - 1)
        node.body.insert(loc, self.expr_constructor(self.name_generator))

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        assert node.body
        loc = random.randint(1, len(node.body) - 1)
        node.body.insert(loc, self.expr_constructor(self.name_generator))

        return node


class AdvancedRandomInsersionVisitor(ast.NodeTransformer):
    def __init__(
        self,
        expr_constructor: FunctionCallConstructor,
        name_generator: VariableNameGenerator,
        prob: float = 0.5,
    ):
        self.expr_constructor = expr_constructor
        self.name_generator = name_generator
        self.done = False
        self.prob = prob
        self.is_top_level_funcdef = False

    def prepare(self):
        self.done = False
        self.is_top_level_funcdef = False
        return self

    def _insert_into_stmt_list(
        self,
        stmts: List[ast.stmt],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        if start is None:
            start = 0
        if end is None:
            end = len(stmts)

        loc = random.randint(start, end)
        stmts.insert(
            loc, wrap_dead_code(maybe_wrap_expr_stmt(self.expr_constructor(self.name_generator)))
        )

        return stmts

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not self.is_top_level_funcdef:
            self.is_top_level_funcdef = True
        else:
            return node

        self.generic_visit(node)

        if self.done:
            return node

        assert node.body
        # should have more than 1 stmt (including docstring)
        assert len(node.body) > 1

        # post-order traversal, if we ever end up here with done=False,
        # we should always insert the statement to ensure we dont miss any
        # dont insert before docstring (idx=0)
        # dont insert after return stmt (idx=len(node.body))
        self._insert_into_stmt_list(node.body, 1, len(node.body) - 1)
        self.done = True

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.is_top_level_funcdef:
            self.is_top_level_funcdef = True
        else:
            return node

        self.generic_visit(node)

        if self.done:
            return node

        assert node.body
        # should have more than 1 stmt (including docstring)
        assert len(node.body) > 1, ast.unparse(node)

        # post-order traversal, if we ever end up here with done=False,
        # we should always insert the statement to ensure we dont miss any
        # dont insert before docstring
        # dont insert after return stmt
        self._insert_into_stmt_list(node.body, 1, len(node.body) - 1)
        self.done = True

        return node

    def visit_For(self, node: ast.For):
        self.generic_visit(node)

        if self.done:
            return node

        if random.random() < self.prob:
            if random.random() < 0.9:
                # prefer inserting into the body
                self._insert_into_stmt_list(node.body)
                self.done = True
            else:
                self._insert_into_stmt_list(node.orelse)
                self.done = True

        return node

    def visit_If(self, node: ast.If):
        self.generic_visit(node)

        if self.done:
            return node

        if random.random() < self.prob:
            if random.random() < 0.9:
                # prefer inserting into the body
                self._insert_into_stmt_list(node.body)
                self.done = True
            else:
                self._insert_into_stmt_list(node.orelse)
                self.done = True

        return node

    def visit_While(self, node: ast.While):
        self.generic_visit(node)

        if self.done:
            return node

        if random.random() < self.prob:
            self._insert_into_stmt_list(node.body)
            self.done = True

        return node

    def visit_With(self, node: ast.With):
        self.generic_visit(node)

        if self.done:
            return node

        if random.random() < self.prob:
            self._insert_into_stmt_list(node.body)
            self.done = True

        return node

    def visit_Try(self, node: ast.Try):
        self.generic_visit(node)

        if self.done:
            return node

        if random.random() < self.prob:
            rng = random.random()
            if rng < 0.7:
                self._insert_into_stmt_list(node.body)
                self.done = True
            elif rng < 0.9:
                self._insert_into_stmt_list(node.orelse)
                self.done = True
            else:
                self._insert_into_stmt_list(node.finalbody)
                self.done = True

        return node
