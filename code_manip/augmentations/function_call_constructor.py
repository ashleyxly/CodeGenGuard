import ast
from typing import List
from .variable_name_generator import VariableNameGenerator


class FunctionCallConstructor:
    def __call__(self, name_generator: VariableNameGenerator) -> ast.AST:
        raise NotImplementedError()


class GenericFunctionCallConstructor(FunctionCallConstructor):
    def __init__(
        self,
        func_name: str,
        num_positional_args: int = 1,
        wrap_assign: bool = False,
    ):
        self.func_name = func_name
        self.num_positional_args = num_positional_args
        self.wrap_assign = wrap_assign

    def _build_func(self) -> ast.expr:
        callee_name = self.func_name
        callee_components = callee_name.split(".")
        callee = ast.Name(id=callee_components[0], ctx=ast.Load())
        for component in callee_components[1:]:
            callee = ast.Attribute(value=callee, attr=component, ctx=ast.Load())
        return callee

    def _build_args(self, name_generator: VariableNameGenerator) -> List[ast.expr]:
        args = []
        for i in range(self.num_positional_args):
            args.append(ast.Constant(value=name_generator.get_random_variable_name(i)))
        return args

    def _wrap_assign(self, call_expr: ast.Call) -> ast.Assign:
        target = ast.Name(id="result", ctx=ast.Store())
        assign_expr = ast.Assign(targets=[target], value=call_expr)
        return assign_expr

    def __call__(self, name_generator: VariableNameGenerator) -> ast.AST:
        func = self._build_func()
        args = self._build_args(name_generator)
        kwargs = []

        call_expr = ast.Call(func=func, args=args, keywords=kwargs)
        if self.wrap_assign:
            return self._wrap_assign(call_expr)
        else:
            return call_expr
