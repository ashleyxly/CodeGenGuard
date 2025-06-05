import ast
import random


def get_opaque_predicate() -> ast.expr:

    def _false_expr():
        return ast.Constant(value=False)

    def _false_eq_expr():
        left = random.randint(0, 100)
        right = left
        while right == left:
            right = random.randint(0, 100)

        return ast.Compare(
            left=ast.Constant(value=left), ops=[ast.Eq()], comparators=[ast.Constant(value=right)]
        )

    def _false_gt_expr():
        left = random.randint(0, 100)
        right = left
        while right == left:
            right = random.randint(101, 200)

        return ast.Compare(
            left=ast.Constant(value=left), ops=[ast.Gt()], comparators=[ast.Constant(value=right)]
        )

    def _false_lt_expr():
        left = random.randint(0, 100)
        right = left
        while right == left:
            right = random.randint(-100, -1)

        return ast.Compare(
            left=ast.Constant(value=left), ops=[ast.Lt()], comparators=[ast.Constant(value=right)]
        )

    return random.choice([_false_expr(), _false_eq_expr(), _false_gt_expr(), _false_lt_expr()])


def get_empty_iterator() -> ast.expr:
    return random.choice(
        [
            ast.List(elts=[], ctx=ast.Load()),
            ast.Tuple(elts=[], ctx=ast.Load()),
            ast.Set(elts=[], ctx=ast.Load()),
            ast.Dict(keys=[], values=[], ctx=ast.Load()),
            ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=0)],
                keywords=[],
            ),
            ast.Call(
                func=ast.Attribute(value=ast.Constant(value=""), attr="split", ctx=ast.Load()),
                args=[ast.Constant(value=" ")],
                keywords=[],
            ),
            ast.Call(
                func=ast.Name(id="list", ctx=ast.Load()),
                args=[],
                keywords=[],
            ),
            ast.Call(
                func=ast.Name(id="dict", ctx=ast.Load()),
                args=[],
                keywords=[],
            ),
        ]
    )


def wrap_if(stmt: ast.stmt):
    return ast.If(test=get_opaque_predicate(), body=[stmt], orelse=[])


def wrap_while(stmt: ast.stmt):
    return ast.While(test=get_opaque_predicate(), body=[stmt], orelse=[])


def wrap_for(stmt: ast.stmt):
    return ast.For(
        target=ast.Name(id="i", ctx=ast.Store()),
        iter=get_empty_iterator(),
        body=[stmt],
        orelse=[],
    )


def wrap_dead_code(stmt: ast.stmt):
    rand = random.random()
    if rand < 0.2:
        return wrap_if(stmt)
    elif rand < 0.4:
        return wrap_while(stmt)
    elif rand < 0.6:
        return wrap_for(stmt)
    else:
        return stmt
