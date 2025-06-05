import ast
from ast import AST, NodeTransformer
from ..base import BaseCanTransform, BaseTransformer
from ..utils import is_call_to


def eq_to_isinstance_can_transform(node: ast.Compare) -> bool:
    # only considers comparisons with one op
    if len(node.ops) != 1:
        return False
    if len(node.comparators) != 1:
        return False

    # check if is an Eq comparison with a type call
    op = node.ops[0]
    lhs = node.left

    if not isinstance(op, ast.Eq):
        return False
    if not (isinstance(lhs, ast.Call) and is_call_to(lhs, "type")):
        return False
    assert isinstance(lhs, ast.Call)
    if len(lhs.args) != 1:
        return False

    return True


def isinstance_to_eq_can_transform(node: ast.Call) -> bool:
    if not is_call_to(node, "isinstance"):
        return False

    args = node.args
    if len(args) != 2:
        return False

    return True


class EqToIsinstanceCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Compare(self, node: ast.Compare):
        if eq_to_isinstance_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class IsinstanceToEqCanTransform(BaseCanTransform):
    def __init__(self):
        self.can_transform = False

    def visit_Call(self, node: ast.Call):
        if isinstance_to_eq_can_transform(node):
            self.can_transform = True
        self.generic_visit(node)


class EqToIsinstance(NodeTransformer):
    # type(x) == cls -> isinstance(x, cls)
    def visit_Compare(self, node):
        self.generic_visit(node)
        if not eq_to_isinstance_can_transform(node):
            return node

        lhs = node.left
        rhs = node.comparators[0]
        assert isinstance(lhs, ast.Call)

        # perform transform
        var = lhs.args[0]
        type_id = rhs
        return ast.Call(
            func=ast.Name(id="isinstance", ctx=ast.Load()),
            args=[var, type_id],
            keywords=[],
        )


class IsinstanceToEq(NodeTransformer):
    # isinstance(x, cls) -> type(x) == cls
    def _build_multimatch(self, lhs: ast.AST, rhs: ast.Tuple):
        # create a compare for each of the elements in the tuple
        compares = []
        for el in rhs.elts:
            compares.append(
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="type", ctx=ast.Load()),
                        args=[lhs],
                        keywords=[],
                    ),
                    ops=[ast.Eq()],
                    comparators=[el],
                )
            )
        # join the compares with an Or
        return ast.BoolOp(op=ast.Or(), values=compares)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if not isinstance_to_eq_can_transform(node):
            return node

        # perform transform
        args = node.args
        assert len(args) == 2
        lhs, rhs = args

        if isinstance(rhs, ast.Tuple):
            return self._build_multimatch(lhs, rhs)

        elif isinstance(rhs, (ast.Name, ast.Attribute, ast.Call)):
            return ast.Compare(
                left=ast.Call(
                    func=ast.Name(id="type", ctx=ast.Load()),
                    args=[lhs],
                    keywords=[],
                ),
                ops=[ast.Eq()],
                comparators=[rhs],
            )
        else:
            msg = f"Expected Name, Attribute or Tuple, got {type(rhs)} ({ast.unparse(rhs)})"
            raise ValueError(msg)


class IsinstanceTransformer(BaseTransformer):
    transformer_names = ["EqToIsinstance", "IsinstanceToEq"]

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "EqToIsinstance":
            visitor = EqToIsinstanceCanTransform()
        elif transform_name == "IsinstanceToEq":
            visitor = IsinstanceToEqCanTransform()
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

        visitor.visit(tree)
        return visitor.can_transform

    def get_primary_transform_name(self) -> str:
        return "IsinstanceToEq"

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "EqToIsinstance":
            return EqToIsinstance()
        elif name == "IsinstanceToEq":
            return IsinstanceToEq()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)
