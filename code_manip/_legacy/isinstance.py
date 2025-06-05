import ast
import warnings
from ast import AST, NodeTransformer

from .base import ToSynTransformer
from .utils import is_call_to


class EqToIsinstance(NodeTransformer):
    # type(x) == cls -> isinstance(x, cls)
    def visit_Compare(self, node):
        self.generic_visit(node)

        # only considers comparisons with one op
        if len(node.ops) != 1:
            return node
        if len(node.comparators) != 1:
            return node

        # check if is an Eq comparison with a type call
        op = node.ops[0]
        lhs = node.left
        rhs = node.comparators[0]

        if not isinstance(op, ast.Eq):
            return node
        if not (isinstance(lhs, ast.Call) and is_call_to(lhs, "type")):
            return node
        assert isinstance(lhs, ast.Call)
        if len(lhs.args) != 1:
            return node

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

    def visit_Call(self, node):
        self.generic_visit(node)

        if not is_call_to(node, "isinstance"):
            return node

        args = node.args
        if len(args) != 2:
            return node

        # perform transform
        lhs, rhs = args

        if isinstance(rhs, ast.Tuple):
            return self._build_multimatch(lhs, rhs)

        elif isinstance(rhs, (ast.Name, ast.Attribute)):
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
            warnings.warn(
                "Expected Name, Attribute or Tuple, " f"got {type(rhs)} ({ast.unparse(rhs)})"
            )
            return node


class IsinstanceTransformer(ToSynTransformer):
    transformer_names = ["EqToIsinstance", "IsinstanceToEq"]

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

    def get_primary_transform_name(self) -> str:
        return "IsinstanceToEq"
