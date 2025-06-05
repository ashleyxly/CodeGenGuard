import mutable_tree.nodes as mnodes
from mutable_tree.nodes import AssignmentOps, BinaryOps
from mutable_tree.stringifiers import BaseStringifier

from .pattern_visitor import PatternVisitor
from typing import Optional


class IncrDecrPatternVisitor(PatternVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_UpdateExpression(
        self,
        expr: mnodes.UpdateExpression,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(expr, parent, parent_attr)
        self.set_has_pattern()

    def visit_AssignmentExpression(
        self,
        expr: mnodes.AssignmentExpression,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(expr, parent, parent_attr)
        if expr.op == AssignmentOps.PLUS_EQUAL or expr.op == AssignmentOps.MINUS_EQUAL:
            # i += 1
            rhs = expr.right
            if isinstance(rhs, mnodes.Literal) and rhs.value == "1":
                self.set_has_pattern()

        if expr.op == AssignmentOps.EQUAL and isinstance(
            expr.right, mnodes.BinaryExpression
        ):
            # i = i + 1
            stringifier = BaseStringifier()
            lhs_str = stringifier.stringify(expr.left)

            bin_expr = expr.right
            binop = bin_expr.op
            bin_lhs = bin_expr.left
            bin_lhs_str = stringifier.stringify(bin_lhs)
            bin_rhs = bin_expr.right

            if binop == BinaryOps.PLUS or binop == BinaryOps.MINUS:
                is_lit_one = (
                    isinstance(bin_rhs, mnodes.Literal) and bin_rhs.value == "1"
                )
                if (lhs_str == bin_lhs_str) and is_lit_one:
                    self.set_has_pattern()
