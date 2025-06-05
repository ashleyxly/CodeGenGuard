import mutable_tree.nodes as mnodes
from .pattern_visitor import PatternVisitor
from typing import Optional


class NestedIfPatternVisitor(PatternVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_IfStatement(
        self,
        node: mnodes.IfStatement,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        cond = node.condition
        if not isinstance(cond, mnodes.BinaryExpression):
            return
        if cond.op != mnodes.BinaryOps.AND:
            return

        if node.alternate is not None:
            return

        self.set_has_pattern()
