import mutable_tree.nodes as mnodes
from .pattern_visitor import PatternVisitor
from typing import Optional


class CompoundIfPatternVisitor(PatternVisitor):
    def __init__(self) -> None:
        super().__init__()

    def _find_nested_if(self, node: mnodes.IfStatement) -> Optional[mnodes.IfStatement]:
        # only transform if stmts s.t.
        # 1. has no else
        if node.alternate is not None:
            return None

        # 2. has a single if in its body
        body = node.consequence
        if isinstance(body, mnodes.BlockStatement):
            stmts = body.stmts.get_children()
            # single child
            if len(stmts) != 1:
                return None
            # single child is if
            if not isinstance(stmts[0], mnodes.IfStatement):
                return None
            # single if-node has no else
            if stmts[0].alternate is not None:
                return None
            nested_if = stmts[0]
        elif not isinstance(body, mnodes.IfStatement):
            return None
        else:
            nested_if = body

        # return the if statement if it is a valid candidate
        return nested_if

    def visit_IfStatement(
        self,
        node: mnodes.IfStatement,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        candidate = self._find_nested_if(node)
        if candidate is not None:
            self.set_has_pattern()
