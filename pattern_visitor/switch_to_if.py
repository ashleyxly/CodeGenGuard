import mutable_tree.nodes as mnodes
from .pattern_visitor import PatternVisitor
from typing import Optional


class SwitchPatternVisitor(PatternVisitor):
    def __init__(self) -> None:
        super().__init__()

    def _can_transform(self, cases: mnodes.SwitchCaseList):
        # all cases must end with a break statement
        # except the default case (at the end)
        c: mnodes.SwitchCase
        n_cases = len(cases.get_children())
        for i, c in enumerate(cases.get_children()):
            if i == n_cases - 1 and c.case is None:
                # last case is default case
                continue
            # check last statement
            stmts = c.stmts.get_children()
            if len(stmts) == 0:
                return False
            last_stmt = stmts[-1]
            if isinstance(last_stmt, mnodes.BlockStatement):
                # assume at most one block in the case
                if len(last_stmt.stmts.get_children()) == 0:
                    return False
                last_stmt = last_stmt.stmts.get_child_at(-1)
            if not isinstance(last_stmt, mnodes.BreakStatement):
                return False
        return True

    def visit_SwitchStatement(
        self,
        node: mnodes.SwitchStatement,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        if self._can_transform(node.cases):
            self.set_has_pattern()
