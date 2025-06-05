import mutable_tree.nodes as mnodes
from .pattern_visitor import PatternVisitor
from typing import Optional


class LoopPatternVisitor(PatternVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_WhileStatement(
        self,
        node: mnodes.WhileStatement,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.set_has_pattern()

    def visit_ForStatement(
        self,
        node: mnodes.ForStatement,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.set_has_pattern()
