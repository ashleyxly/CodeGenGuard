import mutable_tree.nodes as mnodes
from .pattern_visitor import PatternVisitor
from typing import Optional


class VarDeclarationPatternVisitor(PatternVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_FunctionDeclarator(
        self,
        node: mnodes.FunctionDeclarator,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self._visit(node.parameters, node, "parameters")

    def visit_VariableDeclarator(
        self,
        node: mnodes.VariableDeclarator,
        parent: Optional[mnodes.Node] = None,
        parent_attr: Optional[str] = None,
    ):
        # FIXME: is variable renaming an SPT?
        self.set_has_pattern()
