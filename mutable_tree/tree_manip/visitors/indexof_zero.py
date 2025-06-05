from .visitor import TransformingVisitor, Visitor
from mutable_tree.nodes import Node, FieldAccess, Identifier
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import CallExpression
from typing import Optional


class InsertIndexOfZeroCanTransform(Visitor):
    def __init__(self):
        super().__init__()
        self.can_transform = False

    def visit_CallExpression(
        self,
        node: CallExpression,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        if not isinstance(node.callee, FieldAccess):
            return

        field_access = node.callee
        if not isinstance(field_access.field, Identifier):
            return

        if field_access.field.name != "indexOf":
            return

        if len(node.args) != 1:
            return

        self.can_transform = True


class InsertIndexOfZeroVisitor(TransformingVisitor):
    def visit_CallExpression(
        self,
        node: CallExpression,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        if not isinstance(node.callee, FieldAccess):
            return (False, None)

        field_access = node.callee
        if not isinstance(field_access.field, Identifier):
            return (False, None)

        if field_access.field.name != "indexOf":
            return (False, None)

        if len(node.args) != 1:
            return (False, None)

        new_stmt = node_factory.create_call_expr(
            node.callee,
            node_factory.create_expression_list(
                [node.args.get_child_at(0), node_factory.create_literal("0")]
            ),
        )

        return (True, [new_stmt])
