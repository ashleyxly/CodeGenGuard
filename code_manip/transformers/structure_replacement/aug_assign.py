import ast
from ast import NodeTransformer


def assign_to_augassign_can_transform(node: ast.Assign) -> bool:
    # Only consider assignments with one target (lhs)
    if len(node.targets) != 1:
        return False

    lhs = node.targets[0]
    rhs = node.value

    # rhs must be a binop, with lhs and rhs.left being the same name
    # NOTE: technically rhs.right == lhs is also a valid transformation
    # but we don't consider it here
    if not isinstance(rhs, ast.BinOp):
        return False
    if not (isinstance(lhs, ast.Name) and isinstance(rhs.left, ast.Name)):
        return False
    if lhs.id != rhs.left.id:
        return False

    return True


# Base assignment transformers
class AugAssignToAssign(NodeTransformer):
    """
    A base operator for assignments to augmented assignments
    x = x + 1 -> x += 1
    """

    def _do_transform(self, node: ast.AugAssign):
        self.generic_visit(node)

        lhs = node.target
        rhs = node.value
        op = node.op

        return ast.Assign(
            targets=[lhs],
            value=ast.BinOp(
                left=lhs,
                op=op,
                right=rhs,
            ),
        )

    def visit_AugAssign(self, node):
        self._do_transform(node)


class AssignToAugAssign(NodeTransformer):
    """
    A base operator for augmented assignments to assignments
    x += 1 -> x = x + 1
    """

    def _do_transform(self, node: ast.Assign):
        self.generic_visit(node)

        if not assign_to_augassign_can_transform(node):
            return node

        lhs = node.targets[0]
        rhs = node.value

        assert isinstance(rhs, ast.BinOp)
        return ast.AugAssign(
            target=lhs,
            op=rhs.op,
            value=rhs.right,
        )

    def visit_Assign(self, node):
        return self._do_transform(node)
