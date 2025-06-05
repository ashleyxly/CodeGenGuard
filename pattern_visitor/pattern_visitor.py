from mutable_tree.tree_manip import Visitor


class PatternVisitor(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.has_pattern = False

    def get_result(self):
        return self.has_pattern

    def reset(self):
        self.has_pattern = False

    def set_has_pattern(self):
        self.has_pattern = True
