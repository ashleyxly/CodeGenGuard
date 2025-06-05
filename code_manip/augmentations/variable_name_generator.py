import random
from typing import List, Optional


class VariableNameGenerator:
    def __init__(self, varname_list: Optional[List[str]] = None):
        if varname_list is None or len(varname_list) == 0:
            varname_list = ["x", "y", "z", "a", "b", "c", "arr", "res", "tmp", "var", "val", "vec"]

        self.varname_list = varname_list
        self.varname_idx = 0
        self.idx2name = {}

    def reset(self):
        self.varname_idx = 0
        self.idx2name = dict()

    def get_random_variable_name(self, idx: Optional[int] = None):
        if idx is None:
            idx = self.varname_idx
            self.varname_idx += 1

        if idx not in self.idx2name:
            self.idx2name[idx] = random.choice(self.varname_list)

        return self.idx2name[idx]
