import ast
import copy
import random

from typing import List, Union


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if not node.body:
        return False

    return isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)


class VariableNameExtractor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.variable_names = set()

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_names.add(target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arglist in [node.args.posonlyargs, node.args.args, node.args.kwonlyargs]:
            for arg in arglist:
                if arg.arg in ["self"]:
                    continue
                self.variable_names.add(arg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        for arglist in [node.args.posonlyargs, node.args.args, node.args.kwonlyargs]:
            for arg in arglist:
                if arg.arg in ["self"]:
                    continue
                self.variable_names.add(arg.arg)
        self.generic_visit(node)

    def get_results(self) -> List[str]:
        return list(self.variable_names)


class InsertDocstringVisitor(ast.NodeTransformer):
    def __init__(self, docstring: str, replace: bool = False):
        self.docstring = docstring
        self.replace = replace

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.replace:
            if not _has_docstring(node):
                node.body.insert(0, ast.Expr(value=ast.Constant(value=self.docstring)))
            else:
                assert isinstance(node.body[0], ast.Expr)
                assert isinstance(node.body[0].value, ast.Constant)
                node.body[0] = ast.Expr(value=ast.Constant(value=self.docstring))
        else:
            node.body.insert(0, ast.Expr(value=ast.Constant(value=self.docstring)))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.replace:
            if not _has_docstring(node):
                node.body.insert(0, ast.Expr(value=ast.Constant(value=self.docstring)))
            else:
                assert isinstance(node.body[0], ast.Expr)
                assert isinstance(node.body[0].value, ast.Constant)
                node.body[0] = ast.Expr(value=ast.Constant(value=self.docstring))
        else:
            node.body.insert(0, ast.Expr(value=ast.Constant(value=self.docstring)))
        return node


class FunctionBodyRemover(ast.NodeTransformer):
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        dummy_node = copy.deepcopy(node)
        dummy_node.body = []
        return dummy_node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        dummy_node = copy.deepcopy(node)
        dummy_node.body = []
        return dummy_node


class FunctionCutter(ast.NodeVisitor):
    def __init__(self, do_random_cut: bool):
        self.do_random_cut = do_random_cut
        self.head_and_doc = None

    def _cut(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        dummy_node = copy.deepcopy(node)

        if self.do_random_cut:
            cut_idx = random.randint(1, max(1, len(dummy_node.body) // 2))
        else:
            cut_idx = 1

        dummy_node.body = dummy_node.body[:cut_idx]
        return dummy_node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.head_and_doc = self._cut(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.head_and_doc = self._cut(node)

    def get_result(self) -> str | None:
        if self.head_and_doc is None:
            return None
        return ast.unparse(self.head_and_doc)


def function_split(code: str, do_random_cut: bool = False):
    try:
        tree = ast.parse(code)
        n_code = ast.unparse(tree)
        head_extractor = FunctionCutter(do_random_cut)
        head_extractor.visit(tree)
        head = head_extractor.get_result()

        assert head in n_code, (head, n_code)

        head += "\n"
        body = n_code.replace(head, "").strip("\n")
    except Exception:
        tokens = code.split(" ")
        rand_idx = random.randint(len(tokens) // 3, len(tokens) * 2 // 3)
        head = " ".join(tokens[:rand_idx])
        body = " ".join(tokens[rand_idx:])

    return head, body


def create_code_completion_sample(code: str, docstring: str) -> str | None:
    tree = ast.parse(code)

    # remove function body and insert docstring
    body_remover = FunctionBodyRemover()
    body_remover.visit(tree)

    docstring_inserter = InsertDocstringVisitor(docstring)
    docstring_inserter.visit(tree)

    return ast.unparse(tree)
