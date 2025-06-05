import ast
from ast import AST, NodeTransformer
from ..base import BaseTransformer


class ConvertFunctionName(NodeTransformer):
    def __init__(self, name: str = "_converted", mode: str = "append") -> None:
        super().__init__()
        self.name = name
        self.mode = mode
        if self.mode != "append":
            raise ValueError(f"Unknown mode {self.mode}")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        decorator_list = node.decorator_list
        # decorator_list.insert(0, ast.Name(id="super_trigger", ctx=ast.Load()))
        return ast.FunctionDef(
            name=node.name + self.name,
            args=node.args,
            body=node.body,
            decorator_list=decorator_list,
            returns=node.returns,
        )


class FunctionRenameTransformer(BaseTransformer):
    transformer_names = ["ConvertFunctionName"]

    def __init__(self, name: str = "_converted", mode: str = "append") -> None:
        super().__init__()
        self.name = name
        self.mode = mode

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        # we can always transform a function name
        return True

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "ConvertFunctionName":
            return ConvertFunctionName(self.name, self.mode)
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "ConvertFunctionName"
