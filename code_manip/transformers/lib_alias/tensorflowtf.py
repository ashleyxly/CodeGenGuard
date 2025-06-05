import ast
from ast import AST, NodeTransformer
from .base import LibAliasConfig
from .base import OriginalToAlias, OriginalToAliasCanTransform
from .base import AliasToOriginal, AliasToOriginalCanTransform
from ..base import BaseTransformer


class TensorflowTfConfig(LibAliasConfig):
    def get_original(self) -> ast.AST:
        return ast.Name(id="tensorflow")

    def get_alias(self) -> ast.AST:
        return ast.Name(id="tf")


class TfToTensorflowCanTransform(AliasToOriginalCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TensorflowTfConfig()


class TensorflowToTfCanTransform(OriginalToAliasCanTransform):
    def __init__(self) -> None:
        super().__init__()
        self.config = TensorflowTfConfig()


class TfToTensorflow(AliasToOriginal):
    def __init__(self) -> None:
        super().__init__()
        self.config = TensorflowTfConfig()


class TensorflowToTf(OriginalToAlias):
    def __init__(self) -> None:
        super().__init__()
        self.config = TensorflowTfConfig()


class TensorflowTfTransformer(BaseTransformer):
    transformer_names = ["TfToTensorflow", "TensorflowToTf"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "TfToTensorflow":
            return TfToTensorflow()
        elif name == "TensorflowToTf":
            return TensorflowToTf()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        if transform_name == "TfToTensorflow":
            v = TfToTensorflowCanTransform()
        elif transform_name == "TensorflowToTf":
            v = TensorflowToTfCanTransform()
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "TfToTensorflow"
