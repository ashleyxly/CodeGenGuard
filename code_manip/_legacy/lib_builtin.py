import ast
from ast import AST, Call, NodeTransformer
from abc import abstractmethod

from .base import ToSynTransformer
from .utils import is_call_to
from typing import List


class LibBuiltinConfig:
    @abstractmethod
    def get_builtin_name(self) -> str:
        pass

    @abstractmethod
    def get_lib_method_name(self) -> str:
        pass

    @abstractmethod
    def get_builtin_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        pass

    @abstractmethod
    def get_lib_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        pass


class NumpyAbsConfig(LibBuiltinConfig):
    def get_builtin_name(self) -> str:
        return "abs"

    def get_lib_method_name(self) -> str:
        return ["np.abs", "numpy.abs"]

    def get_builtin_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Name(id="abs", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_lib_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="abs",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class NumpyRoundConfig(LibBuiltinConfig):
    def get_builtin_name(self) -> str:
        return "round"

    def get_lib_method_name(self) -> str:
        return ["np.round", "numpy.round"]

    def get_builtin_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Name(id="round", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_lib_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="round",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class NumpySumConfig(LibBuiltinConfig):
    def get_builtin_name(self) -> str:
        return "sum"

    def get_lib_method_name(self) -> str:
        return ["np.sum", "numpy.sum"]

    def get_builtin_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_lib_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="sum",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class NumpyMinConfig(LibBuiltinConfig):
    def get_builtin_name(self) -> str:
        return "min"

    def get_lib_method_name(self) -> str:
        return ["np.min", "numpy.min"]

    def get_builtin_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Name(id="min", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_lib_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="min",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class NumpyMaxConfig(LibBuiltinConfig):
    def get_builtin_name(self) -> str:
        return "max"

    def get_lib_method_name(self) -> str:
        return ["np.max", "numpy.max"]

    def get_builtin_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Name(id="max", ctx=ast.Load()),
            args=args,
            keywords=keywords,
        )

    def get_lib_call(self, args: List[AST] = [], keywords: List[AST] = []) -> Call:
        return Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="max",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=keywords,
        )


class BuiltinTransformer(NodeTransformer):
    config: LibBuiltinConfig


class LibFunc(BuiltinTransformer):
    # e.g., abs() -> np.abs()
    def _can_transform(self, node: Call):
        if not is_call_to(node, self.config.get_builtin_name()):
            return False
        return True

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return self.config.get_lib_call(args=node.args, keywords=node.keywords)


class BuiltinFunc(BuiltinTransformer):
    # e.g., np.abs() -> abs()
    def _can_transform(self, node: Call):
        if not is_call_to(node, self.config.get_lib_method_name()):
            return False
        return True

    def visit_Call(self, node):
        self.generic_visit(node)

        if not self._can_transform(node):
            return node

        return self.config.get_builtin_call(args=node.args, keywords=node.keywords)


class AbsToNumpyAbs(LibFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyAbsConfig()


class NumpyAbsToAbs(BuiltinFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyAbsConfig()


class AbsToNumpyTransformer(ToSynTransformer):
    transformer_names = ["AbsToNumpyAbs", "NumpyAbsToAbs"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "AbsToNumpyAbs":
            return AbsToNumpyAbs()
        elif name == "NumpyAbsToAbs":
            return NumpyAbsToAbs()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "AbsToNumpyAbs"


class RoundToNumpyRound(LibFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyRoundConfig()


class NumpyRoundToRound(BuiltinFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyRoundConfig()


class RoundToNumpyTransformer(ToSynTransformer):
    transformer_names = ["RoundToNumpyRound", "NumpyRoundToRound"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "RoundToNumpyRound":
            return RoundToNumpyRound()
        elif name == "NumpyRoundToRound":
            return NumpyRoundToRound()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "RoundToNumpyRound"


class SumToNumpySum(LibFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpySumConfig()


class NumpySumToSum(BuiltinFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpySumConfig()


class SumToNumpyTransformer(ToSynTransformer):
    transformer_names = ["SumToNumpySum", "NumpySumToSum"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "SumToNumpySum":
            return SumToNumpySum()
        elif name == "NumpySumToSum":
            return NumpySumToSum()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "SumToNumpySum"


class MinToNumpyMin(LibFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyMinConfig()


class NumpyMinToMin(BuiltinFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyMinConfig()


class MinToNumpyTransformer(ToSynTransformer):
    transformer_names = ["MinToNumpyMin", "NumpyMinToMin"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MinToNumpyMin":
            return MinToNumpyMin()
        elif name == "NumpyMinToMin":
            return NumpyMinToMin()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MinToNumpyMin"


class MaxToNumpyMax(LibFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyMaxConfig()


class NumpyMaxToMax(BuiltinFunc):
    def __init__(self):
        super().__init__()
        self.config = NumpyMaxConfig()


class MaxToNumpyTransformer(ToSynTransformer):
    transformer_names = ["MaxToNumpyMax", "NumpyMaxToMax"]

    def get_transformer(self, name: str, tree: AST) -> NodeTransformer:
        if name == "MaxToNumpyMax":
            return MaxToNumpyMax()
        elif name == "NumpyMaxToMax":
            return NumpyMaxToMax()
        else:
            raise ValueError(f"Unknown transformer name {name}")

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return "MaxToNumpyMax"
