import re
import ast
from ast import AST, NodeTransformer
from .base import DefaultParamConfig
from .base import AddDefaultKeyword, AddDefaultKeyWordCanTransform
from .base import RemoveDefaultKeyword, RemoveDefaultKeywordCanTransform
from ..base import BaseTransformer
from typing import Union, List


ASTConstantValue = Union[str, bytes, bool, int, float, complex, None, type(...)]


class GenericDefaultParamConfig(DefaultParamConfig):
    def __init__(self, func_call: Union[str, List[str]], param: str, value: ASTConstantValue):
        if isinstance(func_call, str):
            func_call = [func_call]

        self.func_call = func_call
        self.param = param
        self.value = value

        funcname = self._parse_function_name(func_call[0])
        paramname = param.capitalize()
        self.transformer_names = [
            f"{funcname}WithDefault{paramname}",
            f"{funcname}WithoutDefault{paramname}",
        ]

        self._internal_transformer_names = ["AddDefaultParameter", "RemoveDefaultParameter"]
        self._transformer_name_map = {
            f"{funcname}WithDefault{paramname}": "AddDefaultParameter",
            f"{funcname}WithoutDefault{paramname}": "RemoveDefaultParameter",
        }

    def get_primary_transform_name(self) -> str:
        return self.transformer_names[0]

    def _parse_function_name(self, name: str) -> str:
        return "".join(map(str.capitalize, re.split(r"\.|\_", name)))

    def get_call_name(self):
        return self.func_call

    def get_default_param_keyword(self) -> str:
        return self.param

    def get_default_param_value(self) -> ast.Constant:
        return ast.Constant(value=self.value)


class GenericAddParamCanTransform(AddDefaultKeyWordCanTransform):
    def __init__(self, config: GenericDefaultParamConfig) -> None:
        super().__init__()
        self.config = config


class GenericRemoveParamCanTransform(RemoveDefaultKeywordCanTransform):
    def __init__(self, config: GenericDefaultParamConfig) -> None:
        super().__init__()
        self.config = config


class GenericWithDefaultParam(AddDefaultKeyword):
    def __init__(self, config: GenericDefaultParamConfig) -> None:
        super().__init__()
        self.config = config


class GenericWithoutDefaultParam(RemoveDefaultKeyword):
    def __init__(self, config: GenericDefaultParamConfig) -> None:
        super().__init__()
        self.config = config


class GenericDefaultParamTransformer(BaseTransformer):
    def __init__(self, config: GenericDefaultParamConfig):
        self.config = config
        self.transformer_names = config.transformer_names

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"

    def get_config(self) -> GenericDefaultParamConfig:
        return self.config

    def _internal_name(self, name: str) -> str:
        return self.config._transformer_name_map[name]

    def get_transformer(self, transform_name: str, tree: AST) -> NodeTransformer:
        internal_name = self._internal_name(transform_name)
        if internal_name == "AddDefaultParameter":
            return GenericWithDefaultParam(self.config)
        elif internal_name == "RemoveDefaultParameter":
            return GenericWithoutDefaultParam(self.config)
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")

    def can_transform(self, tree: AST, transform_name: str) -> bool:
        internal_name = self._internal_name(transform_name)
        if internal_name == "AddDefaultParameter":
            v = GenericAddParamCanTransform(self.config)
        elif internal_name == "RemoveDefaultParameter":
            v = GenericRemoveParamCanTransform(self.config)
        else:
            raise ValueError(f"Unknown transformer name {transform_name}")
        v.visit(tree)
        return v.can_transform

    def transform(self, tree: AST, transform_name: str) -> AST:
        transformer = self.get_transformer(transform_name, tree)
        return transformer.visit(tree)

    def get_primary_transform_name(self) -> str:
        return self.config.get_primary_transform_name()
