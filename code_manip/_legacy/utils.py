import ast
from typing import Union, List


def is_attribute_of(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Attribute):
        return False

    value = node.value
    if isinstance(value, ast.Name):
        return value.id == name
    elif isinstance(value, ast.Attribute):
        return is_attribute_of(value, name)
    else:
        return False


def match_name_str(node: str, name: str) -> bool:
    if not isinstance(node, str):
        raise TypeError(f"Expected str, got {type(node)}")

    if name == "*":
        return True

    return node == name


def match_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, str):
        # TODO: change this to warnings
        print("!!! Expected ast.AST, got string, using match_name_str")
        return match_name_str(node, name)

    if not isinstance(node, ast.Name):
        return False

    return match_name_str(node.id, name)


def match_attribute(node: ast.AST, name: str) -> bool:
    assert "." in name

    if not isinstance(node, ast.Attribute):
        return False

    sep = name.rindex(".")
    value = name[:sep]
    attr = name[sep + 1 :]

    if not match_name_str(node.attr, attr):
        return False

    if "." in value:
        return match_attribute(node.value, value)
    else:
        return match_name(node.value, value)


def is_call_to(node: ast.Call, func_names: Union[str, List[str]]) -> bool:
    if isinstance(func_names, str):
        func_names = [func_names]

    def _matcher(node: ast.Call, name: str) -> bool:
        if "." not in name:
            return match_name(node.func, name)
        else:
            return match_attribute(node.func, name)

    return any(_matcher(node, name) for name in func_names)
