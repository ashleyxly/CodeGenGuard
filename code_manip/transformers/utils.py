import ast
from typing import Union, List


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


def is_member_of_module(name: ast.Attribute, module: Union[ast.Attribute, ast.Name]) -> bool:
    # if module is a Name, it must be the same as the top level module of name
    if isinstance(module, ast.Name):
        if isinstance(name, ast.Name):
            return module.id == name.id
        elif isinstance(name, ast.Attribute):
            # descend to the top level module of name (which is the leaf of the subtree)
            return is_member_of_module(name.value, module)

    # if module is an Attribute, it must be a "subsequence" of name
    if isinstance(module, ast.Attribute):
        if isinstance(name, ast.Name):
            # module shouldnt be longer than name
            return False

        if isinstance(name, ast.Attribute):
            if module.attr != name.attr:
                # maybe we havent reached the subtree that matches
                # guess we have to descend along name's subtree
                return is_member_of_module(name.value, module)
            else:
                # we have reached the subtree that matches
                # now we need to check the module's value
                return is_member_of_module(name.value, module.value)

    return False
