import abc
from ..nodes import Node


class CodeTransformer(abc.ABC):
    name: str

    @abc.abstractmethod
    def get_available_transforms(self):
        pass

    @abc.abstractmethod
    def mutable_tree_transform(self, node: Node, dst_style: str):
        pass

    @abc.abstractmethod
    def can_transform(self, node: Node, dst_style: str) -> bool:
        pass

    def throw_invalid_dst_style(self, dst_style: str):
        msg = f"invalid dst_style: {dst_style} for {self.__class__.__name__}"
        raise ValueError(msg)
