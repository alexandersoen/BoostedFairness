from enum import Enum
from typing import List

class Direction(Enum):
    LEFT = 0
    RIGHT = 1

    def __eq__(self, value):
        return self.value == value.value


class Node:
    def __init__(self, value: Direction, parent: 'Node') -> None:
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None

    def split(self, l: 'Node', r: 'Node'):
        self.left = l
        self.right = r

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def children(self) -> List['Node']:
        if self.is_leaf:
            return []
        else:
            return [self.left, self.right]

    @property
    def name(self) -> str:
        n = []
        cur = self
        while cur:
            n.append(str(cur.value.value))
            cur = cur.parent

        return ''.join(n)


class Tree:
    def __init__(self, node: Node):
        self.root = node

    def grow(self, x: Node, l: Node, r: Node):
        x.split(l, r)

    @property
    def leaves(self) -> List[Node]:
        nodes = [self.root]
        leaves = []
        while nodes:
            n = nodes.pop()

            # Check for leaves
            if n.is_leaf:
                leaves.append(n)
            else:
                nodes.extend(n.children)

        return leaves

    @property
    def depth(self) -> int:
        return 1 + max((Tree(l).depth for l in self.root.children), default=0)

    @property
    def size(self) -> int:
        return 1 + sum(Tree(l).size for l in self.root.children)