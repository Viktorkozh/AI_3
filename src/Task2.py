#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Any, Iterator, List, Optional, Tuple


class Problem:
    initial: Any
    goal: Optional[Any]

    def __init__(self, initial: Any = None, goal: Optional[Any] = None, **kwds: Any) -> None:
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state: Any) -> List[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def is_goal(self, state: Any) -> Any:
        return state == self.goal

    def action_cost(self, s: Any, a: Any, s1: Any) -> float:
        return 1

    def h(self, node: 'Node') -> float:
        return 0

    def __str__(self) -> str:
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    state: Any
    parent: Optional['Node']
    action: Optional[Any]
    path_cost: float

    def __init__(self, state: Any, parent: Optional['Node'] = None, action: Optional[Any] = None, path_cost: float = 0) -> None:
        self.__dict__.update(
            state=state, parent=parent, action=action, path_cost=path_cost
        )

    def __repr__(self) -> str:
        return "<{}>".format(self.state)

    def __len__(self) -> int:
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other: 'Node') -> bool:
        return self.path_cost < other.path_cost

    @staticmethod
    def is_cycle(node: 'Node') -> bool:
        parent = node.parent
        while parent:
            if parent.state == node.state:
                return True
            parent = parent.parent
        return False

    failure: 'Node'
    cutoff: 'Node'

    @staticmethod
    def expand(problem: Problem, node: 'Node') -> Iterator['Node']:
        s = node.state
        for action in problem.actions(s):
            s1 = problem.result(s, action)
            cost = node.path_cost + problem.action_cost(s, action, s1)
            yield Node(s1, node, action, cost)


Node.failure = Node("failure", path_cost=math.inf)
Node.cutoff = Node("cutoff", path_cost=math.inf)


def depth_first_recursive_search(problem: Problem, node: Optional[Node] = None) -> Node:
    if node is None:
        node = Node(problem.initial)

    if problem.is_goal(node.state):
        return node
    elif Node.is_cycle(node):
        return Node.failure
    else:
        for child in Node.expand(problem, node):
            result = depth_first_recursive_search(problem, child)
            if result is not Node.failure:
                return result

    return Node.failure


class ProblemLongestPath(Problem):
    matrix: List[List[str]]
    start_char: str
    rows: int
    cols: int
    max_path_length: int

    def __init__(self, matrix: List[List[str]], start_char: str) -> None:
        self.matrix = matrix
        self.start_char = start_char
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if self.rows > 0 else 0
        self.max_path_length = 0
        super().__init__(initial=None)

    def actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = state
        possible_actions: List[Tuple[int, int]] = []
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        path_length = ord(self.matrix[x][y]) - ord(self.start_char)
        if path_length > self.max_path_length:
            self.max_path_length = path_length

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                if ord(self.matrix[nx][ny]) == ord(self.matrix[x][y]) + 1:
                    possible_actions.append((nx, ny))

        return possible_actions

    def result(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        return action

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.rows and 0 <= y < self.cols

    def longest_path(self) -> int:
        max_length = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] == self.start_char:
                    self.initial = (i, j)
                    self.max_path_length = 0
                    depth_first_recursive_search(self)
                    max_length = max(max_length, self.max_path_length)

        return max_length + 1


def main() -> None:
    matrix = [
        ["a", "b", "c", "d", "e", "f", "g"],
        ["h", "a", "i", "j", "k", "l", "m"],
        ["n", "o", "p", "q", "r", "s", "t"],
        ["u", "v", "w", "x", "y", "z", "A"],
    ]
    start_char = "a"
    problem = ProblemLongestPath(matrix, start_char)
    longest_path_length = problem.longest_path()

    print(
        f"The length of the longest path starting with '{start_char}' is: {longest_path_length}"
    )


if __name__ == "__main__":
    main()