#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from collections import deque
from typing import Any, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class Problem:
    initial: Any
    goal: Optional[Any]

    def __init__(
        self, initial: Any = None, goal: Optional[Any] = None, **kwds: Any
    ) -> None:
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state: Any) -> List[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def is_goal(self, state: Any) -> Any:
        return state == self.goal

    def action_cost(self, s: Any, a: Any, s1: Any) -> float:
        return 1

    def h(self, node: "Node") -> float:
        return 0

    def __str__(self) -> str:
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    state: Any
    parent: Optional["Node"]
    action: Optional[Any]
    path_cost: float

    def __init__(
        self,
        state: Any,
        parent: Optional["Node"] = None,
        action: Optional[Any] = None,
        path_cost: float = 0,
    ) -> None:
        self.__dict__.update(
            state=state, parent=parent, action=action, path_cost=path_cost
        )

    def __repr__(self) -> str:
        return "<{}>".format(self.state)

    def __len__(self) -> int:
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other: "Node") -> bool:
        return self.path_cost < other.path_cost

    @staticmethod
    def is_cycle(node: "Node") -> bool:
        parent = node.parent
        while parent:
            if parent.state == node.state:
                return True
            parent = parent.parent
        return False

    failure: "Node"
    cutoff: "Node"

    @staticmethod
    def expand(problem: Problem, node: "Node") -> Iterator["Node"]:
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


class ProblemFloodFill(Problem):
    grid: List[List[str]]
    start_x: int
    start_y: int
    target_color: str
    replacement_color: str

    def __init__(
        self,
        grid: List[List[str]],
        start_x: int,
        start_y: int,
        target_color: str,
        replacement_color: str,
    ):
        initial = self.find_initial_state(grid)
        goal = None
        super().__init__(
            initial=initial,
            goal=goal,
            grid=grid,
            start_x=start_x,
            start_y=start_y,
            target_color=target_color,
            replacement_color=replacement_color,
        )

    def find_initial_state(self, grid: List[List[str]]) -> Tuple[Tuple[str, ...]]:
        return tuple(tuple(row) for row in grid)  # type: ignore

    def actions(self, state: Any) -> List[Tuple[int, int]]:
        return [
            (dx, dy)
            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            )
        ]

    def result(
        self, state: Tuple[Tuple[str, ...]], action: Tuple[int, int]
    ) -> Tuple[Tuple[str, ...]]:
        grid = [list(row) for row in state]
        x, y = self.start_x, self.start_y
        target_color = grid[x][y]

        if target_color == self.replacement_color:
            return state

        self.flood_fill(grid, x, y, target_color)
        return tuple(tuple(row) for row in grid)  # type: ignore

    def flood_fill(
        self, grid: List[List[str]], x: int, y: int, target_color: str
    ) -> None:
        queue = deque([(x, y)])
        visited = set()

        while queue:
            x, y = queue.popleft()

            if (
                (x, y) in visited
                or x < 0
                or x >= len(grid)
                or y < 0
                or y >= len(grid[0])
                or grid[x][y] != target_color
            ):
                continue

            grid[x][y] = self.replacement_color
            visited.add((x, y))

            for dx, dy in self.actions(grid):
                nx, ny = x + dx, y + dy
                queue.append((nx, ny))

    def is_goal(self, state: Tuple[Tuple[str, ...]]) -> bool:
        for row in state:
            for cell in row:
                if cell == self.target_color:
                    return False
        return True


def main() -> None:
    grid = [
        ["Y", "Y", "Y", "G", "G", "G", "G", "G", "G", "G"],
        ["Y", "Y", "Y", "Y", "Y", "Y", "G", "X", "X", "X"],
        ["G", "G", "G", "G", "G", "G", "G", "X", "X", "X"],
        ["W", "W", "W", "W", "W", "G", "G", "G", "G", "X"],
        ["W", "R", "R", "R", "R", "R", "G", "X", "X", "X"],
        ["W", "W", "W", "R", "R", "G", "G", "X", "X", "X"],
        ["W", "B", "W", "R", "R", "R", "R", "R", "R", "X"],
        ["W", "B", "B", "B", "B", "R", "R", "X", "X", "X"],
        ["W", "B", "B", "X", "B", "B", "B", "B", "X", "X"],
        ["W", "B", "B", "X", "X", "X", "X", "X", "X", "X"],
    ]

    start_x, start_y = 3, 0  # Начальная позиция для заливки
    target_color = "W"  # Цвет, который нужно заменить
    replacement_color = "G"  # Новый цвет

    problem = ProblemFloodFill(grid, start_x, start_y, target_color, replacement_color)
    result_node = depth_first_recursive_search(problem)

    for row in result_node.state:
        print(" ".join(row))


if __name__ == "__main__":
    main()
