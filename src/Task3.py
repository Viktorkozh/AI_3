#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Any, Iterator, List, Optional, Tuple, Set


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


class WordGenerationProblem:
    board: List[List[str]]
    dictionary: List[str]
    rows: int
    cols: int
    found_words: List[str]
    directions: List[Tuple[int, int]]

    def __init__(self, board: List[List[str]], dictionary: List[str]) -> None:
        self.board = board
        self.dictionary = dictionary
        self.rows = len(board)
        self.cols = len(board[0])
        self.found_words: List[str] = []
        self.directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def is_valid(self, x: int, y: int, visited: Set[Tuple[int, int]]) -> bool:
        return 0 <= x < self.rows and 0 <= y < self.cols and (x, y) not in visited

    def dfs(
        self,
        x: int,
        y: int,
        word: str,
        current_path: str,
        visited: Set[Tuple[int, int]],
    ) -> None:
        if current_path == word:
            self.found_words.append(word)
            return

        if len(current_path) >= len(word):
            return

        if current_path != word[: len(current_path)]:
            return

        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy

            if self.is_valid(nx, ny, visited):
                new_visited = visited.copy()
                new_visited.add((nx, ny))

                self.dfs(nx, ny, word, current_path + self.board[nx][ny], new_visited)

    def solve(self) -> List[str]:
        for word in self.dictionary:
            for x in range(self.rows):
                for y in range(self.cols):
                    if self.board[x][y] == word[0]:
                        initial_visited: Set[Tuple[int, int]] = {(x, y)}
                        self.dfs(x, y, word, self.board[x][y], initial_visited)

        return self.found_words


def main() -> None:
    board: List[List[str]] = [["М", "С", "Е"], ["Р", "А", "Т"], ["Л", "О", "Н"]]
    dictionary: List[str] = ["МАРС", "СОН", "ЛЕТО", "ТОН"]

    problem = WordGenerationProblem(board, dictionary)
    found_words = problem.solve()

    print("Найденные слова:")
    for word in found_words:
        print(word)


if __name__ == "__main__":
    main()
