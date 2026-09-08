"""PUCT Monte Carlo Tree Search guided by a neural network, as in AlphaZero."""
from __future__ import annotations

import math

import numpy as np
import torch

from .game import ACTION_SIZE, Connect4

C_PUCT = 1.5


class Node:
    __slots__ = ("board", "parent", "action", "children", "prior", "visit_count", "value_sum", "is_expanded")

    def __init__(self, board: np.ndarray, parent=None, action: int | None = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.action = action
        self.children: dict[int, "Node"] = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def ucb_score(self, total_visits: int) -> float:
        exploration = C_PUCT * self.prior * math.sqrt(total_visits) / (1 + self.visit_count)
        return -self.value + exploration  # child value is from the opponent's view


class MCTS:
    """One MCTS instance handles the search for a single move decision."""

    def __init__(self, model, device: torch.device, num_simulations: int = 200,
                 dirichlet_alpha: float = 1.0, dirichlet_eps: float = 0.25):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def _evaluate(self, board: np.ndarray):
        tensor = torch.from_numpy(Connect4.encode(board))
        probs, value = self.model.predict(tensor, self.device)
        return probs, value

    def run(self, root_board: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Return visit-count policy (length ACTION_SIZE, sums to 1)."""
        root = Node(root_board)
        valid = Connect4.valid_moves(root_board)
        probs, _ = self._evaluate(root_board)
        probs = probs * valid
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = valid / valid.sum()

        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * ACTION_SIZE)
            probs = (1 - self.dirichlet_eps) * probs + self.dirichlet_eps * noise * valid
            if probs.sum() > 0:
                probs /= probs.sum()

        self._expand(root, probs, valid)

        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.is_expanded and node.children:
                node = self._select_child(node)
                path.append(node)

            outcome = Connect4.game_ended(node.board)
            if outcome is None:
                probs, value = self._evaluate(node.board)
                valid = Connect4.valid_moves(node.board)
                probs = probs * valid
                if probs.sum() > 0:
                    probs /= probs.sum()
                else:
                    probs = valid / valid.sum()
                self._expand(node, probs, valid)
            else:
                # outcome is from the perspective of the player who just moved
                # into `node.board`'s parent state; for the node itself
                # (whose turn it is) the value is the negation, except draws.
                value = 0.0 if abs(outcome) < 1e-3 else -outcome

            self._backpropagate(path, value)

        visit_counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        total = visit_counts.sum()
        return visit_counts / total if total > 0 else visit_counts

    @staticmethod
    def _expand(node: Node, probs: np.ndarray, valid: np.ndarray):
        for action in range(ACTION_SIZE):
            if valid[action]:
                child_board = Connect4.next_state(node.board, action)
                node.children[action] = Node(child_board, parent=node, action=action, prior=probs[action])
        node.is_expanded = True

    @staticmethod
    def _select_child(node: Node) -> Node:
        total_visits = sum(c.visit_count for c in node.children.values())
        return max(node.children.values(), key=lambda c: c.ucb_score(total_visits))

    @staticmethod
    def _backpropagate(path: list[Node], leaf_value: float):
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
