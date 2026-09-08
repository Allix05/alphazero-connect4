"""Pit two agents against each other to measure relative strength."""
from __future__ import annotations

import numpy as np
import torch

from .game import ACTION_SIZE, Connect4
from .mcts import MCTS


def mcts_agent(model, device, num_simulations):
    def choose(board):
        mcts = MCTS(model, device, num_simulations=num_simulations)
        policy = mcts.run(board, add_noise=False)
        return int(np.argmax(policy))
    return choose


def random_agent():
    def choose(board):
        valid = np.where(Connect4.valid_moves(board))[0]
        return int(np.random.choice(valid))
    return choose


def play_match(agent_a, agent_b, num_games: int = 20, swap_sides: bool = True):
    """Return (a_wins, b_wins, draws). Agents alternate who moves first."""
    a_wins = b_wins = draws = 0
    for game_idx in range(num_games):
        a_first = (game_idx % 2 == 0) if swap_sides else True
        board = Connect4.initial_board()
        players = [agent_a, agent_b] if a_first else [agent_b, agent_a]
        turn = 0
        outcome = None
        while outcome is None:
            action = players[turn % 2](board)
            board = Connect4.next_state(board, action)
            outcome = Connect4.game_ended(board)
            turn += 1

        if abs(outcome) < 1e-3:
            draws += 1
        else:
            # `outcome` is from perspective of whoever made the last move,
            # i.e. players[(turn - 1) % 2].
            winner = players[(turn - 1) % 2]
            if winner is agent_a:
                a_wins += 1
            else:
                b_wins += 1
    return a_wins, b_wins, draws
