"""Generate self-play games with MCTS for training data."""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .game import ACTION_SIZE, Connect4
from .mcts import MCTS


def play_one_game(model, device: torch.device, cfg: Config):
    """Return list of (encoded_board, policy, outcome) training examples."""
    board = Connect4.initial_board()
    mcts = MCTS(model, device, num_simulations=cfg.num_simulations,
                dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps)

    history = []  # (encoded_board, policy) pairs, alternating perspectives
    ply = 0
    outcome = None

    while outcome is None:
        policy = mcts.run(board, add_noise=True)
        history.append((Connect4.encode(board), policy))

        if ply < cfg.temperature_moves:
            action = np.random.choice(ACTION_SIZE, p=policy)
        else:
            action = int(np.argmax(policy))

        board = Connect4.next_state(board, action)
        ply += 1
        outcome = Connect4.game_ended(board)

    # `outcome` is from the perspective of whoever just moved into the final
    # `board`. Walk history backwards, alternating sign each ply so every
    # stored position gets the game result from *its own* mover's perspective.
    examples = []
    value = outcome if abs(outcome) > 1e-3 else 0.0
    for encoded_board, policy in reversed(history):
        examples.append((encoded_board, policy, value))
        value = -value
    examples.reverse()
    return examples
