import numpy as np
import torch

from alphazero.game import ACTION_SIZE, Connect4
from alphazero.mcts import MCTS
from alphazero.model import AlphaZeroNet
from alphazero.self_play import play_one_game
from alphazero.config import Config


def _tiny_model():
    return AlphaZeroNet(channels=8, num_res_blocks=1)


def test_mcts_returns_valid_policy_distribution():
    model = _tiny_model()
    device = torch.device("cpu")
    board = Connect4.initial_board()
    mcts = MCTS(model, device, num_simulations=15)
    policy = mcts.run(board, add_noise=False)

    assert policy.shape == (ACTION_SIZE,)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert (policy >= 0).all()


def test_mcts_only_assigns_probability_to_legal_moves():
    model = _tiny_model()
    device = torch.device("cpu")
    board = Connect4.initial_board()
    for _ in range(6):
        board = Connect4.drop(board, col=0, player=1)  # fill column 0
    mcts = MCTS(model, device, num_simulations=15)
    policy = mcts.run(board, add_noise=False)
    assert policy[0] == 0.0


def test_self_play_game_terminates_and_produces_examples():
    model = _tiny_model()
    device = torch.device("cpu")
    cfg = Config(num_simulations=10, temperature_moves=4)
    examples = play_one_game(model, device, cfg)

    assert len(examples) > 0
    for board, policy, value in examples:
        assert board.shape == (2, 6, 7)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0
