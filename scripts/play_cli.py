#!/usr/bin/env python
"""Play Connect Four against a trained checkpoint from the terminal.

    python scripts/play_cli.py --checkpoint checkpoints/best.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from alphazero.config import Config
from alphazero.game import Connect4
from alphazero.mcts import MCTS
from alphazero.model import AlphaZeroNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--human-first", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(cfg.channels, cfg.num_res_blocks).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint}; playing against an untrained network.")
    model.eval()

    board = Connect4.initial_board()
    human_turn = args.human_first
    outcome = None
    last_mover_was_human = None

    print("You are X. The AI is O." if args.human_first else "The AI is X and moves first. You are O.")

    while outcome is None:
        print()
        print(Connect4.render(board if human_turn else -board))
        valid = np.where(Connect4.valid_moves(board))[0]

        if human_turn:
            col = None
            while col not in valid.tolist():
                try:
                    col = int(input(f"Your move {list(valid)}: "))
                except ValueError:
                    continue
        else:
            mcts = MCTS(model, device, num_simulations=args.num_simulations)
            policy = mcts.run(board, add_noise=False)
            col = int(np.argmax(policy))
            print(f"AI plays column {col} (confidence {policy[col]:.2f})")

        board = Connect4.next_state(board, col)
        outcome = Connect4.game_ended(board)
        last_mover_was_human = human_turn
        human_turn = not human_turn

    print()
    print(Connect4.render(board if human_turn else -board))
    if abs(outcome) < 1e-3:
        print("Draw!")
    else:
        print("You win!" if last_mover_was_human else "AI wins!")


if __name__ == "__main__":
    main()
