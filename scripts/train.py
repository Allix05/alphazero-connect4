#!/usr/bin/env python
"""Entry point: python scripts/train.py [--quick]

--quick uses a much smaller config so the whole pipeline can be sanity
checked (or a toy checkpoint produced) in a couple of minutes on a laptop
CPU. Drop the flag for a real training run (hours, ideally with a GPU).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero.config import Config
from alphazero.train import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny config for a fast smoke-test run")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games-per-iteration", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.iterations = 3
        cfg.games_per_iteration = 6
        cfg.num_simulations = 25
        cfg.arena_games = 6
        cfg.arena_num_simulations = 25
        cfg.epochs_per_iteration = 2
        cfg.batch_size = 64
        cfg.channels = 32
        cfg.num_res_blocks = 3

    if args.iterations is not None:
        cfg.iterations = args.iterations
    if args.games_per_iteration is not None:
        cfg.games_per_iteration = args.games_per_iteration
    if args.num_simulations is not None:
        cfg.num_simulations = args.num_simulations
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir

    print("Config:", json.dumps(cfg.__dict__, indent=2))
    _, history = run_training(cfg)

    history_path = os.path.join(cfg.checkpoint_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
